"""漂移归因诊断：用**可测证据**把漂移分到 8 类之一。

对应 docs/03 §4.2 契约：
    Diagnoser(cfg, n_features, n_classes, feature_groups)
    .diagnose(chunk, model, report, history, oracle) -> Diagnosis

设计原则
--------
1. 每个类别都对应"能被当前数据测出来的量"，不做拍脑袋规则：
   - covariate_affine：用探针标签实测 `align_predict_proba` 的**准确率增益**与
     可恢复比（gain / 原始下降幅度）；
   - novel_class：探针标签里出现 `y >= model.n_classes` 的新标签，或"novelty 高 +
     错分样本 novelty 集中 + 无新标签"；
   - label_shift：`label_prior_em`（BBSE/EM）估计目标先验 VS 训练先验的 TV 距离；
   - noise_inflate：逐特征 std 比（噪声膨胀）；
   - repair：MMD 对 active 参考高、对基座参考低 + 探针准确率恢复；
   - condition_switch / concept：仿射增益无法解释的残差漂移（结构化重参数化 →
     工况切换；残差大且无其它证据 → 概念漂移）；
   - benign：无任何事件/统计量越界。
2. 探针标签严格走 oracle 预算，且优先复用 monitor 已购买的缓存标签（不重复付费）；
   预算不足时退化为**无标签诊断**并降低 confidence，绝不抛异常（封闭评测安全）。
3. recommended 先验来自配置表 `diagnosis.priors`，planner 会在其上叠加 bandit 后验。

阈值来源见 docs/04 §5 与 docs/05 §3；所有数值都在配置里。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.metrics import accuracy, label_prior_em, macro_f1, per_class_recall
from ..core.spec import Cost, Diagnosis, make_cost
from .defaults import resolve_config
from .utils import get_labeler, to_native


class Diagnoser:
    def __init__(self, cfg: dict, n_features: int, n_classes: int,
                 feature_groups: Optional[dict] = None):
        self.cfg = resolve_config(cfg)
        dcfg = self.cfg["diagnosis"]
        self.dcfg = dcfg
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.feature_groups = dict(feature_groups or {i: f"S{i // 6}" for i in range(n_features)})
        self.topk = int(dcfg.get("topk_features", 12))
        self.severity_high = float(dcfg.get("severity_high", 0.55))
        self.affine_recoverability = float(dcfg.get("affine_recoverability", 0.6))
        self.probe_labels = int(dcfg.get("probe_labels", 24))
        self.min_probe_labels = int(dcfg.get("min_probe_labels", 10))
        self.label_reserve = int(dcfg.get("label_reserve", 40))
        self.affine_min_gain = float(dcfg.get("affine_min_gain", 0.03))
        self.affine_min_drop = float(dcfg.get("affine_min_drop", 0.015))
        self.affine_residual_tol = float(dcfg.get("affine_residual_tol", 0.08))
        self.affine_pvalue = float(dcfg.get("affine_pvalue", 0.10))
        self.corr_change_thresh = float(dcfg.get("corr_change_thresh", 0.10))
        self.mean_shift_thresh = float(dcfg.get("mean_shift_thresh", 0.15))
        self.probe_spend_min_drop = float(dcfg.get("probe_spend_min_drop", 0.03))
        self.novelty_ratio_thresh = float(dcfg.get("novelty_ratio_thresh", 0.35))
        self.novel_label_frac = float(dcfg.get("novel_label_frac", 0.02))
        self.error_conc_thresh = float(
            dcfg.get("error_concentration", dcfg.get("error_conc_thresh", 0.5)))
        self.label_shift_tv = float(dcfg.get("label_shift_tv", 0.15))
        self.noise_std_ratio = float(dcfg.get("noise_std_ratio", 1.25))
        self.noise_ks_frac = float(dcfg.get("noise_ks_frac", 0.5))
        self.repair_drop_ratio = float(dcfg.get("repair_drop_ratio", 0.5))
        self.repair_acc_recover = float(dcfg.get("repair_acc_recover", 0.03))
        self.rec_top_k = int(dcfg.get("recommended_top_k", 4))
        self.priors = dict(dcfg.get("priors", {}))
        self.max_total_labels = int(dcfg.get("max_total_labels", 32))
        self._spent_labels = 0
        self.seed = int(self.cfg.get("seed", 0))

    # ------------------------------------------------------------------
    def diagnose(self, chunk, model, report, history=None, oracle=None) -> Diagnosis:
        t = int(getattr(chunk, "t", 0))
        phase = int(getattr(chunk, "phase", 0))
        state = dict(getattr(report, "state", {}) or {})
        events = list(getattr(report, "events", []) or [])
        aux = dict(getattr(report, "aux", {}) or {})
        x = np.asarray(chunk.x, dtype=float)
        n_model_classes = int(getattr(model, "n_classes", self.n_classes))

        ev: dict = {
            "detectors": [getattr(e, "detector", "") for e in events],
            "severity": float(state.get("severity", max(
                [getattr(e, "severity", 0.0) for e in events], default=0.0))),
            "mmd": float(state.get("mmd", 0.0)),
            "mmd_baseline": float(state.get("mmd_baseline", 0.0)),
            "novelty": float(state.get("novelty", 0.0)),
            "ks_count": int(state.get("ks_count", 0)),
            "pred_hist_tv": float(state.get("pred_hist_tv", 0.0)),
            "repair_score": float(state.get("repair_score", 0.0)),
            "std_ratio_median": float(state.get("std_ratio_median", 1.0)),
            "corr_change": float(state.get("corr_change", 0.0)),
            "mean_shift_ratio_mean": float(state.get("mean_shift_ratio_mean", 0.0)),
            "mean_shift_ratio_max": float(state.get("mean_shift_ratio_max", 0.0)),
            "noise_frac": float(state.get("noise_frac", 0.0)),
            "probe_n": 0,
            "label_probe_used": False,
        }

        # ---- 1) 预算内探针（复用 monitor 缓存标签；无性能风险时不买标签） ----
        cost = Cost()
        spend = self._should_spend_probe(state, oracle)
        ev["probe_spend"] = bool(spend)
        labels = self._get_probe_labels(chunk, report, oracle) if spend else {
            "x": np.zeros((0, self.n_features)), "y": np.zeros(0, dtype=np.int64),
            "cost": Cost()}
        x_p, y_p = labels.get("x"), labels.get("y")
        cost = cost.add(labels.get("cost", Cost()))
        self._spent_labels += int(cost.labels)
        ev["probe_n"] = int(len(y_p)) if y_p is not None else 0
        ev["label_probe_used"] = bool(ev["probe_n"] >= self.min_probe_labels)

        # ---- 2) 探针证据：静态准确率 / 仿射校准增益 / 类召回 / 新标签 ----
        if ev["label_probe_used"]:
            self._probe_evidence(model, x, x_p, y_p, state, ev)
        else:
            ev["acc_static"] = float("nan")
            ev["affine_gain"] = 0.0
            ev["affine_recoverability"] = 0.0
            ev["novel_label_frac"] = 0.0

        # ---- 3) 标签先验证据（BBSE/EM，无需标签） ----
        self._prior_evidence(model, x, ev)

        # ---- 4) 误差集中度（novelty 高分的样本是否更容易错） ----
        self._error_concentration_evidence(model, aux, y_p, ev)

        # ---- 5) repair 证据（历史 + 准确率恢复） ----
        self._repair_evidence(state, history, ev)

        # ---- 6) 分类决策树 ----
        kind, confidence, scores = self._decide(events, ev)
        ev["kind_scores"] = scores

        # ---- 7) 影响范围与先验 ----
        affected_features = self._affected_features(state, ev, kind)
        affected_groups = sorted({self.feature_groups.get(int(i), "unknown")
                                  for i in affected_features})
        affected_classes, prior_delta = self._affected_classes(ev, kind, n_model_classes)
        recommended = self._recommend(kind, ev)

        severity = float(min(1.0, max(ev.get("severity", 0.0), state.get("mmd", 0.0)
                                     / max(1.0, state.get("mmd_threshold", 0.35) * 2))))
        diag = Diagnosis(
            t=t, phase=phase, kind=kind, severity=severity,
            confidence=float(confidence),
            affected_features=[int(i) for i in affected_features],
            affected_groups=affected_groups,
            affected_classes=[int(c) for c in affected_classes],
            prior_delta_estimate=[float(v) for v in prior_delta],
            recommended=recommended,
            evidence=to_native(ev),
            cost=cost,
        )
        return diag

    # ------------------------------------------------------------------
    # 探针
    # ------------------------------------------------------------------
    def _get_probe_labels(self, chunk, report, oracle) -> dict:
        """优先复用 monitor 探针缓存标签；不足且预算允许时追加购买。"""
        empty = {"x": np.zeros((0, self.n_features)), "y": np.zeros(0, dtype=np.int64),
                 "cost": Cost()}
        if oracle is None:
            return empty
        labeler = get_labeler(oracle)
        if labeler is None:
            return empty
        remaining = labeler.remaining()
        # 诊断探针上限：min(配置, 20, 剩余/6)——为 playbook 留出大部分预算
        max_n = int(min(self.probe_labels, 20, max(self.min_probe_labels, remaining // 6)))
        if max_n < self.min_probe_labels:
            return empty
        ids = np.asarray(chunk.ids, dtype=np.int64)
        cached = sum(1 for i in ids.tolist() if int(i) in labeler.y_by_id)
        need = max(0, max_n - cached)
        n_allowed = max(0, remaining - self.label_reserve)
        n_query = int(min(need, n_allowed))
        if cached < self.min_probe_labels and n_query < self.min_probe_labels - cached:
            n_query = int(min(max_n, n_allowed))
        got = labeler.acquire(chunk, int(min(max_n, cached + n_query)),
                              reason=f"diagnosis@{getattr(chunk, 't', 0)}")
        if len(got["y"]) < self.min_probe_labels:
            # 退回到"只用缓存"路径（绝不越过为 playbook 预留的预算）
            cached_ids = [int(i) for i in np.asarray(chunk.ids, dtype=np.int64).tolist()
                          if int(i) in labeler.y_by_id]
            if cached_ids:
                got = labeler.query_ids(np.asarray(cached_ids, dtype=np.int64),
                                        reason="diagnosis_cache_only")
                got["requested"] = len(cached_ids)
            else:
                return empty
        return got

    def _should_spend_probe(self, state: dict, oracle) -> bool:
        """只有"可能真的有性能风险"时才买诊断标签，节省预算给 playbook。"""
        if oracle is None:
            return False
        labeler = get_labeler(oracle)
        if labeler is None:
            return False
        remaining = labeler.remaining()
        if remaining < self.label_reserve + self.min_probe_labels:
            return False
        if self._spent_labels >= self.max_total_labels:
            return False
        dets = set(state.get("detectors") or [])
        if dets & {"probe_error", "page_hinkley_probe"}:
            return True
        if "novelty" in dets and float(state.get("novelty", 0.0)) > 1.2 * float(
                self.cfg["monitor"].get("novelty_threshold", 0.6)):
            return True
        ref_acc = state.get("probe_acc_ref", float("nan"))
        cur = state.get("probe_acc", float("nan"))
        recent = state.get("probe_acc_recent") or []
        try:
            ref_acc, cur = float(ref_acc), float(cur)
        except (TypeError, ValueError):
            ref_acc, cur = float("nan"), float("nan")
        if np.isfinite(ref_acc):
            # 用最近 2 次探针的均值降噪（单次 4-6 条标签噪声过大）
            vals = [float(v) for v in recent[-3:]] if recent else ([cur] if np.isfinite(cur) else [])
            drop = ref_acc - float(np.mean(vals)) if vals else 0.0
            # 4-6 条探针噪声大：要求平均下降更明显（0.06）才花诊断标签
            if drop >= max(self.probe_spend_min_drop, 0.06):
                return True
            return False
        # 没有性能探针可比（探针预留耗尽/首窗口）：只在预算很充裕时花标签确认；
        # 否则走无标签诊断 + NOOP/零标签动作，把预算留给后续真正需要的适应动作。
        try:
            total = int(getattr(labeler, "oracle", None).budget_state().labels_total)
        except Exception:
            total = 0
        if total and remaining <= 0.55 * total:
            return False
        if dets & {"mmd", "ks_topk"} and float(
                state.get("severity", 0.0)) >= self.severity_high:
            return True
        return False

    @staticmethod
    def _sign_test_p(b: int, c: int) -> float:
        """McNemar 精确符号检验（双侧），b/c 为两种预测一错一对的样本数。"""
        import math

        n = int(b) + int(c)
        if n == 0:
            return 1.0
        k = min(int(b), int(c))
        tail = sum(math.comb(n, i) for i in range(0, k + 1)) / (2.0 ** n)
        return float(min(1.0, 2.0 * tail))

    def _probe_evidence(self, model, x_chunk, x_p, y_p, state, ev) -> None:
        y_p = np.asarray(y_p, dtype=np.int64)
        x_p = np.asarray(x_p, dtype=float)
        pred_static = model.predict(x_p)
        ev["acc_static"] = float(accuracy(y_p, pred_static))
        known = y_p < model.n_classes
        if known.any():
            ev["acc_static_known"] = float(accuracy(y_p[known], pred_static[known]))
        else:
            ev["acc_static_known"] = float("nan")
        ev["probe_true_prior"] = [float(v) for v in
                                  (np.bincount(np.clip(y_p, 0, max(self.n_classes,
                                                                   int(y_p.max()) + 1) - 1),
                                               minlength=max(self.n_classes,
                                                             int(y_p.max()) + 1))
                                   / max(1, len(y_p)))] if len(y_p) else []
        # 新标签证据
        ev["novel_label_frac"] = float(np.mean(y_p >= model.n_classes)) if len(y_p) else 0.0
        ev["n_novel_labels"] = int(np.sum(y_p >= model.n_classes))
        # 仿射校准实测增益
        target_scaler = model.estimate_target_scaler(x_chunk)
        proba_aff = model.align_predict_proba(x_p, target_scaler)
        pred_aff = np.argmax(proba_aff, axis=1)
        ev["acc_affine"] = float(accuracy(y_p, pred_aff))
        ev["affine_gain"] = float(ev["acc_affine"] - ev["acc_static"])
        # 配对显著性：仿射校准改对的样本数 b vs 改错的样本数 c
        right_static = pred_static == y_p
        right_affine = pred_aff == y_p
        b = int(np.sum(~right_static & right_affine))
        c = int(np.sum(right_static & ~right_affine))
        ev["affine_b"], ev["affine_c"] = b, c
        ev["affine_pvalue"] = self._sign_test_p(b, c)
        ev["affine_significant"] = bool(
            ev["affine_gain"] >= self.affine_min_gain and ev["affine_pvalue"] <= self.affine_pvalue)
        ref_acc = state.get("probe_acc_ref", float("nan"))
        try:
            ref_acc = float(ref_acc)
        except (TypeError, ValueError):
            ref_acc = float("nan")
        ev["ref_acc"] = ref_acc
        drop = (ref_acc - ev["acc_static"]) if np.isfinite(ref_acc) else float("nan")
        ev["acc_drop"] = float(drop) if np.isfinite(drop) else float("nan")
        if np.isfinite(drop) and drop > self.affine_min_drop:
            ev["affine_recoverability"] = float(np.clip(
                ev["affine_gain"] / max(drop, 1e-6), 0.0, 2.0))
        else:
            ev["affine_recoverability"] = (1.0 if ev["affine_gain"] >= self.affine_min_gain
                                           else 0.0)
        # 类别召回
        n_eff = max(self.n_classes, int(y_p.max()) + 1 if len(y_p) else self.n_classes)
        rec = per_class_recall(np.clip(y_p, 0, n_eff - 1), pred_static, n_eff)
        ev["probe_per_class_recall"] = {str(k): float(v) for k, v in rec.items()}
        vals = [v for v in rec.values() if np.isfinite(v)]
        ev["probe_class_recall_min"] = float(min(vals)) if vals else float("nan")
        # 类别先验估计（探针经验先验，24 条噪声大，仅作辅助）
        counts = np.bincount(np.clip(y_p, 0, self.n_classes - 1), minlength=self.n_classes)
        ev["probe_prior"] = [float(v / max(1, len(y_p))) for v in counts]

    # ------------------------------------------------------------------
    def _prior_evidence(self, model, x: np.ndarray, ev: dict) -> None:
        try:
            proba = model.predict_proba(x)
            prior_hat, _ = label_prior_em(proba, model.n_classes)
        except Exception:
            prior_hat = np.ones(model.n_classes) / max(1, model.n_classes)
        ref_prior = np.ones(model.n_classes) / max(1, model.n_classes)
        ev["prior_hat"] = [float(v) for v in prior_hat]
        ev["prior_tv"] = float(0.5 * np.sum(np.abs(prior_hat - ref_prior)))
        # 与探针经验先验融合（若探针可用）
        if ev.get("probe_prior"):
            pp = np.asarray(ev["probe_prior"], dtype=float)
            k = min(len(pp), len(prior_hat))
            fused = 0.5 * np.asarray(prior_hat[:k]) + 0.5 * pp[:k]
            ev["prior_fused"] = [float(v) for v in fused]
            ev["prior_fused_tv"] = float(0.5 * np.sum(np.abs(fused - ref_prior[:k])))
        else:
            ev["prior_fused"] = [float(v) for v in prior_hat]
            ev["prior_fused_tv"] = ev["prior_tv"]

    def _error_concentration_evidence(self, model, aux, y_p, ev: dict) -> None:
        """无标签误差集中度：低置信样本的 novelty 是否显著高于高置信样本。

        novel 类样本天然 novelty 高且模型低置信；若两者相关性强，则支持
        "错分集中在新奇样本上"（novel_class 的证据之一，不需要额外标签）。
        """
        sn = aux.get("sample_novelty")
        proba = aux.get("proba")
        ev["error_conc"] = float("nan")
        ev["novelty_err_mean"] = float("nan")
        ev["novelty_ok_mean"] = float("nan")
        if sn is None or proba is None or len(sn) == 0 or len(proba) != len(sn):
            return
        sn = np.asarray(sn, dtype=float)
        conf = np.max(np.asarray(proba, dtype=float), axis=1)
        try:
            q = float(np.quantile(conf, 0.3))
        except Exception:
            return
        low = conf <= q
        if low.sum() < 3 or (~low).sum() < 3:
            return
        err_mean = float(np.mean(sn[low]))
        ok_mean = float(np.mean(sn[~low]))
        scale = max(abs(ok_mean), 1e-6)
        ev["novelty_err_mean"] = err_mean
        ev["novelty_ok_mean"] = ok_mean
        ev["error_conc"] = float((err_mean - ok_mean) / scale)

    # ------------------------------------------------------------------
    def _repair_evidence(self, state: dict, history, ev: dict) -> None:
        ev["repair"] = float(state.get("repair_score", 0.0))
        ev["recent_drift_kinds"] = self._history_kinds(history)
        # 探针准确率相对近期的恢复
        acc = state.get("probe_acc", float("nan"))
        recent = state.get("probe_acc_recent") or []
        try:
            acc = float(acc)
        except (TypeError, ValueError):
            acc = float("nan")
        rec = [float(v) for v in recent if v is not None]
        ev["acc_recent_min"] = float(min(rec)) if rec else float("nan")
        if np.isfinite(acc) and rec:
            ev["acc_recover_delta"] = float(acc - min(rec))
        else:
            ev["acc_recover_delta"] = 0.0
        repair_hist = (ev["repair"] >= self.repair_drop_ratio
                       and ev.get("mmd_baseline", 0.0) <= state.get(
                           "mmd_threshold", 0.35) * self.repair_drop_ratio)
        ev["repair_evidence"] = bool(repair_hist)

    @staticmethod
    def _history_kinds(history) -> list:
        kinds = []
        for h in (history or []):
            if isinstance(h, dict):
                k = h.get("diagnosis_kind") or h.get("kind") or ""
            else:
                k = getattr(h, "kind", "")
            if k:
                kinds.append(str(k))
        return kinds[-6:]

    # ------------------------------------------------------------------
    # 决策树
    # ------------------------------------------------------------------
    def _decide(self, events, ev: dict) -> tuple[str, float, dict]:
        """按"可测证据强度"的优先级做确定性分类（阈值见 docs/04 §5）。"""
        detectors = set(ev.get("detectors") or [])
        severity = float(ev.get("severity", 0.0))
        any_event = bool(events)
        probe_used = bool(ev.get("label_probe_used"))
        novel_lab = float(ev.get("novel_label_frac", 0.0))
        novelty = float(ev.get("novelty", 0.0))
        affine_gain = float(ev.get("affine_gain", 0.0) or 0.0)
        aff_sig = bool(ev.get("affine_significant", False))
        recover = float(ev.get("affine_recoverability", 0.0) or 0.0)
        residual = (ev.get("ref_acc", float("nan")) - ev.get("acc_affine", float("nan")))
        has_probe = probe_used
        acc_drop = ev.get("acc_drop", float("nan"))
        has_drop = bool(has_probe and np.isfinite(acc_drop)
                        and float(acc_drop) > self.affine_min_drop)
        tv = float(ev.get("prior_fused_tv", ev.get("prior_tv", 0.0)) or 0.0)
        tv_raw = float(ev.get("prior_tv", 0.0) or 0.0)
        noise_ratio = float(ev.get("std_ratio_median", 1.0) or 1.0)
        noise_frac = float(ev.get("noise_frac", 0.0) or 0.0)
        corr = float(ev.get("corr_change", 0.0) or 0.0)
        mean_shift = float(ev.get("mean_shift_ratio_mean", 0.0) or 0.0)
        err_conc = ev.get("error_conc", float("nan"))
        scores = {k: 0.0 for k in ("benign", "covariate_affine", "condition_switch",
                                   "concept", "novel_class", "label_shift",
                                   "noise_inflate", "repair")}

        def out(kind: str, conf: float):
            scores[kind] = float(max(scores.get(kind, 0.0), conf))
            return kind, float(min(0.98, max(0.05, conf))), scores

        if not any_event and severity <= 1e-9:
            return out("benign", 0.80)

        # 1) repair：对 active 参考高、对基座参考低（数据回到基线分布）
        if ev.get("repair_evidence"):
            return out("repair", 0.65 + 0.3 * min(1.0, float(ev.get("repair", 0.0))))

        # 2) novel_class：探针里出现模型不认识的新标签（最强证据）
        if probe_used and novel_lab >= self.novel_label_frac:
            return out("novel_class", 0.6 + 0.4 * min(1.0, novel_lab / 0.15))
        if (probe_used and novelty > self.novelty_threshold_effective()
                and np.isfinite(err_conc) and float(err_conc) > self.error_conc_thresh):
            return out("novel_class", 0.55 + 0.25 * min(1.0, novelty / 2.0))

        # 3) label_shift：EM/BBSE 先验显著偏移、特征分布形状未变（std 比正常、
        #    性能下降有限、仿射无显著增益）—— 先验重加权即为正确动作
        mild_drop = (not has_drop) or (np.isfinite(acc_drop) and float(acc_drop) < 0.12)
        if (max(tv, tv_raw) >= self.label_shift_tv and not aff_sig
                and novel_lab < self.novel_label_frac and mild_drop
                and noise_ratio < self.noise_std_ratio):
            return out("label_shift", 0.55 + 0.4 * min(1.0, max(tv, tv_raw) / 0.3))
        if (float(ev.get("pred_hist_tv", 0.0) or 0.0) >= 0.15 and mild_drop
                and not aff_sig and noise_ratio < self.noise_std_ratio):
            return out("label_shift", 0.45 + 0.3 * min(1.0, float(ev["pred_hist_tv"])))

        # 4) noise_inflate：逐特征 std 比显著上升，**且仿射校准无法把差距拉回**
        residual_known = np.isfinite(residual)
        affine_cannot_fix = not aff_sig and (not residual_known
                                             or float(residual) > self.affine_residual_tol)
        if (noise_ratio >= self.noise_std_ratio
                and (noise_frac >= self.noise_ks_frac or noise_ratio >= 1.5)
                and affine_cannot_fix):
            return out("noise_inflate", 0.5 + 0.35 * min(1.0, noise_ratio - 1.0))

        # 5) 有标签：用仿射校准的**配对显著性**区分协变量漂移 vs 结构重参数化
        if has_probe:
            if aff_sig:
                nearly = np.isfinite(residual) and float(residual) <= self.affine_residual_tol
                # 相关系数结构显著变化 = 存在旋转/重参数化 → 工况切换（即使仿射部分可恢复）
                if corr >= self.corr_change_thresh:
                    return out("condition_switch", 0.6 + 0.25 * min(1.0, recover))
                if nearly or recover >= 0.85:
                    return out("covariate_affine", 0.6 + 0.35 * min(1.0, recover))
                return out("condition_switch", 0.55 + 0.3 * min(1.0, recover))
            # 仿射不显著：相关系数结构变化 → 工况切换（潜空间旋转/重参数化）
            if corr >= self.corr_change_thresh and has_drop:
                return out("condition_switch", 0.55 + 0.3 * min(1.0, corr / 0.2))
            # 均值/尺度改变（磨损、增益老化）→ 协变量漂移（标签无关证据）
            ref_minus_affine = residual
            if has_drop and (mean_shift >= self.mean_shift_thresh
                             or noise_ratio >= 1.10
                             or (np.isfinite(ref_minus_affine)
                                 and float(ref_minus_affine) <= self.affine_residual_tol)):
                return out("covariate_affine", 0.45 + 0.2 * min(1.0, mean_shift))
            # 仿射后仍明显不达标（绝对准确率仍低）→ 升级为概念漂移
            if has_drop and float(acc_drop) >= 0.12:
                return out("concept", 0.45 + 0.2 * min(1.0, float(acc_drop)))
            # 有漂移信号但性能未明显受损 → 保守判 benign（不动作），置信度低
            if any_event or has_drop:
                return out("benign", 0.42)

        # 6) 无标签（预算不足）路径：只用结构证据，置信度低
        if corr >= self.corr_change_thresh:
            return out("condition_switch", 0.4 + 0.2 * min(1.0, corr / 0.3))
        if mean_shift >= 1.5 * self.mean_shift_thresh or noise_ratio >= 1.25:
            return out("covariate_affine", 0.38)
        if any_event:
            if severity >= self.severity_high:
                return out("concept", 0.35)
            return out("benign", 0.35)
        return out("benign", 0.5)

    def novelty_threshold_effective(self) -> float:
        return float(self.cfg["monitor"].get("novelty_threshold", 0.6))

    # ------------------------------------------------------------------
    # 影响范围
    # ------------------------------------------------------------------
    def _affected_features(self, state: dict, ev: dict, kind: str) -> list:
        feats: list[int] = []
        if kind == "covariate_affine":
            feats = list(state.get("mean_shift_features") or [])
        if not feats:
            feats = list(state.get("ks_topk") or [])
        if not feats:
            feats = list(state.get("std_ratio_top") or [])
        return feats[: max(self.topk, 6)]

    def _affected_classes(self, ev: dict, kind: str, n_model_classes: int):
        rec = ev.get("probe_per_class_recall") or {}
        bad = []
        for k, v in rec.items():
            try:
                vv = float(v)
            except (TypeError, ValueError):
                continue
            if np.isfinite(vv) and vv < 0.5:
                bad.append(int(k))
        prior_delta = []
        prior = ev.get("prior_fused") or ev.get("prior_hat") or []
        k = max(int(n_model_classes), len(prior))
        for i in range(k):
            base = 1.0 / max(1, len(prior)) if prior else 1.0 / max(1, k)
            val = float(prior[i]) if i < len(prior) else 0.0
            prior_delta.append(val - base)
        if kind == "novel_class":
            # 新类索引 = 基座类别数（0 起）
            if n_model_classes < self.n_classes:
                bad.append(int(n_model_classes))
                # prior_delta 长度至少覆盖到新类
                while len(prior_delta) <= n_model_classes:
                    prior_delta.append(0.0)
                prior_delta[n_model_classes] = max(
                    prior_delta[n_model_classes], 0.05)
        if not bad:
            order = np.argsort(-np.abs(np.asarray(prior_delta)))[:3] if prior_delta else []
            bad = [int(c) for c in order if abs(prior_delta[int(c)]) > 0.02]
        return sorted(set(int(c) for c in bad))[: self.n_classes], prior_delta

    # ------------------------------------------------------------------
    # 推荐先验
    # ------------------------------------------------------------------
    def _recommend(self, kind: str, ev: dict) -> list[dict]:
        table = self.priors.get(kind) or self.priors.get("concept") or {}
        items = [{"playbook": pb, "prior": float(p), "why": self._why(kind, pb, ev)}
                 for pb, p in table.items()]
        items.sort(key=lambda d: -d["prior"])
        n_keep = max(2, self.rec_top_k)
        items = items[:n_keep]
        total = sum(d["prior"] for d in items) or 1.0
        for d in items:
            d["prior"] = float(d["prior"] / total)
        return items

    @staticmethod
    def _why(kind: str, playbook: str, ev: dict) -> str:
        base = {
            "benign": "无漂移，保持 champion 以节省预算",
            "covariate_affine": "探针实测仿射校准可恢复大部分准确率，零标注成本",
            "condition_switch": "结构化重参数化，需带标签微调/重训",
            "concept": "条件分布变化，仿射无法解释，需带标签适应",
            "novel_class": "检测到未知类别数据，需扩展输出层",
            "label_shift": "类别先验偏移，特征分布未变，先验重加权即可",
            "noise_inflate": "噪声膨胀，低风险/验证收益后再决定",
            "repair": "分布回到历史基线，回滚/保持稳定版本",
        }.get(kind, "由策略表先验决定")
        return f"{base}（{playbook}）"
