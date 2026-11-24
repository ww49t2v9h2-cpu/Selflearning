"""漂移监测器：无标签统计检验 + 预算内标注探针。

对应 docs/03 §4.1 契约：
    DriftMonitor(cfg, n_features, n_classes)
    .reset_reference(x_ref)
    .update(chunk, model, oracle) -> MonitorReport(events, state)

检测器（详见 docs/04）
--------------------
1. MMD(RBF)^2：当前 chunk 对 **active 参考窗口**（最近被接受的数据域）；
2. 逐特征 KS：对 active 参考窗口做双侧 KS，统计显著特征数 + top-k 统计量；
3. 代理指标：预测熵 / 边际 / 预测类别直方图的 z 分数与 TV 距离；
4. Page-Hinkley：在"标签无关代理误差"序列上检测**渐变**漂移；
5. novelty：kNN 距离比（当前窗口到基座参考分布的 k 近邻距离 / 参考分布自身尺度）；
6. 探针：按 `probe_every` 用 oracle 买 `probe_labels` 条标签测准确率，预算不足自动跳过；
7. repair 信号：MMD 对 active 参考高、但对 fixed 基线参考低 → 数据回到基线分布。

触发去抖
--------
真实闭环里"同一段漂移"会连续多窗口发事件；若每个窗口都触发实验会耗尽预算。
监测器内部维护 `event_cooldown`：一次事件后 N 个窗口内只更新统计量、不再重复发事件；
同时用"滚动窗口稳定性"判定进入新的稳定域并重设 active 参考窗口，因此同一轮漂移
只触发一次、稳态期不再误报（消融 trigger_mode=always 由 loop 覆盖，不依赖本行为）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.metrics import (accuracy, ks_2samp, macro_f1, marginal_entropy,
                            mean_entropy, mean_margin, mmd_rbf,
                            per_class_recall, prediction_histogram)
from ..core.spec import Cost, DriftEvent
from .defaults import resolve_config
from .utils import get_labeler, to_native


# --------------------------------------------------------------------------
# Page-Hinkley（渐变漂移检测）
# --------------------------------------------------------------------------
class PageHinkley:
    """Page-Hinkley 变点检测（本工程变体，见 docs/04 §4）。

    对标准化后的代理误差序列 z_t（目标均值 0）：
        S_t = Σ_{i≤t} (z_i - delta)
        M_t = min_{i≤t} S_i,   PH_t = S_t - M_t
    当 PH_t > threshold 时报警；报警后在 PH_t 回落到 reset_ratio*threshold
    以下才重新武装（迟滞，避免逐窗口重复报警）。
    """

    def __init__(self, delta: float = 0.002, threshold: float = 8.0,
                 reset_ratio: float = 0.5):
        self.delta = float(delta)
        self.threshold = float(threshold)
        self.reset_ratio = float(reset_ratio)
        self.reset()

    def reset(self) -> None:
        self.cum = 0.0
        self.min_cum = 0.0
        self.stat = 0.0
        self.armed = True
        self.fired = False
        self.n = 0

    def update(self, z: float) -> float:
        z = float(z)
        self.n += 1
        self.cum += z - self.delta
        self.min_cum = min(self.min_cum, self.cum)
        self.stat = self.cum - self.min_cum
        self.fired = False
        if self.armed and self.stat > self.threshold:
            self.fired = True
            self.armed = False
        elif not self.armed and self.stat < self.reset_ratio * self.threshold:
            self.armed = True
        return self.stat


# --------------------------------------------------------------------------
# 报告对象
# --------------------------------------------------------------------------
@dataclass
class MonitorReport:
    """一个窗口的监测报告（events/state 进记录，aux 只在本进程内给诊断器用）。"""

    events: list = field(default_factory=list)
    state: dict = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    aux: dict = field(default_factory=dict)   # 不序列化：样本级 novelty、参考统计量等

    def to_dict(self) -> dict:
        return {"events": [e.to_dict() for e in self.events],
                "state": to_native(self.state),
                "cost": self.cost.to_dict()}


# --------------------------------------------------------------------------
# 监测器
# --------------------------------------------------------------------------
class DriftMonitor:
    def __init__(self, cfg: dict, n_features: int, n_classes: int):
        self.cfg = resolve_config(cfg)
        mcfg = self.cfg["monitor"]
        self.n_features = int(n_features)
        self.n_classes = int(n_classes)
        self.mcfg = mcfg
        self.warmup_chunks = int(mcfg.get("warmup_chunks", 4))
        self.ref_max_samples = int(mcfg.get("ref_max_samples", 800))
        self.ref_max_chunks = int(mcfg.get("ref_max_chunks", mcfg.get("window", 5)))
        self.rolling_len = int(mcfg.get("window", 5))
        # 阈值优先取 engine.monitor 的实测量程（MMD^2 值域差异大，见 docs/04 §3）；
        # 兼容 DEFAULT_CONFIG 里的 monitor.mmd_threshold。
        self.mmd_threshold = float(mcfg.get("mmd_threshold_calibrated",
                                            mcfg.get("mmd_threshold", 0.35)))
        self.mmd_subsample = int(mcfg.get("mmd_subsample", 400))
        self.ks_alpha = float(mcfg.get("ks_alpha", 0.001))
        self.ks_topk = int(mcfg.get("ks_topk", mcfg.get("topk_features", 12)))
        self.ks_count_threshold = int(mcfg.get("ks_count_threshold", 6))
        self.ks_stat_threshold = float(mcfg.get("ks_stat_threshold", 0.10))
        self.entropy_z_threshold = float(mcfg.get("entropy_z_threshold", 3.0))
        self.margin_z_threshold = float(mcfg.get("margin_z_threshold", 3.0))
        self.pred_hist_tv_threshold = float(mcfg.get("pred_hist_tv_threshold", 0.15))
        self.novelty_k = int(mcfg.get("novelty_k", 8))
        self.novelty_threshold = float(mcfg.get("novelty_threshold", 0.6))
        self.novelty_subsample = int(mcfg.get("novelty_subsample", 300))
        self.probe_every = max(1, int(mcfg.get("probe_every", 2)))
        self.probe_labels = int(mcfg.get("probe_labels", 6))
        self.probe_min_labels = int(mcfg.get("probe_min_labels", 6))
        self.proxy_drop_threshold = float(mcfg.get("proxy_drop_threshold", 0.055))
        self.adapt_reserve = int(mcfg.get("adapt_reserve", 60))
        self.min_gap = int(mcfg.get("min_gap", 1))
        self.ref_patience = int(mcfg.get("ref_patience", 3))
        self.stability_ratio = float(mcfg.get("stability_ratio", 0.5))
        self.repair_ratio = float(mcfg.get("repair_ratio", 0.5))
        self.event_cooldown = int(mcfg.get("event_cooldown", 3))
        self.probe_control_every = int(mcfg.get("probe_control_every", 4))
        self.adapt_reserve_frac = float(mcfg.get("adapt_reserve_frac", 0.60))
        self.anchor_labels = int(mcfg.get("anchor_labels", 4))
        self.rolling_samples = int(mcfg.get("rolling_samples_per_chunk", 200))
        self.repair_patience = int(mcfg.get("repair_patience", 1))

        # 参考窗口状态
        self.baseline_ref: Optional[np.ndarray] = None     # 固定基座参考（原始已知域）
        self.active_ref: Optional[np.ndarray] = None       # 当前被接受的域（可重设）
        self.rolling: list[np.ndarray] = []                # 最近若干窗口样本（判稳定性）
        self._knn_ref_scale = 1.0
        self._novelty_ref_sample: Optional[np.ndarray] = None
        self._novelty_ref_knn: Optional[np.ndarray] = None
        # 代理基线（用当前 champion 在参考窗口上的预测统计）
        self._model_key: Any = None
        self._ref_entropy = 0.0
        self._ref_entropy_std = 1.0
        self._ref_margin = 0.0
        self._ref_margin_std = 1.0
        self._ref_pred_hist: Optional[np.ndarray] = None
        # 运行状态
        self._n_update = 0
        self.last_event_t = -10 ** 9
        self.last_probe_t = -10 ** 9
        self._probe_acc_ref: Optional[float] = None
        self._probe_accs: list[float] = []
        self._last_probe_acc: Optional[float] = None
        self._last_probe_n = 0
        self._stable_count = 0
        self._repair_pending = 0
        self._ph_proxy = PageHinkley(delta=float(mcfg.get("ph_delta", 0.002)),
                                     threshold=float(mcfg.get("ph_threshold", 8.0)),
                                     reset_ratio=float(mcfg.get("ph_lambda", 0.1)))
        # ---- 双门不变量 / 失效方向指示器（notes/学习_dyn3三篇 映射） ----
        self.provisional_ref: Optional[np.ndarray] = None   # 候选参考域：仅用于抑制重复告警
        self._accepted_updates = 0        # 门禁通过（registry 变更）次数
        self._trigger_history: list[int] = []
        self._accept_history: list[int] = []
        self._suppress_history: list[int] = []
        self._model_watch_key: Any = None
        self._z_ent = 0.0
        self._z_mar = 0.0
        self._ph_probe = PageHinkley(delta=float(mcfg.get("ph_delta", 0.002)),
                                     threshold=float(mcfg.get("ph_threshold", 8.0)),
                                     reset_ratio=float(mcfg.get("ph_lambda", 0.1)))
        self._ref_ready = False

    # ------------------------------------------------------------------
    # 参考窗口
    # ------------------------------------------------------------------
    def reset_reference(self, x_ref: np.ndarray) -> None:
        """用初始（无漂移）窗口建立基座参考；同时清空运行状态。"""
        x_ref = np.asarray(x_ref, dtype=float)
        if x_ref.size == 0:
            return
        self.baseline_ref = self._subsample(x_ref, self.ref_max_samples)
        self.active_ref = self.baseline_ref.copy()
        self.rolling = []
        self._n_update = 0
        self.last_event_t = -10 ** 9
        self.last_probe_t = -10 ** 9
        self._probe_acc_ref = None
        self._probe_accs = []
        self._last_probe_acc = None
        self._last_probe_n = 0
        self._stable_count = 0
        self._repair_pending = 0
        self._model_key = None
        self._ref_ready = True
        self._ph_proxy.reset()
        self._ph_probe.reset()
        self.provisional_ref = None
        self._accepted_updates = 0
        self._trigger_history = []
        self._accept_history = []
        self._suppress_history = []
        self._compute_novelty_scale()

    def _subsample(self, x: np.ndarray, max_n: int,
                   rng: Optional[np.random.Generator] = None) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        if len(x) <= max_n:
            return x.copy()
        rng = rng or np.random.default_rng(12345)
        idx = rng.choice(len(x), size=max_n, replace=False)
        return x[np.sort(idx)].copy()

    def _compute_novelty_scale(self) -> None:
        ref = self.baseline_ref
        if ref is None or len(ref) < self.novelty_k + 2:
            self._knn_ref_scale = 1.0
            self._novelty_ref_sample = None
            return
        sub = self._subsample(ref, self.novelty_subsample,
                              np.random.default_rng(777))
        self._novelty_ref_sample = sub
        d = _pairwise_sq_dists(sub, sub)
        d[np.arange(len(sub)), np.arange(len(sub))] = np.inf
        kth = np.sqrt(np.partition(d, self.novelty_k - 1, axis=1)[:, self.novelty_k - 1])
        self._knn_ref_scale = float(np.mean(kth)) if np.mean(kth) > 1e-9 else 1.0

    # ------------------------------------------------------------------
    # 主入口
    # ------------------------------------------------------------------
    def update(self, chunk, model, oracle=None) -> MonitorReport:
        events: list[DriftEvent] = []
        state: dict = {}
        cost = Cost()
        t = int(getattr(chunk, "t", self._n_update))
        phase = int(getattr(chunk, "phase", 0))
        x = np.asarray(chunk.x, dtype=float)
        if not self._ref_ready:
            if len(x) == 0:
                return MonitorReport(events=[], state={"drift_any": False})
            self.reset_reference(x)

        warmup = self._n_update < self.warmup_chunks
        self._n_update += 1
        if oracle is not None:
            try:
                self._total_labels = int(oracle.budget_state().labels_total)
            except Exception:
                pass
        # 模型层版本变化 = 门禁通过的上线/回滚（"双门不变量"：此时才允许参考域更新，
        # 由 loop 调用 on_promote() 完成；这里只记账与刷新代理基线）
        if (not warmup and self._model_watch_key is not None
                and id(model) != self._model_watch_key):
            self._accepted_updates += 1
            self._accept_history.append(int(t))
            self._on_external_model_change()
        self._model_watch_key = id(model)

        # ---- 1. 用当前模型刷新参考窗口的代理统计（模型变化时重算） ----
        self._ensure_proxy_reference(model)

        # ---- 2. MMD（对 active 参考 / 对基座参考） ----
        mmd_active = 0.0
        mmd_baseline = 0.0
        if self.active_ref is not None and len(self.active_ref) >= 4 and len(x) >= 4:
            mmd_active = mmd_rbf(x, self._subsample(self.active_ref, self.mmd_subsample,
                                                    np.random.default_rng(t)),
                                 max_n=self.mmd_subsample, seed=t)
        if self.baseline_ref is not None and len(self.baseline_ref) >= 4 and len(x) >= 4:
            mmd_baseline = mmd_rbf(x, self._subsample(self.baseline_ref, self.mmd_subsample,
                                                      np.random.default_rng(t + 1)),
                                   max_n=self.mmd_subsample, seed=t + 1)
        state["mmd"] = float(mmd_active)
        state["mmd_active"] = float(mmd_active)
        state["mmd_baseline"] = float(mmd_baseline)
        state["mmd_threshold"] = float(self.mmd_threshold)

        # ---- 3. 逐特征 KS（对 active 参考） ----
        ks_stat_topk, ks_features, ks_count, ks_mean = self._ks_features(x)
        state["ks_topk"] = [int(i) for i in ks_features[: self.ks_topk]]
        state["ks_count"] = int(ks_count)
        state["ks_stat_topk"] = float(ks_mean)
        state["ks_stat_max"] = float(ks_stat_topk)
        state["ks_alpha"] = float(self.ks_alpha)
        state["corr_change"] = self._corr_change(x)
        fs = self.feature_stats(x)
        state["std_ratio_median"] = fs["std_ratio_median"]
        state["std_ratio_mean"] = fs["std_ratio_mean"]
        state["noise_frac"] = fs["noise_frac"]
        state["std_ratio_top"] = fs["std_ratio_top"]
        state["mean_shift_top"] = fs["mean_shift_top"]
        state["mean_shift_ratio_mean"] = fs["mean_shift_ratio_mean"]
        state["mean_shift_ratio_max"] = fs["mean_shift_ratio_max"]
        state["mean_shift_features"] = [int(i) for i in fs["mean_shift_top"]]

        # ---- 4. 预测代理：熵 / 边际 / 直方图 TV ----
        proba = model.predict_proba(x) if len(x) else np.zeros((0, self.n_classes))
        ent = mean_entropy(proba) if len(proba) else 0.0
        mar = mean_margin(proba) if len(proba) else 0.0
        ent_z = (ent - self._ref_entropy) / max(self._ref_entropy_std, 1e-3)
        mar_z = (self._ref_margin - mar) / max(self._ref_margin_std, 1e-3)
        hist = (prediction_histogram(proba, max(self.n_classes, model.n_classes))
                if len(proba) else np.zeros(max(self.n_classes, model.n_classes)))
        if self._ref_pred_hist is not None:
            k = min(len(hist), len(self._ref_pred_hist))
            tv = float(0.5 * np.sum(np.abs(hist[:k] - self._ref_pred_hist[:k])))
        else:
            tv = 0.0
        self._z_ent = float(ent_z)
        self._z_mar = float(mar_z)
        state.update({"proxy_entropy": float(ent), "proxy_margin": float(mar),
                      "entropy_z": float(ent_z), "margin_z": float(mar_z),
                      "pred_hist_tv": float(tv),
                      "pred_hist": [float(v) for v in hist],
                      "ref_entropy": float(self._ref_entropy),
                      "ref_margin": float(self._ref_margin)})

        # ---- 5. Page-Hinkley（标签无关代理误差，检测渐变漂移） ----
        proxy_err = max(0.0, ent_z) + max(0.0, mar_z)
        ph_stat = self._ph_proxy.update(proxy_err)
        state["ph_stat"] = float(ph_stat)
        state["ph_fired"] = bool(self._ph_proxy.fired)
        state["proxy_err"] = float(proxy_err)

        # ---- 6. novelty（kNN 距离比） ----
        novelty_stat, sample_novelty = self._novelty(x)
        state["novelty"] = float(novelty_stat)
        state["novelty_threshold"] = float(self.novelty_threshold)

        # ---- 7. repair 信号（回到基座分布） ----
        repair_score = 0.0
        if (mmd_active > self.mmd_threshold
                and mmd_baseline < self.repair_ratio * self.mmd_threshold):
            repair_score = float(min(1.0, (mmd_active - mmd_baseline)
                                     / max(mmd_active, 1e-9)))
        state["repair_score"] = float(repair_score)

        # ---- 7b. 候选参考域（provisional）抑制重复告警，但不写"已接受参考域" ----
        mmd_prov = float("nan")
        suppressed_by_provisional = False
        if self.provisional_ref is not None and len(x) >= 4:
            prov = self._subsample(self.provisional_ref, self.mmd_subsample,
                                   np.random.default_rng(t + 21))
            mmd_prov = float(mmd_rbf(x, prov, max_n=self.mmd_subsample, seed=t + 22))
            state["mmd_provisional"] = mmd_prov
            if mmd_prov <= self.mmd_threshold:
                suppressed_by_provisional = True
            else:
                # 数据再次变化：候选参考域失效，清空后重新判定
                self.provisional_ref = None
                self._stable_count = 0

        # ---- 8. 组装无标签事件 ----
        if not warmup:
            sev = float(min(1.0, max(0.0, (mmd_active - self.mmd_threshold)
                                     / max(self.mmd_threshold, 1e-9))))
            if mmd_active > self.mmd_threshold:
                events.append(DriftEvent(
                    t=t, phase=phase, detector="mmd", statistic=float(mmd_active),
                    threshold=float(self.mmd_threshold), p_value=None, severity=sev,
                    features=[int(i) for i in ks_features[: self.ks_topk]],
                    note=f"MMD 对 active 参考={mmd_active:.3f}，对基座参考={mmd_baseline:.3f}"))
            if (ks_count >= self.ks_count_threshold
                    and ks_mean >= self.ks_stat_threshold):
                events.append(DriftEvent(
                    t=t, phase=phase, detector="ks_topk",
                    statistic=float(ks_mean),
                    threshold=float(self.ks_stat_threshold),
                    p_value=None,
                    severity=float(min(1.0, 2.0 * ks_count / max(self.n_features, 1))),
                    features=[int(i) for i in ks_features[: self.ks_topk]],
                    note=f"{ks_count}/{self.n_features} 个特征 KS p<{self.ks_alpha}"))
            if self._ph_proxy.fired:
                events.append(DriftEvent(
                    t=t, phase=phase, detector="page_hinkley_proxy",
                    statistic=float(ph_stat), threshold=float(self._ph_proxy.threshold),
                    p_value=None, severity=float(min(1.0, ph_stat / (2.0 * self._ph_proxy.threshold))),
                    features=[], note=f"代理误差 z 累积 PH={ph_stat:.2f}（渐变漂移）"))
            if novelty_stat > self.novelty_threshold:
                events.append(DriftEvent(
                    t=t, phase=phase, detector="novelty",
                    statistic=float(novelty_stat), threshold=float(self.novelty_threshold),
                    p_value=None,
                    severity=float(min(1.0, novelty_stat / max(2.0 * self.novelty_threshold, 1e-6))),
                    features=[], note="kNN 距离比超出阈值（可能是新故障模式/新工况）"))
            if repair_score > 0.0 and self._repair_pending >= self.repair_patience:
                events.append(DriftEvent(
                    t=t, phase=phase, detector="repair",
                    statistic=float(repair_score), threshold=float(self.repair_ratio),
                    p_value=None, severity=float(min(1.0, repair_score)),
                    features=[], note="数据分布回到基座参考（维修复位）"))
        if repair_score > 0.0:
            self._repair_pending += 1
        else:
            self._repair_pending = 0

        # ---- 9. 预算内标注探针 ----
        self._probe_suspicious = bool(events)
        self._current_events_flag = bool(events)
        self._current_mmd = float(mmd_active)
        if self._should_probe(t, warmup, events):
            probe = self._run_probe(chunk, model, oracle, t)
            cost = cost.add(probe.get("cost", Cost()))
            if probe:
                state.update(probe["state"])
                pe = probe.get("events") or []
                events.extend(pe)
        state["probe_acc_ref"] = (float(self._probe_acc_ref)
                                  if self._probe_acc_ref is not None else float("nan"))
        # 探针准确率"粘滞"：非探针窗口也提供最近一次探针结果，供 loop 门禁/看门狗使用
        state["probe_acc"] = (float(self._last_probe_acc)
                              if self._last_probe_acc is not None else float("nan"))
        state["probe_n"] = int(self._last_probe_n)
        if self._probe_accs:
            state["probe_acc_recent"] = [float(v) for v in self._probe_accs[-4:]]

        # ---- 10. 触发去抖 + 参考窗口重设 ----
        raw_events = list(events)
        if warmup:
            raw_events = list(events)
            events = []
        elif suppressed_by_provisional and events:
            # 只记录"候选域内仍报事件"的抑制意图，不真正压制（保证漂移覆盖不丢失；
            # 参考域更新交由 on_promote 的门禁后语义，见 docs/04 §6 双门不变量）
            state["provisional_overlap"] = True
        if events:
            if t - self.last_event_t <= self.event_cooldown:
                state["events_suppressed"] = True
                events = []
            else:
                self.last_event_t = t
        state["drift_any"] = bool(events)
        state["severity"] = float(max([e.severity for e in events], default=0.0))
        state["detectors"] = sorted({e.detector for e in events})
        state.update(self._direction_state(t))
        if raw_events and not warmup:
            self._trigger_history.append(int(t))
            self._trigger_history = self._trigger_history[-200:]
        self._maybe_rebase(t, x, mmd_active, mmd_baseline, raw_events, state)
        self._update_rolling(x)

        # 事件再次裁剪（重设参考窗口的窗口不再报事件，避免"已接受新域"仍触发）
        if state.get("rebase") and self.event_cooldown > 0:
            pass
        return MonitorReport(events=events, state=to_native(state), cost=cost,
                             aux={"sample_novelty": to_native(sample_novelty),
                                  "proba": proba})

    # ------------------------------------------------------------------
    # 各检测器实现
    # ------------------------------------------------------------------
    def _ensure_proxy_reference(self, model) -> None:
        key = id(model)
        if self._ref_ready and key == self._model_key and self._ref_pred_hist is not None:
            return
        self._model_key = key
        ref = self.baseline_ref if self.baseline_ref is not None else self.active_ref
        if ref is None or len(ref) == 0:
            return
        proba = model.predict_proba(ref)
        ent = marginal_entropy(proba)
        mar = np.sort(proba, axis=1)
        mar = mar[:, -1] - mar[:, -2]
        self._ref_entropy = float(np.mean(ent))
        self._ref_entropy_std = float(np.std(ent)) if np.std(ent) > 1e-6 else 1.0
        self._ref_margin = float(np.mean(mar))
        self._ref_margin_std = float(np.std(mar)) if np.std(mar) > 1e-6 else 1.0
        self._ref_pred_hist = prediction_histogram(proba, max(self.n_classes, model.n_classes))

    def _ks_features(self, x: np.ndarray) -> tuple[float, list[int], int, float]:
        ref = self.active_ref if self.active_ref is not None else self.baseline_ref
        if ref is None or len(ref) < 8 or len(x) < 8:
            return 0.0, [], 0, 0.0
        ref = self._subsample(ref, self.ref_max_samples, np.random.default_rng(31))
        stats: list[float] = []
        for j in range(x.shape[1]):
            d, p = ks_2samp(x[:, j], ref[:, j])
            stats.append(float(d) if p < self.ks_alpha else 0.0)
        order = np.argsort(stats)[::-1]
        top = [int(j) for j in order[: self.ks_topk]]
        sig = [j for j in order if stats[j] > 0.0]
        mean_top = float(np.mean([stats[j] for j in top])) if top else 0.0
        return (float(max(stats)), top, len(sig), mean_top)

    def _corr_change(self, x: np.ndarray) -> float:
        """当前窗口与基座参考的相关系数结构平均绝对变化（工况旋转/重参数化证据）。"""
        ref = self.baseline_ref
        if ref is None or len(x) < 8 or len(ref) < 8:
            return 0.0
        try:
            xs = self._subsample(x, min(len(x), 200), np.random.default_rng(77))
            rs = self._subsample(ref, min(len(ref), 200), np.random.default_rng(78))
            if xs.shape[1] != rs.shape[1]:
                return 0.0
            cx = np.corrcoef(xs, rowvar=False)
            cr = np.corrcoef(rs, rowvar=False)
            cx = np.nan_to_num(cx, nan=0.0)
            cr = np.nan_to_num(cr, nan=0.0)
            n = cx.shape[0]
            iu = np.triu_indices(n, k=1)
            return float(np.mean(np.abs(cx[iu] - cr[iu])))
        except Exception:
            return 0.0

    def _novelty(self, x: np.ndarray) -> tuple[float, np.ndarray]:
        ref = self._novelty_ref_sample
        if ref is None or len(x) == 0 or len(ref) < 4:
            return 0.0, np.zeros(len(x))
        xs = self._subsample(x, min(len(x), 200), np.random.default_rng(99))
        d = _pairwise_sq_dists(xs, ref)
        k = min(self.novelty_k, len(ref))
        kth = np.sqrt(np.partition(d, k - 1, axis=1)[:, k - 1])
        dist = kth / max(self._knn_ref_scale, 1e-9)
        stat = float(np.mean(dist) - 1.0)
        return stat, dist

    def _should_probe(self, t: int, warmup: bool, events: list) -> bool:
        """探针策略：代理指标可疑时立刻探针；否则每 probe_control_every 窗口做一次
        控制探针（预算优先留给 playbook，见 docs/04 §6 与 docs/05 §2）。"""
        if self.probe_every <= 0 and self.probe_control_every <= 0:
            return False
        gap = t - self.last_probe_t
        suspicious = (bool(events) or self._ph_proxy.fired
                      or self._z_ent >= 1.5 or self._z_mar >= 1.5)
        if warmup:
            return gap >= 2
        if suspicious and gap >= max(self.min_gap, 2):
            return True
        if self.probe_control_every > 0 and gap >= self.probe_control_every:
            return True
        return False

    def _probe_budget_reserve(self) -> int:
        reserve = int(self.adapt_reserve)
        try:
            total = int(self._oracle_total_labels())
            reserve = max(reserve, int(self.adapt_reserve_frac * total))
        except Exception:
            pass
        return reserve

    def _oracle_total_labels(self) -> int:
        return int(self._total_labels) if getattr(self, "_total_labels", 0) else 0

    def _run_probe(self, chunk, model, oracle, t: int) -> dict:
        if oracle is None:
            return {}
        labeler = get_labeler(oracle)
        if labeler is None:
            return {}
        remaining = labeler.remaining()
        reserve = self._probe_budget_reserve()
        if remaining < reserve + self.probe_min_labels:
            return {}
        suspicious = (self._ph_proxy.fired or self._z_ent >= 1.5
                      or self._z_mar >= 1.5)
        try:
            suspicious = suspicious or bool(getattr(self, "_probe_suspicious", False))
        except Exception:
            pass
        cap = remaining - reserve
        n = int(min(self.probe_labels, cap, int(self.mcfg.get("probe_max_labels", 4))))
        self._last_probe_was_control = not suspicious
        if n < min(self.probe_min_labels, 3):
            return {}
        got = labeler.acquire(chunk, n, reason=f"monitor_probe@t{t}")
        y = got["y"]
        x = got["x"]
        # 无论样本是否够验证，本窗口已经花过标签：必须更新 last_probe_t，否则会逐窗口重复探针
        self.last_probe_t = t
        if len(y) < min(self.probe_min_labels, 3):
            return {}
        self.last_probe_t = t
        pred = model.predict(x)
        # 部署口径准确率：model 不认识的类（novel）必然判错，必须计入错误率
        acc = float(np.mean(pred == y)) if len(y) else 0.0
        n_classes_eval = max(self.n_classes, model.n_classes)
        rec = per_class_recall(np.clip(y, 0, n_classes_eval - 1), pred, n_classes_eval)
        f1 = macro_f1(np.clip(y, 0, n_classes_eval - 1), pred, n_classes_eval)
        self._probe_accs.append(acc)
        self._last_probe_acc = acc
        self._last_probe_n = int(len(y))
        first_probe = self._probe_acc_ref is None
        if first_probe:
            # 首探针样本少（6 条）→ 用保守下限 0.85 作为参考准确率初值
            self._probe_acc_ref = max(acc, 0.85)
            self._warmup_probe_accs = [acc]
        elif len(getattr(self, "_warmup_probe_accs", [])) < 2:
            self._warmup_probe_accs = getattr(self, "_warmup_probe_accs", []) + [acc]
            mean_acc = float(np.mean(self._warmup_probe_accs))
            self._probe_acc_ref = max(mean_acc, 0.85)
        elif (not bool(getattr(self, "_current_events_flag", False))
              and float(getattr(self, "_current_mmd", 1.0)) < self.mmd_threshold
              and abs(float(getattr(self, "_z_ent", 0.0))) < 1.0
              and abs(float(getattr(self, "_z_mar", 0.0))) < 1.0
              and self._probe_acc_ref - acc < self.proxy_drop_threshold):
            # 仅在"健康"窗口缓慢更新参考准确率（防止把持续劣化吸收成新常态——
            # 即 notes/学习_dyn3 映射里的"接受一切"失效方向）
            self._probe_acc_ref = 0.92 * self._probe_acc_ref + 0.08 * acc
        events: list[DriftEvent] = []
        err = float(1.0 - acc)
        state = {
            "probe_acc": float(acc),
            "probe_macro_f1": float(f1),
            "probe_n": int(len(y)),
            "probe_novel_frac": float(np.mean(y >= model.n_classes)),
            "probe_error": err,
            "probe_per_class_recall": {str(k): float(v) for k, v in rec.items()},
            "probe_ids": [int(i) for i in got["ids"]],
            "probe_cost_labels": int(got["cost"].labels),
            "probe_ref_acc": float(self._probe_acc_ref),
        }
        if (not first_probe
                and self._probe_acc_ref - acc > self.proxy_drop_threshold):
            events.append(DriftEvent(
                t=int(t), phase=int(getattr(chunk, "phase", 0)),
                detector="probe_error", statistic=float(self._probe_acc_ref - acc),
                threshold=float(self.proxy_drop_threshold), p_value=None,
                severity=float(min(1.0, (self._probe_acc_ref - acc)
                                   / max(2.0 * self.proxy_drop_threshold, 1e-6))),
                features=[], note=f"探针准确率 {acc:.2f} < 参考 {self._probe_acc_ref:.2f}"))
        # 探针误差 PH（检测渐变劣化；只在有探针时更新）
        if self._probe_acc_ref is not None and len(self._probe_accs) >= 2:
            z = (err - (1.0 - self._probe_acc_ref)) / 0.1
            self._ph_probe.update(z)
            state["ph_probe_stat"] = float(self._ph_probe.stat)
            if self._ph_probe.fired:
                events.append(DriftEvent(
                    t=int(t), phase=int(getattr(chunk, "phase", 0)),
                    detector="page_hinkley_probe", statistic=float(self._ph_probe.stat),
                    threshold=float(self._ph_probe.threshold), p_value=None,
                    severity=float(min(1.0, self._ph_probe.stat
                                       / max(2.0 * self._ph_probe.threshold, 1e-6))),
                    features=[], note="探针误差 PH 报警（渐变劣化）"))
        return {"state": to_native(state), "events": events,
                "cost": got["cost"]}

    # ------------------------------------------------------------------
    # 参考窗口重设
    # ------------------------------------------------------------------
    def _update_rolling(self, x: np.ndarray) -> None:
        self.rolling.append(self._subsample(x, self.rolling_samples,
                                            np.random.default_rng(len(self.rolling) + 7)))
        if len(self.rolling) > self.rolling_len:
            self.rolling = self.rolling[-self.rolling_len:]

    def _maybe_rebase(self, t: int, x: np.ndarray, mmd_active: float,
                      mmd_baseline: float, events: list, state: dict) -> None:
        """新域稳定判定：当前窗口与滚动窗口互相相似、但与 active 参考差异大 → 重设 active 参考。"""
        if len(self.rolling) < 2 or self.active_ref is None:
            return
        recent = np.vstack(self.rolling[-2:]) if self.rolling else x
        if len(recent) > 400:
            recent = self._subsample(recent, 400, np.random.default_rng(t + 5))
        mmd_recent = mmd_rbf(x, recent, max_n=300, seed=t + 6) if len(x) >= 4 else 0.0
        state["mmd_recent"] = float(mmd_recent)
        changed = bool(events) or mmd_active > self.mmd_threshold
        stable = mmd_recent < self.stability_ratio * max(self.mmd_threshold, 1e-6)
        if changed and stable:
            self._stable_count += 1
            state["stable_count"] = int(self._stable_count)
            if self._stable_count >= self.ref_patience:
                # "双门不变量"：只写候选参考域（provisional），不写已接受参考域；
                # 已接受参考域仅在门禁通过（on_promote）后更新。
                pool = np.vstack(self.rolling[-self.ref_max_chunks:] + [x]) \
                    if self.rolling else x
                self.provisional_ref = self._subsample(
                    pool, self.ref_max_samples, np.random.default_rng(t + 11))
                self._stable_count = 0
                state["provisional_ref_update"] = True
                state["provisional_t"] = int(t)
        elif not changed:
            self._stable_count = 0

    # ------------------------------------------------------------------
    # 诊断辅助：参考分布 / 样本级 novelty
    # ------------------------------------------------------------------
    def reference_sample(self, max_n: int = 600) -> Optional[np.ndarray]:
        ref = self.baseline_ref if self.baseline_ref is not None else self.active_ref
        if ref is None:
            return None
        return self._subsample(ref, max_n, np.random.default_rng(202))

    def feature_stats(self, x: np.ndarray) -> dict:
        """当前窗口逐特征均值/标准差与基线参考的比值（诊断噪声膨胀/噪声证据）。"""
        ref = self.baseline_ref
        x = np.asarray(x, dtype=float)
        if ref is None or len(x) == 0:
            return {"std_ratio_mean": 1.0, "std_ratio_top": [], "mean_shift_top": []}
        ref_sub = self._subsample(ref, self.ref_max_samples, np.random.default_rng(55))
        ref_std = np.std(ref_sub, axis=0) + 1e-9
        cur_std = np.std(x, axis=0) + 1e-9
        ratio = cur_std / ref_std
        order = np.argsort(ratio)[::-1]
        mean_shift = np.abs(np.mean(x, axis=0) - np.mean(ref_sub, axis=0)) / ref_std
        return {"std_ratio_mean": float(np.mean(ratio)),
                "std_ratio_median": float(np.median(ratio)),
                "std_ratio_top": [int(i) for i in order[: self.ks_topk]],
                "std_ratio_top_values": [float(ratio[i]) for i in order[:5]],
                "mean_shift_top": [int(i) for i in np.argsort(mean_shift)[::-1][: self.ks_topk]],
                "mean_shift_ratio_mean": float(np.mean(mean_shift)),
                "mean_shift_ratio_max": float(np.max(mean_shift)) if len(mean_shift) else 0.0,
                "noise_frac": float(np.mean(ratio > 1.25))}

    # ------------------------------------------------------------------
    # 外部通知
    # ------------------------------------------------------------------
    def on_promote(self, model=None) -> None:
        """门禁通过 / 回滚后由 loop 调用：**此时才更新已接受参考域**。

        - 若存在 provisional（新域已稳定但尚未通过门禁）→ 接受它作为 active 参考；
        - 否则用最近滚动窗口作为 active 参考（已适应的域）；
        - 刷新代理基线、清空探针参考与 PH（新 champion 需要重新建立性能基线）。
        """
        self._accepted_updates += 1
        if model is not None:
            self._model_watch_key = id(model)
        pool = None
        if self.provisional_ref is not None:
            pool = self.provisional_ref
        elif self.rolling:
            pool = np.vstack(self.rolling)
        if pool is not None and len(pool):
            self.active_ref = self._subsample(pool, self.ref_max_samples,
                                              np.random.default_rng(4242))
        self.provisional_ref = None
        self._stable_count = 0
        self._model_key = None
        self._ref_pred_hist = None
        self._probe_acc_ref = None
        self._probe_accs = []
        self._last_probe_acc = None
        self._last_probe_n = 0
        self._ph_proxy.reset()
        self._ph_probe.reset()
        self._n_update = max(self._n_update, self.warmup_chunks)

    def promote_reference(self) -> None:
        """同 on_promote 的参考域更新语义（兼容旧调用名）。"""
        self.on_promote()

    def _on_external_model_change(self) -> None:
        """监测到模型对象变化但未收到 on_promote：只刷新代理基线，不写参考域。"""
        self._model_key = None
        self._ref_pred_hist = None
        self._probe_acc_ref = None
        self._probe_accs = []
        self._last_probe_acc = None
        self._last_probe_n = 0
        self._ph_proxy.reset()
        self._ph_probe.reset()

    def note_decision(self, accepted: bool, t: int = -1) -> None:
        """供 loop/工具记录门禁结论（可选）：用于失效方向指示器。"""
        if accepted:
            self._accept_history.append(int(t))
        else:
            self._reject_history = getattr(self, "_reject_history", [])
            self._reject_history.append(int(t))

    def _direction_state(self, t: int) -> dict:
        """失效方向指示器：接受一切 / 拒绝一切 / 均衡（审计与回滚判断用）。"""
        window = int(self.cfg["monitor"].get("direction_window", 12))
        n_trig = sum(1 for x in self._trigger_history if t - x < window)
        n_acc = sum(1 for x in self._accept_history if t - x < window)
        n_sup = sum(1 for x in self._suppress_history if t - x < window)
        n_rej = sum(1 for x in getattr(self, "_reject_history", []) if t - x < window)
        accept_rate = float(n_acc / max(1, n_trig))
        reject_rate = float(1.0 - accept_rate) if n_trig else 0.0
        direction = "balanced"
        if n_trig >= 2 and accept_rate >= 0.8:
            direction = "accept_all"
        elif n_trig >= 2 and accept_rate <= 0.01 and n_rej + n_sup >= 2:
            direction = "reject_all"
        return {"accept_rate": accept_rate, "reject_rate": reject_rate,
                "n_trigger_recent": int(n_trig), "n_accept_recent": int(n_acc),
                "n_reject_recent": int(n_rej), "n_suppressed_recent": int(n_sup),
                "failure_direction": direction,
                "provisional": bool(self.provisional_ref is not None),
                "n_accepted_total": int(self._accepted_updates)}


def _pairwise_sq_dists(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    aa = np.sum(a * a, axis=1)[:, None]
    bb = np.sum(b * b, axis=1)[None, :]
    d = aa + bb - 2.0 * (a @ b.T)
    return np.maximum(d, 0.0)
