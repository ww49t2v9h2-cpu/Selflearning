"""Playbook 动作库：9 个可跑的模型层/策略层适应动作。

对应 docs/03 §4.3 契约：
    PlaybookLibrary(cfg, oracle, cost_model)
    .specs() -> list[PlaybookSpec]
    .estimate_cost(name, diagnosis, chunk, budget) -> Cost
    .run(name, ctx: PlaybookContext) -> ExperimentResult

工程要点（详见 docs/05）
------------------------
1. **统一实验协议**：先买"评估留出集"（labels_eval，候选/冠军在同一批标签上比较，
   保证门禁公平），再买"训练集"，二者 id 不相交；标签全程走 oracle 预算 +
   共享缓存（同一 id 只付费一次）。
2. **预算自适应**：`plan_labels()` 按剩余标签与配置下限裁剪训练标签数，预算不足时
   昂贵 playbook 返回不可行（由 planner 过滤），不会越权超支。
3. **成本核算**：`Cost.labels` 报告实际新购标签数（oracle 已扣预算，本层不重复扣），
   `Cost.compute_units = sample_updates / base_compute_unit`（由 loop 统一记账）。
4. **指标齐全**：每个 ExperimentResult.metrics 至少含
   probe_macro_f1 / baseline_probe_macro_f1 / pred_stability / class_recall_min / cost。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import numpy as np

from ..core.metrics import label_prior_em
from ..core.model import MLP, class_weights_from_labels, expand_model_classes
from ..core.spec import Cost, PlaybookSpec, make_cost
from .defaults import resolve_config
from .utils import (BudgetedLabeler, compute_cost, eval_probe, get_labeler,
                    pred_stability, sum_cost, to_native)


class InfeasiblePlaybook(RuntimeError):
    """预算/标签不足以执行该 playbook（planner 应已过滤，此处是最后一道硬门）。"""


@dataclass
class PlaybookContext:
    """一次候选实验的上下文（字段由 docs/03 §4.3 冻结）。"""

    model: MLP
    chunk: Any
    base_x: np.ndarray
    base_y: np.ndarray
    config: dict
    n_classes: int
    history: list = field(default_factory=list)
    diagnosis: Any = None
    budget: Any = None
    rng: Any = None
    probe_x: Optional[np.ndarray] = None
    probe_y: Optional[np.ndarray] = None
    # 可选扩展字段（向后兼容；loop 可传入 registry/params/playbook 以支持回滚与参数覆盖）
    registry: Any = None
    params: dict = field(default_factory=dict)
    playbook: str = ""


@dataclass
class ExperimentResult:
    model: Optional[MLP]
    metrics: dict = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    artifacts: list = field(default_factory=list)
    params: dict = field(default_factory=dict)
    note: str = ""


# --------------------------------------------------------------------------
# 静态 playbook 元信息（与 docs/03 §4.3 表格一致）
# --------------------------------------------------------------------------
_PLAYBOOK_META: dict[str, dict] = {
    "NOOP": {"layer": "model", "risk": "low", "needs_labels": False,
             "needs_compute": False,
             "description": "不更新，仅记录当前 champion 指标（成本 0）",
             "applicable_kinds": ["benign", "repair", "label_shift"]},
    "RECALIBRATE_AFFINE": {"layer": "model", "risk": "low", "needs_labels": False,
                           "needs_compute": False,
                           "description": "无标签目标域统计量仿射校准（align_predict_proba）",
                           "applicable_kinds": ["covariate_affine", "noise_inflate",
                                                "benign", "repair", "condition_switch"]},
    "PRIOR_CORRECTION": {"layer": "model", "risk": "low", "needs_labels": False,
                         "needs_compute": False,
                         "description": "EM/BBSE 估计类别先验并对 logits 做先验校正",
                         "applicable_kinds": ["label_shift", "concept", "novel_class"]},
    "LAST_LAYER_FT": {"layer": "model", "risk": "low", "needs_labels": True,
                      "needs_compute": True,
                      "description": "冻结隐藏层，仅用少量标签训练最后一层",
                      "applicable_kinds": ["covariate_affine", "condition_switch",
                                           "concept", "noise_inflate", "label_shift"]},
    "FULL_FINETUNE_REPLAY": {"layer": "model", "risk": "medium",
                             "needs_labels": True, "needs_compute": True,
                             "description": "全量微调 + 基座域回放缓冲，标签预算内",
                             "applicable_kinds": ["condition_switch", "concept",
                                                  "covariate_affine", "noise_inflate"]},
    "RETRAIN_MIXED": {"layer": "model", "risk": "medium", "needs_labels": True,
                      "needs_compute": True,
                      "description": "从零在回放+新标注混合集重训",
                      "applicable_kinds": ["concept", "condition_switch"]},
    "NOVEL_CLASS_EXTEND": {"layer": "model", "risk": "high", "needs_labels": True,
                           "needs_compute": True,
                           "description": "检测新类→扩展输出层→带标签训练",
                           "applicable_kinds": ["novel_class", "concept"]},
    "SHADOW_ONLY": {"layer": "strategy", "risk": "low", "needs_labels": True,
                    "needs_compute": True,
                    "description": "候选模型仅做影子评估，不替换 champion",
                    "applicable_kinds": ["benign", "noise_inflate", "concept",
                                         "novel_class", "condition_switch"]},
    "ROLLBACK": {"layer": "strategy", "risk": "medium", "needs_labels": False,
                 "needs_compute": True,
                 "description": "champion 回退到历史稳定版本（无可访问注册表时回退到基座重训）",
                 "applicable_kinds": ["repair", "benign"]},
}


class PlaybookLibrary:
    def __init__(self, cfg: dict, oracle=None, cost_model: Optional[dict] = None):
        self.cfg = resolve_config(cfg)
        self.raw_cfg = cfg or {}
        self.pcfg = self.cfg["playbooks"]
        self.gates_cfg = self.cfg["gates"]
        self.mcfg = self.cfg.get("model", {})
        self.oracle = oracle
        self.cost_model = cost_model or self.cfg.get("cost_model", {})
        self.base_compute_unit = float(self.cost_model.get("base_compute_unit", 4000.0)) or 4000.0
        seed = int(self.cfg.get("seed", 0))
        self.labeler: Optional[BudgetedLabeler] = get_labeler(oracle, seed=seed)
        self.seed = seed
        self._defaults = {
            "reserve_labels": int(self.pcfg.get("reserve_labels", 24)),
            "eval_frac": float(self.pcfg.get("eval_frac", 0.25)),
        }

    # ------------------------------------------------------------------
    # 规格与成本估计
    # ------------------------------------------------------------------
    def specs(self) -> list[PlaybookSpec]:
        out = []
        for name, meta in _PLAYBOOK_META.items():
            cost, _ = self._nominal_cost(name)
            out.append(PlaybookSpec(
                name=name, layer=meta["layer"], description=meta["description"],
                est_cost=cost, risk=meta["risk"], needs_labels=meta["needs_labels"],
                needs_compute=meta["needs_compute"],
                applicable_kinds=list(meta["applicable_kinds"]),
                contract=[f"预算内标签 {cost.labels}", f"compute_units≈{cost.compute_units:.2f}"],
            ))
        return out

    def _p(self, name: str) -> dict:
        base = {"labels": 0, "labels_eval": 0, "min_train": 0}
        merged = dict(base)
        merged.update(self.pcfg.get(name, {}) or {})
        return merged

    def _nominal_cost(self, name: str) -> tuple[Cost, bool]:
        p = self._p(name)
        n_eval = int(p.get("labels_eval", 0))
        n_train = int(p.get("labels", 0))
        est_compute = self._estimate_compute(name, p)
        return make_cost(compute_units=est_compute, labels=n_eval + n_train), True

    def _estimate_compute(self, name: str, p: dict) -> float:
        epochs = float(p.get("epochs", 0))
        if name in ("NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION", "SHADOW_ONLY"):
            if name != "SHADOW_ONLY":
                return 0.0
            sub = str(p.get("sub_playbook", "RECALIBRATE_AFFINE"))
            return self._estimate_compute(sub, self._p(sub))
        n_train = int(p.get("labels", 0))
        if name in ("FULL_FINETUNE_REPLAY", "NOVEL_CLASS_EXTEND"):
            n = n_train + int(p.get("replay_n", 0))
        elif name == "RETRAIN_MIXED":
            n = n_train + int(p.get("replay_n", 0))
        elif name == "ROLLBACK":
            n = int(p.get("rollback_n", 4000))
        else:
            n = n_train
        return float(n * epochs / self.base_compute_unit)

    def plan_labels(self, name: str, diagnosis=None, chunk=None, budget=None) -> dict:
        """按剩余预算给出 (n_eval, n_train)，并判断是否可行。"""
        p = self._p(name)
        ev_cfg = int(p.get("labels_eval", 0))
        train_cfg = int(p.get("labels", 0))
        min_train = int(p.get("min_train", 0))
        max_labels = int(p.get("max_labels", 10 ** 6))
        left = self._labels_left(budget)
        reserve = self._defaults["reserve_labels"]
        cap = max(0, left - reserve)
        min_eval = int(p.get("min_eval",
                             min(ev_cfg, int(self.gates_cfg.get("min_labels_for_verdict", 12)))))
        if ev_cfg == 0 and train_cfg == 0 and min_train == 0:
            return {"playbook": name, "n_eval": 0, "n_train": 0, "feasible": True,
                    "reason": "no-label playbook"}
        # 先保证 eval 最低量，再给 train 留够下限
        n_eval = min(ev_cfg, cap)
        if train_cfg > 0 or min_train > 0:
            if cap - n_eval < min_train:
                n_eval = max(min_eval, min(ev_cfg, cap - min_train))
            if cap - n_eval < min_train:
                return {"playbook": name, "n_eval": int(n_eval), "n_train": 0,
                        "feasible": False,
                        "reason": f"标签不足: cap={cap} < eval{min_eval}+min_train{min_train}"}
            n_train = min(train_cfg, cap - n_eval)
            n_train = max(0, min(n_train, max_labels))
            feasible = (n_train >= min_train) and (n_eval >= min_eval)
            return {"playbook": name, "n_eval": int(n_eval), "n_train": int(n_train),
                    "feasible": bool(feasible),
                    "reason": "" if feasible else "labels below minimum"}
        feasible = n_eval >= min_eval
        return {"playbook": name, "n_eval": int(n_eval), "n_train": 0,
                "feasible": bool(feasible), "reason": "" if feasible else "eval labels below minimum"}

    def _labels_left(self, budget) -> int:
        if budget is not None:
            try:
                return int(budget.labels_left())
            except Exception:
                pass
        if self.labeler is not None:
            return int(self.labeler.remaining())
        # 无预算/无 oracle：返回名义可行域，便于 specs()/离线估计
        return 10 ** 6

    def estimate_cost(self, name: str, diagnosis=None, chunk=None, budget=None) -> Cost:
        plan = self.plan_labels(name, diagnosis, chunk, budget)
        if not plan["feasible"]:
            # 不可行：故意返回超出当前预算的成本，让 planner/loop 的硬过滤拒绝
            left = self._labels_left(budget)
            return make_cost(compute_units=1e9, labels=left + 1000)
        p = self._p(name)
        est_compute = self._estimate_compute(name, p)
        return make_cost(compute_units=est_compute,
                         labels=int(plan["n_eval"] + plan["n_train"]))

    # ------------------------------------------------------------------
    # 入口
    # ------------------------------------------------------------------
    def run(self, name: str, ctx: PlaybookContext) -> ExperimentResult:
        if name not in _PLAYBOOK_META:
            raise KeyError(f"未知 playbook: {name}")
        plan = self.plan_labels(name, ctx.diagnosis, ctx.chunk, ctx.budget)
        if not plan["feasible"]:
            raise InfeasiblePlaybook(f"{name}: {plan.get('reason', 'infeasible')}")
        handler = {
            "NOOP": self._run_noop,
            "RECALIBRATE_AFFINE": self._run_recalibrate,
            "PRIOR_CORRECTION": self._run_prior_correction,
            "LAST_LAYER_FT": self._run_last_layer,
            "FULL_FINETUNE_REPLAY": self._run_full_finetune,
            "RETRAIN_MIXED": self._run_retrain_mixed,
            "NOVEL_CLASS_EXTEND": self._run_novel_extend,
            "SHADOW_ONLY": self._run_shadow,
            "ROLLBACK": self._run_rollback,
        }[name]
        res = handler(ctx, plan)
        if res.params is None:
            res.params = {}
        res.params.setdefault("label_plan", plan)
        res.params.setdefault("seed", int(self.seed))
        res.params.setdefault("n_classes_benchmark", int(ctx.n_classes))
        return res

    # ------------------------------------------------------------------
    # 实验公共流程
    # ------------------------------------------------------------------
    def _acquire_eval(self, ctx, n_eval: int) -> dict:
        if n_eval <= 0 or self.labeler is None:
            return {"ids": np.zeros(0, dtype=np.int64), "y": np.zeros(0, dtype=np.int64),
                    "x": np.zeros((0, np.asarray(ctx.chunk.x).shape[1])), "cost": Cost()}
        return self.labeler.acquire(ctx.chunk, int(n_eval),
                                    reason=f"eval:{ctx.diagnosis.kind if ctx.diagnosis else 'na'}"
                                           f"@t{getattr(ctx.chunk, 't', 0)}")

    def _acquire_train(self, ctx, n_train: int, exclude_ids=None) -> dict:
        if n_train <= 0 or self.labeler is None:
            d = self._acquire_eval(ctx, 0)
            return d
        got = self.labeler.acquire(ctx.chunk, int(n_train),
                                   reason=f"train@{getattr(ctx.chunk, 't', 0)}",
                                   exclude_ids=exclude_ids)
        # 允许 loop 传进来的免费探针标签补充训练集（不占本次预算）
        if ctx.probe_x is not None and len(getattr(ctx, "probe_y", []) or []) > 0:
            try:
                px = np.asarray(ctx.probe_x, dtype=float)
                py = np.asarray(ctx.probe_y, dtype=np.int64)
                if len(px) == len(py) and len(px) > 0:
                    ex = set(int(i) for i in np.asarray(exclude_ids).tolist()) \
                        if exclude_ids is not None and len(exclude_ids) else set()
                    if not (ex and set(range(len(py))) & ex):
                        got["x"] = np.vstack([got["x"], px])
                        got["y"] = np.concatenate([got["y"], py])
            except Exception:
                pass
        return got

    def _eval_metrics(self, ctx, model, x_eval, y_eval) -> dict:
        n_cls = int(ctx.n_classes)
        cand = eval_probe(model, x_eval, y_eval, n_cls) if len(y_eval) else {
            "n_eval": 0, "probe_macro_f1": float("nan"),
            "probe_accuracy": float("nan"), "class_recall_min": float("nan"),
            "ece": float("nan"), "per_class_recall": {}}
        base = eval_probe(ctx.model, x_eval, y_eval, n_cls) if len(y_eval) else {
            "n_eval": 0, "probe_macro_f1": float("nan"),
            "probe_accuracy": float("nan"), "class_recall_min": float("nan"),
            "ece": float("nan"), "per_class_recall": {}}
        stability = pred_stability(model, ctx.model, np.asarray(ctx.chunk.x, dtype=float))
        return cand, base, stability

    def _finish(self, ctx, model, x_eval, y_eval, cost: Cost, params: dict,
                artifacts: list, note: str, extra: Optional[dict] = None) -> ExperimentResult:
        cand, base, stability = self._eval_metrics(ctx, model, x_eval, y_eval)
        metrics = {
            "probe_macro_f1": cand["probe_macro_f1"],
            "probe_accuracy": cand["probe_accuracy"],
            "class_recall_min": cand["class_recall_min"],
            "per_class_recall": cand.get("per_class_recall", {}),
            "ece": cand["ece"],
            "n_eval": cand["n_eval"],
            "baseline_probe_macro_f1": base["probe_macro_f1"],
            "baseline_probe_accuracy": base["probe_accuracy"],
            "baseline_class_recall_min": base["class_recall_min"],
            "baseline_ece": base["ece"],
            "pred_stability": float(stability),
            "cost": float(cost.scalar),
            "labels": int(cost.labels),
            "compute_units": float(cost.compute_units),
        }
        if extra:
            metrics.update({k: v for k, v in extra.items() if isinstance(v, (int, float, str, bool))})
        return ExperimentResult(model=model, metrics=to_native(metrics), cost=cost,
                                artifacts=list(artifacts), params=dict(params), note=note)

    def _eval_and_train(self, ctx, plan: dict) -> tuple[dict, dict]:
        ev = self._acquire_eval(ctx, plan["n_eval"])
        tr = self._acquire_train(ctx, plan["n_train"], exclude_ids=ev.get("ids"))
        return ev, tr

    def _replay(self, ctx, n: int, exclude_ids=None, rng=None) -> tuple[np.ndarray, np.ndarray]:
        """基座域回放缓冲（历史训练集）；n<=0 时返回空。"""
        n = int(max(0, n))
        bx = np.asarray(ctx.base_x, dtype=float)
        by = np.asarray(ctx.base_y, dtype=np.int64)
        if n <= 0 or len(bx) == 0:
            return np.zeros((0, bx.shape[1] if bx.ndim == 2 else 0)), np.zeros(0, dtype=np.int64)
        if n >= len(bx):
            idx = np.arange(len(bx))
        else:
            rng = rng or np.random.default_rng(self.seed + 17)
            idx = rng.choice(len(bx), size=n, replace=False)
        return bx[idx], by[idx]

    @staticmethod
    def _known_only(x, y, n_model_classes) -> tuple[np.ndarray, np.ndarray]:
        y = np.asarray(y, dtype=np.int64)
        if len(y) == 0:
            return x, y
        mask = (y >= 0) & (y < n_model_classes)
        return np.asarray(x, dtype=float)[mask], y[mask]

    def _class_weights(self, y: np.ndarray, n_classes: int, power: float = 1.0) -> np.ndarray:
        w = class_weights_from_labels(np.asarray(y, dtype=np.int64), n_classes)
        power = float(power)
        if power <= 0:
            return np.ones(n_classes)
        return np.power(w, power)

    def _train_cost(self, sample_updates: int, wall_s: float = 0.0) -> Cost:
        return Cost(compute_units=float(sample_updates) / self.base_compute_unit,
                    wall_s=float(wall_s))

    def _artifact_check(self, model: MLP, x: np.ndarray) -> list:
        """可复现自检：state 往返后预测一致（门禁“可复现”的依据之一）。"""
        try:
            m2 = MLP.from_state({k: np.asarray(v).copy() for k, v in model.state().items()})
            p1 = model.predict_proba(np.asarray(x, dtype=float))
            p2 = m2.predict_proba(np.asarray(x, dtype=float))
            ok = bool(np.max(np.abs(p1 - p2)) <= float(
                self.gates_cfg.get("repro_tol", 1e-6)))
        except Exception:
            ok = False
        return [f"reproducibility:state_roundtrip={ok}", f"classes:{model.n_classes}"]

    # ------------------------------------------------------------------
    # 1) NOOP
    # ------------------------------------------------------------------
    def _run_noop(self, ctx, plan) -> ExperimentResult:
        # 零成本：只读缓存标签评估 champion（不新购标签，避免浪费）
        x_eval = np.zeros((0, np.asarray(ctx.chunk.x).shape[1]))
        y_eval = np.zeros(0, dtype=np.int64)
        cost = Cost()
        if self.labeler is not None:
            ids = np.asarray(ctx.chunk.ids, dtype=np.int64)
            cached = [int(i) for i in ids.tolist() if int(i) in self.labeler.y_by_id]
            if cached:
                got = self.labeler.query_ids(np.asarray(cached, dtype=np.int64),
                                             reason="noop_eval_cache")
                x_eval, y_eval = got["x"], got["y"]
        model = ctx.model.clone()
        return self._finish(ctx, model, x_eval, y_eval, cost,
                            params={"no_update": True},
                            artifacts=["noop"], note="不更新，仅记录 champion 指标")

    # ------------------------------------------------------------------
    # 2) RECALIBRATE_AFFINE
    # ------------------------------------------------------------------
    def _recalibrate_core(self, ctx) -> MLP:
        """测试时自适应归一化（TTA）：不修改权重/不固化目标域 scaler。

        推理时用当前 batch 统计量把目标域仿射对齐回源域尺度，因此同一版本可跨窗口、
        跨阶段泛化；对"传感器老化/工况切换"等仿射成分有效，且零标签零训练成本。
        （Lead 接管修复：原实现把目标域 scaler 固化进模型版本，导致后续阶段失配。）
        """
        p = self._p("RECALIBRATE_AFFINE")
        blend = float(np.clip(p.get("blend", 1.0), 0.0, 1.0))
        model = ctx.model.clone()
        model.adaptive = True
        model.adaptive_blend = blend
        return model

    def _run_recalibrate(self, ctx, plan) -> ExperimentResult:
        ev = self._acquire_eval(ctx, plan["n_eval"])
        model = self._recalibrate_core(ctx)
        cost = sum_cost(ev.get("cost"))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"blend": float(self._p("RECALIBRATE_AFFINE").get("blend", 1.0))},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note="测试时自适应归一化（TTA，零标签）")

    # ------------------------------------------------------------------
    # 3) PRIOR_CORRECTION
    # ------------------------------------------------------------------
    def _run_prior_correction(self, ctx, plan) -> ExperimentResult:
        p = self._p("PRIOR_CORRECTION")
        ev = self._acquire_eval(ctx, plan["n_eval"])
        model = ctx.model.clone()
        x = np.asarray(ctx.chunk.x, dtype=float)
        prior_hat, _ = label_prior_em(model.predict_proba(x), model.n_classes)
        prior_hat = np.clip(prior_hat, 1e-6, None)
        prior_hat = prior_hat / prior_hat.sum()
        strength = float(p.get("strength", 1.0))
        temp = max(float(p.get("posterior_temp", 1.0)), 1e-6)
        last = f"b{len(model.hidden) + 1}"
        adj = strength * np.log(prior_hat) / temp
        model.params[last] = model.params[last] + adj
        cost = sum_cost(ev.get("cost"))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"prior_hat": [float(v) for v in prior_hat],
                                    "strength": strength},
                            artifacts=self._artifact_check(model, x),
                            note="EM/BBSE 类先验估计 + logits 先验校正")

    # ------------------------------------------------------------------
    # 4) LAST_LAYER_FT
    # ------------------------------------------------------------------
    def _run_last_layer(self, ctx, plan) -> ExperimentResult:
        p = self._p("LAST_LAYER_FT")
        ev, tr = self._eval_and_train(ctx, plan)
        x_tr, y_tr = self._known_only(tr["x"], tr["y"], ctx.model.n_classes)
        if len(y_tr) < max(8, int(p.get("min_train", 32)) // 3):
            # 已知类标签太少：退化为仿射校准（无标签），避免崩坏
            model = self._recalibrate_core(ctx)
            cost = sum_cost(ev.get("cost"), tr.get("cost"))
            return self._finish(ctx, model, ev["x"], ev["y"], cost,
                                params={"fallback": "recalibrate_affine"},
                                artifacts=["fallback_affine"],
                                note="已知类训练标签不足，回退仿射校准")
        model = ctx.model.clone()
        oversample = float(p.get("oversample", 1.0) or 1.0)
        if oversample != 1.0:
            w = self._class_weights(y_tr, ctx.model.n_classes,
                                    float(p.get("class_weight_power", 1.0)))
            w = np.power(w, max(oversample, 0.0))
            rng = ctx.rng if ctx.rng is not None else np.random.default_rng(self.seed)
            prob = np.asarray(w[y_tr], dtype=float)
            prob = prob / prob.sum()
            idx = rng.choice(len(y_tr), size=len(y_tr), replace=True, p=prob)
            x_tr, y_tr = x_tr[idx], y_tr[idx]
        res = model.fit_head(
            x_tr, y_tr,
            epochs=int(p.get("epochs", 80)),
            lr=float(p.get("lr", 0.01)),
            batch=int(p.get("batch", 64)),
            seed=self.seed,
        )
        cost = sum_cost(ev.get("cost"), tr.get("cost"),
                        self._train_cost(res.sample_updates, res.wall_s))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"epochs": int(p.get("epochs", 80)),
                                    "lr": float(p.get("lr", 0.01)),
                                    "n_train_known": int(len(y_tr)),
                                    "refit_scaler": False},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note="冻结隐藏层，仅最后一层在少量标签上微调")

    # ------------------------------------------------------------------
    # 5) FULL_FINETUNE_REPLAY
    # ------------------------------------------------------------------
    def _run_full_finetune(self, ctx, plan) -> ExperimentResult:
        p = self._p("FULL_FINETUNE_REPLAY")
        ev, tr = self._eval_and_train(ctx, plan)
        x_tr, y_tr = self._known_only(tr["x"], tr["y"], ctx.model.n_classes)
        if len(y_tr) < 8:
            model = self._recalibrate_core(ctx)
            cost = sum_cost(ev.get("cost"), tr.get("cost"))
            return self._finish(ctx, model, ev["x"], ev["y"], cost,
                                params={"fallback": "recalibrate_affine"},
                                artifacts=["fallback_affine"],
                                note="训练标签不足，回退仿射校准")
        rp_x, rp_y = self._replay(ctx, int(p.get("replay_n", 900)), rng=ctx.rng)
        x_ft = np.vstack([rp_x, x_tr]) if len(rp_x) else x_tr
        y_ft = np.concatenate([rp_y, y_tr]) if len(rp_x) else y_tr
        model = ctx.model.clone()
        cw = self._class_weights(y_ft, ctx.model.n_classes,
                                 float(p.get("class_weight_power", 0.5)))
        res = model.fit(x_ft, y_ft,
                        epochs=int(p.get("epochs", 25)),
                        lr=float(p.get("lr", 0.002)),
                        batch=int(p.get("batch", 128)),
                        class_weights=cw, seed=self.seed,
                        refit_scaler=bool(p.get("refit_scaler", False)))
        cost = sum_cost(ev.get("cost"), tr.get("cost"),
                        self._train_cost(res.sample_updates, res.wall_s))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"epochs": int(p.get("epochs", 25)),
                                    "lr": float(p.get("lr", 0.002)),
                                    "replay_n": int(p.get("replay_n", 900)),
                                    "n_train_known": int(len(y_tr)),
                                    "refit_scaler": bool(p.get("refit_scaler", False))},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note="全量微调 + 基座域回放缓冲")

    # ------------------------------------------------------------------
    # 6) RETRAIN_MIXED
    # ------------------------------------------------------------------
    def _run_retrain_mixed(self, ctx, plan) -> ExperimentResult:
        p = self._p("RETRAIN_MIXED")
        ev, tr = self._eval_and_train(ctx, plan)
        x_tr, y_tr = self._known_only(tr["x"], tr["y"], ctx.model.n_classes)
        if len(y_tr) < 8:
            model = self._recalibrate_core(ctx)
            cost = sum_cost(ev.get("cost"), tr.get("cost"))
            return self._finish(ctx, model, ev["x"], ev["y"], cost,
                                params={"fallback": "recalibrate_affine"},
                                artifacts=["fallback_affine"],
                                note="训练标签不足，回退仿射校准")
        rp_x, rp_y = self._replay(ctx, int(p.get("replay_n", 1600)), rng=ctx.rng)
        x_all = np.vstack([rp_x, x_tr]) if len(rp_x) else x_tr
        y_all = np.concatenate([rp_y, y_tr]) if len(rp_y) else y_tr
        hidden = tuple(self.cfg.get("model", {}).get("hidden", [64, 32]))
        model = MLP(x_all.shape[1], ctx.model.n_classes, hidden=hidden, seed=self.seed,
                    l2=float(self.cfg.get("model", {}).get("l2", 1e-4)))
        cw = self._class_weights(y_all, ctx.model.n_classes,
                                 float(p.get("class_weight_power", 0.5)))
        res = model.fit(x_all, y_all,
                        epochs=int(p.get("epochs", 35)),
                        lr=float(p.get("lr", 0.0035)),
                        batch=int(p.get("batch", 128)),
                        class_weights=cw, seed=self.seed,
                        refit_scaler=bool(p.get("refit_scaler", True)))
        cost = sum_cost(ev.get("cost"), tr.get("cost"),
                        self._train_cost(res.sample_updates, res.wall_s))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"epochs": int(p.get("epochs", 35)),
                                    "lr": float(p.get("lr", 0.0035)),
                                    "replay_n": int(p.get("replay_n", 1600)),
                                    "n_train_known": int(len(y_tr))},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note="从零在回放+新标注混合集重训")

    # ------------------------------------------------------------------
    # 7) NOVEL_CLASS_EXTEND
    # ------------------------------------------------------------------
    def _run_novel_extend(self, ctx, plan) -> ExperimentResult:
        p = self._p("NOVEL_CLASS_EXTEND")
        ev, tr = self._eval_and_train(ctx, plan)
        y_tr_raw = np.asarray(tr["y"], dtype=np.int64)
        if len(y_tr_raw) == 0:
            raise InfeasiblePlaybook("NOVEL_CLASS_EXTEND: 无训练标签")
        n_target = int(max(ctx.model.n_classes, int(y_tr_raw.max()) + 1))
        if ctx.diagnosis is not None:
            for c in (ctx.diagnosis.affected_classes or []):
                if 0 <= int(c) < int(ctx.n_classes):
                    n_target = max(n_target, int(c) + 1)
        n_target = int(min(max(n_target, ctx.model.n_classes), int(ctx.n_classes)))
        model = expand_model_classes(ctx.model, n_target, seed=self.seed)
        rp_x, rp_y = self._replay(ctx, int(p.get("replay_n", 1000)), rng=ctx.rng)
        x_all = np.vstack([rp_x, tr["x"]]) if len(rp_x) else tr["x"]
        y_all = np.concatenate([rp_y, y_tr_raw]) if len(rp_y) else y_tr_raw
        cw = self._class_weights(y_all, n_target, float(p.get("class_weight_power", 0.5)))
        novel_ids = [i for i in range(ctx.model.n_classes, n_target)]
        if novel_ids:
            mult = float(p.get("novel_weight", 2.0))
            for c in novel_ids:
                cw[c] = cw[c] * mult
        res = model.fit(x_all, y_all,
                        epochs=int(p.get("epochs", 30)),
                        lr=float(p.get("lr", 0.0025)),
                        batch=int(p.get("batch", 128)),
                        class_weights=cw, seed=self.seed,
                        refit_scaler=bool(p.get("refit_scaler", True)))
        cost = sum_cost(ev.get("cost"), tr.get("cost"),
                        self._train_cost(res.sample_updates, res.wall_s))
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"epochs": int(p.get("epochs", 30)),
                                    "lr": float(p.get("lr", 0.0025)),
                                    "replay_n": int(p.get("replay_n", 1000)),
                                    "n_target_classes": int(n_target),
                                    "novel_train_n": int(np.sum(y_tr_raw >= ctx.model.n_classes))},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note=f"扩展输出层到 {n_target} 类并用新标注训练")

    # ------------------------------------------------------------------
    # 8) SHADOW_ONLY
    # ------------------------------------------------------------------
    def _run_shadow(self, ctx, plan) -> ExperimentResult:
        sub = str(self._p("SHADOW_ONLY").get("sub_playbook", "FULL_FINETUNE_REPLAY"))
        sub_plan = self.plan_labels(sub, ctx.diagnosis, ctx.chunk, ctx.budget)
        if not sub_plan["feasible"]:
            raise InfeasiblePlaybook(f"SHADOW_ONLY: 子 playbook {sub} 不可行")
        res = self._dispatch_sub(sub, ctx, sub_plan)
        res.params = dict(res.params or {})
        res.params.update({"shadow": True, "sub_playbook": sub})
        # 影子候选不进入 registry：返回 None，交给 loop 记录为"未上线"
        res.model = None
        res.note = f"影子评估：{sub} 候选，不替换 champion（{res.note}）"
        res.artifacts = list(res.artifacts) + ["shadow_only"]
        return res

    def _dispatch_sub(self, sub: str, ctx, plan) -> ExperimentResult:
        handler = {
            "RECALIBRATE_AFFINE": self._run_recalibrate,
            "PRIOR_CORRECTION": self._run_prior_correction,
            "LAST_LAYER_FT": self._run_last_layer,
            "FULL_FINETUNE_REPLAY": self._run_full_finetune,
            "RETRAIN_MIXED": self._run_retrain_mixed,
        }.get(sub)
        if handler is None:
            raise InfeasiblePlaybook(f"SHADOW_ONLY: 子 playbook {sub} 不支持")
        return handler(ctx, plan)

    # ------------------------------------------------------------------
    # 9) ROLLBACK
    # ------------------------------------------------------------------
    def _run_rollback(self, ctx, plan) -> ExperimentResult:
        p = self._p("ROLLBACK")
        ev = self._acquire_eval(ctx, plan["n_eval"])
        # 优先用注册表回退到上一稳定版本；loop 未传 registry 时回退到基座重训
        model = None
        version = None
        if getattr(ctx, "registry", None) is not None:
            try:
                hist = ctx.registry.history()
                candidates = [m for m in hist if m.status in ("archived", "champion")
                              and int(m.version) != int(ctx.registry.champion()[0].version)]
                if candidates:
                    best = max(candidates, key=lambda m: int(m.version))
                    model = ctx.registry.get(int(best.version))[1].clone()
                    version = int(best.version)
            except Exception:
                model = None
        cost = sum_cost(ev.get("cost"))
        if model is None:
            hidden = tuple(self.cfg.get("model", {}).get("hidden", [64, 32]))
            model = MLP(np.asarray(ctx.base_x).shape[1], ctx.model.n_classes,
                        hidden=hidden, seed=self.seed,
                        l2=float(self.cfg.get("model", {}).get("l2", 1e-4)))
            res = model.fit(np.asarray(ctx.base_x, dtype=float),
                            np.asarray(ctx.base_y, dtype=np.int64),
                            epochs=int(p.get("epochs", 20)),
                            lr=float(p.get("lr", 0.004)),
                            batch=int(p.get("batch", 128)),
                            seed=self.seed, refit_scaler=True)
            cost = cost.add(self._train_cost(res.sample_updates, res.wall_s))
            note = "回退：注册表不可用，重训基座稳定版本（只用基座历史数据）"
        else:
            note = f"回退到注册表版本 v{version}"
        return self._finish(ctx, model, ev["x"], ev["y"], cost,
                            params={"rollback_to": version if version is not None
                                    else "base_retrained",
                                    "epochs": int(p.get("epochs", 20))},
                            artifacts=self._artifact_check(model, np.asarray(ctx.chunk.x)),
                            note=note)
