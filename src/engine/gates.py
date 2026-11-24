"""上线门禁：候选模型相对 champion 的多维判定。

对应 docs/03 §4.5 契约：
    Gate(cfg).evaluate(candidate: dict, incumbent: dict, ctx: dict) -> GateResult

判定项（全部写入 GateResult.criteria，便于审计与答辩解释）
--------------------------------------------------------
1. probe_gain      候选 probe macro-F1 相对 champion 的增益 ≥ min_gain_points；
                   优先用候选自带的 `baseline_probe_macro_f1`（同一批留出标签上的
                   champion 指标 → 公平比较），否则退回采信 incumbent 指标。
2. recall_drop     最低类召回相对 champion 的下降 ≤ max_recall_drop（类召回不塌陷），
                   并检查绝对下限 min_class_recall_floor。
3. calibration     候选 ECE 相对 champion 的恶化 ≤ calibration_slack（不可用则记录
                   unavailable 且不阻塞，避免因观测缺失误杀）。
4. cost            单次更新标量成本 ≤ max_cost_per_update。
5. reproducibility 候选可加载/可复现（可传入 model 做 state 往返实测；否则由
                   registry.add 的 npz 往返校验负责并在此记录依据）。
6. high_risk       高风险 playbook 需更大增益 + human_gate：校赛封闭评测默认
                   auto_approve_high_risk=True（以 escalation 记录代替人工确认，
                   见 docs/06 §5）；关闭时无人工批准一律拒绝。
7. budget          剩余标签/算力非负（硬约束，loop 已先过滤，这里再兜一道）。

保守原则：候选指标缺失/NaN 时判 **拒绝**，绝不抛异常（封闭评测里异常=事故）。
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from ..core.spec import Cost, GateResult
from .defaults import resolve_config
from .utils import state_roundtrip_identical


def _finite(v) -> bool:
    try:
        return bool(np.isfinite(float(v)))
    except (TypeError, ValueError):
        return False


def _f(v, default: float = float("nan")) -> float:
    try:
        return float(v)
    except (TypeError, ValueError):
        return default


class Gate:
    def __init__(self, cfg: dict):
        self.cfg = resolve_config(cfg)
        g = self.cfg["gates"]
        self.g = g
        self.min_gain = float(g.get("min_gain_points", 0.01))
        self.min_probe_gain = float(g.get("min_probe_gain", 0.005))
        self.max_recall_drop = float(g.get("max_recall_drop", 0.12))
        self.max_cost = float(g.get("max_cost_per_update", 40.0))
        self.calibration_slack = float(g.get("calibration_slack", 0.04))
        self.recall_floor = float(max(g.get("min_class_recall_floor", 0.0),
                                      g.get("min_class_recall_drop_abs", 0.0)))
        self.high_risk_mult = float(g.get("high_risk_gain_multiplier", 1.5))
        self.shadow_required = bool(g.get("shadow_required", False))
        self.shadow_windows = int(g.get("shadow_windows", 2))
        self.repro_tol = float(g.get("repro_tol", 1e-6))
        self.min_labels = int(g.get("min_labels_for_verdict", 12))
        safety = self.cfg["safety"]
        self.high_risk_playbooks = set(safety.get("high_risk_playbooks", []) or [])
        self.auto_approve_high_risk = bool(safety.get("auto_approve_high_risk", False))
        self.human_gate = bool(safety.get("human_gate", True))

    # ------------------------------------------------------------------
    def evaluate(self, candidate: dict, incumbent: dict, ctx: Optional[dict] = None) -> GateResult:
        candidate = dict(candidate or {})
        incumbent = dict(incumbent or {})
        ctx = dict(ctx or {})
        criteria: dict = {}
        critical_fail: list[str] = []
        soft_fail: list[str] = []
        risks: list[str] = []
        playbook = str(ctx.get("playbook") or candidate.get("playbook") or "")

        # ---------- 0. 基础可用性 ----------
        cand_f1 = _f(candidate.get("probe_macro_f1"))
        inc_f1 = _f(incumbent.get("probe_macro_f1"))
        base_f1 = cand_f1
        fair = _f(candidate.get("baseline_probe_macro_f1"))
        if _finite(fair):
            base_f1 = fair
            base_src = "candidate.baseline_probe_macro_f1(同留出集)"
        elif _finite(inc_f1):
            base_f1 = inc_f1
            base_src = "incumbent.probe_macro_f1"
        else:
            base_src = "unavailable"
        if not _finite(cand_f1) or not _finite(base_f1):
            criteria["probe_gain"] = {"value": None, "threshold": self.min_gain,
                                      "pass": False,
                                      "note": f"候选/基线指标缺失(base={base_src})，保守拒绝"}
            critical_fail.append("probe_gain")
            gain = float("nan")
        else:
            gain = float(cand_f1 - base_f1)
            threshold = self.min_gain
            if playbook in self.high_risk_playbooks:
                threshold = max(threshold, self.min_gain * self.high_risk_mult)
            ok = gain >= threshold - 1e-12
            criteria["probe_gain"] = {"value": round(gain, 6), "threshold": threshold,
                                      "pass": bool(ok), "note": f"基线来源: {base_src}"}
            if not ok:
                critical_fail.append("probe_gain")
        # 次要：min_probe_gain（更严格的信息项，不单独阻塞，只记录）
        criteria["probe_gain_minor"] = {
            "value": round(gain, 6) if _finite(gain) else None,
            "threshold": self.min_probe_gain, "pass": bool(_finite(gain) and gain >= self.min_probe_gain),
            "note": "min_probe_gain 辅助判据"}

        # ---------- 1. 类召回 ----------
        cand_rec = _f(candidate.get("class_recall_min"))
        base_rec = _f(candidate.get("baseline_class_recall_min"))
        if not _finite(base_rec):
            base_rec = _f(incumbent.get("class_recall_min"))
        if not _finite(cand_rec):
            criteria["recall_drop"] = {"value": None, "threshold": self.max_recall_drop,
                                       "pass": False, "note": "候选类召回缺失，保守拒绝"}
            critical_fail.append("recall_drop")
        else:
            if not _finite(base_rec):
                drop = 0.0
                note = "基线类召回缺失，按 0 降幅处理"
            else:
                drop = float(base_rec - cand_rec)
                note = ""
            ok = drop <= self.max_recall_drop + 1e-12
            criteria["recall_drop"] = {"value": round(float(drop), 6),
                                       "threshold": self.max_recall_drop,
                                       "pass": bool(ok), "note": note}
            if not ok:
                critical_fail.append("recall_drop")
            floor_ok = cand_rec >= self.recall_floor - 1e-12
            criteria["recall_floor"] = {"value": round(float(cand_rec), 6),
                                        "threshold": self.recall_floor,
                                        "pass": bool(floor_ok), "note": ""}
            if not floor_ok:
                critical_fail.append("recall_floor")

        # ---------- 2. 校准 ----------
        cand_ece = _f(candidate.get("ece"))
        base_ece = _f(candidate.get("baseline_ece"))
        if not _finite(base_ece):
            base_ece = _f(incumbent.get("ece"))
        if _finite(cand_ece) and _finite(base_ece):
            delta = float(cand_ece - base_ece)
            ok = delta <= self.calibration_slack + 1e-12
            criteria["calibration"] = {"value": round(delta, 6),
                                       "threshold": self.calibration_slack,
                                       "pass": bool(ok), "note": "ECE 变化量"}
            if not ok:
                soft_fail.append("calibration")
        else:
            criteria["calibration"] = {"value": None, "threshold": self.calibration_slack,
                                       "pass": True, "note": "ECE 不可用，不阻塞"}

        # ---------- 3. 成本 ----------
        cost_val = _f(candidate.get("cost"))
        if not _finite(cost_val):
            est = ctx.get("est_cost_scalar")
            cost_val = _f(est)
        if _finite(cost_val):
            ok = cost_val <= self.max_cost + 1e-12
            criteria["cost"] = {"value": round(float(cost_val), 6),
                                "threshold": self.max_cost, "pass": bool(ok)}
            if not ok:
                soft_fail.append("cost")
        else:
            criteria["cost"] = {"value": None, "threshold": self.max_cost,
                                "pass": True, "note": "成本不可用，不阻塞"}

        # ---------- 4. 可复现 ----------
        repro_ok, repro_note = True, "registry.add 时 npz 往返校验"
        if "reproducible" in candidate:
            repro_ok = bool(candidate.get("reproducible"))
            repro_note = "playbook 自检 state_roundtrip"
        model = candidate.get("model")
        eval_x = ctx.get("eval_x")
        if model is not None and eval_x is not None:
            repro_ok = state_roundtrip_identical(model, eval_x, tol=self.repro_tol)
            repro_note = "state 往返实测"
        criteria["reproducibility"] = {"value": bool(repro_ok), "threshold": True,
                                       "pass": bool(repro_ok), "note": repro_note}
        if not repro_ok:
            critical_fail.append("reproducibility")

        # ---------- 5. 预算硬门（兜底） ----------
        budget = ctx.get("budget") or {}
        labels_left = None
        try:
            if isinstance(budget, dict) and "labels_total" in budget:
                labels_left = int(budget["labels_total"]) - int(budget.get("labels_used", 0))
        except Exception:
            labels_left = None
        if labels_left is not None:
            ok = labels_left >= 0
            criteria["budget"] = {"value": int(labels_left), "threshold": 0,
                                  "pass": bool(ok), "note": "剩余标签数"}
            if not ok:
                critical_fail.append("budget")

        # ---------- 6. 高风险升级 ----------
        is_high = bool(ctx.get("high_risk")) or playbook in self.high_risk_playbooks
        if is_high:
            approval = ctx.get("human_approval")
            if approval is True:
                mode, ok = "human_approved", True
            elif self.auto_approve_high_risk:
                mode, ok = "escalation_auto_approved", True
            elif not self.human_gate:
                mode, ok = "blocked", False
            else:
                mode, ok = "awaiting_human", False
            criteria["human_gate"] = {"value": mode, "threshold": True, "pass": bool(ok),
                                      "note": "高风险动作升级记录（校赛封闭评测默认自动批准并留痕）"}
            if not ok:
                critical_fail.append("human_gate")
                risks.append(f"高风险动作 {playbook} 未获批准")

        # ---------- 7. 影子要求 ----------
        if self.shadow_required and not ctx.get("shadow_ok"):
            criteria["shadow"] = {"value": int(ctx.get("shadow_windows", 0)),
                                  "threshold": self.shadow_windows, "pass": False,
                                  "note": "要求先影子评估 N 窗口"}
            soft_fail.append("shadow")
        else:
            criteria["shadow"] = {"value": int(ctx.get("shadow_windows", self.shadow_windows)),
                                  "threshold": self.shadow_windows, "pass": True,
                                  "note": "shadow_required=false"}

        # ---------- 汇总 ----------
        passed = not critical_fail
        if passed and soft_fail:
            # 软失败：降低决策强度但允许上线（记录理由），保持校赛阶段正向适应
            decision = "promote"
            reason = f"通过（软告警: {','.join(soft_fail)}）"
        elif passed:
            decision = "promote"
            reason = "全部门禁通过"
        else:
            decision = "reject"
            reason = "拒绝: " + ",".join(critical_fail)
        if risks:
            reason += "；" + "；".join(risks)
        gain_txt = f"增益={gain:+.4f}" if _finite(gain) else "增益=NA"
        return GateResult(criteria=criteria, passed=bool(passed), decision=decision,
                          reason=f"{reason}（{gain_txt}）")
