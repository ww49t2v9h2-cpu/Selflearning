"""Harness 驱动层：DeterministicDriver（默认）+ LLMDriver（可插拔 LLM）。

统一契约（docs/03 §5.2）
-----------------------
```python
class HarnessDriver(Protocol):
    def decide(self, snapshot: dict, tools: ToolRegistry) -> Trajectory
```
`snapshot` 由 loop 构造：`{"t","phase","monitor_report","diagnosis","candidates",
"budget","registry_summary","skills"}`；`tools` 为 `ToolRegistry`。
两种 driver 输出统一的 `Trajectory`（steps=实际 ToolCall、final_decision=Decision
草案、tokens/wall_s/driver/outcome）。

DeterministicDriver
-------------------
默认离线驱动，规则来自 `skills/*.md` 的可执行子集（阈值与 DEFAULT_CONFIG 同源）：
no-drift→NOOP；benign→NOOP；repair→ROLLBACK；其余按"cost/prior/severity"排序
选一个候选跑实验 → 评估 → 门禁通过则 deploy；预算不足/工具被拒→换下一个或 NOOP。
全程通过工具执行，轨迹完整可审计；不消耗 token。

LLMDriver
---------
把 skills 注入 system prompt、工具 JSON Schema 作为可用动作，模型每步输出
`{"thought","tool","args"}` 或 `{"thought","final":{...}}`。严格的 JSON 解析 +
工具名/参数校验；连续解析失败、token 超预算、provider 不可用 → **自动降级**
`DeterministicDriver`（`Trajectory.driver="llm-fallback"` 且 outcome 记录原因）。
token 计量来自 `ProviderResult`（真实 usage 优先，mock 按字符估算），并计入
`Trajectory.tokens` / `Decision.cost.tokens`。
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional, Protocol, runtime_checkable

from ..core.config import load_config
from ..core.spec import Cost, Decision, Diagnosis, RunRecord, ToolCall, Trajectory
from ..providers.base import (
    RESULT_PREFIX,
    SNAPSHOT_PREFIX,
    ProviderError,
    ProviderResult,
    ProviderUnavailable,
)
from ..providers.mock_llm import MockLLMProvider
from ..providers.openai_compat import OpenAICompatProvider
from . import skills as skills_mod
from .tools import (
    ToolContext,
    ToolRegistry,
    _ctx_budget,
    _ctx_current,
    _ctx_is_approved,
    _ctx_state,
    _kill_switch_active,
    build_tool_registry,
    to_plain,
)

ROOT = Path(__file__).resolve().parents[2]

_FINAL_KEYS = {"chosen_playbook", "action", "reasoning", "deploy", "version",
               "confidence", "outcome", "note"}


# --------------------------------------------------------------------------
# 通用辅助
# --------------------------------------------------------------------------
def _as_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _shrink(obj: Any, depth: int = 0, max_list: int = 12,
            max_str: int = 500) -> Any:
    """把 snapshot 压缩成 LLM 可读的紧凑 JSON（避免把大数组塞进 prompt）。"""
    if depth > 6:
        return "…"
    if obj is None or isinstance(obj, (bool, int, float)):
        return obj
    if isinstance(obj, str):
        return obj if len(obj) <= max_str else obj[:max_str] + "…"
    if hasattr(obj, "tolist"):
        obj = obj.tolist()
    if isinstance(obj, dict):
        return {str(k): _shrink(v, depth + 1, max_list, max_str)
                for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        items = list(obj)
        out = [_shrink(v, depth + 1, max_list, max_str) for v in items[:max_list]]
        if len(items) > max_list:
            out.append(f"…(+{len(items) - max_list})")
        return out
    if hasattr(obj, "to_dict"):
        try:
            return _shrink(obj.to_dict(), depth + 1, max_list, max_str)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)[:max_str]


def _maybe_diagnosis(obj: Any) -> Optional[Diagnosis]:
    if obj is None or isinstance(obj, Diagnosis):
        return obj
    if isinstance(obj, dict):
        known = set(Diagnosis.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in obj.items() if k in known}
        kwargs.setdefault("t", 0)
        kwargs.setdefault("phase", 0)
        kwargs.setdefault("kind", "benign")
        kwargs.setdefault("severity", 0.0)
        kwargs.setdefault("confidence", 0.0)
        try:
            return Diagnosis(**kwargs)
        except TypeError:
            return None
    return None


def _snapshot_diagnosis(snapshot: dict) -> Optional[Diagnosis]:
    return _maybe_diagnosis(snapshot.get("diagnosis"))


def _snapshot_budget(snapshot: dict) -> dict:
    budget = snapshot.get("budget")
    if budget is None:
        return {}
    if hasattr(budget, "to_dict"):
        return dict(budget.to_dict())
    if isinstance(budget, dict):
        return dict(budget)
    return {}


def _cost_of_dict(d: Any) -> Cost:
    if isinstance(d, Cost):
        return Cost(**{k: getattr(d, k) for k in
                       ("compute_units", "labels", "tokens", "wall_s")})
    if isinstance(d, dict):
        return Cost(compute_units=_as_float(d.get("compute_units", 0.0)),
                    labels=_as_int(d.get("labels", 0)),
                    tokens=_as_int(d.get("tokens", 0)),
                    wall_s=_as_float(d.get("wall_s", 0.0)))
    return Cost()


def _cost_sum(calls: list[ToolCall]) -> Cost:
    total = Cost()
    for call in calls:
        total = total.add(call.cost)
    return total


def _normalize_candidates(snapshot: dict, cfg: dict) -> list[dict]:
    """把 loop 传入的候选统一成 {playbook, prior, why, est_cost, est_scalar}。"""
    raw = snapshot.get("candidates") or []
    out: list[dict] = []
    seen: set[str] = set()
    for cand in raw:
        if isinstance(cand, str):
            cand = {"playbook": cand}
        if not isinstance(cand, dict):
            continue
        name = str(cand.get("playbook") or cand.get("name") or "").strip()
        if not name or name in seen:
            continue
        seen.add(name)
        est_raw = cand.get("est_cost")
        if hasattr(est_raw, "to_dict"):
            est_raw = est_raw.to_dict()
        if not isinstance(est_raw, dict):
            est_raw = {}
        est_cost = _cost_of_dict(est_raw)
        est_scalar = cand.get("est_cost_scalar")
        if est_scalar is None:
            est_scalar = est_cost.scalar
        out.append({
            "playbook": name,
            "prior": _as_float(cand.get("prior", 0.0)),
            "why": str(cand.get("why", "") or ""),
            "est_cost": {"compute_units": est_cost.compute_units,
                         "labels": est_cost.labels,
                         "tokens": est_cost.tokens,
                         "wall_s": est_cost.wall_s},
            "est_scalar": _as_float(est_scalar),
            "risk": str(cand.get("risk", "") or ""),
        })
    return out


def _candidate_affordable(cand: dict, budget_info: dict,
                          allowed: set[str]) -> tuple[bool, str]:
    name = cand["playbook"]
    if allowed and name not in allowed:
        return False, "不在动作白名单"
    if not budget_info:
        return True, ""
    est = cand.get("est_cost") or {}
    if budget_info.get("updates_left_phase", 1) <= 0 and name not in ("NOOP", "ROLLBACK"):
        return False, "当前 phase 更新次数已用满"
    if _as_int(est.get("labels", 0)) > _as_int(budget_info.get("labels_left", 10 ** 9)):
        return False, f"标签预算不足（需 {est.get('labels')}，余 {budget_info.get('labels_left')}）"
    if _as_float(est.get("compute_units", 0.0)) > _as_float(
            budget_info.get("compute_left", 1e18)):
        return False, f"算力预算不足（需 {est.get('compute_units')}，余 {budget_info.get('compute_left')}）"
    if _as_int(est.get("tokens", 0)) > _as_int(budget_info.get("tokens_left", 10 ** 9)):
        return False, "token 预算不足"
    return True, ""


def _order_candidates(candidates: list[dict], cfg: dict,
                      diagnosis: Optional[Diagnosis],
                      budget_info: dict) -> list[dict]:
    """按 skills 规则排序候选：benign→NOOP、repair→ROLLBACK、高严重度先验优先、
    其余 cheapest-first；不可负担的候选排到后面并标注原因。"""
    allowed = set((cfg.get("safety", {}) or {}).get("allowed_playbooks", []) or [])
    kind = str(getattr(diagnosis, "kind", "") or "")
    severity = _as_float(getattr(diagnosis, "severity", 0.0))
    severity_high = _as_float((cfg.get("diagnosis", {}) or {}).get("severity_high", 0.55))
    ordered: list[dict] = []
    for cand in candidates:
        affordable, reason = _candidate_affordable(cand, budget_info, allowed)
        item = dict(cand)
        item["affordable"] = affordable
        item["blocked_reason"] = reason
        ordered.append(item)

    def key(item: dict):
        name = item["playbook"]
        boost = 0.0
        if kind == "benign" and name == "NOOP":
            boost -= 1e9
        elif kind in ("repair",) and name == "ROLLBACK":
            boost -= 1e9
        elif kind == "repair" and name == "NOOP":
            boost -= 1e8
        elif name == "NOOP":
            boost += 1e7          # 正常漂移下优先做动作，而不是直接放弃
        elif name == "SHADOW_ONLY":
            boost += 1e6          # 影子评估优先于更重的动作
        if severity >= severity_high:
            return (boost, -item["prior"], item["est_scalar"], name)
        return (boost, item["est_scalar"], -item["prior"], name)

    ordered.sort(key=key)
    # 可负担的排前面；不可负担的保留在后面（用于 rationale/alternatives）
    return [c for c in ordered if c["affordable"]] + \
           [c for c in ordered if not c["affordable"]]


def _budget_after(tools: ToolRegistry, snapshot: dict) -> dict:
    budget = _ctx_budget(tools.ctx)
    if budget is not None:
        phase = _as_int(snapshot.get("phase", 0))
        cfg = getattr(tools.ctx, "cfg", {}) or {}
        per_phase = _as_int((cfg.get("budget", {}) or {}).get("per_phase_updates", 0))
        if per_phase:
            updates_left_phase = budget.updates_left(phase, per_phase)
        else:
            updates_left_phase = max(0, budget.updates_total - budget.updates_used)
        return {
            "labels_total": budget.labels_total, "labels_used": budget.labels_used,
            "labels_left": budget.labels_left(),
            "compute_total": budget.compute_total, "compute_used": budget.compute_used,
            "compute_left": budget.compute_left(),
            "tokens_total": budget.tokens_total, "tokens_used": budget.tokens_used,
            "tokens_left": budget.tokens_left(),
            "updates_total": budget.updates_total, "updates_used": budget.updates_used,
            "updates_left_phase": updates_left_phase,
        }
    return _snapshot_budget(snapshot)


def _collect_evidence(calls: list[ToolCall]) -> list[str]:
    refs: list[str] = []
    for call in calls:
        if not call.ok or not isinstance(call.result, dict):
            continue
        for key in ("run_id", "exp_id", "version"):
            value = call.result.get(key)
            if value not in (None, "", -1) and str(value) not in refs:
                refs.append(str(value))
        for art in call.result.get("artifacts") or []:
            if art and str(art) not in refs:
                refs.append(str(art))
    return refs


def _make_decision(*, t: int, phase: int, diagnosis: Optional[Diagnosis],
                   chosen_playbook: str, alternatives: list, budget_before: dict,
                   budget_after: dict, cost: Cost, outcome: dict, rationale: str,
                   evidence_refs: list, driver: str, safety: dict,
                   tokens: int = 0) -> Decision:
    cost = Cost(**{k: getattr(cost, k) for k in
                   ("compute_units", "labels", "tokens", "wall_s")})
    cost.tokens = int(cost.tokens) + int(tokens)
    return Decision(
        t=int(t), phase=int(phase), diagnosis=diagnosis,
        chosen_playbook=str(chosen_playbook),
        alternatives=to_plain(alternatives), budget_before=to_plain(budget_before),
        budget_after=to_plain(budget_after), cost=cost,
        outcome=to_plain(outcome), rationale=str(rationale),
        evidence_refs=to_plain(evidence_refs), driver=driver,
        safety=to_plain(safety),
    )


def _is_high_risk(cfg: dict, playbook: str) -> bool:
    high = set((cfg.get("safety", {}) or {}).get("high_risk_playbooks", []) or [])
    return playbook in high


# --------------------------------------------------------------------------
# HarnessDriver 协议
# --------------------------------------------------------------------------
@runtime_checkable
class HarnessDriver(Protocol):
    def decide(self, snapshot: dict, tools: ToolRegistry) -> Trajectory: ...


# --------------------------------------------------------------------------
# 确定性驱动
# --------------------------------------------------------------------------
@dataclass
class DeterministicRules:
    """skills/00 的机器可执行子集（可在 tests/配置中覆写）。"""

    min_severity_act: float = 0.15
    low_confidence: float = 0.40
    benign_kinds: tuple = ("benign",)
    repair_kinds: tuple = ("repair",)
    noop_playbook: str = "NOOP"
    rollback_playbook: str = "ROLLBACK"
    shadow_playbook: str = "SHADOW_ONLY"
    max_experiments_per_turn: int = 1


class DeterministicDriver:
    """默认离线驱动：规则来自 skills，全部动作通过工具执行并记账。"""

    name = "deterministic"
    driver_tag = "deterministic_policy"

    def __init__(self, cfg: Optional[dict] = None,
                 rules: Optional[DeterministicRules] = None,
                 skills_dir: Optional[str | Path] = None):
        self.cfg = cfg or load_config()
        self.rules = rules or DeterministicRules(
            low_confidence=float((self.cfg.get("diagnosis", {}) or {}).get(
                "low_confidence", 0.40)),
        )
        self.skills_dir = skills_dir
        self.provider_info: dict = {"driver": self.name, "provider": None,
                                    "network": False}
        # 供 loop 打印：本驱动使用的技能文档
        self.skill_ids = list(skills_mod.load_skills(skills_dir))

    # ------------------------------------------------------------------
    def decide(self, snapshot: dict, tools: ToolRegistry) -> Trajectory:
        t0 = time.perf_counter()
        steps: list[ToolCall] = []
        snapshot = snapshot or {}
        t = _as_int(snapshot.get("t", 0))
        phase = _as_int(snapshot.get("phase", 0))
        budget_before = _snapshot_budget(snapshot)
        diagnosis = _snapshot_diagnosis(snapshot)
        candidates = _normalize_candidates(snapshot, self.cfg)
        try:
            return self._decide_inner(snapshot, tools, steps, t, phase,
                                      budget_before, diagnosis, candidates, t0)
        except Exception as exc:  # noqa: BLE001 - driver 永不抛异常给 loop
            outcome = {"action": "noop", "error": f"{type(exc).__name__}: {exc}"}
            decision = _make_decision(
                t=t, phase=phase, diagnosis=diagnosis, chosen_playbook="NOOP",
                alternatives=[], budget_before=budget_before,
                budget_after=_budget_after(tools, snapshot), cost=_cost_sum(steps),
                outcome=outcome,
                rationale=f"驱动内部异常（{type(exc).__name__}），保守 NOOP 并保留轨迹",
                evidence_refs=_collect_evidence(steps),
                driver=self.driver_tag,
                safety={"kill_switch": _kill_switch_active(tools.ctx),
                        "exception": type(exc).__name__})
            return Trajectory(t=t, phase=phase, steps=steps,
                              final_decision=decision, tokens=0,
                              wall_s=time.perf_counter() - t0,
                              driver=self.driver_tag, outcome=outcome)

    # ------------------------------------------------------------------
    def _decide_inner(self, snapshot: dict, tools: ToolRegistry,
                      steps: list[ToolCall], t: int, phase: int,
                      budget_before: dict, diagnosis: Optional[Diagnosis],
                      candidates: list[dict], t0: float) -> Trajectory:
        # 1) 预算 + 监测（只读，完整留痕）
        budget_call = tools.call("budget_status", {"phase": phase})
        steps.append(budget_call)
        budget_info = dict(budget_call.result or {}) if budget_call.ok else dict(budget_before)
        inspect_call = tools.call("inspect_drift", {})
        steps.append(inspect_call)
        monitor_severity = 0.0
        if inspect_call.ok and isinstance(inspect_call.result, dict):
            monitor_severity = _as_float(inspect_call.result.get("max_severity", 0.0))
        severity = max(monitor_severity,
                       _as_float(getattr(diagnosis, "severity", 0.0)))
        kind = str(getattr(diagnosis, "kind", "") or "")
        confidence = _as_float(getattr(diagnosis, "confidence", 1.0))
        alternatives: list[dict] = []
        outcome: dict = {}
        safety: dict = {
            "kill_switch": _kill_switch_active(tools.ctx),
            "human_gate": "escalation_recorded",
            "checks": [],
        }

        def finish(chosen: str, rationale: str, **extra) -> Trajectory:
            decision = _make_decision(
                t=t, phase=phase, diagnosis=diagnosis, chosen_playbook=chosen,
                alternatives=alternatives, budget_before=budget_before,
                budget_after=_budget_after(tools, snapshot), cost=_cost_sum(steps),
                outcome={**outcome, **extra}, rationale=rationale,
                evidence_refs=_collect_evidence(steps), driver=self.driver_tag,
                safety=safety)
            return Trajectory(t=t, phase=phase, steps=steps,
                              final_decision=decision, tokens=0,
                              wall_s=time.perf_counter() - t0,
                              driver=self.driver_tag,
                              outcome={**outcome, **extra})

        # 2) 安全/预算前置判断
        if safety["kill_switch"]:
            outcome.update({"action": "noop", "reason": "kill_switch"})
            return finish("NOOP", "kill-switch 生效：停止自主更新，等待人工/下一阶段")
        updates_left_phase = _as_int(budget_info.get("updates_left_phase", 1), 1)
        if updates_left_phase <= 0:
            outcome.update({"action": "noop", "reason": "updates_exhausted"})
            return finish("NOOP", f"phase={phase} 的更新次数已用满（"
                                  f"{budget_info.get('updates_used')}/"
                                  f"{budget_info.get('updates_total')}），本轮收手")
        if not candidates:
            outcome.update({"action": "noop", "reason": "no_candidates"})
            return finish("NOOP", "候选为空（无漂移或规划器未给出候选）：按安全红线 NOOP")
        if confidence < self.rules.low_confidence and kind not in self.rules.repair_kinds:
            outcome.update({"action": "noop", "reason": "low_confidence"})
            return finish("NOOP", f"诊断置信度 {confidence:.2f} < "
                                  f"{self.rules.low_confidence:.2f}，归因不明不动作")

        # 3) 候选排序（skills 规则）
        ordered = _order_candidates(candidates, self.cfg, diagnosis, budget_info)
        affordable = [c for c in ordered if c.get("affordable")]
        for cand in ordered:
            if not cand.get("affordable"):
                alternatives.append({"playbook": cand["playbook"],
                                     "why_not": cand.get("blocked_reason", "不可负担")})
        if not affordable:
            outcome.update({"action": "noop", "reason": "budget_exhausted"})
            return finish("NOOP", f"所有候选均不可负担（预算/白名单/phase 次数）："
                                  f"{[c['playbook'] for c in candidates]}")

        # 4) 逐个候选尝试（最多 max_experiments_per_turn 个）
        executed = 0
        for cand in affordable:
            playbook = cand["playbook"]
            if executed >= self.rules.max_experiments_per_turn:
                alternatives.append({"playbook": playbook, "why_not": "本轮实验次数已用"})
                continue
            if playbook == self.rules.noop_playbook:
                outcome.update({"action": "noop", "reason": "candidate_NOOP"})
                return finish("NOOP", "候选中的最优动作是 NOOP：当前无必要更新，"
                                      "记录证据并保留预算")
            if playbook == self.rules.rollback_playbook:
                rb = tools.call("rollback", {"reason": f"诊断 {kind} → 回滚 SOP"})
                steps.append(rb)
                if rb.ok:
                    executed += 1
                    outcome.update({
                        "action": "rollback",
                        "rollback": rb.result,
                        "from_version": rb.result.get("from_version"),
                        "to_version": rb.result.get("to_version"),
                    })
                    safety["checks"].append(rb.result.get("safety", {}))
                    return finish("ROLLBACK",
                                  f"诊断={kind}（repair）：按回滚 SOP 从 "
                                  f"v{rb.result.get('from_version')} 回滚到 "
                                  f"v{rb.result.get('to_version')}，恢复稳定版本")
                alternatives.append({"playbook": playbook,
                                     "why_not": rb.error or "rollback 失败"})
                continue
            if playbook == self.rules.shadow_playbook:
                shadow = tools.call("shadow_eval", {})
                steps.append(shadow)
                if shadow.ok:
                    result = shadow.result or {}
                    outcome.update({"action": "shadow_only", "shadow": result})
                    return finish("SHADOW_ONLY",
                                  "候选为 SHADOW_ONLY：仅做影子评估（"
                                  f"一致率={result.get('agreement')}），不改线上模型")
                alternatives.append({"playbook": playbook,
                                     "why_not": shadow.error or "shadow_eval 失败"})
                continue
            # 4a) 常规 playbook：跑实验
            hypothesis = (f"诊断 {kind}（severity={severity:.2f}）下，"
                          f"{playbook} 应带来 probe 增益且成本可控")
            exp_call = tools.call("run_experiment",
                                  {"playbook": playbook, "hypothesis": hypothesis})
            steps.append(exp_call)
            if not exp_call.ok:
                alternatives.append({"playbook": playbook,
                                     "why_not": f"run_experiment 被拒: {exp_call.error}"})
                if (exp_call.result or {}).get("error_kind") in (
                        "kill_switch", "safety_veto", "playbook_not_allowed"):
                    break
                continue
            executed += 1
            exp = dict(exp_call.result or {})
            exp_id = exp.get("exp_id")
            outcome.update({"action": "experiment", "playbook": playbook,
                            "exp_id": exp_id, "run_id": exp.get("run_id"),
                            "metrics": exp.get("metrics", {}),
                            "cost_actual": exp.get("cost", {})})
            if exp.get("escalation"):
                safety["escalations"] = to_plain(exp.get("escalation"))
            # 4b) 评估门禁
            eval_call = tools.call("evaluate_candidate", {"exp_id": exp_id})
            steps.append(eval_call)
            if not eval_call.ok:
                shadow = tools.call("shadow_eval", {"exp_id": exp_id})
                steps.append(shadow)
                outcome.update({"action": "shadow_only", "reason": "evaluate_failed",
                                "shadow": shadow.result if shadow.ok else None})
                return finish(playbook,
                              f"{playbook} 实验完成但评估失败（{eval_call.error}），"
                              "转影子评估、不上线")
            ev = dict(eval_call.result or {})
            gate = ev.get("gate") or {}
            outcome.update({"gate": gate, "probe_gain": ev.get("probe_gain"),
                            "decision": ev.get("decision")})
            safety["checks"].append(ev.get("gate", {}))
            if not ev.get("passed"):
                shadow = tools.call("shadow_eval", {"exp_id": exp_id})
                steps.append(shadow)
                outcome.update({"action": "shadow_only", "reason": "gate_rejected",
                                "shadow": shadow.result if shadow.ok else None})
                return finish(playbook,
                              f"{playbook} 实验未过门禁：{gate.get('reason', '')}；"
                              "候选归档为 shadow 证据，本轮不上线")
            # 4c) 上线
            version = ev.get("version")
            if version is None:
                outcome.update({"action": "shadow_only", "reason": "no_version"})
                return finish(playbook,
                              f"{playbook} 门禁通过但候选未入库（无 version），"
                              "记录证据、维持 champion")
            deploy_call = tools.call("deploy", {
                "version": _as_int(version),
                "reason": f"门禁通过（probe 增益={ev.get('probe_gain')}），"
                          f"诊断={kind}，playbook={playbook}"})
            steps.append(deploy_call)
            if deploy_call.ok:
                dep = dict(deploy_call.result or {})
                outcome.update({"action": "deploy", "deployed": True,
                                "version": dep.get("version"),
                                "previous_version": dep.get("previous_version"),
                                "escalation": dep.get("escalation")})
                safety["checks"].append(dep.get("safety", {}))
                return finish(playbook,
                              f"{playbook} 实验门禁通过（probe 增益 "
                              f"{ev.get('probe_gain')}），v{version} 已上线 "
                              f"（previous=v{dep.get('previous_version')}）；"
                              "后续按看门狗观察，掉点则回滚")
            alternatives.append({"playbook": playbook,
                                 "why_not": f"deploy 被拒: {deploy_call.error}"})
            outcome.update({"action": "experiment", "deployed": False})
            return finish(playbook,
                          f"{playbook} 实验门禁通过但上线被拒"
                          f"（{deploy_call.error}）；维持 champion，记录 escalation")
        # 5) 所有候选都失败
        outcome.update({"action": "noop", "reason": "all_candidates_failed"})
        return finish("NOOP",
                      "所有候选均执行失败/被拒：本轮 NOOP，失败证据已记录，"
                      "下轮重新规划（不重复同一失败配置）")


# --------------------------------------------------------------------------
# LLM 驱动
# --------------------------------------------------------------------------
class LLMDriver:
    """LLM 驱动：OpenAI 兼容 JSON 工具调用协议 + 严格校验 + 失败降级。"""

    name = "llm"
    driver_tag = "llm"

    def __init__(self, provider: Any, cfg: Optional[dict] = None,
                 max_steps: Optional[int] = None,
                 temperature: Optional[float] = None,
                 max_tokens: Optional[int] = None,
                 skills_dir: Optional[str | Path] = None,
                 fallback: Optional[HarnessDriver] = None,
                 max_parse_failures: int = 2,
                 max_tokens_per_decision: Optional[int] = None,
                 charge_budget_tokens: bool = False,
                 system_prompt_extra: str = ""):
        self.provider = provider
        self.cfg = cfg or load_config()
        hcfg = self.cfg.get("harness", {}) or {}
        scfg = self.cfg.get("safety", {}) or {}
        self.max_steps = int(max_steps if max_steps is not None
                             else hcfg.get("max_steps", 8))
        self.temperature = float(temperature if temperature is not None
                                 else hcfg.get("temperature", 0.2))
        self.max_tokens = int(max_tokens or 1024)
        self.max_tokens_per_decision = int(
            max_tokens_per_decision if max_tokens_per_decision is not None
            else scfg.get("max_tokens_per_decision", 20000))
        self.skills_dir = skills_dir
        self.fallback = fallback or DeterministicDriver(self.cfg,
                                                        skills_dir=skills_dir)
        self.max_parse_failures = max(1, int(max_parse_failures))
        # 默认不直接扣全局 BudgetState 的 token：闭环（loop）在每轮统一记账，
        # 否则会和 loop 的 budget.charge(tokens) 重复计数；独立使用 Harness 时
        # 可传 charge_budget_tokens=True 自行记账。
        self.charge_budget_tokens = bool(charge_budget_tokens)
        self.system_prompt_extra = system_prompt_extra
        provider_name = getattr(provider, "name", type(provider).__name__)
        self.provider_info = {
            "driver": self.name,
            "provider": provider_name,
            "model": getattr(provider, "model", ""),
            "network": bool(getattr(provider, "enabled", False)),
            "max_steps": self.max_steps,
        }
        self.skill_ids = list(skills_mod.load_skills(skills_dir))

    # ------------------------------------------------------------------
    def decide(self, snapshot: dict, tools: ToolRegistry) -> Trajectory:
        t0 = time.perf_counter()
        snapshot = snapshot or {}
        t = _as_int(snapshot.get("t", 0))
        phase = _as_int(snapshot.get("phase", 0))
        budget_before = _snapshot_budget(snapshot)
        diagnosis = _snapshot_diagnosis(snapshot)
        candidates = _normalize_candidates(snapshot, self.cfg)
        candidate_names = {c["playbook"] for c in candidates}
        steps: list[ToolCall] = []
        tokens = 0
        parse_failures = 0
        messages = self._initial_messages(snapshot, tools)

        for step_idx in range(self.max_steps):
            budget = _ctx_budget(tools.ctx)
            if budget is not None and budget.tokens_left() < 256:
                return self._fallback_trajectory(
                    snapshot, tools, "budget_tokens_low", tokens, steps, t0,
                    parse_failures, t, phase, budget_before, diagnosis)
            # safety.max_tokens_per_decision 作为【单次 LLM 调用】上限；
            # 整轮/整个 run 的总量由 budget.tokens_total 硬约束（见 docs/07）。
            try:
                result: ProviderResult = self.provider.complete(
                    messages, temperature=self.temperature,
                    max_tokens=self.max_tokens)
            except ProviderUnavailable as exc:
                return self._fallback_trajectory(
                    snapshot, tools, f"provider_unavailable: {exc}", tokens,
                    steps, t0, parse_failures, t, phase, budget_before, diagnosis)
            except ProviderError as exc:
                return self._fallback_trajectory(
                    snapshot, tools, f"provider_error: {exc}", tokens, steps,
                    t0, parse_failures, t, phase, budget_before, diagnosis)
            tokens += int(result.total_tokens)
            if int(result.total_tokens) > self.max_tokens_per_decision:
                return self._fallback_trajectory(
                    snapshot, tools,
                    f"single_call_token_cap_exceeded({result.total_tokens})",
                    tokens, steps, t0, parse_failures, t, phase, budget_before,
                    diagnosis)
            self._charge_tokens(tools, tokens, phase)
            text = result.text or ""
            messages.append({"role": "assistant", "content": text[:8000]})
            try:
                obj = self._parse_step(text)
            except ValueError as exc:
                parse_failures += 1
                messages.append({"role": "user", "content":
                                 f"【格式错误】{exc}。请只输出一个 JSON 对象，"
                                 "要么 {\"thought\",\"tool\",\"args\"}，"
                                 "要么 {\"thought\",\"final\":{...}}，不要多余文字。"})
                if parse_failures >= self.max_parse_failures:
                    return self._fallback_trajectory(
                        snapshot, tools, f"json_parse_failed: {exc}", tokens, steps,
                        t0, parse_failures, t, phase, budget_before, diagnosis)
                continue
            # ---------- 最终决策 ----------
            if "final" in obj:
                final = obj.get("final")
                if not isinstance(final, dict):
                    parse_failures += 1
                    messages.append({"role": "user", "content":
                                     "【格式错误】final 必须是 JSON 对象。"})
                    if parse_failures >= self.max_parse_failures:
                        return self._fallback_trajectory(
                            snapshot, tools, "final_not_object", tokens, steps, t0,
                            parse_failures, t, phase, budget_before, diagnosis)
                    continue
                chosen = str(final.get("chosen_playbook") or "").strip()
                if chosen not in candidate_names and chosen != "NOOP":
                    parse_failures += 1
                    messages.append({"role": "user", "content":
                                     f"【决策非法】chosen_playbook={chosen!r} 必须来自"
                                     f"本轮 candidates 或 NOOP；可用: "
                                     f"{sorted(candidate_names) or ['NOOP']}。"
                                     "请重新输出 final。"})
                    if parse_failures >= self.max_parse_failures:
                        return self._fallback_trajectory(
                            snapshot, tools, f"invalid_chosen_playbook:{chosen}",
                            tokens, steps, t0, parse_failures, t, phase,
                            budget_before, diagnosis)
                    continue
                decision = self._final_decision(
                    obj.get("thought", ""), final, t, phase, diagnosis,
                    budget_before, _budget_after(tools, snapshot), steps, tokens,
                    candidates, tools)
                outcome = {
                    "action": decision.outcome.get("action", "final"),
                    "deployed": bool(decision.outcome.get("deployed", False)),
                    "chosen_playbook": decision.chosen_playbook,
                    "llm_steps": step_idx + 1,
                }
                return Trajectory(t=t, phase=phase, steps=steps,
                                  final_decision=decision, tokens=tokens,
                                  wall_s=time.perf_counter() - t0,
                                  driver=self.driver_tag, outcome=outcome)
            # ---------- 工具调用 ----------
            tool_name = str(obj.get("tool") or "").strip()
            args = obj.get("args") or {}
            if tool_name not in tools.names():
                parse_failures += 1
                messages.append({"role": "user", "content":
                                 f"【工具不存在】{tool_name!r}；可用: {tools.names()}。"
                                 "请重新输出 JSON。"})
                if parse_failures >= self.max_parse_failures:
                    return self._fallback_trajectory(
                        snapshot, tools, f"unknown_tool:{tool_name}", tokens,
                        steps, t0, parse_failures, t, phase, budget_before,
                        diagnosis)
                continue
            if not isinstance(args, dict):
                parse_failures += 1
                messages.append({"role": "user", "content":
                                 "【参数错误】args 必须是 JSON 对象。"})
                continue
            call = tools.call(tool_name, args)
            steps.append(call)
            messages.append({"role": "user", "content": RESULT_PREFIX + json.dumps(
                {"tool": tool_name, "ok": call.ok,
                 "result": _shrink(call.result, max_list=8),
                 "error": call.error, "error_kind":
                     (call.result or {}).get("error_kind")},
                ensure_ascii=False)})
            if not call.ok:
                messages.append({"role": "user", "content":
                                 f"【工具被拒/失败】{tool_name}: {call.error}。"
                                 "读 error_kind 决定换策略或收手，不要重复同一失败调用。"})
        # 步数上限
        return self._fallback_trajectory(
            snapshot, tools, f"max_steps_reached({self.max_steps})", tokens,
            steps, t0, parse_failures, t, phase, budget_before, diagnosis)

    # ------------------------------------------------------------------
    def _initial_messages(self, snapshot: dict, tools: ToolRegistry) -> list[dict]:
        kind = str(getattr(_snapshot_diagnosis(snapshot), "kind", "") or "")
        selected_ids = skills_mod.select_skill_ids(kind)
        selected = skills_mod.skills_by_ids(selected_ids, self.skills_dir)
        tool_briefs = None
        brief_fn = getattr(tools, "brief_specs", None)
        if callable(brief_fn):
            tool_briefs = brief_fn()
        system = skills_mod.build_system_prompt(
            tool_specs=None, tool_briefs=tool_briefs, skills=selected,
            dirs=self.skills_dir, extra_rules=self.system_prompt_extra)
        compact = _shrink(snapshot, max_list=8, max_str=400)
        user = SNAPSHOT_PREFIX + json.dumps(
            {"type": "snapshot", "snapshot": compact,
             "candidates": [{"playbook": c["playbook"], "prior": c["prior"],
                             "est_cost": c["est_cost"], "why": c["why"]}
                            for c in _normalize_candidates(snapshot, self.cfg)]},
            ensure_ascii=False)
        return [{"role": "system", "content": system},
                {"role": "user", "content": user}]

    @staticmethod
    def _parse_step(text: str) -> dict:
        """严格解析：容忍 ```json 围栏，提取第一个平衡 JSON 对象。"""
        if not isinstance(text, str) or not text.strip():
            raise ValueError("空响应")
        cleaned = re.sub(r"^\s*```(?:json)?\s*", "", text.strip(),
                         flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```\s*$", "", cleaned).strip()
        start = cleaned.find("{")
        if start < 0:
            raise ValueError("响应中没有 JSON 对象")
        depth, in_str, escape, end = 0, False, False, -1
        for i in range(start, len(cleaned)):
            ch = cleaned[i]
            if in_str:
                if escape:
                    escape = False
                elif ch == "\\":
                    escape = True
                elif ch == '"':
                    in_str = False
                continue
            if ch == '"':
                in_str = True
            elif ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        if end < 0:
            raise ValueError("JSON 对象不完整")
        try:
            obj = json.loads(cleaned[start:end])
        except ValueError as exc:
            raise ValueError(f"JSON 解析失败: {exc}") from exc
        if not isinstance(obj, dict):
            raise ValueError("顶层必须是 JSON 对象")
        if "final" not in obj and "tool" not in obj:
            raise ValueError("缺少 'tool' 或 'final' 字段")
        return obj

    def _final_decision(self, thought: Any, final: dict, t: int, phase: int,
                        diagnosis: Optional[Diagnosis], budget_before: dict,
                        budget_after: dict, steps: list[ToolCall], tokens: int,
                        candidates: list[dict],
                        tools: ToolRegistry) -> Decision:
        chosen = str(final.get("chosen_playbook"))
        deployed = any(call.name == "deploy" and call.ok for call in steps)
        version = None
        exp_id = None
        for call in reversed(steps):
            if call.ok and isinstance(call.result, dict):
                version = version or call.result.get("version")
                exp_id = exp_id or call.result.get("exp_id")
                if version is not None and exp_id is not None:
                    break
        llm_outcome = final.get("outcome") if isinstance(final.get("outcome"), dict) else {}
        outcome = {
            "action": str(final.get("action") or ("deploy" if deployed else "noop")),
            "deployed": bool(deployed and final.get("deploy", True)),
            "version": version,
            "exp_id": exp_id,
            "llm_confidence": _as_float(final.get("confidence", 0.0)),
            "llm_action": final.get("action"),
            "llm_claimed_deploy": bool(final.get("deploy", False)),
            **to_plain(llm_outcome),
        }
        if bool(final.get("deploy", False)) and not deployed:
            outcome["deploy_not_executed"] = True
        alternatives = [{"playbook": c["playbook"],
                         "why_not": "LLM 未选择"}
                        for c in candidates if c["playbook"] != chosen]
        rationale = str(final.get("reasoning") or thought or "LLM 最终决策")
        safety = {
            "driver": "llm",
            "provider": getattr(self.provider, "name", ""),
            "kill_switch": _kill_switch_active(tools.ctx),
            "human_gate": "escalation_recorded",
            "tool_calls": [c.name for c in steps],
            "denied_calls": [c.name for c in steps if not c.ok],
        }
        if outcome.get("deploy_not_executed"):
            rationale += "（注意：模型声称 deploy 但未调用 deploy 工具，实际未上线）"
        return _make_decision(
            t=t, phase=phase, diagnosis=diagnosis, chosen_playbook=chosen,
            alternatives=alternatives, budget_before=budget_before,
            budget_after=budget_after, cost=_cost_sum(steps), outcome=outcome,
            rationale=rationale,
            evidence_refs=_collect_evidence(steps),
            driver=self.driver_tag, safety=safety, tokens=tokens)

    def _charge_tokens(self, tools: ToolRegistry, total_tokens: int,
                       phase: int) -> None:
        if not self.charge_budget_tokens:
            return
        ctx = tools.ctx
        charge = getattr(ctx, "charge", None)
        if not callable(charge):
            return
        try:
            # 只补扣新增 token（BudgetState.charge 累加），labels 已由 oracle 计
            state = _ctx_state(ctx)
            prev = int(state.get("_llm_tokens_charged", 0))
            delta = max(0, int(total_tokens) - prev)
            if delta <= 0:
                return
            charge(Cost(tokens=delta), phase=phase, labels_already_charged=True)
            state["_llm_tokens_charged"] = int(total_tokens)
        except Exception:  # noqa: BLE001 - 预算记账失败不打断决策
            pass

    def _fallback_trajectory(self, snapshot: dict, tools: ToolRegistry,
                             reason: str, tokens: int, steps: list[ToolCall],
                             t0: float, parse_failures: int, t: int, phase: int,
                             budget_before: dict,
                             diagnosis: Optional[Diagnosis]) -> Trajectory:
        fallback_snapshot = dict(snapshot)
        try:
            traj = self.fallback.decide(fallback_snapshot, tools)
        except Exception as exc:  # noqa: BLE001
            outcome = {"reason": reason, "fallback_error":
                       f"{type(exc).__name__}: {exc}"}
            decision = _make_decision(
                t=t, phase=phase, diagnosis=diagnosis, chosen_playbook="NOOP",
                alternatives=[], budget_before=budget_before,
                budget_after=_budget_after(tools, snapshot), cost=_cost_sum(steps),
                outcome=outcome, rationale="LLM 失败且确定性回退也异常：NOOP",
                evidence_refs=_collect_evidence(steps),
                driver="llm-fallback",
                safety={"driver": "llm-fallback", "reason": reason})
            return Trajectory(t=t, phase=phase, steps=steps,
                              final_decision=decision, tokens=tokens,
                              wall_s=time.perf_counter() - t0,
                              driver="llm-fallback", outcome=outcome)
        merged_steps = list(steps) + list(traj.steps)
        decision = traj.final_decision
        if decision is not None:
            decision.driver = "llm-fallback"
            decision.cost.tokens = int(decision.cost.tokens) + int(tokens)
            decision.safety = {**to_plain(decision.safety),
                               "llm_fallback_reason": reason,
                               "llm_tokens": tokens,
                               "parse_failures": parse_failures}
        outcome = {**to_plain(traj.outcome), "fallback_reason": reason,
                   "llm_tokens": tokens, "parse_failures": parse_failures,
                   "llm_steps": len(steps)}
        return Trajectory(t=t, phase=phase, steps=merged_steps,
                          final_decision=decision, tokens=tokens,
                          wall_s=time.perf_counter() - t0,
                          driver="llm-fallback", outcome=outcome)


# --------------------------------------------------------------------------
# 工厂（loop 集成入口）
# --------------------------------------------------------------------------
def make_provider(cfg: Optional[dict] = None,
                  provider: Optional[str] = None) -> tuple[Any, dict]:
    """按 cfg.harness.provider 构造 provider，返回 (provider|None, meta)。"""
    cfg = cfg or load_config()
    hcfg = cfg.get("harness", {}) or {}
    name = str(provider or hcfg.get("provider", "mock")).lower()
    model = str(hcfg.get("model", "") or "")
    if name in ("mock", "offline", "mock_llm"):
        prov = MockLLMProvider(model=model or "mock-trainer-v1")
        return prov, {"provider": "mock", "model": prov.model,
                      "enabled": True, "network": False}
    if name in ("openai_compat", "openai", "pytri0", "llm"):
        prov = OpenAICompatProvider.from_env(cfg=cfg)
        meta = prov.health()
        if not prov.enabled:
            return None, {**meta, "fallback_reason":
                          "未配置 ZHIXUN_LLM_BASE_URL/ZHIXUN_LLM_MODEL 等环境变量"}
        return prov, meta
    return None, {"provider": name, "enabled": False,
                  "fallback_reason": f"未知 provider {name!r}（支持 mock|openai_compat）"}


def build_driver(cfg: Optional[dict] = None, tools: Optional[ToolRegistry] = None,
                 provider: Optional[Any] = None,
                 skills_dir: Optional[str | Path] = None) -> HarnessDriver:
    """loop 集成入口：`build_driver(cfg, tools=None) -> HarnessDriver`。

    - `cfg.harness.driver=deterministic`（默认）→ DeterministicDriver（离线、零 token）；
    - `deterministic` 之外为 `llm`/`auto`：
      * provider=mock → LLMDriver(MockLLMProvider)（可离线验证完整 LLM 协议）；
      * provider=openai_compat → 读环境变量构造；未配置/连接失败 → **降级**
        DeterministicDriver（auto 模式未配置时直接确定性，避免无谓开销）；
    - 返回对象带 `.provider_info` / `.skill_ids` 供 loop 打印审计信息。
    """
    cfg = cfg or load_config()
    hcfg = cfg.get("harness", {}) or {}
    mode = str(hcfg.get("driver", "deterministic")).lower()
    if mode not in ("deterministic", "llm", "auto"):
        mode = "deterministic"
    if mode == "deterministic":
        driver = DeterministicDriver(cfg, skills_dir=skills_dir)
        driver.provider_info = {"driver": "deterministic", "provider": None,
                                "enabled": True, "network": False}
        return driver
    if provider is None:
        provider, meta = make_provider(cfg)
    else:
        meta = getattr(provider, "health", lambda: {})()
    if provider is None:
        driver = DeterministicDriver(cfg, skills_dir=skills_dir)
        driver.provider_info = {"driver": "deterministic",
                                "fallback_from": mode, **meta}
        return driver
    if mode == "auto":
        # auto：只有真实 LLM（非 mock）可用时才走 LLM，否则直接用确定性，
        # 避免"以为在用大模型、其实在跑 mock"的审计歧义。
        if meta.get("provider") != "openai_compat":
            driver = DeterministicDriver(cfg, skills_dir=skills_dir)
            driver.provider_info = {"driver": "deterministic",
                                    "fallback_from": "auto", **meta}
            return driver
    driver = LLMDriver(provider, cfg, skills_dir=skills_dir,
                       fallback=DeterministicDriver(cfg, skills_dir=skills_dir))
    driver.provider_info = {**driver.provider_info, **meta}
    return driver


def harness_turn(cfg: Optional[dict] = None, snapshot: Optional[dict] = None,
                 tools: Optional[ToolRegistry] = None,
                 driver: Optional[HarnessDriver] = None,
                 ctx: Optional[ToolContext] = None) -> Trajectory:
    """便捷入口：一次完整的 Harness 决策回合（loop 可直接调用）。"""
    cfg = cfg or load_config()
    tools = tools or build_tool_registry(ctx=ctx, cfg=cfg)
    driver = driver or build_driver(cfg, tools)
    snapshot = snapshot or tools.ctx.snapshot()  # type: ignore[attr-defined]
    return driver.decide(snapshot, tools)


__all__ = [
    "HarnessDriver", "DeterministicRules", "DeterministicDriver", "LLMDriver",
    "build_driver", "make_provider", "harness_turn",
    "SNAPSHOT_PREFIX", "RESULT_PREFIX",
]
