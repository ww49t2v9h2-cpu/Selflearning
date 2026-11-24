"""确定性离线 Mock LLM Provider。

用途
----
1. **无网/无额度演示**：完整跑通 LLM 驱动的"思考→工具调用 JSON→最终决策"协议，
   证明 Harness 的 LLM 接口是真实可接线（不是硬编码分支）；
2. **单测**：`script=[...]` 可精确构造各种坏输出（非 JSON / 未知工具 / 非法参数），
   验证 LLMDriver 的重试与降级；
3. **token 计量**：按字符估算 prompt/completion tokens，与 `providers/base.py`
   同一口径，保证离线预算核算可复现。

协议
----
模型每一步只输出**一个** JSON 对象：

- 调工具：`{"thought": "...", "tool": "inspect_drift", "args": {...}}`
- 最终决策：`{"thought": "...", "final": {"chosen_playbook": "...", "action": "...",
  "reasoning": "...", "deploy": true, "version": 3, "outcome": {...}}}`

默认 `mode="rule"`：按"看漂移→诊断→查预算→跑实验→评估→上线/收手"的确定性
剧本消费工具结果，行为与 `DeterministicDriver` 同源（skills/*.md 的可执行子集）。
"""

from __future__ import annotations

import json
import time
from typing import Any, Optional

from .base import (
    RESULT_PREFIX,
    SNAPSHOT_PREFIX,
    ProviderError,
    ProviderResult,
    count_message_tokens,
    estimate_tokens,
)


def _try_json(text: Any) -> Any:
    if not isinstance(text, str):
        return None
    text = text.strip()
    if not text.startswith("{"):
        return None
    try:
        return json.loads(text)
    except ValueError:
        return None


def _extract_blocks(messages: list[dict]) -> tuple[dict, list[dict]]:
    """从对话里还原 (snapshot, tool_results)。"""
    snapshot: dict = {}
    results: list[dict] = []
    for msg in messages or []:
        content = msg.get("content", "") if isinstance(msg, dict) else ""
        if not isinstance(content, str):
            continue
        if content.startswith(SNAPSHOT_PREFIX):
            data = _try_json(content[len(SNAPSHOT_PREFIX):])
            if isinstance(data, dict):
                snapshot = data.get("snapshot", data)
        elif content.startswith(RESULT_PREFIX):
            data = _try_json(content[len(RESULT_PREFIX):])
            if isinstance(data, dict):
                results.append(data)
    return snapshot, results


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


class MockLLMProvider:
    """离线规则模拟 provider（确定性、无网络、token 按字符估算）。"""

    name = "mock"

    def __init__(self, model: str = "mock-trainer-v1",
                 script: Optional[list[str]] = None,
                 mode: str = "rule"):
        self.model = model or "mock-trainer-v1"
        self.script = list(script) if script else None
        self.mode = mode
        self.calls: list[dict] = []
        self._step = 0

    # ------------------------------------------------------------------
    def complete(self, messages: list[dict], temperature: float = 0.2,
                 max_tokens: int = 2048) -> ProviderResult:
        t0 = time.time()
        if self.script is not None:
            idx = min(self._step, len(self.script) - 1)
            text = self.script[idx]
            self._step += 1
        else:
            text = self._rule_response(messages)
        prompt_tokens = count_message_tokens(messages)
        completion_tokens = estimate_tokens(text)
        result = ProviderResult(
            text=text,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            wall_s=time.time() - t0,
            model=self.model,
            finish_reason="stop",
        )
        self.calls.append({"messages": len(messages or []),
                           "text": text,
                           "prompt_tokens": prompt_tokens,
                           "completion_tokens": completion_tokens})
        return result

    # ------------------------------------------------------------------
    def _rule_response(self, messages: list[dict]) -> str:
        snapshot, results = _extract_blocks(messages)
        names = [r.get("tool") for r in results if isinstance(r, dict)]
        last = results[-1] if results else {}
        last_name = last.get("tool", "")
        last_ok = bool(last.get("ok", False))
        last_result = last.get("result", {}) if isinstance(last.get("result"), dict) else {}
        seq = len(results)

        if seq >= 9:  # 硬上限，防止 mock 自己死循环
            return self._final(snapshot, "NOOP", "达到 mock 步数上限，收手记录 NOOP。")

        if not results:
            return self._tool("inspect_drift", {},
                              "先看监测报告：漂移是否存在、强度与受影响特征。")

        # 1) 看完漂移：无漂移 → 直接 NOOP
        if last_name == "inspect_drift":
            if not last_ok:
                return self._final(snapshot, "NOOP",
                                   f"监测读取失败（{last.get('error', '')}），保守收手。")
            detected = bool(last_result.get("drift_detected"))
            severity = float(last_result.get("max_severity", 0.0) or 0.0)
            if not detected or severity <= 0.05:
                return self._final(snapshot, "NOOP",
                                   "监测未发现显著漂移（severity≤0.05），本轮不更新。")
            return self._tool("diagnose", {"force": True},
                              "存在显著漂移，先做归因诊断（类型/范围/先验）。")

        # 2) 诊断完成：查预算
        if last_name == "diagnose":
            if not last_ok:
                return self._final(snapshot, "NOOP",
                                   "诊断工具失败，保守收手（不盲目实验）。")
            return self._tool("budget_status", {},
                              "确认剩余 labels/compute/tokens 与 per-phase 更新额度。")

        # 3) 预算：从候选里选 playbook 跑实验
        if last_name == "budget_status":
            candidates = self._pick_candidates(snapshot, last_result)
            if not candidates:
                return self._final(snapshot, "NOOP", "候选为空或预算不足，收手。")
            chosen = candidates[0]
            playbook = str(chosen.get("playbook", "RECALIBRATE_AFFINE"))
            return self._tool("run_experiment",
                              {"playbook": playbook,
                               "hypothesis": "按候选先验与 cheapest-first 规则选择"},
                              f"按技能规则选择 {playbook}（预算内、风险最低且先验最高）。")

        # 4) 实验完成：评估
        if last_name == "run_experiment":
            if not last_ok:
                return self._final(snapshot,
                                   self._playbook_from_candidates(snapshot),
                                   f"实验执行失败（{last.get('error', '')}），"
                                   "未产生可上线候选，收手并留待下轮。")
            exp_id = last_result.get("exp_id", "")
            version = _as_int(last_result.get("version"))
            args = {"exp_id": exp_id} if exp_id else {"version": version}
            return self._tool("evaluate_candidate", args,
                              "候选实验完成，按门禁口径评估 probe 增益/召回/成本。")

        # 5) 评估完成：门禁通过则上线
        if last_name == "evaluate_candidate":
            if not last_ok:
                return self._final(snapshot, self._playbook_from_candidates(snapshot),
                                   f"评估失败（{last.get('error', '')}），候选不上线。")
            gate = last_result.get("gate", {}) if isinstance(last_result.get("gate"), dict) else {}
            passed = bool(last_result.get("passed", gate.get("passed", False)))
            version = _as_int(last_result.get("version"), -1)
            playbook = str(last_result.get("playbook")
                           or self._playbook_from_candidates(snapshot))
            if not passed or version < 0:
                return self._final(snapshot, playbook,
                                   f"门禁未通过（{gate.get('reason', '未达门槛')}），"
                                   "候选保留为 shadow 证据，不上线。")
            return self._tool("deploy", {"version": version,
                                         "reason": "门禁通过，按技能上线门禁规则晋级"},
                              f"候选 v{version} 过门禁，执行上线（记录 escalation 供审计）。")

        # 6) 上线完成：给出最终决策
        if last_name == "deploy":
            playbook = str(last_result.get("playbook")
                           or self._playbook_from_candidates(snapshot))
            if last_ok and last_result.get("ok", True):
                return self._final(snapshot, playbook,
                                   f"已上线 v{last_result.get('version')}；"
                                   "继续监测，若回滚窗口内掉点则触发回滚。",
                                   deploy=True,
                                   version=_as_int(last_result.get("version"), -1),
                                   outcome={"deployed": True,
                                            "version": last_result.get("version")})
            return self._final(snapshot, playbook,
                               f"上线被拒/需升级：{last.get('error', '')}；"
                               "保留当前 champion，记录 escalation。",
                               outcome={"deployed": False})

        if last_name == "shadow_eval":
            return self._final(snapshot, self._playbook_from_candidates(snapshot),
                               "影子评估完成：候选与 champion 的稳定性差异已记录，"
                               "本轮不改线上模型。")

        # 兜底
        return self._final(snapshot, "NOOP", f"未知工具结果 {last_name!r}，保守收手。")

    # ------------------------------------------------------------------
    @staticmethod
    def _final(snapshot: dict, playbook: str, reasoning: str,
               deploy: bool = False, version: int = -1, outcome: Optional[dict] = None) -> str:
        candidates = [str(c.get("playbook", "")) for c in
                      (snapshot.get("candidates") or []) if isinstance(c, dict)]
        if playbook not in candidates and playbook != "NOOP":
            playbook = candidates[0] if candidates else "NOOP"
        payload = {
            "thought": reasoning,
            "final": {
                "chosen_playbook": playbook,
                "action": "deploy" if deploy else ("noop" if playbook == "NOOP" else "run_experiment"),
                "reasoning": reasoning,
                "deploy": bool(deploy),
                "version": version,
                "confidence": 0.7 if playbook != "NOOP" else 0.85,
                "outcome": outcome or {},
            },
        }
        return json.dumps(payload, ensure_ascii=False)

    @staticmethod
    def _tool(name: str, args: dict, thought: str) -> str:
        return json.dumps({"thought": thought, "tool": name, "args": args},
                          ensure_ascii=False)

    @staticmethod
    def _playbook_from_candidates(snapshot: dict) -> str:
        cands = snapshot.get("candidates") or []
        for c in cands:
            if isinstance(c, dict) and c.get("playbook"):
                return str(c["playbook"])
        return "NOOP"

    def _pick_candidates(self, snapshot: dict, budget_result: dict) -> list[dict]:
        cands = [c for c in (snapshot.get("candidates") or []) if isinstance(c, dict)]
        if not cands:
            return []
        # 预算硬约束（工具侧仍会再检查一次）
        labels_left = _as_int(budget_result.get("labels_left"), 10 ** 9)
        compute_left = float(budget_result.get("compute_left", 1e9) or 1e9)
        updates_left = _as_int(budget_result.get("updates_left_phase"), 10 ** 9)
        if updates_left <= 0:
            return []
        avail = []
        for c in cands:
            est = c.get("est_cost") or {}
            if isinstance(est, dict):
                if _as_int(est.get("labels", 0)) > labels_left:
                    continue
                if float(est.get("compute_units", 0.0) or 0.0) > compute_left:
                    continue
            avail.append(c)
        return avail


__all__ = ["MockLLMProvider"]
