"""安全护栏（SafetyGuard）与 sha256 哈希链审计日志（AuditLog）。

对应 docs/03 §4.7 契约：
    SafetyGuard(cfg, budget).check(action, payload, state) -> (allowed, checks)
    SafetyGuard(cfg, budget).should_kill() -> bool
    AuditLog(path).append(record) -> dict；.verify() -> bool

护栏检查项（每项写入 Decision.safety，可审计）
--------------------------------------------
1. kill-switch：`STOP` 文件存在时只允许 NOOP/ROLLBACK 等收尾动作；
2. 动作白名单：playbook 必须在 `safety.allowed_playbooks`；
3. 预算硬门：标签/算力/token 不得超剩余预算；单词决策标签 ≤ max_labels_per_decision；
4. 更新频率：单窗口更新次数 ≤ max_updates_per_window；每 phase 更新数受
   BudgetState.per_phase_updates 约束；
5. 连续失败上限：连续失败达到 max_consecutive_failures 后禁止再实验，只允许回滚；
6. 决策/token 上限：单次决策 token ≤ max_tokens_per_decision；
7. 越权/网络禁止：项目运行期不允许网络动作（allow_network=false）；
8. 高风险升级：高风险 playbook 需人工批准或（校赛封闭评测）自动升级记录留痕。

审计日志哈希链：
    hash_i = sha256(prev_hash + canonical_json(record_i))，prev_hash_0 = "0"*64
`verify()` 逐行重算并检查 seq/prev_hash/hash 连续性；任何篡改都会被发现。
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Optional

from ..core.config import ROOT
from .defaults import resolve_config
from .utils import canonical_json_bytes, now_ts, read_json

GENESIS = "0" * 64


# --------------------------------------------------------------------------
# SafetyGuard
# --------------------------------------------------------------------------
class SafetyGuard:
    # 只读 / 收尾动作即使在 kill-switch 下也允许
    SAFE_ACTIONS = {"NOOP", "ROLLBACK", "inspect_drift", "budget_status", "query_history"}
    EXPERIMENT_ACTIONS = {"LAST_LAYER_FT", "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED",
                          "NOVEL_CLASS_EXTEND", "SHADOW_ONLY", "RECALIBRATE_AFFINE",
                          "PRIOR_CORRECTION", "ROLLBACK"}

    def __init__(self, cfg: dict, budget):
        self.cfg = resolve_config(cfg)
        self.scfg = self.cfg["safety"]
        self.budget = budget
        self.allowed_playbooks = set(self.scfg.get("allowed_playbooks", []))
        self.high_risk = set(self.scfg.get("high_risk_playbooks", []) or [])
        self.max_consecutive_failures = int(self.scfg.get("max_consecutive_failures", 3))
        self.max_tokens = int(self.scfg.get("max_tokens_per_decision", 20000))
        self.max_decisions = int(self.scfg.get("max_decisions", 80))
        self.max_labels_per_decision = int(self.scfg.get("max_labels_per_decision", 160))
        self.max_updates_per_window = int(self.scfg.get("max_updates_per_window", 1))
        self.allow_network = bool(self.scfg.get("allow_network", False))
        self.auto_approve_high_risk = bool(self.scfg.get("auto_approve_high_risk", False))
        self.human_gate = bool(self.scfg.get("human_gate", True))
        per_phase = self.cfg.get("budget", {}).get("per_phase_updates", 3)
        self.per_phase_updates = int(per_phase or 3)
        self.decisions_made = 0
        self.consecutive_failures = 0
        self.window_updates: dict = {}
        self.escalations: list[dict] = []
        self._kill_override = False

    # ------------------------------------------------------------------
    def should_kill(self) -> bool:
        if self._kill_override:
            return True
        files = []
        single = self.scfg.get("kill_switch_file")
        if single:
            files.append(str(single))
        files.extend([str(f) for f in (self.scfg.get("kill_switch_files") or [])])
        for f in files:
            p = Path(f)
            cands = [p] if p.is_absolute() else [ROOT / p, Path.cwd() / p]
            for c in cands:
                if c.exists():
                    return True
        return False

    def trigger_kill(self, value: bool = True) -> None:
        """测试/演练用：程序化触发 kill-switch。"""
        self._kill_override = bool(value)

    # ------------------------------------------------------------------
    def check(self, action: str, payload: Optional[dict] = None,
              state: Optional[dict] = None) -> tuple[bool, dict]:
        action = str(action or "")
        payload = dict(payload or {})
        state = dict(state or {})
        checks: dict = {}

        def add(name: str, ok: bool, value=None, limit=None, note: str = "") -> None:
            checks[name] = {"pass": bool(ok), "value": value, "limit": limit,
                            "note": note}

        # 1) kill-switch
        kill = self.should_kill()
        safe = action in self.SAFE_ACTIONS
        add("kill_switch", (not kill) or safe,
            value=bool(kill), limit=False,
            note="kill-switch 文件存在" if kill else "")
        blocked_by_kill = kill and not safe

        # 2) 动作白名单
        if action:
            add("allowed_playbook", action in self.allowed_playbooks,
                value=action, limit=sorted(self.allowed_playbooks))
        else:
            add("allowed_playbook", True, value="", limit="empty-action")

        # 3) 预算硬门
        try:
            labels_left = int(self.budget.labels_left())
            compute_left = float(self.budget.compute_left())
            tokens_left = int(self.budget.tokens_left())
        except Exception:
            labels_left, compute_left, tokens_left = 10 ** 9, 1e9, 10 ** 9
        est_labels = payload.get("labels_est", payload.get("labels", 0))
        if est_labels is None:
            est_labels = 0
        est_compute = payload.get("compute_units_est", payload.get("compute_units", 0.0))
        if est_compute is None:
            est_compute = 0.0
        tokens = int(payload.get("tokens", state.get("tokens", 0)) or 0)
        add("budget_labels", int(est_labels) <= labels_left,
            value=int(est_labels), limit=labels_left, note="标签预算硬门")
        add("budget_compute", float(est_compute) <= compute_left + 1e-9,
            value=float(est_compute), limit=compute_left, note="算力预算硬门")
        add("per_decision_labels", int(est_labels) <= self.max_labels_per_decision,
            value=int(est_labels), limit=self.max_labels_per_decision)
        if action in self.EXPERIMENT_ACTIONS and int(est_labels) <= 0:
            # 估计缺失时用配置里的 playbook 名义标签需求做保守检查
            pb = self.cfg.get("playbooks", {}).get(action, {}) or {}
            nominal = int(pb.get("labels", 0)) + int(pb.get("labels_eval", 0))
            add("nominal_labels_available", nominal <= labels_left,
                value=nominal, limit=labels_left,
                note="playbook 名义标签需求 vs 剩余预算")

        # 4) 更新频率
        phase = payload.get("phase", state.get("phase", None))
        window = payload.get("t", state.get("t", None))
        in_exp = action in self.EXPERIMENT_ACTIONS
        if in_exp:
            try:
                ups_left = int(self.budget.updates_left(
                    int(phase) if phase is not None else 0, self.per_phase_updates))
            except Exception:
                ups_left = 10 ** 9
            n_window = int(self.window_updates.get(int(window), 0)) if window is not None else 0
            add("updates_budget", ups_left > 0, value=ups_left, limit=0)
            add("updates_per_window", n_window < self.max_updates_per_window,
                value=n_window, limit=self.max_updates_per_window)

        # 5) 连续失败
        add("consecutive_failures", self.consecutive_failures < self.max_consecutive_failures,
            value=self.consecutive_failures, limit=self.max_consecutive_failures,
            note="连续失败达到上限后只允许回滚/NOOP")
        blocked_by_failures = (self.consecutive_failures >= self.max_consecutive_failures
                               and not safe)

        # 6) 决策 / token 上限
        add("decision_cap", self.decisions_made < self.max_decisions,
            value=self.decisions_made, limit=self.max_decisions)
        add("token_cap", tokens <= self.max_tokens, value=tokens, limit=self.max_tokens)

        # 7) 网络禁止
        wants_network = bool(payload.get("network")) or action.lower() in (
            "network", "http", "download")
        add("network_guard", (not wants_network) or self.allow_network,
            value=bool(wants_network), limit=False, note="运行期禁止网络访问")

        # 8) 高风险升级
        if action in self.high_risk:
            if self.auto_approve_high_risk:
                mode, ok = "escalation_auto_approved", True
            elif payload.get("human_approval") is True:
                mode, ok = "human_approved", True
            elif not self.human_gate:
                mode, ok = "blocked", False
            else:
                mode, ok = "awaiting_human", False
            add("high_risk_gate", ok, value=mode, limit="human_or_auto",
                note="高风险动作升级记录")
            if ok:
                self.escalations.append({
                    "ts": now_ts(), "action": action, "mode": mode,
                    "t": payload.get("t"), "phase": payload.get("phase"),
                    "diagnosis": payload.get("diagnosis"),
                })
                self.escalations = self.escalations[-200:]
            blocked_high = not ok
        else:
            blocked_high = False

        allowed = (not blocked_by_kill and not blocked_by_failures and not blocked_high
                   and all(bool(c["pass"]) for c in checks.values()))
        return bool(allowed), checks

    # ------------------------------------------------------------------
    # 运行期记账（loop 可选调用；不调用也不影响判定）
    # ------------------------------------------------------------------
    def record_decision(self, t: Optional[int] = None) -> None:
        self.decisions_made += 1
        if t is not None:
            self.window_updates[int(t)] = self.window_updates.get(int(t), 0) + 1

    def record_outcome(self, success: bool) -> None:
        self.consecutive_failures = 0 if success else self.consecutive_failures + 1

    def reset_failures(self) -> None:
        self.consecutive_failures = 0

    def summary(self) -> dict:
        return {"decisions_made": self.decisions_made,
                "consecutive_failures": self.consecutive_failures,
                "kill_switch": self.should_kill(),
                "escalations": list(self.escalations[-10:])}


# --------------------------------------------------------------------------
# AuditLog（sha256 哈希链）
# --------------------------------------------------------------------------
class AuditLog:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._last_hash = GENESIS
        self._seq = 0
        self._load_tail()

    def _load_tail(self) -> None:
        if not self.path.exists() or self.path.stat().st_size == 0:
            self._seq, self._last_hash = 0, GENESIS
            return
        last = None
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    last = line
        if last is None:
            self._seq, self._last_hash = 0, GENESIS
            return
        try:
            rec = json.loads(last)
            self._seq = int(rec["seq"]) + 1
            self._last_hash = str(rec["hash"])
        except Exception:
            # 尾部损坏：不静默修复，标记使 verify() 失败并拒绝继续追加
            self._seq = -1
            self._last_hash = "BROKEN"

    @staticmethod
    def _hash(prev_hash: str, record: dict) -> str:
        h = hashlib.sha256()
        h.update(str(prev_hash).encode("utf-8"))
        h.update(canonical_json_bytes(record))
        return h.hexdigest()

    def append(self, record: dict) -> dict:
        if self._seq < 0:
            raise RuntimeError("审计日志尾部损坏，拒绝追加（请人工检查）")
        rec = {"seq": int(self._seq), "prev_hash": self._last_hash,
               "hash": self._hash(self._last_hash, record), "record": record}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False, sort_keys=True) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._seq += 1
        self._last_hash = rec["hash"]
        return rec

    def verify(self) -> bool:
        if not self.path.exists():
            return True
        prev = GENESIS
        expect_seq = 0
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    rec = json.loads(line)
                    if int(rec.get("seq", -1)) != expect_seq:
                        return False
                    if str(rec.get("prev_hash")) != prev:
                        return False
                    body = rec.get("record")
                    if self._hash(prev, body) != str(rec.get("hash")):
                        return False
                    prev = str(rec["hash"])
                    expect_seq += 1
        except Exception:
            return False
        return True

    def last_hash(self) -> str:
        return self._last_hash

    def __len__(self) -> int:
        return max(0, self._seq)

    def records(self) -> list[dict]:
        out = []
        if not self.path.exists():
            return out
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    out.append(json.loads(line))
        return out
