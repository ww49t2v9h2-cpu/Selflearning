"""预算感知的候选规划器 + 策略层自改进（bandit）。

对应 docs/03 §4.4 契约：
    Planner(cfg, budget, library, policy_path)
    .propose(diagnosis, chunk, state) -> list[Candidate]
    .order(candidates) -> list[Candidate]
    .update_strategy(diagnosis_kind, playbook, reward, cost) -> None
    Candidate = {"playbook": str, "est_cost": Cost, "prior": float, "why": str}

算法（详见 docs/05 §4）
----------------------
1. **预算硬过滤**：`library.plan_labels()` 判定不可行（标签/算力不够）直接剔除；
   白名单（safety.allowed_playbooks）之外的 playbook 不进候选。
2. **bandit 后验**：策略表按 `(diagnosis_kind, playbook)` 维护
   Beta 后验成功次数（成功 = 门禁通过且增益 > 0）与 EMA 增益/成本；
   打分 = (1-w)·诊断先验 + w·后验成功率 + UCB 探索项 − cost_penalty·估计成本。
   默认确定性 UCB（无随机数），保证同配置可复现；`planner.mode=random` 留给消融。
3. **cheapest-first**：`order()` 按估计成本升序、同价按分数排序；`propose()` 末尾
   保证至少一个"便宜低风险"候选（避免预算紧时只剩昂贵动作）。
4. **持久化**：策略表原子写 `experiments/strategy_policy.json`（相对 ROOT），
   含 schema 版本、更新次数、每格后验统计——这是"策略层自改进"的可审计证据。
"""

from __future__ import annotations

import copy
import math
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import ROOT
from ..core.spec import Cost, make_cost
from .defaults import resolve_config
from .utils import now_ts, read_json, atomic_write_json, to_native

POLICY_SCHEMA = 1


class Planner:
    def __init__(self, cfg: dict, budget, library, policy_path=None):
        self.cfg = resolve_config(cfg)
        self.pcfg = self.cfg["planner"]
        self.budget = budget
        self.library = library
        self.allowed = set(self.cfg["safety"].get("allowed_playbooks", []))
        self.max_candidates = int(self.pcfg.get("max_candidates", 3))
        self.ucb_c = float(self.pcfg.get("ucb_c", 0.35))
        self.prior_weight = float(np.clip(self.pcfg.get("prior_weight", 0.5), 0.0, 1.0))
        self.cost_penalty = float(self.pcfg.get("cost_penalty", 0.02))
        self.prior_strength = float(self.pcfg.get("prior_strength", 2.0))
        self.reward_ema = float(np.clip(self.pcfg.get("reward_ema", 0.6), 0.0, 1.0))
        self.mode = str(self.pcfg.get("mode", "bandit"))
        self.cheapest_first = bool(self.pcfg.get("cheapest_first", True))
        self.always_keep_cheap = bool(self.pcfg.get("always_keep_cheap", True))
        self.seed = int(self.cfg.get("seed", 0))
        if policy_path is None:
            policy_path = self.pcfg.get("policy_path", "experiments/strategy_policy.json")
        p = Path(policy_path)
        self.policy_path = p if p.is_absolute() else (ROOT / p)
        self.policy: dict = self._load_policy()
        self.n_updates = 0

    # ------------------------------------------------------------------
    # 策略表
    # ------------------------------------------------------------------
    def _empty_policy(self) -> dict:
        return {"schema": POLICY_SCHEMA, "created_at": now_ts(), "updated_at": now_ts(),
                "n_updates": 0, "table": {}, "events": []}

    def _load_policy(self) -> dict:
        p = read_json(self.policy_path, default=None)
        if not isinstance(p, dict) or "table" not in p:
            return self._empty_policy()
        p.setdefault("schema", POLICY_SCHEMA)
        p.setdefault("events", [])
        p.setdefault("n_updates", 0)
        return p

    def save(self) -> None:
        self.policy["updated_at"] = now_ts()
        self.policy["n_updates"] = int(self.policy.get("n_updates", 0))
        atomic_write_json(self.policy_path, to_native(self.policy))

    def stats(self, diagnosis_kind: str, playbook: str) -> dict:
        cell = (self.policy.get("table", {}).get(str(diagnosis_kind), {}) or {}).get(playbook)
        if not cell:
            return {"n": 0, "successes": 0, "failures": 0, "success_rate": 0.0,
                    "ema_reward": 0.0, "ema_cost": 0.0, "last_gain": 0.0,
                    "posterior_success": 0.0, "trials": 0}
        n = int(cell.get("n", 0))
        succ = int(cell.get("successes", 0))
        return {**cell, "n": n, "successes": succ,
                "success_rate": float(succ / max(1, n)),
                "trials": n}

    def _posterior(self, diagnosis_kind: str, playbook: str,
                   prior: float) -> tuple[float, float, dict]:
        """返回 (后验成功率, UCB 探索项, 统计快照)。"""
        cell = self.stats(diagnosis_kind, playbook)
        n = int(cell.get("n", 0))
        succ = int(cell.get("successes", 0))
        post = (succ + self.prior_strength * float(prior)) / max(
            1e-9, n + self.prior_strength)
        total = int(self.policy.get("n_updates", 0)) + 1
        explore = self.ucb_c * math.sqrt(math.log(total + 1.0) / (n + 1.0))
        return float(post), float(explore), cell

    def update_strategy(self, diagnosis_kind: str, playbook: str, reward: float,
                        cost: Optional[Cost] = None) -> None:
        """把一次候选实验的实际增益/成本写回策略表（策略层自改进的唯一入口）。"""
        reward = float(reward)
        cost = cost or Cost()
        kind = str(diagnosis_kind or "unknown")
        table = self.policy.setdefault("table", {})
        cell = table.setdefault(kind, {}).setdefault(playbook, {
            "n": 0, "successes": 0, "failures": 0,
            "ema_reward": 0.0, "ema_cost": 0.0, "sum_reward": 0.0,
            "sum_cost": 0.0, "last_gain": 0.0,
        })
        cell["n"] = int(cell.get("n", 0)) + 1
        success = reward > 0.0
        if success:
            cell["successes"] = int(cell.get("successes", 0)) + 1
        else:
            cell["failures"] = int(cell.get("failures", 0)) + 1
        a = self.reward_ema
        cell["ema_reward"] = float(a * float(cell.get("ema_reward", 0.0))
                                  + (1 - a) * reward)
        cell["ema_cost"] = float(a * float(cell.get("ema_cost", 0.0))
                                 + (1 - a) * float(cost.scalar))
        cell["sum_reward"] = float(cell.get("sum_reward", 0.0)) + reward
        cell["sum_cost"] = float(cell.get("sum_cost", 0.0)) + float(cost.scalar)
        cell["last_gain"] = float(reward)
        cell["last_cost"] = cost.to_dict()
        self.policy["n_updates"] = int(self.policy.get("n_updates", 0)) + 1
        self.n_updates += 1
        events = self.policy.setdefault("events", [])
        events.append({"t": int(cell["n"]), "kind": kind, "playbook": playbook,
                       "reward": round(reward, 6),
                       "success": bool(success),
                       "cost_scalar": round(float(cost.scalar), 6)})
        self.policy["events"] = events[-200:]
        self.save()

    # ------------------------------------------------------------------
    # 候选生成
    # ------------------------------------------------------------------
    def propose(self, diagnosis, chunk, state=None) -> list[dict]:
        state = state or {}
        kind = str(getattr(diagnosis, "kind", "unknown"))
        recs = list(getattr(diagnosis, "recommended", []) or [])
        if not recs:
            recs = [{"playbook": "NOOP", "prior": 1.0, "why": "无诊断推荐"}]
        candidates: list[dict] = []
        seen = set()
        for rec in recs:
            pb = str(rec.get("playbook", "NOOP"))
            if pb in seen:
                continue
            seen.add(pb)
            if pb not in self.allowed:
                continue
            plan = self.library.plan_labels(pb, diagnosis, chunk, self.budget) \
                if hasattr(self.library, "plan_labels") else {"feasible": True}
            if not plan.get("feasible", True):
                continue
            est = self.library.estimate_cost(pb, diagnosis, chunk, self.budget)
            if not self._budget_can_afford(est):
                continue
            prior = float(rec.get("prior", 0.0))
            post, explore, cell = self._posterior(kind, pb, prior)
            score = (self.prior_weight * prior + (1.0 - self.prior_weight) * post
                     + explore - self.cost_penalty * float(est.scalar))
            # 预算压力：单个实验可消耗的标签不超过剩余的 ~55%（低于绝对下限时不罚）
            try:
                left = float(self.budget.labels_left()) if self.budget is not None else 1e9
            except Exception:
                left = 1e9
            cap = max(48.0, 0.55 * left)
            if float(est.labels) > cap:
                score *= 0.45
            # 标签事实不足（诊断置信度低）时，昂贵带标签动作降权（优先 NOOP/零标签动作）
            conf = float(getattr(diagnosis, "confidence", 0.0) or 0.0)
            if conf < 0.50 and float(est.labels) > 40:
                score *= 0.30
            # 证据充分（novel 高置信）时才允许把整段预算押在扩展类别等昂贵动作上
            if kind == "novel_class" and conf >= 0.60:
                score *= 1.20
            if pb == "SHADOW_ONLY" and float(getattr(diagnosis, "severity", 0.0)) >= 0.55:
                score *= 0.5      # 严重漂移时影子评估不能替代真实动作
            score = float(np.clip(score, 1e-6, 1.0))
            n_hist = int(cell.get("n", 0))
            why = str(rec.get("why", ""))
            if n_hist > 0:
                why = (f"{why}；策略表历史 n={n_hist} 成功率="
                       f"{float(cell.get('successes', 0)) / max(1, n_hist):.2f} "
                       f"EMA收益={float(cell.get('ema_reward', 0.0)):.3f}")
            candidates.append({"playbook": pb, "est_cost": est, "prior": score,
                               "why": why, "feasible": True})
        if not candidates:
            candidates = [{"playbook": "NOOP", "est_cost": make_cost(),
                           "prior": 1.0, "why": "无满足预算/白名单的候选（兜底 NOOP）",
                           "feasible": True}]
        # bandit：按分数取 top-k；再保证至少一个便宜低风险候选
        candidates.sort(key=lambda c: -float(c["prior"]))
        top = candidates[: self.max_candidates]
        if (self.always_keep_cheap and len(top) < self.max_candidates
                and not any(c["playbook"] in ("NOOP", "RECALIBRATE_AFFINE",
                                              "PRIOR_CORRECTION", "SHADOW_ONLY")
                            for c in top)):
            cheap = [c for c in candidates if c["playbook"] in (
                "NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION",
                "LAST_LAYER_FT", "SHADOW_ONLY") and c not in top]
            if cheap:
                cheap.sort(key=lambda c: float(c["est_cost"].scalar))
                top.append(cheap[0])
        return self.order(top)

    def order(self, candidates: list[dict]) -> list[dict]:
        """cheapest-first：先试便宜动作；同成本按先验降序。"""
        if not candidates:
            return []
        if self.cheapest_first:
            return sorted(candidates, key=lambda c: (float(c["est_cost"].scalar),
                                                     -float(c.get("prior", 0.0))))
        return sorted(candidates, key=lambda c: -float(c.get("prior", 0.0)))

    def _budget_can_afford(self, cost: Cost) -> bool:
        if self.budget is None:
            return True
        try:
            if cost.labels > self.budget.labels_left():
                return False
            if cost.compute_units > self.budget.compute_left() + 1e-9:
                return False
            return True
        except Exception:
            return True

    # ------------------------------------------------------------------
    # 观测辅助（供记录/报告）
    # ------------------------------------------------------------------
    def table_snapshot(self) -> dict:
        return copy.deepcopy(self.policy.get("table", {}))

    def strategy_summary(self) -> dict:
        """给结果报告用的策略层摘要：每个 (kind, playbook) 的后验统计。"""
        table = self.policy.get("table", {})
        out = {}
        for kind, cells in table.items():
            out[kind] = {}
            for pb, cell in cells.items():
                n = int(cell.get("n", 0))
                out[kind][pb] = {
                    "n": n,
                    "success_rate": round(float(cell.get("successes", 0)) / max(1, n), 3),
                    "ema_reward": round(float(cell.get("ema_reward", 0.0)), 4),
                    "ema_cost": round(float(cell.get("ema_cost", 0.0)), 4),
                }
        return {"n_updates": int(self.policy.get("n_updates", 0)),
                "policy_path": str(self.policy_path), "table": out}
