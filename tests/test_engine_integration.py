"""引擎层集成测试：用合成漂移基准的 mini 闭环走通
监测 → 诊断 → 规划 → 实验 → 门禁 → 上线/回滚 → 策略表持久化 → 审计链校验。

为了测试速度，benchmark 配置被缩小（phase_n=300, chunk_size=100, base_train_n=600），
但数据生成、oracle 预算、引擎组件全部是生产代码路径。
"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config                       # noqa: E402
from src.core.metrics import evaluate                         # noqa: E402
from src.core.model import MLP                                # noqa: E402
from src.core.spec import BudgetState, Cost, Diagnosis        # noqa: E402
from src.data.synth import SyntheticOracle, make_benchmark    # noqa: E402
from src.engine.diagnosis import Diagnoser                    # noqa: E402
from src.engine.gates import Gate                             # noqa: E402
from src.engine.monitor import DriftMonitor                   # noqa: E402
from src.engine.planner import Planner                        # noqa: E402
from src.engine.playbooks import (InfeasiblePlaybook, PlaybookContext,  # noqa: E402
                                  PlaybookLibrary)
from src.engine.registry import ModelRegistry                 # noqa: E402
from src.engine.safety import AuditLog, SafetyGuard           # noqa: E402
from src.engine.utils import make_feature_groups              # noqa: E402
from src.engine.watchdog import Watchdog                      # noqa: E402


def small_cfg(scenario: str = "S7_full_mix", labels_total: int = 140) -> dict:
    return load_config(overrides={
        "benchmark": {"scenario": scenario, "phase_n": 300, "chunk_size": 100,
                      "base_train_n": 600, "base_val_n": 200},
        "model": {"epochs": 15, "lr": 0.005, "batch": 64},
        "budget": {"labels_total": labels_total, "compute_total": 120.0,
                   "tokens_total": 40000, "updates_total": 20,
                   "per_phase_updates": 3},
    })


def fit_base(cfg: dict, bench, seed: int = 0) -> MLP:
    model = MLP(bench.d, bench.base_classes, hidden=cfg["model"]["hidden"], seed=seed)
    model.fit(bench.base_x, bench.base_y, epochs=cfg["model"]["epochs"],
              lr=cfg["model"]["lr"], batch=cfg["model"]["batch"], seed=seed)
    return model


class TestEngineIntegration(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = tempfile.TemporaryDirectory()
        cls.root = Path(cls.tmp.name)
        cls.cfg = small_cfg()
        cls.bench = make_benchmark(cls.cfg, seed=0)
        cls.budget = BudgetState(
            labels_total=cls.cfg["budget"]["labels_total"],
            compute_total=float(cls.cfg["budget"]["compute_total"]),
            tokens_total=int(cls.cfg["budget"]["tokens_total"]),
            updates_total=int(cls.cfg["budget"]["updates_total"]))
        cls.oracle = SyntheticOracle(cls.bench.visible_y, cls.budget)
        cls.base_model = fit_base(cls.cfg, cls.bench, seed=0)

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    # ------------------------------------------------------------------
    def _components(self, suffix: str):
        cfg = self.cfg
        run_dir = self.root / f"run-{suffix}" / "registry"
        registry = ModelRegistry(run_dir, self.root / f"artifacts-{suffix}",
                                 self.bench.d, self.bench.base_classes, cfg)
        monitor = DriftMonitor(cfg, self.bench.d, self.bench.n_classes)
        n_warm = int(cfg["monitor"].get("warmup_chunks", 4))
        monitor.reset_reference(np.vstack([self.bench.chunks[t].x
                                           for t in range(n_warm)]))
        diagnoser = Diagnoser(cfg, self.bench.d, self.bench.n_classes,
                              make_feature_groups(self.bench.d))
        library = PlaybookLibrary(cfg, self.oracle, cfg.get("cost_model", {}))
        planner = Planner(cfg, self.budget, library,
                          self.root / f"run-{suffix}" / "strategy_policy.json")
        gate = Gate(cfg)
        audit = AuditLog(self.root / f"run-{suffix}" / "audit.jsonl")
        safety = SafetyGuard(cfg, self.budget)
        watchdog = Watchdog(cfg)
        v0 = registry.add(self.base_model, {"kind": "base", "playbook": "BASE",
                                            "created_t": -1, "metrics":
                                            evaluate(self.bench.val_y,
                                                     self.base_model.predict(
                                                         self.bench.val_x),
                                                     self.bench.base_classes)})
        registry.promote(v0.version)
        return registry, monitor, diagnoser, library, planner, gate, audit, safety, watchdog

    # ------------------------------------------------------------------
    def test_mini_closed_loop(self):
        (registry, monitor, diagnoser, library, planner, gate, audit, safety,
         watchdog) = self._components("loop")
        n_warm = int(self.cfg["monitor"].get("warmup_chunks", 4))
        decisions = 0
        deploys = 0
        for t, chunk in enumerate(self.bench.chunks):
            champ_meta, champion = registry.champion()
            report = monitor.update(chunk, champion, self.oracle)
            # 看门狗（无漂移窗口也应正常返回 none）
            wd = watchdog.observe(t, chunk.phase, 0.9, champ_meta.version)
            self.assertIn(wd["action"], ("none", "rollback", "recover"))
            if not report.events or t < n_warm:
                continue
            diag = diagnoser.diagnose(chunk, champion, report, [], self.oracle)
            self.assertGreaterEqual(int(diag.cost.labels), 0)
            cands = planner.propose(diag, chunk, report.state)
            self.assertLessEqual(len(cands),
                                 int(self.cfg["planner"]["max_candidates"]))
            chosen = None
            for cand in cands:
                allowed, checks = safety.check(cand["playbook"],
                                               {"t": t, "phase": chunk.phase,
                                                "labels_est": int(cand["est_cost"].labels)},
                                               {"t": t, "phase": chunk.phase})
                self.assertIn("allowed_playbook", checks)
                if allowed:
                    chosen = cand["playbook"]
                    break
            decisions += 1
            if chosen in (None, "NOOP"):
                audit.append({"type": "decision", "t": t, "chosen": "NOOP"})
                continue
            ctx = PlaybookContext(model=champion, chunk=chunk,
                                  base_x=self.bench.base_x, base_y=self.bench.base_y,
                                  config=self.cfg, n_classes=self.bench.n_classes,
                                  history=[], diagnosis=diag, budget=self.budget,
                                  rng=np.random.default_rng(t), registry=registry)
            try:
                res = library.run(chosen, ctx)
            except InfeasiblePlaybook:
                audit.append({"type": "infeasible", "t": t, "playbook": chosen})
                continue
            metrics = dict(res.metrics or {})
            candidate = {"playbook": chosen, "metrics": metrics,
                         "cost": dict(res.cost.__dict__)}
            incumbent = {"probe_macro_f1": metrics.get("baseline_probe_macro_f1",
                                                       float("nan")),
                         "class_recall_min": metrics.get("baseline_class_recall_min",
                                                         float("nan"))}
            gate_res = gate.evaluate(candidate, incumbent,
                                     {"t": t, "phase": chunk.phase,
                                      "budget": {"labels_total": self.budget.labels_total,
                                                 "labels_used": self.budget.labels_used},
                                      "high_risk": chosen in
                                      self.cfg["safety"]["high_risk_playbooks"]})
            audit.append({"type": "gate", "t": t, "playbook": chosen,
                          "passed": bool(gate_res.passed)})
            self.budget.charge(res.cost, phase=chunk.phase)
            planner.update_strategy(diag.kind, chosen,
                                    reward=1.0 if gate_res.passed else -0.01,
                                    cost=res.cost)
            if gate_res.passed and res.model is not None:
                meta = registry.add(res.model, {
                    "created_t": t, "parent": champ_meta.version,
                    "kind": chosen.lower(), "playbook": chosen,
                    "metrics": metrics, "cost": res.cost.to_dict()})
                registry.promote(meta.version)
                monitor.on_promote(res.model)
                watchdog.reset()
                deploys += 1
                audit.append({"type": "deploy", "t": t, "version": meta.version,
                              "playbook": chosen})
        # ---- 不变量 ----
        self.assertGreater(decisions, 0, "mini 流上至少应触发一次诊断")
        self.assertLessEqual(self.budget.labels_used, self.budget.labels_total,
                             "标签预算不得超支")
        self.assertGreaterEqual(self.bench.n_chunks, 8)
        self.assertTrue(audit.verify(), "审计哈希链必须校验通过")
        self.assertGreaterEqual(len(registry.history()), 1)
        policy_path = self.root / "run-loop" / "strategy_policy.json"
        self.assertTrue(policy_path.exists(), "策略表必须持久化")
        policy = json.loads(policy_path.read_text(encoding="utf-8"))
        self.assertGreaterEqual(policy["n_updates"], 1)
        self.assertTrue(policy["table"])
        # 门禁拒绝的候选不应把 champion 改坏：champion 始终可加载
        meta, model = registry.champion()
        self.assertIsNotNone(model)
        self.assertEqual(meta.status, "champion")
        self.assertGreaterEqual(deploys, 0)

    # ------------------------------------------------------------------
    def test_budget_exceeded_is_rejected(self):
        # 直接用一个几乎为空的预算：planner 不应给出可行候选，library 拒绝执行
        cfg = small_cfg()
        budget = BudgetState(labels_total=0, compute_total=0.0, tokens_total=0,
                             updates_total=1)
        oracle = SyntheticOracle(self.bench.visible_y, budget)
        library = PlaybookLibrary(cfg, oracle, cfg.get("cost_model", {}))
        planner = Planner(cfg, budget, library, self.root / "tiny_policy.json")
        chunk = self.bench.chunks[1]
        d = Diagnosis(t=0, phase=0, kind="condition_switch", severity=0.9,
                      confidence=0.9,
                      recommended=[{"playbook": "FULL_FINETUNE_REPLAY",
                                    "prior": 1.0, "why": "test"},
                                   {"playbook": "NOOP", "prior": 0.1, "why": ""}])
        cands = planner.propose(d, chunk, {})
        self.assertEqual([c["playbook"] for c in cands], ["NOOP"])
        with self.assertRaises(InfeasiblePlaybook):
            library.run("FULL_FINETUNE_REPLAY", PlaybookContext(
                model=self.base_model, chunk=chunk, base_x=self.bench.base_x,
                base_y=self.bench.base_y, config=cfg, n_classes=self.bench.n_classes,
                history=[], diagnosis=d, budget=budget,
                rng=np.random.default_rng(0)))

    # ------------------------------------------------------------------
    def test_rollback_and_watchdog(self):
        registry, monitor, diagnoser, library, planner, gate, audit, safety, watchdog = \
            self._components("rb")
        champ = registry.champion()
        # 造一个候选版本并回滚
        v1 = registry.add(self.base_model.clone(), {"parent": champ[0].version,
                                                    "playbook": "RECALIBRATE_AFFINE",
                                                    "kind": "recalibrated"})
        registry.promote(v1.version)
        meta = registry.rollback(champ[0].version, reason="集成测试回滚")
        self.assertEqual(meta.version, champ[0].version)
        self.assertEqual(registry.champion()[0].version, champ[0].version)
        # watchdog：持续下降后必须给出 rollback 决策
        actions = []
        for t in range(4):
            actions.append(watchdog.observe(t, 0, 0.95, champ[0].version)["action"])
        for t in range(4, 9):
            actions.append(watchdog.observe(t, 0, 0.30, champ[0].version)["action"])
        self.assertIn("rollback", actions)
        self.assertTrue(audit.verify())

    # ------------------------------------------------------------------
    def test_strategy_layer_self_improvement(self):
        """策略表必须记录 (diagnosis_kind, playbook) 的后验并跨实例持久化。"""
        registry, monitor, diagnoser, library, planner, gate, audit, safety, watchdog = \
            self._components("strat")
        p1 = planner.propose(type("D", (), {"kind": "covariate_affine",
                                            "severity": 0.6, "confidence": 0.9,
                                            "recommended": [
                                                {"playbook": "RECALIBRATE_AFFINE",
                                                 "prior": 0.7, "why": ""},
                                                {"playbook": "NOOP", "prior": 0.3,
                                                 "why": ""}]})(),
                             self.bench.chunks[1], {})
        before = {c["playbook"]: c["prior"] for c in p1}
        planner.update_strategy("covariate_affine", "RECALIBRATE_AFFINE",
                                reward=0.3, cost=Cost(labels=16))
        st = planner.stats("covariate_affine", "RECALIBRATE_AFFINE")
        self.assertEqual(st["n"], 1)
        self.assertEqual(st["successes"], 1)
        self.assertGreater(st["ema_reward"], 0)
        path = self.root / "run-strat" / "strategy_policy.json"
        self.assertTrue(path.exists())
        planner2 = Planner(self.cfg, self.budget, library, path)
        st2 = planner2.stats("covariate_affine", "RECALIBRATE_AFFINE")
        self.assertEqual(st2["n"], 1)
        self.assertGreaterEqual(st2["ema_reward"], 0)
        after = {c["playbook"]: c["prior"]
                 for c in planner2.propose(type("D", (), {
                     "kind": "covariate_affine", "severity": 0.6, "confidence": 0.9,
                     "recommended": [
                         {"playbook": "RECALIBRATE_AFFINE", "prior": 0.7, "why": ""},
                         {"playbook": "NOOP", "prior": 0.3, "why": ""}]})(),
                     self.bench.chunks[1], {})}
        self.assertIn("RECALIBRATE_AFFINE", after)
        self.assertGreaterEqual(after.get("RECALIBRATE_AFFINE", 0) + 1e-9,
                                before.get("RECALIBRATE_AFFINE", 0))

    # ------------------------------------------------------------------
    def test_audit_tamper_detected(self):
        audit = AuditLog(self.root / "tamper_audit.jsonl")
        audit.append({"type": "decision", "t": 0})
        audit.append({"type": "deploy", "version": 1})
        self.assertTrue(audit.verify())
        path = Path(audit.path)
        lines = path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[1])
        rec["record"]["version"] = 99
        lines[1] = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(audit.verify())


if __name__ == "__main__":
    unittest.main(verbosity=2)
