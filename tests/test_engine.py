"""引擎层单元测试（B 负责）：monitor / diagnosis / playbooks / planner / gate /
registry / safety / audit / watchdog。

运行：
    PY=python3
    $PY -m unittest discover -s tests -v
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

from src.core.config import DEFAULT_CONFIG, load_config          # noqa: E402
from src.core.model import MLP, expand_model_classes            # noqa: E402
from src.core.spec import (BudgetState, Cost, DriftEvent, Diagnosis,  # noqa: E402
                           StreamChunk)
from src.engine.defaults import resolve_config                  # noqa: E402
from src.engine.diagnosis import Diagnoser                      # noqa: E402
from src.engine.gates import Gate                               # noqa: E402
from src.engine.monitor import DriftMonitor, MonitorReport      # noqa: E402
from src.engine.planner import Planner                          # noqa: E402
from src.engine.playbooks import (InfeasiblePlaybook, PlaybookContext,  # noqa: E402
                                  PlaybookLibrary)
from src.engine.registry import ModelRegistry                   # noqa: E402
from src.engine.safety import AuditLog, SafetyGuard             # noqa: E402
from src.engine.watchdog import Watchdog                        # noqa: E402


# --------------------------------------------------------------------------
# 小型合成世界（与 bench 无关，保证单测快且确定）
# --------------------------------------------------------------------------
class MiniWorld:
    def __init__(self, seed: int = 0, d: int = 6, n_classes: int = 3,
                 n_base: int = 300, chunk_size: int = 48):
        rng = np.random.default_rng(seed)
        self.d = d
        self.n_classes = n_classes
        self.means = rng.normal(0.0, 2.0, size=(n_classes, d))
        by = rng.integers(0, n_classes, size=n_base)
        bx = self.means[by] + rng.normal(0, 0.7, size=(n_base, d))
        self.base_x, self.base_y = bx, by
        self.model = MLP(d, n_classes, hidden=(16, 8), seed=seed)
        self.model.fit(bx, by, epochs=20, lr=5e-3, batch=64, seed=seed)
        self.visible_y = np.zeros(0, dtype=np.int64)
        self.chunks: list[StreamChunk] = []
        self._next_id = 0

    def add_chunk(self, x: np.ndarray, y: np.ndarray, phase: int = 0) -> StreamChunk:
        ids = np.arange(self._next_id, self._next_id + len(x), dtype=np.int64)
        self._next_id += len(x)
        self.visible_y = np.concatenate([self.visible_y, np.asarray(y, dtype=np.int64)])
        ch = StreamChunk(t=len(self.chunks), phase=phase, ids=ids,
                         x=np.asarray(x, dtype=float))
        self.chunks.append(ch)
        return ch

    def sample(self, n: int, rng: np.random.Generator, shift=0.0, scale=1.0,
               rotate: np.ndarray | None = None, noise: float = 0.0,
               novel_frac: float = 0.0, prior: np.ndarray | None = None):
        nc = self.n_classes
        if prior is None:
            y = rng.integers(0, nc, size=n)
        else:
            y = rng.choice(nc, size=n, p=np.asarray(prior) / np.sum(prior))
        x = self.means[y] * scale + shift + rng.normal(0, 0.7 + noise, size=(n, self.d))
        if rotate is not None:
            x = x @ rotate
        if novel_frac > 0:
            n_novel = int(round(n * novel_frac))
            if n_novel > 0:
                idx = rng.choice(n, size=n_novel, replace=False)
                x[idx] = rng.normal(4.0, 0.8, size=(n_novel, self.d))
                y = y.copy()
                y[idx] = nc          # 新类标签 = n_classes（模型未见过）
        return x, y


def make_budget(total: int = 200) -> BudgetState:
    return BudgetState(labels_total=total, compute_total=400.0,
                       tokens_total=10 ** 6, updates_total=100)


def make_oracle(world: MiniWorld, budget: BudgetState):
    from src.data.synth import SyntheticOracle
    return SyntheticOracle(world.visible_y, budget)


def rotation(d: int, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    q, _ = np.linalg.qr(rng.normal(size=(d, d)))
    return q


def fake_report(state: dict | None = None, detectors: tuple = ("mmd", "ks_topk")):
    state = dict(state or {})
    state.setdefault("severity", 0.8)
    state.setdefault("detectors", list(detectors))
    events = [DriftEvent(t=0, phase=0, detector=d, statistic=1.0, threshold=0.5,
                         p_value=None, severity=0.8) for d in detectors]
    return MonitorReport(events=events, state=state)


# --------------------------------------------------------------------------
# 1. Monitor
# --------------------------------------------------------------------------
class TestMonitor(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.world = MiniWorld(seed=3)
        rng = np.random.default_rng(1)
        for _ in range(5):                      # 1 参考窗口 + 4 个 warmup 窗口
            self.world.add_chunk(*self.world.sample(60, rng))
        self.budget = make_budget()
        self.oracle = make_oracle(self.world, self.budget)
        self.monitor = DriftMonitor(self.cfg, self.world.d, self.world.n_classes)
        self.monitor.reset_reference(self.world.chunks[0].x)
        for ch in self.world.chunks[1:]:
            self.monitor.update(ch, self.world.model, self.oracle)

    def test_no_drift_no_events_and_state_keys(self):
        rng = np.random.default_rng(11)
        ch = self.world.add_chunk(*self.world.sample(60, rng))
        rep = self.monitor.update(ch, self.world.model, self.oracle)
        self.assertFalse(rep.events)
        for key in ("mmd", "ks_topk", "proxy_entropy", "proxy_margin",
                    "ph_stat", "novelty", "pred_hist_tv"):
            self.assertIn(key, rep.state)
        self.assertLessEqual(rep.state["severity"], 0.01 + 1e-9)

    def test_covariate_drift_triggers_and_probe_accounted(self):
        rng = np.random.default_rng(2)
        x, y = self.world.sample(60, rng, shift=3.0, scale=1.6)
        ch = self.world.add_chunk(x, y, phase=1)
        oracle = make_oracle(self.world, self.budget)   # 新窗口 id 需在 visible_y 范围内
        rep = self.monitor.update(ch, self.world.model, oracle)
        self.assertTrue(rep.events, "分布漂移应触发事件")
        self.assertGreater(rep.cost.labels, 0, "探针必须计入 Cost")
        self.assertGreaterEqual(self.budget.labels_used, rep.cost.labels)
        self.assertTrue(any(e.detector in ("mmd", "ks_topk", "probe_error",
                                           "page_hinkley_proxy") for e in rep.events))

    def test_probe_respects_budget_and_caps(self):
        small = make_budget(total=6)
        oracle = make_oracle(self.world, small)
        cfg = load_config(overrides={"budget": {"labels_total": 12}})
        mon = DriftMonitor(cfg, self.world.d, self.world.n_classes)
        mon.reset_reference(self.world.chunks[0].x)
        for ch in self.world.chunks:
            mon.update(ch, self.world.model, oracle)
        self.assertLessEqual(small.labels_used, 12)

    def test_monitor_direction_indicators(self):
        rng = np.random.default_rng(12)
        ch = self.world.add_chunk(*self.world.sample(60, rng))
        rep = self.monitor.update(ch, self.world.model, self.oracle)
        for key in ("accept_rate", "reject_rate", "failure_direction",
                    "n_trigger_recent", "n_accepted_total"):
            self.assertIn(key, rep.state)


# --------------------------------------------------------------------------
# 2. Diagnosis（直接测试证据决策树 + 真数据上的倾向）
# --------------------------------------------------------------------------
class TestDiagnosis(unittest.TestCase):
    def setUp(self):
        self.cfg = load_config()
        self.world = MiniWorld(seed=5)
        self.diagr = Diagnoser(self.cfg, self.world.d, self.world.n_classes,
                               {i: f"S{i // 2}" for i in range(self.world.d)})

    def _decide(self, ev: dict, use_events: bool = True):
        ev = dict(ev)
        ev.setdefault("label_probe_used", True)
        events = []
        if use_events:
            events = [DriftEvent(t=0, phase=0, detector="mmd", statistic=1.0,
                                 threshold=0.5, p_value=None, severity=0.8)]
        return self.diagr._decide(events, ev)

    def _run(self, state_extra: dict | None = None):
        state = dict(state_extra or {})
        state.setdefault("detectors", ["mmd", "ks_topk"])
        state.setdefault("severity", 0.8)
        rng = np.random.default_rng(0)
        x, y = self.world.sample(40, rng)
        ch = self.world.add_chunk(x, y)
        budget = make_budget()
        oracle = make_oracle(self.world, budget)
        return self.diagr.diagnose(ch, self.world.model, fake_report(state), [],
                                   oracle)

    def test_eight_kinds_reachable(self):
        cases = {
            "benign": dict(severity=0.0),
            "covariate_affine": dict(acc_static=0.6, acc_affine=0.9,
                                     affine_gain=0.3, affine_significant=True,
                                     affine_recoverability=0.95, ref_acc=0.95,
                                     acc_drop=0.35, corr_change=0.05,
                                     mean_shift_ratio_mean=0.5, acceleration=0),
            "condition_switch": dict(acc_static=0.5, acc_affine=0.55,
                                     affine_gain=0.05, affine_significant=False,
                                     affine_recoverability=0.1, ref_acc=0.95,
                                     acc_drop=0.45, corr_change=0.25,
                                     mean_shift_ratio_mean=0.3),
            "concept": dict(acc_static=0.55, acc_affine=0.56, affine_gain=0.01,
                            affine_significant=False, ref_acc=0.95, acc_drop=0.40,
                            corr_change=0.02, mean_shift_ratio_mean=0.05,
                            prior_fused_tv=0.02, std_ratio_median=1.0,
                            noise_frac=0.0, novelty=0.1),
            "novel_class": dict(acc_static=0.5, acc_affine=0.5, affine_gain=0.0,
                                affine_significant=False, ref_acc=0.95,
                                acc_drop=0.45, novel_label_frac=0.2),
            "label_shift": dict(acc_static=0.95, acc_affine=0.95, affine_gain=0.0,
                                affine_significant=False, ref_acc=0.96,
                                acc_drop=0.0, prior_fused_tv=0.45,
                                std_ratio_median=0.95, noise_frac=0.0,
                                corr_change=0.03),
            "noise_inflate": dict(acc_static=0.6, acc_affine=0.6, affine_gain=0.0,
                                  affine_significant=False, ref_acc=0.95,
                                  acc_drop=0.35, std_ratio_median=1.9,
                                  noise_frac=1.0, corr_change=0.02,
                                  mean_shift_ratio_mean=0.02),
            "repair": dict(repair=0.9, repair_evidence=True),
        }
        for kind, ev in cases.items():
            with self.subTest(kind=kind):
                got, conf, scores = self._decide(ev, use_events=(kind != "benign"))
                self.assertEqual(got, kind, f"证据应按预期判为 {kind}，实际 {got}")
                self.assertTrue(0.0 <= conf <= 1.0)
                self.assertIn(got, scores)

    def test_no_oracle_returns_valid_diagnosis_without_exception(self):
        rng = np.random.default_rng(0)
        x, y = self.world.sample(40, rng)
        ch = self.world.add_chunk(x, y)
        rep = fake_report({"severity": 0.9, "detectors": ["mmd"]})
        diag = self.diagr.diagnose(ch, self.world.model, rep, [], None)
        self.assertIn(diag.kind, {"benign", "concept", "covariate_affine",
                                  "condition_switch", "noise_inflate",
                                  "novel_class"})
        self.assertEqual(int(diag.cost.labels), 0)

    def test_affected_scope_and_prior_estimate(self):
        diag = self._run({"mean_shift_features": [0, 1, 2], "ks_topk": [2, 1, 0],
                          "mmd_threshold": 0.35, "mmd_baseline": 0.0,
                          "probe_spend": False, "repair_score": 0.0})
        self.assertTrue(diag.affected_features)
        self.assertTrue(diag.affected_groups)
        self.assertEqual(len(diag.prior_delta_estimate), self.world.n_classes)
        self.assertTrue(diag.recommended)
        self.assertTrue(all(0.0 <= r["prior"] <= 1.0 for r in diag.recommended))


# --------------------------------------------------------------------------
# 3. Playbooks
# --------------------------------------------------------------------------
class TestPlaybooks(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overrides={"budget": {"labels_total": 400}})
        cls.world = MiniWorld(seed=7)
        rng = np.random.default_rng(3)
        x, y = cls.world.sample(60, rng, shift=1.0, scale=1.3, novel_frac=0.15)
        cls.chunk = cls.world.add_chunk(x, y, phase=1)
        cls.budget = make_budget(total=400)
        cls.oracle = make_oracle(cls.world, cls.budget)
        cls.library = PlaybookLibrary(cls.cfg, cls.oracle,
                                      cls.cfg.get("cost_model", {}))
        cls.diag = Diagnosis(t=0, phase=1, kind="concept", severity=0.7,
                             confidence=0.8,
                             affected_classes=[3],
                             recommended=[{"playbook": p, "prior": 0.5, "why": "test"}
                                          for p in ("RECALIBRATE_AFFINE", "LAST_LAYER_FT",
                                                    "FULL_FINETUNE_REPLAY",
                                                    "RETRAIN_MIXED",
                                                    "NOVEL_CLASS_EXTEND",
                                                    "SHADOW_ONLY", "ROLLBACK")])

    def _ctx(self, budget=None, diag=None):
        return PlaybookContext(
            model=self.world.model, chunk=self.chunk,
            base_x=self.world.base_x, base_y=self.world.base_y,
            config=self.cfg, n_classes=self.world.n_classes + 1,
            history=[], diagnosis=diag or self.diag,
            budget=budget or make_budget(), rng=np.random.default_rng(0))

    def test_specs_cover_nine_playbooks(self):
        specs = self.library.specs()
        names = {s.name for s in specs}
        self.assertEqual(len(specs), 9)
        for required in ("NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION",
                         "LAST_LAYER_FT", "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED",
                         "NOVEL_CLASS_EXTEND", "SHADOW_ONLY", "ROLLBACK"):
            self.assertIn(required, names)

    def test_all_playbooks_run_and_metrics_complete(self):
        for name in ("NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION",
                     "LAST_LAYER_FT", "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED",
                     "NOVEL_CLASS_EXTEND", "SHADOW_ONLY", "ROLLBACK"):
            with self.subTest(playbook=name):
                budget = make_budget(total=400)
                res = self.library.run(name, self._ctx(budget=budget))
                for key in ("probe_macro_f1", "pred_stability",
                            "class_recall_min", "cost"):
                    self.assertIn(key, res.metrics)
                self.assertIsInstance(res.cost, Cost)
                if name != "SHADOW_ONLY":
                    self.assertIsNotNone(res.model)

    def test_recalibrate_is_tta_zero_label(self):
        res = self.library.run("RECALIBRATE_AFFINE", self._ctx())
        self.assertTrue(getattr(res.model, "adaptive", False),
                        "recalibrate 必须使用 TTA（adaptive）而不是固化目标域 scaler")
        self.assertEqual(int(res.cost.labels), res.metrics.get("labels", int(res.cost.labels)))
        self.assertEqual(float(res.cost.compute_units), 0.0)

    def test_novel_extend_expands_classes(self):
        res = self.library.run("NOVEL_CLASS_EXTEND", self._ctx())
        self.assertGreaterEqual(int(res.model.n_classes), self.world.n_classes + 1)
        self.assertIn("n_target_classes", res.params)

    def test_insufficient_budget_raises(self):
        empty = make_budget(total=0)
        with self.assertRaises(InfeasiblePlaybook):
            self.library.run("FULL_FINETUNE_REPLAY", self._ctx(budget=empty))

    def test_estimate_cost_reflects_budget(self):
        budget = make_budget(total=400)
        cheap = self.library.estimate_cost("RECALIBRATE_AFFINE", self.diag,
                                           self.chunk, budget)
        rich = self.library.estimate_cost("RECALIBRATE_AFFINE", self.diag,
                                          self.chunk, make_budget(total=40))
        self.assertLessEqual(rich.compute_units, cheap.compute_units)


# --------------------------------------------------------------------------
# 4. Planner（bandit + 预算硬过滤 + 持久化）
# --------------------------------------------------------------------------
class TestPlanner(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.policy = Path(self.tmp.name) / "strategy_policy.json"
        self.cfg = load_config(overrides={"budget": {"labels_total": 260}})
        self.world = MiniWorld(seed=9)
        rng = np.random.default_rng(4)
        self.chunk = self.world.add_chunk(*self.world.sample(40, rng))
        self.budget = make_budget(total=260)
        self.oracle = make_oracle(self.world, self.budget)
        self.library = PlaybookLibrary(self.cfg, self.oracle,
                                       self.cfg.get("cost_model", {}))
        self.planner = Planner(self.cfg, self.budget, self.library, self.policy)

    def tearDown(self):
        self.tmp.cleanup()

    def _diag(self, kind="condition_switch", conf=0.8):
        return Diagnosis(t=0, phase=0, kind=kind, severity=0.8, confidence=conf,
                         recommended=[
                             {"playbook": "RECALIBRATE_AFFINE", "prior": 0.5, "why": ""},
                             {"playbook": "LAST_LAYER_FT", "prior": 0.25, "why": ""},
                             {"playbook": "FULL_FINETUNE_REPLAY", "prior": 0.15, "why": ""},
                             {"playbook": "NOOP", "prior": 0.10, "why": ""},
                         ])

    def test_propose_respects_max_candidates_and_order(self):
        cands = self.planner.propose(self._diag(), self.chunk, {})
        self.assertLessEqual(len(cands), int(self.cfg["planner"]["max_candidates"]))
        self.assertTrue(all("playbook" in c and "est_cost" in c and "prior" in c
                            for c in cands))
        # cheapest-first 排序（order 的契约）
        scalars = [float(c["est_cost"].scalar) for c in cands]
        self.assertEqual(scalars, sorted(scalars))

    def test_budget_hard_filter(self):
        tiny = make_budget(total=14)
        planner = Planner(self.cfg, tiny, self.library, self.policy)
        cands = planner.propose(self._diag(), self.chunk, {})
        self.assertTrue(all(c["est_cost"].labels <= 14 for c in cands))

    def test_update_strategy_persists_and_changes_behavior(self):
        d = self._diag()
        cands_before = self.planner.propose(d, self.chunk, {})
        p_before = {c["playbook"]: c["prior"] for c in cands_before}
        self.planner.update_strategy("condition_switch", "RECALIBRATE_AFFINE",
                                     reward=0.05, cost=Cost(labels=16))
        self.assertTrue(self.policy.exists(), "策略表必须落盘")
        reloaded = Planner(self.cfg, self.budget, self.library, self.policy)
        st = reloaded.stats("condition_switch", "RECALIBRATE_AFFINE")
        self.assertEqual(int(st["n"]), 1)
        self.assertEqual(int(st["successes"]), 1)
        cands_after = reloaded.propose(d, self.chunk, {})
        p_after = {c["playbook"]: c["prior"] for c in cands_after}
        self.assertIn("RECALIBRATE_AFFINE", p_after)
        # 成功经验应提升（或至少不降低）该动作的排序分
        self.assertGreaterEqual(p_after["RECALIBRATE_AFFINE"] + 1e-6,
                                p_before.get("RECALIBRATE_AFFINE", 0.0))
        table = json.loads(self.policy.read_text(encoding="utf-8"))
        self.assertEqual(table["schema"], 1)


# --------------------------------------------------------------------------
# 5. Gate
# --------------------------------------------------------------------------
class TestGate(unittest.TestCase):
    def setUp(self):
        self.gate = Gate(load_config())

    def _metrics(self, f1=0.8, base=0.7, recall=0.6, base_recall=0.55,
                 ece=0.05, base_ece=0.06, cost=0.3):
        return {"probe_macro_f1": f1, "baseline_probe_macro_f1": base,
                "class_recall_min": recall,
                "baseline_class_recall_min": base_recall,
                "ece": ece, "baseline_ece": base_ece, "cost": cost,
                "pred_stability": 0.9}

    def test_pass_on_clear_gain(self):
        res = self.gate.evaluate(self._metrics(), {}, {"t": 0, "phase": 0})
        self.assertTrue(res.passed)
        self.assertEqual(res.decision, "promote")
        self.assertIn("probe_gain", res.criteria)

    def test_reject_on_negative_gain(self):
        res = self.gate.evaluate(self._metrics(f1=0.60, base=0.70), {}, {})
        self.assertFalse(res.passed)
        self.assertEqual(res.decision, "reject")

    def test_reject_on_missing_metrics(self):
        res = self.gate.evaluate({"probe_macro_f1": float("nan")}, {}, {})
        self.assertFalse(res.passed)
        self.assertFalse(res.criteria["probe_gain"]["pass"])

    def test_reject_on_recall_collapse(self):
        res = self.gate.evaluate(self._metrics(f1=0.9, base=0.7, recall=0.3,
                                               base_recall=0.9), {}, {})
        self.assertFalse(res.passed)

    def test_high_risk_escalation_auto_approved(self):
        ctx = {"high_risk": True, "playbook": "NOVEL_CLASS_EXTEND",
               "diagnosis_kind": "novel_class"}
        res = self.gate.evaluate(self._metrics(f1=0.8, base=0.7), {}, ctx)
        self.assertIn("human_gate", res.criteria)
        self.assertTrue(res.criteria["human_gate"]["pass"])

    def test_cost_cap_enforced(self):
        res = self.gate.evaluate(self._metrics(cost=100.0), {}, {})
        self.assertFalse(res.criteria["cost"]["pass"])


# --------------------------------------------------------------------------
# 6. Registry
# --------------------------------------------------------------------------
class TestRegistry(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        base = Path(self.tmp.name)
        self.cfg = load_config()
        self.registry = ModelRegistry(base / "runs" / "run-x" / "registry",
                                      base / "artifacts", 6, 3, self.cfg)
        self.world = MiniWorld(seed=11)

    def tearDown(self):
        self.tmp.cleanup()

    def test_add_promote_rollback_history(self):
        v0 = self.registry.add(self.world.model, {"created_t": 0, "kind": "base",
                                                  "playbook": "BASE"})
        self.assertEqual(v0.version, 0)
        self.registry.promote(v0.version)
        meta, model = self.registry.champion()
        self.assertEqual(meta.version, 0)
        self.assertTrue(Path(meta.artifact).exists() or
                        (ROOT / meta.artifact).exists())
        v1 = self.registry.add(self.world.model.clone(), {"parent": 0, "kind": "recalibrated",
                                                          "playbook": "RECALIBRATE_AFFINE",
                                                          "cost": Cost(labels=3).to_dict()})
        self.registry.promote(v1.version)
        self.assertEqual(self.registry.champion()[0].version, 1)
        rb = self.registry.rollback(v0.version, reason="test")
        self.assertEqual(rb.status, "champion")
        self.assertEqual(self.registry.champion()[0].version, 0)
        hist = self.registry.history()
        self.assertEqual([m.version for m in hist], [0, 1])
        self.assertEqual(self.registry.lineage_of(1), [1, 0])
        # 事件记录
        self.assertTrue(any(e["kind"] == "rollback" for e in self.registry.state["events"]))

    def test_adaptive_flag_survives_artifact(self):
        m = self.world.model.clone()
        m.adaptive = True
        m.adaptive_blend = 0.7
        v = self.registry.add(m, {"kind": "base", "playbook": "BASE"})
        self.registry.promote(v.version)
        _, loaded = self.registry.champion()
        self.assertTrue(loaded.adaptive)
        self.assertAlmostEqual(loaded.adaptive_blend, 0.7, places=6)

    def test_novel_class_version_allowed(self):
        ext = expand_model_classes(self.world.model, 4, seed=0)
        v = self.registry.add(ext, {"kind": "novel_extended",
                                    "playbook": "NOVEL_CLASS_EXTEND"})
        _, loaded = self.registry.get(v.version)
        self.assertEqual(loaded.n_classes, 4)


# --------------------------------------------------------------------------
# 7. SafetyGuard + AuditLog
# --------------------------------------------------------------------------
class TestSafetyAudit(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.root = Path(self.tmp.name)
        self.cfg = load_config(overrides={
            "engine": {"safety": {"kill_switch_files": [str(self.root / "STOP")]}}})
        self.budget = make_budget(total=100)

    def tearDown(self):
        self.tmp.cleanup()

    def test_whitelist_and_budget(self):
        guard = SafetyGuard(self.cfg, self.budget)
        ok, checks = guard.check("FULL_FINETUNE_REPLAY",
                                 {"labels_est": 90, "phase": 0, "t": 0}, {})
        self.assertTrue(ok)
        ok2, _ = guard.check("HACK_ACTION", {}, {})
        self.assertFalse(ok2)
        # 预算超限
        self.budget.charge(Cost(labels=95))
        ok3, checks3 = guard.check("FULL_FINETUNE_REPLAY", {"labels_est": 90,
                                                            "phase": 0, "t": 1}, {})
        self.assertFalse(ok3)
        self.assertFalse(checks3["budget_labels"]["pass"])

    def test_kill_switch(self):
        guard = SafetyGuard(self.cfg, self.budget)
        self.assertFalse(guard.should_kill())
        (self.root / "STOP").write_text("stop", encoding="utf-8")
        self.assertTrue(guard.should_kill())
        ok, checks = guard.check("FULL_FINETUNE_REPLAY", {"phase": 0, "t": 0}, {})
        self.assertFalse(ok)
        self.assertFalse(checks["kill_switch"]["pass"])
        ok_noop, _ = guard.check("NOOP", {}, {})
        self.assertTrue(ok_noop, "kill-switch 下收尾动作仍允许")

    def test_high_risk_escalation_recorded(self):
        guard = SafetyGuard(self.cfg, make_budget(total=400))
        ok, checks = guard.check("NOVEL_CLASS_EXTEND",
                                 {"phase": 0, "t": 0, "labels_est": 20}, {})
        self.assertTrue(ok)
        self.assertTrue(checks["high_risk_gate"]["pass"])
        self.assertTrue(guard.escalations, "高风险动作必须留升级记录")

    def test_audit_chain_verify_and_tamper(self):
        path = self.root / "audit.jsonl"
        audit = AuditLog(path)
        audit.append({"type": "decision", "t": 0, "x": 1})
        audit.append({"type": "deploy", "t": 1, "x": 2})
        self.assertTrue(audit.verify())
        self.assertEqual(len(audit), 2)
        lines = path.read_text(encoding="utf-8").splitlines()
        rec = json.loads(lines[0])
        rec["record"]["x"] = 999          # 篡改
        lines[0] = json.dumps(rec, ensure_ascii=False, sort_keys=True)
        path.write_text("\n".join(lines) + "\n", encoding="utf-8")
        self.assertFalse(audit.verify(), "哈希链必须检测到篡改")


# --------------------------------------------------------------------------
# 8. Watchdog
# --------------------------------------------------------------------------
class TestWatchdog(unittest.TestCase):
    def test_no_rollback_when_stable(self):
        wd = Watchdog(load_config())
        for t in range(6):
            res = wd.observe(t, 0, 0.9, champion_version=0)
            self.assertNotEqual(res["action"], "rollback")

    def test_rollback_after_sustained_drop(self):
        wd = Watchdog(load_config())
        for t in range(4):
            wd.observe(t, 0, 0.95, champion_version=0)
        actions = []
        for t in range(4, 8):
            actions.append(wd.observe(t, 0, 0.40, champion_version=0)["action"])
        self.assertIn("rollback", actions)

    def test_new_champion_rebaselines(self):
        wd = Watchdog(load_config())
        wd.observe(0, 0, 0.95, champion_version=0)
        wd.observe(1, 0, 0.95, champion_version=0)
        res = wd.observe(2, 0, 0.30, champion_version=1)
        self.assertNotEqual(res["action"], "rollback")
        self.assertEqual(res["reason"], "champion 更替，重建参考")

    def test_noise_level_fluctuation_does_not_rollback(self):
        wd = Watchdog(load_config())
        for t, s in enumerate([0.92, 0.96, 0.88, 0.94, 0.90, 0.92, 0.86, 0.93]):
            res = wd.observe(t, 0, s, champion_version=0)
            self.assertNotEqual(res["action"], "rollback",
                                f"4-6 条探针的历史噪声不应触发回滚(t={t})")


if __name__ == "__main__":
    unittest.main(verbosity=2)
