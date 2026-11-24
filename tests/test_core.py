"""核心层测试：数据基准确定性、隐藏标签隔离、预算、模型、指标、评测器、CLI 审计。"""

from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.cli import verify_audit                                   # noqa: E402
from src.core.config import load_config, set_global_seed           # noqa: E402
from src.core.evaluator import HiddenEvaluator                     # noqa: E402
from src.core.metrics import (evaluate, ks_2samp, label_prior_em,  # noqa: E402
                              macro_f1, mmd_rbf, recovery_time)
from src.core.model import (MLP, ScalerState, expand_model_classes)  # noqa: E402
from src.core.spec import BudgetState, Cost                         # noqa: E402
from src.data.synth import make_benchmark, SyntheticOracle          # noqa: E402


class TestBenchmark(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.cfg = load_config(overrides={"benchmark": {"scenario": "S7_full_mix"}})

    def test_deterministic_across_calls(self):
        b1 = make_benchmark(self.cfg, seed=3)
        b2 = make_benchmark(self.cfg, seed=3)
        self.assertEqual(b1.n_chunks, b2.n_chunks)
        for c1, c2 in zip(b1.chunks, b2.chunks):
            np.testing.assert_allclose(c1.x, c2.x, rtol=0, atol=0)
            np.testing.assert_array_equal(c1.ids, c2.ids)
        np.testing.assert_array_equal(b1.visible_y, b2.visible_y)

    def test_shapes_and_isolation(self):
        b = make_benchmark(self.cfg, seed=0)
        self.assertEqual(len(b.visible_y), sum(len(c.ids) for c in b.chunks))
        self.assertEqual(len(b.hidden_by_chunk), b.n_chunks)
        # 隐藏标签与可见标签不是同一份
        hx, hy = b.hidden_by_chunk[0]
        self.assertEqual(len(hx), len(hy))
        # oracle 只能拿 visible 标签
        budget = BudgetState(labels_total=10)
        orc = SyntheticOracle(b.visible_y, budget)
        q = orc.query(b.chunks[0].ids[:3], "test")
        np.testing.assert_array_equal(q.y, b.visible_y[q.ids])
        self.assertEqual(budget.labels_used, 3)
        self.assertEqual(orc.remaining(), 7)

    def test_drift_degrades_static_baseline(self):
        b = make_benchmark(self.cfg, seed=0)
        set_global_seed(0)
        m = MLP(b.d, b.base_classes, hidden=[64, 32], seed=0)
        m.fit(b.base_x, b.base_y, epochs=30, lr=4e-3, batch=128, seed=0)
        ph0 = [i for i, c in enumerate(b.chunks) if c.phase == 0]
        f1_0 = np.mean([evaluate(b.hidden_by_chunk[i][1], m.predict(b.hidden_by_chunk[i][0]), 5)["macro_f1"]
                        for i in ph0])
        # 工况切换阶段（S7 P03）应显著退化
        ph3 = [i for i, c in enumerate(b.chunks) if c.phase == 3]
        f1_3 = np.mean([evaluate(b.hidden_by_chunk[i][1], m.predict(b.hidden_by_chunk[i][0]), 5)["macro_f1"]
                        for i in ph3])
        self.assertGreater(f1_0, 0.85)
        self.assertLess(f1_3, f1_0 - 0.25)


class TestBudgetAndOracle(unittest.TestCase):
    def test_budget_caps(self):
        b = BudgetState(labels_total=5, compute_total=1.0)
        c = Cost(labels=3, compute_units=0.5)
        self.assertTrue(b.can_afford(c))
        b.charge(c, phase=0)
        self.assertEqual(b.labels_used, 3)
        self.assertFalse(b.can_afford(Cost(labels=3)))
        self.assertFalse(b.can_afford(Cost(compute_units=1.0)))


class TestModel(unittest.TestCase):
    def test_save_load_and_predict(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(200, 8))
        y = (x[:, 0] > 0).astype(int)
        m = MLP(8, 2, hidden=[16], seed=0)
        m.fit(x, y, epochs=20, lr=5e-3, batch=32, seed=0)
        p1 = m.predict_proba(x)
        with tempfile.TemporaryDirectory() as td:
            m.save(str(Path(td) / "m.npz"))
            m2 = MLP.load(str(Path(td) / "m.npz"))
        np.testing.assert_allclose(p1, m2.predict_proba(x), atol=1e-12)

    def test_expand_classes(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(100, 6))
        y = rng.integers(0, 3, size=100)
        m = MLP(6, 3, hidden=[8], seed=0)
        m.fit(x, y, epochs=5, batch=32, seed=0)
        m2 = expand_model_classes(m, 5, seed=0)
        self.assertEqual(m2.n_classes, 5)
        proba = m2.predict_proba(x)
        self.assertEqual(proba.shape[1], 5)
        # 旧类别偏置保留
        np.testing.assert_allclose(proba[:, :3].sum(axis=1) + proba[:, 3:].sum(axis=1), 1.0, atol=1e-9)

    def test_align_predict_proba(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 6))
        y = (x[:, 0] > 0).astype(int)
        m = MLP(6, 2, hidden=[8], seed=0)
        m.fit(x, y, epochs=30, lr=5e-3, batch=32, seed=0)
        # 增益漂移后的目标域：x*2+1
        xd = x * 2.0 + 1.0
        tgt = m.estimate_target_scaler(xd)
        p_aligned = m.align_predict_proba(xd, tgt)
        acc_aligned = np.mean(np.argmax(p_aligned, axis=1) == y)
        self.assertGreater(acc_aligned, 0.9)


class TestMetrics(unittest.TestCase):
    def test_basic(self):
        y = np.array([0, 0, 1, 1, 2])
        p = np.array([0, 0, 1, 2, 2])
        self.assertAlmostEqual(macro_f1(y, p, 3), (1.0 + 2 / 3 + 2 / 3) / 3, places=5)
        self.assertAlmostEqual(evaluate(y, p, 3)["accuracy"], 0.8)

    def test_two_sample_tests(self):
        rng = np.random.default_rng(0)
        a = rng.normal(size=500)
        b = rng.normal(size=500)
        d, p = ks_2samp(a, b)
        self.assertLess(p, 0.5)
        self.assertLess(mmd_rbf(a.reshape(-1, 1), b.reshape(-1, 1)), 0.1)
        c = rng.normal(loc=1.0, size=500)
        self.assertGreater(mmd_rbf(a.reshape(-1, 1), c.reshape(-1, 1)), 0.01)

    def test_label_prior(self):
        rng = np.random.default_rng(0)
        prior_true = np.array([0.7, 0.2, 0.1])
        y = rng.choice(3, size=5000, p=prior_true)
        proba = np.eye(3)[y] * 0.9 + rng.uniform(0, 0.1, size=(5000, 3))
        proba /= proba.sum(axis=1, keepdims=True)
        prior_hat, _ = label_prior_em(proba, 3)
        np.testing.assert_allclose(np.sort(prior_hat), np.sort(prior_true), atol=0.08)

    def test_recovery_time(self):
        curve = [0.9, 0.9, 0.5, 0.6, 0.85, 0.9, 0.9]
        self.assertEqual(recovery_time(curve, 0.9, window=2, tol=0.02), 3)
        self.assertEqual(recovery_time([0.9, 0.9], 0.9), 0)


class TestEvaluator(unittest.TestCase):
    def test_hidden_eval(self):
        cfg = load_config(overrides={"benchmark": {"scenario": "S2_sensor_gain"}})
        b = make_benchmark(cfg, seed=0)
        set_global_seed(0)
        m = MLP(b.d, b.base_classes, hidden=[32], seed=0)
        m.fit(b.base_x, b.base_y, epochs=10, lr=5e-3, batch=128, seed=0)
        ev = HiddenEvaluator(b)
        scores = ev.score_stream(m)
        self.assertEqual(len(scores.scores), b.n_chunks)
        self.assertTrue(0.0 <= scores.summary()["stream_macro_f1"] <= 1.0)


class TestAuditChain(unittest.TestCase):
    def test_verify_chain(self):
        # 模拟 engine 的哈希链格式：append 记录 + prev_hash + hash
        import hashlib

        prev, lines = "", []
        for i in range(5):
            body = {"seq": i, "prev_hash": prev, "record": {"x": i}}
            canon = json.dumps(body, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
            h = hashlib.sha256((prev + canon).encode()).hexdigest()
            body["hash"] = h
            prev = h
            lines.append(json.dumps(body, ensure_ascii=False))
        with tempfile.TemporaryDirectory() as td:
            p = Path(td) / "audit.jsonl"
            p.write_text("\n".join(lines) + "\n", encoding="utf-8")
            ok, n, _ = verify_audit(p)
            self.assertTrue(ok)
            self.assertEqual(n, 5)
            # 篡改一条记录后校验失败
            bad = lines[:2] + [lines[2].replace('"x": 2', '"x": 3')] + lines[3:]
            p.write_text("\n".join(bad) + "\n", encoding="utf-8")
            ok2, _, _ = verify_audit(p)
            self.assertFalse(ok2)


if __name__ == "__main__":
    unittest.main()
