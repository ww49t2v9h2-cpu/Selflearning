"""数据与基准层测试（负责人 A：src/data/**）。

覆盖：
1. Benchmark 结构契约（形状/编号/visible-hidden 对齐）；
2. 可复现性（同 seed 两次完全一致）× 跨 seed 难度稳定性；
3. 各场景漂移退化幅度（静态基线难度表的核心断言）；
4. SyntheticOracle / VisibleLabelOracle 的预算语义与去重；
5. OfficialAdapter：合法数据包加载、缺目录/缺文件/缺列/标签泄漏等错误路径；
6. load_benchmark 数据源选择；
7. AI4I 公开数据：结构、确定性、缺数据错误路径（数据未就绪时 skip）。
"""

from __future__ import annotations

import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config                                    # noqa: E402
from src.core.metrics import evaluate                                      # noqa: E402
from src.core.model import MLP                                             # noqa: E402
from src.core.spec import BudgetState                                      # noqa: E402
from src.data.adapter import (                                              # noqa: E402
    DatasetAdapter, OfficialAdapter, VisibleLabelOracle, load_benchmark, resolve_root,
)
from src.data.synth import (                                                # noqa: E402
    Benchmark, SyntheticOracle, make_benchmark, scenario_specs,
)

SCENARIOS = ["S1_gradual_wear", "S2_sensor_gain", "S3_condition_switch",
             "S4_novel_class", "S5_repair_cycle", "S6_label_shift", "S7_full_mix"]

PY = sys.executable


# --------------------------------------------------------------------------
# 工具
# --------------------------------------------------------------------------
def _cfg(scenario: str = "S1_gradual_wear", seed: int = 7) -> dict:
    return load_config(overrides={"benchmark": {"scenario": scenario}, "seed": seed})


def _fit_base(bench: Benchmark, cfg: dict, seed: int = 7) -> MLP:
    mcfg = cfg["model"]
    model = MLP(bench.d, bench.base_classes, hidden=mcfg["hidden"], seed=seed)
    model.fit(bench.base_x, bench.base_y, epochs=mcfg["epochs"], lr=mcfg["lr"],
              batch=mcfg["batch"], seed=seed)
    return model


def _phase_static_f1(model: MLP, bench: Benchmark) -> list[float]:
    out = []
    for p in range(len(bench.phase_names)):
        vals = []
        for ci, chunk in enumerate(bench.chunks):
            if chunk.phase != p:
                continue
            hx, hy = bench.hidden_by_chunk[ci]
            vals.append(evaluate(hy, model.predict(hx), bench.n_classes)["macro_f1"])
        out.append(float(np.mean(vals)))
    return out


def _make_official_package(root: Path, drop: set[str] | None = None,
                           bad_checksum: bool = False) -> Path:
    """构造一个最小官方数据包（供 loader 测试；2 特征 3 类 2 阶段 4 窗口）。"""
    drop = drop or set()
    root.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    n_feat = 2

    def _rows(n):
        x = np.round(rng.normal(size=(n, n_feat)), 3)
        y = rng.integers(0, 3, size=n)
        return x, y

    def _write_base(name, n):
        x, y = _rows(n)
        lines = ["f0,f1,label"] + [f"{a},{b},{c}" for (a, b), c in zip(x, y)]
        (root / name).write_text("\n".join(lines) + "\n", encoding="utf-8")

    stream_rows = ["id,chunk,phase,f0,f1"]
    labels_rows = ["id,label"]
    hidden_rows = ["chunk,f0,f1,label"]
    src_id = 1000
    for chunk in range(4):
        phase = 0 if chunk < 2 else 1
        for _ in range(3):
            x = np.round(rng.normal(size=n_feat), 3)
            stream_rows.append(f"{src_id},{chunk},{phase},{x[0]},{x[1]}")
            if src_id % 2 == 0:            # 一半样本有 visible 标注
                labels_rows.append(f"{src_id},{int(rng.integers(0, 3))}")
            src_id += 1
        for _ in range(2):
            x = np.round(rng.normal(size=n_feat), 3)
            hidden_rows.append(f"{chunk},{x[0]},{x[1]},{int(rng.integers(0, 3))}")

    files = {
        "base_train.csv": None,
        "base_val.csv": None,
        "stream.csv": "\n".join(stream_rows) + "\n",
        "stream_labels.csv": "\n".join(labels_rows) + "\n",
        "hidden.csv": "\n".join(hidden_rows) + "\n",
    }
    _write_base("base_train.csv", 30)
    _write_base("base_val.csv", 12)
    for name, content in files.items():
        if content is not None and name not in drop:
            (root / name).write_text(content, encoding="utf-8")
    manifest = {
        "schema_version": "1.0",
        "name": "official_unittest_v1",
        "dataset_version": "test",
        "base_classes": 3,
        "n_classes": 3,
        "d_features": n_feat,
        "class_names": ["正常", "故障A", "故障B"],
        "phase_names": ["阶段A", "阶段B"],
        "feature_columns": ["f0", "f1"],
        "label_column": "label",
        "chunk_column": "chunk",
        "id_column": "id",
        "license": "CC BY 4.0",
    }
    if bad_checksum:
        manifest["checksums"] = {"stream.csv": "0" * 64}
    if "manifest.json" not in drop:
        (root / "manifest.json").write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return root


# --------------------------------------------------------------------------
# 1. Benchmark 结构契约
# --------------------------------------------------------------------------
class TestBenchmarkContract(unittest.TestCase):
    def setUp(self):
        self.bench = make_benchmark(_cfg("S7_full_mix"), seed=7)

    def test_scenarios_exist(self):
        for name in SCENARIOS:
            specs = scenario_specs(name)
            self.assertEqual(len(specs), 12 if name == "S7_full_mix" else 6, name)
            self.assertEqual(len({s.name for s in specs}), len(specs))
            with self.assertRaises(KeyError):
                scenario_specs("S999_unknown")

    def test_shapes_and_alignment(self):
        b = self.bench
        self.assertEqual(b.d, 48)
        self.assertEqual(b.n_classes, 5)
        self.assertEqual(b.base_classes, 4)
        self.assertEqual(b.base_x.shape, (4000, 48))
        self.assertEqual(b.val_x.shape, (1000, 48))
        self.assertTrue(np.issubdtype(b.base_y.dtype, np.integer))
        self.assertEqual(b.n_chunks, len(b.chunks))
        self.assertEqual(b.n_chunks, 60)  # 12 阶段 × 1800 样本 × 0.66 / 200 ≈ 5/阶段
        n_visible = sum(len(c.x) for c in b.chunks)
        self.assertEqual(len(b.visible_y), n_visible)
        # ids 全局连续递增、t 递增、与 visible_y 对齐
        concat_ids = np.concatenate([c.ids for c in b.chunks])
        self.assertTrue(np.array_equal(concat_ids, np.arange(n_visible)))
        self.assertEqual([c.t for c in b.chunks], list(range(b.n_chunks)))
        self.assertEqual([c.phase for c in b.chunks], b.phase_of_chunk)
        # chunks 不带标签；hidden 逐窗口非空
        for c in b.chunks:
            self.assertFalse(hasattr(c, "y"))
        self.assertEqual(len(b.hidden_by_chunk), b.n_chunks)
        for hx, hy in b.hidden_by_chunk:
            self.assertEqual(hx.shape[1], b.d)
            self.assertEqual(len(hx), len(hy))
            self.assertGreater(len(hx), 0)
        # describe 自描述
        desc = b.describe()
        self.assertEqual(desc["name"], b.name)
        self.assertEqual(len(desc["phase_names"]), len(b.phase_names))
        self.assertEqual(desc["chunks_per_phase"][0],
                         sum(1 for p in b.phase_of_chunk if p == 0))

    def test_determinism_same_seed(self):
        b1 = make_benchmark(_cfg("S3_condition_switch"), seed=11)
        b2 = make_benchmark(_cfg("S3_condition_switch"), seed=11)
        self.assertTrue(np.array_equal(b1.base_x, b2.base_x))
        self.assertTrue(np.array_equal(b1.base_y, b2.base_y))
        self.assertTrue(np.array_equal(b1.visible_y, b2.visible_y))
        for c1, c2 in zip(b1.chunks, b2.chunks):
            self.assertTrue(np.array_equal(c1.x, c2.x))
            self.assertTrue(np.array_equal(c1.ids, c2.ids))
        for (hx1, hy1), (hx2, hy2) in zip(b1.hidden_by_chunk, b2.hidden_by_chunk):
            self.assertTrue(np.array_equal(hx1, hx2))
            self.assertTrue(np.array_equal(hy1, hy2))

    def test_different_seed_different_sampling(self):
        b1 = make_benchmark(_cfg("S4_novel_class"), seed=7)
        b2 = make_benchmark(_cfg("S4_novel_class"), seed=8)
        self.assertFalse(np.array_equal(b1.base_x, b2.base_x))
        self.assertFalse(np.array_equal(b1.visible_y, b2.visible_y))

    def test_world_is_fixed_across_seeds(self):
        """世界固定：不同 seed 的 base 分布同源（均值差远小于特征尺度），难度可比。"""
        b1 = make_benchmark(_cfg("S2_sensor_gain"), seed=7)
        b2 = make_benchmark(_cfg("S2_sensor_gain"), seed=17)
        diff = np.abs(np.mean(b1.base_x, axis=0) - np.mean(b2.base_x, axis=0))
        scale = np.std(b1.base_x, axis=0) + 1e-9
        # 抽样误差级别（3σ/√n ≈ 0.05），不是"两个不同世界"
        self.assertLess(float(np.max(diff / scale)), 0.25)

    def test_meta_reproducibility_fields(self):
        b = make_benchmark(_cfg("S1_gradual_wear"), seed=9)
        self.assertEqual(b.meta["scenario"], "S1_gradual_wear")
        self.assertEqual(b.meta["seed"], 9)
        self.assertIn("world_seed", b.meta)
        self.assertIn("generator_version", b.meta)


# --------------------------------------------------------------------------
# 2. 漂移退化幅度（静态难度表的核心断言）
# --------------------------------------------------------------------------
class TestDriftDegradation(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cfg = _cfg("S1_gradual_wear", seed=7)
        cls.bench_s1 = make_benchmark(cfg, seed=7)
        cls.curves = {}
        for sc in ["S1_gradual_wear", "S3_condition_switch",
                   "S4_novel_class", "S6_label_shift"]:
            cfg = _cfg(sc, seed=7)
            bench = make_benchmark(cfg, seed=7)
            model = _fit_base(bench, cfg, seed=7)
            cls.curves[sc] = (_phase_static_f1(model, bench), bench)

    def test_s1_gradual_wear_monotone_degradation(self):
        curve, _ = self.curves["S1_gradual_wear"]
        # 渐变磨损：静态 F1 近似单调下降，深度磨损显著退化（标定目标 ~0.62）
        self.assertGreater(curve[0], 0.95)
        self.assertLess(curve[-1], 0.78)
        self.assertLess(curve[-1], curve[0] - 0.25)
        self.assertLessEqual(curve[-1], curve[1] + 0.02)   # 末阶段不比前期更高
        # 逐阶段大体不升（允许统计波动 0.03）
        diffs = np.diff(curve)
        self.assertLessEqual(int(np.sum(diffs > 0.03)), 2)

    def test_s3_condition_switch_concept_drift(self):
        curve, _ = self.curves["S3_condition_switch"]
        self.assertGreater(curve[0], 0.95)
        self.assertLess(np.mean(curve[1:5]), 0.65)   # 两个工况的平均静态 F1 明显下降
        self.assertGreater(curve[5], 0.95)           # 切回工况 A 后恢复

    def test_s4_novel_class_structural_drop(self):
        curve, bench = self.curves["S4_novel_class"]
        self.assertGreater(curve[0], 0.95)
        self.assertLess(curve[4], 0.75)              # novel 主导阶段（5 类评测）下降
        # hidden 标签确实含 novel 类别
        novel_present = any(int((hy == bench.base_classes).sum()) > 0
                            for _, hy in bench.hidden_by_chunk)
        self.assertTrue(novel_present)

    def test_s6_label_shift_is_benign_for_macro_f1(self):
        curve, _ = self.curves["S6_label_shift"]
        # 标签先验漂移不改变特征条件分布：macro-F1 应基本不变（"良性漂移"）
        self.assertGreater(min(curve), 0.94)
        self.assertLess(abs(curve[-1] - curve[0]), 0.05)


# --------------------------------------------------------------------------
# 3. Oracle 语义
# --------------------------------------------------------------------------
class TestOracles(unittest.TestCase):
    def test_synthetic_oracle_budget_and_dedup(self):
        bench = make_benchmark(_cfg("S1_gradual_wear"), seed=7)
        budget = BudgetState(labels_total=10)
        oracle = SyntheticOracle(bench.visible_y, budget)
        self.assertEqual(oracle.remaining(), 10)
        qr = oracle.query(np.array([3, 3, 5, 7]), reason="unit_test")
        self.assertEqual(len(qr.ids), 3)            # 去重
        self.assertEqual(qr.cost.labels, 3)
        self.assertTrue(np.array_equal(qr.y, bench.visible_y[[3, 5, 7]]))
        self.assertEqual(oracle.remaining(), 7)
        qr2 = oracle.query(np.array([0, 1, 2, 4, 8, 9, 10, 11]), reason="overflow")
        self.assertEqual(qr2.cost.labels, 7)        # 预算硬上限，不会超发
        self.assertEqual(oracle.remaining(), 0)
        self.assertEqual(len(oracle.queries), 2)
        self.assertEqual(oracle.budget_state(), budget)

    def test_visible_label_oracle_wrapper(self):
        bench = make_benchmark(_cfg("S2_sensor_gain"), seed=7)
        oracle = VisibleLabelOracle(bench, labels_total=5)
        qr = oracle.query(np.arange(8), reason="wrapper")
        self.assertEqual(qr.cost.labels, 5)
        self.assertEqual(oracle.bench, bench)
        self.assertTrue(np.array_equal(qr.y, bench.visible_y[:5]))


# --------------------------------------------------------------------------
# 4. OfficialAdapter
# --------------------------------------------------------------------------
class TestOfficialAdapter(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp(prefix="zhixun_official_"))
        self.addCleanup(shutil.rmtree, self.tmp, ignore_errors=True)

    def test_load_valid_package(self):
        root = _make_official_package(self.tmp / "pkg")
        adapter = OfficialAdapter()
        report = adapter.validate(root)
        self.assertTrue(report["ok"], report["problems"])
        bench = adapter.load(root, {})
        self.assertIsInstance(bench, Benchmark)
        self.assertEqual(bench.d, 2)
        self.assertEqual(bench.n_classes, 3)
        self.assertEqual(len(bench.phase_names), 2)
        self.assertEqual(bench.n_chunks, 4)
        self.assertEqual(len(bench.visible_y), sum(len(c.ids) for c in bench.chunks))
        # visible 重新编号为 0..N-1，原始 id 保存在 meta
        concat_ids = np.concatenate([c.ids for c in bench.chunks])
        self.assertTrue(np.array_equal(concat_ids, np.arange(len(concat_ids))))
        self.assertIn("source_ids", bench.chunks[0].meta)
        self.assertEqual(len(bench.hidden_by_chunk), bench.n_chunks)
        for hx, hy in bench.hidden_by_chunk:
            self.assertEqual(len(hx), len(hy))
            self.assertGreater(len(hx), 0)
        # oracle 可用
        oracle = VisibleLabelOracle(bench, labels_total=100)
        qr = oracle.query(bench.chunks[0].ids, reason="official")
        self.assertGreater(len(qr.ids), 0)

    def test_is_protocol_instance(self):
        self.assertIsInstance(OfficialAdapter(), DatasetAdapter)

    def test_missing_root(self):
        with self.assertRaises(FileNotFoundError) as cm:
            OfficialAdapter().load(self.tmp / "not_exists", {})
        self.assertIn("官方数据包目录不存在", str(cm.exception))

    def test_missing_manifest(self):
        root = _make_official_package(self.tmp / "pkg", drop={"manifest.json"})
        with self.assertRaises(FileNotFoundError) as cm:
            OfficialAdapter().load(root, {})
        self.assertIn("manifest.json", str(cm.exception))
        self.assertFalse(OfficialAdapter().validate(root)["ok"])

    def test_missing_required_file(self):
        root = _make_official_package(self.tmp / "pkg", drop={"hidden.csv"})
        with self.assertRaises(FileNotFoundError) as cm:
            OfficialAdapter().load(root, {})
        self.assertIn("hidden.csv", str(cm.exception))

    def test_missing_visible_labels(self):
        root = _make_official_package(self.tmp / "pkg", drop={"stream_labels.csv"})
        with self.assertRaises(FileNotFoundError) as cm:
            OfficialAdapter().load(root, {})
        self.assertIn("stream_labels.csv", str(cm.exception))

    def test_bad_manifest_key(self):
        root = _make_official_package(self.tmp / "pkg")
        manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
        del manifest["feature_columns"]
        (root / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")
        report = OfficialAdapter().validate(root)
        self.assertFalse(report["ok"])
        self.assertIn("feature_columns", " ".join(report["problems"]))

    def test_label_leak_in_stream_rejected(self):
        root = _make_official_package(self.tmp / "pkg")
        stream = root / "stream.csv"
        text = stream.read_text(encoding="utf-8")
        lines = text.strip().splitlines()
        lines[0] = lines[0] + ",label"
        lines = [ln + (",0" if i > 0 else "") for i, ln in enumerate(lines)]
        stream.write_text("\n".join(lines) + "\n", encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            OfficialAdapter().load(root, {})
        self.assertIn("标签列", str(cm.exception))

    def test_unknown_visible_label_id(self):
        root = _make_official_package(self.tmp / "pkg")
        (root / "stream_labels.csv").write_text("id,label\n999999,0\n", encoding="utf-8")
        with self.assertRaises(ValueError) as cm:
            OfficialAdapter().load(root, {})
        self.assertIn("999999", str(cm.exception))

    def test_checksum_mismatch(self):
        root = _make_official_package(self.tmp / "pkg", bad_checksum=True)
        report = OfficialAdapter().validate(root)
        self.assertFalse(report["ok"])
        self.assertFalse(report["checksum_ok"])
        with self.assertRaises(FileNotFoundError):
            OfficialAdapter().load(root, {})

    def test_load_benchmark_dispatch(self):
        cfg = load_config(overrides={"benchmark": {"adapter": "synth",
                                                   "scenario": "S1_gradual_wear"}})
        bench = load_benchmark(cfg, seed=7)
        self.assertEqual(bench.n_chunks, 30)
        cfg_bad = load_config(overrides={"benchmark": {"adapter": "nope"}})
        with self.assertRaises(ValueError):
            load_benchmark(cfg_bad, seed=7)
        self.assertEqual(resolve_root(cfg, "official"), ROOT / "data" / "official")


# --------------------------------------------------------------------------
# 5. AI4I 公开数据（数据未就绪时 skip）
# --------------------------------------------------------------------------
AI4I_ROOT = ROOT / "data" / "external"
AI4I_READY = (AI4I_ROOT / "ai4i2020.csv").is_file() or (AI4I_ROOT / "ai4i2020.zip").is_file()


@unittest.skipUnless(AI4I_READY, "AI4I 数据未就绪（运行 python -m src.data.public_ai4i --download）")
class TestAI4IPublicData(unittest.TestCase):
    def test_status_ready(self):
        from src.data.public_ai4i import ai4i_status, ensure_ai4i

        info = ensure_ai4i(AI4I_ROOT, verbose=False)
        self.assertTrue(info["ok"])
        self.assertTrue(Path(info["csv"]).is_file())
        self.assertEqual(Path(info["csv"]).name, "ai4i2020.csv")
        self.assertTrue(ai4i_status(AI4I_ROOT)["csv_exists"])

    def test_benchmark_structure_and_determinism(self):
        from src.data.public_ai4i import make_ai4i_benchmark

        cfg = load_config()
        b1 = make_ai4i_benchmark(cfg, seed=3)
        b2 = make_ai4i_benchmark(cfg, seed=3)
        self.assertEqual(b1.name, "ai4i_predictive_maintenance")
        self.assertEqual(b1.d, 11)
        self.assertEqual(b1.n_classes, 2)
        self.assertEqual(len(b1.phase_names), 4)
        self.assertGreater(b1.n_chunks, 20)
        self.assertEqual(len(b1.visible_y), sum(len(c.ids) for c in b1.chunks))
        self.assertEqual(len(b1.hidden_by_chunk), b1.n_chunks)
        self.assertIn("license", b1.meta)
        self.assertEqual(b1.meta["license"], "CC BY 4.0")
        self.assertTrue(np.array_equal(b1.base_x, b2.base_x))
        self.assertTrue(np.array_equal(b1.visible_y, b2.visible_y))
        # 真实数据的两类标签都应出现
        self.assertEqual(set(np.unique(b1.base_y)) | set(np.unique(b1.visible_y)), {0, 1})

    def test_missing_data_clear_error(self):
        from src.data.public_ai4i import make_ai4i_benchmark

        empty = Path(tempfile.mkdtemp(prefix="zhixun_ai4i_empty_"))
        self.addCleanup(shutil.rmtree, empty, ignore_errors=True)
        with self.assertRaises(FileNotFoundError) as cm:
            make_ai4i_benchmark({}, seed=0, root=empty)
        msg = str(cm.exception)
        self.assertIn("AI4I", msg)
        self.assertIn("download", msg)


if __name__ == "__main__":
    unittest.main()
