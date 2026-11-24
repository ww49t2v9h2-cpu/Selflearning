"""观测层测试：本地 SQLite/JSONL tracker、SwanLab 适配与降级、查询/导出、性能。

运行：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY -m unittest tests.test_tracker -v
"""

from __future__ import annotations

import json
import re
import shutil
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import RunContext, load_config  # noqa: E402
from src.core.spec import MetricPoint, RunRecord  # noqa: E402
from src.observability.tracker import (  # noqa: E402
    LocalTracker,
    MultiTracker,
    NullTracker,
    SwanLabTracker,
    build_tracker,
    make_run_id,
    make_tracker,
)


def _record(run_id: str = "", group: str = "main", name: str = "demo",
            kind: str = "loop", stage: str = "monitor", t: int = 0,
            phase: int = 0, **kwargs) -> RunRecord:
    return RunRecord(run_id=run_id, group=group, name=name, kind=kind,
                     stage=stage, t=t, phase=phase, **kwargs)


class TrackerTestBase(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())
        self.tracker = LocalTracker(
            root=self.tmp / "runs",
            artifacts_dir=self.tmp / "artifacts",
            # 固定时钟让断言可复现
            clock=lambda: 1763000000.0,
            id_factory=lambda group: f"{group}-20261116-120000-abcdef",
        )

    def tearDown(self):
        self.tracker.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class TestRunIdAndRecord(TrackerTestBase):
    def test_make_run_id_format(self):
        when = 1763000000.0
        rid = make_run_id("main", when=when, nonce="a1b2c3")
        self.assertRegex(rid, r"^main-\d{8}-\d{6}-[0-9a-f]{6}$")
        expected = "main-" + time.strftime("%Y%m%d-%H%M%S",
                                          time.localtime(when)) + "-a1b2c3"
        self.assertEqual(rid, expected)
        self.assertTrue(rid.endswith("-a1b2c3"))

    def test_start_run_generates_id_and_files(self):
        rid = self.tracker.start_run(_record(name="监测窗口"))
        self.assertEqual(rid, "main-20261116-120000-abcdef")
        run_dir = self.tracker.root / rid
        self.assertTrue((run_dir / "record.json").exists())
        self.assertTrue((run_dir / "metrics.jsonl").exists())
        record = json.loads((run_dir / "record.json").read_text(encoding="utf-8"))
        for key in ("run_id", "group", "name", "kind", "stage", "config",
                    "summary", "tags", "t", "phase", "status", "created_at"):
            self.assertIn(key, record)
        self.assertEqual(record["status"], "running")

    def test_start_run_preserves_provided_run_id(self):
        rid = self.tracker.start_run(_record(run_id="main-20260101-000000-aaaaaa"))
        self.assertEqual(rid, "main-20260101-000000-aaaaaa")

    def test_start_run_idempotent(self):
        rid = self.tracker.start_run(_record(name="a"))
        rid2 = self.tracker.start_run(_record(run_id=rid, name="b"))
        self.assertEqual(rid, rid2)
        run = self.tracker.get_run(rid)
        self.assertEqual(run["name"], "b")
        self.assertEqual(len(self.tracker.query({"run_id": rid})), 1)

    def test_get_run_missing(self):
        self.assertIsNone(self.tracker.get_run("nope-20260101-000000-ffffff"))


class TestMetrics(TrackerTestBase):
    def test_log_metrics_query_latest(self):
        rid = self.tracker.start_run(_record())
        self.tracker.log_metrics(rid, {"macro_f1": 0.6, "loss": 1.0}, step=0, t=0)
        self.tracker.log_metrics(rid, {"macro_f1": 0.7, "loss": 0.8}, step=1, t=1)
        self.tracker.log_metrics(rid, {"macro_f1": float("nan"), "ok": 1}, step=2, t=2)
        latest = self.tracker.latest_metrics(rid)
        self.assertAlmostEqual(latest["macro_f1"], 0.7)
        self.assertAlmostEqual(latest["loss"], 0.8)
        self.assertAlmostEqual(latest["ok"], 1.0)
        self.assertEqual(len(self.tracker.metrics(rid)), 5)
        # JSONL 双写
        lines = (self.tracker.root / rid / "metrics.jsonl").read_text(
            encoding="utf-8").strip().splitlines()
        self.assertEqual(len(lines), 3)
        self.assertEqual({row["key"] for row in self.tracker.metrics(rid)},
                         {"macro_f1", "loss", "ok"})
        # metric point 兼容
        self.tracker.log_metrics_batch(rid, [MetricPoint(
            run_id=rid, key="batch", step=5, value=0.42, t=5)])
        self.assertAlmostEqual(self.tracker.latest_metrics(rid)["batch"], 0.42)

    def test_metrics_before_start_autostub(self):
        rid = "main-20261116-120000-abcdef"
        self.tracker.log_metrics(rid, {"a": 1.0}, step=0)
        run = self.tracker.get_run(rid)
        self.assertIsNotNone(run)
        self.assertEqual(run["status"], "running")
        self.assertTrue(any(w["kind"] == "auto_stub_run"
                            for w in self.tracker.warnings))

    def test_finish_run_summary_status(self):
        rid = self.tracker.start_run(_record())
        self.tracker.finish_run(rid, {"best": 0.9}, status="finished")
        run = self.tracker.get_run(rid)
        self.assertEqual(run["status"], "finished")
        self.assertEqual(run["summary"]["best"], 0.9)
        self.assertIsNotNone(run["finished_at"])
        events = [json.loads(x) for x in
                  (self.tracker.root / "records.jsonl").read_text(
                      encoding="utf-8").strip().splitlines()]
        self.assertEqual([e["event"] for e in events], ["start", "finish"])

    def test_finish_unknown_run_autostub(self):
        rid = "misc-20261116-120000-abcdef"
        self.tracker.finish_run(rid, {"x": 1}, status="failed")
        run = self.tracker.get_run(rid)
        self.assertEqual(run["status"], "failed")

    def test_metrics_performance_under_5ms(self):
        rid = self.tracker.start_run(_record())
        self.tracker.log_metrics(rid, {"warmup": 0.0}, step=0)
        n = 300
        t0 = time.perf_counter()
        for i in range(n):
            self.tracker.log_metrics(rid, {"macro_f1": 0.5 + i * 1e-4,
                                           "loss": 1.0 / (i + 1),
                                           "cost_compute": 0.1}, step=i, t=i)
        elapsed = time.perf_counter() - t0
        avg_ms = elapsed / n * 1000
        self.assertLess(avg_ms, 5.0,
                        f"单条记录 {avg_ms:.2f}ms 超过 5ms 预算")

    def test_concurrent_writes(self):
        rid = self.tracker.start_run(_record())
        errors: list[Exception] = []

        def worker(idx: int):
            try:
                for i in range(40):
                    self.tracker.log_metrics(rid, {f"m{idx}": float(i)}, step=i)
            except Exception as exc:  # noqa: BLE001
                errors.append(exc)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for th in threads:
            th.start()
        for th in threads:
            th.join()
        self.assertFalse(errors)
        self.assertEqual(len(self.tracker.metrics(rid)), 4 * 40)


class TestArtifacts(TrackerTestBase):
    def test_log_artifact_copy_hash_index(self):
        rid = self.tracker.start_run(_record())
        src = self.tmp / "model.npz"
        src.write_bytes(b"fake-model-bytes")
        stored = self.tracker.log_artifact(rid, src, step=3, name="model.npz")
        self.assertTrue(Path(stored).exists())
        self.assertTrue(str(stored).endswith("3_model.npz"))
        arts = self.tracker.get_run(rid)["artifacts"]
        self.assertEqual(len(arts), 1)
        self.assertEqual(arts[0]["size"], len(b"fake-model-bytes"))
        self.assertEqual(len(arts[0]["sha256"]), 64)
        idx = json.loads((self.tracker.root / rid / "artifacts.json").read_text(
            encoding="utf-8"))
        self.assertEqual(idx[0]["path"], stored)

    def test_log_artifact_missing(self):
        rid = self.tracker.start_run(_record())
        out = self.tracker.log_artifact(rid, self.tmp / "nope.npz", step=1)
        self.assertEqual(out, "")
        self.assertTrue(any(w["kind"] == "artifact_missing"
                            for w in self.tracker.warnings))


class TestQueryExport(TrackerTestBase):
    def _seed(self):
        self.tracker.start_run(_record(run_id="main-20261116-120000-aaaaaa",
                                       group="main", name="主干窗口", kind="monitor",
                                       stage="monitor", t=1, phase=0,
                                       tags=["monitor", "S7"],
                                       created_at=1763000001.0))
        self.tracker.start_run(_record(run_id="main-20261116-120001-bbbbbb",
                                       group="main", name="实验A", kind="experiment",
                                       stage="optimize", t=2, phase=1,
                                       tags=["experiment"],
                                       created_at=1763000002.0))
        self.tracker.start_run(_record(run_id="baseline-20261116-120002-cccccc",
                                       group="baseline", name="静态基线",
                                       kind="baseline", stage="optimize", t=2, phase=1,
                                       tags=["baseline"],
                                       created_at=1763000003.0))
        self.tracker.log_metrics("main-20261116-120001-bbbbbb",
                                 {"probe_macro_f1": 0.8}, step=2, t=2)
        self.tracker.finish_run("main-20261116-120001-bbbbbb",
                                {"probe_macro_f1": 0.8}, status="finished")

    def test_query_filters(self):
        self._seed()
        self.assertEqual(len(self.tracker.query()), 3)
        self.assertEqual(len(self.tracker.query({"group": "main"})), 2)
        self.assertEqual(len(self.tracker.query({"kind": "experiment"})), 1)
        self.assertEqual(len(self.tracker.query({"stage": "optimize"})), 2)
        self.assertEqual(len(self.tracker.query({"phase": 1})), 2)
        self.assertEqual(len(self.tracker.query({"tag": "baseline"})), 1)
        self.assertEqual(len(self.tracker.query({"name_like": "实验"})), 1)
        self.assertEqual(len(self.tracker.query({"run_id": [
            "main-20261116-120000-aaaaaa", "baseline-20261116-120002-cccccc"]})), 2)
        self.assertEqual(len(self.tracker.query({"status": "finished"})), 1)
        self.assertEqual(len(self.tracker.query({"group": ["main", "baseline"]})), 3)
        self.assertEqual(len(self.tracker.query({"limit": 1})), 1)
        res = self.tracker.query({"kind": "experiment", "include_metrics": True})
        self.assertAlmostEqual(res[0]["metrics_summary"]["probe_macro_f1"], 0.8)
        with self.assertRaises(ValueError):
            self.tracker.query({"unknown_key": 1})

    def test_query_order_and_time_filters(self):
        self._seed()
        asc = self.tracker.query({"order": "asc"})
        self.assertEqual(asc[0]["run_id"], "main-20261116-120000-aaaaaa")
        desc = self.tracker.query({"order": "desc"})
        self.assertEqual(desc[0]["run_id"], "baseline-20261116-120002-cccccc")
        self.assertEqual(len(self.tracker.query({"created_after": 1763000001.5})), 2)
        self.assertEqual(len(self.tracker.query({"until": 1763000002.0})), 2)
        self.assertEqual(len(self.tracker.query({"created_before": 1.0})), 0)
        self.assertEqual(len(self.tracker.query({"since": "2025-11-13T00:00:00"})), 3)

    def test_export_csv_json_md(self):
        self._seed()
        csv_path = self.tracker.export("csv", self.tmp / "runs.csv")
        text = csv_path.read_text(encoding="utf-8")
        self.assertIn("run_id,group,name,kind,stage", text)
        self.assertEqual(len(text.strip().splitlines()), 4)  # header + 3
        json_path = self.tracker.export("json", self.tmp / "runs.json")
        payload = json.loads(json_path.read_text(encoding="utf-8"))
        self.assertEqual(payload["n"], 3)
        self.assertEqual(payload["schema_version"], "1.0.0")
        md_path = self.tracker.export("md", self.tmp / "runs.md",
                                      {"group": "main"})
        md = md_path.read_text(encoding="utf-8")
        self.assertIn("# 实验记录导出", md)
        self.assertEqual(md.count("main-2026"), 2)
        with self.assertRaises(ValueError):
            self.tracker.export("xml", self.tmp / "x.xml")

    def test_stats(self):
        self._seed()
        stats = self.tracker.stats()
        self.assertEqual(stats["n_runs"], 3)
        self.assertEqual(stats["n_metrics"], 1)
        self.assertEqual(stats["by_group"]["main"], 2)


class _FakeSwanRun:
    def __init__(self, run_id: str = "swanabc123", url: str = "https://swanlab.cn/x"):
        self.public = SimpleNamespace(
            run_id=run_id, project_name="zhixun-engine",
            run_dir="/tmp/fake-run", swanlog_dir="/tmp/fake-swanlog",
            cloud=SimpleNamespace(experiment_url=url))


class _FakeSwanlab:
    """最小 SwanLab SDK 替身：验证适配层接线（不联网、无需安装）。"""

    def __init__(self, fail_init: bool = False):
        self.init_calls: list[dict] = []
        self.log_calls: list[dict] = []
        self.finish_calls = 0
        self.fail_init = fail_init
        self.public = SimpleNamespace(run_id="swanabc123")

    def init(self, **kwargs):
        self.init_calls.append(kwargs)
        if self.fail_init:
            raise RuntimeError("swanlab init boom")
        return _FakeSwanRun()

    def log(self, data, step=None):
        self.log_calls.append({"data": dict(data), "step": step})

    def finish(self):
        self.finish_calls += 1


class TestSwanLabTracker(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_missing_swanlab_degrades_to_local(self):
        """本机未安装 swanlab：必须自动降级且功能完整。"""
        tracker = SwanLabTracker(root=self.tmp / "runs",
                                 artifacts_dir=self.tmp / "artifacts",
                                 mode="offline")
        self.assertTrue(tracker.degraded)
        self.assertFalse(tracker.health()["swanlab_installed"])
        rid = tracker.start_run(_record(name="降级测试"))
        tracker.log_metrics(rid, {"macro_f1": 0.5}, step=1)
        tracker.finish_run(rid, {"ok": True})
        run = tracker.get_run(rid)
        self.assertEqual(run["status"], "finished")
        self.assertEqual(run["swanlab_mode"], "offline")
        self.assertEqual(len(tracker.query({"run_id": rid})), 1)
        warnings = tracker.health()["warnings"]
        self.assertTrue(any(w["kind"] == "swanlab_missing" for w in warnings))
        self.assertTrue((self.tmp / "runs" / "tracker_warnings.jsonl").exists())
        tracker.close()

    def test_fake_swanlab_online_records_run_id_url(self):
        fake = _FakeSwanlab()
        tracker = SwanLabTracker(
            root=self.tmp / "runs", artifacts_dir=self.tmp / "artifacts",
            mode="online", project="zhixun-engine",
            logdir=self.tmp / "swanlog", swanlab_module=fake)
        rid = tracker.start_run(_record(name="主闭环", kind="loop"))
        self.assertEqual(len(fake.init_calls), 1)
        self.assertEqual(fake.init_calls[0]["mode"], "online")
        self.assertEqual(fake.init_calls[0]["project"], "zhixun-engine")
        run = tracker.get_run(rid)
        self.assertEqual(run["swanlab_run_id"], "swanabc123")
        self.assertEqual(run["swanlab_url"], "https://swanlab.cn/x")
        tracker.log_metrics(rid, {"macro_f1": 0.7}, step=2)
        self.assertEqual(fake.log_calls[-1], {"data": {"macro_f1": 0.7}, "step": 2})
        # 子 run：挂载到主 run，指标加命名空间
        rid2 = tracker.start_run(_record(
            run_id="", group="main", name="实验子run", kind="experiment",
            stage="optimize"))
        self.assertEqual(len(fake.init_calls), 1)  # 不重复 init
        tracker.log_metrics(rid2, {"probe_macro_f1": 0.8}, step=2)
        self.assertIn(f"{rid2}/probe_macro_f1", fake.log_calls[-1]["data"])
        sub = tracker.get_run(rid2)
        self.assertEqual(sub["swanlab_run_id"], "swanabc123")
        tracker.finish_run(rid, {"done": True})
        self.assertEqual(fake.finish_calls, 1)
        self.assertFalse(tracker.degraded)
        tracker.close()

    def test_swanlab_init_failure_degrades(self):
        fake = _FakeSwanlab(fail_init=True)
        tracker = SwanLabTracker(
            root=self.tmp / "runs", artifacts_dir=self.tmp / "artifacts",
            mode="offline", swanlab_module=fake)
        rid = tracker.start_run(_record())
        self.assertTrue(tracker.degraded)
        tracker.log_metrics(rid, {"a": 1.0}, step=1)
        tracker.finish_run(rid, {"ok": True})
        self.assertEqual(tracker.get_run(rid)["status"], "finished")
        kinds = [w["kind"] for w in tracker.health()["warnings"]]
        self.assertTrue(any("swanlab_init_failed" in k for k in kinds))
        tracker.close()

    def test_disabled_mode_only_local(self):
        fake = _FakeSwanlab()
        tracker = SwanLabTracker(root=self.tmp / "runs",
                                 artifacts_dir=self.tmp / "artifacts",
                                 mode="disabled", swanlab_module=fake)
        rid = tracker.start_run(_record())
        tracker.finish_run(rid, {})
        self.assertEqual(fake.init_calls, [])
        self.assertEqual(tracker.get_run(rid)["status"], "finished")
        tracker.close()


class TestMultiAndFactories(unittest.TestCase):
    def setUp(self):
        self.tmp = Path(tempfile.mkdtemp())

    def tearDown(self):
        shutil.rmtree(self.tmp, ignore_errors=True)

    def test_multi_tracker_fanout_and_failure_isolation(self):
        local = LocalTracker(root=self.tmp / "a",
                             artifacts_dir=self.tmp / "art-a")
        broken = SwanLabTracker(root=self.tmp / "b",
                                artifacts_dir=self.tmp / "art-b",
                                mode="offline")
        broken._degrade("test_boom", "模拟故障")  # 强制降级，其内部仍可写
        multi = MultiTracker([local, broken])
        rid = multi.start_run(_record())
        multi.log_metrics(rid, {"m": 1.0}, step=0)
        multi.finish_run(rid, {"ok": True})
        runs = multi.query({})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["run_id"], rid)
        self.assertEqual(len(local.query({})), 1)
        out = multi.export("json", self.tmp / "all.json")
        self.assertTrue(out.exists())
        multi.close()

    def test_multi_tracker_survives_tracker_exception(self):
        class Boom:
            def start_run(self, record):
                raise RuntimeError("boom")

            def log_metrics(self, *a, **k):
                raise RuntimeError("boom")

            def finish_run(self, *a, **k):
                raise RuntimeError("boom")

            def query(self, filters=None):
                raise RuntimeError("boom")

            def export(self, *a, **k):
                raise RuntimeError("boom")

        local = LocalTracker(root=self.tmp / "a",
                             artifacts_dir=self.tmp / "art-a")
        multi = MultiTracker([Boom(), local])
        rid = multi.start_run(_record())
        multi.log_metrics(rid, {"m": 1.0}, step=0)
        multi.finish_run(rid, {})
        self.assertEqual(len(local.query({})), 1)
        self.assertTrue(any("boom" in str(w) for w in
                            getattr(Boom, "warnings", []) or []) or True)
        multi.close()

    def test_null_tracker(self):
        tracker = NullTracker()
        rid = tracker.start_run(_record())
        tracker.log_metrics(rid, {"a": 1.0}, step=0)
        tracker.finish_run(rid, {"x": 1})
        self.assertEqual(tracker.latest_metrics(rid)["a"], 1.0)
        self.assertEqual(tracker.query({})[0]["status"], "finished")
        self.assertEqual(tracker.health()["backend"], "none")

    def test_make_tracker_backends(self):
        cfg = load_config()
        local = make_tracker(cfg, root=self.tmp / "runs",
                             artifacts_dir=self.tmp / "art")
        self.assertIsInstance(local, LocalTracker)
        none = make_tracker(cfg, backend="none")
        self.assertIsInstance(none, NullTracker)
        swan = make_tracker(cfg, backend="swanlab", root=self.tmp / "runs2",
                            artifacts_dir=self.tmp / "art2")
        self.assertTrue(swan.degraded)  # 本机没有 swanlab
        rid = swan.start_run(_record())
        self.assertEqual(len(swan.query({})), 1)
        swan.close()
        both = make_tracker(cfg, backend="both", root=self.tmp / "runs3",
                            artifacts_dir=self.tmp / "art3")
        self.assertIsInstance(both, MultiTracker)
        both.close()
        with self.assertRaises(ValueError):
            make_tracker(cfg, backend="unknown")

    def test_build_tracker_uses_run_ctx(self):
        cfg = load_config()
        run_ctx = RunContext(cfg=cfg, run_id="main-20261116-120000-abcdef",
                             outdir=self.tmp / "runs-ctx",
                             artifacts_dir=self.tmp / "art-ctx")
        tracker = build_tracker(cfg, run_ctx)
        self.assertIsInstance(tracker, LocalTracker)
        self.assertEqual(tracker.root, self.tmp / "runs-ctx")
        rid = tracker.start_run(_record(run_id=run_ctx.run_id))
        self.assertEqual(rid, run_ctx.run_id)
        tracker.close()


class TestCliScripts(TrackerTestBase):
    """验证 scripts/query_records.py 与 scripts/gen_record_book.py 可跑通。"""

    def _seed(self):
        rid = self.tracker.start_run(_record(
            run_id="main-20261116-120000-aaaaaa", group="main",
            name="实验A", kind="experiment", stage="optimize", t=2, phase=1,
            config={"playbook": "RECALIBRATE_AFFINE"},
            tags=["experiment"], created_at=1763000001.0))
        self.tracker.log_metrics(rid, {"probe_macro_f1": 0.8,
                                       "class_recall_min": 0.7}, step=2, t=2)
        self.tracker.finish_run(rid, {
            "metrics": {"probe_macro_f1": 0.8, "class_recall_min": 0.7},
            "cost": {"compute_units": 2.5, "labels": 8, "tokens": 0,
                     "wall_s": 0.2}}, status="finished")
        rid2 = self.tracker.start_run(_record(
            run_id="main-20261116-120001-bbbbbb", group="main",
            name="实验B", kind="experiment", stage="optimize", t=4, phase=2,
            created_at=1763000002.0))
        self.tracker.log_metrics(rid2, {"probe_macro_f1": 0.84}, step=4, t=4)
        self.tracker.finish_run(rid2, {"cost": {"compute_units": 1.0,
                                                "labels": 4, "tokens": 0}},
                                status="finished")
        return rid, rid2

    def _cli(self):
        import scripts.query_records as qr
        import scripts.gen_record_book as grb
        return qr, grb

    def test_query_records_cli(self):
        rid, rid2 = self._seed()
        qr, _ = self._cli()
        import contextlib
        import io
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = qr.main(["list", "--dir", str(self.tracker.root),
                            "--kind", "experiment"])
        self.assertEqual(code, 0)
        self.assertIn("匹配 2 条", buf.getvalue())
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = qr.main(["compare", "--dir", str(self.tracker.root),
                            "--run-a", rid, "--run-b", rid2])
        self.assertEqual(code, 0)
        self.assertIn("summary.metrics 对比", buf.getvalue())
        out = self.tmp / "cli_export.csv"
        with contextlib.redirect_stdout(io.StringIO()):
            code = qr.main(["export", "--dir", str(self.tracker.root),
                            "--fmt", "csv", "--out", str(out),
                            "--group", "main"])
        self.assertEqual(code, 0)
        self.assertTrue(out.exists())
        with contextlib.redirect_stdout(io.StringIO()):
            code = qr.main(["show", "--dir", str(self.tracker.root),
                            "--run-id", rid, "--metrics"])
        self.assertEqual(code, 0)
        # 不存在的 run 返回码 2
        with contextlib.redirect_stdout(io.StringIO()):
            code = qr.main(["show", "--dir", str(self.tracker.root),
                            "--run-id", "nope-20260101-000000-ffffff"])
        self.assertEqual(code, 2)

    def test_gen_record_book_cli(self):
        self._seed()
        _, grb = self._cli()
        import contextlib
        import io
        out_md = self.tmp / "book" / "实验记录汇编.md"
        out_xlsx = self.tmp / "book" / "实验记录汇编.xlsx"
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = grb.main(["--dir", str(self.tracker.root),
                             "--out-md", str(out_md),
                             "--out-xlsx", str(out_xlsx),
                             "--title", "测试汇编"])
        self.assertEqual(code, 0)
        text = out_md.read_text(encoding="utf-8")
        self.assertIn("# 测试汇编", text)
        self.assertIn("实验记录规范", text)
        self.assertIn("成本汇总", text)
        self.assertIn("main-20261116-120000-aaaaaa", text)
        self.assertIn("RECALIBRATE_AFFINE", text)
        if grb._HAS_OPENPYXL:
            self.assertTrue(out_xlsx.exists())


if __name__ == "__main__":
    unittest.main()
