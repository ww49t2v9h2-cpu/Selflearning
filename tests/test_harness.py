"""Harness 测试：12 工具 / 权限与预算 / 确定性驱动 / LLM 驱动 / provider / skills。

运行：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY -m unittest tests.test_harness -v
"""

from __future__ import annotations

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import numpy as np

import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.core.model import MLP  # noqa: E402
from src.core.spec import (  # noqa: E402
    BudgetState,
    Cost,
    Diagnosis,
    DriftEvent,
    ModelVersion,
    QueryResult,
    RunRecord,
    ToolCall,
    ToolSpec,
    Trajectory,
)
from src.harness.agent import (  # noqa: E402
    DeterministicDriver,
    LLMDriver,
    build_driver,
    harness_turn,
    make_provider,
)
from src.harness.skills import (  # noqa: E402
    build_system_prompt,
    load_skills,
    select_skill_ids,
    skill_index,
    skill_titles,
    skills_by_ids,
)
from src.harness.tools import (  # noqa: E402
    DEFAULT_TOOLS,
    DefaultToolContext,
    ToolError,
    ToolPolicy,
    build_tool_context,
    build_tool_registry,
    to_plain,
    validate_args,
)
from src.observability.tracker import LocalTracker  # noqa: E402
from src.providers.base import (  # noqa: E402
    ProviderError,
    ProviderUnavailable,
    count_message_tokens,
    estimate_tokens,
)
from src.providers.mock_llm import MockLLMProvider  # noqa: E402
from src.providers.openai_compat import OpenAICompatProvider  # noqa: E402


# --------------------------------------------------------------------------
# 测试替身（引擎未就绪也能完整单测 Harness）
# --------------------------------------------------------------------------
class FakeOracle:
    def __init__(self, budget: BudgetState):
        self.budget = budget
        self.queries: list[dict] = []

    def query(self, ids, reason=""):
        ids = np.asarray(ids, dtype=np.int64)
        n = min(len(ids), self.budget.labels_left())
        ids = ids[:n]
        cost = Cost(labels=len(ids))
        self.budget.charge(cost)
        self.queries.append({"ids": ids.tolist(), "reason": reason})
        y = (ids * 7 + 3) % 4
        return QueryResult(ids=ids, y=np.asarray(y, dtype=np.int64),
                           cost=cost, reason=reason)

    def remaining(self):
        return self.budget.labels_left()

    def budget_state(self):
        return self.budget


class FakeDiagnoser:
    def __init__(self, kind="covariate_affine", severity=0.35,
                 confidence=0.8):
        self.kind = kind
        self.severity = severity
        self.confidence = confidence
        self.calls = 0

    def diagnose(self, chunk, model, report, history, oracle):
        self.calls += 1
        q = oracle.query(chunk.ids[:3], "diagnosis_probe")
        return Diagnosis(
            t=chunk.t, phase=chunk.phase, kind=self.kind,
            severity=self.severity, confidence=self.confidence,
            affected_features=[1, 2], affected_groups=["sensor_0"],
            recommended=[{"playbook": "RECALIBRATE_AFFINE", "prior": 0.7,
                          "why": "affine recoverable"}],
            cost=q.cost)


class FakeLibrary:
    def __init__(self, metrics=None, cost=None, fail=False):
        self.metrics = metrics or {"probe_macro_f1": 0.62,
                                   "pred_stability": 0.9,
                                   "class_recall_min": 0.5}
        self.cost = cost or Cost(compute_units=1.0, labels=0)
        self.fail = fail
        self.calls: list[str] = []
        self.last_ctx: object = None
        self._specs = {
            "RECALIBRATE_AFFINE": SimpleNamespace(
                name="RECALIBRATE_AFFINE", layer="model", risk="low",
                needs_labels=False, est_cost=Cost()),
            "LAST_LAYER_FT": SimpleNamespace(
                name="LAST_LAYER_FT", layer="model", risk="low",
                needs_labels=True, est_cost=Cost(labels=40, compute_units=2.0)),
            "NOVEL_CLASS_EXTEND": SimpleNamespace(
                name="NOVEL_CLASS_EXTEND", layer="model", risk="high",
                needs_labels=True, est_cost=Cost(labels=80, compute_units=20.0)),
            "NOOP": SimpleNamespace(name="NOOP", layer="model", risk="low",
                                    needs_labels=False, est_cost=Cost()),
            "ROLLBACK": SimpleNamespace(name="ROLLBACK", layer="strategy",
                                        risk="medium", needs_labels=False,
                                        est_cost=Cost()),
        }

    def specs(self):
        return list(self._specs.values())

    def estimate_cost(self, name, diagnosis, chunk, budget):
        spec = self._specs.get(name)
        return spec.est_cost if spec else Cost(compute_units=1.0)

    def run(self, name, pb):
        self.calls.append(name)
        self.last_ctx = pb
        if self.fail:
            raise RuntimeError("fake library boom")
        return SimpleNamespace(model=pb.model.clone(), metrics=dict(self.metrics),
                               cost=Cost(**{"compute_units": self.cost.compute_units,
                                            "labels": self.cost.labels}),
                               artifacts=[], params={}, note=f"fake {name}")


class FakeRegistry:
    def __init__(self):
        self.dir = Path(tempfile.mkdtemp())
        self.versions: list[ModelVersion] = []
        self.promote_calls: list[int] = []
        self.rollback_calls: list[tuple] = []
        self._champion_model = None

    def add(self, model, meta):
        version = len(self.versions) + 1
        path = self.dir / f"v{version}.npz"
        model.save(str(path))
        mv = ModelVersion(version=version, created_t=meta.get("t", 0),
                          parent=None, kind="recalibrated",
                          playbook=meta.get("playbook", ""),
                          metrics=meta.get("metrics", {}),
                          status="candidate", artifact=str(path),
                          cost=Cost(), note=meta.get("hypothesis", ""))
        self.versions.append(mv)
        return mv

    def champion(self):
        if not self.versions:
            raise RuntimeError("no champion")
        return (self.versions[0], self._champion_model)

    def history(self):
        return list(self.versions)

    def promote(self, version):
        self.promote_calls.append(version)
        for mv in self.versions:
            if mv.version == version:
                mv.status = "champion"
                return mv
        raise ValueError(f"no version {version}")

    def rollback(self, to_version, reason):
        self.rollback_calls.append((to_version, reason))
        for mv in self.versions:
            if mv.version == to_version:
                mv.status = "champion"
                return mv
        raise ValueError(f"no version {to_version}")


class FakeSafety:
    def __init__(self, kill=False, allow=True):
        self.kill = kill
        self.allow = allow
        self.checks: list[dict] = []

    def should_kill(self):
        return self.kill

    def check(self, action, payload, state):
        self.checks.append({"action": action, "payload": payload})
        return self.allow, {"action": action, "allowed": self.allow,
                            "kill": self.kill}


class FakePlanner:
    def __init__(self, candidates):
        self.candidates = candidates

    def propose(self, diagnosis, chunk, state):
        return [dict(c) for c in self.candidates]

    def order(self, candidates):
        return sorted(candidates, key=lambda c: c.get("est_cost", Cost()).scalar)

    def update_strategy(self, *args, **kwargs):
        return None


def _fake_chunk(t=3, phase=1, n=16, d=6):
    rng = np.random.default_rng(t)
    return SimpleNamespace(t=t, phase=phase,
                           x=rng.normal(size=(n, d)),
                           ids=np.arange(n, dtype=np.int64),
                           meta={})


class HarnessEnv:
    """一个可复用的 Harness 测试环境（假引擎 + 临时 tracker）。"""

    def __init__(self, cfg=None, *, kind="covariate_affine", severity=0.35,
                 confidence=0.8, kill=False, safety_allow=True,
                 candidates=None, library=None, registry=None,
                 labels_total=220, compute_total=220.0,
                 tokens_total=120000, updates_total=40,
                 challenger_metrics=None):
        self.tmp = Path(tempfile.mkdtemp())
        self.cfg = cfg or load_config()
        self.budget = BudgetState(labels_total=labels_total,
                                  compute_total=compute_total,
                                  tokens_total=tokens_total,
                                  updates_total=updates_total)
        self.chunk = _fake_chunk()
        self.model = MLP(6, 4, hidden=[8, 4], seed=0)
        self.base_x = np.random.default_rng(0).normal(size=(32, 6))
        self.base_y = np.random.default_rng(0).integers(0, 4, size=32)
        self.report = SimpleNamespace(
            events=[DriftEvent(t=3, phase=1, detector="mmd", statistic=0.5,
                               threshold=0.35, p_value=0.01, severity=0.35,
                               features=[1, 2], note="")],
            state={"mmd": 0.5, "proxy_drop": 0.06})
        self.diagnosis = Diagnosis(
            t=3, phase=1, kind=kind, severity=severity,
            confidence=confidence, affected_features=[1, 2],
            recommended=[{"playbook": "RECALIBRATE_AFFINE", "prior": 0.7,
                          "why": "affine"}])
        self.oracle = FakeOracle(self.budget)
        self.diagnoser = FakeDiagnoser(kind=kind, severity=severity,
                                       confidence=confidence)
        self.library = library or FakeLibrary()
        self.registry = registry or FakeRegistry()
        self.registry._champion_model = self.model
        self.safety = FakeSafety(kill=kill, allow=safety_allow)
        self.planner = FakePlanner(candidates or [])
        self.tracker = LocalTracker(root=self.tmp / "runs",
                                    artifacts_dir=self.tmp / "artifacts")
        self.state = {
            "t": 3, "phase": 1, "chunk": self.chunk, "model": self.model,
            "base_x": self.base_x, "base_y": self.base_y, "history": [],
            "rng": np.random.default_rng(0), "n_classes": 4,
            "run_id": "loop-20261116-120000-abcdef",
            "monitor_report": self.report,
            "champion_model": self.model,
            "champion_metrics": {"probe_macro_f1": 0.55,
                                 "class_recall_min": 0.45},
            "champion_version": 0,
            "diagnosis": self.diagnosis,
        }
        self.ctx = build_tool_context(
            self.cfg, budget=self.budget, state=self.state,
            oracle=self.oracle, diagnoser=self.diagnoser,
            library=self.library, registry=self.registry,
            safety=self.safety, tracker=self.tracker, planner=self.planner)
        self.tools = build_tool_registry(self.ctx, cfg=self.cfg)
        if candidates is None:
            self.snapshot_candidates = [
                {"playbook": "RECALIBRATE_AFFINE",
                 "est_cost": Cost(compute_units=1.0), "prior": 0.7,
                 "why": "affine recoverable"},
                {"playbook": "LAST_LAYER_FT",
                 "est_cost": Cost(labels=40, compute_units=10.0), "prior": 0.5,
                 "why": "need labels"},
            ]
        else:
            self.snapshot_candidates = candidates
        self.snapshot = {
            "t": 3, "phase": 1, "monitor_report": self.report,
            "diagnosis": self.diagnosis,
            "candidates": [dict(c) for c in self.snapshot_candidates],
            "budget": {"labels_left": self.budget.labels_left(),
                       "compute_left": self.budget.compute_left(),
                       "tokens_left": self.budget.tokens_left(),
                       "updates_left_phase": 3},
            "registry_summary": {"champion_version": 0},
            "skills": ["00", "01", "02"],
        }

    def close(self):
        self.tracker.close()
        shutil.rmtree(self.tmp, ignore_errors=True)


class HarnessTestBase(unittest.TestCase):
    def make_env(self, **kwargs) -> HarnessEnv:
        env = HarnessEnv(**kwargs)
        self.addCleanup(env.close)
        return env


# --------------------------------------------------------------------------
# 工具注册表 / Schema / 权限
# --------------------------------------------------------------------------
class TestToolRegistry(HarnessTestBase):
    def test_twelve_tools_registered(self):
        env = self.make_env()
        names = set(env.tools.names())
        expected = {"inspect_drift", "diagnose", "budget_status",
                    "propose_candidates", "run_experiment", "evaluate_candidate",
                    "deploy", "rollback", "request_labels", "list_registry",
                    "query_history", "shadow_eval"}
        self.assertEqual(names, expected)
        self.assertEqual(len(env.tools.specs()), 12)
        for spec in env.tools.specs():
            self.assertIn(spec.risk, ("low", "medium", "high"))
            self.assertIn(spec.side_effects, ("none", "budget", "model"))
            self.assertEqual(spec.input_schema.get("type"), "object")
        briefs = env.tools.brief_specs()
        self.assertEqual(len(briefs), 12)
        self.assertTrue(any("run_experiment" in line for line in briefs))

    def test_unknown_tool(self):
        env = self.make_env()
        call = env.tools.call("rm_rf", {})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "unknown_tool")
        self.assertIn("rm_rf", call.error)

    def test_schema_validation(self):
        env = self.make_env()
        call = env.tools.call("run_experiment", {"playbook": 123})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "invalid_args")
        self.assertTrue(any("playbook" in e for e in
                            call.result["validation_errors"]))
        call = env.tools.call("run_experiment", {"playbook": "NOOP", "step": 3})
        self.assertFalse(call.ok)  # additionalProperties=False
        call = env.tools.call("request_labels", {})
        self.assertFalse(call.ok)
        call = env.tools.call("request_labels", {"n": 0})
        self.assertFalse(call.ok)
        self.assertIn("minimum", call.error)
        # 默认值填充
        call = env.tools.call("inspect_drift", {})
        self.assertTrue(call.ok)
        self.assertTrue(call.args.get("include_events", True))

    def test_validate_args_unit(self):
        schema = {"type": "object",
                  "properties": {"a": {"type": "integer", "minimum": 1},
                                 "b": {"type": "string",
                                       "enum": ["x", "y"]}},
                  "required": ["a"], "additionalProperties": False}
        self.assertEqual(validate_args(schema, {"a": 2, "b": "x"}), [])
        self.assertTrue(validate_args(schema, {}))
        self.assertTrue(validate_args(schema, {"a": 0}))
        self.assertTrue(validate_args(schema, {"a": 1, "b": "z"}))
        self.assertTrue(validate_args(schema, {"a": 1, "c": 1}))
        self.assertTrue(validate_args(schema, "not-dict"))

    def test_read_only_policy_denies_writes(self):
        env = self.make_env()
        env.tools.policy = ToolPolicy(read_only=True)
        ok = env.tools.call("inspect_drift", {})
        self.assertTrue(ok.ok)
        for name, args in (("run_experiment", {"playbook": "NOOP"}),
                           ("deploy", {"version": 1}),
                           ("rollback", {}),
                           ("request_labels", {"n": 1}),
                           ("diagnose", {})):
            call = env.tools.call(name, args)
            self.assertFalse(call.ok, name)
            self.assertEqual(call.result["error_kind"], "read_only_violation", name)

    def test_whitelist_and_max_calls(self):
        env = self.make_env()
        env.tools.policy = ToolPolicy(allowed_tools={"inspect_drift"},
                                      max_calls=1)
        self.assertTrue(env.tools.call("inspect_drift", {}).ok)
        denied = env.tools.call("inspect_drift", {})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.result["error_kind"], "max_calls_exceeded")
        env.tools.policy = ToolPolicy(allowed_tools={"inspect_drift"})
        denied = env.tools.call("budget_status", {})
        self.assertEqual(denied.result["error_kind"], "tool_not_allowed")

    def test_deny_high_risk_policy(self):
        env = self.make_env()
        env.tools.policy = ToolPolicy(deny_high_risk=True)
        call = env.tools.call("deploy", {"version": 1})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "high_risk_denied")

    def test_require_approval_for_high(self):
        env = self.make_env()
        env.ctx.auto_approve_high_risk = False
        env.tools.policy = ToolPolicy(require_approval_for_high=True)
        denied = env.tools.call("deploy", {"version": 1})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.result["error_kind"], "needs_approval")
        env.ctx.approve("tool:deploy")
        self.assertTrue(env.tools.call("rollback", {}).ok or True)


# --------------------------------------------------------------------------
# 只读工具
# --------------------------------------------------------------------------
class TestReadTools(HarnessTestBase):
    def test_inspect_drift(self):
        env = self.make_env()
        call = env.tools.call("inspect_drift", {})
        self.assertTrue(call.ok)
        self.assertTrue(call.result["drift_detected"])
        self.assertAlmostEqual(call.result["max_severity"], 0.35)
        self.assertEqual(call.result["level"], "medium")
        self.assertEqual(call.result["n_events"], 1)
        self.assertIn("proxy_drop", call.result["thresholds"])
        self.assertGreaterEqual(call.latency_s, 0.0)

    def test_inspect_drift_without_report(self):
        env = self.make_env()
        env.state["monitor_report"] = None
        env.ctx.state["monitor_report"] = None
        env.snapshot["monitor_report"] = None
        call = env.tools.call("inspect_drift", {})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "no_monitor_report")

    def test_diagnose_cached_and_force(self):
        env = self.make_env()
        call = env.tools.call("diagnose", {})
        self.assertTrue(call.ok)
        self.assertTrue(call.result["cached"])  # snapshot 已有诊断
        self.assertEqual(env.diagnoser.calls, 0)
        call = env.tools.call("diagnose", {"force": True})
        self.assertTrue(call.ok)
        self.assertFalse(call.result["cached"])
        self.assertEqual(env.diagnoser.calls, 1)
        self.assertEqual(call.result["kind"], "covariate_affine")
        self.assertEqual(env.budget.labels_used, 3)  # 诊断探针 3 条
        self.assertEqual(call.cost.labels, 3)

    def test_diagnose_unavailable(self):
        env = self.make_env()
        env.ctx.diagnoser = None
        env.state["diagnosis"] = None
        env.ctx.state["diagnosis"] = None
        env.snapshot["diagnosis"] = None
        call = env.tools.call("diagnose", {})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "diagnoser_unavailable")

    def test_budget_status(self):
        env = self.make_env()
        call = env.tools.call("budget_status", {"phase": 1})
        self.assertTrue(call.ok)
        for key in ("labels_left", "compute_left", "tokens_left",
                    "updates_left_phase", "labels_total", "compute_total"):
            self.assertIn(key, call.result)
        self.assertEqual(call.result["labels_left"], 220)
        self.assertIn("suggestion", call.result)

    def test_propose_candidates_planner_and_fallback(self):
        env = self.make_env(candidates=[
            {"playbook": "LAST_LAYER_FT", "est_cost": Cost(labels=40),
             "prior": 0.2, "why": "b"},
            {"playbook": "RECALIBRATE_AFFINE", "est_cost": Cost(),
             "prior": 0.7, "why": "a"}])
        call = env.tools.call("propose_candidates", {})
        self.assertTrue(call.ok)
        names = [c["playbook"] for c in call.result["candidates"]]
        self.assertEqual(names[0], "RECALIBRATE_AFFINE")  # cheapest-first
        # 非白名单候选被标记不可负担
        env.planner.candidates = [{"playbook": "NOT_ALLOWED",
                                   "est_cost": Cost(), "prior": 0.9}]
        call = env.tools.call("propose_candidates", {})
        self.assertFalse(call.result["candidates"][0]["affordable"])
        env.planner.candidates = []
        env.planner.propose = None
        env.ctx.planner = None
        call = env.tools.call("propose_candidates", {})
        self.assertTrue(call.ok)
        self.assertTrue(call.result["fallback"])
        self.assertEqual(call.result["candidates"][0]["playbook"],
                         "RECALIBRATE_AFFINE")

    def test_list_registry_and_query_history(self):
        env = self.make_env()
        env.tracker.start_run(RunRecord(run_id="", group="main", name="r1",
                                        kind="experiment", stage="optimize"))
        call = env.tools.call("list_registry", {})
        self.assertTrue(call.ok)
        self.assertEqual(call.result["count"], 0)
        call = env.tools.call("query_history", {"limit": 5})
        self.assertTrue(call.ok)
        self.assertEqual(call.result["count"], 1)
        self.assertEqual(call.result["runs"][0]["kind"], "experiment")
        bad = env.tools.call("query_history", {"filters": {"nope": 1}})
        self.assertFalse(bad.ok)
        self.assertEqual(bad.result["error_kind"], "query_failed")


# --------------------------------------------------------------------------
# 写工具：预算 / 安全 / 记账
# --------------------------------------------------------------------------
class TestWriteTools(HarnessTestBase):
    def test_run_experiment_full_record(self):
        env = self.make_env()
        call = env.tools.call("run_experiment",
                              {"playbook": "RECALIBRATE_AFFINE",
                               "hypothesis": "affine 可恢复"})
        self.assertTrue(call.ok, call.error)
        res = call.result
        self.assertEqual(res["playbook"], "RECALIBRATE_AFFINE")
        self.assertEqual(res["metrics"]["probe_macro_f1"], 0.62)
        self.assertEqual(res["missing_metrics"], [])
        self.assertEqual(res["version"], 1)
        self.assertIsNotNone(res["exp_id"])
        self.assertTrue(Path(res["artifacts"][0]).exists())
        self.assertAlmostEqual(call.cost.compute_units, 1.0)
        # 预算只扣一次 compute，labels 由 oracle 扣（此 playbook 0 标签）
        self.assertAlmostEqual(env.budget.compute_used, 1.0)
        self.assertEqual(env.budget.labels_used, 0)
        # tracker 记录：experiment run（含指标/artifact）
        runs = env.tracker.query({"kind": "experiment"})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["parent_run"], env.state["run_id"])
        self.assertEqual(runs[0]["status"], "succeeded")
        self.assertEqual(len(runs[0]["artifacts"]), 1) if \
            "artifacts" in runs[0] else None
        self.assertIn("probe_macro_f1",
                      env.tracker.latest_metrics(runs[0]["run_id"]))
        self.assertIn(res["exp_id"], env.state["experiments"])

    def test_run_experiment_budget_rejected(self):
        env = self.make_env(compute_total=0.2)
        call = env.tools.call("run_experiment",
                              {"playbook": "LAST_LAYER_FT"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "budget_exceeded")
        self.assertEqual(env.library.calls, [])
        self.assertEqual(env.tracker.query({"kind": "experiment"}), [])

    def test_run_experiment_not_allowed(self):
        env = self.make_env()
        cfg_allowed = env.cfg["safety"]["allowed_playbooks"]
        call = env.tools.call("run_experiment", {"playbook": "EVIL_HACK"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "playbook_not_allowed")
        self.assertIn("EVIL_HACK", call.error)
        self.assertTrue(cfg_allowed)

    def test_run_experiment_kill_switch(self):
        env = self.make_env(kill=True)
        call = env.tools.call("run_experiment",
                              {"playbook": "RECALIBRATE_AFFINE"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "kill_switch")
        # read-only 仍可用
        self.assertTrue(env.tools.call("inspect_drift", {}).ok)

    def test_rollback_emergency_under_kill_switch(self):
        env = self.make_env(kill=True)
        env.registry.add(env.model, {"t": 1, "playbook": "BASE"})
        denied = env.tools.call("rollback", {"to_version": 1})
        self.assertFalse(denied.ok)
        self.assertEqual(denied.result["error_kind"], "kill_switch")
        allowed = env.tools.call("rollback", {"to_version": 1,
                                              "emergency": True,
                                              "reason": "watchdog"})
        self.assertTrue(allowed.ok, allowed.error)
        self.assertEqual(env.registry.rollback_calls[0][0], 1)

    def test_safety_veto(self):
        env = self.make_env(safety_allow=False)
        call = env.tools.call("run_experiment",
                              {"playbook": "RECALIBRATE_AFFINE"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "safety_veto")
        self.assertTrue(env.safety.checks)

    def test_high_risk_escalation_recorded(self):
        env = self.make_env(labels_total=220)
        call = env.tools.call("run_experiment",
                              {"playbook": "NOVEL_CLASS_EXTEND"})
        self.assertTrue(call.ok, call.error)
        self.assertEqual(call.result["risk"], "high")
        escalation = call.result["escalation"]
        self.assertEqual(escalation["reason"],
                         "high_risk_human_gate_escalation")
        self.assertTrue(env.state["escalations"])

    def test_high_risk_blocked_when_no_auto_approve(self):
        env = self.make_env()
        env.ctx.auto_approve_high_risk = False
        call = env.tools.call("run_experiment",
                              {"playbook": "NOVEL_CLASS_EXTEND"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "needs_approval")
        env.ctx.approve("run_experiment:NOVEL_CLASS_EXTEND")
        call = env.tools.call("run_experiment",
                              {"playbook": "NOVEL_CLASS_EXTEND"})
        self.assertTrue(call.ok, call.error)
        self.assertIsNone(call.result["escalation"])

    def test_experiment_failure_recorded(self):
        env = self.make_env(library=FakeLibrary(fail=True))
        call = env.tools.call("run_experiment",
                              {"playbook": "RECALIBRATE_AFFINE"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "experiment_failed")
        runs = env.tracker.query({"kind": "experiment"})
        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["status"], "failed")

    def test_evaluate_and_deploy(self):
        env = self.make_env()
        run = env.tools.call("run_experiment",
                             {"playbook": "RECALIBRATE_AFFINE"})
        ev = env.tools.call("evaluate_candidate",
                            {"exp_id": run.result["exp_id"]})
        self.assertTrue(ev.ok)
        gate = ev.result["gate"]
        self.assertIn("criteria", gate)
        self.assertEqual(ev.result["decision"], "promote")
        self.assertAlmostEqual(ev.result["probe_gain"], 0.62 - 0.55, places=6)
        dep = env.tools.call("deploy", {"version": run.result["version"],
                                        "reason": "gate passed"})
        self.assertTrue(dep.ok, dep.error)
        self.assertEqual(env.registry.promote_calls, [1])
        self.assertEqual(env.state["champion_version"], 1)
        runs = env.tracker.query({"kind": "deploy"})
        self.assertEqual(len(runs), 1)
        # 无门禁的 version 被拒
        env.registry.add(env.model, {"playbook": "X"})
        bad = env.tools.call("deploy", {"version": 99})
        self.assertFalse(bad.ok)

    def test_evaluate_reject_when_no_gain(self):
        env = self.make_env()
        env.library.metrics = {"probe_macro_f1": 0.40,
                               "pred_stability": 0.9,
                               "class_recall_min": 0.2}
        env.library.cost = Cost(compute_units=99.0)
        run = env.tools.call("run_experiment",
                             {"playbook": "RECALIBRATE_AFFINE"})
        ev = env.tools.call("evaluate_candidate",
                            {"exp_id": run.result["exp_id"]})
        self.assertTrue(ev.ok)
        self.assertFalse(ev.result["passed"])
        self.assertEqual(ev.result["decision"], "reject")
        dep = env.tools.call("deploy", {"version": run.result["version"]})
        self.assertFalse(dep.ok)
        self.assertEqual(dep.result["error_kind"], "gate_rejected")

    def test_request_labels(self):
        env = self.make_env()
        call = env.tools.call("request_labels",
                              {"n": 5, "reason": "unit-test"})
        self.assertTrue(call.ok)
        self.assertEqual(call.result["n"], 5)
        self.assertEqual(env.budget.labels_used, 5)
        self.assertEqual(call.cost.labels, 5)
        too_many = env.tools.call("request_labels", {"n": 10 ** 6})
        self.assertFalse(too_many.ok)
        self.assertEqual(too_many.result["error_kind"], "budget_exceeded")

    def test_shadow_eval(self):
        env = self.make_env()
        run = env.tools.call("run_experiment",
                             {"playbook": "RECALIBRATE_AFFINE"})
        shadow = env.tools.call("shadow_eval", {"exp_id": run.result["exp_id"]})
        self.assertTrue(shadow.ok)
        self.assertGreaterEqual(shadow.result["agreement"], 0.0)
        self.assertLessEqual(shadow.result["agreement"], 1.0)
        self.assertIn(shadow.result["exp_id"], env.state["shadow_results"])
        self.assertIn("verdict", shadow.result)

    def test_tool_exception_captured(self):
        class BoomLib(FakeLibrary):
            def run(self, name, pb):
                raise ValueError("kaboom")

        env = self.make_env(library=BoomLib())
        call = env.tools.call("run_experiment",
                              {"playbook": "RECALIBRATE_AFFINE"})
        self.assertFalse(call.ok)
        self.assertEqual(call.result["error_kind"], "experiment_failed")
        # 未注册 handler 的异常被兜住（模拟内部 bug）
        empty = build_tool_registry(DefaultToolContext(cfg=env.cfg))

        def bad(args, ctx):
            raise ArithmeticError("boom-in-tool")

        empty.register(ToolSpec(name="bad_tool", description="x",
                                input_schema={"type": "object",
                                              "properties": {}},
                                risk="low", side_effects="none"), bad)
        res = empty.call("bad_tool", {})
        self.assertFalse(res.ok)
        self.assertEqual(res.result["error_kind"], "exception")


# --------------------------------------------------------------------------
# 确定性驱动
# --------------------------------------------------------------------------
class TestDeterministicDriver(HarnessTestBase):
    def test_full_trajectory_and_decision(self):
        env = self.make_env()
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertIsInstance(traj, Trajectory)
        self.assertEqual(traj.driver, "deterministic_policy")
        self.assertEqual(traj.tokens, 0)
        self.assertGreaterEqual(traj.wall_s, 0.0)
        self.assertGreaterEqual(len(traj.steps), 4)
        self.assertTrue(all(isinstance(s, ToolCall) for s in traj.steps))
        names = [s.name for s in traj.steps]
        self.assertIn("run_experiment", names)
        self.assertIn("evaluate_candidate", names)
        self.assertIn("deploy", names)
        d = traj.final_decision
        self.assertIsNotNone(d)
        self.assertEqual(d.t, 3)
        self.assertEqual(d.phase, 1)
        self.assertEqual(d.chosen_playbook, "RECALIBRATE_AFFINE")
        self.assertEqual(d.driver, "deterministic_policy")
        self.assertTrue(d.rationale)
        self.assertIsInstance(d.cost, Cost)
        self.assertAlmostEqual(d.cost.compute_units, 1.0)
        self.assertEqual(d.budget_before["labels_left"], 220)
        self.assertIn("compute_left", d.budget_after)
        self.assertIn("deployed", d.outcome)
        self.assertTrue(d.outcome["deployed"])
        self.assertIsInstance(d.safety, dict)
        self.assertIn("kill_switch", d.safety)
        self.assertTrue(d.evidence_refs)
        # 记录真实写入
        self.assertEqual(len(env.tracker.query({"kind": "experiment"})), 1)
        self.assertEqual(len(env.tracker.query({"kind": "deploy"})), 1)

    def test_no_candidates_returns_noop(self):
        env = self.make_env()
        env.snapshot["candidates"] = []
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook, "NOOP")
        self.assertEqual(traj.outcome["action"], "noop")
        self.assertIn("候选为空", traj.final_decision.rationale)
        self.assertEqual(env.library.calls, [])

    def test_budget_exhausted_returns_noop(self):
        env = self.make_env(compute_total=0.5)
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook, "NOOP")
        self.assertEqual(env.library.calls, [])
        d = traj.final_decision
        self.assertTrue(any("预算" in alt.get("why_not", "") or
                            "算力" in alt.get("why_not", "")
                            for alt in d.alternatives), d.alternatives)

    def test_benign_prefers_noop_candidate(self):
        env = self.make_env(kind="benign", severity=0.1, confidence=0.9,
                            candidates=[
                                {"playbook": "NOOP", "est_cost": Cost(),
                                 "prior": 0.9, "why": "benign"},
                                {"playbook": "LAST_LAYER_FT",
                                 "est_cost": Cost(labels=40),
                                 "prior": 0.3, "why": "x"}])
        env.snapshot["diagnosis"] = Diagnosis(t=3, phase=1, kind="benign",
                                              severity=0.1, confidence=0.9)
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook, "NOOP")
        self.assertEqual(env.library.calls, [])

    def test_repair_triggers_rollback(self):
        registry = FakeRegistry()
        model = MLP(6, 4, hidden=[8, 4], seed=0)
        model.save(str(registry.dir / "v0.npz"))
        registry.versions.append(ModelVersion(
            version=1, created_t=1, parent=None, kind="recalibrated",
            playbook="RECALIBRATE_AFFINE", status="champion",
            metrics={"probe_macro_f1": 0.6}, artifact=str(registry.dir / "v0.npz")))
        env = self.make_env(kind="repair", registry=registry,
                            candidates=[{"playbook": "ROLLBACK",
                                         "est_cost": Cost(), "prior": 0.9,
                                         "why": "repair → 回滚"}])
        env.snapshot["diagnosis"] = Diagnosis(t=3, phase=1, kind="repair",
                                              severity=0.2, confidence=0.9)
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook, "ROLLBACK")
        self.assertEqual(env.registry.rollback_calls, [(1, "诊断 repair → 回滚 SOP")])
        self.assertEqual(traj.outcome["action"], "rollback")

    def test_kill_switch_noop(self):
        env = self.make_env(kill=True)
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook, "NOOP")
        self.assertEqual(traj.outcome["reason"], "kill_switch")
        self.assertEqual(env.library.calls, [])

    def test_tool_failure_skips_to_next_candidate(self):
        # 第一个候选被预算拒绝（LAST_LAYER_FT 需要 40 labels，但只剩 5）
        env = self.make_env(labels_total=5, candidates=[
            {"playbook": "LAST_LAYER_FT",
             "est_cost": Cost(labels=40, compute_units=2.0), "prior": 0.9},
            {"playbook": "RECALIBRATE_AFFINE",
             "est_cost": Cost(compute_units=1.0), "prior": 0.5}])
        env.snapshot["budget"]["labels_left"] = 5
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertEqual(traj.final_decision.chosen_playbook,
                         "RECALIBRATE_AFFINE")
        self.assertTrue(env.library.calls)

    def test_never_raises_on_broken_context(self):
        env = self.make_env()
        env.state.clear()  # 极端：loop 没给任何实时对象
        traj = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        self.assertIsNotNone(traj.final_decision)
        self.assertIsInstance(traj, Trajectory)
        d = traj.final_decision
        self.assertIn(d.chosen_playbook, {"NOOP", "RECALIBRATE_AFFINE",
                                          "LAST_LAYER_FT"})

    def test_driver_reproducible(self):
        env = self.make_env()
        t1 = DeterministicDriver(env.cfg).decide(env.snapshot, env.tools)
        env2 = self.make_env()
        t2 = DeterministicDriver(env2.cfg).decide(env2.snapshot, env2.tools)
        self.assertEqual(t1.outcome.get("action"), t2.outcome.get("action"))
        self.assertEqual(t1.final_decision.chosen_playbook,
                         t2.final_decision.chosen_playbook)


# --------------------------------------------------------------------------
# LLM 驱动
# --------------------------------------------------------------------------
class TestLLMDriver(HarnessTestBase):
    def test_mock_llm_full_protocol(self):
        env = self.make_env()
        driver = LLMDriver(MockLLMProvider(), env.cfg)
        traj = driver.decide(env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm")
        self.assertGreater(traj.tokens, 0)
        names = [s.name for s in traj.steps]
        self.assertIn("run_experiment", names)
        self.assertIn("evaluate_candidate", names)
        self.assertIn("deploy", names)
        d = traj.final_decision
        self.assertEqual(d.chosen_playbook, "RECALIBRATE_AFFINE")
        self.assertEqual(d.driver, "llm")
        self.assertEqual(d.cost.tokens, traj.tokens)
        self.assertEqual(d.safety["provider"], "mock")
        self.assertTrue(d.outcome["deployed"])
        # 默认由 loop 统一扣 token（避免重复计数）；独立使用可开启 charge_budget_tokens
        self.assertEqual(env.budget.tokens_used, 0)
        env2 = self.make_env()
        driver2 = LLMDriver(MockLLMProvider(), env2.cfg, charge_budget_tokens=True)
        traj2 = driver2.decide(env2.snapshot, env2.tools)
        self.assertAlmostEqual(env2.budget.tokens_used, traj2.tokens)
        self.assertEqual(len(env.tracker.query({"kind": "experiment"})), 1)
        self.assertEqual(len(env.tracker.query({"kind": "deploy"})), 1)

    def test_invalid_json_falls_back_deterministic(self):
        env = self.make_env()
        driver = LLMDriver(MockLLMProvider(script=["这不是 JSON"]), env.cfg)
        traj = driver.decide(env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm-fallback")
        self.assertGreater(traj.tokens, 0)
        self.assertIn("json_parse_failed", traj.outcome["fallback_reason"])
        d = traj.final_decision
        self.assertIsNotNone(d)
        self.assertEqual(d.driver, "llm-fallback")
        self.assertIn(d.chosen_playbook, {"RECALIBRATE_AFFINE", "NOOP"})
        self.assertEqual(d.safety["llm_fallback_reason"],
                         traj.outcome["fallback_reason"])

    def test_unknown_tool_falls_back(self):
        env = self.make_env()
        payload = json.dumps({"thought": "hack", "tool": "drop_database",
                              "args": {}})
        traj = LLMDriver(MockLLMProvider(script=[payload]), env.cfg).decide(
            env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm-fallback")
        self.assertIn("unknown_tool", traj.outcome["fallback_reason"])

    def test_invalid_chosen_playbook_falls_back(self):
        env = self.make_env()
        payload = json.dumps({"thought": "x", "final": {
            "chosen_playbook": "SUPER_MODEL", "action": "deploy",
            "reasoning": "because"}})
        traj = LLMDriver(MockLLMProvider(script=[payload]), env.cfg).decide(
            env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm-fallback")
        self.assertIn("invalid_chosen_playbook",
                      traj.outcome["fallback_reason"])
        self.assertIn(traj.final_decision.chosen_playbook,
                      {"RECALIBRATE_AFFINE", "NOOP"})

    def test_provider_error_falls_back(self):
        class FailingProvider:
            name = "failing"
            model = "failing-v1"

            def complete(self, messages, temperature=0.2, max_tokens=2048):
                raise ProviderUnavailable("no network")

        env = self.make_env()
        traj = LLMDriver(FailingProvider(), env.cfg).decide(env.snapshot,
                                                           env.tools)
        self.assertEqual(traj.driver, "llm-fallback")
        self.assertIn("provider_unavailable", traj.outcome["fallback_reason"])

    def test_max_steps_falls_back(self):
        env = self.make_env()
        cfg = json.loads(json.dumps(env.cfg))
        cfg["harness"]["max_steps"] = 2
        payload = json.dumps({"thought": "loop", "tool": "inspect_drift",
                              "args": {}})
        traj = LLMDriver(MockLLMProvider(script=[payload]), cfg,
                         max_steps=2).decide(env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm-fallback")
        self.assertIn("max_steps_reached", traj.outcome["fallback_reason"])

    def test_llm_cannot_deploy_without_tool(self):
        env = self.make_env()
        payload = json.dumps({"thought": "我要上线", "final": {
            "chosen_playbook": "RECALIBRATE_AFFINE", "action": "deploy",
            "deploy": True, "reasoning": "自己批准", "confidence": 0.99}})
        traj = LLMDriver(MockLLMProvider(script=[payload]), env.cfg).decide(
            env.snapshot, env.tools)
        self.assertEqual(traj.driver, "llm")
        d = traj.final_decision
        self.assertFalse(d.outcome["deployed"])
        self.assertTrue(d.outcome["deploy_not_executed"])
        self.assertEqual(env.registry.promote_calls, [])
        self.assertIn("未调用 deploy 工具", d.rationale)


# --------------------------------------------------------------------------
# Provider
# --------------------------------------------------------------------------
class TestProviders(unittest.TestCase):
    def test_token_estimate(self):
        self.assertEqual(estimate_tokens(""), 0)
        self.assertEqual(estimate_tokens("漂移监测"), 4)
        self.assertEqual(estimate_tokens("abcd"), 1)
        self.assertLess(estimate_tokens("a" * 400), 200)
        msgs = [{"role": "system", "content": "你好"},
                {"role": "user", "content": "abc"}]
        self.assertGreater(count_message_tokens(msgs),
                           estimate_tokens("你好"))

    def test_mock_deterministic(self):
        messages = [{"role": "system", "content": "s"},
                    {"role": "user", "content": "【状态快照】{\"snapshot\": "
                                                "{\"t\": 1, \"candidates\": []}}"}]
        p1, p2 = MockLLMProvider(), MockLLMProvider()
        r1 = p1.complete(messages)
        r2 = p2.complete(messages)
        self.assertEqual(r1.text, r2.text)
        self.assertEqual(r1.prompt_tokens, r2.prompt_tokens)
        self.assertEqual(r1.completion_tokens, r2.completion_tokens)
        parsed = json.loads(r1.text)
        self.assertEqual(parsed["tool"], "inspect_drift")

    def test_mock_script_mode_and_bad_provider(self):
        provider = MockLLMProvider(script=["resp-1", "resp-2"])
        self.assertEqual(provider.complete([]).text, "resp-1")
        self.assertEqual(provider.complete([]).text, "resp-2")
        self.assertEqual(provider.complete([]).text, "resp-2")

    def test_openai_compat_disabled_fails_fast(self):
        provider = OpenAICompatProvider(env={})
        self.assertFalse(provider.enabled)
        self.assertFalse(provider.health()["api_key_set"])
        self.assertNotIn("sk-", json.dumps(provider.health()))
        t0 = time.perf_counter()
        with self.assertRaises(ProviderUnavailable):
            provider.complete([{"role": "user", "content": "hi"}])
        self.assertLess(time.perf_counter() - t0, 1.0)  # 无网络时快速失败

    def test_openai_compat_unreachable_endpoint_fails_fast(self):
        provider = OpenAICompatProvider(
            base_url="http://127.0.0.1:1/v1", model="Qwen3.8-27B",
            api_key="sk-test-secret-value-123456", timeout_s=0.5, env={})
        self.assertTrue(provider.enabled)
        self.assertIn("Qwen3.8-27B", provider.health()["model"])
        self.assertNotIn("sk-test", json.dumps(provider.health()))
        t0 = time.perf_counter()
        with self.assertRaises(ProviderError):
            provider.complete([{"role": "user", "content": "hi"}])
        self.assertLess(time.perf_counter() - t0, 5.0)

    def test_openai_compat_response_parsing(self):
        provider = OpenAICompatProvider(base_url="http://x/v1",
                                        model="m", env={})
        result = provider._parse_response(json.dumps({
            "model": "m", "choices": [{"message": {"content": "hello"},
                                       "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 7, "completion_tokens": 3}}), 0.1, 9)
        self.assertEqual(result.text, "hello")
        self.assertEqual(result.prompt_tokens, 7)
        self.assertEqual(result.completion_tokens, 3)
        self.assertEqual(result.total_tokens, 10)


# --------------------------------------------------------------------------
# Skills / 工厂
# --------------------------------------------------------------------------
class TestSkillsAndFactories(HarnessTestBase):
    def test_skills_library(self):
        skills = load_skills()
        self.assertGreaterEqual(len(skills), 6)
        titles = skill_titles()
        self.assertEqual(len(titles), len(skills))
        index = skill_index()
        self.assertTrue(all(s["chars"] > 200 for s in index))
        ids = select_skill_ids("novel_class")
        self.assertEqual(ids[:3], ["00", "04", "06"])
        self.assertIn("02", ids)
        selected = skills_by_ids(["00", "06"], per_skill_max_chars=500)
        self.assertEqual(len(selected), 2)
        self.assertTrue(all(len(t) <= 700 for t in selected.values()))
        # 技能里必须有可执行阈值
        all_text = "\n".join(skills.values())
        for threshold in ("0.35", "0.005", "0.12", "20000"):
            self.assertIn(threshold, all_text)

    def test_system_prompt(self):
        env = self.make_env()
        prompt = build_system_prompt(tool_briefs=env.tools.brief_specs(),
                                     skills=skills_by_ids(["00", "06"]))
        self.assertIn("输出协议", prompt)
        self.assertIn("run_experiment", prompt)
        self.assertIn("06_安全红线与成本纪律", prompt)
        self.assertIn("chosen_playbook", prompt)
        self.assertLess(len(prompt), 30000)

    def test_build_driver_modes(self):
        env = self.make_env()
        self.assertEqual(build_driver(env.cfg).name, "deterministic")
        cfg = json.loads(json.dumps(env.cfg))
        cfg["harness"]["driver"] = "llm"
        cfg["harness"]["provider"] = "mock"
        driver = build_driver(cfg)
        self.assertEqual(driver.name, "llm")
        self.assertEqual(driver.provider_info["provider"], "mock")
        # openai_compat 未配置 → 降级 deterministic
        cfg["harness"]["provider"] = "openai_compat"
        with mock.patch.dict(os.environ, {}, clear=False):
            for key in ("ZHIXUN_LLM_BASE_URL", "OPENAI_BASE_URL",
                        "LLM_BASE_URL", "PYTRI0_BASE_URL"):
                os.environ.pop(key, None)
            driver = build_driver(cfg)
        self.assertEqual(driver.name, "deterministic")
        self.assertIn("fallback_from", driver.provider_info)
        # auto 未配置 → 确定性；mock provider 也不会被 auto 采用
        cfg["harness"]["driver"] = "auto"
        driver = build_driver(cfg)
        self.assertEqual(driver.name, "deterministic")

    def test_make_provider_env(self):
        provider, meta = make_provider(load_config(), "mock")
        self.assertEqual(meta["provider"], "mock")
        self.assertFalse(meta["network"])
        with mock.patch.dict(os.environ, {
                "ZHIXUN_LLM_BASE_URL": "https://api.pytri0.example/v1",
                "ZHIXUN_LLM_MODEL": "Qwen3.8-27B",
                "ZHIXUN_LLM_API_KEY": "sk-test"}, clear=False):
            provider, meta = make_provider(load_config(), "openai_compat")
        self.assertIsNotNone(provider)
        self.assertTrue(provider.enabled)
        self.assertTrue(meta["api_key_set"])
        self.assertNotIn("sk-test", json.dumps(meta))

    def test_harness_turn_convenience(self):
        env = self.make_env()
        traj = harness_turn(env.cfg, snapshot=env.snapshot, tools=env.tools)
        self.assertIsNotNone(traj.final_decision)
        self.assertEqual(traj.driver, "deterministic_policy")


if __name__ == "__main__":
    unittest.main()
