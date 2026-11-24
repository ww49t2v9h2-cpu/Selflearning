"""自进化闭环编排：漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚。

设计原则
--------
1. 闭环是**状态机**，每个窗口走固定管线；无漂移时零成本 NOOP；
2. 所有决策写入 `Decision`（含诊断、候选、预算前后、成本、护栏、证据 run_id），
   并追加到 sha256 哈希链审计日志；
3. 隐藏测试分数只用于**报告**，绝不参与任何决策（决策只用无标签信号与探针标签）；
4. 消融开关（`cfg.ablation.*`）在编排层实现，不污染引擎代码。

用法（由 src/cli.py 调用）：
    result = SelfEvolvingLoop(cfg, run_ctx).run()
"""

from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Any, Optional

import numpy as np

from .core.config import ROOT, EXPERIMENTS_DIR, RunContext, load_config, set_global_seed
from .core.evaluator import HiddenEvaluator, StreamEvaluation, compare_curves
from .core.metrics import evaluate, mean_entropy, mean_margin, prediction_histogram
from .core.model import MLP
from .core.preprocess import RobustSanitizer
from .core.spec import (
    SCHEMA_VERSION,
    BudgetState,
    Cost,
    Decision,
    Diagnosis,
    Experiment,
    GateResult,
    MetricPoint,
    ModelVersion,
    RunRecord,
    StreamChunk,
    Trajectory,
    make_cost,
)
from .data.adapter import load_benchmark
from .data.synth import Benchmark, SyntheticOracle, make_benchmark

# 引擎（B 实现，接口见 docs/03 §4）
from .engine.monitor import DriftMonitor
from .engine.diagnosis import Diagnoser
from .engine.playbooks import PlaybookContext, PlaybookLibrary
from .engine.planner import Planner
from .engine.gates import Gate
from .engine.registry import ModelRegistry
from .engine.safety import AuditLog, SafetyGuard
from .engine.watchdog import Watchdog


RUN_ID_ALPHABET = "abcdefghijklmnopqrstuvwxyz0123456789"


def new_run_id(prefix: str = "run") -> str:
    import datetime

    stamp = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    short = "".join(uuid.uuid4().hex[i] for i in range(0, 12, 2))
    return f"{prefix}-{stamp}-{short}"


def make_feature_groups(n_features: int, n_sensors: int = 8) -> dict:
    """特征分组（供诊断归因输出人类可读的受影响传感器）。"""
    stats = n_features // max(1, n_sensors)
    return {i: f"S{i // max(1, stats)}" for i in range(n_features)}


class SelfEvolvingLoop:
    def __init__(self, cfg: dict, run_ctx: Optional[RunContext] = None,
                 benchmark: Optional[Benchmark] = None):
        self.cfg = cfg
        self.run_ctx = run_ctx or RunContext(cfg=cfg, run_id=new_run_id(cfg["run"]["name"]))
        self.seed = int(cfg.get("seed", 0))
        set_global_seed(self.seed)
        self.rng = np.random.default_rng(self.seed)

        # ---- 数据 / oracle / 评测器 ----
        self.benchmark = benchmark or load_benchmark(cfg, seed=self.seed)
        # 抗脏数据清洗器：统计量只来自干净的初始训练集
        dcfg = cfg.get("data", {}) or {}
        self.sanitizer = None
        if bool(dcfg.get("sanitize", True)):
            self.sanitizer = RobustSanitizer(
                clip_quantile=float(dcfg.get("clip_quantile", 0.9995)),
                fill=str(dcfg.get("fill", "median"))).fit(self.benchmark.base_x)
        self.budget = BudgetState(
            labels_total=cfg["budget"]["labels_total"],
            compute_total=float(cfg["budget"]["compute_total"]),
            tokens_total=int(cfg["budget"]["tokens_total"]),
            updates_total=int(cfg["budget"]["updates_total"]),
        )
        self.oracle = SyntheticOracle(self.benchmark.visible_y, self.budget)
        self.evaluator = HiddenEvaluator(self.benchmark, sanitizer=self.sanitizer)

        # ---- 基座模型 ----
        t0 = time.time()
        self.base_model = MLP(self.benchmark.d, self.benchmark.base_classes,
                              hidden=cfg["model"]["hidden"], seed=self.seed)
        base_res = self.base_model.fit(
            self.benchmark.base_x, self.benchmark.base_y,
            epochs=cfg["model"]["epochs"], lr=cfg["model"]["lr"],
            batch=cfg["model"]["batch"], seed=self.seed,
        )
        self.base_cost = make_cost(
            compute_units=base_res.sample_updates / cfg["cost_model"]["base_compute_unit"],
            wall_s=time.time() - t0,
        )
        val_metrics = evaluate(self.benchmark.val_y, self.base_model.predict(self.benchmark.val_x),
                               self.benchmark.base_classes)

        # ---- 引擎组件 ----
        self.registry = ModelRegistry(self.run_ctx.run_dir / "registry",
                                      self.run_ctx.artifacts_dir,
                                      self.benchmark.d, self.benchmark.base_classes, cfg)
        v0 = self.registry.add(self.base_model, {
            "created_t": -1, "parent": None, "kind": "base", "playbook": "BASE",
            "config": {"model": cfg["model"], "seed": self.seed},
            "metrics": val_metrics, "status": "champion", "note": "初始基座模型",
            "cost": self.base_cost.to_dict(),
        })
        self.registry.promote(v0.version)

        self.monitor = DriftMonitor(cfg, self.benchmark.d, self.benchmark.n_classes)
        self._init_monitor_reference()
        self.diagnoser = Diagnoser(cfg, self.benchmark.d, self.benchmark.n_classes,
                                   make_feature_groups(self.benchmark.d))
        self.library = PlaybookLibrary(cfg, self.oracle, cfg.get("cost_model", {}))
        # 策略表作用域：默认按 run 隔离（跨 run 不继承 → 评测公平、可复现）；
        # 需要"跨 run 累积经验"时置 engine.planner.policy_scope="global"。
        eng_planner = self.cfg.get("engine", {}).get("planner", {})
        if str(eng_planner.get("policy_scope", "run")) == "global":
            policy_path = Path(eng_planner.get("policy_path", "experiments/strategy_policy.json"))
        else:
            policy_path = self.run_ctx.run_dir / "strategy_policy.json"
        policy_path.parent.mkdir(parents=True, exist_ok=True)
        self.planner = Planner(cfg, self.budget, self.library, policy_path)
        self.gate = Gate(cfg)
        self.gate_adapter = _GateAdapter(self.gate)
        self.safety = SafetyGuard(cfg, self.budget)
        self.safety_adapter = _SafetyAdapter(self.safety, cfg)
        self.audit = AuditLog(Path(cfg["safety"]["audit_dir"]) / f"{self.run_ctx.run_id}.jsonl")
        self.watchdog = Watchdog(cfg)

        # ---- Harness / 记录 ----
        self.tracker = self._build_tracker()
        self.driver = self._build_driver()
        self.tracker.start_run(RunRecord(
            run_id=self.run_ctx.run_id, group=cfg["run"]["name"],
            name=f"{self.benchmark.name}-seed{self.seed}",
            kind="loop", stage="system",
            config=self._flat_config(), tags=[self.benchmark.name, f"seed={self.seed}"],
            created_at=time.time(),
        ))

        # ---- 运行状态 ----
        self.timeline: list[dict] = []
        self.decisions: list[dict] = []
        self.experiments: list[dict] = []
        self.hidden_scores = []
        self.strategy_updates: list[dict] = []
        self.infer_units = 0.0
        self.total_cost = Cost()
        self.records: list[str] = [self.run_ctx.run_id]
        self.running = True
        # Harness 工具上下文的持久状态（跨窗口保留 experiment/gate 句柄）
        self._tool_experiments: dict = {}
        self._tool_gate_results: dict = {}

    # ------------------------------------------------------------------
    # 初始化辅助
    # ------------------------------------------------------------------
    def _init_monitor_reference(self) -> None:
        """用前 warmup_chunks 个窗口做参考分布（校准期，不触发更新）。"""
        w = int(self.cfg["monitor"].get("warmup_chunks", 4))
        w = max(1, min(w, self.benchmark.n_chunks))
        xs = [self.benchmark.chunks[t].x for t in range(w)]
        try:
            self.monitor.reset_reference(np.vstack(xs))
        except Exception:
            pass
        self.warmup_chunks = w

    def _build_tracker(self):
        try:
            from .observability.tracker import build_tracker
            return build_tracker(self.cfg, self.run_ctx)
        except Exception as e:  # 观测层尚未就绪时的兜底
            try:
                from .observability.tracker import LocalTracker
                return LocalTracker(self.cfg, self.run_ctx)
            except Exception:
                return _NullTracker(str(e))

    def _build_driver(self):
        try:
            from .harness.agent import build_driver
            return build_driver(self.cfg, tools=None)
        except Exception:
            return _HeuristicDriver(self.cfg)

    def _flat_config(self) -> dict:
        flat = {}

        def walk(prefix, obj):
            if isinstance(obj, dict):
                for k, v in obj.items():
                    walk(f"{prefix}.{k}" if prefix else k, v)
            else:
                flat[prefix] = obj

        walk("", self.cfg)
        return flat

    # ------------------------------------------------------------------
    # 记录辅助
    # ------------------------------------------------------------------
    def _track(self, kind: str, stage: str, name: str, summary: dict,
               metrics: Optional[dict] = None, cost: Optional[Cost] = None,
               tags: Optional[list] = None) -> str:
        rid = new_run_id(kind)
        try:
            self.tracker.start_run(RunRecord(
                run_id=rid, group=self.cfg["run"]["name"],
                name=name, kind=kind, stage=stage,
                config={"t": summary.get("t", -1), "phase": summary.get("phase", -1)},
                summary=summary, tags=tags or [],
                t=int(summary.get("t", -1)), phase=int(summary.get("phase", -1)),
                parent_run=self.run_ctx.run_id, created_at=time.time(),
            ))
            if metrics:
                self.tracker.log_metrics(rid, {k: float(v) for k, v in metrics.items()
                                               if isinstance(v, (int, float, np.floating, np.integer))},
                                         step=int(summary.get("t", 0)))
            self.tracker.finish_run(rid, summary=summary, status="finished")
        except Exception as e:  # 记录失败不能中断闭环，但要留痕
            self.audit.append({"type": "tracker_error", "run": rid, "error": str(e)})
        self.records.append(rid)
        return rid

    # ------------------------------------------------------------------
    # 主循环
    # ------------------------------------------------------------------
    def run(self) -> dict:
        t_start = time.time()
        abl = self.cfg.get("ablation", {}) or {}
        meta_version, champion = self.registry.champion()
        prev_stable_version = meta_version.version

        for t, chunk in enumerate(self.benchmark.chunks):
            step_t0 = time.time()
            if self.sanitizer is not None and len(chunk.x) > 0:
                chunk = StreamChunk(t=chunk.t, phase=chunk.phase, ids=chunk.ids,
                                    x=self.sanitizer.transform(chunk.x), meta=dict(chunk.meta))
            champion_meta, champion = self.registry.champion()
            model_for_pred = champion

            # 1) 观测：预测 + 代理指标（无标签）
            proba = model_for_pred.predict_proba(chunk.x)
            pred = np.argmax(proba, axis=1)
            entropy = mean_entropy(proba)
            margin = mean_margin(proba)
            hist = prediction_histogram(proba, self.benchmark.n_classes)
            self.infer_units += len(chunk.x) / self.cfg["cost_model"]["base_compute_unit"] * 0.25

            # 2) 隐藏测试打分（只用于报告，不参与决策）
            hs = self.evaluator.score_chunk(model_for_pred, t)
            self.hidden_scores.append(hs)

            # 3) 监测（含预算内探针标签）
            report = self.monitor.update(chunk, model_for_pred, self.oracle)
            events = list(getattr(report, "events", []) or [])
            state = dict(getattr(report, "state", {}) or {})
            probe_acc = float(state.get("probe_acc", float("nan")))

            # 4) 上线看门狗（用无标签/探针信号，绝不用隐藏分数）
            wd = {"action": "none"}
            if abl.get("watchdog", True):
                wd_score = probe_acc if np.isfinite(probe_acc) else margin
                wd = self.watchdog.observe(t, chunk.phase, float(wd_score), champion_meta.version)
            if wd.get("action") == "rollback" and champion_meta.version > 0:
                info = self.registry.rollback(prev_stable_version,
                                              reason=f"watchdog: {wd.get('reason','')}")
                self._track("rollback", "rollback", f"watchdog-rollback@t{t}",
                            {"t": t, "phase": chunk.phase, **{k: v for k, v in wd.items()}},
                            tags=["rollback", "watchdog"])
                self.audit.append({"type": "rollback", "t": t, "reason": wd.get("reason", ""),
                                   "to_version": prev_stable_version})
                self.watchdog.reset()
                champion_meta, champion = self.registry.champion()
                self._notify_monitor_promote(champion, rebase=True)
            else:
                prev_stable_version = champion_meta.version

            # 5) 触发判定
            triggered = bool(events)
            if abl.get("trigger_mode") == "always":
                triggered = True
            elif abl.get("trigger_mode") == "never":
                triggered = False
            if t < self.warmup_chunks:
                triggered = False

            row = {
                "t": t, "phase": chunk.phase, "phase_name": self.benchmark.phase_names[chunk.phase],
                "entropy": entropy, "margin": margin, "probe_acc": probe_acc,
                "drift_events": [e.to_dict() for e in events],
                "triggered": triggered, "champion_version": champion_meta.version,
                "hidden_macro_f1": float(hs.macro_f1), "hidden_accuracy": float(hs.accuracy),
                "watchdog": wd, "diagnosis_kind": "", "chosen_playbook": "NOOP",
                "gate_passed": None, "deploy": False, "rollback": False,
                "cost_cumulative": self._cost_cumulative(),
            }

            if not triggered:
                self._log_window_metrics(t, hs, entropy, probe_acc, champion_meta.version)
                row["wall_s"] = time.time() - step_t0
                self.timeline.append(row)
                continue

            # 6) 诊断归因
            budget_before = self._budget_snapshot()
            diag = self.diagnoser.diagnose(chunk, model_for_pred, report,
                                           self._history_snapshot(), self.oracle)
            if not abl.get("diagnosis", True):
                diag = Diagnosis(t=t, phase=chunk.phase, kind="concept",
                                 severity=float(state.get("mmd", 0.5)), confidence=0.3,
                                 recommended=[{"playbook": "FULL_FINETUNE_REPLAY",
                                               "prior": 1.0, "why": "ablation: no diagnosis"}])
            self.total_cost = self.total_cost.add(diag.cost)
            self._track("diagnosis", "diagnose", f"diag-{diag.kind}@t{t}",
                        {"t": t, "phase": chunk.phase, "kind": diag.kind,
                         "severity": diag.severity, "confidence": diag.confidence,
                         "reason": self._diag_reason(diag)},
                        metrics={"severity": diag.severity, "confidence": diag.confidence},
                        tags=["diagnosis", diag.kind])
            row["diagnosis_kind"] = diag.kind

            # 7) 规划候选（预算/白名单过滤）
            candidates = self.planner.propose(diag, chunk, state)
            candidates = [self._as_candidate(c) for c in (candidates or [])]
            candidates = [c for c in candidates if self._candidate_allowed(c)]
            if not candidates:
                candidates = [{"playbook": "NOOP", "est_cost": make_cost(),
                               "prior": 1.0, "why": "无满足预算/白名单的候选"}]

            # 8-11) Harness 工具驱动闭环：实验 → 门禁 → 上线/回滚（安全护栏内置）
            cycle = self._harness_cycle(t, chunk, model_for_pred, report, diag,
                                        candidates, probe_acc, budget_before)
            row["chosen_playbook"] = cycle["chosen_playbook"]
            row["gate_passed"] = cycle["gate_passed"]
            row["deploy"] = cycle["deploy"]
            row["rollback"] = cycle["rollback"]
            row["champion_version"] = self.registry.champion()[0].version
            decision = cycle["decision"]
            self.decisions.append(decision.to_dict())
            self.audit.append({"type": "decision", **decision.to_dict()})
            row["cost_cumulative"] = self._cost_cumulative()
            row["wall_s"] = time.time() - step_t0
            self.timeline.append(row)
            self._log_window_metrics(t, hs, entropy, probe_acc,
                                     self.registry.champion()[0].version)
            if not self.running:
                break

        # ---- 汇总 ----
        wall = time.time() - t_start
        ev = StreamEvaluation(scores=self.hidden_scores,
                              phase_of_chunk=list(self.benchmark.phase_of_chunk),
                              phase_names=list(self.benchmark.phase_names),
                              n_classes=self.benchmark.n_classes)
        static_scores, static_ev = self._static_baseline()
        result = {
            "meta": {
                "schema_version": SCHEMA_VERSION,
                "run_id": self.run_ctx.run_id,
                "group": self.cfg["run"]["name"],
                "scenario": self.benchmark.name,
                "seed": self.seed,
                "wall_s": wall,
                "base_model_cost": self.base_cost.to_dict(),
                "driver": self.cfg["harness"]["driver"],
                "n_windows": len(self.timeline),
                "sanitizer": ({**self.sanitizer.stats.to_dict(),
                               "enabled": True, "clip_quantile": self.sanitizer.clip_quantile}
                              if self.sanitizer else {"enabled": False}),
            },
            "config": self.cfg,
            "benchmark": self.benchmark.describe(),
            "budget": self._budget_snapshot(),
            "cost": {
                "compute_units": self.budget.compute_used,
                "labels": self.budget.labels_used,
                "tokens": self.budget.tokens_used,
                "wall_s": self.total_cost.wall_s,
                "infer_units": self.infer_units,
                "adaptation_units": self.budget.compute_used,
                "scalar": (self.budget.compute_used
                           + self.cfg["cost_model"]["label_weight"] * self.budget.labels_used
                           + self.cfg["cost_model"]["token_weight"] * self.budget.tokens_used),
            },
            "hidden": ev.summary(),
            "static_baseline_hidden": static_ev.summary(),
            "comparison": compare_curves(ev, static_ev),
            "timeline": self.timeline,
            "decisions": self.decisions,
            "experiments": self.experiments,
            "registry": [m.to_dict() for m in self.registry.history()],
            "strategy_updates": self.strategy_updates,
            "records": {"loop_run_id": self.run_ctx.run_id, "runs": self.records,
                        "n_tracker_runs": len(self.records)},
            "audit": {"path": str(self.audit.path),
                      "last_hash": self.audit.last_hash() if hasattr(self.audit, "last_hash") else "",
                      "verified": bool(self.audit.verify())},
            "static_curve": static_ev.curve(),
            "method_curve": [float(s.macro_f1) for s in self.hidden_scores],
        }
        try:
            self.tracker.finish_run(self.run_ctx.run_id, summary={
                "hidden": ev.summary(), "cost": result["cost"],
                "comparison": result["comparison"], "audit_verified": result["audit"]["verified"],
            }, status="finished")
        except Exception:
            pass
        with open(self.run_ctx.run_dir / "result.json", "w", encoding="utf-8") as f:
            json.dump(result, f, ensure_ascii=False, indent=1, default=str)
        return result

    # ------------------------------------------------------------------
    # 静态基线（同一次 run 内计算，用于对比；不进入系统决策）
    # ------------------------------------------------------------------
    def _static_baseline(self):
        ev = self.evaluator.score_stream(self.base_model)
        return [s.macro_f1 for s in ev.scores], ev

    # ------------------------------------------------------------------
    # Harness 工具驱动闭环
    # ------------------------------------------------------------------
    def _cost_cumulative(self) -> dict:
        c = self.cfg.get("cost_model", {})
        return {"compute_units": self.budget.compute_used,
                "labels": self.budget.labels_used,
                "tokens": self.budget.tokens_used,
                "scalar": (self.budget.compute_used
                           + c.get("label_weight", 0.02) * self.budget.labels_used
                           + c.get("token_weight", 1e-5) * self.budget.tokens_used)}

    def _ensure_tools(self, t: int, chunk: StreamChunk, model: MLP, report,
                      diag: Diagnosis, candidates: list, probe_acc: float):
        """构造 Harness ToolContext + ToolRegistry（失败返回 None，走 legacy 兜底）。"""
        try:
            from .harness.tools import DefaultToolContext, build_tool_registry
        except Exception as e:
            self.audit.append({"type": "tools_unavailable", "error": str(e)})
            return None
        try:
            champ_metrics = {
                "probe_macro_f1": float(probe_acc) if np.isfinite(probe_acc) else 0.0,
                "class_recall_min": 1.0 / max(1, self.benchmark.n_classes),
                "pred_stability": 1.0,
            }
            ctx = DefaultToolContext(
                cfg=self.cfg,
                budget=self.budget,
                state={
                    "t": t, "phase": chunk.phase, "chunk": chunk, "model": model,
                    "base_x": self.benchmark.base_x, "base_y": self.benchmark.base_y,
                    "history": self._history_snapshot(), "rng": self.rng,
                    "n_classes": self.benchmark.n_classes,
                    "monitor_report": report, "diagnosis": diag,
                    "candidates": candidates,
                    "registry_summary": {**self._registry_summary(),
                                         "champion_metrics": champ_metrics},
                    "champion_metrics": champ_metrics,
                    "gate_results": self._tool_gate_results,
                    "experiments": self._tool_experiments,
                    "run_id": self.run_ctx.run_id,
                    "group": self.cfg["run"]["name"],
                    "escalations": [], "shadow_results": {},
                    "skills": list(getattr(self.driver, "skill_ids", []) or []),
                },
                oracle=self.oracle,
                diagnoser=self.diagnoser,
                planner=self.planner,
                library=self.library,
                registry=self.registry,
                safety=self.safety_adapter,
                tracker=self.tracker,
                gate=self.gate_adapter,
                evaluate_model=self._evaluate_model_shadow,
                build_playbook_context=lambda playbook, params: self._build_playbook_ctx(
                    model, chunk, diag, params=params, playbook=playbook),
                auto_approve_high_risk=bool(
                    self.cfg.get("safety", {}).get("auto_approve_high_risk", True)),
            )
            return build_tool_registry(ctx)
        except Exception as e:
            self.audit.append({"type": "tool_context_error", "error": str(e)})
            return None

    def _evaluate_model_shadow(self, candidate, ctx, meta: dict) -> dict:
        """给 tools.shadow_eval 用的无标签一致性评估。"""
        champion = self.registry.champion()[1]
        model = candidate if hasattr(candidate, "predict") else None
        if model is None and isinstance(candidate, dict):
            model = candidate.get("model")
        if model is None:
            return {"agreement": None, "n": 0, "note": "no_model"}
        chunk = (ctx.current() if hasattr(ctx, "current") else {}).get("chunk")
        if chunk is None:
            return {"agreement": None, "n": 0, "note": "no_chunk"}
        try:
            a = model.predict(chunk.x)
            b = champion.predict(chunk.x)
            return {"agreement": float(np.mean(a == b)), "n": int(len(chunk.x)),
                    "mode": (meta or {}).get("mode", "shadow")}
        except Exception as e:
            return {"agreement": None, "n": 0, "error": str(e)}

    def _harness_cycle(self, t: int, chunk: StreamChunk, model: MLP, report,
                       diag: Diagnosis, candidates: list, probe_acc: float,
                       budget_before: dict) -> dict:
        """一次受控闭环：由 Harness 驱动（工具执行），loop 负责记录/护栏/策略学习。"""
        tools = self._ensure_tools(t, chunk, model, report, diag, candidates, probe_acc)
        snapshot = {
            "t": t, "phase": chunk.phase,
            "monitor_report": report, "diagnosis": diag,
            "candidates": candidates, "budget": self._budget_snapshot(),
            "registry_summary": self._registry_summary(),
            "history": self._history_snapshot(limit=5),
            "skills": list(getattr(self.driver, "skill_ids", []) or []),
        }
        traj = None
        if tools is not None:
            traj = self._driver_decide(snapshot, tools)
        else:
            traj = self._driver_decide(snapshot, None)

        deployed = False
        rollback = False
        gate_passed = None
        chosen = "NOOP"
        decision = None
        if traj is not None and getattr(traj, "final_decision", None) is not None:
            dec = traj.final_decision
            outcome = dict(getattr(dec, "outcome", {}) or {})
            chosen = str(getattr(dec, "chosen_playbook", "NOOP") or "NOOP")
            deployed = bool(outcome.get("deployed") or outcome.get("action") == "deploy")
            rollback = bool(outcome.get("action") == "rollback")
            gate = outcome.get("gate")
            if isinstance(gate, dict):
                gate_passed = bool(gate.get("passed"))
            elif "passed" in outcome:
                gate_passed = bool(outcome.get("passed"))
            # 统一字段 + 轨迹审计
            dec.t = t
            dec.phase = chunk.phase
            if getattr(dec, "diagnosis", None) is None:
                dec.diagnosis = diag
            dec.budget_before = budget_before
            dec.budget_after = self._budget_snapshot()
            dec.driver = str(getattr(traj, "driver", self.cfg["harness"]["driver"]))
            saf = dict(getattr(dec, "safety", {}) or {})
            saf["tool_steps"] = [
                {"name": getattr(st, "name", ""), "ok": bool(getattr(st, "ok", False)),
                 "latency_s": round(float(getattr(st, "latency_s", 0.0)), 4),
                 "error": str(getattr(st, "error", ""))[:300],
                 "cost": getattr(st, "cost", Cost()).to_dict()
                 if hasattr(getattr(st, "cost", None), "to_dict") else {}}
                for st in (getattr(traj, "steps", []) or [])
            ]
            saf["tokens"] = int(getattr(traj, "tokens", 0) or 0)
            dec.safety = saf
            if not dec.evidence_refs:
                dec.evidence_refs = self.records[-4:]
            decision = dec
        else:
            # 兜底：legacy 执行路径（Harness 不可用时仍能闭环）
            chosen = candidates[0]["playbook"] if candidates else "NOOP"
            exp_obj, gate_res, deployed = (None, None, False)
            if chosen and chosen != "NOOP":
                cm, ch = self.registry.champion()
                exp_obj, gate_res, deployed = self._run_experiment(
                    chosen, chunk, t, diag, candidates, cm, ch, probe_acc)
                gate_passed = bool(gate_res.passed) if gate_res else None
            decision = self._make_decision(
                t, chunk, diag, candidates, chosen, exp_obj, gate_res, deployed,
                budget_before, {}, traj)

        # token 记账（driver 层 token；预算硬门在工具内已检查）
        tokens = int(getattr(traj, "tokens", 0) or 0) if traj is not None else 0
        if tokens > 0 and self.budget.tokens_left() >= tokens:
            self.budget.charge(Cost(tokens=tokens))

        # 上线/回滚后的监测器与看门狗状态维护
        if deployed:
            try:
                self.watchdog.reset()
            except Exception:
                pass
            self._notify_monitor_promote(self.registry.champion()[1])
        if rollback:
            self._notify_monitor_promote(self.registry.champion()[1], rebase=True)
            try:
                self.watchdog.reset()
            except Exception:
                pass

        # 策略层自改进（用真实 outcome 的增益/成本）
        abl = self.cfg.get("ablation", {}) or {}
        if abl.get("strategy_learning", True) and decision is not None:
            self._update_strategy_from_outcome(t, diag, chosen, decision, probe_acc)

        # 收集工具侧实验句柄（用于结果汇总/审计；剔除不可序列化的 model）
        try:
            for exp_id, handle in list(self._tool_experiments.items()):
                clean = {k: v for k, v in dict(handle).items() if k not in ("model",)}
                if not any(e.get("exp_id") == exp_id for e in self.experiments):
                    self.experiments.append(clean)
        except Exception:
            pass

        # 决策级记录（工具已自记 experiment/deploy run；这里记决策）
        self._track("decision", "decide", f"decision-{chosen}@t{t}", {
            "t": t, "phase": chunk.phase, "kind": diag.kind,
            "chosen_playbook": chosen, "deployed": deployed, "rollback": rollback,
            "gate_passed": gate_passed,
            "rationale": str(getattr(decision, "rationale", ""))[:500],
            "driver": str(getattr(decision, "driver", "")),
        }, metrics={"severity": diag.severity, "confidence": diag.confidence},
            tags=["decision", chosen, diag.kind])
        return {"chosen_playbook": chosen, "gate_passed": gate_passed,
                "deploy": deployed, "rollback": rollback, "decision": decision}

    def _update_strategy_from_outcome(self, t: int, diag: Diagnosis, chosen: str,
                                      decision, probe_acc: float) -> None:
        outcome = dict(getattr(decision, "outcome", {}) or {})
        cost = getattr(decision, "cost", None) or Cost()
        gain = outcome.get("probe_gain")
        if gain is None:
            metrics = outcome.get("metrics") or {}
            try:
                gain = float(metrics.get("probe_macro_f1", 0.0)) - (
                    float(probe_acc) if np.isfinite(probe_acc) else 0.0)
            except Exception:
                gain = 0.0
        passed = bool(outcome.get("deployed") or outcome.get("passed")
                      or outcome.get("action") == "deploy")
        reward = (float(gain) if passed else -0.01) - 0.02 * float(getattr(cost, "scalar", 0.0))
        try:
            self.planner.update_strategy(diag.kind, chosen, reward, cost)
            self.strategy_updates.append({"t": t, "kind": diag.kind, "playbook": chosen,
                                          "reward": float(reward), "passed": passed,
                                          "gain": float(gain), "cost": cost.to_dict()})
        except Exception as e:
            self.audit.append({"type": "strategy_error", "error": str(e)})

    # ------------------------------------------------------------------
    # 实验执行 + 门禁 + 上线
    # ------------------------------------------------------------------
    def _run_experiment(self, playbook: str, chunk: StreamChunk, t: int,
                        diag: Diagnosis, candidates: list, champion_meta: ModelVersion,
                        champion: MLP, probe_acc: float):
        meta, incumbent = self.registry.champion()
        ctx = self._build_playbook_ctx(incumbent, chunk, diag)
        exp_id = new_run_id("exp")
        best = None
        for _attempt in range(1):
            t0 = time.time()
            try:
                res = self.library.run(playbook, ctx)
            except Exception as e:
                self.audit.append({"type": "playbook_error", "t": t, "playbook": playbook,
                                   "error": str(e)})
                self._track("experiment", "optimize", f"exp-{playbook}@t{t}",
                            {"t": t, "phase": chunk.phase, "playbook": playbook,
                             "status": "failed", "error": str(e)},
                            tags=["experiment", playbook])
                return None, None, False
            cost = getattr(res, "cost", None) or Cost()
            metrics = dict(getattr(res, "metrics", {}) or {})
            self.total_cost = self.total_cost.add(cost)
            # 预算记账：labels 由 oracle 记；这里只记 compute（并计一次 update）
            self.budget.charge(Cost(compute_units=float(cost.compute_units),
                                    wall_s=float(cost.wall_s)), phase=chunk.phase)
            exp = Experiment(
                exp_id=exp_id, run_id=new_run_id("exp"), t=t, phase=chunk.phase,
                playbook=playbook, params=dict(getattr(res, "params", {}) or {}),
                hypothesis=self._hypothesis(diag, playbook), driver=self.cfg["harness"]["driver"],
                diagnosis_kind=diag.kind, cost_est=candidates[0]["est_cost"] if candidates else Cost(),
                cost_actual=cost, metrics=metrics,
                status="succeeded", started_at=t0,
            )
            self.experiments.append(exp.to_dict())
            self._track("experiment", "optimize", f"exp-{playbook}@t{t}",
                        {"t": t, "phase": chunk.phase, "playbook": playbook,
                         "status": "succeeded", **{k: v for k, v in metrics.items()
                                                   if isinstance(v, (int, float))}},
                        metrics={k: v for k, v in metrics.items()
                                 if isinstance(v, (int, float, np.floating, np.integer))},
                        tags=["experiment", playbook, diag.kind])
            if not hasattr(res, "model") or res.model is None:
                return None, None, False
            candidate_model = res.model
            break

        # 候选指标 vs 现任指标
        model_metrics = dict(getattr(res, "metrics", {}) or {})
        candidate_metrics = {
            "probe_macro_f1": float(model_metrics.get("probe_macro_f1", float("nan"))),
            "class_recall_min": float(model_metrics.get("class_recall_min", float("nan"))),
            "pred_stability": float(model_metrics.get("pred_stability", 1.0)),
        }
        incumbent_metrics = {
            "probe_macro_f1": float(probe_acc) if np.isfinite(probe_acc) else 0.0,
            "class_recall_min": float(self._incumbent_min_recall(champion, chunk)),
            "pred_stability": 1.0,
        }
        gate_ctx = {"t": t, "phase": chunk.phase, "diagnosis_kind": diag.kind,
                    "budget": self._budget_snapshot(),
                    "high_risk": playbook in set(self.cfg["safety"].get("high_risk_playbooks", []))}
        gate_res = self.gate.evaluate(candidate_metrics, incumbent_metrics, gate_ctx)
        if not self.cfg.get("ablation", {}).get("gate", True):
            gate_res.passed = True
            gate_res.decision = "promote"
            gate_res.reason = "ablation: gate disabled"

        deployed = False
        if gate_res.passed and gate_res.decision in ("promote", "shadow"):
            try:
                meta_new = self.registry.add(candidate_model, {
                    "created_t": t, "parent": meta.version, "kind": playbook.lower(),
                    "playbook": playbook, "config": {"playbook": playbook,
                                                     "params": exp.params, "seed": self.seed},
                    "metrics": model_metrics, "status": "candidate",
                    "note": self._hypothesis(diag, playbook),
                    "cost": cost.to_dict(),
                })
                self.registry.promote(meta_new.version)
                deployed = True
                self._track("deploy", "evaluate", f"deploy-v{meta_new.version}@t{t}",
                            {"t": t, "phase": chunk.phase, "playbook": playbook,
                             "version": meta_new.version, "parent": meta.version,
                             "domain_gap": gate_res.__dict__.get("criteria", {})},
                            metrics={k: v for k, v in candidate_metrics.items()
                                     if isinstance(v, (int, float))},
                            tags=["deploy", playbook])
                self.audit.append({"type": "deploy", "t": t, "playbook": playbook,
                                   "version": meta_new.version, "parent": meta.version,
                                   "metrics": candidate_metrics,
                                   "gate": gate_res.to_dict()})
                self.watchdog.reset()
                self._notify_monitor_promote(candidate_model)
            except Exception as e:
                self.audit.append({"type": "deploy_error", "t": t, "playbook": playbook,
                                   "error": str(e)})
        else:
            self._track("experiment_rejected", "optimize", f"reject-{playbook}@t{t}",
                        {"t": t, "phase": chunk.phase, "playbook": playbook,
                         "gate_passed": False, "reason": gate_res.reason},
                        tags=["rejected", playbook])
        return exp, gate_res, deployed

    def _build_playbook_ctx(self, model: MLP, chunk: StreamChunk, diag: Diagnosis,
                            params: Optional[dict] = None,
                            playbook: str = ""):
        extra = {"params": dict(params or {}), "playbook": playbook}
        try:
            return PlaybookContext(
                model=model, chunk=chunk,
                base_x=self.benchmark.base_x, base_y=self.benchmark.base_y,
                config=self.cfg, n_classes=self.benchmark.n_classes,
                history=self._history_snapshot(limit=5), diagnosis=diag,
                budget=self.budget, rng=self.rng,
                probe_x=np.zeros((0, self.benchmark.d)),
                probe_y=np.zeros((0,), dtype=np.int64),
                registry=self.registry,
                **extra,
            )
        except TypeError:  # 引擎 PlaybookContext 字段与契约有出入时兼容
            return PlaybookContext(
                model=model, chunk=chunk,
                base_x=self.benchmark.base_x, base_y=self.benchmark.base_y,
                config=self.cfg, n_classes=self.benchmark.n_classes,
                history=self._history_snapshot(limit=5), diagnosis=diag,
                budget=self.budget, rng=self.rng,
                probe_x=np.zeros((0, self.benchmark.d)),
                probe_y=np.zeros((0,), dtype=np.int64),
                registry=self.registry,
            )

    def _notify_monitor_promote(self, model: MLP, rebase: bool = False) -> None:
        """部署/回滚后更新监测器的参考域（避免对已适应的新域持续误报）。"""
        try:
            if hasattr(self.monitor, "on_promote"):
                self.monitor.on_promote(model)
            elif hasattr(self.monitor, "promote_reference"):
                self.monitor.promote_reference()
        except Exception as e:
            self.audit.append({"type": "monitor_rebase_error", "error": str(e)})

    def _incumbent_min_recall(self, model: MLP, chunk: StreamChunk) -> float:
        proba = model.predict_proba(chunk.x)
        pred = np.argmax(proba, axis=1)
        hist = np.bincount(pred, minlength=self.benchmark.n_classes).astype(float)
        hist = hist / max(1.0, hist.sum())
        return float(hist.min())

    # ------------------------------------------------------------------
    # Harness 与决策
    # ------------------------------------------------------------------
    def _driver_decide(self, snapshot: dict, tools=None) -> Trajectory:
        try:
            import inspect

            sig = inspect.signature(self.driver.decide)
            if len(sig.parameters) >= 2:
                return self.driver.decide(snapshot, tools)
            return self.driver.decide(snapshot)
        except Exception as e:
            self.audit.append({"type": "driver_error", "error": str(e)})
            return Trajectory(t=snapshot["t"], phase=snapshot["phase"],
                              driver="fallback", tokens=0,
                              final_decision=None)

    def _choose_playbook(self, traj, candidates: list, diag: Diagnosis) -> str:
        cand_names = [c["playbook"] for c in candidates]
        mode = (self.cfg.get("ablation", {}) or {}).get("planner", "bandit")
        if mode == "random" and candidates:
            return candidates[int(self.rng.integers(0, len(candidates)))]["playbook"]
        if mode == "fixed":
            return "FULL_FINETUNE_REPLAY" if "FULL_FINETUNE_REPLAY" in cand_names else cand_names[0]
        chosen = ""
        try:
            dec = getattr(traj, "final_decision", None)
            if dec is not None:
                if isinstance(dec, dict):
                    chosen = dec.get("chosen_playbook", "") or ""
                else:
                    chosen = getattr(dec, "chosen_playbook", "") or ""
        except Exception:
            chosen = ""
        if chosen not in cand_names:
            # 兜底：先验 × 成本排序第一
            ranked = sorted(candidates, key=lambda c: (-float(c.get("prior", 0)),
                                                       float(getattr(c.get("est_cost", Cost()), "scalar", 0))))
            chosen = ranked[0]["playbook"] if ranked else "NOOP"
        return chosen

    def _candidate_allowed(self, cand: dict) -> bool:
        name = cand.get("playbook")
        allowed = self.cfg["safety"].get("allowed_playbooks", [])
        if name not in allowed:
            return False
        est = cand.get("est_cost", Cost())
        can = self.budget.can_afford(est)
        return bool(can)

    def _as_candidate(self, c) -> dict:
        if isinstance(c, dict):
            return {"playbook": c.get("playbook", "NOOP"),
                    "est_cost": c.get("est_cost", Cost()),
                    "prior": float(c.get("prior", 0.0)),
                    "why": c.get("why", "")}
        return {"playbook": getattr(c, "playbook", "NOOP"),
                "est_cost": getattr(c, "est_cost", Cost()),
                "prior": float(getattr(c, "prior", 0.0)),
                "why": getattr(c, "why", "")}

    def _make_decision(self, t, chunk, diag, candidates, chosen, exp_obj,
                       gate_res, deployed, budget_before, checks, traj) -> Decision:
        cost = Cost()
        if exp_obj is not None:
            cost = cost.add(exp_obj.cost_actual)
        if diag is not None:
            cost = cost.add(diag.cost)
        outcome = {
            "deployed": bool(deployed),
            "gate": gate_res.to_dict() if gate_res is not None else None,
            "champion_version": self.registry.champion()[0].version,
        }
        if exp_obj is not None:
            outcome["metrics"] = exp_obj.metrics
        return Decision(
            t=t, phase=chunk.phase, diagnosis=diag, chosen_playbook=chosen,
            alternatives=[{"playbook": c["playbook"], "prior": c["prior"],
                           "est_cost": c["est_cost"].to_dict()} for c in candidates],
            budget_before=budget_before, budget_after=self._budget_snapshot(),
            cost=cost, outcome=outcome,
            rationale=self._rationale(diag, chosen, exp_obj, gate_res, deployed),
            evidence_refs=[r for r in self.records[-4:]],
            driver=self.cfg["harness"]["driver"],
            safety={"checks": checks, "driver_tokens": int(getattr(traj, "tokens", 0) or 0)},
        )

    def _rationale(self, diag, chosen, exp_obj, gate_res, deployed) -> str:
        parts = []
        if diag is not None:
            parts.append(f"诊断={diag.kind}(sev={diag.severity:.2f},conf={diag.confidence:.2f})")
        parts.append(f"选择={chosen}")
        if exp_obj is not None:
            m = exp_obj.metrics
            parts.append(f"候选probeF1={m.get('probe_macro_f1', float('nan')):.3f}")
        if gate_res is not None:
            parts.append(f"门禁={'通过' if gate_res.passed else '拒绝'}({gate_res.reason})")
        parts.append("已上线" if deployed else "未上线")
        return "；".join(parts)

    def _hypothesis(self, diag: Diagnosis, playbook: str) -> str:
        return f"漂移类型[{diag.kind}]下，{playbook} 的期望增益/成本优于当前 champion"

    def _diag_reason(self, diag: Diagnosis) -> str:
        rec = diag.recommended[0]["playbook"] if diag.recommended else "NOOP"
        return f"{diag.kind}: 推荐 {rec}；证据特征 {diag.affected_features[:5]}"

    def _update_strategy(self, diag, chosen, exp_obj, gate_res, probe_acc):
        done = bool(gate_res and gate_res.passed)
        gain = (float(exp_obj.metrics.get("probe_macro_f1", 0.0)) -
                (float(probe_acc) if np.isfinite(probe_acc) else 0.0))
        reward = (gain if done else -0.01) - 0.02 * exp_obj.cost_actual.scalar
        try:
            self.planner.update_strategy(diag.kind, chosen, reward, exp_obj.cost_actual)
            self.strategy_updates.append({"t": exp_obj.t, "kind": diag.kind,
                                          "playbook": chosen, "reward": reward,
                                          "passed": done, "cost": exp_obj.cost_actual.to_dict()})
        except Exception as e:
            self.audit.append({"type": "strategy_error", "error": str(e)})

    # ------------------------------------------------------------------
    # 小工具
    # ------------------------------------------------------------------
    def _budget_snapshot(self) -> dict:
        b = self.budget
        return {"labels_total": b.labels_total, "labels_used": b.labels_used,
                "compute_total": b.compute_total, "compute_used": b.compute_used,
                "tokens_total": b.tokens_total, "tokens_used": b.tokens_used,
                "updates_used": b.updates_used, "updates_total": b.updates_total}

    def _history_snapshot(self, limit: int = 10) -> list:
        return self.timeline[-limit:]

    def _registry_summary(self) -> dict:
        meta, _ = self.registry.champion()
        return {"champion": meta.to_dict(), "n_versions": len(self.registry.history())}

    def _log_window_metrics(self, t, hs, entropy, probe_acc, version):
        try:
            self.tracker.log_metrics(self.run_ctx.run_id, {
                "hidden_macro_f1": float(hs.macro_f1),
                "entropy": float(entropy),
                "probe_acc": float(probe_acc) if np.isfinite(probe_acc) else 0.0,
                "champion_version": float(version),
                "cost_compute": float(self.total_cost.compute_units),
                "cost_labels": float(self.total_cost.labels),
                "cost_tokens": float(self.total_cost.tokens),
            }, step=t)
        except Exception:
            pass


# --------------------------------------------------------------------------
# 兜底实现（观测层/Harness 未就绪时仍能独立跑通验证）
# --------------------------------------------------------------------------
class _NullTracker:
    def __init__(self, reason=""):
        self.reason = reason

    def start_run(self, record):
        return record.run_id

    def log_metrics(self, run_id, metrics, step=0):
        return None

    def log_artifact(self, run_id, path):
        return None

    def finish_run(self, run_id, summary=None, status="finished"):
        return None

    def query(self, filters=None):
        return []

    def export(self, fmt="json", out=None):
        return out


class _HeuristicDriver:
    """最小兜底驱动：按候选先验/成本排序选第一个（真实驱动由 harness 提供）。"""

    def __init__(self, cfg):
        self.cfg = cfg

    def decide(self, snapshot: dict, tools=None) -> Trajectory:
        cands = snapshot.get("candidates", [])
        chosen = "NOOP"
        if cands:
            ranked = sorted(cands, key=lambda c: (-float(c.get("prior", 0)),
                                                  float(getattr(c.get("est_cost", Cost()), "scalar", 0))))
            chosen = ranked[0]["playbook"]
        dec = Decision(t=snapshot.get("t", 0), phase=snapshot.get("phase", 0),
                       diagnosis=snapshot.get("diagnosis"), chosen_playbook=chosen,
                       rationale="fallback heuristic")
        return Trajectory(t=snapshot.get("t", 0), phase=snapshot.get("phase", 0),
                          final_decision=dec, driver="fallback", tokens=0)


def run_one(cfg: dict, group: str = "main", seed: Optional[int] = None) -> dict:
    cfg = json.loads(json.dumps(cfg))  # deep copy
    cfg.setdefault("run", {})["name"] = group
    if seed is not None:
        cfg["seed"] = int(seed)
    ctx = RunContext(cfg=cfg, run_id=new_run_id(group))
    loop = SelfEvolvingLoop(cfg, ctx)
    result = loop.run()
    for d in [ROOT / "results" / group]:
        d.mkdir(parents=True, exist_ok=True)
    scenario = result["meta"]["scenario"]
    out = ROOT / "results" / group / f"{scenario}_seed{cfg['seed']}.json"
    with open(out, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=1, default=str)
    return result


class _SafetyAdapter:
    """把 Harness 工具级动作名映射为 playbook 级动作名，交给 SafetyGuard 判定。

    C 的 tools 以工具名（run_experiment/deploy/rollback/...）调用安全护栏，
    而 B 的 SafetyGuard 以 playbook 名做白名单/频率/升级判定。本适配器在
    loop 集成层完成映射，不侵入任何一方；预算硬门仍由 tools 在调用前检查。
    """

    TOOL_TO_PLAYBOOK = {
        "run_experiment": None,       # 取 payload["playbook"]
        "deploy": None,               # 取 payload["playbook"]，缺省 ROLLBACK 以外最保守
        "rollback": "ROLLBACK",
        "shadow_eval": "SHADOW_ONLY",
        "request_labels": "NOOP",
        "diagnose": "NOOP",
        "inspect_drift": "NOOP",
        "budget_status": "NOOP",
        "propose_candidates": "NOOP",
        "list_registry": "NOOP",
        "query_history": "NOOP",
        "evaluate_candidate": "NOOP",
    }

    def __init__(self, guard, cfg: dict):
        self.guard = guard
        self.cfg = cfg

    def _effective(self, action: str, payload: dict) -> str:
        if action in ("run_experiment", "deploy"):
            pb = payload.get("playbook")
            if pb:
                return str(pb)
            # deploy 缺 playbook：按中等风险动作处理
            return "SHADOW_ONLY"
        return self.TOOL_TO_PLAYBOOK.get(action) or action

    def check(self, action: str, payload=None, state=None):
        payload = dict(payload or {})
        effective = self._effective(str(action or ""), payload)
        try:
            allowed, checks = self.guard.check(effective, payload, state or {})
        except Exception as e:
            return False, {"safety": "error", "error": str(e), "action": action}
        checks = dict(checks or {})
        checks["tool_action"] = action
        checks["effective_action"] = effective
        return bool(allowed), checks

    def should_kill(self) -> bool:
        try:
            return bool(self.guard.should_kill())
        except Exception:
            return False

    def __getattr__(self, item):
        return getattr(self.guard, item)


class _GateAdapter:
    """兼容 Gate 的两种调用风格：tools 传 {"metrics": {...}}（嵌套），
    engine.Gate 期望扁平指标。适配器展平后交给引擎 Gate 判定，不侵入双方。
    """

    def __init__(self, gate):
        self.gate = gate

    def evaluate(self, candidate, incumbent, ctx=None):
        cand = dict(candidate or {})
        metrics = dict(cand.get("metrics") or {})
        flat = dict(metrics)
        for k, v in cand.items():
            if k != "metrics":
                flat.setdefault(k, v)
        if "cost" in flat and isinstance(flat["cost"], dict) and "compute_units" not in flat:
            flat["compute_units"] = flat["cost"].get("compute_units", 0.0)
        return self.gate.evaluate(flat, dict(incumbent or {}), dict(ctx or {}))

    def __getattr__(self, item):
        return getattr(self.gate, item)
