"""Harness 工具层：ToolRegistry + 12 个工具（docs/03 §5.1）。

设计要点
--------
1. **依赖注入**：工具不直接 import 引擎/仓库，只依赖 `ToolContext` 协议。
   loop 把实时对象放进 `ctx.state`（t/phase/chunk/model/...），把引擎组件挂到
   `ctx.oracle/diagnoser/planner/library/registry/safety/tracker/gate` 上，
   Harness 即可驱动与单测（fake 引擎即可，不必等引擎实现完成）。
2. **权限与安全**：只读/写工具分级（`ToolSpec.side_effects`）；动作白名单、
   风险分级、kill-switch、高危 escalation、预算硬门都在调用前检查；
   所有异常被捕获成 `ToolCall(ok=False)`，绝不打断主闭环。
3. **可审计记账**：每次调用计时；工具把资源消耗写进 `result["cost"]`，
   ToolRegistry 汇总到 `ToolCall.cost`；预算由 `BudgetState.charge` 统一扣除
   （labels 由 oracle 扣、compute/tokens 由本层补扣，避免重复计数）。
4. **LLM 友好**：每个工具都有轻量 JSON Schema，`ToolRegistry.specs_json()`
   可直接拼进 system prompt；`call()` 返回 JSON 可序列化结果。

12 个工具
--------
只读: inspect_drift / budget_status / propose_candidates / list_registry / query_history
评估: shadow_eval
预算型: diagnose / request_labels
写   : run_experiment / evaluate_candidate / deploy / rollback
"""

from __future__ import annotations

import contextlib
import copy
import dataclasses
import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional, Protocol, Sequence, runtime_checkable

import numpy as np

from ..core.spec import (
    BudgetState,
    Cost,
    Decision,
    Diagnosis,
    Experiment,
    GateResult,
    MetricPoint,
    ModelVersion,
    RunRecord,
    ToolCall,
    ToolSpec,
)
from ..observability.tracker import make_run_id

ROOT = Path(__file__).resolve().parents[2]

# 风险等级 / 副作用分类（用于只读模式与 kill-switch 判定）
RISK_LEVELS = ("low", "medium", "high")
SIDE_EFFECTS = ("none", "budget", "model")   # none=只读；budget=耗预算；model=改模型/仓库


class ToolError(Exception):
    """工具内部可预期的失败（参数/预算/权限/引擎异常），由 Registry 捕获。"""

    def __init__(self, code: str, message: str, result: Optional[dict] = None):
        super().__init__(message)
        self.code = str(code)
        self.message = str(message)
        self.result: dict = dict(result or {})


# --------------------------------------------------------------------------
# 通用转换
# --------------------------------------------------------------------------
def to_plain(obj: Any, _depth: int = 0) -> Any:
    """把 dataclass / numpy / Cost / Path 递归转成 JSON 可序列化对象。"""
    if _depth > 12:
        return str(obj)
    if obj is None or isinstance(obj, (bool, int, float, str)):
        return obj
    if hasattr(obj, "tolist"):  # numpy.ndarray / 标量
        try:
            return to_plain(obj.tolist(), _depth + 1)
        except Exception:  # noqa: BLE001
            return str(obj)
    if dataclasses.is_dataclass(obj) and not isinstance(obj, type):
        try:
            return to_plain(dataclasses.asdict(obj), _depth + 1)
        except Exception:  # noqa: BLE001
            return {k: to_plain(v, _depth + 1) for k, v in vars(obj).items()}
    if isinstance(obj, dict):
        return {str(k): to_plain(v, _depth + 1) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [to_plain(v, _depth + 1) for v in obj]
    if hasattr(obj, "to_dict"):
        try:
            return to_plain(obj.to_dict(), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    if isinstance(obj, Path):
        return str(obj)
    if hasattr(obj, "__dict__"):  # 引擎 dataclass/SimpleNamespace 等
        try:
            return to_plain(vars(obj), _depth + 1)
        except Exception:  # noqa: BLE001
            pass
    return str(obj)


def _as_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _as_cost(obj: Any) -> Cost:
    """把 Cost / dict / None 统一成 Cost。"""
    if isinstance(obj, Cost):
        return Cost(**dataclasses.asdict(obj))
    if isinstance(obj, dict):
        return Cost(compute_units=float(obj.get("compute_units", 0.0) or 0.0),
                    labels=int(obj.get("labels", 0) or 0),
                    tokens=int(obj.get("tokens", 0) or 0),
                    wall_s=float(obj.get("wall_s", 0.0) or 0.0))
    return Cost()


def _cost_dict(cost: Cost) -> dict:
    return {"compute_units": float(cost.compute_units), "labels": int(cost.labels),
            "tokens": int(cost.tokens), "wall_s": float(cost.wall_s)}


# --------------------------------------------------------------------------
# ToolContext 协议与默认实现
# --------------------------------------------------------------------------
@runtime_checkable
class ToolContext(Protocol):
    """Harness 工具访问闭环状态的唯一入口（依赖注入，便于 fake 单测）。"""

    cfg: dict

    def snapshot(self) -> dict: ...

    def current(self) -> dict: ...


@dataclass
class DefaultToolContext:
    """`ToolContext` 的默认实现：loop 只需把实时对象塞进 `state`/字段即可。

    state 约定（由 loop 维护，工具只读；少数工具会写入 last_experiment 等结果键）::

        {"t": int, "phase": int, "chunk": StreamChunk, "model": MLP,
         "base_x": ndarray, "base_y": ndarray, "history": list, "rng": Generator,
         "n_classes": int, "monitor_report": MonitorReport,
         "diagnosis": Diagnosis, "candidates": list[dict],
         "registry_summary": dict, "champion_metrics": dict,
         "gate_results": dict, "experiments": dict, "run_id": str(loop run),
         "group": str, "escalations": list, "shadow_results": dict}

    也可设置 `build_playbook_context(name, params) -> Any` 完全接管
    PlaybookContext 的构造（否则本层用 SimpleNamespace 按属性构造，兼容引擎契约）。
    """

    cfg: dict = field(default_factory=dict)
    budget: Optional[BudgetState] = None
    state: dict = field(default_factory=dict)
    oracle: Any = None
    diagnoser: Any = None
    planner: Any = None
    library: Any = None
    registry: Any = None
    safety: Any = None
    tracker: Any = None
    gate: Any = None
    evaluate_model: Optional[Callable] = None
    build_playbook_context: Optional[Callable] = None
    approvals: set = field(default_factory=set)
    auto_approve_high_risk: Optional[bool] = None
    warnings: list = field(default_factory=list)

    def __post_init__(self):
        if not self.cfg:
            from ..core.config import load_config
            self.cfg = load_config()
        if self.budget is None:
            b = self.cfg.get("budget", {})
            self.budget = BudgetState(
                labels_total=int(b.get("labels_total", 0)),
                compute_total=float(b.get("compute_total", 0.0)),
                tokens_total=int(b.get("tokens_total", 0)),
                updates_total=int(b.get("updates_total", 0)),
            )
        for key, default in (("t", 0), ("phase", 0), ("history", []),
                             ("candidates", []), ("registry_summary", {}),
                             ("champion_metrics", {}), ("gate_results", {}),
                             ("experiments", {}), ("escalations", []),
                             ("shadow_results", {}), ("group", "main")):
            self.state.setdefault(key, default)

    # -- 协议 ----------------------------------------------------------
    def snapshot(self) -> dict:
        budget = self.budget
        return {
            "t": int(self.state.get("t", 0)),
            "phase": int(self.state.get("phase", 0)),
            "monitor_report": self.state.get("monitor_report"),
            "diagnosis": self.state.get("diagnosis"),
            "candidates": self.state.get("candidates", []),
            "budget": _budget_dict(budget, self.state.get("phase", 0),
                                   self.cfg) if budget is not None else {},
            "registry_summary": self.state.get("registry_summary", {}),
            "skills": self.state.get("skills", []),
        }

    def current(self) -> dict:
        return self.state

    # -- 便捷 ----------------------------------------------------------
    def approve(self, key: str) -> None:
        self.approvals.add(str(key))

    def revoke(self, key: str) -> None:
        self.approvals.discard(str(key))

    def is_approved(self, key: str) -> bool:
        if self.auto_approve_high_risk is None:
            mode = str(self.cfg.get("safety", {}).get("escalation_mode",
                                                      "record_and_proceed"))
            auto = mode in ("record_and_proceed", "auto", "record")
        else:
            auto = bool(self.auto_approve_high_risk)
        return auto or str(key) in self.approvals

    def warn(self, kind: str, message: str, **extra) -> dict:
        item = {"kind": kind, "message": message, "ts": time.time(), **extra}
        self.warnings.append(item)
        return item

    def charge(self, cost: Cost, phase: Optional[int] = None,
               labels_already_charged: bool = False) -> dict:
        if self.budget is None:
            return {}
        labels = 0 if labels_already_charged else int(cost.labels)
        charge = Cost(compute_units=float(cost.compute_units), labels=labels,
                      tokens=int(cost.tokens), wall_s=0.0)
        self.budget.charge(charge, phase=phase)
        return _budget_dict(self.budget, phase if phase is not None
                            else self.state.get("phase", 0), self.cfg)


def build_tool_context(cfg: Optional[dict] = None, budget: Optional[BudgetState] = None,
                       state: Optional[dict] = None, **bindings) -> DefaultToolContext:
    """工厂：`build_tool_context(cfg, budget, state, oracle=..., registry=..., ...)`。"""
    # 注意：state 是 loop 的**实时字典**，必须共享引用（工具会写 last_experiment
    # / gate_results / experiments 等结果键，loop 也要看到）
    return DefaultToolContext(cfg=cfg or {}, budget=budget,
                              state=state if state is not None else {},
                              **bindings)


# --------------------------------------------------------------------------
# 轻量 JSON Schema 校验（只覆盖工具协议用到的子集）
# --------------------------------------------------------------------------
def _type_ok(typ: str, value: Any) -> bool:
    if typ == "object":
        return isinstance(value, dict)
    if typ == "array":
        return isinstance(value, (list, tuple))
    if typ == "string":
        return isinstance(value, str)
    if typ == "integer":
        return isinstance(value, int) and not isinstance(value, bool)
    if typ == "number":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    if typ == "boolean":
        return isinstance(value, bool)
    if typ == "null":
        return value is None
    return True


def _validate_node(schema: dict, value: Any, path: str, errors: list[str]) -> None:
    if not isinstance(schema, dict):
        return
    typ = schema.get("type")
    if typ:
        types = typ if isinstance(typ, list) else [typ]
        if not any(_type_ok(t, value) for t in types):
            errors.append(f"{path}: 期望 {typ}，实际 {type(value).__name__}")
            return
    if "enum" in schema and value not in schema["enum"]:
        errors.append(f"{path}: 取值 {value!r} 不在允许集合 {schema['enum']} 内")
    if isinstance(value, dict):
        props = schema.get("properties") or {}
        for req in schema.get("required") or []:
            if req not in value:
                errors.append(f"{path}.{req}: 缺少必填参数")
        if schema.get("additionalProperties") is False:
            for key in value:
                if key not in props:
                    errors.append(f"{path}.{key}: schema 未声明的参数")
        for key, sub in props.items():
            if key in value:
                _validate_node(sub, value[key], f"{path}.{key}", errors)
    elif isinstance(value, (list, tuple)) and "items" in schema:
        for i, item in enumerate(value):
            _validate_node(schema["items"], item, f"{path}[{i}]", errors)
    elif isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            errors.append(f"{path}: {value} < minimum {schema['minimum']}")
        if "maximum" in schema and value > schema["maximum"]:
            errors.append(f"{path}: {value} > maximum {schema['maximum']}")
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            errors.append(f"{path}: {value} <= exclusiveMinimum {schema['exclusiveMinimum']}")


def validate_args(schema: dict, args: Any) -> list[str]:
    """返回错误列表；空列表=合法。"""
    if not isinstance(args, dict):
        return ["args 必须是 JSON 对象（dict）"]
    errors: list[str] = []
    _validate_node(schema, args, "args", errors)
    return errors


def apply_defaults(schema: dict, args: dict) -> dict:
    """填充 schema 中声明的默认值（返回新 dict，不改调用方对象）。"""
    out = copy.deepcopy(args) if isinstance(args, dict) else {}
    props = (schema or {}).get("properties") or {}
    for key, sub in props.items():
        if key not in out and isinstance(sub, dict) and "default" in sub:
            out[key] = copy.deepcopy(sub["default"])
    return out


def _obj(schema_props: Optional[dict] = None, required: Sequence[str] = (),
         additional: bool = False) -> dict:
    return {"type": "object", "properties": schema_props or {},
            "required": list(required), "additionalProperties": bool(additional)}


# --------------------------------------------------------------------------
# 权限策略与 ToolRegistry
# --------------------------------------------------------------------------
@dataclass
class ToolPolicy:
    """工具级权限策略（loop 可下发只读/受限模式）。"""

    allowed_tools: Optional[set] = None       # None=全部已注册工具
    read_only: bool = False                    # True: 只允许 side_effects=none
    deny_high_risk: bool = False               # True: 直接拒绝 risk=high 的工具
    require_approval_for_high: bool = False    # True: high 风险需 ctx 审批
    max_calls: int = 0                         # 单次决策最大工具调用数（0=不限）


class ToolRegistry:
    """工具注册表：校验 + 权限 + 异常捕获 + 计时/记账（契约 §5.1）。"""

    def __init__(self, ctx: ToolContext, policy: Optional[ToolPolicy] = None):
        self.ctx = ctx
        self.policy = policy or ToolPolicy()
        self._specs: dict[str, ToolSpec] = {}
        self._handlers: dict[str, Callable[[dict, ToolContext], dict]] = {}
        self.calls: list[ToolCall] = []

    # -- 注册 ----------------------------------------------------------
    def register(self, spec: ToolSpec, fn: Callable[[dict, ToolContext], dict]) -> None:
        if not isinstance(spec, ToolSpec):
            raise TypeError("register 需要 ToolSpec")
        if spec.risk not in RISK_LEVELS:
            raise ValueError(f"非法 risk={spec.risk!r}，支持 {RISK_LEVELS}")
        if spec.side_effects not in SIDE_EFFECTS:
            raise ValueError(f"非法 side_effects={spec.side_effects!r}，支持 {SIDE_EFFECTS}")
        if not callable(fn):
            raise TypeError("handler 必须可调用")
        self._specs[spec.name] = spec
        self._handlers[spec.name] = fn

    def specs(self) -> list[ToolSpec]:
        return [copy.deepcopy(s) for s in self._specs.values()]

    def spec(self, name: str) -> Optional[ToolSpec]:
        return copy.deepcopy(self._specs.get(name))

    def names(self) -> list[str]:
        return list(self._specs)

    def specs_json(self) -> list[dict]:
        return [s.to_dict() for s in self.specs()]

    def brief_specs(self) -> list[str]:
        """紧凑工具说明（LLM prompt 用，比完整 JSON Schema 省 ~70% token）。"""
        out = []
        for spec in self._specs.values():
            props = []
            schema = spec.input_schema or {}
            required = set(schema.get("required") or [])
            for key, sub in (schema.get("properties") or {}).items():
                typ = sub.get("type", "any") if isinstance(sub, dict) else "any"
                suffix = ".必填" if key in required else ""
                extra = ""
                if isinstance(sub, dict):
                    if "enum" in sub:
                        extra = "=" + "|".join(map(str, sub["enum"]))
                    elif "default" in sub:
                        extra = f"={sub['default']}"
                props.append(f"{key}:{typ}{extra}{suffix}")
            out.append(f'- {spec.name}({", ".join(props)}) [risk={spec.risk},'
                       f' side_effects={spec.side_effects}] {spec.description}')
        return out

    def calls_since(self, n: int = 0) -> list[ToolCall]:
        return list(self.calls[n:])

    # -- 调用 ----------------------------------------------------------
    def call(self, name: str, args: Optional[dict] = None) -> ToolCall:
        t0 = time.perf_counter()
        args = {} if args is None else args
        call = self._deny(name, args, t0)
        if call is not None:
            return call
        spec = self._specs[name]
        # 参数校验（先填默认值再校验）
        try:
            norm_args = apply_defaults(spec.input_schema, args)
        except Exception as exc:  # noqa: BLE001 - 参数本身畸形
            return self._finish(ToolCall(name=name, args=to_plain(args), ok=False,
                                         error=f"参数解析失败: {exc}"), t0)
        errors = validate_args(spec.input_schema, norm_args)
        if errors:
            return self._finish(
                ToolCall(name=name, args=to_plain(norm_args), ok=False,
                         error="参数校验失败: " + "; ".join(errors),
                         result={"error_kind": "invalid_args",
                                 "validation_errors": errors}),
                t0)
        # 注意：权限/预算/安全由各 handler 内部再做细粒度检查（依赖具体上下文）
        try:
            raw = self._handlers[name](norm_args, self.ctx)
        except ToolError as exc:
            result = dict(exc.result or {})
            result.setdefault("error_kind", exc.code)
            return self._finish(ToolCall(name=name, args=to_plain(norm_args),
                                         ok=False, error=exc.message,
                                         result=to_plain(result)), t0)
        except Exception as exc:  # noqa: BLE001 - 工具异常绝不外泄
            result = {"error_kind": "exception",
                      "exception": type(exc).__name__,
                      "detail": str(exc)[:800]}
            with contextlib.suppress(Exception):
                _ctx_warn(self.ctx, "tool_exception",
                          f"{name} 执行异常: {type(exc).__name__}: {exc}")
            return self._finish(ToolCall(name=name, args=to_plain(norm_args),
                                         ok=False, error=str(exc), result=result),
                                t0)
        if raw is None:
            raw = {}
        if not isinstance(raw, dict):
            raw = {"value": raw}
        cost = _as_cost(raw.get("cost"))
        result = to_plain(raw)
        return self._finish(ToolCall(name=name, args=to_plain(norm_args), ok=True,
                                     result=result, cost=cost), t0)

    def _deny(self, name: str, args: Any, t0: float) -> Optional[ToolCall]:
        """调用前权限检查：未知工具 / 白名单 / 只读 / 高风险 / kill-switch / 次数。"""
        if name not in self._handlers:
            return self._finish(
                ToolCall(name=name, args=to_plain(args), ok=False,
                         error=f"未注册的工具: {name!r}；可用: {sorted(self._handlers)}",
                         result={"error_kind": "unknown_tool",
                                 "available": sorted(self._handlers)}), t0)
        spec = self._specs[name]
        policy = self.policy
        if policy.allowed_tools is not None and name not in policy.allowed_tools:
            return self._end_denied(name, args, t0, "tool_not_allowed",
                                    f"工具 {name!r} 不在本轮白名单 {sorted(policy.allowed_tools)} 内")
        if policy.read_only and spec.side_effects != "none":
            return self._end_denied(name, args, t0, "read_only_violation",
                                    f"只读模式禁止 {name!r}（side_effects={spec.side_effects}）")
        if policy.deny_high_risk and spec.risk == "high":
            return self._end_denied(name, args, t0, "high_risk_denied",
                                    f"策略禁止高风险工具 {name!r}（risk=high）")
        if policy.require_approval_for_high and spec.risk == "high":
            if not _ctx_is_approved(self.ctx, f"tool:{name}"):
                return self._end_denied(name, args, t0, "needs_approval",
                                        f"高风险工具 {name!r} 需要审批（ctx.approve）")
        if policy.max_calls and len(self.calls) >= policy.max_calls:
            return self._end_denied(name, args, t0, "max_calls_exceeded",
                                    f"已调用 {len(self.calls)} 次，达到上限 {policy.max_calls}")
        if spec.side_effects in ("budget", "model", "external"):
            killed = _kill_switch_active(self.ctx)
            if killed:
                emergency_rollback = (name == "rollback"
                                      and bool(args.get("emergency")))
                if not emergency_rollback:
                    return self._end_denied(name, args, t0, "kill_switch",
                                            f"kill-switch 生效中，拒绝写动作 {name!r}")
        return None

    def _end_denied(self, name: str, args: Any, t0: float, code: str,
                    message: str) -> ToolCall:
        call = ToolCall(name=name, args=to_plain(args), ok=False, error=message,
                        result={"error_kind": code, "denied": True})
        with contextlib.suppress(Exception):
            state = _ctx_state(self.ctx)
            state.setdefault("denied_calls", []).append(
                {"tool": name, "code": code, "reason": message,
                 "t": state.get("t", 0)})
        return self._finish(call, t0)

    def _finish(self, call: ToolCall, t0: float) -> ToolCall:
        call.latency_s = max(0.0, time.perf_counter() - t0)
        self.calls.append(call)
        return call


# --------------------------------------------------------------------------
# ctx 访问辅助（对自定义 ToolContext 宽容）
# --------------------------------------------------------------------------
def _ctx_state(ctx: Any) -> dict:
    state = getattr(ctx, "state", None)
    if isinstance(state, dict):
        return state
    current = getattr(ctx, "current", None)
    if callable(current):
        try:
            cur = current()
            if isinstance(cur, dict):
                return cur
        except Exception:  # noqa: BLE001
            pass
    return {}


def _ctx_current(ctx: Any) -> dict:
    current = getattr(ctx, "current", None)
    if callable(current):
        try:
            cur = current()
            if isinstance(cur, dict):
                return cur
        except Exception:  # noqa: BLE001
            pass
    return _ctx_state(ctx)


def _ctx_snapshot(ctx: Any) -> dict:
    snap = getattr(ctx, "snapshot", None)
    if callable(snap):
        try:
            out = snap()
            if isinstance(out, dict):
                return out
        except Exception:  # noqa: BLE001
            pass
    return _ctx_state(ctx)


def _ctx_cfg(ctx: Any) -> dict:
    cfg = getattr(ctx, "cfg", None)
    return cfg if isinstance(cfg, dict) else {}


def _ctx_budget(ctx: Any) -> Optional[BudgetState]:
    budget = getattr(ctx, "budget", None)
    return budget if isinstance(budget, BudgetState) else None


def _ctx_get(ctx: Any, name: str) -> Any:
    return getattr(ctx, name, None)


def _ctx_warn(ctx: Any, kind: str, message: str, **extra) -> None:
    warn = getattr(ctx, "warn", None)
    if callable(warn):
        with contextlib.suppress(Exception):
            warn(kind, message, **extra)
            return
    warnings = getattr(ctx, "warnings", None)
    if isinstance(warnings, list):
        warnings.append({"kind": kind, "message": message,
                         "ts": time.time(), **extra})


def _ctx_is_approved(ctx: Any, key: str) -> bool:
    fn = getattr(ctx, "is_approved", None)
    if callable(fn):
        with contextlib.suppress(Exception):
            return bool(fn(key))
    approvals = getattr(ctx, "approvals", None)
    if isinstance(approvals, (set, list, tuple)):
        return str(key) in approvals
    return False


def _ctx_explicitly_approved(ctx: Any, key: str) -> bool:
    """只认 loop/人类显式注入的批准（不允许 auto/escalation 模式冒充）。"""
    approvals = getattr(ctx, "approvals", None)
    if isinstance(approvals, (set, list, tuple)):
        return str(key) in approvals
    return False


def _ctx_auto_approve(ctx: Any) -> bool:
    """校赛阶段高危动作以 escalation 记录代替人工审批（docs/06）。"""
    flag = getattr(ctx, "auto_approve_high_risk", None)
    if flag is not None:
        return bool(flag)
    cfg = _ctx_cfg(ctx)
    mode = str(cfg.get("safety", {}).get("escalation_mode",
                                          "record_and_proceed"))
    return mode in ("record_and_proceed", "auto", "record")


def _kill_switch_active(ctx: Any) -> bool:
    """优先问引擎 SafetyGuard；未绑定时回退到 cfg.safety.kill_switch_file。

    兼容 loop 的 `_SafetyAdapter`：适配器本身可能只暴露 check()，真正的
    SafetyGuard 挂在 `<adapter>.guard` 上，这里递归探测一层。
    """
    safety = _ctx_get(ctx, "safety")
    for candidate in (safety, getattr(safety, "guard", None) if safety is not None
                      else None):
        if candidate is not None and hasattr(candidate, "should_kill"):
            try:
                return bool(candidate.should_kill())
            except Exception:  # noqa: BLE001
                return False
    cfg = _ctx_cfg(ctx)
    name = str(cfg.get("safety", {}).get("kill_switch_file", "") or "")
    if not name:
        return False
    path = Path(name)
    if not path.is_absolute():
        path = ROOT / path
    return path.exists()


def _safety_check(ctx: Any, action: str, payload: dict, state: dict) -> tuple[bool, dict]:
    safety = _ctx_get(ctx, "safety")
    if safety is None or not hasattr(safety, "check"):
        return True, {"safety": "unbound", "action": action}
    try:
        allowed, checks = safety.check(action, payload, state)
        return bool(allowed), to_plain(checks if isinstance(checks, dict)
                                       else {"checks": checks})
    except Exception as exc:  # noqa: BLE001 - 安全组件异常按拒绝处理（保守）
        return False, {"safety": "error", "error": str(exc), "action": action}


def _charge_cost(ctx: Any, cost: Cost, phase: Optional[int],
                 labels_before: Optional[int] = None) -> dict:
    """扣预算：labels 若已被 oracle 扣过则只补扣 compute/tokens（防重复计数）。"""
    budget = _ctx_budget(ctx)
    if budget is None:
        return {}
    labels_uncovered = int(cost.labels)
    if labels_before is not None:
        spent_by_oracle = max(0, int(budget.labels_used) - int(labels_before))
        labels_uncovered = max(0, int(cost.labels) - spent_by_oracle)
    charge = Cost(compute_units=float(cost.compute_units),
                  labels=labels_uncovered, tokens=int(cost.tokens), wall_s=0.0)
    try:
        budget.charge(charge, phase=phase)
    except Exception:  # noqa: BLE001
        return {}
    return {"charged": _cost_dict(charge),
            "budget_after": _budget_dict(budget, phase or 0, _ctx_cfg(ctx))}


def _budget_dict(budget: Optional[BudgetState], phase: int = 0,
                 cfg: Optional[dict] = None) -> dict:
    cfg = cfg or {}
    per_phase = int(cfg.get("budget", {}).get("per_phase_updates", 0) or 0)
    if budget is None:
        return {}
    updates_left_phase = (budget.updates_left(int(phase), per_phase)
                          if per_phase else
                          max(0, budget.updates_total - budget.updates_used))
    return {
        "labels_total": budget.labels_total, "labels_used": budget.labels_used,
        "labels_left": budget.labels_left(),
        "compute_total": budget.compute_total, "compute_used": budget.compute_used,
        "compute_left": budget.compute_left(),
        "tokens_total": budget.tokens_total, "tokens_used": budget.tokens_used,
        "tokens_left": budget.tokens_left(),
        "updates_total": budget.updates_total, "updates_used": budget.updates_used,
        "updates_left": max(0, budget.updates_total - budget.updates_used),
        "per_phase_updates": per_phase,
        "updates_left_phase": updates_left_phase,
        "phase": int(phase),
    }


def _phase_of(ctx: Any, args: Optional[dict] = None) -> int:
    args = args or {}
    if args.get("phase") is not None:
        return int(args["phase"])
    return int(_ctx_current(ctx).get("phase", _ctx_state(ctx).get("phase", 0)) or 0)


def _risk_of(cfg: dict, playbook: str) -> str:
    high = set(cfg.get("safety", {}).get("high_risk_playbooks", []) or [])
    if playbook in high:
        return "high"
    if playbook in ("NOOP", "RECALIBRATE_AFFINE", "PRIOR_CORRECTION",
                    "LAST_LAYER_FT", "SHADOW_ONLY"):
        return "low"
    return "medium"


def _allowed_playbooks(cfg: dict) -> list[str]:
    return list(cfg.get("safety", {}).get("allowed_playbooks", []) or [])


def _record_escalation(ctx: Any, action: str, payload: dict) -> dict:
    """校赛阶段以 escalation 记录代替人工审批（仍需门禁/kill-switch 约束）。"""
    item = {"action": action, "payload": to_plain(payload),
            "t": int(_ctx_current(ctx).get("t", 0) or 0),
            "phase": int(_ctx_current(ctx).get("phase", 0) or 0),
            "reason": "high_risk_human_gate_escalation",
            "ts": time.time()}
    state = _ctx_state(ctx)
    state.setdefault("escalations", []).append(item)
    return item


# --------------------------------------------------------------------------
# 实验/追踪辅助
# --------------------------------------------------------------------------
def _next_exp_id(ctx: Any) -> str:
    state = _ctx_state(ctx)
    t = int(state.get("t", 0) or 0)
    phase = int(state.get("phase", 0) or 0)
    seq = len(state.get("experiments", {}) or {}) + 1
    exp_id = f"EXP-p{phase}-t{t:04d}-{seq:03d}"
    # 极端情况：同 t 重复调用，追加后缀保证唯一
    while exp_id in (state.get("experiments") or {}):
        seq += 1
        exp_id = f"EXP-p{phase}-t{t:04d}-{seq:03d}"
    return exp_id


def _tracker_start(ctx: Any, *, kind: str, stage: str, name: str,
                   config: Optional[dict] = None, tags: Optional[list] = None,
                   t: Optional[int] = None, phase: Optional[int] = None,
                   parent_run: Optional[str] = None,
                   group: Optional[str] = None) -> str:
    tracker = _ctx_get(ctx, "tracker")
    state = _ctx_state(ctx)
    t = int(state.get("t", 0) if t is None else t)
    phase = int(state.get("phase", 0) if phase is None else phase)
    group = str(group or state.get("group", "main"))
    parent = parent_run or str(state.get("run_id") or "")
    run_id = make_run_id(group)
    if tracker is None or not hasattr(tracker, "start_run"):
        return run_id
    record = RunRecord(run_id=run_id, group=group, name=name, kind=kind,
                       stage=stage, config=to_plain(config or {}),
                       tags=list(tags or []), t=t, phase=phase,
                       parent_run=parent or None, created_at=time.time())
    try:
        return tracker.start_run(record) or run_id
    except Exception as exc:  # noqa: BLE001 - 记录失败不打断主流程
        _ctx_warn(ctx, "tracker_start_failed", f"{kind}/{name}: {exc}")
        return run_id


def _tracker_metrics(ctx: Any, run_id: str, metrics: dict, step: int, t: int) -> None:
    tracker = _ctx_get(ctx, "tracker")
    if tracker is None or not hasattr(tracker, "log_metrics") or not metrics:
        return
    try:
        tracker.log_metrics(run_id, to_plain(metrics), step=int(step), t=int(t))
    except Exception as exc:  # noqa: BLE001
        _ctx_warn(ctx, "tracker_metrics_failed", f"{run_id}: {exc}")


def _tracker_artifact(ctx: Any, run_id: str, artifact: Any, step: int) -> str:
    tracker = _ctx_get(ctx, "tracker")
    if tracker is None or not hasattr(tracker, "log_artifact"):
        return ""
    if not artifact:
        return ""
    path = Path(str(artifact))
    if not path.is_absolute():
        path = ROOT / path
    if not path.exists():
        return ""
    try:
        return tracker.log_artifact(run_id, path, step=int(step)) or ""
    except Exception as exc:  # noqa: BLE001
        _ctx_warn(ctx, "tracker_artifact_failed", f"{run_id}: {exc}")
        return ""


def _tracker_finish(ctx: Any, run_id: str, summary: dict,
                    status: str = "finished") -> None:
    tracker = _ctx_get(ctx, "tracker")
    if tracker is None or not hasattr(tracker, "finish_run"):
        return
    try:
        tracker.finish_run(run_id, to_plain(summary), status=status)
    except Exception as exc:  # noqa: BLE001
        _ctx_warn(ctx, "tracker_finish_failed", f"{run_id}: {exc}")


def _as_diagnosis(ctx: Any, obj: Any) -> Optional[Diagnosis]:
    if obj is None or isinstance(obj, Diagnosis):
        return obj
    if isinstance(obj, dict):
        known = set(Diagnosis.__dataclass_fields__)  # type: ignore[attr-defined]
        kwargs = {k: v for k, v in obj.items() if k in known}
        kwargs.setdefault("t", int(_ctx_current(ctx).get("t", 0) or 0))
        kwargs.setdefault("phase", int(_ctx_current(ctx).get("phase", 0) or 0))
        kwargs.setdefault("kind", "benign")
        kwargs.setdefault("severity", 0.0)
        kwargs.setdefault("confidence", 0.0)
        try:
            return Diagnosis(**kwargs)
        except TypeError:
            return None
    return None


def _build_playbook_context(ctx: Any, playbook: str, params: dict,
                            diagnosis: Any) -> Any:
    builder = _ctx_get(ctx, "build_playbook_context")
    if callable(builder):
        return builder(playbook, dict(params or {}))
    cur = _ctx_current(ctx)
    cfg = _ctx_cfg(ctx)
    try:  # 引擎实现后优先用真 PlaybookContext 类型
        from ..engine.playbooks import PlaybookContext  # type: ignore
        return PlaybookContext(
            model=cur.get("model"), chunk=cur.get("chunk"),
            base_x=cur.get("base_x"), base_y=cur.get("base_y"),
            config=cfg, n_classes=int(cur.get("n_classes", cfg.get("benchmark", {}).get("n_classes", 4))),
            history=list(cur.get("history") or []), diagnosis=diagnosis,
            budget=_ctx_budget(ctx), rng=cur.get("rng") or np.random.default_rng(0))
    except Exception:  # noqa: BLE001 - 引擎未就绪/构造失败 → 属性鸭子类型
        from types import SimpleNamespace
        return SimpleNamespace(
            model=cur.get("model"), chunk=cur.get("chunk"),
            base_x=cur.get("base_x"), base_y=cur.get("base_y"),
            config=cfg,
            n_classes=int(cur.get("n_classes", cfg.get("benchmark", {}).get("n_classes", 4))),
            history=list(cur.get("history") or []), diagnosis=diagnosis,
            budget=_ctx_budget(ctx), rng=cur.get("rng") or np.random.default_rng(0),
            params=dict(params or {}), playbook=playbook)


# --------------------------------------------------------------------------
# 12 个工具实现
# --------------------------------------------------------------------------
def _drift_level(severity: float) -> str:
    if severity <= 0.05:
        return "none"
    if severity < 0.15:
        return "weak"
    if severity < 0.55:
        return "medium"
    return "strong"


def tool_inspect_drift(args: dict, ctx: Any) -> dict:
    cur = _ctx_current(ctx)
    snap = _ctx_snapshot(ctx)
    report = cur.get("monitor_report") or snap.get("monitor_report")
    if report is None:
        raise ToolError("no_monitor_report",
                        "当前没有监测报告：loop 需把 MonitorReport 放入 "
                        "ctx.state['monitor_report'] 或 snapshot['monitor_report']",
                        {"hint": "先运行 DriftMonitor.update(chunk, model, oracle)"})
    rep = to_plain(report)
    if not isinstance(rep, dict):
        rep = {"events": [], "state": {}}
    events = rep.get("events")
    state = rep.get("state")
    if not isinstance(events, list):
        events = []
    if not isinstance(state, dict):
        state = {}
    cfg = _ctx_cfg(ctx)
    mon = cfg.get("monitor", {}) or {}
    rep_t = _as_int(rep.get("t", 0))
    t = int(cur.get("t", snap.get("t", rep_t)) or 0)
    rep_phase = _as_int(rep.get("phase", 0))
    phase = int(cur.get("phase", snap.get("phase", rep_phase)) or 0)
    severities = [float(e.get("severity", 0.0) or 0.0) for e in events
                  if isinstance(e, dict)]
    max_sev = max(severities, default=0.0)
    include_events = bool(args.get("include_events", True))
    max_events = int(args.get("max_events", 12) or 12)
    warmup = t < int(mon.get("warmup_chunks", 4) or 0)
    return {
        "t": t, "phase": phase,
        "drift_detected": bool(events) and (max_sev > 0.0 or not warmup),
        "warmup": warmup,
        "n_events": len(events),
        "max_severity": round(max_sev, 4),
        "level": _drift_level(max_sev),
        "detectors": sorted({str(e.get("detector", "")) for e in events
                             if isinstance(e, dict)}),
        "events": events[:max_events] if include_events else [],
        "state": state,
        "thresholds": {
            "mmd": mon.get("mmd_threshold"),
            "ks_alpha": mon.get("ks_alpha"),
            "proxy_drop": mon.get("proxy_drop_threshold"),
            "novelty": mon.get("novelty_threshold"),
            "severity_high": cfg.get("diagnosis", {}).get("severity_high"),
        },
        "source": "monitor_report",
        "cost": _cost_dict(Cost()),
    }


def tool_diagnose(args: dict, ctx: Any) -> dict:
    cur = _ctx_current(ctx)
    snap = _ctx_snapshot(ctx)
    cached = cur.get("diagnosis") or snap.get("diagnosis")
    force = bool(args.get("force", False))
    if cached is not None and not force:
        dia = _as_diagnosis(ctx, cached)
        return {"diagnosis": to_plain(dia or cached), "cached": True,
                "kind": getattr(dia, "kind", None), "cost": _cost_dict(Cost()),
                "note": "复用本轮已有诊断（force=true 可重算）"}
    diagnoser = _ctx_get(ctx, "diagnoser")
    if diagnoser is None or not hasattr(diagnoser, "diagnose"):
        raise ToolError("diagnoser_unavailable",
                        "未绑定 Diagnoser：loop 需设置 ctx.diagnoser（含 diagnose 方法）",
                        {"cached_available": cached is not None})
    chunk = cur.get("chunk")
    model = cur.get("model")
    report = cur.get("monitor_report") or snap.get("monitor_report")
    history = cur.get("history") or []
    oracle = _ctx_get(ctx, "oracle")
    if chunk is None or model is None or report is None:
        raise ToolError("context_missing",
                        "诊断需要 ctx.state 中的 chunk/model/monitor_report",
                        {"has_chunk": chunk is not None, "has_model": model is not None,
                         "has_report": report is not None})
    budget = _ctx_budget(ctx)
    labels_before = int(budget.labels_used) if budget is not None else None
    try:
        dia = diagnoser.diagnose(chunk, model, report, history, oracle)
    except Exception as exc:  # noqa: BLE001 - 引擎异常转工具错误
        raise ToolError("diagnose_failed",
                        f"Diagnoser.diagnose 失败: {type(exc).__name__}: {exc}") from exc
    dia_obj = _as_diagnosis(ctx, dia)
    if dia_obj is None:
        raise ToolError("diagnose_invalid", f"诊断返回类型非法: {type(dia).__name__}")
    state = _ctx_state(ctx)
    state["diagnosis"] = dia_obj
    cost = _as_cost(getattr(dia_obj, "cost", None))
    phase = _phase_of(ctx, args)
    charged = _charge_cost(ctx, cost, phase, labels_before=labels_before)
    return {
        "diagnosis": to_plain(dia_obj), "cached": False,
        "kind": dia_obj.kind, "severity": float(dia_obj.severity),
        "confidence": float(dia_obj.confidence),
        "affected_features": list(dia_obj.affected_features),
        "affected_groups": list(dia_obj.affected_groups),
        "recommended": list(dia_obj.recommended),
        "cost": _cost_dict(cost), "budget_charge": charged,
    }


def tool_budget_status(args: dict, ctx: Any) -> dict:
    budget = _ctx_budget(ctx)
    if budget is None:
        raise ToolError("budget_unavailable", "未绑定 BudgetState")
    phase = _phase_of(ctx, args)
    info = _budget_dict(budget, phase, _ctx_cfg(ctx))
    info.update({"ok": True, "cost": _cost_dict(Cost()),
                 "suggestion": _budget_suggestion(info, _ctx_cfg(ctx))})
    return info


def _budget_suggestion(info: dict, cfg: dict) -> str:
    if info.get("updates_left_phase", 1) <= 0:
        return "当前 phase 更新次数已用满：只能 NOOP/ROLLBACK，等待下一 phase"
    if info.get("labels_left", 0) < 2 * int(cfg.get("budget", {}).get("probe_labels", 6) or 6):
        return "标签预算接近耗尽：停止实验，保留监测探针预算"
    if info.get("compute_left", 0.0) <= 0:
        return "算力预算耗尽：只能选择零算力 playbook（NOOP/RECALIBRATE_AFFINE/PRIOR_CORRECTION）"
    if info.get("tokens_left", 0) <= 0.3 * max(1, info.get("tokens_total", 1)):
        return "token 预算低于 30%：LLM 驱动应降级为确定性策略"
    return "预算充足，可按 cheapest-first 选择候选"


_DEFAULT_PLAYBOOK_BY_KIND = {
    "benign": "NOOP",
    "repair": "ROLLBACK",
    "covariate_affine": "RECALIBRATE_AFFINE",
    "condition_switch": "LAST_LAYER_FT",
    "concept": "FULL_FINETUNE_REPLAY",
    "novel_class": "NOVEL_CLASS_EXTEND",
    "label_shift": "PRIOR_CORRECTION",
    "noise_inflate": "RECALIBRATE_AFFINE",
}


def _library_specs(library: Any) -> dict:
    out: dict[str, Any] = {}
    if library is None or not hasattr(library, "specs"):
        return out
    try:
        for spec in library.specs() or []:
            name = getattr(spec, "name", None)
            if name:
                out[str(name)] = spec
    except Exception:  # noqa: BLE001
        return out
    return out


def tool_propose_candidates(args: dict, ctx: Any) -> dict:
    cfg = _ctx_cfg(ctx)
    cur = _ctx_current(ctx)
    snap = _ctx_snapshot(ctx)
    dia_raw = cur.get("diagnosis") or snap.get("diagnosis")
    dia = _as_diagnosis(ctx, dia_raw)
    if dia is None:
        raise ToolError("no_diagnosis", "没有诊断结果：先调 diagnose 再做候选规划")
    budget = _ctx_budget(ctx)
    phase = _phase_of(ctx, args)
    max_n = int(args.get("max_candidates",
                         cfg.get("planner", {}).get("max_candidates", 3)) or 3)
    planner = _ctx_get(ctx, "planner")
    chunk = cur.get("chunk")
    used_fallback = False
    raw_candidates: list = []
    if planner is not None and hasattr(planner, "propose") and chunk is not None:
        try:
            raw_candidates = list(planner.propose(dia, chunk, cur) or [])
            if raw_candidates and hasattr(planner, "order"):
                with contextlib.suppress(Exception):
                    raw_candidates = list(planner.order(raw_candidates) or raw_candidates)
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "planner_propose_failed", f"planner.propose 失败: {exc}")
            raw_candidates = []
    if not raw_candidates:
        used_fallback = True
        raw_candidates = _fallback_candidates(ctx, dia, cfg)
    allowed = set(_allowed_playbooks(cfg))
    lib_specs = _library_specs(_ctx_get(ctx, "library"))
    normalized: list[dict] = []
    for cand in raw_candidates:
        if isinstance(cand, str):
            cand = {"playbook": cand}
        if not isinstance(cand, dict):
            continue
        name = str(cand.get("playbook") or cand.get("name") or "")
        if not name:
            continue
        spec = lib_specs.get(name)
        est = _as_cost(cand.get("est_cost")
                       if cand.get("est_cost") is not None
                       else getattr(spec, "est_cost", None))
        prior = float(cand.get("prior", 0.0) or 0.0)
        why = str(cand.get("why", "") or "")
        risk = str(getattr(spec, "risk", None) or _risk_of(cfg, name))
        needs_labels = bool(getattr(spec, "needs_labels", est.labels > 0))
        allowed_ok = (not allowed) or name in allowed
        affordable = True
        blocked = ""
        if not allowed_ok:
            affordable, blocked = False, "不在动作白名单"
        elif budget is not None:
            if budget.updates_left(phase, int(cfg.get("budget", {}).get(
                    "per_phase_updates", 0) or 0) or 10 ** 9) <= 0:
                affordable, blocked = False, "当前 phase 更新次数已用满"
            elif not budget.can_afford(est, phase,
                                       int(cfg.get("budget", {}).get(
                                           "per_phase_updates", 0) or 0) or None):
                affordable, blocked = False, "预算不足（labels/compute/tokens/updates）"
        normalized.append({
            "playbook": name, "prior": round(prior, 4), "why": why,
            "est_cost": _cost_dict(est), "est_cost_scalar": round(est.scalar, 6),
            "risk": risk, "needs_labels": needs_labels,
            "layer": str(getattr(spec, "layer", "") or ""),
            "affordable": affordable, "blocked_reason": blocked,
        })
    # 按确定性规则排序：高严重度先验优先，否则 cheapest-first
    severity = float(dia.severity)
    severity_high = float(cfg.get("diagnosis", {}).get("severity_high", 0.55))
    if severity >= severity_high:
        normalized.sort(key=lambda c: (-c["prior"], c["est_cost_scalar"], c["playbook"]))
    else:
        normalized.sort(key=lambda c: (c["est_cost_scalar"], -c["prior"], c["playbook"]))
    normalized = normalized[:max_n]
    state = _ctx_state(ctx)
    state["candidates"] = [dict(c) for c in normalized]
    return {"candidates": normalized, "count": len(normalized),
            "fallback": used_fallback, "diagnosis_kind": dia.kind,
            "severity": severity, "budget": _budget_dict(budget, phase, cfg),
            "cost": _cost_dict(Cost())}


def _fallback_candidates(ctx: Any, dia: Diagnosis, cfg: dict) -> list[dict]:
    """规划器未绑定时，用诊断推荐 + skills 矩阵兜底（确定性、可复现）。"""
    out: list[dict] = []
    for rec in dia.recommended or []:
        if isinstance(rec, dict) and rec.get("playbook"):
            out.append({"playbook": rec["playbook"],
                        "prior": float(rec.get("prior", 0.5) or 0.5),
                        "why": str(rec.get("why", "") or "诊断推荐")})
    if not out:
        name = _DEFAULT_PLAYBOOK_BY_KIND.get(dia.kind, "NOOP")
        out.append({"playbook": name, "prior": 0.5,
                    "why": f"skills 矩阵：{dia.kind} → {name}"})
        if dia.kind not in ("benign", "repair") and name != "NOOP":
            out.append({"playbook": "SHADOW_ONLY", "prior": 0.2,
                        "why": "兜底：证据不足时只做影子评估"})
    lib = _ctx_get(ctx, "library")
    for cand in out:
        spec = _library_specs(lib).get(cand["playbook"])
        if "est_cost" not in cand or cand["est_cost"] is None:
            cand["est_cost"] = getattr(spec, "est_cost", Cost())
    return out


def tool_run_experiment(args: dict, ctx: Any) -> dict:
    cfg = _ctx_cfg(ctx)
    cur = _ctx_current(ctx)
    snap = _ctx_snapshot(ctx)
    name = str(args.get("playbook") or "")
    params = dict(args.get("params") or {})
    hypothesis = str(args.get("hypothesis") or "")
    allowed = _allowed_playbooks(cfg)
    if allowed and name not in allowed:
        raise ToolError("playbook_not_allowed",
                        f"playbook {name!r} 不在动作白名单 {allowed} 内",
                        {"allowed_playbooks": allowed})
    library = _ctx_get(ctx, "library")
    if library is None or not hasattr(library, "run"):
        raise ToolError("library_unavailable",
                        "未绑定 PlaybookLibrary（需提供 run(name, ctx) 方法）")
    missing = [k for k in ("model", "chunk", "base_x", "base_y")
               if cur.get(k) is None]
    if missing:
        raise ToolError("context_missing",
                        f"实验需要 ctx.state 中的实时对象，缺少: {missing}",
                        {"missing": missing})
    if _kill_switch_active(ctx):
        raise ToolError("kill_switch", "kill-switch 生效，禁止启动新实验")
    dia = _as_diagnosis(ctx, cur.get("diagnosis") or snap.get("diagnosis"))
    budget = _ctx_budget(ctx)
    phase = _phase_of(ctx, args)
    per_phase = int(cfg.get("budget", {}).get("per_phase_updates", 0) or 0)
    est = None
    if hasattr(library, "estimate_cost"):
        try:
            est = _as_cost(library.estimate_cost(name, dia, cur.get("chunk"), budget))
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "estimate_cost_failed", f"{name}: {exc}")
            est = None
    if est is None:
        spec = _library_specs(library).get(name)
        est = _as_cost(getattr(spec, "est_cost", None))
    if budget is not None and not budget.can_afford(
            est, phase, per_phase if per_phase > 0 else None):
        raise ToolError("budget_exceeded",
                        f"预算不足，无法执行 {name}：估算 {_cost_dict(est)}，"
                        f"余额 {_budget_dict(budget, phase, cfg)}",
                        {"est_cost": _cost_dict(est),
                         "budget": _budget_dict(budget, phase, cfg),
                         "playbook": name})
    allowed_safety, safety_checks = _safety_check(
        ctx, "run_experiment",
        {"playbook": name, "params": params, "t": cur.get("t", 0)}, cur)
    if not allowed_safety:
        raise ToolError("safety_veto", f"安全护栏拒绝 run_experiment({name})",
                        {"checks": safety_checks, "playbook": name})
    risk = _risk_of(cfg, name)
    escalation = None
    if risk == "high" and not _ctx_explicitly_approved(ctx, f"run_experiment:{name}"):
        if not _ctx_auto_approve(ctx):
            raise ToolError("needs_approval",
                            f"高风险 playbook {name!r} 需要 human_gate 审批"
                            "（显式 ctx.approvals 或 escalation_mode=record_and_proceed）",
                            {"playbook": name, "risk": risk})
        escalation = _record_escalation(
            ctx, "run_experiment", {"playbook": name, "risk": risk,
                                    "est_cost": _cost_dict(est),
                                    "mode": "record_and_proceed"})
    exp_id = _next_exp_id(ctx)
    t = int(cur.get("t", 0) or 0)
    loop_run_id = str(_ctx_state(ctx).get("run_id") or "")
    run_id = _tracker_start(
        ctx, kind="experiment", stage="optimize",
        name=f"{name}@t{t}", config={
            "playbook": name, "params": params, "hypothesis": hypothesis,
            "diagnosis_kind": getattr(dia, "kind", ""), "exp_id": exp_id,
            "est_cost": _cost_dict(est), "risk": risk,
            "loop_run_id": loop_run_id},
        tags=["experiment", name, f"phase{phase}", f"t{t}"], t=t, phase=phase)
    labels_before = int(budget.labels_used) if budget is not None else None
    pb_ctx = _build_playbook_context(ctx, name, params, dia)
    wall0 = time.perf_counter()
    try:
        result = library.run(name, pb_ctx)
    except Exception as exc:  # noqa: BLE001
        cost_fail = Cost(wall_s=time.perf_counter() - wall0)
        _tracker_finish(ctx, run_id, {"status": "failed", "playbook": name,
                                      "error": f"{type(exc).__name__}: {exc}",
                                      "exp_id": exp_id}, status="failed")
        raise ToolError(
            "experiment_failed",
            f"{name} 执行失败（已记录 failed run）: {type(exc).__name__}: {exc}",
            {"exp_id": exp_id, "run_id": run_id, "playbook": name,
             "cost": _cost_dict(cost_fail)}) from exc
    wall_s = time.perf_counter() - wall0
    metrics = to_plain(getattr(result, "metrics", {}) or {})
    cost = _as_cost(getattr(result, "cost", None))
    cost.wall_s = max(float(cost.wall_s), wall_s)
    charged = _charge_cost(ctx, cost, phase, labels_before=labels_before)
    required = ("probe_macro_f1", "pred_stability", "class_recall_min")
    missing_metrics = [k for k in required if k not in (metrics or {})]
    raw_artifacts = to_plain(getattr(result, "artifacts", []) or [])
    model_new = getattr(result, "model", None) or cur.get("model")
    version: Optional[int] = None
    artifact = ""
    registry = _ctx_get(ctx, "registry")
    if registry is not None and hasattr(registry, "add") and model_new is not None:
        meta = {"playbook": name, "t": t, "phase": phase,
                "diagnosis_kind": getattr(dia, "kind", ""),
                "hypothesis": hypothesis, "exp_id": exp_id,
                "metrics": metrics, "cost": _cost_dict(cost),
                "note": str(getattr(result, "note", "") or ""),
                "params": params}
        try:
            mv = registry.add(model_new, meta)
            version = int(getattr(mv, "version", -1))
            version = version if version >= 0 else None
            artifact = str(getattr(mv, "artifact", "") or "")
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "registry_add_failed",
                      f"{name} 候选入库失败（实验记录保留）: {exc}")
    if artifact:
        _tracker_artifact(ctx, run_id, artifact, t)
    experiment = Experiment(
        exp_id=exp_id, run_id=run_id, t=t, phase=phase, playbook=name,
        params=params, hypothesis=hypothesis, driver="harness",
        diagnosis_kind=getattr(dia, "kind", ""), cost_est=est,
        cost_actual=cost, metrics=metrics, status="succeeded",
        started_at=time.time() - wall_s)
    state = _ctx_state(ctx)
    handle = {"exp_id": exp_id, "run_id": run_id, "playbook": name,
              "model": model_new, "metrics": metrics, "cost": cost,
              "version": version, "artifact": artifact,
              "experiment": experiment, "t": t, "phase": phase,
              "risk": risk}
    state.setdefault("experiments", {})[exp_id] = handle
    state["last_experiment"] = exp_id
    _tracker_metrics(ctx, run_id, metrics, step=t, t=t)
    _tracker_finish(ctx, run_id, {
        "status": "succeeded", "exp_id": exp_id, "playbook": name,
        "version": version, "metrics": metrics,
        "cost": _cost_dict(cost), "gain_per_compute": _gain_per_compute(metrics, cost),
        "artifact": artifact, "hypothesis": hypothesis}, status="succeeded")
    return {
        "exp_id": exp_id, "run_id": run_id, "playbook": name,
        "version": version, "status": "succeeded",
        "hypothesis": hypothesis, "params": params,
        "metrics": metrics, "missing_metrics": missing_metrics,
        "cost": _cost_dict(cost), "cost_est": _cost_dict(est),
        "budget_charge": charged,
        "artifacts": ([artifact] if artifact else []) + raw_artifacts,
        "note": str(getattr(result, "note", "") or ""),
        "diagnosis_kind": getattr(dia, "kind", ""),
        "risk": risk, "escalation": to_plain(escalation),
        "safety": to_plain(safety_checks), "t": t, "phase": phase,
        "gain_per_compute": _gain_per_compute(metrics, cost),
    }


def _gain_per_compute(metrics: dict, cost: Cost) -> float:
    probe = metrics.get("probe_macro_f1") if isinstance(metrics, dict) else None
    if probe is None:
        return 0.0
    return round(float(probe) / max(1e-9, float(cost.compute_units) + 1e-9)
                 if cost.compute_units > 0 else 0.0, 6)


def _find_experiment(ctx: Any, exp_id: Optional[str],
                     version: Optional[int]) -> dict:
    experiments = _ctx_state(ctx).get("experiments", {}) or {}
    if exp_id:
        handle = experiments.get(str(exp_id))
        if handle is None:
            raise ToolError("experiment_not_found",
                            f"未找到实验 {exp_id!r}（可用: {sorted(experiments)[-5:]}）")
        return handle
    if version is not None:
        for handle in experiments.values():
            if int(handle.get("version") or -1) == int(version):
                return handle
        raise ToolError("experiment_not_found", f"未找到 version={version} 的候选实验")
    last = _ctx_state(ctx).get("last_experiment")
    if last and last in experiments:
        return experiments[last]
    raise ToolError("experiment_not_found", "没有候选实验：先 run_experiment")


def _champion_metrics(ctx: Any) -> dict:
    state = _ctx_state(ctx)
    snap = _ctx_snapshot(ctx)
    out: dict = {}
    summary = snap.get("registry_summary")
    if isinstance(summary, dict):
        if isinstance(summary.get("champion_metrics"), dict):
            out = dict(summary["champion_metrics"])
        elif isinstance(summary.get("metrics"), dict):
            out = dict(summary["metrics"])
        elif isinstance(summary.get("champion"), dict):
            out = dict(summary["champion"].get("metrics") or {})
    if not out:
        out = dict(state.get("champion_metrics") or {})
    if not out:
        registry = _ctx_get(ctx, "registry")
        if registry is not None and hasattr(registry, "champion"):
            try:
                champ = registry.champion()
                mv = champ[0] if isinstance(champ, (tuple, list)) else champ
                out = dict(getattr(mv, "metrics", {}) or {})
            except Exception:  # noqa: BLE001
                out = {}
    return out


def _fallback_gate(cfg: dict, metrics: dict, champ_metrics: dict,
                   cost: Cost) -> dict:
    """门禁兜底实现（引擎 Gate 未绑定时使用；阈值与 skills/04 同源）。"""
    gates = cfg.get("gates", {}) or {}
    min_gain = float(gates.get("min_probe_gain", 0.005))
    max_recall_drop = float(gates.get("max_recall_drop", 0.12))
    max_cost = float(gates.get("max_cost_per_update", 40.0))
    criteria: dict = {}
    probe = metrics.get("probe_macro_f1")
    champ_probe = champ_metrics.get("probe_macro_f1")
    if probe is not None:
        criteria["probe_available"] = {"value": float(probe), "threshold": 0.0,
                                       "pass": True}
    else:
        criteria["probe_available"] = {"value": None, "threshold": 0.0,
                                       "pass": False}
    if probe is not None and champ_probe is not None:
        gain = float(probe) - float(champ_probe)
        criteria["probe_gain"] = {"value": round(gain, 4),
                                  "threshold": min_gain,
                                  "pass": gain >= min_gain}
    recall_min = metrics.get("class_recall_min")
    champ_recall = champ_metrics.get("class_recall_min")
    if recall_min is not None:
        floor = (float(champ_recall) - max_recall_drop) if champ_recall is not None \
            else 0.0
        criteria["class_recall_min"] = {"value": float(recall_min),
                                        "threshold": round(floor, 4),
                                        "pass": float(recall_min) >= floor}
    criteria["cost"] = {"value": round(float(cost.compute_units), 4),
                        "threshold": max_cost,
                        "pass": float(cost.compute_units) <= max_cost}
    if not criteria:
        return {"criteria": {}, "passed": False, "decision": "reject",
                "reason": "没有可用评估指标"}
    passed = all(bool(c.get("pass")) for c in criteria.values())
    probe_gain = criteria.get("probe_gain", {}).get("value")
    if passed:
        decision, reason = "promote", "全部门禁项通过"
    elif probe_gain is not None and float(probe_gain) >= min_gain:
        decision, reason = "shadow", "有增益但部分门禁项不通过，转影子评估"
    else:
        decision, reason = "reject", "增益不足或指标塌陷"
    return {"criteria": criteria, "passed": passed, "decision": decision,
            "reason": reason}


def tool_evaluate_candidate(args: dict, ctx: Any) -> dict:
    cfg = _ctx_cfg(ctx)
    handle = _find_experiment(ctx, args.get("exp_id"), args.get("version"))
    metrics = dict(handle.get("metrics") or {})
    cost = _as_cost(handle.get("cost"))
    champ_metrics = _champion_metrics(ctx)
    gate_res: Optional[dict] = None
    gate = _ctx_get(ctx, "gate")
    if gate is not None and hasattr(gate, "evaluate"):
        candidate = {"exp_id": handle.get("exp_id"),
                     "playbook": handle.get("playbook"),
                     "metrics": metrics,
                     "cost": _cost_dict(cost),
                     "version": handle.get("version")}
        try:
            res = gate.evaluate(candidate, champ_metrics, _ctx_state(ctx))
            gate_res = to_plain(res)
            if hasattr(res, "criteria") and not isinstance(gate_res, dict):
                gate_res = to_plain(res)
            if not isinstance(gate_res, dict):
                gate_res = {"criteria": {}, "passed": False,
                            "decision": "reject", "reason": "Gate 返回非法"}
            gate_res.setdefault("criteria", {})
            gate_res.setdefault("passed", False)
            gate_res.setdefault("decision", "promote" if gate_res["passed"] else "reject")
            gate_res.setdefault("reason", "")
            if "passed" not in gate_res and "criteria" in gate_res:
                gate_res["passed"] = bool(gate_res.get("decision") == "promote")
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "gate_failed", f"Gate.evaluate 失败，使用兜底规则: {exc}")
    if gate_res is None:
        gate_res = _fallback_gate(cfg, metrics, champ_metrics, cost)
    probe = metrics.get("probe_macro_f1")
    champ_probe = champ_metrics.get("probe_macro_f1")
    delta = (float(probe) - float(champ_probe)
             if probe is not None and champ_probe is not None else None)
    state = _ctx_state(ctx)
    state.setdefault("gate_results", {})[str(handle.get("exp_id"))] = gate_res
    if handle.get("version") is not None:
        state["gate_results"][str(handle["version"])] = gate_res
    state["last_gate"] = gate_res
    if gate_res.get("passed"):
        state["champion_metrics"] = metrics  # 尚未 deploy，但门禁口径以候选为参考
    required = ("probe_macro_f1", "pred_stability", "class_recall_min")
    return {
        "exp_id": handle.get("exp_id"), "version": handle.get("version"),
        "playbook": handle.get("playbook"), "metrics": metrics,
        "champion_metrics": champ_metrics,
        "probe_gain": delta,
        "gate": gate_res, "passed": bool(gate_res.get("passed")),
        "decision": gate_res.get("decision", "reject"),
        "missing_metrics": [k for k in required if k not in metrics],
        "candidate_cost": _cost_dict(cost), "cost": _cost_dict(Cost()),
        "t": handle.get("t"), "phase": handle.get("phase"),
    }


def tool_deploy(args: dict, ctx: Any) -> dict:
    cfg = _ctx_cfg(ctx)
    version = args.get("version")
    if version is None:
        raise ToolError("invalid_args", "deploy 需要 version 参数")
    version = int(version)
    reason = str(args.get("reason") or "")
    force = bool(args.get("force", False))
    registry = _ctx_get(ctx, "registry")
    if registry is None or not hasattr(registry, "promote"):
        raise ToolError("registry_unavailable", "未绑定 ModelRegistry（需 promote 方法）")
    if _kill_switch_active(ctx):
        raise ToolError("kill_switch", "kill-switch 生效，禁止上线（rollback 除外）")
    state = _ctx_state(ctx)
    gate_results = state.get("gate_results", {}) or {}
    gate_res = gate_results.get(str(version)) or state.get("last_gate")
    handle = None
    for h in (state.get("experiments", {}) or {}).values():
        if int(h.get("version") or -1) == version:
            handle = h
            break
    if not isinstance(gate_res, dict):
        # 无门禁记录：有候选时现场算一次，否则拒绝
        if handle is not None:
            gate_res = _fallback_gate(cfg, dict(handle.get("metrics") or {}),
                                      _champion_metrics(ctx),
                                      _as_cost(handle.get("cost")))
        else:
            raise ToolError("gate_missing",
                            f"deploy v{version} 前没有门禁结果：先 evaluate_candidate")
    passed = bool(gate_res.get("passed"))
    if not passed and not force:
        raise ToolError("gate_rejected",
                        f"门禁未通过，拒绝上线 v{version}：{gate_res.get('reason', '')}",
                        {"gate": to_plain(gate_res), "version": version})
    playbook = str((handle or {}).get("playbook") or "")
    risk = _risk_of(cfg, playbook) if playbook else "medium"
    escalation = None
    if risk == "high" and not _ctx_explicitly_approved(ctx, f"deploy:{version}"):
        if not _ctx_auto_approve(ctx):
            raise ToolError("needs_approval",
                            f"高风险上线 v{version}（{playbook}）需要 human_gate 审批",
                            {"version": version, "playbook": playbook})
        escalation = _record_escalation(
            ctx, "deploy", {"version": version, "playbook": playbook,
                            "gate_decision": gate_res.get("decision"),
                            "mode": "record_and_proceed"})
    allowed_safety, checks = _safety_check(
        ctx, "deploy", {"version": version, "reason": reason,
                        "playbook": playbook}, state)
    if not allowed_safety:
        raise ToolError("safety_veto", f"安全护栏拒绝 deploy v{version}",
                        {"checks": checks, "version": version})
    previous = _current_champion_version(ctx)
    try:
        mv = registry.promote(version)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("deploy_failed",
                        f"ModelRegistry.promote({version}) 失败: {exc}") from exc
    new_version = int(getattr(mv, "version", version))
    state["champion_version"] = new_version
    state["champion_playbook"] = playbook
    if handle is not None:
        state["champion_model"] = handle.get("model")
        state["champion_metrics"] = dict(handle.get("metrics") or {})
    summary = {"version": new_version, "previous_version": previous,
               "playbook": playbook, "gate": gate_res,
               "reason": reason, "escalation": escalation}
    run_id = _tracker_start(ctx, kind="deploy", stage="evaluate",
                            name=f"deploy-v{new_version}",
                            config={"version": new_version, "playbook": playbook,
                                    "gate": gate_res, "reason": reason},
                            tags=["deploy", playbook or "unknown"])
    _tracker_finish(ctx, run_id, summary, status="finished")
    return {"ok": True, "version": new_version, "previous_version": previous,
            "playbook": playbook, "gate": to_plain(gate_res),
            "safety": to_plain(checks), "escalation": to_plain(escalation),
            "run_id": run_id, "cost": _cost_dict(Cost()),
            "metrics": (handle or {}).get("metrics", {}), "reason": reason}


def _current_champion_version(ctx: Any) -> Optional[int]:
    state = _ctx_state(ctx)
    if state.get("champion_version") is not None:
        return int(state["champion_version"])
    registry = _ctx_get(ctx, "registry")
    if registry is not None and hasattr(registry, "champion"):
        try:
            champ = registry.champion()
            mv = champ[0] if isinstance(champ, (tuple, list)) else champ
            return int(getattr(mv, "version"))
        except Exception:  # noqa: BLE001
            return None
    summary = _ctx_snapshot(ctx).get("registry_summary")
    if isinstance(summary, dict) and summary.get("champion_version") is not None:
        return int(summary["champion_version"])
    return None


def tool_rollback(args: dict, ctx: Any) -> dict:
    registry = _ctx_get(ctx, "registry")
    if registry is None or not hasattr(registry, "rollback"):
        raise ToolError("registry_unavailable", "未绑定 ModelRegistry（需 rollback 方法）")
    reason = str(args.get("reason") or "rollback")
    emergency = bool(args.get("emergency", False))
    to_version = args.get("to_version")
    history: list = []
    if hasattr(registry, "history"):
        try:
            history = list(registry.history() or [])
        except Exception as exc:  # noqa: BLE001
            raise ToolError("registry_history_failed", f"读取版本历史失败: {exc}") from exc
    current = _current_champion_version(ctx)
    if to_version is None:
        candidates = []
        for mv in history:
            v = int(getattr(mv, "version", -1))
            status = str(getattr(mv, "status", ""))
            if v < 0:
                continue
            if current is not None and v == current:
                continue
            if status in ("champion", "archived", "rolled_back", "candidate"):
                candidates.append(v)
        if not candidates:
            raise ToolError("rollback_unavailable",
                            "没有可回滚的历史版本（registry.history 为空或只有当前版本）")
        to_version = max(candidates)
    to_version = int(to_version)
    if current is not None and to_version == current:
        raise ToolError("rollback_noop", f"目标版本 v{to_version} 已是当前 champion")
    allowed_safety, checks = _safety_check(
        ctx, "rollback", {"to_version": to_version, "reason": reason,
                          "emergency": emergency}, _ctx_state(ctx))
    if not allowed_safety:
        raise ToolError("safety_veto", f"安全护栏拒绝 rollback v{to_version}",
                        {"checks": checks})
    try:
        mv = registry.rollback(to_version, reason)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("rollback_failed",
                        f"ModelRegistry.rollback({to_version}) 失败: {exc}") from exc
    new_version = int(getattr(mv, "version", to_version))
    state = _ctx_state(ctx)
    state["champion_version"] = new_version
    state["champion_metrics"] = dict(getattr(mv, "metrics", {}) or {})
    run_id = _tracker_start(ctx, kind="rollback", stage="rollback",
                            name=f"rollback-v{current}-to-v{new_version}",
                            config={"from_version": current, "to_version": new_version,
                                    "reason": reason, "emergency": emergency},
                            tags=["rollback", "emergency" if emergency else "sop"])
    summary = {"from_version": current, "to_version": new_version,
               "reason": reason, "emergency": emergency,
               "safety": to_plain(checks)}
    _tracker_finish(ctx, run_id, summary, status="finished")
    return {"ok": True, "to_version": new_version, "from_version": current,
            "reason": reason, "emergency": emergency, "run_id": run_id,
            "safety": to_plain(checks), "cost": _cost_dict(Cost()),
            "metrics": dict(getattr(mv, "metrics", {}) or {})}


def tool_request_labels(args: dict, ctx: Any) -> dict:
    oracle = _ctx_get(ctx, "oracle")
    if oracle is None or not hasattr(oracle, "query"):
        raise ToolError("oracle_unavailable", "未绑定 LabelOracle")
    n = args.get("n")
    if n is None:
        raise ToolError("invalid_args", "request_labels 需要 n（正整数）")
    n = int(n)
    if n <= 0:
        raise ToolError("invalid_args", f"n 必须为正整数，收到 {n}")
    reason = str(args.get("reason") or "harness_request")
    budget = _ctx_budget(ctx)
    if budget is not None and budget.labels_left() < n:
        raise ToolError("budget_exceeded",
                        f"标签预算不足：请求 {n}，剩余 {budget.labels_left()}",
                        {"requested": n, "labels_left": budget.labels_left(),
                         "budget": _budget_dict(budget, _phase_of(ctx), _ctx_cfg(ctx))})
    ids = args.get("ids")
    if ids is None:
        chunk = _ctx_current(ctx).get("chunk")
        if chunk is None or not hasattr(chunk, "ids"):
            raise ToolError("context_missing",
                            "未提供 ids 且 ctx.state 无 current chunk")
        ids = list(np.asarray(chunk.ids)[:n])
    ids_arr = np.asarray(list(ids), dtype=np.int64)[:n]
    if ids_arr.size == 0:
        raise ToolError("invalid_args", "ids 为空")
    phase = _phase_of(ctx, args)
    labels_before = int(budget.labels_used) if budget is not None else None
    try:
        res = oracle.query(ids_arr, reason)
    except Exception as exc:  # noqa: BLE001
        raise ToolError("label_query_failed",
                        f"LabelOracle.query 失败: {exc}") from exc
    cost = _as_cost(getattr(res, "cost", None))
    charged = _charge_cost(ctx, cost, phase, labels_before=labels_before)
    got_ids = np.asarray(getattr(res, "ids", ids_arr)).astype(int).tolist()
    got_y = np.asarray(getattr(res, "y", [])).astype(int).tolist()
    return {"n": len(got_ids), "ids": got_ids, "y": got_y,
            "reason": reason, "cost": _cost_dict(cost),
            "budget_charge": charged,
            "labels_left": (budget.labels_left() if budget is not None else None)}


def tool_list_registry(args: dict, ctx: Any) -> dict:
    registry = _ctx_get(ctx, "registry")
    include_archived = bool(args.get("include_archived", True))
    if registry is None:
        snap = _ctx_snapshot(ctx)
        summary = snap.get("registry_summary") or {}
        return {"champion": to_plain(summary), "versions": [], "count": 0,
                "note": "registry_unbound（返回 registry_summary）",
                "cost": _cost_dict(Cost())}
    history: list = []
    if hasattr(registry, "history"):
        try:
            history = list(registry.history() or [])
        except Exception as exc:  # noqa: BLE001
            raise ToolError("registry_history_failed", f"读取版本历史失败: {exc}") from exc
    champion = None
    champion_version = _current_champion_version(ctx)
    versions = []
    for mv in history:
        item = {
            "version": int(getattr(mv, "version", -1)),
            "kind": str(getattr(mv, "kind", "")),
            "playbook": str(getattr(mv, "playbook", "")),
            "status": str(getattr(mv, "status", "")),
            "created_t": int(getattr(mv, "created_t", 0)),
            "parent": getattr(mv, "parent", None),
            "metrics": to_plain(getattr(mv, "metrics", {}) or {}),
            "artifact": str(getattr(mv, "artifact", "") or ""),
            "note": str(getattr(mv, "note", "") or ""),
        }
        if item["version"] == champion_version:
            champion = item
        if include_archived or item["status"] in ("champion", "candidate", "shadow"):
            versions.append(item)
    if champion is None and versions:
        champion = max(versions, key=lambda v: v["version"])
    return {"champion": champion, "champion_version": champion_version,
            "versions": versions, "count": len(versions),
            "history_count": len(history), "cost": _cost_dict(Cost())}


def tool_query_history(args: dict, ctx: Any) -> dict:
    filters = dict(args.get("filters") or {})
    limit = int(args.get("limit", 20) or 20)
    filters.setdefault("limit", limit)
    filters.setdefault("order", "desc")
    tracker = _ctx_get(ctx, "tracker")
    runs: list = []
    if tracker is not None and hasattr(tracker, "query"):
        try:
            runs = list(tracker.query(filters) or [])
        except Exception as exc:  # noqa: BLE001
            raise ToolError("query_failed", f"tracker.query 失败: {exc}") from exc
    else:
        snap = _ctx_snapshot(ctx)
        history = snap.get("history") or _ctx_state(ctx).get("history") or []
        runs = [to_plain(r) for r in history][-limit:]
    compact = []
    for run in runs:
        if not isinstance(run, dict):
            compact.append({"value": to_plain(run)})
            continue
        compact.append({
            "run_id": run.get("run_id"), "group": run.get("group"),
            "name": run.get("name"), "kind": run.get("kind"),
            "stage": run.get("stage"), "status": run.get("status"),
            "t": run.get("t"), "phase": run.get("phase"),
            "created_at": run.get("created_at"),
            "summary": to_plain(run.get("summary") or {}),
            "swanlab_run_id": run.get("swanlab_run_id"),
            "swanlab_url": run.get("swanlab_url"),
        })
    return {"runs": compact, "count": len(compact), "filters": to_plain(filters),
            "cost": _cost_dict(Cost())}


def tool_shadow_eval(args: dict, ctx: Any) -> dict:
    handle = _find_experiment(ctx, args.get("exp_id"), args.get("version"))
    cur = _ctx_current(ctx)
    candidate = handle.get("model")
    champion = cur.get("champion_model") or _ctx_state(ctx).get("champion_model")
    chunk = cur.get("chunk")
    n = 0
    agreement: Optional[float] = None
    entropy_delta: Optional[float] = None
    if candidate is not None and chunk is not None and hasattr(chunk, "x"):
        x = np.asarray(chunk.x)
        try:
            pred_c = np.asarray(candidate.predict(x)).astype(int)
            if champion is not None and hasattr(champion, "predict"):
                pred_h = np.asarray(champion.predict(x)).astype(int)
                n = int(min(len(pred_c), len(pred_h)))
                agreement = float(np.mean(pred_c[:n] == pred_h[:n])) if n else None
            from ..core.metrics import mean_entropy
            if hasattr(candidate, "predict_proba"):
                ent_c = float(mean_entropy(candidate.predict_proba(x)))
                if champion is not None and hasattr(champion, "predict_proba"):
                    ent_h = float(mean_entropy(champion.predict_proba(x)))
                    entropy_delta = round(ent_c - ent_h, 6)
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "shadow_eval_failed", f"影子评估失败: {exc}")
    external = None
    evaluate_model = _ctx_get(ctx, "evaluate_model")
    if callable(evaluate_model):
        try:
            external = to_plain(evaluate_model(candidate, ctx, {"mode": "shadow",
                                                               "handle": handle}))
        except Exception as exc:  # noqa: BLE001
            _ctx_warn(ctx, "evaluate_model_failed", f"evaluate_model(shadow) 失败: {exc}")
    key = str(handle.get("exp_id"))
    result = {"exp_id": key, "version": handle.get("version"),
              "playbook": handle.get("playbook"), "n": n,
              "agreement": agreement, "entropy_delta": entropy_delta,
              "external": external,
              "verdict": ("一致率偏低，候选可能是换了一种错法"
                          if agreement is not None and agreement < 0.6
                          else "一致率在可评估区间，可进入正式门禁评估"),
              "cost": _cost_dict(Cost())}
    _ctx_state(ctx).setdefault("shadow_results", {})[key] = result
    return result


# --------------------------------------------------------------------------
# 工具规格（12 个）
# --------------------------------------------------------------------------
def _candidate_item_schema() -> dict:
    return {"type": "object",
            "properties": {"playbook": {"type": "string"},
                           "est_cost": {"type": "object"},
                           "prior": {"type": "number"},
                           "why": {"type": "string"}},
            "required": ["playbook"]}


DEFAULT_TOOLS: list[tuple[ToolSpec, Callable[[dict, Any], dict]]] = [
    (ToolSpec(
        name="inspect_drift",
        description="读取当前窗口的漂移监测报告（事件/统计量/受影响特征），判断是否需要干预。只读。",
        input_schema=_obj({
            "include_events": {"type": "boolean", "default": True,
                               "description": "是否返回事件明细"},
            "max_events": {"type": "integer", "minimum": 0, "maximum": 50,
                           "default": 12},
        }),
        risk="low", side_effects="none"), tool_inspect_drift),
    (ToolSpec(
        name="diagnose",
        description="对当前漂移做归因诊断（kind/severity/confidence/受影响特征/推荐 playbook）。"
                    "可能消耗少量标注预算。",
        input_schema=_obj({
            "force": {"type": "boolean", "default": False,
                      "description": "忽略缓存重新诊断"},
            "phase": {"type": "integer", "minimum": 0},
        }),
        risk="low", side_effects="budget"), tool_diagnose),
    (ToolSpec(
        name="budget_status",
        description="查看预算余额：labels/compute/tokens/updates 及当前 phase 剩余更新次数。只读。",
        input_schema=_obj({"phase": {"type": "integer", "minimum": 0}}),
        risk="low", side_effects="none"), tool_budget_status),
    (ToolSpec(
        name="propose_candidates",
        description="基于诊断与预算生成候选 playbook（含估算成本/先验/风险/是否可负担），"
                    "按 skills 规则排序。只读。",
        input_schema=_obj({
            "max_candidates": {"type": "integer", "minimum": 1, "maximum": 10},
            "phase": {"type": "integer", "minimum": 0},
        }),
        risk="low", side_effects="none"), tool_propose_candidates),
    (ToolSpec(
        name="run_experiment",
        description="执行一个候选 playbook 实验：预算/白名单/安全护栏前置校验 → 训练/更新 → "
                    "候选入库 → 全量记录（RunRecord/指标/成本）。写动作。",
        input_schema=_obj({
            "playbook": {"type": "string", "description": "候选 playbook 名（来自 candidates）"},
            "params": {"type": "object", "description": "playbook 参数覆盖"},
            "hypothesis": {"type": "string", "description": "实验假设（可证伪的一句话）"},
        }, required=["playbook"]),
        risk="medium", side_effects="budget"), tool_run_experiment),
    (ToolSpec(
        name="evaluate_candidate",
        description="按上线门禁口径评估候选（probe 增益/召回/成本/复现），返回 promote|shadow|reject。",
        input_schema=_obj({
            "exp_id": {"type": "string"},
            "version": {"type": "integer", "minimum": 1},
        }, additional=True),
        risk="medium", side_effects="none"), tool_evaluate_candidate),
    (ToolSpec(
        name="deploy",
        description="把通过门禁的候选版本提升为 champion（写模型仓库）。高危动作需 escalation 记录。",
        input_schema=_obj({
            "version": {"type": "integer", "minimum": 1},
            "reason": {"type": "string"},
            "force": {"type": "boolean", "default": False,
                      "description": "门禁未过时强制上线（仅人类/loop 可用，LLM 不应使用）"},
        }, required=["version"]),
        risk="high", side_effects="model"), tool_deploy),
    (ToolSpec(
        name="rollback",
        description="灰度/上线后掉点时回滚到历史稳定版本；kill-switch 下仅允许 emergency 回滚。",
        input_schema=_obj({
            "to_version": {"type": "integer", "minimum": 1,
                           "description": "目标版本；缺省时自动选最近非当前 champion"},
            "reason": {"type": "string"},
            "emergency": {"type": "boolean", "default": False},
        }),
        risk="high", side_effects="model"), tool_rollback),
    (ToolSpec(
        name="request_labels",
        description="按预算向 LabelOracle 请求标注（只返回预算内条数）。写预算动作。",
        input_schema=_obj({
            "n": {"type": "integer", "minimum": 1, "description": "请求条数"},
            "ids": {"type": "array", "items": {"type": "integer"},
                    "description": "可选：指定样本 id；缺省取当前窗口前 n 条"},
            "reason": {"type": "string"},
            "phase": {"type": "integer", "minimum": 0},
        }, required=["n"]),
        risk="low", side_effects="budget"), tool_request_labels),
    (ToolSpec(
        name="list_registry",
        description="列出模型仓库版本（champion/候选/历史）与元数据，用于回滚/审计。只读。",
        input_schema=_obj({
            "include_archived": {"type": "boolean", "default": True},
        }),
        risk="low", side_effects="none"), tool_list_registry),
    (ToolSpec(
        name="query_history",
        description="查询历史实验记录（按 group/kind/stage/phase/tag/时间过滤），用于复盘与对比。只读。",
        input_schema=_obj({
            "filters": {"type": "object",
                        "description": "group/kind/stage/phase/tag/since/until/status/run_id 等"},
            "limit": {"type": "integer", "minimum": 1, "maximum": 200, "default": 20},
        }),
        risk="low", side_effects="none"), tool_query_history),
    (ToolSpec(
        name="shadow_eval",
        description="对候选做影子评估（与 champion 的预测一致率/熵差），不上线、默认不耗标签。",
        input_schema=_obj({
            "exp_id": {"type": "string"},
            "version": {"type": "integer", "minimum": 1},
        }, additional=True),
        risk="low", side_effects="none"), tool_shadow_eval),
]


def build_tool_registry(ctx: Optional[ToolContext] = None,
                        cfg: Optional[dict] = None,
                        policy: Optional[ToolPolicy] = None) -> ToolRegistry:
    """构建含 12 个默认工具的注册表（loop 集成入口）。"""
    if ctx is None:
        ctx = DefaultToolContext(cfg=cfg or {})
    registry = ToolRegistry(ctx, policy=policy)
    for spec, handler in DEFAULT_TOOLS:
        registry.register(spec, handler)
    return registry


__all__ = [
    "ToolError", "ToolContext", "DefaultToolContext", "ToolPolicy",
    "ToolRegistry", "build_tool_registry", "build_tool_context",
    "validate_args", "apply_defaults", "to_plain",
    "tool_inspect_drift", "tool_diagnose", "tool_budget_status",
    "tool_propose_candidates", "tool_run_experiment", "tool_evaluate_candidate",
    "tool_deploy", "tool_rollback", "tool_request_labels", "tool_list_registry",
    "tool_query_history", "tool_shadow_eval", "DEFAULT_TOOLS",
]
