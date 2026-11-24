"""冻结接口契约（FROZEN CONTRACT）—— 智训引擎·工业模型自进化系统。

本文件是整个系统的唯一数据契约来源。所有模块（数据、引擎、Harness、
观测）都必须实现/消费这里声明的类型，不得私自更改已发布字段的语义。

设计原则
--------
1. 闭环的五个阶段各自有明确的输入/输出，阶段间通过本文件中的数据结构通信；
2. 所有决策必须携带证据（evidence）与成本（cost），以支撑审计与"单位增益消耗"核算；
3. 评估（hidden evaluation）与系统（agent）严格隔离：系统只能通过
   LabelOracle 按预算取标签，永远拿不到 hidden test 的标签。
"""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable, Iterable, Optional, Protocol, Sequence

import numpy as np

# --------------------------------------------------------------------------
# 全局常量
# --------------------------------------------------------------------------
SCHEMA_VERSION = "1.0.0"

# 闭环五阶段（顺序不可变，文档与审计日志共用这些标识）
STAGES = ("monitor", "diagnose", "optimize", "evaluate", "rollback")
# 受控 RSI 的两层：模型层自迭代 / 策略层自改进
LAYERS = ("model", "strategy")


def _np(obj: Any) -> Any:
    """把 numpy 类型递归转成 JSON 可序列化的原生类型。"""
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if isinstance(obj, (np.integer, int)):
        return int(obj)
    if isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    if isinstance(obj, dict):
        return {str(k): _np(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_np(v) for v in obj]
    return obj


class JsonMixin:
    """提供 to_dict()/to_json()，保证所有记录可写入实验平台（SwanLab/本地镜像）。"""

    def to_dict(self) -> dict:
        return _np(asdict(self))

    def to_json(self) -> str:
        import json

        return json.dumps(self.to_dict(), ensure_ascii=False, sort_keys=True)


# --------------------------------------------------------------------------
# 0. 预算与成本
# --------------------------------------------------------------------------
@dataclass
class Cost(JsonMixin):
    """一次操作/一个阶段/整个 run 的资源消耗。

    compute_units: 归一化算力单位。定义：1.0 unit = 在 base 训练集上
                   完成 1 个 epoch 的样本更新量（samples_epochs / base_n）。
    labels:        向 oracle 请求的标注条数。
    tokens:        驱动 Harness 的大模型 token 数（prompt+completion）。
    wall_s:        实测墙钟秒（CPU），仅用于可观测性，不参与打分。
    """

    compute_units: float = 0.0
    labels: int = 0
    tokens: int = 0
    wall_s: float = 0.0

    def add(self, other: "Cost") -> "Cost":
        return Cost(
            compute_units=self.compute_units + other.compute_units,
            labels=self.labels + other.labels,
            tokens=self.tokens + other.tokens,
            wall_s=self.wall_s + other.wall_s,
        )

    @property
    def scalar(self) -> float:
        """标量成本（用于排序）。权重可在运行时由 CostModel 覆写。"""
        return self.compute_units + 0.02 * self.labels + 1e-5 * self.tokens


@dataclass
class BudgetState(JsonMixin):
    """剩余预算。所有消耗型工具必须先检查该状态（安全护栏之一）。"""

    labels_total: int = 0
    labels_used: int = 0
    compute_total: float = 0.0
    compute_used: float = 0.0
    tokens_total: int = 0
    tokens_used: int = 0
    updates_used: int = 0
    updates_total: int = 0
    per_phase_updates_used: dict = field(default_factory=dict)

    # ---- 查询 ----
    def labels_left(self) -> int:
        return max(0, self.labels_total - self.labels_used)

    def compute_left(self) -> float:
        return max(0.0, self.compute_total - self.compute_used)

    def tokens_left(self) -> int:
        return max(0, self.tokens_total - self.tokens_used)

    def updates_left(self, phase: int, per_phase: int) -> int:
        return max(0, min(self.updates_total - self.updates_used,
                          per_phase - self.per_phase_updates_used.get(phase, 0)))

    # ---- 记账 ----
    def charge(self, cost: Cost, phase: Optional[int] = None) -> None:
        self.labels_used += int(cost.labels)
        self.compute_used += float(cost.compute_units)
        self.tokens_used += int(cost.tokens)
        if phase is not None and cost.compute_units > 0:
            self.updates_used += 1
            self.per_phase_updates_used[phase] = self.per_phase_updates_used.get(phase, 0) + 1

    def can_afford(self, cost: Cost, phase: Optional[int] = None,
                   per_phase: Optional[int] = None) -> bool:
        if cost.labels > self.labels_left():
            return False
        if cost.compute_units > self.compute_left() + 1e-9:
            return False
        if cost.tokens > self.tokens_left():
            return False
        if per_phase is not None and phase is not None:
            if self.updates_left(phase, per_phase) <= 0:
                return False
        return True


# --------------------------------------------------------------------------
# 1. 数据层
# --------------------------------------------------------------------------
@dataclass
class StreamChunk(JsonMixin):
    """部署流按窗口切片后交给系统的一个 chunk。**不含标签**。"""

    t: int                      # 全局窗口序号（0 开始）
    phase: int                  # 漂移阶段编号（数据生成器定义）
    ids: np.ndarray             # 全局样本 id，shape [n]
    x: np.ndarray               # 特征，shape [n, d]
    meta: dict = field(default_factory=dict)   # 时间戳、工况提示等（可为空）


@dataclass
class QueryResult(JsonMixin):
    """oracle 返回的标签。"""

    ids: np.ndarray
    y: np.ndarray
    cost: Cost
    reason: str = ""


class LabelOracle(Protocol):
    """标注预算的唯一入口。系统只能通过它获得部署流标签。"""

    def query(self, ids: np.ndarray, reason: str) -> QueryResult: ...
    def remaining(self) -> int: ...
    def budget_state(self) -> BudgetState: ...


@dataclass
class HiddenScore(JsonMixin):
    """平台侧（hidden test）对一个模型在一个窗口上的评分。系统不可见。"""

    t: int
    phase: int
    macro_f1: float
    accuracy: float
    per_class_recall: dict = field(default_factory=dict)
    n: int = 0


# --------------------------------------------------------------------------
# 2. 模型层
# --------------------------------------------------------------------------
class Predictor(Protocol):
    def predict_proba(self, x: np.ndarray) -> np.ndarray: ...
    def predict(self, x: np.ndarray) -> np.ndarray: ...
    def state(self) -> dict: ...


@dataclass
class ModelVersion(JsonMixin):
    """模型仓库中的一个不可变版本（champion/challenger 指针指向 version）。"""

    version: int
    created_t: int
    parent: Optional[int]
    kind: str                  # base|recalibrated|prior_fixed|finetuned|retrained|novel_extended|rolledback
    playbook: str
    config: dict = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)     # 候选评估指标（probe/proxy）
    status: str = "candidate"  # candidate|shadow|champion|archived|rolled_back
    artifact: str = ""         # 相对路径 experiments/artifacts/...
    cost: Cost = field(default_factory=Cost)
    lineage: list = field(default_factory=list)     # 生成该版本的 experiment_id 链
    note: str = ""


# --------------------------------------------------------------------------
# 3. 漂移监测 / 诊断归因
# --------------------------------------------------------------------------
@dataclass
class DriftEvent(JsonMixin):
    """监测器在某个窗口上发出的漂移信号（可多个并存）。"""

    t: int
    phase: int
    detector: str          # mmd|ks_topk|page_hinkley_proxy|probe_error|novelty
    statistic: float
    threshold: float
    p_value: Optional[float]
    severity: float        # 0~1，超过阈值的归一化强度
    features: list = field(default_factory=list)    # 受影响的特征下标（top-k）
    note: str = ""


@dataclass
class Diagnosis(JsonMixin):
    """诊断归因结果：漂移类型 + 影响范围 + 推荐 playbook 先验。"""

    t: int
    phase: int
    kind: str              # benign|covariate_affine|condition_switch|concept|novel_class|label_shift|noise_inflate|repair
    severity: float
    confidence: float
    affected_features: list = field(default_factory=list)
    affected_groups: list = field(default_factory=list)
    affected_classes: list = field(default_factory=list)
    prior_delta_estimate: list = field(default_factory=list)   # 估计的类别先验变化
    recommended: list = field(default_factory=list)  # [{"playbook":..., "prior": 0.6, "why": ...}]
    evidence: dict = field(default_factory=dict)     # 统计量、检验结果、特征排名等
    cost: Cost = field(default_factory=Cost)


# --------------------------------------------------------------------------
# 4. 实验优化（playbook / 候选实验）
# --------------------------------------------------------------------------
@dataclass
class PlaybookSpec(JsonMixin):
    """动作库中一个 playbook 的静态描述（成本模型 + 风险等级 + 适用条件）。"""

    name: str
    layer: str             # model|strategy
    description: str
    est_cost: Cost
    risk: str              # low|medium|high
    needs_labels: bool
    needs_compute: bool
    applicable_kinds: list = field(default_factory=list)
    contract: list = field(default_factory=list)   # 前置/后置条件说明


@dataclass
class Experiment(JsonMixin):
    """一次候选更新实验。id 用于关联 SwanLab run。"""

    exp_id: str
    run_id: str
    t: int
    phase: int
    playbook: str
    params: dict = field(default_factory=dict)
    hypothesis: str = ""
    driver: str = "harness"      # harness|deterministic_policy
    diagnosis_kind: str = ""
    cost_est: Cost = field(default_factory=Cost)
    cost_actual: Cost = field(default_factory=Cost)
    metrics: dict = field(default_factory=dict)
    stdout_tail: str = ""
    status: str = "running"      # running|succeeded|failed|skipped|reverted
    started_at: float = 0.0


@dataclass
class GateResult(JsonMixin):
    """上线门禁的逐项判定结果。"""

    criteria: dict = field(default_factory=dict)   # {name: {"value":.., "threshold":.., "pass": bool}}
    passed: bool = False
    decision: str = "reject"   # promote|shadow|reject
    reason: str = ""


@dataclass
class Decision(JsonMixin):
    """一步闭环的完整决策记录（审计核心对象）。"""

    t: int
    phase: int
    diagnosis: Optional[Diagnosis]
    chosen_playbook: str
    alternatives: list = field(default_factory=list)
    budget_before: dict = field(default_factory=dict)
    budget_after: dict = field(default_factory=dict)
    cost: Cost = field(default_factory=Cost)
    outcome: dict = field(default_factory=dict)    # 部署/回滚/拒绝 + 关键指标
    rationale: str = ""
    evidence_refs: list = field(default_factory=list)  # run_id / artifact 路径
    driver: str = "harness"
    safety: dict = field(default_factory=dict)     # 护栏检查结果


# --------------------------------------------------------------------------
# 5. 实验记录 / 可观测性
# --------------------------------------------------------------------------
@dataclass
class RunRecord(JsonMixin):
    """与 SwanLab run 一一对应的记录（同时写入本地镜像 SQLite）。"""

    run_id: str
    group: str                  # 实验组：main|baseline|ablation|safety|route
    name: str
    kind: str                   # drift|diagnosis|experiment|evaluation|deploy|rollback|loop|baseline
    stage: str                  # monitor|diagnose|optimize|evaluate|rollback|system
    config: dict = field(default_factory=dict)
    summary: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    t: int = 0
    phase: int = 0
    parent_run: Optional[str] = None
    artifacts: list = field(default_factory=list)
    created_at: float = 0.0


@dataclass
class MetricPoint(JsonMixin):
    run_id: str
    key: str
    step: int
    value: float
    t: int = 0


# --------------------------------------------------------------------------
# 6. Harness / 工具
# --------------------------------------------------------------------------
@dataclass
class ToolCall(JsonMixin):
    name: str
    args: dict = field(default_factory=dict)
    ok: bool = False
    result: dict = field(default_factory=dict)
    error: str = ""
    cost: Cost = field(default_factory=Cost)
    latency_s: float = 0.0


@dataclass
class ToolSpec(JsonMixin):
    name: str
    description: str
    input_schema: dict
    risk: str = "low"
    side_effects: str = "none"


@dataclass
class Trajectory(JsonMixin):
    """一次 Harness 决策回合的完整轨迹（用于策略层自改进 / 训练师模型微调）。"""

    t: int
    phase: int
    steps: list = field(default_factory=list)      # list[ToolCall]
    final_decision: Optional[Decision] = None
    tokens: int = 0
    wall_s: float = 0.0
    driver: str = "harness"
    outcome: dict = field(default_factory=dict)


# --------------------------------------------------------------------------
# 7. 工具函数
# --------------------------------------------------------------------------
def make_cost(compute_units: float = 0.0, labels: int = 0, tokens: int = 0,
              wall_s: float = 0.0) -> Cost:
    return Cost(compute_units=compute_units, labels=labels, tokens=tokens, wall_s=wall_s)


def chunk_to_arrays(chunk: StreamChunk) -> tuple[np.ndarray, np.ndarray]:
    return chunk.x, chunk.ids
