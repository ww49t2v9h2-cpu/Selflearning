"""合成工业漂移流基准（Synthetic Industrial Drift Stream）。

为什么需要它
------------
官方数据包（训练/验证/隐藏测试流）在校赛阶段尚未发放。为了在校赛阶段
就能训练并验证"漂移监测—诊断归因—实验优化—评估上线—回滚"全闭环，
本模块构造一个**可控、可复现、可替换**的工业故障诊断漂移基准：

- 潜变量 = 6 个特征频带（对应轴承特征频率 BPFO/BPFI/BSF 等）；
- 观测 = 8 路传感器 × 6 种统计量 = 48 维"预提取特征"（与官方"预提取特征"同形）；
- 4 个基础类别（正常/内圈/外圈/滚动体故障）+ 1 个"新故障模式"（novel）；
- 6 类真实工业漂移机制：渐变磨损、传感器增益老化、工况切换（潜空间旋转）、
  未知故障模式、标签先验漂移、维修复位。

接口契约：`make_benchmark(cfg, seed) -> Benchmark`，其中
- `chunks`        系统可见的部署流（只有 x，无标签）；
- `hidden_by_chunk` 平台侧隐藏测试标签（评测器专用，系统不可见）；
- `visible_y`     仅供 LabelOracle 按预算取标签。

可复现性设计（v1.1，2026-11 标定）
----------------------------------
- **世界固定**：传感器混合矩阵 A/C/W、噪声尺度、增益敏感度、工况旋转/偏置等
  “物理结构”只用 `WORLD_SEED` 生成，对所有 seed 相同；
  seed 只控制数据采样（标签抽样、观测噪声、hidden/visible 划分）与模型初始化。
  这样不同 seed 的难度差异只反映统计波动，多种子均值/方差可比。
- **磨损幅度**：`PhaseSpec.wear` 的潜空间位移方向为 `WEAR_DIR`，幅度经静态
  基线扫描标定，使 S1 深度磨损静态 macro-F1 降到 ~0.6（见 docs/09 难度表）。
- **增益老化**：`GAIN_NOISE_COUPLING=0.35`，让 S2 以“无标签仿射校准可恢复”的
  协变量漂移为主；噪声膨胀由 `PhaseSpec.noise` 单独控制（S2 P04/P05、S7）。

替换为官方数据时，只需实现同名字段的 adapter（见 docs/15）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np

from ..core.spec import StreamChunk

N_BANDS = 6
N_SENSORS = 8
STATS_PER_SENSOR = 6
D_FEATURES = N_SENSORS * STATS_PER_SENSOR

# 基准“物理世界”种子：传感器混合矩阵、频带响应、工况旋转/偏置等**结构性**随机量
# 固定为同一个标准世界，使不同 seed 的难度可比（seed 只改变数据采样与模型初始化）。
WORLD_SEED = 20261116

# 磨损潜空间位移方向。`PhaseSpec.wear` 为强度 τ，τ=1.0 时位移约 1 倍
# 类间判别距离的量级；幅度经过静态基线扫描标定（见 docs/09 难度表）。
WEAR_DIR = np.array([0.625, 0.25, 0.45, 0.55, 0.375, 0.25])

# 增益老化附加的噪声膨胀耦合系数（1.2=老化时噪声显著膨胀，使仿射校准不能完全恢复；
# 取 0.35 让 S2 以“可仿射恢复”为主，噪声膨胀作为独立机制留给 S2 P04/P05 与 S7）。
GAIN_NOISE_COUPLING = 0.35

# 基础类别
CLASS_NAMES_BASE = ["正常", "内圈故障", "外圈故障", "滚动体故障"]
CLASS_NAME_NOVEL = "复合新故障"

# 6 个频带上的类别"指纹"（模拟 BPFO/BPFI/BSF 特征频率能量）
CLASS_PROFILES = np.array([
    [0.30, 0.16, 0.12, 0.10, 0.08, 0.05],   # 正常
    [0.35, 1.60, 0.22, 0.10, 0.16, 0.08],   # 内圈
    [0.35, 0.20, 1.60, 0.18, 0.12, 0.08],   # 外圈
    [0.35, 0.18, 0.16, 1.50, 0.60, 0.10],   # 滚动体
], dtype=float)
NOVEL_PROFILE = np.array([0.30, 0.70, 0.65, 0.55, 0.90, 0.65], dtype=float)


def _orthogonal(rng: np.random.Generator, n: int) -> np.ndarray:
    q, r = np.linalg.qr(rng.normal(size=(n, n)))
    return q * np.sign(np.diag(r))[None, :]


@dataclass
class PhaseSpec:
    """一个漂移阶段的数据生成配置。"""

    name: str
    n: int = 1800
    cond: int = 0                 # 工况编号（0=基线，1/2=切换后）
    wear: float = 0.0             # 渐变磨损强度 τ
    gain: float = 0.0             # 传感器增益漂移强度（老化）
    noise: float = 0.0            # 噪声膨胀系数
    novel_frac: float = 0.0       # 未知故障模式占比
    prior: Optional[list] = None  # 4 类先验（None=均衡）
    rotate: bool = False          # 潜空间是否施加工况旋转（概念漂移）
    note: str = ""


# --------------------------------------------------------------------------
# 场景剧本（校赛可控实验用；对应 docs/09）
# --------------------------------------------------------------------------
def scenario_specs(name: str) -> list[PhaseSpec]:
    balanced = [0.25, 0.25, 0.25, 0.25]
    wear_prior = [0.25, 0.25, 0.25, 0.25]
    S: dict[str, list[PhaseSpec]] = {
        # S1 渐变磨损：磨损持续累积，无突变
        "S1_gradual_wear": [
            PhaseSpec("基线", wear=0.0, prior=balanced),
            PhaseSpec("轻度磨损", wear=0.25, prior=balanced),
            PhaseSpec("中度磨损", wear=0.50, prior=balanced),
            PhaseSpec("重度磨损", wear=0.75, prior=balanced),
            PhaseSpec("持续磨损", wear=0.95, prior=balanced),
            PhaseSpec("深度磨损", wear=1.10, prior=balanced),
        ],
        # S2 传感器增益老化：可被无标签仿射校准修复
        "S2_sensor_gain": [
            PhaseSpec("基线", gain=0.0, prior=balanced),
            PhaseSpec("轻微增益漂移", gain=0.25, prior=balanced),
            PhaseSpec("中度增益漂移", gain=0.50, prior=balanced),
            PhaseSpec("重度增益漂移", gain=0.80, prior=balanced),
            PhaseSpec("增益+噪声", gain=1.00, noise=0.25, prior=balanced),
            PhaseSpec("极端老化", gain=1.25, noise=0.40, prior=balanced),
        ],
        # S3 工况切换：潜空间旋转（概念漂移），需要带标签微调
        "S3_condition_switch": [
            PhaseSpec("工况A", cond=0, prior=balanced),
            PhaseSpec("切换到工况B", cond=1, rotate=True, prior=balanced,
                      note="负载/转速切换，特征关系变化"),
            PhaseSpec("工况B持续", cond=1, rotate=True, prior=balanced),
            PhaseSpec("切换到工况C", cond=2, rotate=True, prior=balanced),
            PhaseSpec("工况C持续", cond=2, rotate=True, prior=balanced),
            PhaseSpec("切回工况A", cond=0, rotate=False, prior=balanced),
        ],
        # S4 未知故障模式：新类别逐步出现
        "S4_novel_class": [
            PhaseSpec("基线", prior=balanced),
            PhaseSpec("零星新故障", novel_frac=0.05, prior=balanced),
            PhaseSpec("新故障增多", novel_frac=0.10, prior=balanced),
            PhaseSpec("新故障显著", novel_frac=0.18, prior=balanced),
            PhaseSpec("新故障主导", novel_frac=0.25, prior=balanced),
            PhaseSpec("回归常态", novel_frac=0.02, prior=balanced),
        ],
        # S5 维修复位：漂移后回到基线分布，检验回滚/回放记忆
        "S5_repair_cycle": [
            PhaseSpec("基线", prior=balanced),
            PhaseSpec("增益漂移", gain=0.50, prior=balanced),
            PhaseSpec("磨损+漂移", gain=0.70, wear=0.40, prior=balanced),
            PhaseSpec("维修复位", gain=0.0, wear=0.0, prior=balanced, note="更换轴承后分布回归基线"),
            PhaseSpec("复位后稳定", gain=0.0, wear=0.0, prior=balanced),
            PhaseSpec("二次磨损", wear=0.40, prior=balanced),
        ],
        # S6 标签先验漂移：特征不变，仅类别比例突变
        "S6_label_shift": [
            PhaseSpec("均衡", prior=balanced),
            PhaseSpec("正常主导", prior=[0.70, 0.10, 0.10, 0.10]),
            PhaseSpec("故障主导", prior=[0.10, 0.30, 0.30, 0.30]),
            PhaseSpec("内圈主导", prior=[0.10, 0.70, 0.10, 0.10]),
            PhaseSpec("外圈主导", prior=[0.10, 0.10, 0.70, 0.10]),
            PhaseSpec("恢复均衡", prior=balanced),
        ],
        # S7 综合漂移（主线实验）：六类机制混合，考察策略选择
        "S7_full_mix": [
            PhaseSpec("基线", prior=[0.25, 0.25, 0.25, 0.25]),
            PhaseSpec("渐变磨损", wear=0.35, prior=[0.25, 0.25, 0.25, 0.25]),
            PhaseSpec("传感器老化", gain=0.60, noise=0.20, wear=0.45, prior=[0.25, 0.25, 0.25, 0.25]),
            PhaseSpec("工况切换", cond=1, rotate=True, gain=0.60, prior=[0.25, 0.25, 0.25, 0.25],
                      note="工况B + 老化"),
            PhaseSpec("工况B稳定", cond=1, rotate=True, gain=0.70, wear=0.55,
                      prior=[0.18, 0.30, 0.30, 0.22]),
            PhaseSpec("新故障出现", cond=1, rotate=True, gain=0.70, wear=0.55, novel_frac=0.10,
                      prior=[0.18, 0.28, 0.28, 0.16]),
            PhaseSpec("新故障增多", cond=1, rotate=True, gain=0.60, wear=0.50, novel_frac=0.18,
                      prior=[0.15, 0.25, 0.25, 0.20]),
            PhaseSpec("维修复位", cond=0, rotate=False, gain=0.0, wear=0.0, novel_frac=0.0,
                      prior=[0.25, 0.25, 0.25, 0.25], note="维修后回到基线"),
            PhaseSpec("复位后漂移", gain=0.35, wear=0.20, prior=[0.30, 0.25, 0.25, 0.20]),
            PhaseSpec("再次老化", gain=0.75, wear=0.60, prior=[0.25, 0.25, 0.25, 0.25]),
            PhaseSpec("新故障再临", gain=0.75, wear=0.60, novel_frac=0.12,
                      prior=[0.22, 0.26, 0.26, 0.14]),
            PhaseSpec("深度漂移", cond=2, rotate=True, gain=0.80, wear=0.70, novel_frac=0.10,
                      prior=[0.20, 0.25, 0.25, 0.20]),
        ],
    }
    if name not in S:
        raise KeyError(f"未知场景 {name}，可选：{sorted(S)}")
    return S[name]


# --------------------------------------------------------------------------
# 基准对象
# --------------------------------------------------------------------------
@dataclass
class Benchmark:
    """一个可复现的工业漂移基准实例。"""

    name: str
    d: int
    n_classes: int                 # 评测类别数（含 novel=5）
    base_classes: int              # 初始模型输出数（4）
    base_x: np.ndarray
    base_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    chunks: list                   # list[StreamChunk]（仅 x）
    visible_y: np.ndarray          # 与 chunks 的 ids 对齐的标签（仅供 oracle）
    hidden_by_chunk: list          # list[(x, y)] 平台隐藏测试
    phase_of_chunk: list
    phase_names: list
    phase_specs: list
    class_names: list
    meta: dict = field(default_factory=dict)

    @property
    def n_chunks(self) -> int:
        return len(self.chunks)

    def chunk_labels_visible(self) -> np.ndarray:
        return self.visible_y

    def describe(self) -> dict:
        return {
            "name": self.name,
            "d": self.d,
            "n_classes": self.n_classes,
            "base_classes": self.base_classes,
            "n_chunks": self.n_chunks,
            "chunk_size": int(len(self.chunks[0].x)) if self.chunks else 0,
            "chunks_per_phase": [int(sum(1 for p in self.phase_of_chunk if p == i))
                                 for i in range(len(self.phase_names))],
            "phase_names": list(self.phase_names),
            "phases": [
                {"idx": i, "name": sp.name, "cond": sp.cond, "wear": sp.wear,
                 "gain": sp.gain, "noise": sp.noise, "novel_frac": sp.novel_frac,
                 "prior": sp.prior, "note": sp.note}
                for i, sp in enumerate(self.phase_specs)
            ],
            "class_names": self.class_names,
            "meta": dict(self.meta),
        }


class BenchmarkBuilder:
    """数据生成器：固定随机结构，按场景生成时间有序漂移流。"""

    def __init__(self, cfg: dict, seed: int = 0):
        self.cfg = cfg
        self.seed = int(seed)
        bcfg = cfg["benchmark"]
        self.scenario = bcfg.get("scenario", "S7_full_mix")
        self.phase_n = int(bcfg.get("phase_n", 1800))
        self.chunk_size = int(bcfg.get("chunk_size", 200))
        self.hidden_ratio = float(bcfg.get("hidden_ratio", 0.34))
        self.noise_scale = float(bcfg.get("noise_scale", 2.0))
        self.base_n = int(bcfg.get("base_train_n", 4000))
        self.val_n = int(bcfg.get("base_val_n", 1000))
        self.base_classes = int(bcfg.get("n_classes", 4))
        self.rot_alpha = float(bcfg.get("rot_alpha", 0.55))
        self.offset_scale = float(bcfg.get("offset_scale", 0.30))
        # 结构随机量只用 WORLD_SEED：不同 seed 共享同一“世界”，只改变采样。
        rng = np.random.default_rng(WORLD_SEED)
        self.rng = rng
        self._build_fixed_structure()

    # ------------------------------------------------------------------
    def _build_fixed_structure(self) -> None:
        rng = self.rng
        # 传感器敏感度 A: (8 传感器, 6 频带)
        A = np.zeros((N_SENSORS, N_BANDS))
        for s in range(4):          # 0-3 振动传感器：全频带敏感，结构不同
            A[s] = 1.0 + 0.35 * rng.normal(size=N_BANDS)
            A[s, (s + 1) % N_BANDS] *= 1.6
        A[4] = np.array([1.2, 0.9, 0.5, 0.3, 0.2, 0.1])   # 电流
        A[5] = np.array([1.1, 0.7, 0.6, 0.4, 0.3, 0.2])   # 电流
        A[6] = np.array([1.4, 0.25, 0.1, 0.05, 0.05, 0.02])  # 温度：低频敏感
        A[7] = np.array([0.6, 0.9, 1.0, 0.8, 0.7, 0.6])   # 声发射
        # 每个传感器内部的 6 种统计量混合矩阵 C: (6 统计量, 6 频带)
        C = rng.normal(size=(STATS_PER_SENSOR, N_BANDS)) * 0.5
        C = C + np.eye(STATS_PER_SENSOR, N_BANDS)
        # 特征权重矩阵 W: (48, 6)
        W = np.zeros((D_FEATURES, N_BANDS))
        for s in range(N_SENSORS):
            for j in range(STATS_PER_SENSOR):
                W[s * STATS_PER_SENSOR + j] = A[s] * C[j]
        # 噪声尺度：不同传感器噪声水平不同
        noise = np.array([0.22, 0.25, 0.24, 0.26, 0.18, 0.18, 0.30, 0.32])
        noise = np.repeat(noise, STATS_PER_SENSOR)
        # 增益漂移敏感度：温度/声发射/电流更容易老化
        gain_sens = np.array([0.35, 0.30, 0.30, 0.32, 0.65, 0.60, 1.00, 0.85])
        gain_sens = np.repeat(gain_sens, STATS_PER_SENSOR)
        # 磨损方向（潜空间），影响所有类别但幅度不同
        wear_dir = WEAR_DIR.copy()
        # 工况旋转矩阵（每个工况一个固定正交阵）
        rotations = {}
        offsets = {}
        # 工况切换的潜空间混合强度（1.0=完全旋转，0=不变）
        for c in range(3):
            if c == 0:
                rotations[c] = np.eye(N_BANDS)
                offsets[c] = np.zeros(N_BANDS)
            else:
                R = _orthogonal(rng, N_BANDS)
                rotations[c] = (1.0 - self.rot_alpha) * np.eye(N_BANDS) + self.rot_alpha * R
                offsets[c] = rng.normal(0, self.offset_scale, size=N_BANDS)
        self.W = W
        self.noise = noise
        self.gain_sens = gain_sens
        self.wear_dir = wear_dir
        self.rotations = rotations
        self.offsets = offsets
        self.profiles = CLASS_PROFILES
        self.novel_profile = NOVEL_PROFILE

    # ------------------------------------------------------------------
    def _latent(self, y: np.ndarray, phase: PhaseSpec, phase_idx: int) -> np.ndarray:
        z = np.empty((len(y), N_BANDS))
        base_mask = y < len(CLASS_PROFILES)
        z[base_mask] = CLASS_PROFILES[y[base_mask]]
        if (~base_mask).any():
            z[~base_mask] = self.novel_profile
        z = z + phase.wear * self.wear_dir[None, :]
        if phase.rotate or phase.cond != 0:
            R = self.rotations[phase.cond]
            z = z @ R.T + self.offsets[phase.cond][None, :]
        return z

    def _observe(self, z: np.ndarray, phase: PhaseSpec, rng) -> np.ndarray:
        x = z @ self.W.T
        # 传感器增益/偏置老化（标签保持的仿射变换）
        if phase.gain > 0:
            g = 1.0 + phase.gain * self.gain_sens
            b = phase.gain * 0.12 * self.gain_sens
            x = x * g[None, :] + b[None, :]
        # 噪声（随老化膨胀）
        scale = self.noise_scale * self.noise * (1.0 + GAIN_NOISE_COUPLING * phase.gain + phase.noise)
        x = x + rng.normal(size=x.shape) * scale[None, :]
        return x

    def _make_phase(self, phase: PhaseSpec, rng) -> tuple[np.ndarray, np.ndarray]:
        n = phase.n
        prior = np.array(phase.prior if phase.prior is not None
                         else [1.0 / self.base_classes] * self.base_classes, dtype=float)
        prior = prior / prior.sum()
        n_novel = int(round(n * phase.novel_frac)) if phase.novel_frac > 0 else 0
        n_base = n - n_novel
        y_base = rng.choice(self.base_classes, size=n_base, p=prior)
        y = np.concatenate([y_base, np.full(n_novel, self.base_classes, dtype=np.int64)])
        perm = rng.permutation(len(y))
        y = y[perm]
        z = self._latent(y, phase, 0)
        x = self._observe(z, phase, rng)
        return x, y

    # ------------------------------------------------------------------
    def build(self) -> Benchmark:
        rng = np.random.default_rng(self.seed * 104729 + 7)
        specs = scenario_specs(self.scenario)
        # base 训练/验证集：条件 0、无磨损、无增益、均衡先验
        base_spec = PhaseSpec("base", n=self.base_n, cond=0, wear=0.0, gain=0.0,
                              prior=[1 / self.base_classes] * self.base_classes)
        bx, by = self._make_phase(base_spec, rng)
        val_spec = PhaseSpec("val", n=self.val_n, cond=0, wear=0.0, gain=0.0,
                             prior=[1 / self.base_classes] * self.base_classes)
        vx, vy = self._make_phase(val_spec, rng)

        chunks: list[StreamChunk] = []
        visible_y_parts: list[np.ndarray] = []
        hidden_by_chunk: list = []
        phase_of_chunk: list[int] = []
        next_id = 0
        for pidx, phase in enumerate(specs):
            x, y = self._make_phase(phase, rng)
            n_hidden = int(round(len(x) * self.hidden_ratio))
            # hidden 与 visible 随机划分（同分布），hidden 标签物理隔离
            perm = rng.permutation(len(x))
            hid_idx, vis_idx = perm[:n_hidden], perm[n_hidden:]
            hidden_by_chunk.append((x[hid_idx], y[hid_idx]))
            vis_x, vis_y = x[vis_idx], y[vis_idx]
            n_chunks = len(vis_x) // self.chunk_size
            for ci in range(n_chunks):
                sl = slice(ci * self.chunk_size, (ci + 1) * self.chunk_size)
                ids = np.arange(next_id, next_id + len(vis_x[sl]), dtype=np.int64)
                next_id += len(vis_x[sl])
                chunks.append(StreamChunk(t=len(chunks), phase=pidx, ids=ids,
                                          x=vis_x[sl].copy(), meta={}))
                phase_of_chunk.append(pidx)
                visible_y_parts.append(vis_y[sl].copy())
        visible_y = np.concatenate(visible_y_parts)

        # 窗口级 hidden：把 phase hidden 均分给该 phase 的各窗口
        hidden_per_chunk: list = [None] * len(chunks)
        for pidx in range(len(specs)):
            hx, hy = hidden_by_chunk[pidx]
            chunk_ids = [i for i, p in enumerate(phase_of_chunk) if p == pidx]
            if not chunk_ids:
                continue
            split_idx = np.array_split(np.arange(len(hx)), len(chunk_ids))
            for ci, idxs in zip(chunk_ids, split_idx):
                if len(idxs) == 0:
                    hidden_per_chunk[ci] = (hx[:0], hy[:0])
                else:
                    hidden_per_chunk[ci] = (hx[idxs], hy[idxs])

        class_names = CLASS_NAMES_BASE + [CLASS_NAME_NOVEL]
        return Benchmark(
            name=f"synth_{self.scenario}",
            d=D_FEATURES,
            n_classes=len(class_names),
            base_classes=self.base_classes,
            base_x=bx.astype(np.float64), base_y=by.astype(np.int64),
            val_x=vx.astype(np.float64), val_y=vy.astype(np.int64),
            chunks=chunks,
            visible_y=visible_y,
            hidden_by_chunk=hidden_per_chunk,
            phase_of_chunk=phase_of_chunk,
            phase_names=[s.name for s in specs],
            phase_specs=specs,
            class_names=class_names,
            meta={"scenario": self.scenario, "seed": self.seed,
                  "world_seed": WORLD_SEED,
                  "base_classes": self.base_classes, "n_classes": len(class_names),
                  "chunk_size": self.chunk_size, "phase_n": self.phase_n,
                  "generator_version": "synth-v1.1"},
        )


def make_benchmark(cfg: dict, seed: int = 0) -> Benchmark:
    return BenchmarkBuilder(cfg, seed=seed).build()


# --------------------------------------------------------------------------
# 标注 oracle（隐藏测试标签不可达，只能按预算取 visible 标签）
# --------------------------------------------------------------------------
class SyntheticOracle:
    """按预算返回 visible 流标签；隐藏测试集标签物理隔离。"""

    def __init__(self, visible_y: np.ndarray, budget_state):
        self._y = np.asarray(visible_y, dtype=np.int64)
        self._budget = budget_state
        self.queries: list[dict] = []

    def query(self, ids: np.ndarray, reason: str = ""):
        from ..core.spec import Cost, QueryResult

        ids = np.asarray(ids, dtype=np.int64)
        ids = ids[(ids >= 0) & (ids < len(self._y))]
        # 去重
        ids = np.unique(ids)
        n = min(len(ids), self._budget.labels_left())
        ids = ids[:n]
        cost = Cost(labels=len(ids), compute_units=0.0, tokens=0,
                    wall_s=0.0)
        self._budget.charge(cost)
        self.queries.append({"ids": ids.tolist(), "reason": reason, "n": len(ids)})
        return QueryResult(ids=ids, y=self._y[ids], cost=cost, reason=reason)

    def remaining(self) -> int:
        return self._budget.labels_left()

    def budget_state(self):
        return self._budget
