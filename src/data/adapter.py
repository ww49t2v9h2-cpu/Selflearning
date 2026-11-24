"""数据适配层：统一 `DatasetAdapter` 协议 + 官方数据包适配 + 数据源选择器。

为什么需要适配层
----------------
`src/core/spec.py` 与 `src/data/synth.py` 冻结了 `Benchmark` 的数据契约；校赛阶段用
合成漂移基准跑通闭环，正式赛/测试阶段官方会发数据包（目录与文件格式见 docs/15）。
本模块把"数据从哪来"与"闭环代码"解耦：

- `DatasetAdapter`  协议：`load(root, cfg) -> Benchmark`；任何数据源只要实现它，
  闭环、基线、评测代码全部无需改动；
- `OfficialAdapter` 官方数据包（train/val/stream/hidden 四个 CSV + manifest.json）
  的加载、校验与错误提示；
- `VisibleLabelOracle` 通用标注 oracle（任何 `visible_y` 按 id 连续编号的 Benchmark）；
- `load_benchmark(cfg, seed)` 按 `cfg["benchmark"]["adapter"]` 选择 synth/official/ai4i。

官方数据包目录格式（与 docs/15 一致）
------------------------------------
```
<official_root>/
├── manifest.json        # 版本、类别、特征列名、阶段名、可选 sha256 校验和
├── base_train.csv       # 初始有标签训练集（特征列 + label）
├── base_val.csv         # 初始验证集
├── stream.csv           # 部署流：id, chunk, phase, <feature_columns...>（无标签）
├── stream_labels.csv    # visible 标注（id, label），供 LabelOracle 按预算取用
└── hidden.csv           # 平台隐藏测试：chunk, <feature_columns...>, label
```

安全约束：`hidden.csv` 只允许**评测平台**在读入 Benchmark 时加载；闭环系统侧只接收
`chunks` 与经 `LabelOracle` 发放的少量标签。`OfficialAdapter.load` 不向系统返回
隐藏标签，只填入 `Benchmark.hidden_by_chunk`（由 evaluator 消费）。

编号约定：加载后统一把 visible 样本重新编号为 `0..N-1`（按 chunk 时间顺序），
原始 id 保存在 `chunk.meta["source_ids"]`；这样 `visible_y[id]` 是该 id 的标签，
`VisibleLabelOracle`/`SyntheticOracle` 可直接使用。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np

from ..core.spec import BudgetState, StreamChunk
from .synth import Benchmark, SyntheticOracle, make_benchmark

MANIFEST_NAME = "manifest.json"
BASE_TRAIN_NAME = "base_train.csv"
BASE_VAL_NAME = "base_val.csv"
STREAM_NAME = "stream.csv"
STREAM_LABELS_NAME = "stream_labels.csv"
HIDDEN_NAME = "hidden.csv"

REQUIRED_MANIFEST_KEYS = [
    "schema_version", "name", "base_classes", "n_classes",
    "class_names", "phase_names", "feature_columns",
]
REQUIRED_FILES = [BASE_TRAIN_NAME, BASE_VAL_NAME, STREAM_NAME, HIDDEN_NAME]


# --------------------------------------------------------------------------
# 协议
# --------------------------------------------------------------------------
@runtime_checkable
class DatasetAdapter(Protocol):
    """所有数据源适配器必须实现的协议（docs/03 §2）。"""

    name: str

    def load(self, root: Path, cfg: dict) -> Benchmark:  # pragma: no cover - 协议
        ...

    def validate(self, root: Path) -> dict:  # pragma: no cover - 协议
        ...


# --------------------------------------------------------------------------
# 通用标注 oracle
# --------------------------------------------------------------------------
class VisibleLabelOracle(SyntheticOracle):
    """任何 `visible_y` 按 id 连续编号的 Benchmark 通用 oracle。

    与 `SyntheticOracle` 完全同一实现（同一预算语义），单独命名是为了让
    official / AI4I 等真实数据的调用点语义清晰。隐藏测试标签不可达。
    """

    def __init__(self, bench: Benchmark, labels_total: int = 10 ** 9):
        super().__init__(bench.visible_y, BudgetState(labels_total=int(labels_total)))
        self.bench = bench


# --------------------------------------------------------------------------
# 官方数据包适配器
# --------------------------------------------------------------------------
class OfficialAdapter:
    """官方数据包适配器。缺数据/格式不符时抛出**可行动**的清晰错误。"""

    name = "official"

    # ---------------- 校验 ----------------
    def validate(self, root: Path) -> dict:
        root = Path(root)
        problems: list[str] = []
        files: dict[str, bool] = {}
        if not root.is_dir():
            return {
                "ok": False,
                "root": str(root),
                "files": {},
                "problems": [f"官方数据包目录不存在：{root}"],
            }
        for fn in REQUIRED_FILES:
            files[fn] = (root / fn).is_file()
            if not files[fn]:
                problems.append(f"缺少文件：{fn}")
        manifest_path = root / MANIFEST_NAME
        files[MANIFEST_NAME] = manifest_path.is_file()
        manifest = None
        if not manifest_path.is_file():
            problems.append(f"缺少文件：{MANIFEST_NAME}")
        else:
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except json.JSONDecodeError as e:
                problems.append(f"{MANIFEST_NAME} 不是合法 JSON：{e}")
            else:
                for k in REQUIRED_MANIFEST_KEYS:
                    if k not in manifest:
                        problems.append(f"{MANIFEST_NAME} 缺少必需字段：{k}")
        stream_labels_present = (root / STREAM_LABELS_NAME).is_file()
        files[STREAM_LABELS_NAME] = stream_labels_present
        if not stream_labels_present:
            problems.append(f"缺少文件：{STREAM_LABELS_NAME}（标注 oracle 的数据来源）")
        checksum_ok = None
        if manifest and "checksums" in manifest:
            checksum_ok = True
            for fn, expected in manifest["checksums"].items():
                p = root / fn
                if not p.is_file():
                    checksum_ok = False
                    problems.append(f"checksum 引用的文件不存在：{fn}")
                elif _sha256(p) != str(expected).replace("sha256:", ""):
                    checksum_ok = False
                    problems.append(f"checksum 不匹配：{fn}")
        return {
            "ok": len(problems) == 0,
            "root": str(root),
            "files": files,
            "manifest": manifest,
            "checksum_ok": checksum_ok,
            "problems": problems,
        }

    # ---------------- 加载 ----------------
    def load(self, root: Path, cfg: dict | None = None) -> Benchmark:
        root = Path(root)
        cfg = cfg or {}
        report = self.validate(root)
        if not report["ok"]:
            hint = (
                f"官方数据包未就绪：{root}\n"
                f"  - 问题：{'; '.join(report['problems'])}\n"
                f"  - 目录格式见 docs/15《与官方数据包对接》；\n"
                f"  - 可先用 configs/official_template.json 作为配置骨架，"
                f"或改用合成基准（benchmark.adapter=synth）。"
            )
            raise FileNotFoundError(hint)
        manifest = report["manifest"]
        try:
            import pandas as pd
        except ImportError as e:  # pragma: no cover - 环境保证有 pandas
            raise RuntimeError("加载 CSV 需要 pandas（数据层允许依赖 numpy/pandas）") from e

        feature_cols = list(manifest["feature_columns"])
        label_col = str(manifest.get("label_column", "label"))
        chunk_col = str(manifest.get("chunk_column", "chunk"))
        id_col = str(manifest.get("id_column", "id"))
        n_classes = int(manifest["n_classes"])
        base_classes = int(manifest["base_classes"])
        class_names = list(manifest["class_names"])
        phase_names = list(manifest["phase_names"])
        d = len(feature_cols)
        if int(manifest.get("d_features", d)) != d:
            raise ValueError(
                f"manifest.d_features={manifest.get('d_features')} 与 feature_columns 长度 {d} 不一致"
            )
        if len(class_names) != n_classes:
            raise ValueError(f"class_names 长度 {len(class_names)} != n_classes {n_classes}")

        def _read(fn: str) -> "pd.DataFrame":
            path = root / fn
            try:
                return pd.read_csv(path)
            except Exception as e:  # 空文件 / 编码错误等
                raise ValueError(f"读取 {fn} 失败：{e}") from e

        def _check_cols(df, fn: str, cols: list[str], where: str) -> None:
            missing = [c for c in cols if c not in df.columns]
            if missing:
                raise ValueError(
                    f"{fn} 缺少{where}列 {missing[:8]}"
                    f"{' ...' if len(missing) > 8 else ''}（manifest.feature_columns 定义）"
                )

        # ---- base train / val ----
        base_train = _read(BASE_TRAIN_NAME)
        base_val = _read(BASE_VAL_NAME)
        _check_cols(base_train, BASE_TRAIN_NAME, feature_cols + [label_col], "特征/标签")
        _check_cols(base_val, BASE_VAL_NAME, feature_cols + [label_col], "特征/标签")
        base_x = base_train[feature_cols].to_numpy(dtype=np.float64)
        base_y = _as_labels(base_train[label_col], base_classes, BASE_TRAIN_NAME)
        val_x = base_val[feature_cols].to_numpy(dtype=np.float64)
        val_y = _as_labels(base_val[label_col], base_classes, BASE_VAL_NAME)
        if len(base_x) == 0 or len(val_x) == 0:
            raise ValueError("base_train.csv / base_val.csv 不能为空")

        # ---- stream（无标签）----
        stream = _read(STREAM_NAME)
        _check_cols(stream, STREAM_NAME, [id_col, chunk_col, "phase"] + feature_cols, "id/chunk/phase/特征")
        if stream[feature_cols].isna().to_numpy().any():
            raise ValueError("stream.csv 存在缺失特征值")
        # 标签列若意外混入 stream.csv 直接拒绝（防止标签泄漏给系统）
        if label_col in stream.columns:
            raise ValueError(
                f"stream.csv 不允许包含标签列 '{label_col}'（部署流必须无标签）；"
                f"请把 visible 标签放到 {STREAM_LABELS_NAME}"
            )
        # 分 chunk；chunk 号升序即为时间顺序
        chunk_groups = list(stream.groupby(chunk_col, sort=True))
        if not chunk_groups:
            raise ValueError("stream.csv 为空")
        chunks: list[StreamChunk] = []
        phase_of_chunk: list[int] = []
        visible_parts: list[np.ndarray] = []
        orig_id_parts: list[np.ndarray] = []
        next_id = 0
        for chunk_no, g in chunk_groups:
            g = g.sort_values(id_col, kind="stable")
            ids = np.arange(next_id, next_id + len(g), dtype=np.int64)
            next_id += len(g)
            phase = int(g["phase"].iloc[0])
            if not (0 <= phase < len(phase_names)):
                raise ValueError(
                    f"chunk={chunk_no} 的 phase={phase} 超出 phase_names（0..{len(phase_names)-1}）"
                )
            if g["phase"].nunique() != 1:
                raise ValueError(f"chunk={chunk_no} 内 phase 不唯一（每个 chunk 必须属于单一阶段）")
            chunks.append(StreamChunk(
                t=len(chunks), phase=phase, ids=ids,
                x=g[feature_cols].to_numpy(dtype=np.float64),
                meta={"source_ids": g[id_col].to_numpy(), "official_chunk": int(chunk_no)},
            ))
            phase_of_chunk.append(phase)
            orig_id_parts.append(g[id_col].to_numpy())
        orig_ids = np.concatenate(orig_id_parts)
        id_pos = {int(o): i for i, o in enumerate(orig_ids)}

        # ---- stream_labels（visible 标注）----
        labels_df = _read(STREAM_LABELS_NAME)
        _check_cols(labels_df, STREAM_LABELS_NAME, [id_col, label_col], "id/标签")
        visible_y = np.zeros(len(orig_ids), dtype=np.int64)
        seen = np.zeros(len(orig_ids), dtype=bool)
        for src_id, lab in zip(labels_df[id_col].to_numpy(), labels_df[label_col].to_numpy()):
            sid = int(src_id)
            if sid not in id_pos:
                raise ValueError(
                    f"{STREAM_LABELS_NAME} 中的 id={sid} 不在 stream.csv 内（可见标注必须属于部署流）"
                )
            pos = id_pos[sid]
            visible_y[pos] = _as_label_scalar(lab, n_classes, STREAM_LABELS_NAME)
            seen[pos] = True
        if not seen.any():
            raise ValueError(f"{STREAM_LABELS_NAME} 没有任何 visible 标注（LabelOracle 需要非空）")
        if len(labels_df) > len(orig_ids):
            raise ValueError(f"{STREAM_LABELS_NAME} 行数 {len(labels_df)} 超过部署流样本数 {len(orig_ids)}")

        # ---- hidden（平台隐藏测试；按 chunk 对齐）----
        hidden = _read(HIDDEN_NAME)
        _check_cols(hidden, HIDDEN_NAME, [chunk_col] + feature_cols + [label_col], "chunk/特征/标签")
        hidden_groups = dict(list(hidden.groupby(chunk_col, sort=False)))
        hidden_by_chunk: list[tuple[np.ndarray, np.ndarray]] = []
        for ch in chunks:
            official_chunk = ch.meta["official_chunk"]
            g = hidden_groups.get(official_chunk)
            if g is None:
                raise ValueError(
                    f"hidden.csv 缺少 chunk={official_chunk} 的隐藏测试行（每个部署 chunk 必须有对应评测集）"
                )
            hx = g[feature_cols].to_numpy(dtype=np.float64)
            hy = _as_labels(g[label_col], n_classes, HIDDEN_NAME)
            hidden_by_chunk.append((hx, hy))

        return Benchmark(
            name=str(manifest.get("name", "official")),
            d=d,
            n_classes=n_classes,
            base_classes=base_classes,
            base_x=base_x, base_y=base_y,
            val_x=val_x, val_y=val_y,
            chunks=chunks,
            visible_y=visible_y,
            hidden_by_chunk=hidden_by_chunk,
            phase_of_chunk=phase_of_chunk,
            phase_names=phase_names,
            phase_specs=[],   # 官方数据无合成 PhaseSpec
            class_names=class_names,
            meta={
                "adapter": "official",
                "root": str(root),
                "manifest": manifest,
                "source": "official_package",
            },
        )


# --------------------------------------------------------------------------
# 工具函数
# --------------------------------------------------------------------------
def _sha256(path: Path) -> str:
    import hashlib

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _as_labels(series, n_classes: int, where: str) -> np.ndarray:
    y = np.asarray(series).astype(np.int64)
    if len(y) and (y.min() < 0 or y.max() >= n_classes):
        raise ValueError(
            f"{where} 标签取值范围 [{y.min()}, {y.max()}] 超出 n_classes={n_classes}"
        )
    return y


def _as_label_scalar(value, n_classes: int, where: str) -> int:
    v = int(value)
    if not (0 <= v < n_classes):
        raise ValueError(f"{where} 标签 {v} 超出 n_classes={n_classes}")
    return v


# --------------------------------------------------------------------------
# 数据源选择器
# --------------------------------------------------------------------------
def resolve_root(cfg: dict, adapter: str) -> Path:
    """从配置解析数据根目录（相对路径按项目根目录解析）。"""
    from ..core.config import ROOT

    bcfg = cfg.get("benchmark", {})
    key = {"official": "official_root", "ai4i": "ai4i_root"}.get(adapter)
    if key is None:
        return ROOT
    raw = bcfg.get(key, f"data/{'official' if adapter == 'official' else 'external'}")
    p = Path(raw)
    return p if p.is_absolute() else ROOT / p


def load_benchmark(cfg: dict, seed: int = 0, root: Path | str | None = None) -> Benchmark:
    """按 `cfg["benchmark"]["adapter"]` 加载基准（synth|official|ai4i）。

    - synth：调用 `make_benchmark`（S1-S7 合成漂移流）；
    - official：`OfficialAdapter`，root 默认 `cfg["benchmark"]["official_root"]`；
    - ai4i：`make_ai4i_benchmark`（公开真实数据，见 src/data/public_ai4i.py）。
    """
    adapter = str(cfg.get("benchmark", {}).get("adapter", "synth"))
    if adapter == "synth":
        return make_benchmark(cfg, seed=seed)
    if adapter == "official":
        r = Path(root) if root is not None else resolve_root(cfg, "official")
        return OfficialAdapter().load(r, cfg)
    if adapter == "ai4i":
        from .public_ai4i import make_ai4i_benchmark

        r = Path(root) if root is not None else resolve_root(cfg, "ai4i")
        return make_ai4i_benchmark(cfg, seed=seed, root=r)
    raise ValueError(
        f"未知 benchmark.adapter={adapter!r}（可选：synth|official|ai4i）；"
        f"请检查配置或 src/data/adapter.py::load_benchmark"
    )


def make_official_oracle(bench: Benchmark, labels_total: int | None = None) -> VisibleLabelOracle:
    """为官方数据基准构造按预算发放 visible 标签的 oracle。"""
    if labels_total is None:
        labels_total = 10 ** 9
    return VisibleLabelOracle(bench, labels_total=int(labels_total))


__all__ = [
    "DatasetAdapter",
    "OfficialAdapter",
    "VisibleLabelOracle",
    "load_benchmark",
    "make_official_oracle",
    "resolve_root",
]
