"""UCI AI4I 2020 预测性维护公开数据接入（真实工业数据补充验证）。

数据来源
--------
- 名称：AI4I 2020 Predictive Maintenance Dataset（UCI ML Repository, ID 601）
- 论文/说明：S. Matzka, "Explainable Artificial Intelligence for Predictive
  Maintenance: A Framework and a Case Study on a Milling Machine" (2020)
- 许可：CC BY 4.0（允许在注明出处的前提下使用、修改与再分发）
- 规模：10,000 条、14 列；5 个数值传感器（气温/过程温度/转速/扭矩/刀具磨损）、
  产品类型 L/M/H、机器故障标签 + 5 种失效模式（TWF/HDF/PWF/OSF/RNF）。
- 官方直链：
  https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip

为什么用它
----------
赛题校赛阶段无官方数据包，合成基准可控但"世界是造的"。AI4I 是真实公开的工业
预测性维护数据（铣床加工），用它可以验证：① 数据适配层/闭环代码在真实数据上能跑通；
② 刀具磨损累积（渐变）与转速-扭矩工况切换（协变量漂移）在真实数据上同样成立；
③ 真实数据的类不平衡（故障率 3.48%）会让静态模型在漂移后故障召回塌陷/误报膨胀
   ——这正是"自进化"要解决的问题（见 docs/09 §5）。

工况设计依据（实测）：AI4I 的 `Type`(L/M/H) 对 5 个传感器的分布几乎无影响
（各 Type 的温/速/扭矩/磨损均值差 <1%），只轻微影响故障先验；真正的协变量漂移在
**转速-扭矩-功率**方向：低转速(<1450rpm)重载工况故障率 7.9%，中转速 0.7%，
高转速 1.5%，且低转速段扭矩均值 49.7Nm vs 高转速 29.5Nm。因此工况段按转速区间
划分，`Type` 仅作为标签先验的次要因素（不做特征，避免 base 中恒为 0 的 one-hot
在漂移域外推造成伪影）。

转换为 Benchmark 的方式（与 src/data/synth.py 的 Benchmark 同构）
-----------------------------------------------------------------
特征（11 维，工程化）：
    air_temp, proc_temp, temp_diff(过程-气温), rpm, torque, tool_wear,
    power_kw(扭矩×角速度), torque_per_rpm, wear_ratio(磨损/253),
    wear_x_torque(磨损×扭矩/1000), temp_ratio(过程/气温)
标签（2 类）：机器故障 = 0 正常 / 1 故障（原始 `Machine failure`）。
漂移剧本（按转速分工况段 + 刀具磨损分磨损段，阶段内按磨损时间排序）：
    P0 基线延续(重载B,磨损80-140) → P1 深度磨损(重载B,磨损≥140)
    → P2 工况切换A(中转速,磨损≥80) → P3 工况切换C(高转速,全磨损)
base 训练/验证集来自**重载低转速工况 (rpm<1450) & 刀具磨损<80min**（初始健康工况）。
隐藏测试按阶段随机 34% 划分，与 synth 相同的"visible/hidden 物理隔离"约定。

复现性
------
数据集是**静态文件**，系统运行期不联网：
- 准备阶段（开发/评测前）运行 `python -m src.data.public_ai4i --download`；
- 若 `data/external/ai4i2020.csv` 不存在但 `ai4i2020.zip` 存在，则离线解压（无需网络）；
- 下载/解压都会把 URL、sha256、字节数写入 `data/external/download_manifest.json`。
"""

from __future__ import annotations

import hashlib
import io
import json
import time
import urllib.request
import zipfile
from pathlib import Path

import numpy as np

from ..core.spec import StreamChunk
from .synth import Benchmark

AI4I_CSV_NAME = "ai4i2020.csv"
AI4I_ZIP_NAME = "ai4i2020.zip"
README_NAME = "README.md"
DOWNLOAD_MANIFEST_NAME = "download_manifest.json"

# 主直链 + 等价镜像（直链失效时自动尝试；记录实际使用的 URL）
AI4I_SOURCES: list[tuple[str, str]] = [
    ("https://archive.ics.uci.edu/static/public/601/"
     "ai4i+2020+predictive+maintenance+dataset.zip", "zip"),
    ("https://archive.ics.uci.edu/ml/machine-learning-databases/00601/ai4i2020.csv", "csv"),
]

LICENSE = "CC BY 4.0"
LICENSE_URL = "https://creativecommons.org/licenses/by/4.0/"
DATASET_URL = "https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset"

# 原始列名（CSV 首列可能带 UTF-8 BOM，读取后由 pandas 处理）
RAW_COLUMNS = [
    "UDI", "Product ID", "Type", "Air temperature [K]", "Process temperature [K]",
    "Rotational speed [rpm]", "Torque [Nm]", "Tool wear [min]", "Machine failure",
    "TWF", "HDF", "PWF", "OSF", "RNF",
]

# Benchmark 剧本参数（静态基准，写入 meta，便于复现）
BASE_WEAR_MAX = 80.0
BASE_TRAIN_N = 1000
HIDDEN_RATIO = 0.34
CHUNK_SIZE = 200
# 工况段 = 转速区间（低转速=高扭矩重载工况；中/高转速=轻载工况）。
# 不用产品类型做工况：AI4I 中 Type 对特征分布几乎无影响（各 Type 的温/速/扭矩/磨损均值差 <1%），
# 只轻微影响故障先验；真正的协变量漂移在转速-扭矩-功率方向上。
COND_RPM = {"B_low_torque_high": (0.0, 1450.0), "A_mid": (1450.0, 1600.0),
            "C_high": (1600.0, 1e9)}
BASE_CONDITION = "B_low_torque_high"
PHASE_SCRIPT = [
    ("基线延续(重载B,磨损80-140)", "B_low_torque_high", 80.0, 140.0),
    ("深度磨损(重载B,磨损≥140)", "B_low_torque_high", 140.0, 1e9),
    ("工况切换A(中转速,磨损≥80)", "A_mid", 80.0, 1e9),
    ("工况切换C(高转速,全磨损)", "C_high", 0.0, 1e9),
]
FEATURE_NAMES = [
    "air_temp", "proc_temp", "temp_diff", "rpm", "torque", "tool_wear",
    "power_kw", "torque_per_rpm", "wear_ratio", "wear_x_torque", "temp_ratio",
]
CLASS_NAMES = ["正常", "故障"]


# --------------------------------------------------------------------------
# 数据准备（下载 / 离线解压）
# --------------------------------------------------------------------------
def _default_root() -> Path:
    from ..core.config import ROOT

    return ROOT / "data" / "external"


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def _http_get(url: str, timeout: float) -> bytes:
    req = urllib.request.Request(url, headers={"User-Agent": "zhixun-engine/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310 - 仅准备阶段
        return resp.read()


def _extract_csv_from_zip(data: bytes) -> bytes:
    with zipfile.ZipFile(io.BytesIO(data)) as zf:
        names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
        if not names:
            raise ValueError("zip 内不含 csv 文件")
        return zf.read(names[0])


def _validate_csv_bytes(data: bytes) -> None:
    head = data[:4096].decode("utf-8-sig", errors="replace").splitlines()
    if not head:
        raise ValueError("CSV 为空")
    cols = [c.strip() for c in head[0].split(",")]
    missing = [c for c in RAW_COLUMNS if c not in cols]
    if missing:
        raise ValueError(f"CSV 缺少列 {missing[:5]}，疑似不是 AI4I 2020 数据集")
    if len(head) < 2:
        raise ValueError("CSV 只有表头，数据不完整")


def _write_download_manifest(root: Path, url: str, source_kind: str, csv_bytes: bytes,
                             zip_sha: str | None) -> dict:
    manifest = {
        "dataset": "AI4I 2020 Predictive Maintenance Dataset (UCI ID 601)",
        "source_url": url,
        "source_kind": source_kind,
        "dataset_page": DATASET_URL,
        "license": LICENSE,
        "license_url": LICENSE_URL,
        "csv_file": AI4I_CSV_NAME,
        "csv_sha256": _sha256_bytes(csv_bytes),
        "csv_bytes": len(csv_bytes),
        "zip_file": AI4I_ZIP_NAME if zip_sha else None,
        "zip_sha256": zip_sha,
        "downloaded_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "note": ("数据为静态公开文件，已纳入仓库；系统运行期不联网。"
                 "若直链失效，请手动从 dataset_page 下载同名 zip/csv 放入本目录。"),
    }
    (root / DOWNLOAD_MANIFEST_NAME).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True), encoding="utf-8")
    return manifest


def ensure_ai4i(root: Path | str | None = None, force: bool = False,
                timeout: float = 60.0, verbose: bool = True) -> dict:
    """确保 AI4I 原始数据就绪（CSV 在 root 下），返回数据元信息。

    优先顺序：已有 CSV（除非 force）→ 已有 zip 离线解压 → 按 AI4I_SOURCES 下载。
    仅在**准备阶段**调用；闭环运行期只调用 `make_ai4i_benchmark`（不联网）。
    """
    root = Path(root) if root is not None else _default_root()
    root.mkdir(parents=True, exist_ok=True)
    csv_path = root / AI4I_CSV_NAME
    zip_path = root / AI4I_ZIP_NAME
    manifest_path = root / DOWNLOAD_MANIFEST_NAME

    if csv_path.is_file() and not force:
        info = {"ok": True, "csv": str(csv_path), "source": "existing_file",
                "sha256": _sha256_file(csv_path), "bytes": csv_path.stat().st_size,
                "download_manifest": str(manifest_path) if manifest_path.is_file() else None}
        if verbose:
            print(f"[ai4i] 已存在 {csv_path}（{info['bytes']} bytes）")
        return info

    # 1) 离线：本地 zip 解压
    if zip_path.is_file() and not force:
        data = zip_path.read_bytes()
        csv_bytes = _extract_csv_from_zip(data)
        _validate_csv_bytes(csv_bytes)
        csv_path.write_bytes(csv_bytes)
        info = {"ok": True, "csv": str(csv_path), "source": "local_zip",
                "sha256": _sha256_bytes(csv_bytes), "bytes": len(csv_bytes)}
        if not manifest_path.is_file():
            _write_download_manifest(root, "local_zip", "zip", csv_bytes,
                                     _sha256_bytes(data))
        if verbose:
            print(f"[ai4i] 从本地 zip 离线解压 -> {csv_path}")
        return info

    # 2) 联网下载（准备阶段，允许网络；闭环运行期禁止）
    last_err: Exception | None = None
    for url, kind in AI4I_SOURCES:
        try:
            if verbose:
                print(f"[ai4i] 下载 {url}")
            raw = _http_get(url, timeout)
            if kind == "zip":
                csv_bytes = _extract_csv_from_zip(raw)
                zip_path.write_bytes(raw)
                zip_sha = _sha256_bytes(raw)
            else:
                csv_bytes = raw
                zip_sha = None
            _validate_csv_bytes(csv_bytes)
            csv_path.write_bytes(csv_bytes)
            manifest = _write_download_manifest(root, url, kind, csv_bytes, zip_sha)
            if verbose:
                print(f"[ai4i] 完成：{csv_path}（{len(csv_bytes)} bytes，sha256="
                      f"{manifest['csv_sha256'][:16]}...）")
            return {"ok": True, "csv": str(csv_path), "source": "download",
                    "url": url, "sha256": manifest["csv_sha256"],
                    "bytes": len(csv_bytes)}
        except Exception as e:  # noqa: BLE001 - 依次尝试镜像
            last_err = e
            if verbose:
                print(f"[ai4i] 源失败（{url}）：{e}")
    raise RuntimeError(
        f"AI4I 数据准备失败（已尝试 {len(AI4I_SOURCES)} 个源，最后错误：{last_err}）。\n"
        f"请手动从 {DATASET_URL} 下载 'ai4i+2020+predictive+maintenance+dataset.zip'，\n"
        f"解压后把 ai4i2020.csv（或原 zip）放到 {root}/ ，再运行本函数/基线脚本。\n"
        f"离线环境可把 zip/csv 随仓库提交（约 0.5MB），运行期不联网。"
    )


# 语义别名：与 docs/09 中"下载脚本"的说法保持一致
download_ai4i = ensure_ai4i


def ai4i_status(root: Path | str | None = None) -> dict:
    """查询数据是否就绪（不触发下载，供自检/测试使用）。"""
    root = Path(root) if root is not None else _default_root()
    csv_path = root / AI4I_CSV_NAME
    zip_path = root / AI4I_ZIP_NAME
    return {
        "root": str(root),
        "csv_exists": csv_path.is_file(),
        "zip_exists": zip_path.is_file(),
        "csv": str(csv_path),
        "zip": str(zip_path),
        "download_manifest": str(root / DOWNLOAD_MANIFEST_NAME)
        if (root / DOWNLOAD_MANIFEST_NAME).is_file() else None,
    }


# --------------------------------------------------------------------------
# 加载与特征工程
# --------------------------------------------------------------------------
def load_ai4i_frame(root: Path | str | None = None):
    """读取原始 CSV 为 pandas DataFrame；缺数据时抛出带操作指引的 FileNotFoundError。"""
    root = Path(root) if root is not None else _default_root()
    csv_path = root / AI4I_CSV_NAME
    if not csv_path.is_file():
        if (root / AI4I_ZIP_NAME).is_file():
            ensure_ai4i(root)          # 离线解压
            csv_path = root / AI4I_CSV_NAME
        else:
            raise FileNotFoundError(
                f"未找到 AI4I 数据文件：{csv_path}\n"
                f"  准备方式（只需一次，允许联网）：\n"
                f"    python -m src.data.public_ai4i --download --root {root}\n"
                f"  或手动从 {DATASET_URL} 下载 zip/csv 放入 {root}/。\n"
                f"  闭环系统运行期禁止网络访问，因此数据必须在准备阶段落盘。"
            )
    try:
        import pandas as pd
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("读取 AI4I CSV 需要 pandas") from e
    df = pd.read_csv(csv_path)
    missing = [c for c in RAW_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"{csv_path} 缺少列 {missing[:5]}（疑似不是 AI4I 2020 数据集或被截断）"
        )
    if len(df) < 5000:
        raise ValueError(f"AI4I 数据行数异常：{len(df)}（完整数据集应为 10000 行）")
    return df


def build_features(df) -> np.ndarray:
    """把原始 14 列映射为 11 维工程化特征（见模块 docstring）。"""
    air = df["Air temperature [K]"].to_numpy(dtype=np.float64)
    proc = df["Process temperature [K]"].to_numpy(dtype=np.float64)
    rpm = df["Rotational speed [rpm]"].to_numpy(dtype=np.float64)
    torque = df["Torque [Nm]"].to_numpy(dtype=np.float64)
    wear = df["Tool wear [min]"].to_numpy(dtype=np.float64)
    power_kw = torque * rpm * 2.0 * np.pi / 60.0 / 1000.0
    x = np.column_stack([
        air, proc, proc - air, rpm, torque, wear,
        power_kw,
        torque / np.maximum(rpm, 1.0) * 1000.0,
        wear / 253.0,
        wear * torque / 1000.0,
        proc / air,
    ])
    return x


def _phase_masks(df) -> list[tuple[str, np.ndarray]]:
    wear = df["Tool wear [min]"].to_numpy(dtype=np.float64)
    rpm = df["Rotational speed [rpm]"].to_numpy(dtype=np.float64)
    masks = []
    for name, cond, lo, hi in PHASE_SCRIPT:
        rlo, rhi = COND_RPM[cond]
        masks.append((name, (rpm >= rlo) & (rpm < rhi) & (wear >= lo) & (wear < hi)))
    return masks


def make_ai4i_benchmark(cfg: dict | None = None, seed: int = 0,
                        root: Path | str | None = None) -> Benchmark:
    """把 AI4I 2020 转成与 synth.Benchmark 同构的时序漂移流（可复现）。

    参数
    ----
    cfg   : 配置字典（仅用于保持与 `make_benchmark(cfg, seed)` 同签名；当前不使用
            benchmark 段超参，剧本参数为模块常量并写入 meta）；
    seed  : 控制 base 训练/验证划分、每阶段 hidden/visible 划分与 visible 标注顺序；
    root  : 数据目录（默认 data/external）。
    """
    df = load_ai4i_frame(root)
    cfg = cfg or {}
    rng = np.random.default_rng(10_007 + int(seed))
    x_all = build_features(df)
    y_all = df["Machine failure"].to_numpy(dtype=np.int64)
    wear = df["Tool wear [min]"].to_numpy(dtype=np.float64)
    typ = df["Type"].astype(str).to_numpy()
    udi = df["UDI"].to_numpy(dtype=np.int64)

    # ---- base 训练/验证集：重载低转速工况、低磨损（初始健康工况）----
    _rlo, _rhi = COND_RPM[BASE_CONDITION]
    rpm = df["Rotational speed [rpm]"].to_numpy(dtype=np.float64)
    base_pool = np.flatnonzero((rpm >= _rlo) & (rpm < _rhi) & (wear < BASE_WEAR_MAX))
    if len(base_pool) < BASE_TRAIN_N + 100:
        raise ValueError(f"AI4I base 池样本不足：{len(base_pool)}（需 > {BASE_TRAIN_N + 100}）")
    perm = rng.permutation(len(base_pool))
    train_idx = base_pool[perm[:BASE_TRAIN_N]]
    val_idx = base_pool[perm[BASE_TRAIN_N:]]
    train_idx = train_idx[np.argsort(wear[train_idx], kind="stable")]
    base_x, base_y = x_all[train_idx].copy(), y_all[train_idx].copy()
    val_x, val_y = x_all[val_idx].copy(), y_all[val_idx].copy()

    # ---- 部署流：按剧本分阶段；阶段内按 (磨损, UDI) 时间排序 ----
    chunks: list[StreamChunk] = []
    visible_parts: list[np.ndarray] = []
    hidden_by_chunk: list = []
    phase_of_chunk: list[int] = []
    phase_names: list[str] = []
    phase_rows: list[int] = []
    phase_fail: list[int] = []
    next_id = 0
    for pidx, (name, mask) in enumerate(_phase_masks(df)):
        idx = np.flatnonzero(mask)
        if len(idx) < 200:
            raise ValueError(f"AI4I 阶段 {name} 样本过少：{len(idx)}")
        # 时间顺序：磨损累积（同磨损按 UDI）
        idx = idx[np.lexsort((udi[idx], wear[idx]))]
        n_hidden = int(round(len(idx) * HIDDEN_RATIO))
        # hidden/visible 随机划分（同分布），visible 再按磨损时间排序切窗口
        sel = rng.permutation(len(idx))
        hid_idx = idx[sel[:n_hidden]]
        vis_idx = idx[sel[n_hidden:]]
        vis_idx = vis_idx[np.lexsort((udi[vis_idx], wear[vis_idx]))]
        hidden_by_chunk.append((x_all[hid_idx].copy(), y_all[hid_idx].copy()))
        n_chunks = max(1, len(vis_idx) // CHUNK_SIZE)
        splits = np.array_split(np.arange(len(vis_idx)), n_chunks)
        visible_y_parts: list[np.ndarray] = []
        for ci, sl in enumerate(splits):
            chunk_idx = vis_idx[sl]
            ids = np.arange(next_id, next_id + len(chunk_idx), dtype=np.int64)
            next_id += len(chunk_idx)
            meta = {"source_udi": udi[chunk_idx].copy(),
                    "tool_wear": wear[chunk_idx].copy(),
                    "product_type": str(typ[idx[0]])}
            chunks.append(StreamChunk(t=len(chunks), phase=pidx, ids=ids,
                                      x=x_all[chunk_idx].copy(), meta=meta))
            phase_of_chunk.append(pidx)
            visible_y_parts.append(y_all[chunk_idx].copy())
        visible_parts.append(np.concatenate(visible_y_parts))
        phase_names.append(name)
        phase_rows.append(int(len(idx)))
        phase_fail.append(int(y_all[idx].sum()))

    visible_y = np.concatenate(visible_parts)

    # hidden 按阶段均分到该阶段各 chunk（与 synth 相同约定）
    hidden_per_chunk: list = [None] * len(chunks)
    for pidx in range(len(phase_names)):
        chunk_ids = [i for i, p in enumerate(phase_of_chunk) if p == pidx]
        if not chunk_ids:
            continue
        hx, hy = hidden_by_chunk[pidx]
        for ci, sl in zip(chunk_ids, np.array_split(np.arange(len(hx)), len(chunk_ids))):
            hidden_per_chunk[ci] = (hx[sl], hy[sl]) if len(sl) else (hx[:0], hy[:0])

    return Benchmark(
        name="ai4i_predictive_maintenance",
        d=len(FEATURE_NAMES),
        n_classes=len(CLASS_NAMES),
        base_classes=len(CLASS_NAMES),
        base_x=base_x, base_y=base_y,
        val_x=val_x, val_y=val_y,
        chunks=chunks,
        visible_y=visible_y,
        hidden_by_chunk=hidden_per_chunk,
        phase_of_chunk=phase_of_chunk,
        phase_names=phase_names,
        phase_specs=[],   # 真实数据无合成 PhaseSpec
        class_names=list(CLASS_NAMES),
        meta={
            "adapter": "ai4i",
            "source": "UCI AI4I 2020 Predictive Maintenance Dataset",
            "dataset_url": DATASET_URL,
            "license": LICENSE,
            "license_url": LICENSE_URL,
            "feature_names": list(FEATURE_NAMES),
            "label": "Machine failure (0=正常,1=故障; 故障率≈3.48%)",
            "base_condition": f"rpm in [0,1450) (重载工况) & Tool wear<{BASE_WEAR_MAX}min",
            "base_train_n": int(len(base_x)), "base_val_n": int(len(val_x)),
            "hidden_ratio": HIDDEN_RATIO, "chunk_size": CHUNK_SIZE,
            "phase_rows": phase_rows, "phase_failures": phase_fail,
            "generator_version": "ai4i-v1.0",
            "note": "真实公开数据补充验证；故障率低，macro-F1 对稀有类敏感，见 docs/09 §5",
        },
    )


def main(argv: list[str] | None = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(description="AI4I 2020 数据准备/自检")
    ap.add_argument("--download", action="store_true", help="下载并解压原始数据（联网）")
    ap.add_argument("--force", action="store_true", help="强制重新下载")
    ap.add_argument("--root", default=None, help="数据目录（默认 data/external）")
    ap.add_argument("--check", action="store_true", help="只检查数据是否就绪")
    args = ap.parse_args(argv)
    root = args.root
    if args.check:
        print(json.dumps(ai4i_status(root), ensure_ascii=False, indent=2))
        return 0
    if args.download or args.force:
        info = ensure_ai4i(root, force=args.force)
        print(json.dumps(info, ensure_ascii=False, indent=2))
        bench = make_ai4i_benchmark({}, seed=0, root=root)
        print(f"[ai4i] Benchmark 自检通过：{bench.describe()['n_chunks']} chunks × "
              f"{len(bench.chunks[0].x)} samples, d={bench.d}, "
              f"phases={bench.phase_names}")
        return 0
    print(__doc__)
    print("请加 --download 或 --check")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
