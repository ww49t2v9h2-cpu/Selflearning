"""基线实验套件：B0 静态 / B1 周期全量重训 / B2 oracle 上界。

为什么要基线
-----------
赛题要求"闭环系统不低于静态基线"，同时要能证明"单位成本（算力/标注）的增益"。
本脚本给出三条可比参照曲线（全部在**隐藏测试**上评估，与系统可见数据严格隔离）：

- B0 static   部署后不更新（0 标注、0 适应算力）——"不进化"的下界；
- B1 periodic 每 `retrain_every` 个窗口用 `per_retrain_labels` 条新标注 + base 训练集
              从零全量重训（累计标注/算力全部记账）——"勤重训但不做实验设计"的基线；
- B2 oracle   上界：每个漂移阶段取 `oracle_labels_per_phase` 条标注，与 base 训练集
              合并后从零重训（本套件默认 600 条/阶段，与自进化方法的标注预算口径
              放在同一 Pareto 图上比较）。

产物（results/baselines/）
--------------------------
- `{scenario}.json`  该场景 × 全部 seed 的逐 chunk / 逐 phase 记录（自描述字段）；
- `summary.json`     跨场景汇总（均值/方差、标注/算力/墙钟总量）；
- `table.md`         可直接粘进 docs/09 的 Markdown 表（难度表 + B0/B1/B2 对比）；
- `figures_long.csv` 长表（scenario,seed,baseline,chunk,phase,f1,...）供画图；
- `figures_wide.csv` 宽表（scenario,seed,phase,static_f1,periodic_f1,oracle_f1,...）；
- `config_used.json` 运行配置快照（可复现）。

运行：
  PY=python3
  $PY scripts/baseline_suite.py                        # 全量 S1-S7 × 3 seeds
  $PY scripts/baseline_suite.py --scenarios S7_full_mix --seeds 7
  $PY scripts/baseline_suite.py --outdir /tmp/bl       # 自定义输出目录
  $PY scripts/baseline_suite.py --include-ai4i        # 附加真实公开数据 AI4I_maintenance 场景

字段约定（供 scripts/run_all.py 聚合）：逐 chunk 记录含
`scenario, seed, baseline, chunk, phase, phase_name, n_eval, macro_f1, accuracy,
labels_cum, compute_units_cum, compute_seconds_cum`；每个基线含 `totals`
（labels/compute_units/compute_seconds/train_calls）。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config                                    # noqa: E402
from src.core.metrics import area_under, evaluate                          # noqa: E402
from src.core.model import MLP, class_weights_from_labels                  # noqa: E402
from src.core.spec import BudgetState                                      # noqa: E402
from src.data.synth import SyntheticOracle, make_benchmark                 # noqa: E402

SCENARIOS = [
    "S1_gradual_wear",
    "S2_sensor_gain",
    "S3_condition_switch",
    "S4_novel_class",
    "S5_repair_cycle",
    "S6_label_shift",
    "S7_full_mix",
]

BASELINES = ["static", "periodic", "oracle"]

# B1 超参（写进 config_used.json，保证可复现）
PERIODIC_RETRAIN_EVERY = 3       # 每 3 个窗口重训一次
PERIODIC_LABELS = 60             # 每次重训请求的新标注条数
ORACLE_LABELS_PER_PHASE = 600    # B2：每个漂移阶段从零重训可用的标注条数
BIG_LABEL_BUDGET = 10 ** 9       # 基线自身不设总预算上限，但逐次记账


# --------------------------------------------------------------------------
# 基础工具
# --------------------------------------------------------------------------
def build_base_model(bench, cfg: dict, seed: int) -> tuple[MLP, dict]:
    """在 base 训练集上训练初始模型（基座训练不计入适应成本，见 docs/03 §3）。"""
    mcfg = cfg["model"]
    model = MLP(bench.d, bench.base_classes, hidden=mcfg["hidden"], seed=seed)
    t0 = time.time()
    res = model.fit(bench.base_x, bench.base_y, epochs=mcfg["epochs"], lr=mcfg["lr"],
                    batch=mcfg["batch"], seed=seed)
    wall = time.time() - t0
    val = evaluate(bench.val_y, model.predict(bench.val_x), bench.base_classes)
    info = {"wall_s": float(wall), "sample_updates": int(res.sample_updates),
            "val_macro_f1": float(val["macro_f1"]), "val_accuracy": float(val["accuracy"])}
    return model, info


def evaluate_chunks(model: MLP, bench) -> list[dict]:
    """在逐 chunk 的隐藏测试集上评估 macro-F1 / accuracy（评测类别数 = bench.n_classes）。"""
    rows = []
    for ci, chunk in enumerate(bench.chunks):
        hx, hy = bench.hidden_by_chunk[ci]
        pred = model.predict(hx)
        sc = evaluate(hy, pred, bench.n_classes)
        rows.append({
            "chunk": int(ci),
            "t": int(chunk.t),
            "phase": int(chunk.phase),
            "phase_name": str(bench.phase_names[chunk.phase]),
            "n_eval": int(len(hy)),
            "macro_f1": float(sc["macro_f1"]),
            "accuracy": float(sc["accuracy"]),
            "per_class_recall": {str(k): (None if np.isnan(v) else float(v))
                                 for k, v in sc["per_class_recall"].items()},
        })
    return rows


def _n_out(bench, labels: np.ndarray, default: int) -> int:
    """若标注里出现 novel 类别（>= base_classes），输出层需扩到 bench.n_classes。"""
    if len(labels) == 0:
        return default
    return int(max(default, int(np.max(labels)) + 1))


def _query_labels(oracle, ids: np.ndarray, reason: str) -> tuple[np.ndarray, np.ndarray]:
    """通过 LabelOracle 取标签，返回 (ids, y)。保持"系统只能经 oracle 拿标签"的契约。"""
    qr = oracle.query(np.asarray(ids, dtype=np.int64), reason)
    return np.asarray(qr.ids, dtype=np.int64), np.asarray(qr.y, dtype=np.int64)


def _chunk_index_map(chunk) -> dict:
    return {int(i): pos for pos, i in enumerate(chunk.ids)}


def _train_full(cfg: dict, bench, x: np.ndarray, y: np.ndarray, seed: int,
                n_out: int) -> tuple[MLP, float, int]:
    """从零全量重训：新模型 + 自适应算力（sample_updates/base_compute_unit）。"""
    mcfg = cfg["model"]
    model = MLP(bench.d, n_out, hidden=mcfg["hidden"], seed=seed)
    cw = class_weights_from_labels(y, n_out)
    t0 = time.time()
    res = model.fit(x, y, epochs=mcfg["epochs"], lr=mcfg["lr"], batch=mcfg["batch"],
                    class_weights=cw, seed=seed, refit_scaler=True)
    wall = time.time() - t0
    base_unit = float(cfg["cost_model"]["base_compute_unit"])
    compute_units = float(res.sample_updates) / max(base_unit, 1e-9)
    return model, wall, compute_units


def _summarize(rows: list[dict], bench, labels_used: int, compute_units: float,
               wall_s: float, train_calls: int) -> dict:
    """把逐 chunk 记录聚合成逐 phase + 总体指标。"""
    per_phase = []
    for p in range(len(bench.phase_names)):
        rs = [r for r in rows if r["phase"] == p]
        if not rs:
            continue
        per_phase.append({
            "phase": int(p),
            "phase_name": str(bench.phase_names[p]),
            "n_chunks": int(len(rs)),
            "f1": float(np.mean([r["macro_f1"] for r in rs])),
            "f1_std": float(np.std([r["macro_f1"] for r in rs])),
            "accuracy": float(np.mean([r["accuracy"] for r in rs])),
        })
    curve = [r["macro_f1"] for r in rows]
    return {
        "mean_f1": float(np.mean(curve)) if curve else 0.0,
        "std_f1_chunk": float(np.std(curve)) if curve else 0.0,
        "mean_phase_f1": float(np.mean([r["f1"] for r in per_phase])) if per_phase else 0.0,
        "area_under_f1": float(area_under(curve)),
        "per_phase": per_phase,
        "totals": {
            "labels": int(labels_used),
            "compute_units": float(compute_units),
            "compute_seconds": float(wall_s),
            "train_calls": int(train_calls),
        },
    }


# --------------------------------------------------------------------------
# B0 静态基线
# --------------------------------------------------------------------------
def run_static(bench, cfg: dict, seed: int, oracle) -> dict:
    model, info = build_base_model(bench, cfg, seed)
    rows = evaluate_chunks(model, bench)
    out = _summarize(rows, bench, labels_used=0, compute_units=0.0,
                     wall_s=info["wall_s"], train_calls=1)
    out.update({"name": "B0_static", "baseline": "static", "records": rows,
                "base_model": info})
    return out


# --------------------------------------------------------------------------
# B1 周期全量重训
# --------------------------------------------------------------------------
def run_periodic(bench, cfg: dict, seed: int, oracle,
                 retrain_every: int = PERIODIC_RETRAIN_EVERY,
                 labels_per_retrain: int = PERIODIC_LABELS) -> dict:
    model, base_info = build_base_model(bench, cfg, seed)
    buf_x: list[np.ndarray] = []
    buf_y: list[np.ndarray] = []
    labels_used = 0
    compute_units = 0.0
    wall_s = base_info["wall_s"]
    train_calls = 1
    rows: list[dict] = []
    cum: list[dict] = []   # 每 chunk 的累计成本（用于长表绘图）
    for ci, chunk in enumerate(bench.chunks):
        if ci % retrain_every == 0:
            rng = np.random.default_rng(seed * 100_003 + ci)
            n_sel = min(labels_per_retrain, len(chunk.ids))
            sel = rng.choice(len(chunk.ids), size=n_sel, replace=False)
            sel_ids, sel_y = _query_labels(oracle, chunk.ids[sel],
                                           reason=f"B1_periodic@chunk={ci}")
            if len(sel_ids) > 0:
                idx_map = _chunk_index_map(chunk)
                pos = np.array([idx_map[int(i)] for i in sel_ids], dtype=np.int64)
                buf_x.append(chunk.x[pos])
                buf_y.append(sel_y)
                labels_used += int(len(sel_ids))
            train_x = np.vstack([bench.base_x] + buf_x)
            train_y = np.concatenate([bench.base_y] + buf_y)
            n_out = _n_out(bench, train_y, bench.base_classes)
            model, w, cu = _train_full(cfg, bench, train_x, train_y, seed, n_out)
            wall_s += w
            compute_units += cu
            train_calls += 1
        # 逐 chunk 评估（重训后模型立即用于该 chunk）
        sc = evaluate_chunks_single(model, bench, ci)
        rows.append(sc)
        cum.append({"chunk": int(ci), "phase": int(chunk.phase),
                    "phase_name": str(bench.phase_names[chunk.phase]),
                    "labels_cum": int(labels_used),
                    "compute_units_cum": float(compute_units),
                    "compute_seconds_cum": float(wall_s)})
    out = _summarize(rows, bench, labels_used, compute_units, wall_s, train_calls)
    out.update({"name": "B1_periodic", "baseline": "periodic", "records": rows,
                "cost_curve": cum, "base_model": base_info,
                "hyper": {"retrain_every": int(retrain_every),
                          "labels_per_retrain": int(labels_per_retrain)}})
    return out


def evaluate_chunks_single(model: MLP, bench, ci: int) -> dict:
    chunk = bench.chunks[ci]
    hx, hy = bench.hidden_by_chunk[ci]
    pred = model.predict(hx)
    sc = evaluate(hy, pred, bench.n_classes)
    return {
        "chunk": int(ci), "t": int(chunk.t), "phase": int(chunk.phase),
        "phase_name": str(bench.phase_names[chunk.phase]), "n_eval": int(len(hy)),
        "macro_f1": float(sc["macro_f1"]), "accuracy": float(sc["accuracy"]),
        "per_class_recall": {str(k): (None if np.isnan(v) else float(v))
                             for k, v in sc["per_class_recall"].items()},
    }


# --------------------------------------------------------------------------
# B2 oracle 上界：每阶段 N 条标注 + base 从零重训
# --------------------------------------------------------------------------
def run_oracle(bench, cfg: dict, seed: int, oracle,
               labels_per_phase: int = ORACLE_LABELS_PER_PHASE) -> dict:
    model, base_info = build_base_model(bench, cfg, seed)
    labels_used = 0
    compute_units = 0.0
    wall_s = base_info["wall_s"]
    train_calls = 1
    rows: list[dict] = []
    cum: list[dict] = []
    for p in range(len(bench.phase_names)):
        chunk_ids = [ci for ci, c in enumerate(bench.chunks) if c.phase == p]
        if not chunk_ids:
            continue
        # 在该阶段内均匀取标注：按 chunk 平均分配（余数给前面的 chunk），
        # 保证标注总数 <= labels_per_phase 且覆盖整个阶段（口径可与其他方法比较）。
        rng = np.random.default_rng(seed * 100_003 + 7_000 + p)
        base_alloc = labels_per_phase // len(chunk_ids)
        extra = labels_per_phase - base_alloc * len(chunk_ids)
        xs: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        for k, ci in enumerate(chunk_ids):
            chunk = bench.chunks[ci]
            n_sel = min(base_alloc + (1 if k < extra else 0), len(chunk.ids))
            sel = rng.choice(len(chunk.ids), size=n_sel, replace=False)
            sel_ids, sel_y = _query_labels(oracle, chunk.ids[sel],
                                           reason=f"B2_oracle@phase={p}")
            if len(sel_ids) == 0:
                continue
            idx_map = _chunk_index_map(chunk)
            pos = np.array([idx_map[int(i)] for i in sel_ids], dtype=np.int64)
            xs.append(chunk.x[pos])
            ys.append(sel_y)
            labels_used += int(len(sel_ids))
        if not xs:
            continue
        x_aug = np.vstack([bench.base_x] + xs)
        y_aug = np.concatenate([bench.base_y] + ys)
        n_out = _n_out(bench, y_aug, bench.base_classes)
        model, w, cu = _train_full(cfg, bench, x_aug, y_aug, seed, n_out)
        wall_s += w
        compute_units += cu
        train_calls += 1
        for ci in chunk_ids:
            rows.append(evaluate_chunks_single(model, bench, ci))
            cum.append({"chunk": int(ci), "phase": int(p),
                        "phase_name": str(bench.phase_names[p]),
                        "labels_cum": int(labels_used),
                        "compute_units_cum": float(compute_units),
                        "compute_seconds_cum": float(wall_s)})
    rows.sort(key=lambda r: r["chunk"])
    cum.sort(key=lambda r: r["chunk"])
    out = _summarize(rows, bench, labels_used, compute_units, wall_s, train_calls)
    out.update({"name": "B2_oracle", "baseline": "oracle", "records": rows,
                "cost_curve": cum, "base_model": base_info,
                "hyper": {"labels_per_phase": int(labels_per_phase)}})
    return out


# --------------------------------------------------------------------------
# 场景级编排 / 汇总 / 落盘
# --------------------------------------------------------------------------
def run_scenario(scenario: str, seeds: list[int], cfg_base: dict,
                 periodic_kwargs: dict | None = None,
                 oracle_kwargs: dict | None = None,
                 bench_factory=None) -> dict:
    """跑一个场景的全部 seed × 三基线。

    bench_factory: 可选 (cfg, seed) -> Benchmark；默认 synth.make_benchmark。
    非 synth 场景（如 AI4I_maintenance）传入自己的 factory。
    """
    t0 = time.time()
    factory = bench_factory or (lambda cfg, seed: make_benchmark(cfg, seed=int(seed)))
    runs = []
    for seed in seeds:
        cfg = load_config(overrides={
            "benchmark": {"scenario": scenario},
            "seed": int(seed),
        })
        bench = factory(cfg, int(seed))
        oracle = SyntheticOracle(bench.visible_y, BudgetState(labels_total=BIG_LABEL_BUDGET))
        b0 = run_static(bench, cfg, int(seed), oracle)
        b1 = run_periodic(bench, cfg, int(seed), oracle, **(periodic_kwargs or {}))
        b2 = run_oracle(bench, cfg, int(seed), oracle, **(oracle_kwargs or {}))
        # 自描述字段：让 {scenario}.json 的逐条记录可直接被 run_all.py 聚合
        for name, bl in (("static", b0), ("periodic", b1), ("oracle", b2)):
            for rec in bl.get("records", []):
                rec["scenario"] = scenario
                rec["seed"] = int(seed)
                rec["baseline"] = name
                rec["name"] = name          # 别名：与 Lead 的 run_all 字段口径一致
                rec["f1"] = rec["macro_f1"]  # 别名：macro-F1
            for rec in bl.get("cost_curve", []):
                rec["scenario"] = scenario
                rec["seed"] = int(seed)
                rec["baseline"] = name
        runs.append({
            "seed": int(seed),
            "benchmark": bench.describe(),
            "baselines": {"static": b0, "periodic": b1, "oracle": b2},
        })
        print(f"    seed={seed}: static={b0['mean_f1']:.3f} "
              f"periodic={b1['mean_f1']:.3f} oracle={b2['mean_f1']:.3f} "
              f"(labels B1/B2={b1['totals']['labels']}/{b2['totals']['labels']})")
    desc = runs[0]["benchmark"]
    out = {
        "suite": "baseline_suite",
        "suite_version": "1.0",
        "scenario": scenario,
        "seeds": [int(s) for s in seeds],
        "n_phases": int(len(desc.get("phase_names", desc.get("phases", [])))),
        "phase_names": list(desc.get("phase_names", [])),
        "phase_specs": desc.get("phases", []),
        "baselines": BASELINES,
        "runs": runs,
        "cross_seed": _cross_seed(runs),
        "wall_s": float(time.time() - t0),
    }
    return out


def _cross_seed(runs: list[dict]) -> dict:
    """跨 seed 的均值/方差（逐 phase 与总体），供 docs 难度表与图表使用。"""
    out: dict = {}
    n_phase = len(runs[0]["benchmark"].get("phase_names", []))
    for name in BASELINES:
        curves = []
        for r in runs:
            per_phase = r["baselines"][name]["per_phase"]
            by_idx = {pp["phase"]: pp["f1"] for pp in per_phase}
            curves.append([by_idx.get(p, float("nan")) for p in range(n_phase)])
        arr = np.array(curves, dtype=float)
        out[name] = {
            "per_phase_mean": [float(v) for v in np.nanmean(arr, axis=0)],
            "per_phase_std": [float(v) for v in np.nanstd(arr, axis=0)],
            "mean_f1_mean": float(np.mean([r["baselines"][name]["mean_f1"] for r in runs])),
            "mean_f1_std": float(np.std([r["baselines"][name]["mean_f1"] for r in runs])),
            "mean_phase_f1_mean": float(np.mean([r["baselines"][name]["mean_phase_f1"]
                                                  for r in runs])),
            "labels_mean": float(np.mean([r["baselines"][name]["totals"]["labels"]
                                          for r in runs])),
            "compute_units_mean": float(np.mean([r["baselines"][name]["totals"]["compute_units"]
                                                 for r in runs])),
            "compute_seconds_mean": float(np.mean([r["baselines"][name]["totals"]["compute_seconds"]
                                                   for r in runs])),
        }
    return out


def write_long_csv(per_scenario: dict, path: Path) -> None:
    """长表：每行 = (scenario, seed, baseline, chunk)。自描述字段对齐 run_all 聚合。"""
    header = ("scenario,seed,baseline,chunk,phase,phase_name,n_eval,"
              "f1,accuracy,labels_cum,compute_units_cum,compute_seconds_cum,"
              "labels_total,compute_units_total,compute_seconds_total\n")
    lines = [header]
    for scenario, data in per_scenario.items():
        for run in data["runs"]:
            seed = run["seed"]
            for name in BASELINES:
                b = run["baselines"][name]
                cost = {c["chunk"]: c for c in b.get("cost_curve", [])}
                for rec in b["records"]:
                    c = cost.get(rec["chunk"], {})
                    lines.append(
                        f"{scenario},{seed},{name},{rec['chunk']},{rec['phase']},"
                        f"{rec['phase_name']},{rec['n_eval']},{rec['macro_f1']:.6f},"
                        f"{rec['accuracy']:.6f},"
                        f"{c.get('labels_cum', b['totals']['labels'] if name == 'static' else 0)},"
                        f"{c.get('compute_units_cum', b['totals']['compute_units']):.4f},"
                        f"{c.get('compute_seconds_cum', b['totals']['compute_seconds']):.3f},"
                        f"{b['totals']['labels']},{b['totals']['compute_units']:.4f},"
                        f"{b['totals']['compute_seconds']:.3f}\n")
    path.write_text("".join(lines), encoding="utf-8")


def write_wide_csv(per_scenario: dict, path: Path) -> None:
    """宽表：每行 = (scenario, seed, phase)，三基线逐 phase F1 并排，便于画曲线。"""
    header = ("scenario,seed,phase,phase_name,n_chunks,"
              "static_f1,periodic_f1,oracle_f1,static_acc,periodic_acc,oracle_acc,"
              "periodic_labels,oracle_labels\n")
    lines = [header]
    for scenario, data in per_scenario.items():
        for run in data["runs"]:
            seed = run["seed"]
            n_phase = len(run["benchmark"].get("phase_names", []))
            for p in range(n_phase):
                vals = {}
                for name in BASELINES:
                    pp = [x for x in run["baselines"][name]["per_phase"]
                          if x["phase"] == p]
                    vals[name] = pp[0] if pp else {"f1": float("nan"),
                                                   "accuracy": float("nan"),
                                                   "n_chunks": 0,
                                                   "phase_name": ""}
                pname = vals["static"]["phase_name"] or vals["periodic"].get("phase_name", "")
                lines.append(
                    f"{scenario},{seed},{p},{pname},{vals['static']['n_chunks']},"
                    f"{vals['static']['f1']:.6f},{vals['periodic']['f1']:.6f},"
                    f"{vals['oracle']['f1']:.6f},"
                    f"{vals['static']['accuracy']:.6f},{vals['periodic']['accuracy']:.6f},"
                    f"{vals['oracle']['accuracy']:.6f},"
                    f"{run['baselines']['periodic']['totals']['labels']},"
                    f"{run['baselines']['oracle']['totals']['labels']}\n")
    path.write_text("".join(lines), encoding="utf-8")


def build_tables(per_scenario: dict, summary: dict) -> str:
    """生成 Markdown：总体对比表 + 静态难度表（逐 phase，跨 seed 均值±方差）。"""
    lines = ["# 基线实验汇总（自动生成：scripts/baseline_suite.py）", ""]
    lines.append("## 表 1：三基线总体对比（跨 seed 均值 ± 标准差，hidden macro-F1）")
    lines.append("")
    lines.append("| 场景 | B0 static | B1 periodic | B2 oracle | 标签数 B1/B2 | 算力 B1/B2 |")
    lines.append("|---|---|---|---|---|---|")
    for scenario, data in per_scenario.items():
        cs = data["cross_seed"]
        lines.append(
            f"| {scenario} | {cs['static']['mean_f1_mean']:.3f} ± {cs['static']['mean_f1_std']:.3f} "
            f"| {cs['periodic']['mean_f1_mean']:.3f} ± {cs['periodic']['mean_f1_std']:.3f} "
            f"| {cs['oracle']['mean_f1_mean']:.3f} ± {cs['oracle']['mean_f1_std']:.3f} "
            f"| {cs['periodic']['labels_mean']:.0f} / {cs['oracle']['labels_mean']:.0f} "
            f"| {cs['periodic']['compute_units_mean']:.1f} / "
            f"{cs['oracle']['compute_units_mean']:.1f} |")
    lines.append("")
    lines.append("## 表 2：静态基线逐阶段难度表（B0，跨 seed 均值 ± 标准差）")
    lines.append("")
    max_phase = max(len(d["phase_names"]) for d in per_scenario.values())
    head = "| 场景 | " + " | ".join(f"P{i}" for i in range(max_phase)) + " | 均值 |"
    lines.append(head)
    lines.append("|" + "---|" * (max_phase + 2))
    for scenario, data in per_scenario.items():
        cs = data["cross_seed"]["static"]
        cells = []
        for i in range(max_phase):
            if i < len(cs["per_phase_mean"]):
                cells.append(f"{cs['per_phase_mean'][i]:.3f}±{cs['per_phase_std'][i]:.3f}")
            else:
                cells.append("—")
        lines.append(f"| {scenario} | " + " | ".join(cells)
                     + f" | {cs['mean_phase_f1_mean']:.3f} |")
    lines.append("")
    lines.append("## 表 3：B1/B2 逐阶段 F1（跨 seed 均值）")
    lines.append("")
    for scenario, data in per_scenario.items():
        cs = data["cross_seed"]
        lines.append(f"### {scenario}")
        lines.append("")
        lines.append("| 阶段 | " + " | ".join(str(p) for p in data["phase_names"]) + " |")
        lines.append("|" + "---|" * (len(data["phase_names"]) + 1))
        for name, label in (("periodic", "B1 periodic"), ("oracle", "B2 oracle")):
            vals = cs[name]["per_phase_mean"]
            lines.append(f"| {label} | " + " | ".join(f"{v:.3f}" for v in vals) + " |")
        lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="基线实验套件 B0/B1/B2 × S1-S7 × seeds")
    ap.add_argument("--scenarios", nargs="*", default=SCENARIOS,
                    help="场景列表（默认全部 S1-S7）")
    ap.add_argument("--seeds", nargs="*", type=int, default=[7, 17, 27],
                    help="随机种子（默认 7 17 27）")
    ap.add_argument("--outdir", default="results/baselines")
    ap.add_argument("--periodic-every", type=int, default=PERIODIC_RETRAIN_EVERY)
    ap.add_argument("--periodic-labels", type=int, default=PERIODIC_LABELS)
    ap.add_argument("--oracle-labels", type=int, default=ORACLE_LABELS_PER_PHASE)
    ap.add_argument("--include-ai4i", action="store_true",
                    help="附加真实公开数据场景 AI4I_maintenance（UCI AI4I 2020）")
    args = ap.parse_args(argv)

    scenario_jobs: list[tuple[str, object]] = [(s, None) for s in args.scenarios]
    if args.include_ai4i:
        from src.data.public_ai4i import make_ai4i_benchmark

        scenario_jobs.append((
            "AI4I_maintenance",
            lambda cfg, seed: make_ai4i_benchmark(cfg, seed=int(seed)),
        ))

    outdir = Path(args.outdir)
    if not outdir.is_absolute():
        outdir = ROOT / outdir
    outdir.mkdir(parents=True, exist_ok=True)

    t_all = time.time()
    per_scenario: dict = {}
    for scenario, factory in scenario_jobs:
        print(f"[baseline] {scenario} × seeds {args.seeds}")
        per_scenario[scenario] = run_scenario(
            scenario, list(args.seeds), load_config(),
            periodic_kwargs={"retrain_every": args.periodic_every,
                             "labels_per_retrain": args.periodic_labels},
            oracle_kwargs={"labels_per_phase": args.oracle_labels},
            bench_factory=factory,
        )
        with open(outdir / f"{scenario}.json", "w", encoding="utf-8") as f:
            json.dump(per_scenario[scenario], f, ensure_ascii=False, indent=2)

    wall = time.time() - t_all
    summary = {
        "suite": "baseline_suite", "suite_version": "1.0",
        "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "scenarios": list(per_scenario),
        "seeds": list(args.seeds),
        "hyper": {
            "periodic_retrain_every": args.periodic_every,
            "periodic_labels_per_retrain": args.periodic_labels,
            "oracle_labels_per_phase": args.oracle_labels,
        },
        "evaluation": "hidden test macro-F1 / accuracy，评测类别数=bench.n_classes",
        "results": {sc: data["cross_seed"] for sc, data in per_scenario.items()},
        "totals": {
            "wall_s": float(wall),
            "scenario_wall_seconds_sum": float(sum(d["wall_s"] for d in per_scenario.values())),
        },
    }
    with open(outdir / "summary.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    write_long_csv(per_scenario, outdir / "figures_long.csv")
    write_wide_csv(per_scenario, outdir / "figures_wide.csv")
    (outdir / "table.md").write_text(build_tables(per_scenario, summary), encoding="utf-8")
    cfg_snapshot = {
        "suite": "baseline_suite", "suite_version": "1.0",
        "default_config": load_config(),
        "cli": {"scenarios": [s for s, _ in scenario_jobs], "seeds": args.seeds,
                "periodic_every": args.periodic_every,
                "periodic_labels": args.periodic_labels,
                "oracle_labels": args.oracle_labels},
        "files": ["table.md", "summary.json", "figures_long.csv",
                  "figures_wide.csv", "{scenario}.json"],
    }
    with open(outdir / "config_used.json", "w", encoding="utf-8") as f:
        json.dump(cfg_snapshot, f, ensure_ascii=False, indent=2, sort_keys=True)

    # 控制台总结
    print("\n[baseline] 汇总（跨 seed 均值，hidden macro-F1）：")
    print(f"{'scenario':<22s} {'static':>8s} {'periodic':>9s} {'oracle':>8s} "
          f"{'labels(B1/B2)':>14s}")
    for scenario, data in per_scenario.items():
        cs = data["cross_seed"]
        print(f"{scenario:<22s} {cs['static']['mean_f1_mean']:8.3f} "
              f"{cs['periodic']['mean_f1_mean']:9.3f} {cs['oracle']['mean_f1_mean']:8.3f} "
              f"{cs['periodic']['labels_mean']:6.0f}/{cs['oracle']['labels_mean']:<7.0f}")
    print(f"\n[baseline] 产物目录：{outdir}  （总墙钟 {wall:.1f}s）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
