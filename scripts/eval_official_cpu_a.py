"""在官方 CPU-A（气体传感器漂移 6 类）数据上跑自进化闭环。

协议：
  - 基座训练：官方 public/train.jsonl（batch1，445 样本，128 维，6 类）；
  - 基座验证：官方 public/validation.jsonl（batch2，1244 样本）；
  - 部署流：UCI 原始 gas sensor batches 3–10（同一数据集的后续批次，按时间顺序），
    每批 70% 作为可见流（oracle 可控标签），30% 作为隐藏集（只有评测器可见）；
  - 每批一个漂移阶段；静态基线 = 基座模型不再更新；自进化系统 = 我们的闭环。
输出：results/official/cpu_a_evolving/seed{7,17,27}.json + summary.md

用法：PY=.../python3
      $PY scripts/eval_official_cpu_a.py
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import RunContext, load_config  # noqa: E402
from src.core.spec import StreamChunk                     # noqa: E402
from src.data.synth import Benchmark, PhaseSpec          # noqa: E402
from src.loop import SelfEvolvingLoop, new_run_id         # noqa: E402

OFF = ROOT / "data" / "official" / "cpu-a-guide-20261110-v1"
RAW = ROOT / "data" / "external" / "gas_sensor_drift" / "Dataset"
OUT = ROOT / "results" / "official" / "cpu_a_evolving"
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def load_jsonl(path: Path):
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        xs.append(np.asarray(d["inputs"]["features"], dtype=float))
        ys.append(CLASSES.index(d["answer"]))
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def load_batch(path: Path):
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        y = int(parts[0]) - 1
        vec = np.zeros(128, dtype=float)
        for tok in parts[1:]:
            k, v = tok.split(":")
            vec[int(k) - 1] = float(v)
        xs.append(vec)
        ys.append(y)
    return np.vstack(xs), np.asarray(ys, dtype=np.int64)


def build_benchmark(seed: int = 0) -> Benchmark:
    base_x, base_y = load_jsonl(OFF / "public" / "train.jsonl")
    val_x, val_y = load_jsonl(OFF / "public" / "validation.jsonl")
    chunks, hidden, phase_of, phase_names, specs = [], [], [], [], []
    visible_y_parts, ids = [], []
    next_id = 0
    for batch in range(3, 11):
        x, y = load_batch(RAW / f"batch{batch}.dat")
        n = len(x)
        cut = int(n * 0.70)
        vis_x, vis_y = x[:cut], y[:cut]
        hid_x, hid_y = x[cut:], y[cut:]
        phase = batch - 3
        phase_names.append(f"batch{batch}")
        specs.append(PhaseSpec(name=f"batch{batch}", n=cut, note="UCI gas batch（时间顺序，70% 可见/30% 隐藏）"))
        # 可见流按窗口切 chunk
        step = 64
        for start in range(0, cut, step):
            sx, sy = vis_x[start:start + step], vis_y[start:start + step]
            if len(sx) == 0:
                continue
            cid = np.arange(next_id, next_id + len(sx))
            next_id += len(sx)
            visible_y_parts.append(sy)
            chunks.append(StreamChunk(t=len(chunks), phase=phase, ids=cid, x=sx,
                                     meta={"batch": batch, "visible": True}))
            hidden.append((hid_x, hid_y))
            phase_of.append(phase)
    visible_y = np.concatenate(visible_y_parts) if visible_y_parts else np.zeros(0, dtype=np.int64)
    return Benchmark(
        name="official_cpu_a_gas_drift", d=128, n_classes=6, base_classes=6,
        base_x=base_x, base_y=base_y, val_x=val_x, val_y=val_y,
        chunks=chunks, visible_y=visible_y, hidden_by_chunk=hidden,
        phase_of_chunk=phase_of, phase_names=phase_names, phase_specs=specs,
        class_names=list(CLASSES),
        meta={"source": "UCI Gas Sensor Array Drift #224 + official CPU-A public split",
              "base": "official public/train.jsonl (batch1)", "val": "official public/validation.jsonl (batch2)",
              "stream": "raw batches 3-10, 70% visible / 30% hidden", "n_visible": int(len(visible_y))},
    )


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    summary = {"benchmark": "official CPU-A gas drift", "seeds": {}}
    seeds = [7, 17, 27]
    for seed in seeds:
        cfg = copy.deepcopy(load_config("configs/main.json"))
        cfg["seed"] = seed
        cfg["run"]["name"] = "official-cpu-a"
        cfg["budget"]["labels_total"] = 260
        cfg["budget"]["compute_total"] = 260.0
        # 128 维原始特征：适度放宽监测阈值，避免每窗口误触发
        cfg.setdefault("engine", {}).setdefault("monitor", {}).update({
            "mmd_threshold": 0.55, "ks_count_threshold": 10,
            "novelty_threshold": 0.7, "probe_every": 2,
        })
        cfg["data"] = {"sanitize": True, "clip_quantile": 0.9995, "fill": "median"}
        bench = build_benchmark(seed)
        ctx = RunContext(cfg=cfg, run_id=new_run_id(f"official-cpu-a-s{seed}"))
        loop = SelfEvolvingLoop(cfg, ctx, benchmark=bench)
        res = loop.run()
        path = OUT / f"seed{seed}.json"
        path.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        comp = res.get("comparison", {})
        summary["seeds"][str(seed)] = {
            "method_macro_f1": res["hidden"]["stream_macro_f1"],
            "static_macro_f1": res["static_baseline_hidden"]["stream_macro_f1"],
            "gain_points": comp.get("stream_gain_points"),
            "labels": res["cost"]["labels"], "compute_units": res["cost"]["compute_units"],
            "deploys": sum(1 for x in res["timeline"] if x.get("deploy")),
            "rollbacks": sum(1 for x in res["timeline"] if x.get("rollback")),
            "n_chunks": len(bench.chunks),
        }
        print(f"seed{seed}: method={res['hidden']['stream_macro_f1']:.4f} "
              f"static={res['static_baseline_hidden']['stream_macro_f1']:.4f} "
              f"gain={comp.get('stream_gain_points'):+.2f} labels={res['cost']['labels']:.0f}")

    vals = list(summary["seeds"].values())
    summary["mean"] = {
        "method_macro_f1": float(np.mean([v["method_macro_f1"] for v in vals])),
        "static_macro_f1": float(np.mean([v["static_macro_f1"] for v in vals])),
        "gain_points": float(np.mean([v["gain_points"] for v in vals])),
        "labels": float(np.mean([v["labels"] for v in vals])),
        "compute_units": float(np.mean([v["compute_units"] for v in vals])),
    }
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    m = summary["mean"]
    lines = ["# 官方 CPU-A 数据上的自进化闭环（气体传感器漂移）", "",
             f"- 基座：官方 public/train.jsonl（batch1）；部署流：原始 batches 3–10（70% 可见 / 30% 隐藏）",
             f"- 结果（3 seeds 7/17/27）：macro-F1 **{m['static_macro_f1']:.4f}（静态）→ {m['method_macro_f1']:.4f}（自进化）**，"
             f"**{m['gain_points']:+.2f}pt**；成本：{m['labels']:.0f} 标注 / {m['compute_units']:.1f} 算力单位",
             "", "| seed | 静态 | 自进化 | 增益(pt) | 标注 | 部署/回滚 |", "|---|---|---|---|---|---|"]
    for s, v in summary["seeds"].items():
        lines.append(f"| {s} | {v['static_macro_f1']:.4f} | {v['method_macro_f1']:.4f} | "
                     f"{v['gain_points']:+.2f} | {v['labels']:.0f} | {v['deploys']}/{v['rollbacks']} |")
    lines += ["", "说明：本地实验用于验证闭环在官方数据上的可运行性与增益；官方隐藏测试由平台独立评测。", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
