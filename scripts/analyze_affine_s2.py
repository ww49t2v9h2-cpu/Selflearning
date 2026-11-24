"""S2 传感器老化场景的"零标签仿射校准"分析（支撑 docs/09 §5.5）。

对每个 seed：训练 base 模型 → 逐窗口计算
  - static：原模型直接预测（隐藏测试）
  - affine：用当前窗口无标签特征估计目标域 scaler，再 align_predict_proba（0 标注、0 训练）
按阶段聚合，输出 results/diagnostics/s2_affine.json，并打印 Markdown 表。

用法：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY scripts/analyze_affine_s2.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config, set_global_seed    # noqa: E402
from src.core.metrics import evaluate                       # noqa: E402
from src.core.model import MLP                              # noqa: E402
from src.data.synth import make_benchmark                   # noqa: E402

SEEDS = [7, 17, 27]
SCENARIO = "S2_sensor_gain"
OUT = ROOT / "results" / "diagnostics" / "s2_affine.json"


def run_seed(cfg: dict, seed: int) -> dict:
    cfg = json.loads(json.dumps(cfg))
    cfg["benchmark"]["scenario"] = SCENARIO
    set_global_seed(seed)
    bench = make_benchmark(cfg, seed=seed)
    model = MLP(bench.d, bench.base_classes, hidden=cfg["model"]["hidden"], seed=seed)
    model.fit(bench.base_x, bench.base_y, epochs=cfg["model"]["epochs"],
              lr=cfg["model"]["lr"], batch=cfg["model"]["batch"], seed=seed)
    per_chunk = []
    for t, chunk in enumerate(bench.chunks):
        hx, hy = bench.hidden_by_chunk[t]
        static = evaluate(hy, model.predict(hx), bench.n_classes)["macro_f1"]
        tgt = model.estimate_target_scaler(chunk.x)
        pred = np.argmax(model.align_predict_proba(hx, tgt), axis=1)
        affine = evaluate(hy, pred, bench.n_classes)["macro_f1"]
        per_chunk.append({"t": t, "phase": chunk.phase,
                          "static": static, "affine": affine})
    # 按阶段聚合
    phases = {}
    for p in range(len(bench.phase_names)):
        rows = [r for r in per_chunk if r["phase"] == p]
        phases[p] = {
            "name": bench.phase_names[p],
            "static_mean": float(np.mean([r["static"] for r in rows])),
            "affine_mean": float(np.mean([r["affine"] for r in rows])),
            "n_chunks": len(rows),
        }
    return {"seed": seed, "phases": phases}


def main() -> int:
    cfg = load_config("configs/main.json")
    results = [run_seed(cfg, s) for s in SEEDS]
    names = [results[0]["phases"][p]["name"] for p in sorted(results[0]["phases"])]
    summary = []
    for p in sorted(results[0]["phases"]):
        st = [r["phases"][p]["static_mean"] for r in results]
        af = [r["phases"][p]["affine_mean"] for r in results]
        summary.append({
            "phase": p, "name": names[p],
            "static_mean": float(np.mean(st)), "static_std": float(np.std(st)),
            "affine_mean": float(np.mean(af)), "affine_std": float(np.std(af)),
            "gain_points": float((np.mean(af) - np.mean(st)) * 100),
        })
    out = {"scenario": SCENARIO, "seeds": SEEDS, "per_seed": results,
           "summary": summary,
           "note": "affine = 用当前窗口无标签特征估计目标域 scaler 后 align_predict_proba；"
                   "0 标注、0 训练算力，仅推理时变换"}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=1)
    print(f"written {OUT.relative_to(ROOT)}")
    print("\n| 阶段 | B0 static | 仿射校准(0标注) | 增益 |")
    print("|---|---|---|---|")
    for r in summary:
        print(f"| {r['name']} | {r['static_mean']:.4f}±{r['static_std']:.3f} | "
              f"{r['affine_mean']:.4f}±{r['affine_std']:.3f} | {r['gain_points']:+.1f}pt |")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
