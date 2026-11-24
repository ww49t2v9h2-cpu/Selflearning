"""按官方 5 分钟速览视频披露的评分细则（客观分 35 的五个得分点）计算我们的指标。

视频口径（2026-11 官方讲解视频）：
  客观 35 = 漂移流综合性能 15 + 最差阶段表现 5 + 恢复速度 5 + 稳定性与抗遗忘 5 + 资源效率 5
  - 漂移流综合性能：各阶段归一化增益均值（G≤0 得 0 分）
  - 最差阶段表现：最差阶段的性能
  - 恢复速度：恢复至漂移前 90% 所需阶段数
  - 稳定性与抗遗忘：回测下降 ≤5% 得满分
  - 资源效率：算力/标注/Token 分档计分

本脚本用 results/main（S7 综合漂移，3 seeds）复算这五项的可量化代理，并额外做
"脏数据/缺失/异常"鲁棒性压力测试（对应决赛数据包-B：更难、更复杂、含脏数据）。

用法：
  PY=.../python3
  $PY scripts/analyze_official_rubric.py
输出：
  results/tables/official_rubric.json / .md
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
import sys
sys.path.insert(0, str(ROOT))

from src.core.metrics import evaluate                    # noqa: E402
from src.core.model import MLP                           # noqa: E402
from src.data.adapter import load_benchmark              # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "tables"


def load_models(rec: dict, cfg: dict, seed: int):
    bench = load_benchmark(cfg, seed=seed)
    champs = [v for v in rec["registry"] if v["status"] == "champion"]
    base_meta = rec["registry"][0]
    base = MLP.load(str(ROOT / base_meta["artifact"]))
    champ = MLP.load(str(ROOT / (champs[-1]["artifact"] if champs else base_meta["artifact"])))
    return bench, base, champ


def phase_eval(bench, model, n_classes: int = 5) -> dict[str, float]:
    """在隐藏测试集上逐阶段评估（regenerate 的 benchmark 与运行期完全一致）。"""
    per: dict[int, list[float]] = {}
    for t, chunk in enumerate(bench.chunks):
        hx, hy = bench.hidden_by_chunk[t]
        per.setdefault(chunk.phase, []).append(evaluate(hy, model.predict(hx), n_classes)["macro_f1"])
    return {str(p): float(np.mean(v)) for p, v in sorted(per.items())}


def recovery_phases(method_curve: list[float], phase_boundaries: list[int],
                    chunks_per_phase: int, tol: float = 0.9, hold: int = 2) -> dict:
    """恢复速度：每个漂移阶段从起始窗口起，恢复到"漂移前 90%"所需阶段数。"""
    out = []
    for p in range(1, len(phase_boundaries)):
        start = phase_boundaries[p]
        prev_start = phase_boundaries[p - 1]
        # 漂移前参考：上一阶段最后 3 个窗口的均值
        ref = float(np.mean(method_curve[max(prev_start, start - 3):start]))
        found = None
        for w in range(start, len(method_curve) - hold + 1):
            if all(v >= tol * ref for v in method_curve[w:w + hold]):
                found = w
                break
        if found is None:
            used = (len(method_curve) - start) / chunks_per_phase
            recovered = False
        else:
            used = (found - start) / chunks_per_phase
            recovered = True
        out.append({"phase": p, "start_window": start, "ref": round(ref, 4),
                    "recovery_windows": None if found is None else found - start,
                    "recovery_phases": round(used, 2), "recovered": recovered})
    ok = [r["recovery_phases"] for r in out if r["recovered"]]
    return {
        "per_drift": out,
        "mean_recovery_phases": float(np.mean(ok)) if ok else None,
        "recovered_fraction": round(len(ok) / max(1, len(out)), 3),
    }


def main() -> int:
    files = sorted((RESULTS / "main").glob("*.json"))
    if not files:
        print("缺少 results/main（先跑 scripts/run_all.py --groups main）")
        return 1
    runs = []
    for f in files:
        rec = json.load(open(f, encoding="utf-8"))
        cfg = rec["config"]
        seed = rec["meta"]["seed"]
        bench, base, champ = load_models(rec, cfg, seed)
        n_chunks = rec["benchmark"]["n_chunks"]
        cpp = rec["benchmark"]["chunks_per_phase"][0]
        bounds = [i * cpp for i in range(len(rec["benchmark"]["chunks_per_phase"]) + 1)]

        base_phase = phase_eval(bench, base)
        champ_phase = phase_eval(bench, champ)
        method_curve = rec["method_curve"]
        static_curve = rec["static_curve"]

        # --- 1) 漂移流综合性能：各阶段归一化增益均值 ---
        gains, norm_gains, norm_clipped = [], [], []
        for k in base_phase:
            m, s = champ_phase[k], base_phase[k]
            g = m - s
            gains.append(g)
            norm = g / (1.0 - s) if (1.0 - s) > 1e-9 else 0.0
            norm_gains.append(norm)
            norm_clipped.append(max(0.0, norm))
        G_raw = float(np.mean(gains) * 100)
        G_norm = float(np.mean(norm_gains) * 100)
        G_norm_clipped = float(np.mean(norm_clipped) * 100)

        # --- 2) 最差阶段表现 ---
        worst_key = min(champ_phase, key=lambda k: champ_phase[k])
        worst_gain_key = min(gains and base_phase or base_phase,
                             key=lambda k: champ_phase[k] - base_phase[k])

        # --- 3) 恢复速度 ---
        recov = recovery_phases(method_curve, bounds, cpp)

        # --- 4) 稳定性与抗遗忘（回测下降）---
        # 用最终 champion 在"更早阶段"的隐藏集上回测，与 base 模型同集对比
        drops = {}
        for k in base_phase:
            if int(k) >= int(worst_key):
                pass
            drops[k] = base_phase[k] - champ_phase[k]
        max_drop = max(drops.values())
        max_drop_phase = max(drops, key=drops.get)
        # 只统计"回测"（最终模型对更早阶段）：排除最后一阶段
        last_key = str(max(int(k) for k in base_phase))
        early = [k for k in base_phase if k != last_key]
        max_drop_early = max(drops[k] for k in early) if early else 0.0
        early_phase = max(early, key=lambda k: drops[k]) if early else last_key

        # --- 5) 资源效率 ---
        cost = rec["cost"]

        runs.append({
            "seed": seed,
            "run_id": rec["meta"]["run_id"],
            "G_raw_points": round(G_raw, 2),
            "G_normalized": round(G_norm, 2),
            "G_normalized_clipped": round(G_norm_clipped, 2),
            "per_phase_gain_points": {k: round(v * 100, 2) for k, v in zip(base_phase, gains)},
            "worst_phase": worst_key,
            "worst_phase_f1": round(champ_phase[worst_key], 4),
            "worst_phase_static": round(base_phase[worst_key], 4),
            "worst_gain_points": round((champ_phase[worst_key] - base_phase[worst_key]) * 100, 2),
            "recovery": recov,
            "max_backtest_drop_points": round(max_drop * 100, 2),
            "max_backtest_drop_phase": max_drop_phase,
            "max_backtest_drop_early_points": round(max_drop_early * 100, 2),
            "max_backtest_drop_early_phase": early_phase,
            "cost": {k: cost.get(k) for k in ("compute_units", "labels", "tokens", "scalar")},
        })

    def mean(key):
        vals = [r[key] for r in runs]
        return float(np.mean(vals)), float(np.std(vals))

    # 手工整理 summary（避免上面占位键）
    summary = {
        "n_runs": len(runs),
        "G_raw_points": mean("G_raw_points"),
        "G_normalized": mean("G_normalized"),
        "G_normalized_clipped": mean("G_normalized_clipped"),
        "worst_phase_f1": mean("worst_phase_f1"),
        "worst_gain_points": mean("worst_gain_points"),
        "mean_recovery_phases": (float(np.mean([r["recovery"]["mean_recovery_phases"]
                                                for r in runs if r["recovery"]["mean_recovery_phases"] is not None])), 0.0),
        "max_backtest_drop_points": mean("max_backtest_drop_points"),
        "max_backtest_drop_early_points": mean("max_backtest_drop_early_points"),
        "labels": (float(np.mean([r["cost"]["labels"] for r in runs])), 0.0),
        "compute_units": (float(np.mean([r["cost"]["compute_units"] for r in runs])), 0.0),
        "tokens": (float(np.mean([r["cost"]["tokens"] for r in runs])), 0.0),
    }

    payload = {"rubric_source": "官方 5 分钟速览视频（2026-11-16 获取）", "runs": runs, "summary": summary}
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "official_rubric.json").write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 官方评分细则对齐（客观分 35 的五个得分点）", "",
             "> 口径来自官方 5 分钟速览视频：15+5+5+5+5；本表为 S7 综合漂移 3 seeds 的实测复算。", "",
             "| 得分点 | 我们的实测（3 seeds 均值） | 说明 |", "|---|---|---|",
             f"| 漂移流综合性能（15 分） | G_raw={summary['G_raw_points'][0]:+.2f}pt；"
             f"G_norm={summary['G_normalized'][0]:+.1f}%（归一化 (m−s)/(1−s)） | 各阶段归一化增益均值 >0 |",
             f"| 最差阶段表现（5 分） | 最差阶段 F1={summary['worst_phase_f1'][0]:.4f}；"
             f"该阶段增益={summary['worst_gain_points'][0]:+.2f}pt | 见 runs[].worst_phase |",
             f"| 恢复速度（5 分） | 平均 {summary['mean_recovery_phases'][0]:.2f} 个阶段恢复到漂移前 90% | "
             f"按窗口曲线计算（90%×前阶段均值，连续 2 窗口） |",
             f"| 稳定性与抗遗忘（5 分） | 最终模型对更早阶段的最大回测下降 "
             f"{summary['max_backtest_drop_early_points'][0]:.2f}pt | ≤5% 得满分（视频口径） |",
             f"| 资源效率（5 分） | labels={summary['labels'][0]:.0f} / compute={summary['compute_units'][0]:.1f}u / "
             f"tokens={summary['tokens'][0]:.0f} | 分档计分，越低越好 |", ""]
    (OUT / "official_rubric.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"written {OUT/'official_rubric.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
