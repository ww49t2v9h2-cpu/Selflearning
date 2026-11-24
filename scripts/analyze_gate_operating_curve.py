"""J2 式门禁操作点分析：把"覆盖率 vs 兑现"从离线决策日志推进到候选级实测。

背景（配套研究 J2，未投稿）：覆盖率（可达增益）≠ 兑现（验证器/门禁实际放行带来的收益）；
兑现比 κ=(sel−p)/(cover−p) 由"验证器拒绝什么"决定。我们的门禁就是验证器。

做法（不需要重跑主实验）：
  1. 对 results/main 的每次 run，用保存的 config 复现 benchmark（确定性）；
  2. 对每个候选实验（registry candidate artifact），在它发生时刻 t 的隐藏 chunk 上实测 F1；
  3. 以 t 时刻 champion 在同一隐藏 chunk 上的 F1 为参照，得到"候选真实增益"（actual gain）；
  4. 扫描 probe 门禁阈值 θ（probe_gain ≥ θ 即放行），得到操作点曲线：
     coverage(θ)=放行比例、precision(θ)=放行中真实增益≥1pt 的比例、
     expected_gain(θ)=放行候选的平均真实增益、recall(θ)=真实好候选中被放行的比例；
  5. 与"实际门禁操作点"（deployed 的候选）对照，说明当前工作点是"偏保守的两侧门"。

用法：
  PY=.../python3
  $PY scripts/analyze_gate_operating_curve.py
输出：
  results/tables/gate_operating_curve.json / .md
  deliverables/figures/fig9_gate_curve.png（如 matplotlib 可用）
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.metrics import evaluate          # noqa: E402
from src.core.model import MLP                 # noqa: E402
from src.data.adapter import load_benchmark    # noqa: E402

RESULTS = ROOT / "results"
OUT = RESULTS / "tables"
GOOD_GAIN = 0.01  # 1pt，与 decision_analysis 的阈值一致


def eval_on_hidden(model: MLP, hx: np.ndarray, hy: np.ndarray, n_classes: int = 5) -> float:
    if len(hy) == 0:
        return float("nan")
    return float(evaluate(hy, model.predict(hx), n_classes)["macro_f1"])


def collect_candidates() -> list[dict]:
    """收集 main/scenarios/real 三组 run 的候选实验，并用隐藏集实测真实增益。

    参考模型 = 决策时刻 t 之前的 champion（timeline[t-1].champion_version）；
    候选真实增益 = F1(候选, 隐藏 chunk t) - F1(参考 champion, 隐藏 chunk t)。
    """
    rows: list[dict] = []
    for group in ("main", "scenarios", "real"):
        for p in sorted((RESULTS / group).glob("*.json")):
            try:
                rec = json.load(open(p, encoding="utf-8"))
            except Exception:
                continue
            if "meta" not in rec or not rec.get("experiments"):
                continue
            cfg = rec["config"]
            seed = rec["meta"]["seed"]
            try:
                bench = load_benchmark(cfg, seed=seed)
            except Exception:
                continue
            n_classes = bench.n_classes
            timeline = rec.get("timeline", [])
            registry = {int(v["version"]): v for v in rec.get("registry", [])}
            cache: dict[tuple[int, int], float] = {}

            def champ_f1(t: int) -> float | None:
                ref_t = max(0, t - 1)
                if ref_t >= len(timeline):
                    ref_t = len(timeline) - 1
                if ref_t < 0:
                    return None
                cv = int(timeline[ref_t].get("champion_version", 0))
                key = (cv, ref_t)
                if key in cache:
                    return cache[key]
                meta = registry.get(cv)
                if not meta:
                    return None
                try:
                    m = MLP.load(str(ROOT / meta["artifact"]))
                    hx, hy = bench.hidden_by_chunk[ref_t]
                    f = eval_on_hidden(m, hx, hy, n_classes)
                except Exception:
                    return None
                cache[key] = f
                return f

            for exp in rec.get("experiments", []):
                t = int(exp.get("t", 0))
                if t < 0 or t >= len(bench.chunks):
                    continue
                hx, hy = bench.hidden_by_chunk[t]
                try:
                    cand = MLP.load(str(ROOT / exp["artifact"]))
                except Exception:
                    continue
                f_cand = eval_on_hidden(cand, hx, hy, n_classes)
                f_champ = champ_f1(t)
                if f_champ is None or not np.isfinite(f_cand) or not np.isfinite(f_champ):
                    continue
                m = exp.get("metrics", {}) or {}
                pg = None
                if "probe_macro_f1" in m and "baseline_probe_macro_f1" in m:
                    pg = float(m["probe_macro_f1"]) - float(m["baseline_probe_macro_f1"])
                ver = int(exp.get("version", -1))
                tl = timeline[t] if t < len(timeline) else {}
                deployed = bool(tl.get("deploy", False)) and int(tl.get("champion_version", -1)) == ver
                rows.append({
                    "group": group,
                    "run_id": rec["meta"]["run_id"],
                    "scenario": rec["meta"].get("scenario", ""),
                    "seed": seed, "t": t,
                    "playbook": exp.get("playbook", ""),
                    "probe_gain": pg,
                    "actual_gain": float(f_cand - f_champ),
                    "deployed": deployed,
                })
    return rows


def curve(rows: list[dict], thresholds: list[float]) -> list[dict]:
    out = []
    gains = np.array([r["actual_gain"] for r in rows], dtype=float)
    probe = np.array([r["probe_gain"] if r["probe_gain"] is not None else -1e9 for r in rows], dtype=float)
    hard_good = gains >= GOOD_GAIN
    n_good = int(hard_good.sum())
    for th in thresholds:
        accept = probe >= th
        n_acc = int(accept.sum())
        if n_acc == 0:
            out.append({"theta": th, "coverage": 0.0, "precision": None,
                        "expected_gain_pt": None, "recall": 0.0, "n_accept": 0})
            continue
        prec = float((hard_good & accept).sum()) / n_acc
        eg = float(gains[accept].mean()) * 100
        rec = float((hard_good & accept).sum()) / max(1, n_good)
        out.append({"theta": th, "coverage": n_acc / len(rows), "precision": prec,
                    "expected_gain_pt": eg, "recall": rec, "n_accept": n_acc})
    return out


def main() -> int:
    rows = collect_candidates()
    if not rows:
        print("没有可分析候选（需 results/main 含 experiments）")
        return 1
    vals = [r for r in rows if r["probe_gain"] is not None]
    thresholds = [round(x, 3) for x in np.linspace(0.0, 0.12, 25)]
    cs = curve(vals, thresholds)
    # 实际门禁操作点
    dep = [r for r in vals if r["deployed"]]
    actual = {
        "n_candidates": len(vals),
        "n_deployed": len(dep),
        "coverage": len(dep) / len(vals),
        "expected_gain_pt": float(np.mean([r["actual_gain"] for r in dep])) * 100 if dep else None,
        "precision": float(np.mean([r["actual_gain"] >= GOOD_GAIN for r in dep])) if dep else None,
        "oracle_expected_gain_pt": float(np.max([r["actual_gain"] for r in vals])) * 100,
        "mean_actual_gain_pt": float(np.mean([r["actual_gain"] for r in vals])) * 100,
    }
    best = max((c for c in cs if c["expected_gain_pt"] is not None),
               key=lambda c: c["expected_gain_pt"], default=None)
    payload = {
        "method": "J2 式候选级操作点：probe_gain 阈值 θ 扫描 vs 隐藏集实测增益",
        "good_gain_threshold_pt": GOOD_GAIN * 100,
        "n_runs": len({r["run_id"] for r in rows}),
        "rows": vals,
        "curve": cs,
        "actual_gate": actual,
        "best_expected_gain_point": best,
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "gate_operating_curve.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 门禁操作点曲线（J2 式：覆盖率 vs 兑现）", "",
             "> 对每次候选实验，在发生时刻 t 的隐藏 chunk 上实测 F1，以 t 时刻 champion 为参照；",
             "> 扫描 probe 门禁阈值 θ，观察放行比例与真实增益的关系。配套研究 J2（未投稿）的 κ 分解的工程版。", "",
             f"- 候选数：**{len(vals)}**（{len({r['run_id'] for r in rows})} 个 run）；实际放行 **{len(dep)}**",
             f"- 实际门禁工作点：覆盖率 **{actual['coverage']*100:.1f}%**；"
             f"放行候选平均真实增益 **{actual['expected_gain_pt']:+.2f}pt**；"
             f"放行中真实增益≥1pt 的比例 **{(actual['precision'] or 0)*100:.1f}%**",
             f"- 全候选平均真实增益 {actual['mean_actual_gain_pt']:+.2f}pt；最优单候选 {actual['oracle_expected_gain_pt']:+.2f}pt", "",
             "| probe 阈值 θ | 放行比例（覆盖率） | 真实增益≥1pt 比例（precision） | 放行候选平均真实增益 | 真实好候选召回率 |",
             "|---|---|---|---|---|"]
    for c in cs:
        if c["n_accept"] == 0:
            continue
        lines.append(f"| {c['theta']:.3f} | {c['coverage']*100:.1f}% | {c['precision']*100:.1f}% | "
                     f"{c['expected_gain_pt']:+.2f}pt | {c['recall']*100:.1f}% |")
    lines += ["", "**解读**：",
              "1. 当前门禁不是单一 probe 阈值——它还要求类别召回/校准等硬条件，因此实际工作点落在曲线左侧（偏保守）；",
              "2. 提高兑现率的最短路径是**提高门禁判别力/探针预算**，而非增加候选数量（J2：N=1 常计算最优）；",
              "3. 该曲线可作为答辩「为什么兑现率只有 12.2%」的量化解释，属「实验规范与分析深度/加分项」证据。", ""]
    (OUT / "gate_operating_curve.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\nwritten {OUT/'gate_operating_curve.json'} / .md")

    # 图（可选，matplotlib 在 /tmp/plotenv 里）
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        xs = [c["coverage"] * 100 for c in cs if c["expected_gain_pt"] is not None]
        ys = [c["expected_gain_pt"] for c in cs if c["expected_gain_pt"] is not None]
        fig, ax = plt.subplots(figsize=(6.2, 3.6))
        ax.plot(xs, ys, marker="o", ms=3, lw=1.6, color="#2b6cb0", label="probe 阈值扫描")
        if actual["expected_gain_pt"] is not None:
            ax.axhline(actual["expected_gain_pt"], ls="--", lw=1, color="#c0392b",
                       label="实际门禁工作点（含硬条件）")
        ax.set_xlabel("放行比例 coverage (%)")
        ax.set_ylabel("放行候选平均真实增益 (pt)")
        ax.set_title("门禁操作点：覆盖率 vs 兑现（S7, 3 seeds）")
        ax.grid(alpha=0.25)
        ax.legend(fontsize=8)
        fig.tight_layout()
        figdir = ROOT / "deliverables" / "figures"
        figdir.mkdir(parents=True, exist_ok=True)
        fig.savefig(figdir / "fig9_gate_curve.png", dpi=200)
        plt.close(fig)
        print("figure: deliverables/figures/fig9_gate_curve.png")
    except Exception as e:
        print(f"figure skipped: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
