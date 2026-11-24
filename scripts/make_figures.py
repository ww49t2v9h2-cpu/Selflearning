"""生成报告/PPT 用图表（matplotlib，开发期工具，不进入系统依赖）。

运行：
  /tmp/plotenv/bin/python scripts/make_figures.py [--figs 1,2,3]
  # 依赖：系统 Python 的 numpy 不够，需要用装了 matplotlib 的 venv。
  # 若 venv 路径不同：FIGPY=/path/to/python scripts/make_figures.py

数据来源：
  results/SUMMARY.json、results/{main,scenarios,ab-*,budget-*}/**.json、
  results/tables/runs.csv、results/baselines/summary.json（可选）

输出：results/figures/fig{1..7}_*.png|.svg
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from matplotlib import font_manager as _fm  # noqa: E402

_CJK_CANDIDATES = ["PingFang SC", "Hiragino Sans GB", "Arial Unicode MS", "Heiti TC",
                   "Songti SC", "STHeiti", "Noto Sans CJK SC", "Source Han Sans SC"]
_available = {f.name for f in _fm.fontManager.ttflist}
CJK = [c for c in _CJK_CANDIDATES if c in _available][:1] or ["sans-serif"]
plt.rcParams.update({
    "font.family": CJK,
    "axes.unicode_minus": False,
    "figure.dpi": 110,
    "savefig.dpi": 220,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "--",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "font.size": 9,
})

C_METHOD = "#1f6feb"
C_STATIC = "#8b949e"
C_ACCENT = "#d29922"
C_RED = "#cf222e"
C_GREEN = "#2da44e"


def _load(path: Path):
    if not path.exists():
        return None
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _runs(group: str) -> list[dict]:
    d = RESULTS / group
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        r = _load(p)
        if r:
            out.append(r)
    return out


def _phase_boundaries(result: dict) -> tuple[list[int], list[str]]:
    b = result["benchmark"]
    cpp = b["chunks_per_phase"]
    bounds, names = [], []
    acc = 0
    for i, n in enumerate(cpp):
        acc += int(n)
        bounds.append(acc)
        names.append(b["phases"][i]["name"])
    return bounds, names


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    for ext in ("png", "svg"):
        fig.savefig(FIGS / f"{name}.{ext}", bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"  saved results/figures/{name}.png")


# --------------------------------------------------------------------------
# 图 1：主场景（S7）流式性能曲线 + 成本累计
# --------------------------------------------------------------------------
def fig1_main_curve() -> None:
    runs = _runs("main")
    if not runs:
        print("skip fig1: no main runs")
        return
    runs = sorted(runs, key=lambda r: r["meta"]["seed"])
    m = np.array([r["method_curve"] for r in runs])
    s = np.array([r["static_curve"] for r in runs])
    x = np.arange(m.shape[1])
    bounds, names = _phase_boundaries(runs[0])
    fig, axes = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True,
                             gridspec_kw={"height_ratios": [2.5, 1]})
    ax = axes[0]
    ax.plot(x, m.mean(0), color=C_METHOD, lw=2.0, label="自进化系统（本方法）")
    ax.fill_between(x, m.min(0), m.max(0), color=C_METHOD, alpha=0.15)
    ax.plot(x, s.mean(0), color=C_STATIC, lw=1.6, ls="--", label="静态基线（部署后不更新）")
    ax.fill_between(x, s.min(0), s.max(0), color=C_STATIC, alpha=0.10)
    for b, nm in zip(bounds[:-1], names[:-1]):
        ax.axvline(b, color="#d0d7de", lw=0.8, zorder=0)
    for b, nm in zip(bounds[:-1], names[:-1]):
        ax.text(b - 0.3, 1.02, nm, rotation=28, ha="right", va="bottom", fontsize=7.5,
                color="#57606a")
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("隐藏测试 macro-F1")
    ax.set_title("S7 综合漂移场景：静态基线 vs 自进化系统（3 seeds 均值±范围）", fontsize=11)
    ax.legend(loc="lower left", frameon=True, fontsize=8)

    ax2 = axes[1]
    for i, r in enumerate(runs):
        tl = r["timeline"]
        cost = [row["cost_cumulative"]["compute_units"] for row in tl]
        ax2.plot(np.arange(len(cost))[:len(cost)], cost, color=C_METHOD,
                 lw=1.2, alpha=0.8, label="累计适应算力" if i == 0 else None)
        labels = [row["cost_cumulative"]["labels"] for row in tl]
        ax2.plot(np.arange(len(labels)), labels, color=C_ACCENT, lw=1.2, alpha=0.7,
                 label="累计标注条数" if i == 0 else None)
    for b in bounds[:-1]:
        ax2.axvline(b, color="#d0d7de", lw=0.8, zorder=0)
    ax2.set_xlabel("部署流窗口 t")
    ax2.set_ylabel("累计成本")
    ax2.legend(loc="upper left", fontsize=8)
    _save(fig, "fig1_main_curve")


# --------------------------------------------------------------------------
# 图 2：各场景 静态 vs 自进化
# --------------------------------------------------------------------------
def fig2_scenarios() -> None:
    groups = [g for g in ("main", "scenarios") if (RESULTS / g).exists()]
    per: dict[str, dict] = {}
    for g in groups:
        for r in _runs(g):
            scn = r["meta"]["scenario"]
            per.setdefault(scn, {"m": [], "s": [], "c": []})
            per[scn]["m"].append(r["hidden"]["stream_macro_f1"])
            per[scn]["s"].append(r["static_baseline_hidden"]["stream_macro_f1"])
            per[scn]["c"].append(r["cost"]["scalar"])
    if not per:
        print("skip fig2: no scenario runs")
        return
    scns = sorted(per)
    x = np.arange(len(scns))
    m = [np.mean(per[s]["m"]) for s in scns]
    sd = [np.std(per[s]["m"]) for s in scns]
    st = [np.mean(per[s]["s"]) for s in scns]
    fig, ax = plt.subplots(figsize=(9.6, 4.2))
    ax.bar(x - 0.2, st, 0.38, color=C_STATIC, label="静态基线")
    ax.bar(x + 0.2, m, 0.38, color=C_METHOD, yerr=sd, capsize=3, label="自进化系统")
    for i, (a, b) in enumerate(zip(st, m)):
        ax.text(i, max(a, b) + 0.035, f"{(b - a) * 100:+.1f}pt", ha="center",
                fontsize=8.5, color=C_GREEN if b >= a else C_RED, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels([s.split("_", 1)[1] if "_" in s else s for s in scns], rotation=18)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("流式（窗口平均）macro-F1")
    ax.set_title("七种漂移场景：自进化系统 vs 静态基线（3 seeds 均值；误差棒=seed 标准差）")
    ax.legend(loc="lower right")
    _save(fig, "fig2_scenarios")


# --------------------------------------------------------------------------
# 图 3：消融（增益与成本）
# --------------------------------------------------------------------------
def fig3_ablation() -> None:
    groups = sorted([g.name for g in RESULTS.glob("ab-*")
                     if g.is_dir() and not g.name.startswith("ab-S3-")])
    if not groups:
        print("skip fig3: no ablation runs")
        return
    labels, gains, costs = [], [], []
    for g in groups:
        runs = _runs(g)
        if not runs:
            continue
        labels.append(g.replace("ab-", ""))
        gains.append(np.mean([r["comparison"]["stream_gain_points"] for r in runs]))
        costs.append(np.mean([r["cost"]["scalar"] for r in runs]))
    order = np.argsort(gains)
    labels = [labels[i] for i in order]
    gains = [gains[i] for i in order]
    costs = [costs[i] for i in order]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    ax = axes[0]
    bars = ax.barh(labels, gains, color=[C_GREEN if g > 0 else C_RED for g in gains])
    ax.axvline(0, color="#57606a", lw=0.8)
    for i, g in enumerate(gains):
        ax.text(g + (0.15 if g >= 0 else -0.15), i, f"{g:+.1f}",
                va="center", ha="left" if g >= 0 else "right", fontsize=8.5)
    ax.set_xlabel("流式增益（百分点，vs 静态基线）")
    ax.set_title("消融（S7）：诊断与 TTA 是主收益来源")
    ax = axes[1]
    ax.barh(labels, costs, color=C_ACCENT)
    for i, c in enumerate(costs):
        ax.text(c + 0.5, i, f"{c:.0f}", va="center", fontsize=8.5)
    ax.set_xlabel("累计成本（scalar）")
    ax.set_title("消融：成本")
    _save(fig, "fig3_ablation")


# --------------------------------------------------------------------------
# 图 4：标注预算扫描
# --------------------------------------------------------------------------
def fig4_budget() -> None:
    groups = sorted([g.name for g in RESULTS.glob("budget-*") if g.is_dir()],
                    key=lambda s: int(s.split("-")[1]))
    if not groups:
        print("skip fig4: no budget sweep runs")
        return
    budgets = [int(g.split("-")[1]) for g in groups]
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    for scn, color, marker in [("synth_S3_condition_switch", C_METHOD, "o"),
                               ("synth_S7_full_mix", C_ACCENT, "s")]:
        gains = []
        for g in groups:
            runs = [r for r in _runs(g) if r["meta"]["scenario"] == scn]
            gains.append(np.mean([r["comparison"]["stream_gain_points"] for r in runs])
                         if runs else np.nan)
        ax.plot(budgets, gains, color=color, marker=marker, lw=1.8,
                label=scn.replace("_", " "))
    ax.set_xlabel("标注预算（条）")
    ax.set_ylabel("流式增益（百分点）")
    ax.set_title("标注预算 vs 增益：预算越紧，策略选择越重要")
    ax.legend()
    _save(fig, "fig4_budget")


# --------------------------------------------------------------------------
# 图 5：S7 单次 run 的决策时间线
# --------------------------------------------------------------------------
def fig5_timeline() -> None:
    runs = _runs("main")
    if not runs:
        print("skip fig5: no main runs")
        return
    r = sorted(runs, key=lambda x: x["meta"]["seed"])[0]
    tl = r["timeline"]
    t = np.arange(len(tl))
    method = [row["hidden_macro_f1"] for row in tl]
    static = r["static_curve"]
    champ = [row["champion_version"] for row in tl]
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10.5, 5.0), sharex=True,
                                  gridspec_kw={"height_ratios": [2.2, 1]})
    ax.plot(t, method, color=C_METHOD, lw=1.8, label="自进化（隐藏 F1）")
    ax.plot(np.arange(len(static)), static, color=C_STATIC, lw=1.4, ls="--", label="静态基线")
    for i, row in enumerate(tl):
        if row.get("triggered"):
            ax.axvline(i, color=C_ACCENT, lw=0.7, alpha=0.5, zorder=0)
            if row.get("deploy"):
                ax.scatter(i, row["hidden_macro_f1"], marker="^", s=45, color=C_GREEN, zorder=5)
    for i, row in enumerate(tl):
        if row.get("deploy"):
            pb = row.get("chosen_playbook", "")[:4]
            ax.annotate(pb, (i, row["hidden_macro_f1"]), textcoords="offset points",
                        xytext=(2, 8), fontsize=6.8, color=C_GREEN)
    bounds, names = _phase_boundaries(r)
    for b, nm in zip(bounds[:-1], names[:-1]):
        ax.axvline(b, color="#d0d7de", lw=0.8, zorder=0)
    ax.set_ylabel("隐藏测试 macro-F1")
    ax.set_title(f"S7 决策时间线（seed={r['meta']['seed']}）：触发、上线、版本演进")
    ax.legend(loc="lower left", fontsize=8)

    ax2.step(t, champ, where="post", color=C_METHOD, lw=1.6, label="champion 版本")
    for i, row in enumerate(tl):
        if row.get("rollback"):
            ax2.axvline(i, color=C_RED, lw=1.2, ls=":")
    for b in bounds[:-1]:
        ax2.axvline(b, color="#d0d7de", lw=0.8, zorder=0)
    ax2.set_xlabel("部署流窗口 t")
    ax2.set_ylabel("版本号")
    ax2.legend(loc="upper left", fontsize=8)
    _save(fig, "fig5_timeline")


# --------------------------------------------------------------------------
# 图 6：成本-增益 Pareto
# --------------------------------------------------------------------------
def fig6_pareto() -> None:
    pts = []
    keep_ab = {"ab-A2_no_diagnosis", "ab-A7_no_tta"}
    for g in ["main"] + sorted([p.name for p in RESULTS.glob("ab-*") if p.is_dir()]):
        if g != "main" and g not in keep_ab:
            continue
        runs = _runs(g)
        if not runs:
            continue
        gain = np.mean([r["comparison"]["stream_gain_points"] for r in runs])
        cost = np.mean([r["cost"]["scalar"] for r in runs])
        pts.append((cost, gain, "Ours（主实验）" if g == "main" else g.replace("ab-", "消融-")))
    base = _load(RESULTS / "baselines" / "summary.json")
    if base:
        for name, item in base.get("methods", {}).items():
            pts.append((item.get("cost_scalar", 0.0), item.get("gain_points", 0.0), name))
    pts.append((0.0, 0.0, "静态基线"))
    if len(pts) <= 1:
        print("skip fig6: no comparable points")
        return
    fig, ax = plt.subplots(figsize=(7.6, 4.6))
    for cost, gain, name in pts:
        color = C_METHOD if name.startswith("Ours") else (
            C_RED if name.startswith("消融") else C_ACCENT)
        ax.scatter(cost, gain, s=70, color=color, edgecolor="white", zorder=3)
        ax.annotate(name, (cost, gain), textcoords="offset points", xytext=(7, 5),
                    fontsize=8, color="#24292f")
    ax.set_xlabel("累计成本（compute_units + 标注 + token 加权）")
    ax.set_ylabel("流式增益（百分点，vs 静态基线）")
    ax.set_title("成本-增益：越靠左上越优（Pareto 前沿）")
    _save(fig, "fig6_pareto")


# --------------------------------------------------------------------------
# 图 7：诊断类型 × playbook 决策热力图
# --------------------------------------------------------------------------
def fig7_strategy() -> None:
    runs = _runs("main") + _runs("scenarios")
    if not runs:
        print("skip fig7: no runs")
        return
    kinds, plays = [], []
    counter: dict[tuple[str, str], int] = {}
    for r in runs:
        for d in r.get("decisions", []):
            kind = (d.get("diagnosis") or {}).get("kind", "?")
            play = d.get("chosen_playbook", "NOOP")
            kinds.append(kind)
            plays.append(play)
            counter[(kind, play)] = counter.get((kind, play), 0) + 1
    kinds = sorted(set(kinds))
    plays = sorted(set(plays))
    mat = np.zeros((len(kinds), len(plays)))
    for (k, p), v in counter.items():
        mat[kinds.index(k), plays.index(p)] = v
    fig, ax = plt.subplots(figsize=(1.15 * len(plays) + 2.2, 0.55 * len(kinds) + 1.8))
    im = ax.imshow(mat, cmap="Blues", aspect="auto")
    ax.set_xticks(range(len(plays)))
    ax.set_xticklabels([p[:14] for p in plays], rotation=35, ha="right", fontsize=8)
    ax.set_yticks(range(len(kinds)))
    ax.set_yticklabels(kinds, fontsize=8)
    for i in range(len(kinds)):
        for j in range(len(plays)):
            if mat[i, j] > 0:
                ax.text(j, i, int(mat[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if mat[i, j] > mat.max() * 0.6 else "#24292f")
    ax.set_title("诊断类型 × 选择动作（策略层行为）")
    fig.colorbar(im, ax=ax, label="次数")
    _save(fig, "fig7_strategy")



# --------------------------------------------------------------------------
# 图 0：系统架构图（报告/PPT 用）
# --------------------------------------------------------------------------
def fig0_architecture() -> None:
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

    fig, ax = plt.subplots(figsize=(12.6, 6.0))
    ax.set_xlim(0, 118)
    ax.set_ylim(0, 62)
    ax.axis("off")

    def box(x, y, w, h, text, fc="#EAF1F8", ec="#1f6feb", fs=8.2, bold=False,
            tc="#24292f", ls="-"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.35,rounding_size=1.0",
                                    facecolor=fc, edgecolor=ec, linewidth=1.1,
                                    linestyle=ls))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=tc, fontweight="bold" if bold else "normal", linespacing=1.5)

    def arrow(x1, y1, x2, y2, color="#57606a", lw=1.2, ls="-", rad=0.0):
        ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>",
                                     mutation_scale=9, color=color, lw=lw,
                                     linestyle=ls, connectionstyle=f"arc3,rad={rad}"))

    # 顶部：Harness
    ax.add_patch(FancyBboxPatch((3, 52.0), 109, 7.2, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor="#F6F8FA", edgecolor="#afb8c1", linewidth=1.0,
                                linestyle="--"))
    ax.text(57.5, 55.6, "智能体 Harness：12 个工具 + 6 个技能\n"
                        "驱动：确定性策略（默认，0 token）/ LLM 可插拔（PyTRi0 Qwen3.8-27B）",
            ha="center", va="center", fontsize=8.6, color="#24292f", fontweight="bold",
            linespacing=1.5)

    # 主流水线（五阶段）
    y0, hh = 37.0, 8.8
    stages = [
        (3.0, 20.0, "① 漂移监测\nMMD/KS/熵/PH/novelty\n+ 预算内探针", "#EAF1F8", "#1f6feb"),
        (26.0, 19.0, "② 诊断归因\n8 类漂移 + 仿射\n可恢复性实测", "#EAF1F8", "#1f6feb"),
        (48.5, 19.0, "③ 实验优化\n9 个 playbook 成本阶梯\nbandit 规划 + 预算硬门", "#EAF1F8", "#1f6feb"),
        (71.0, 19.0, "④ 评估上线\n门禁：增益/召回/校准/成本", "#EAF1F8", "#1f6feb"),
        (93.5, 19.0, "⑤ 回滚\nwatchdog 持续监控\n版本回退", "#FDF2E3", "#d29922"),
    ]
    for x, w, t, fc, ec in stages:
        box(x, y0, w, hh, t, fc=fc, ec=ec)
    for x1, x2 in [(23.0, 26.0), (45.0, 48.5), (67.5, 71.0), (90.0, 93.5)]:
        arrow(x1, y0 + hh / 2, x2, y0 + hh / 2)

    # 部署流入口
    arrow(0.6, y0 + hh / 2, 3.0, y0 + hh / 2, color="#2e74b5", lw=1.6)
    ax.text(0.8, y0 - 2.4, "部署流 chunks（无标签 x）", fontsize=7.6, color="#2e74b5")

    # 模型仓库（④⑤下方）
    box(71.0, 24.5, 41.5, 6.2, "模型 Registry：不可变版本 + lineage + champion 指针\n（promote / rollback 均留痕）",
        fc="#EAF1F8", ec="#1f6feb", fs=8.0)
    arrow(80.5, y0, 80.5, 30.7)
    arrow(103.0, y0, 103.0, 30.7, color="#d29922", ls="--")

    # 策略层（②③下方）
    box(26.0, 24.5, 41.5, 6.2, "策略层自改进：bandit 策略表\n诊断类型 → playbook → 增益/成本 → 先验",
        fc="#FFF8E6", ec="#d29922", fs=8.0)
    arrow(35.5, y0, 35.5, 30.7)
    arrow(62.0, 30.7, 55.5, y0, color="#d29922", lw=1.2)

    # 平台侧
    ax.add_patch(FancyBboxPatch((3, 3.0), 65.0, 14.0, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor="#FDF6F6", edgecolor="#cf222e", linewidth=1.0,
                                linestyle=":"))
    ax.text(35.5, 14.3, "平台侧（隐藏标签对系统不可见）", ha="center", fontsize=8.8,
            color="#cf222e", fontweight="bold")
    box(5.5, 5.2, 27.0, 6.4, "LabelOracle\n按标注预算返回标签", fc="#FFFFFF", ec="#cf222e", fs=8.2)
    box(36.0, 5.2, 29.0, 6.4, "HiddenEvaluator\n隐藏测试评分（只进报告）", fc="#FFFFFF", ec="#cf222e", fs=8.0)
    # oracle → monitor 的探针标签（虚线，从平台侧顶部直接上行）
    arrow(12.0, 11.6, 12.0, y0, color="#cf222e", lw=1.0, ls="--")
    ax.text(12.6, 19.0, "探针标签", fontsize=7.2, color="#cf222e")

    # 可观测性 / 安全
    ax.add_patch(FancyBboxPatch((71.0, 3.0), 41.5, 14.0, boxstyle="round,pad=0.5,rounding_size=1.0",
                                facecolor="#F0FAF3", edgecolor="#2da44e", linewidth=1.0,
                                linestyle=":"))
    ax.text(91.7, 14.3, "可观测性 / 安全", ha="center", fontsize=8.8, color="#2da44e",
            fontweight="bold")
    box(73.0, 5.2, 19.0, 6.4, "Tracker\nSQLite + SwanLab\n(online/local/offline)", fc="#FFFFFF",
        ec="#2da44e", fs=7.6)
    box(94.0, 5.2, 16.5, 6.4, "AuditLog\nsha256 哈希链\n护栏检查", fc="#FFFFFF", ec="#2da44e", fs=7.6)
    arrow(91.7, 24.3, 91.7, 17.2, color="#2da44e", lw=1.2, ls="--")
    ax.text(93.0, 20.6, "全部决策/实验/版本/成本", fontsize=7.2, color="#2da44e", ha="left")
    _save(fig, "fig0_architecture")



def fig8_ai4i() -> None:
    """真实工业数据（AI4I 2020）闭环结果。"""
    import json as _json
    base = None
    bp = RESULTS / "baselines" / "summary.json"
    if bp.exists():
        with open(bp, "r", encoding="utf-8") as f:
            base = _json.load(f)
    ours = _runs("real")
    if not ours and not base:
        print("skip fig8: no AI4I runs")
        return
    labels, vals, colors = [], [], []
    if base and base.get("results", {}).get("AI4I_maintenance"):
        b = base["results"]["AI4I_maintenance"]
        for key, name in [("static", "B0 静态"), ("periodic", "B1 周期重训"), ("oracle", "B2 oracle")]:
            v = b.get(key) or {}
            if v:
                labels.append(name)
                vals.append(float(v.get("mean_f1_mean", 0)))
                colors.append(C_STATIC if key == "static" else C_ACCENT)
    if ours:
        m = float(np.mean([r["hidden"]["stream_macro_f1"] for r in ours]))
        s_ = float(np.mean([r["static_baseline_hidden"]["stream_macro_f1"] for r in ours]))
        labels.append("Ours（自进化）")
        vals.append(m)
        colors.append(C_METHOD)
        # 用系统同 run 的静态基线替换/补充（更可比）
        labels.insert(0, "B0 静态（同 run）")
        vals.insert(0, s_)
        colors.insert(0, C_STATIC)
    fig, ax = plt.subplots(figsize=(7.2, 3.6))
    bars = ax.bar(labels, vals, color=colors, width=0.55)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, v + 0.01, f"{v:.3f}",
                ha="center", fontsize=9, fontweight="bold")
    ax.set_ylim(0, max(vals) * 1.25)
    ax.set_ylabel("流式 macro-F1（真实工业数据 AI4I）")
    ax.set_title("真实数据闭环：同一套系统仅换数据适配层")
    _save(fig, "fig8_ai4i")


FIG_FUNCS = {0: fig0_architecture, 1: fig1_main_curve, 2: fig2_scenarios, 3: fig3_ablation,
             4: fig4_budget, 5: fig5_timeline, 6: fig6_pareto, 7: fig7_strategy,
             8: fig8_ai4i}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--figs", default="all")
    args = ap.parse_args()
    figs = list(FIG_FUNCS) if args.figs == "all" else [int(x) for x in args.figs.split(",")]
    print(f"[make_figures] 输出目录 {FIGS}")
    for i in figs:
        fn = FIG_FUNCS.get(i)
        if fn:
            try:
                fn()
            except Exception as e:
                print(f"  fig{i} 失败: {e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
