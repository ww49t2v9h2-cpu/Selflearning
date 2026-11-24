"""汇总所有"要写进文档/答辩"的关键数字（单命令输出，便于回填 docs/10/11/14 与 PPT）。

用法：
  PY=.../python3
  $PY scripts/summary_numbers.py
输出：results/NUMBERS.md（同时打印）
"""

from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"


def load(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def main() -> int:
    summary = load(RESULTS / "SUMMARY.json") or {}
    baselines = load(RESULTS / "baselines" / "summary.json") or {}
    acceptance = load(RESULTS / "acceptance.json") or {}
    decisions = load(RESULTS / "tables" / "decision_analysis.json") or {}
    diag = load(RESULTS / "diagnostics" / "s2_affine.json") or {}

    lines = ["# 关键数字汇总（自动生成，供 docs/10/11/14 与 PPT 回填）", ""]

    g = summary.get("groups", {})
    if g:
        lines.append("## 1. 主实验/分组")
        lines.append("| 组 | n | 静态 F1 | 自进化 F1 | 增益(pt) | cost_scalar | labels | compute | 部署 | 回滚 |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for name, v in g.items():
            lines.append(f"| {name} | {v['n']} | {v['static_f1_mean']:.4f} | "
                         f"{v['method_f1_mean']:.4f} | {v['gain_points_mean']:+.2f} | "
                         f"{v['cost_scalar_mean']:.2f} | {v['labels_mean']:.0f} | "
                         f"{v['compute_mean']:.1f} | {v['deploys_mean']:.1f} | "
                         f"{v['rollbacks_mean']:.1f} |")
        lines.append("")
        lines.append("## 2. 分场景（main + scenarios）")
        lines.append("| 场景 | 静态 F1 | 自进化 F1 | 增益(pt) | 成本 |")
        lines.append("|---|---|---|---|---|")
        for grp in ("main", "scenarios"):
            if grp not in g:
                continue
            for scn, v in g[grp]["per_scenario"].items():
                lines.append(f"| {scn} | {v['static_f1']:.4f} | {v['method_f1']:.4f} | "
                             f"{v['gain_points']:+.2f} | {v['cost_scalar']:.2f} |")
        lines.append("")
        lines.append("## 3. 验收断言")
        lines.append("```json")
        lines.append(json.dumps(summary.get("acceptance", {}), ensure_ascii=False, indent=1))
        lines.append("```")
        lines.append("")

    if acceptance:
        lines.append("## 4. acceptance.py 逐项")
        lines.append("| 检查 | 通过 | 详情 |")
        lines.append("|---|---|---|")
        for c in acceptance.get("checks", []):
            lines.append(f"| {c['name']} | {'✅' if c['pass'] else '❌'} | "
                         f"{json.dumps(c.get('detail', {}), ensure_ascii=False)[:180]} |")
        lines.append("")

    if decisions:
        lines.append("## 5. 决策分析（覆盖率/兑现率/门禁判别力）")
        lines.append(f"- 覆盖率（可达增益）：{decisions['coverage_rate'] * 100:.1f}%；"
                     f"兑现率：{decisions['realized_rate'] * 100:.1f}%")
        lines.append(f"- 门禁拒绝「增益不足」候选：{decisions['gate_reject_bad_rate'] * 100:.1f}%；"
                     f"放行「有增益」候选：{decisions['gate_accept_good_rate'] * 100:.1f}%")
        lines.append(f"- 动作分布：{json.dumps(decisions.get('actions', {}), ensure_ascii=False)}")
        lines.append("")

    if baselines.get("results"):
        lines.append("## 6. 基线对照（S7，3 seeds）")
        b7 = baselines["results"].get("S7_full_mix", {})
        for key in ("static", "periodic", "oracle"):
            v = b7.get(key) or {}
            if v:
                lines.append(f"- {key}: F1={v.get('mean_f1_mean', 0):.4f}, "
                             f"labels={v.get('labels_mean', 0):.0f}, "
                             f"compute={v.get('compute_units_mean', 0):.0f}")
        lines.append("")

    if diag.get("summary"):
        lines.append("## 7. S2 零标注仿射校准")
        lines.append("| 阶段 | static | affine(0标注) | 增益 |")
        lines.append("|---|---|---|---|")
        for r in diag["summary"]:
            lines.append(f"| {r['name']} | {r['static_mean']:.4f} | {r['affine_mean']:.4f} | "
                         f"{r['gain_points']:+.1f}pt |")
        lines.append("")

    out = RESULTS / "NUMBERS.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nwritten {out.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
