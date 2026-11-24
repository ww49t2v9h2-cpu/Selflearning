"""决策分析：覆盖率 vs 兑现率、门禁判别力、NOOP 方向（支撑 docs/10 §5.7 与答辩）。

方法论借自配套研究 Paper II：把"多候选实验"拆成
  - 覆盖率（reach）：本轮候选中存在"实测 probe 增益 ≥ 阈值"的候选（可达增益）；
  - 兑现率（realized）：该候选最终被门禁放行并上线（系统真正拿到增益）。
以及门禁判别力：对"实测增益<阈值"的候选拒绝率（越高越好）、对"实测增益≥阈值"的候选放行率。

数据来源：results/{main,scenarios}/*.json 的 decisions（含 outcome.gate/probe_gain）
与 experiments（含 metrics.probe_macro_f1 / baseline_probe_macro_f1）。

用法：
  PY=.../python3
  $PY scripts/analyze_decisions.py
输出：results/tables/decision_analysis.json + .md
"""

from __future__ import annotations

import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
OUT = RESULTS / "tables"


def load_runs() -> list[dict]:
    runs = []
    for group in ("main", "scenarios", "real"):
        d = RESULTS / group
        if not d.exists():
            continue
        for p in sorted(d.glob("*.json")):
            try:
                with open(p, "r", encoding="utf-8") as f:
                    r = json.load(f)
                if "meta" in r:
                    runs.append(r)
            except Exception:
                pass
    return runs


def analyze(runs: list[dict]) -> dict:
    rows = []
    for r in runs:
        scn = r["meta"]["scenario"]
        seed = r["meta"]["seed"]
        for d in r.get("decisions", []):
            oc = d.get("outcome") or {}
            gate = oc.get("gate") if isinstance(oc.get("gate"), dict) else {}
            probe_gain = oc.get("probe_gain")
            if probe_gain is None and isinstance(gate.get("criteria"), dict):
                probe_gain = (gate["criteria"].get("probe_gain") or {}).get("value")
            rows.append({
                "scenario": scn, "seed": seed, "t": d.get("t"),
                "kind": (d.get("diagnosis") or {}).get("kind", ""),
                "chosen": d.get("chosen_playbook", "NOOP"),
                "action": oc.get("action", ""),
                "deployed": bool(oc.get("deployed")),
                "gate_passed": bool(gate.get("passed")) if gate else None,
                "probe_gain": probe_gain,
                "alternatives": [a.get("playbook") for a in d.get("alternatives", [])],
            })

    # ---- 覆盖率 / 兑现率（只在有实验被执行、且有 probe_gain 的决策上算） ----
    tried = [x for x in rows if x["probe_gain"] is not None]
    thr = 0.01
    reach = [x for x in tried if float(x["probe_gain"]) >= thr]
    confirmed = [x for x in reach if x["deployed"]]
    coverage = len(reach) / max(1, len(tried))
    realized = len(confirmed) / max(1, len(reach))
    # ---- 门禁判别力 ----
    bad = [x for x in tried if float(x["probe_gain"]) < thr]
    good = reach
    reject_bad = (sum(1 for x in bad if not x["deployed"]) / max(1, len(bad)))
    accept_good = (sum(1 for x in good if x["deployed"]) / max(1, len(good)))
    # ---- NOOP / 动作分布 ----
    noop_reasons = Counter()
    for x in rows:
        if x["chosen"] == "NOOP" or x["action"] == "noop":
            noop_reasons[x["kind"] or "unknown"] += 1
    actions = Counter(x["action"] or "none" for x in rows)
    per_scn = defaultdict(lambda: {"n": 0, "tried": 0, "reach": 0, "deployed": 0, "rollback": 0})
    for x in rows:
        p = per_scn[x["scenario"]]
        p["n"] += 1
        if x["probe_gain"] is not None:
            p["tried"] += 1
            if float(x["probe_gain"]) >= thr:
                p["reach"] += 1
        if x["deployed"]:
            p["deployed"] += 1
    return {
        "n_decisions": len(rows),
        "n_with_experiment": len(tried),
        "coverage_rate": coverage,
        "realized_rate": realized,
        "gate_reject_bad_rate": reject_bad,
        "gate_accept_good_rate": accept_good,
        "actions": dict(actions),
        "noop_by_kind": dict(noop_reasons),
        "per_scenario": {k: dict(v) for k, v in per_scn.items()},
        "samples": tried[:50],
    }


def to_md(res: dict) -> str:
    lines = ["# 决策分析：覆盖率 / 兑现率 / 门禁判别力（自动生成）", ""]
    lines.append(f"- 决策数：{res['n_decisions']}；有实测候选增益的决策数：{res['n_with_experiment']}")
    lines.append(f"- **覆盖率（可达增益）**：{res['coverage_rate'] * 100:.1f}%")
    lines.append(f"- **兑现率（可达→上线）**：{res['realized_rate'] * 100:.1f}%")
    lines.append(f"- 门禁对「增益不足」候选的拒绝率：{res['gate_reject_bad_rate'] * 100:.1f}%")
    lines.append(f"- 门禁对「有增益」候选的放行率：{res['gate_accept_good_rate'] * 100:.1f}%")
    lines.append("")
    lines.append("## 动作分布")
    lines.append("| action | 次数 |")
    lines.append("|---|---|")
    for k, v in sorted(res["actions"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## NOOP 按诊断类型")
    lines.append("| 诊断类型 | NOOP/未上线次数 |")
    lines.append("|---|---|")
    for k, v in sorted(res["noop_by_kind"].items(), key=lambda kv: -kv[1]):
        lines.append(f"| {k} | {v} |")
    lines.append("")
    lines.append("## 分场景")
    lines.append("| 场景 | 决策 | 有实验 | 可达 | 上线 |")
    lines.append("|---|---|---|---|---|")
    for k, v in sorted(res["per_scenario"].items()):
        lines.append(f"| {k} | {v['n']} | {v['tried']} | {v['reach']} | {v['deployed']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    runs = load_runs()
    if not runs:
        print("没有可分析的 run（先跑 scripts/run_all.py）")
        return 1
    res = analyze(runs)
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "decision_analysis.json", "w", encoding="utf-8") as f:
        json.dump(res, f, ensure_ascii=False, indent=1)
    with open(OUT / "decision_analysis.md", "w", encoding="utf-8") as f:
        f.write(to_md(res))
    print(to_md(res))
    print(f"written {OUT / 'decision_analysis.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
