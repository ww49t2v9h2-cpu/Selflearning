"""实验矩阵运行与聚合：主实验 / 场景 / 消融 / 预算扫描 / 路线对比。

一条命令产出：
  results/<group>/<scenario>_seed<seed>.json   每次闭环的完整结果
  results/tables/runs.csv                      长表（所有 run）
  results/tables/{main,scenarios,ablations,budget}.csv
  results/SUMMARY.json / results/SUMMARY.md    汇总（均值±方差）+ 验收断言

用法：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY scripts/run_all.py --groups main --seeds 7,17,27
  $PY scripts/run_all.py --groups main,scenarios,ablations,budget,real --seeds 7,17,27
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import RESULTS_DIR, load_config          # noqa: E402
from src.loop import run_one                                  # noqa: E402

SCENARIOS = ["S1_gradual_wear", "S2_sensor_gain", "S3_condition_switch",
             "S4_novel_class", "S5_repair_cycle", "S6_label_shift", "S7_full_mix"]

ABLATIONS = {
    "A1_no_trigger": {"ablation": {"trigger_mode": "always"}},
    "A2_no_diagnosis": {"ablation": {"diagnosis": False}},
    "A3_random_planner": {"ablation": {"planner": "random"}},
    "A4_no_gate": {"ablation": {"gate": False}},
    "A5_no_strategy": {"ablation": {"strategy_learning": False}},
    "A6_no_watchdog": {"ablation": {"watchdog": False}},
    "A7_no_tta": {"playbooks": {"RECALIBRATE_AFFINE": {"blend": 0.0}}},
}


@dataclass
class RunSpec:
    group: str
    scenario: str
    seed: int
    overrides: dict = field(default_factory=dict)
    note: str = ""


def build_specs(base: dict, groups: list[str], seeds: list[int]) -> list[RunSpec]:
    specs: list[RunSpec] = []
    main_scn = base["benchmark"].get("scenario", "S7_full_mix")
    if "main" in groups:
        for s in seeds:
            specs.append(RunSpec("main", main_scn, s, {}, "主实验"))
    if "scenarios" in groups:
        for scn in SCENARIOS:
            for s in seeds:
                specs.append(RunSpec("scenarios", scn, s, {}, "全场景"))
    if "ablations" in groups:
        for name, ov in ABLATIONS.items():
            for s in seeds:
                specs.append(RunSpec(f"ab-{name}", main_scn, s,
                                     copy.deepcopy(ov), f"消融 {name}"))
    if "budget" in groups:
        for labels in [60, 120, 220]:
            for scn in ["S3_condition_switch", "S7_full_mix"]:
                for s in seeds:
                    specs.append(RunSpec(f"budget-{labels}", scn, s,
                                         {"budget": {"labels_total": labels}},
                                         f"标注预算={labels}"))
    if "real" in groups:
        for s in seeds:
            specs.append(RunSpec("real", "ai4i_predictive_maintenance", s,
                                 {"benchmark": {"adapter": "ai4i"},
                                  "budget": {"labels_total": 120}},
                                 "真实工业数据闭环（AI4I）"))
    if "route" in groups:
        for s in seeds[:1]:
            specs.append(RunSpec("route-mockllm", main_scn, s,
                                 {"harness": {"driver": "llm", "provider": "mock"}},
                                 "路线对比：LLM(mock) 驱动"))
    return specs


def run_matrix(base_cfg: dict, groups: list[str], seeds: list[int],
               max_runs: int = 0, quiet: bool = False) -> dict:
    specs = build_specs(base_cfg, groups, seeds)
    if max_runs:
        specs = specs[:max_runs]
    if not quiet:
        print(f"[run_all] 共 {len(specs)} 个 run")
    t_all = time.time()
    rows = []
    for i, spec in enumerate(specs, 1):
        t0 = time.time()
        cfg = copy.deepcopy(base_cfg)
        cfg = _apply(cfg, spec.overrides)
        cfg["benchmark"]["scenario"] = spec.scenario
        try:
            result = run_one(cfg, group=spec.group, seed=spec.seed)
        except Exception as e:  # 单个 run 失败不终止整个矩阵，但要保留失败记录
            print(f"[{i}/{len(specs)}] FAIL {spec.group} {spec.scenario} seed={spec.seed}: {e}")
            rows.append({"group": spec.group, "scenario": spec.scenario, "seed": spec.seed,
                         "status": f"failed: {e}", "note": spec.note})
            continue
        r = _row(spec, result)
        rows.append(r)
        if not quiet:
            print(f"[{i}/{len(specs)}] {spec.group:<16s} {spec.scenario:<20s} "
                  f"seed={spec.seed} gain={r['gain_points']:+.2f}pt "
                  f"mF1={r['method_stream_f1']:.3f} static={r['static_stream_f1']:.3f} "
                  f"cost={r['cost_scalar']:.2f} ({time.time()-t0:.1f}s)")
    summary = aggregate(rows, base_cfg, wall_s=time.time() - t_all)
    _write_outputs(rows, summary)
    if not quiet:
        print(f"[run_all] 完成，用时 {summary['wall_s']:.1f}s；汇总见 results/SUMMARY.md")
    return summary


def _apply(base: dict, overrides: dict) -> dict:
    from src.core.config import _deep_merge

    return _deep_merge(base, overrides)


def _row(spec: RunSpec, result: dict) -> dict:
    h = result["hidden"]
    s = result["static_baseline_hidden"]
    c = result["cost"]
    cmp_ = result["comparison"]
    return {
        "group": spec.group,
        "scenario": spec.scenario,
        "seed": spec.seed,
        "note": spec.note,
        "status": "ok",
        "method_stream_f1": h["stream_macro_f1"],
        "static_stream_f1": s["stream_macro_f1"],
        "gain_points": cmp_["stream_gain_points"],
        "method_final_phase_f1": h["final_phase_macro_f1"],
        "static_final_phase_f1": s["final_phase_macro_f1"],
        "win_windows": cmp_["win_windows"],
        "n_windows": cmp_["n_windows"],
        "compute_units": c["compute_units"],
        "labels": c["labels"],
        "tokens": c["tokens"],
        "cost_scalar": c["scalar"],
        "gain_per_cost": (cmp_["stream_gain_points"] / c["scalar"]) if c["scalar"] > 0 else 0.0,
        "n_deploy": sum(1 for x in result["timeline"] if x.get("deploy")),
        "n_rollback": sum(1 for x in result["timeline"] if x.get("rollback")),
        "n_trigger": sum(1 for x in result["timeline"] if x.get("triggered")),
        "audit_verified": bool(result["audit"]["verified"]),
        "driver": result["meta"]["driver"],
        "wall_s": result["meta"]["wall_s"],
        "run_id": result["meta"]["run_id"],
        "per_phase_method": h["per_phase_macro_f1"],
        "per_phase_static": s["per_phase_macro_f1"],
    }


def aggregate(rows: list[dict], base_cfg: dict, wall_s: float = 0.0) -> dict:
    ok = [r for r in rows if r.get("status") == "ok"]
    summary = {
        "wall_s": wall_s,
        "n_runs": len(rows),
        "n_ok": len(ok),
        "groups": {},
        "acceptance": {},
        "config": {"benchmark": base_cfg["benchmark"],
                   "budget": base_cfg["budget"]},
    }
    for g in sorted({r["group"] for r in ok}):
        gs = [r for r in ok if r["group"] == g]
        summary["groups"][g] = {
            "n": len(gs),
            "scenarios": sorted({r["scenario"] for r in gs}),
            "gain_points_mean": float(np.mean([r["gain_points"] for r in gs])),
            "gain_points_std": float(np.std([r["gain_points"] for r in gs])),
            "method_f1_mean": float(np.mean([r["method_stream_f1"] for r in gs])),
            "static_f1_mean": float(np.mean([r["static_stream_f1"] for r in gs])),
            "cost_scalar_mean": float(np.mean([r["cost_scalar"] for r in gs])),
            "labels_mean": float(np.mean([r["labels"] for r in gs])),
            "compute_mean": float(np.mean([r["compute_units"] for r in gs])),
            "tokens_mean": float(np.mean([r["tokens"] for r in gs])),
            "gain_per_cost_mean": float(np.mean([r["gain_per_cost"] for r in gs])),
            "deploys_mean": float(np.mean([r["n_deploy"] for r in gs])),
            "rollbacks_mean": float(np.mean([r["n_rollback"] for r in gs])),
            "audit_all_ok": bool(all(r["audit_verified"] for r in gs)),
            "per_scenario": {
                scn: {
                    "gain_points": float(np.mean([r["gain_points"] for r in gs
                                                 if r["scenario"] == scn])),
                    "method_f1": float(np.mean([r["method_stream_f1"] for r in gs
                                               if r["scenario"] == scn])),
                    "static_f1": float(np.mean([r["static_stream_f1"] for r in gs
                                               if r["scenario"] == scn])),
                    "cost_scalar": float(np.mean([r["cost_scalar"] for r in gs
                                                 if r["scenario"] == scn])),
                }
                for scn in sorted({r["scenario"] for r in gs})
            },
        }
    # ---- 关键验收断言 ----
    main = summary["groups"].get("main")
    if main:
        summary["acceptance"]["not_worse_than_static"] = bool(
            main["method_f1_mean"] >= main["static_f1_mean"] - 1e-9)
        summary["acceptance"]["stream_gain_points"] = main["gain_points_mean"]
    summary["acceptance"]["audit_verified_all"] = bool(
        all(r["audit_verified"] for r in ok)) if ok else False
    summary["acceptance"]["no_failed_runs"] = bool(len(ok) == len(rows))
    return summary


def _write_outputs(rows: list[dict], summary: dict) -> None:
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    tables = RESULTS_DIR / "tables"
    tables.mkdir(parents=True, exist_ok=True)

    scalar_keys = ["group", "scenario", "seed", "note", "status", "method_stream_f1",
                   "static_stream_f1", "gain_points", "method_final_phase_f1",
                   "static_final_phase_f1", "win_windows", "n_windows", "compute_units",
                   "labels", "tokens", "cost_scalar", "gain_per_cost", "n_deploy",
                   "n_rollback", "n_trigger", "audit_verified", "driver", "wall_s", "run_id"]
    with open(tables / "runs.csv", "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k) for k in scalar_keys})

    for name in ["main", "scenarios", "ablations", "budget"]:
        subset = [r for r in rows if r["group"] == name
                  or (name == "ablations" and r["group"].startswith("ab-"))
                  or (name == "budget" and r["group"].startswith("budget-"))]
        if not subset:
            continue
        with open(tables / f"{name}.csv", "w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=scalar_keys, extrasaction="ignore")
            w.writeheader()
            for r in subset:
                w.writerow({k: r.get(k) for k in scalar_keys})

    with open(RESULTS_DIR / "SUMMARY.json", "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=1)
    with open(RESULTS_DIR / "SUMMARY.md", "w", encoding="utf-8") as f:
        f.write(_summary_md(rows, summary))


def _summary_md(rows: list[dict], summary: dict) -> str:
    lines = ["# 实验汇总（自动生成）", ""]
    lines.append(f"- runs: {summary['n_ok']}/{summary['n_runs']} 成功，总耗时 {summary['wall_s']:.1f}s")
    lines.append(f"- 验收断言：{json.dumps(summary['acceptance'], ensure_ascii=False)}")
    lines.append("")
    lines.append("## 分组均值")
    lines.append("| 组 | n | 静态 F1 | 自进化 F1 | 增益(pt) | 成本(scalar) | 增益/成本 | 部署 | 回滚 | 审计 |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|")
    for g, v in summary["groups"].items():
        lines.append(f"| {g} | {v['n']} | {v['static_f1_mean']:.4f} | {v['method_f1_mean']:.4f} | "
                     f"{v['gain_points_mean']:+.2f} | {v['cost_scalar_mean']:.2f} | "
                     f"{v['gain_per_cost_mean']:.2f} | {v['deploys_mean']:.1f} | "
                     f"{v['rollbacks_mean']:.1f} | {v['audit_all_ok']} |")
    lines.append("")
    lines.append("## 分场景（groups=scenarios / main）")
    lines.append("| 场景 | 静态 F1 | 自进化 F1 | 增益(pt) | 成本 |")
    lines.append("|---|---|---|---|---|")
    for g in ["main", "scenarios"]:
        if g not in summary["groups"]:
            continue
        for scn, v in summary["groups"][g]["per_scenario"].items():
            lines.append(f"| {scn} | {v['static_f1']:.4f} | {v['method_f1']:.4f} | "
                         f"{v['gain_points']:+.2f} | {v['cost_scalar']:.2f} |")
    lines.append("")
    lines.append("## 明细")
    hdr = "| group | scenario | seed | method F1 | static F1 | 增益 | cost | labels | compute | deploy | rollback | 审计 |"
    lines.append(hdr)
    lines.append("|" + "---|" * 12)
    for r in rows:
        if r.get("status") != "ok":
            lines.append(f"| {r.get('group')} | {r.get('scenario')} | {r.get('seed')} | "
                         f"FAILED | | | | | | | | |")
            continue
        lines.append(f"| {r['group']} | {r['scenario']} | {r['seed']} | {r['method_stream_f1']:.4f} | "
                     f"{r['static_stream_f1']:.4f} | {r['gain_points']:+.2f} | "
                     f"{r['cost_scalar']:.2f} | {r['labels']} | {r['compute_units']:.1f} | "
                     f"{r['n_deploy']} | {r['n_rollback']} | {r['audit_verified']} |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.json")
    ap.add_argument("--groups", default="main")
    ap.add_argument("--seeds", default="7,17,27")
    ap.add_argument("--max-runs", type=int, default=0)
    args = ap.parse_args()
    base = load_config(args.config)
    run_matrix(base, args.groups.split(","), [int(s) for s in args.seeds.split(",")],
               max_runs=args.max_runs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
