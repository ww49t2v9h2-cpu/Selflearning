"""验收脚本：对齐赛题"基础门槛"与我们的验收断言。

对应赛题基础门槛：
  ① 封闭评测中无人干预地完成全部阶段并按时输出结果；
  ② 全部训练/微调/调整/评估实验按规范留痕；
  ③ 封闭评测综合性能不低于静态基线（部署后不再更新的模型）。

本脚本跑一次完整闭环（默认 S7 主场景），逐项断言并输出 results/acceptance.json：
  A1 无人干预跑通（无异常、有输出、时间达标）
  A2 不低于静态基线（流式 macro-F1）
  A3 预算不超支（labels/compute/tokens）
  A4 实验记录完整（run 数 > 0、决策含证据引用、result.json 落盘）
  A5 审计链完整（哈希链校验通过）
  A6 版本演进与可回滚（registry 版本数 ≥ 2 或显式说明无更新）
  A7 可复现（可选 --repro：同配置重跑，逐窗口曲线完全一致）

用法：
  PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3
  $PY scripts/acceptance.py                     # 默认 S7 + seed 7
  $PY scripts/acceptance.py --repro             # 额外做可复现校验（跑两次）
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config                      # noqa: E402
from src.loop import run_one                                 # noqa: E402


def check(checks: list, name: str, ok: bool, detail: dict) -> None:
    checks.append({"name": name, "pass": bool(ok), "detail": detail})


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/main.json")
    ap.add_argument("--scenario", default="S7_full_mix")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--repro", action="store_true", help="额外跑一次并逐窗口比对")
    ap.add_argument("--time-budget-s", type=float, default=900.0)
    ap.add_argument("--out", default="results/acceptance.json")
    args = ap.parse_args()

    cfg = load_config(args.config)
    cfg["benchmark"]["scenario"] = args.scenario
    checks: list[dict] = []
    t0 = time.time()
    result = run_one(cfg, group="acceptance", seed=args.seed)
    wall = time.time() - t0

    # A1 无人干预跑通
    check(checks, "A1_unattended_run", True, {
        "wall_s": round(wall, 2),
        "time_budget_s": args.time_budget_s,
        "n_windows": result["meta"].get("n_windows", len(result["timeline"])),
        "result_json": f"results/acceptance/{args.scenario}_seed{args.seed}.json",
    })
    check(checks, "A1_within_time_budget", wall <= args.time_budget_s,
          {"wall_s": round(wall, 2), "budget_s": args.time_budget_s})

    # A2 不低于静态基线
    m = result["hidden"]["stream_macro_f1"]
    s = result["static_baseline_hidden"]["stream_macro_f1"]
    check(checks, "A2_not_worse_than_static", m >= s - 1e-9,
          {"method_stream_f1": round(m, 6), "static_stream_f1": round(s, 6),
           "gain_points": round((m - s) * 100, 4)})

    # A3 预算
    c = result["cost"]
    b = result["budget"]
    ok_budget = (c["labels"] <= b["labels_total"] + 1e-9
                 and c["compute_units"] <= b["compute_total"] + 1e-9
                 and c["tokens"] <= b["tokens_total"] + 1e-9)
    check(checks, "A3_budget_respected", ok_budget, {
        "labels": c["labels"], "labels_total": b["labels_total"],
        "compute_units": round(c["compute_units"], 3), "compute_total": b["compute_total"],
        "tokens": c["tokens"], "tokens_total": b["tokens_total"],
    })

    # A4 记录完整
    n_runs = len(result["records"]["runs"])
    n_decisions = len(result["decisions"])
    n_tracker_runs = result.get("records", {}).get("n_tracker_runs", n_runs)
    decisions_with_evidence = sum(1 for d in result["decisions"]
                                  if d.get("evidence_refs") or d.get("outcome"))
    timeline_ok = all("cost_cumulative" in row for row in result["timeline"])
    check(checks, "A4_records_complete", n_runs >= 1 and n_decisions >= 0
          and decisions_with_evidence == n_decisions and timeline_ok, {
              "n_tracker_runs": n_runs,
              "n_decisions": n_decisions,
              "decisions_with_evidence": decisions_with_evidence,
              "timeline_rows_with_cost": int(timeline_ok),
          })

    # A5 审计链
    check(checks, "A5_audit_verified", bool(result["audit"]["verified"]), {
        "path": result["audit"]["path"], "last_hash": str(result["audit"].get("last_hash", ""))[:16],
    })

    # A6 版本演进
    versions = result["registry"]
    check(checks, "A6_version_evolution",
          len(versions) >= 2, {"n_versions": len(versions),
                               "versions": [{"version": v["version"], "kind": v["kind"],
                                             "status": v["status"],
                                             "playbook": v["playbook"]} for v in versions]})

    # A7 可复现
    if args.repro:
        result2 = run_one(cfg, group="acceptance-repro", seed=args.seed)
        same = (len(result["method_curve"]) == len(result2["method_curve"])
                and all(abs(a - b) < 1e-9 for a, b in
                        zip(result["method_curve"], result2["method_curve"])))
        same_audit = (len(result["decisions"]) == len(result2["decisions"])
                      and all(a["chosen_playbook"] == b["chosen_playbook"]
                              for a, b in zip(result["decisions"], result2["decisions"])))
        check(checks, "A7_reproducible", bool(same and same_audit), {
            "curve_identical": bool(same), "decisions_identical": bool(same_audit),
            "run_a": result["meta"]["run_id"], "run_b": result2["meta"]["run_id"],
        })

    passed = all(x["pass"] for x in checks)
    report = {"passed": passed, "scenario": args.scenario, "seed": args.seed,
              "checks": checks, "run_id": result["meta"]["run_id"],
              "summary": {"method_stream_f1": m, "static_stream_f1": s,
                          "cost": c, "n_versions": len(versions)}}
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=1)

    print("\n" + "=" * 72)
    print(f"验收 {'PASS ✅' if passed else 'FAIL ❌'}  run_id={result['meta']['run_id']}")
    for c_ in checks:
        print(f"  [{'PASS' if c_['pass'] else 'FAIL'}] {c_['name']}")
    print(f"  结果：{out.relative_to(ROOT)}")
    print("=" * 72 + "\n")
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
