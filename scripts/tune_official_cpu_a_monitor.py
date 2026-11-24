"""针对官方 CPU-A（气体传感器跨批次漂移）重校准 monitor/门禁配置，目标是让整包闭环取得正增益。

基座/数据协议与 scripts/eval_official_cpu_a.py 一致：
  base = 官方 public/train.jsonl（batch1）；部署流 = 原始 batches 3–10（70% 可见 / 30% 隐藏）。
先单 seed 网格搜索，再用 3 seeds 验证最优配置。

用法：PY=.../python3
      $PY scripts/tune_official_cpu_a_monitor.py
输出：
  results/official/cpu_a_tuning/<config>.json
  results/official/cpu_a_tuning/summary.md / .json
  results/official/cpu_a_evolving_tuned/（最优配置 3 seeds）
"""

from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from src.core.config import RunContext, load_config  # noqa: E402
from src.loop import SelfEvolvingLoop, new_run_id         # noqa: E402
from eval_official_cpu_a import build_benchmark           # noqa: E402

OUT = ROOT / "results" / "official" / "cpu_a_tuning"
BEST_OUT = ROOT / "results" / "official" / "cpu_a_evolving_tuned"

CONFIGS = [
    {"name": "c0_current", "monitor": {}, "gates": {}, "budget": {}, "diagnosis": {}},
    {"name": "c1_probe8", "monitor": {"mmd_threshold": 0.35, "ks_count_threshold": 8,
                                       "probe_max_labels": 8, "probe_labels": 8,
                                       "probe_min_labels": 6, "adapt_reserve_frac": 0.50,
                                       "probe_every": 3},
     "gates": {"min_gain_points": 0.005, "min_probe_gain": 0.0},
     "budget": {"labels_total": 260}, "diagnosis": {"label_shift_tv": 0.25}},
    {"name": "c2_probe12", "monitor": {"mmd_threshold": 0.25, "ks_count_threshold": 8,
                                        "probe_max_labels": 12, "probe_labels": 12,
                                        "probe_min_labels": 8, "adapt_reserve_frac": 0.40,
                                        "probe_every": 4, "proxy_drop_threshold": 0.03},
     "gates": {"min_gain_points": 0.005, "min_probe_gain": 0.0},
     "budget": {"labels_total": 260}, "diagnosis": {"label_shift_tv": 0.25}},
    {"name": "c3_probe16", "monitor": {"mmd_threshold": 0.20, "ks_count_threshold": 6,
                                        "probe_max_labels": 16, "probe_labels": 16,
                                        "probe_min_labels": 10, "adapt_reserve_frac": 0.35,
                                        "probe_every": 5, "proxy_drop_threshold": 0.025},
     "gates": {"min_gain_points": 0.003, "min_probe_gain": 0.0},
     "budget": {"labels_total": 260}, "diagnosis": {"label_shift_tv": 0.30}},
    {"name": "c4_probe16_budget", "monitor": {"mmd_threshold": 0.20, "ks_count_threshold": 6,
                                               "probe_max_labels": 16, "probe_labels": 16,
                                               "probe_min_labels": 10, "adapt_reserve_frac": 0.35,
                                               "probe_every": 5, "proxy_drop_threshold": 0.025},
     "gates": {"min_gain_points": 0.003, "min_probe_gain": 0.0},
     "budget": {"labels_total": 360}, "diagnosis": {"label_shift_tv": 0.30}},
    {"name": "c5_lowmmd", "monitor": {"mmd_threshold": 0.12, "ks_count_threshold": 6,
                                       "probe_max_labels": 16, "probe_labels": 16,
                                       "probe_min_labels": 10, "adapt_reserve_frac": 0.30,
                                       "probe_every": 4, "proxy_drop_threshold": 0.02},
     "gates": {"min_gain_points": 0.003, "min_probe_gain": 0.0},
     "budget": {"labels_total": 360}, "diagnosis": {"label_shift_tv": 0.35}},
    {"name": "c6_affinefirst", "monitor": {"mmd_threshold": 0.20, "ks_count_threshold": 6,
                                            "probe_max_labels": 16, "probe_labels": 16,
                                            "probe_min_labels": 10, "adapt_reserve_frac": 0.35,
                                            "probe_every": 4, "proxy_drop_threshold": 0.025},
     "gates": {"min_gain_points": 0.003, "min_probe_gain": 0.0},
     "budget": {"labels_total": 360},
     "diagnosis": {"label_shift_tv": 0.35, "affine_min_gain": 0.01,
                    "affine_recoverability": 0.5}},
    {"name": "c7_highprobe", "monitor": {"mmd_threshold": 0.25, "ks_count_threshold": 8,
                                          "probe_max_labels": 24, "probe_labels": 24,
                                          "probe_min_labels": 12, "adapt_reserve_frac": 0.30,
                                          "probe_every": 6, "proxy_drop_threshold": 0.03},
     "gates": {"min_gain_points": 0.003, "min_probe_gain": 0.0},
     "budget": {"labels_total": 420}, "diagnosis": {"label_shift_tv": 0.30}},
]


def apply_overrides(cfg: dict, item: dict) -> dict:
    out = copy.deepcopy(cfg)
    out["seed"] = 7
    out["run"]["name"] = "official-cpu-a-tune"
    out["data"] = {"sanitize": True, "clip_quantile": 0.9995, "fill": "median"}
    for k, v in (item.get("monitor") or {}).items():
        out.setdefault("engine", {}).setdefault("monitor", {})[k] = v
    for k, v in (item.get("diagnosis") or {}).items():
        out.setdefault("engine", {}).setdefault("diagnosis", {})[k] = v
    for k, v in (item.get("gates") or {}).items():
        out.setdefault("gates", {})[k] = v
    for k, v in (item.get("budget") or {}).items():
        out.setdefault("budget", {})[k] = v
    return out


def run_one(item: dict, seed: int = 7) -> dict:
    cfg = apply_overrides(load_config("configs/main.json"), item)
    cfg["seed"] = seed
    bench = build_benchmark(seed)
    ctx = RunContext(cfg=cfg, run_id=new_run_id(f"official-cpu-a-{item['name']}-s{seed}"))
    loop = SelfEvolvingLoop(cfg, ctx, benchmark=bench)
    res = loop.run()
    comp = res.get("comparison", {})
    return {
        "config": item["name"], "seed": seed,
        "method": res["hidden"]["stream_macro_f1"],
        "static": res["static_baseline_hidden"]["stream_macro_f1"],
        "gain": comp.get("stream_gain_points"),
        "labels": res["cost"]["labels"], "compute_units": res["cost"]["compute_units"],
        "deploys": sum(1 for x in res["timeline"] if x.get("deploy")),
        "rollbacks": sum(1 for x in res["timeline"] if x.get("rollback")),
        "triggers": sum(1 for x in res["timeline"] if x.get("triggered")),
        "kinds": {k: sum(1 for x in res["timeline"] if x.get("diagnosis_kind") == k)
                  for k in sorted({x.get("diagnosis_kind") for x in res["timeline"] if x.get("diagnosis_kind")})},
        "raw": res,
    }


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for item in CONFIGS:
        print(f"== {item['name']}")
        r = run_one(item, seed=7)
        (OUT / f"{item['name']}.json").write_text(
            json.dumps(r["raw"], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        results.append({k: v for k, v in r.items() if k != "raw"})
        print(f"   gain={r['gain']:+.2f}pt method={r['method']:.4f} static={r['static']:.4f} "
              f"labels={r['labels']:.0f} deploy={r['deploys']} trig={r['triggers']} kinds={r['kinds']}")

    best = max(results, key=lambda x: x["gain"] if x["gain"] is not None else -999)
    summary = {"protocol": "official CPU-A; base=batch1; stream=batches3-10 (70% visible/30% hidden); seed7 tuning",
               "runs": results, "best": best["config"]}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 官方 CPU-A monitor 阈值重校准（单 seed 网格）", "",
             "| 配置 | 方法 F1 | 静态 F1 | 增益(pt) | 标注 | 部署 | 触发 |", "|---|---|---|---|---|---|---|"]
    for r in results:
        lines.append(f"| {r['config']} | {r['method']:.4f} | {r['static']:.4f} | {r['gain']:+.2f} | "
                     f"{r['labels']:.0f} | {r['deploys']} | {r['triggers']} |")
    lines += ["", f"最优配置：**{best['config']}**（seed7 增益 {best['gain']:+.2f}pt）；"
                  f"下一步用 3 seeds 验证。", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))

    # 3 seeds 验证最优
    best_item = next(x for x in CONFIGS if x["name"] == best["config"])
    BEST_OUT.mkdir(parents=True, exist_ok=True)
    seeds = [7, 17, 27]
    rows = []
    for seed in seeds:
        r = run_one(best_item, seed=seed)
        (BEST_OUT / f"seed{seed}.json").write_text(
            json.dumps(r["raw"], ensure_ascii=False, indent=1, default=str), encoding="utf-8")
        rows.append({k: v for k, v in r.items() if k != "raw"})
        print(f"best seed{seed}: gain={r['gain']:+.2f}pt method={r['method']:.4f} "
              f"static={r['static']:.4f} labels={r['labels']:.0f} deploy={r['deploys']}")
    mean = {k: float(np.mean([x[k] for x in rows])) for k in ("method", "static", "gain", "labels")}
    best_summary = {"config": best["config"], "item": best_item, "seeds": rows, "mean": mean}
    (BEST_OUT / "summary.json").write_text(json.dumps(best_summary, ensure_ascii=False, indent=1),
                                          encoding="utf-8")
    lines2 = [f"# 官方 CPU-A 最优配置 3 seeds 验证（{best['config']}）", "",
              f"- 平均：静态 **{mean['static']:.4f}** → 自进化 **{mean['method']:.4f}**"
              f"（**{mean['gain']:+.2f}pt**），标注 {mean['labels']:.0f}", "",
              "| seed | 静态 | 自进化 | 增益(pt) | 标注 | 部署/回滚 |", "|---|---|---|---|---|---|"]
    for r in rows:
        lines2.append(f"| {r['seed']} | {r['static']:.4f} | {r['method']:.4f} | {r['gain']:+.2f} | "
                      f"{r['labels']:.0f} | {r['deploys']}/{r['rollbacks']} |")
    lines2 += ["", "配置见 `results/official/cpu_a_tuning/summary.{json,md}`；结果 JSON 见本目录。", ""]
    (BEST_OUT / "summary.md").write_text("\n".join(lines2), encoding="utf-8")
    print("\n".join(lines2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
