"""引擎离线评测/调参台：playbook 恢复数字、诊断命中率、监测器阈值扫描。

用途
----
1. `playbooks`：在代表性漂移窗口上单跑每个 playbook，与静态基线、离线"可恢复
   上界"（用 600 条当前域标签从零重训的强参考）对比，产出 docs/05 的恢复数字表；
2. `diagnosis`：逐场景逐 phase 调用 Diagnoser（预算充足模式），统计 8 类诊断
   命中率（docs/04 的验证数字）；
3. `monitor`：打印每个窗口的监测统计量（阈值标定用）；
4. `sweep`：对单个 (场景, phase, playbook) 扫描超参，用于调优。

运行（在仓库根目录）：
    PY=python3
    $PY -m src.engine.eval_bench playbooks
    $PY -m src.engine.eval_bench diagnosis
    $PY -m src.engine.eval_bench monitor --scenario S7_full_mix
本模块只用于离线分析与调参，不参与闭环决策。
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

from ..core.config import ROOT, load_config, set_global_seed
from ..core.metrics import evaluate, macro_f1
from ..core.model import MLP
from ..core.spec import BudgetState, Diagnosis
from ..data.synth import SyntheticOracle, make_benchmark
from .diagnosis import Diagnoser
from .monitor import DriftMonitor
from .playbooks import PlaybookContext, PlaybookLibrary
from .utils import make_feature_groups

SCENARIOS = ["S1_gradual_wear", "S2_sensor_gain", "S3_condition_switch",
             "S4_novel_class", "S5_repair_cycle", "S6_label_shift", "S7_full_mix"]

# 场景 → 每个 phase 的"可接受诊断"（人工标注的漂移机制真值；混合机制允许多选）
GT_KINDS: dict[str, list] = {
    # 注：轻度漂移（静态性能未明显受损）phase 的可接受答案包含 benign——"不动作"是
    # 正确的运营决策；机制真值以列表形式给出（混合机制/混合损伤允许多选）。
    "S1_gradual_wear": ["benign", ["covariate_affine", "benign"],
                        ["covariate_affine", "benign"], "covariate_affine",
                        "covariate_affine", "covariate_affine"],
    "S2_sensor_gain": ["benign", ["covariate_affine", "benign"],
                       ["covariate_affine", "benign"], "covariate_affine",
                       ["covariate_affine", "noise_inflate"],
                       ["covariate_affine", "noise_inflate"]],
    "S3_condition_switch": ["benign", "condition_switch", "condition_switch",
                            "condition_switch", "condition_switch", "repair"],
    "S4_novel_class": ["benign", ["novel_class", "benign"], "novel_class",
                       "novel_class", "novel_class", ["novel_class", "benign"]],
    "S5_repair_cycle": ["benign", ["covariate_affine", "benign"],
                        "covariate_affine", "repair", "repair",
                        ["covariate_affine", "benign"]],
    "S6_label_shift": ["benign", ["label_shift", "benign"],
                       ["label_shift", "benign"], ["label_shift", "benign"],
                       ["label_shift", "benign"], "repair"],
    "S7_full_mix": ["benign", ["covariate_affine", "benign"], "covariate_affine",
                    "condition_switch", "condition_switch", "novel_class",
                    "novel_class", "repair", ["covariate_affine", "benign"],
                    "covariate_affine", "novel_class",
                    ["concept", "novel_class", "condition_switch"]],
}

# 代表性 (scenario, phase) → 该窗口上应重点比较的 playbook
REPRESENTATIVE = [
    ("S7_full_mix", 2, ["RECALIBRATE_AFFINE", "LAST_LAYER_FT", "FULL_FINETUNE_REPLAY"]),
    ("S3_condition_switch", 3, ["RECALIBRATE_AFFINE", "LAST_LAYER_FT",
                                "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED"]),
    ("S4_novel_class", 3, ["NOVEL_CLASS_EXTEND", "FULL_FINETUNE_REPLAY"]),
    ("S7_full_mix", 5, ["NOVEL_CLASS_EXTEND", "FULL_FINETUNE_REPLAY", "SHADOW_ONLY"]),
    ("S7_full_mix", 11, ["NOVEL_CLASS_EXTEND", "FULL_FINETUNE_REPLAY", "RETRAIN_MIXED",
                         "RECALIBRATE_AFFINE"]),
    ("S2_sensor_gain", 3, ["RECALIBRATE_AFFINE", "LAST_LAYER_FT"]),
    ("S1_gradual_wear", 4, ["RECALIBRATE_AFFINE", "LAST_LAYER_FT", "FULL_FINETUNE_REPLAY"]),
    ("S6_label_shift", 2, ["PRIOR_CORRECTION", "RECALIBRATE_AFFINE", "NOOP"]),
    ("S5_repair_cycle", 3, ["ROLLBACK", "NOOP"]),
    ("S7_full_mix", 8, ["RECALIBRATE_AFFINE", "LAST_LAYER_FT"]),
]


# --------------------------------------------------------------------------
# 基础构造
# --------------------------------------------------------------------------
_ENV_CACHE: dict = {}


def _env_core(scenario: str, seed: int):
    key = (scenario, int(seed))
    if key not in _ENV_CACHE:
        cfg0 = load_config(overrides={"benchmark": {"scenario": scenario}})
        set_global_seed(cfg0["seed"])
        bench = make_benchmark(cfg0, seed=seed)
        model = MLP(bench.d, bench.base_classes, hidden=cfg0["model"]["hidden"], seed=seed)
        model.fit(bench.base_x, bench.base_y, epochs=cfg0["model"]["epochs"],
                  lr=cfg0["model"]["lr"], batch=cfg0["model"]["batch"], seed=seed)
        _ENV_CACHE[key] = (bench, model)
    bench, model = _ENV_CACHE[key]
    return bench, model.clone()


def make_env(scenario: str, seed: int = 7, cfg_overrides: Optional[dict] = None):
    ov = {"benchmark": {"scenario": scenario}}
    if cfg_overrides:
        _deep_update(ov, cfg_overrides)
    cfg = load_config(overrides=ov)
    set_global_seed(cfg["seed"])
    bench, model = _env_core(scenario, seed)
    return cfg, bench, model


def _deep_update(base: dict, override: dict) -> dict:
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            _deep_update(base[k], v)
        else:
            base[k] = copy.deepcopy(v)
    return base


def phase_chunk_indices(bench, phase: int) -> list[int]:
    return [i for i, c in enumerate(bench.chunks) if int(c.phase) == int(phase)]


def hidden_eval(model, bench, ci: int) -> dict:
    hx, hy = bench.hidden_by_chunk[ci]
    pred = model.predict(hx)
    return evaluate(hy, pred, bench.n_classes)


def chunk_true_labels(bench, ci: int) -> np.ndarray:
    chunk = bench.chunks[ci]
    return bench.visible_y[np.asarray(chunk.ids, dtype=np.int64)]


def fake_diagnosis(bench, ci: int, kind: str) -> Diagnosis:
    chunk = bench.chunks[ci]
    cls = list(bench.class_names)
    rec_map = {
        "covariate_affine": ["RECALIBRATE_AFFINE", "LAST_LAYER_FT", "NOOP"],
        "condition_switch": ["FULL_FINETUNE_REPLAY", "LAST_LAYER_FT", "RETRAIN_MIXED"],
        "concept": ["RETRAIN_MIXED", "FULL_FINETUNE_REPLAY", "NOVEL_CLASS_EXTEND"],
        "novel_class": ["NOVEL_CLASS_EXTEND", "SHADOW_ONLY", "FULL_FINETUNE_REPLAY"],
        "label_shift": ["PRIOR_CORRECTION", "NOOP"],
        "repair": ["NOOP", "ROLLBACK"],
        "benign": ["NOOP", "SHADOW_ONLY"],
        "noise_inflate": ["RECALIBRATE_AFFINE", "SHADOW_ONLY", "NOOP"],
    }
    recs = rec_map.get(kind, ["NOOP"])
    n = len(recs)
    return Diagnosis(
        t=int(chunk.t), phase=int(chunk.phase), kind=kind, severity=0.7, confidence=0.8,
        affected_features=list(range(6)),
        affected_groups=[f"S{i}" for i in range(3)],
        affected_classes=([bench.base_classes] if kind == "novel_class" else []),
        recommended=[{"playbook": p, "prior": float((n - i) / (n * (n + 1) / 2)),
                      "why": f"offline eval kind={kind}"} for i, p in enumerate(recs)],
    )


# --------------------------------------------------------------------------
# Playbook 恢复评测
# --------------------------------------------------------------------------
def run_one(scenario: str, phase: int, playbook: str, labels_total: int = 500,
            seed: int = 7, cfg_overrides: Optional[dict] = None,
            chunk_pos: int = -1) -> dict:
    cfg, bench, base_model = make_env(scenario, seed=seed, cfg_overrides=cfg_overrides)
    idxs = phase_chunk_indices(bench, phase)
    ci = idxs[chunk_pos]
    chunk = bench.chunks[ci]
    static = hidden_eval(base_model, bench, ci)["macro_f1"]
    budget = BudgetState(labels_total=labels_total, compute_total=labels_total * 1.0,
                         tokens_total=10 ** 6, updates_total=100)
    oracle = SyntheticOracle(bench.visible_y, budget)
    lib = PlaybookLibrary(cfg, oracle, cfg.get("cost_model", {}))
    kind_list = GT_KINDS[scenario][phase]
    kind = kind_list[0] if isinstance(kind_list, list) else kind_list
    diag = fake_diagnosis(bench, ci, kind)
    ctx = PlaybookContext(model=base_model, chunk=chunk, base_x=bench.base_x,
                          base_y=bench.base_y, config=cfg, n_classes=bench.n_classes,
                          history=[], diagnosis=diag, budget=budget,
                          rng=np.random.default_rng(seed))
    t0 = time.time()
    try:
        res = lib.run(playbook, ctx)
        err = ""
    except Exception as e:  # noqa: BLE001 - 评测台需要把异常也记录成数字
        return {"scenario": scenario, "phase": phase, "playbook": playbook,
                "static_f1": static, "f1": float("nan"), "gain": float("nan"),
                "error": f"{type(e).__name__}: {e}", "wall_s": time.time() - t0,
                "labels": 0, "compute_units": 0.0}
    cand = res.model
    if cand is None:
        # SHADOW_ONLY 不返回模型：用其子 playbook 的模型做指标估计
        cand_f1 = float(res.metrics.get("probe_macro_f1", float("nan")))
        hidden = {"macro_f1": cand_f1}
    else:
        hidden = hidden_eval(cand, bench, ci)
    return {
        "scenario": scenario, "phase": phase, "playbook": playbook,
        "static_f1": float(static), "f1": float(hidden["macro_f1"]),
        "gain": float(hidden["macro_f1"] - static),
        "probe_macro_f1": float(res.metrics.get("probe_macro_f1", float("nan"))),
        "pred_stability": float(res.metrics.get("pred_stability", float("nan"))),
        "class_recall_min": float(res.metrics.get("class_recall_min", float("nan"))),
        "labels": int(res.cost.labels), "compute_units": float(res.cost.compute_units),
        "budget_labels_used": int(budget.labels_used),
        "budget_labels_total": int(labels_total),
        "train_n": int(res.params.get("label_plan", {}).get("n_train", 0)),
        "eval_n": int(res.params.get("label_plan", {}).get("n_eval", 0)),
        "note": res.note, "wall_s": time.time() - t0,
        "params": {k: v for k, v in (res.params or {}).items()
                   if isinstance(v, (int, float, str, bool))},
    }


def upper_bound(scenario: str, phase: int, seed: int = 7, n_cur: int = 600,
                n_base: int = 2000) -> dict:
    """离线可恢复上界（非系统可见）：用大量当前域标签从零重训的强参考。"""
    cfg, bench, _ = make_env(scenario, seed=seed)
    ci = phase_chunk_indices(bench, phase)[-1]
    chunk = bench.chunks[ci]
    y_cur = chunk_true_labels(bench, ci)
    rng = np.random.default_rng(seed + 1)
    nb = min(n_base, len(bench.base_x))
    bi = rng.choice(len(bench.base_x), size=nb, replace=False)
    if n_cur >= len(chunk.x):
        sel = np.arange(len(chunk.x))
    else:
        sel = rng.choice(len(chunk.x), size=n_cur, replace=False)
    x = np.vstack([bench.base_x[bi], chunk.x[sel]])
    y = np.concatenate([bench.base_y[bi], y_cur[sel]])
    m = MLP(bench.d, bench.n_classes, hidden=cfg["model"]["hidden"], seed=seed)
    m.fit(x, y, epochs=30, lr=0.003, batch=128, seed=seed, refit_scaler=True)
    return {"scenario": scenario, "phase": phase,
            "f1": float(hidden_eval(m, bench, ci)["macro_f1"]), "n_cur": int(len(sel))}


def playbook_table(labels_total: int = 500, seed: int = 7,
                   cfg_overrides: Optional[dict] = None) -> list[dict]:
    rows = []
    for scenario, phase, playbooks in REPRESENTATIVE:
        for pb in playbooks:
            row = run_one(scenario, phase, pb, labels_total=labels_total,
                          seed=seed, cfg_overrides=cfg_overrides)
            rows.append(row)
            print(f"  [{scenario} P{phase:02d}] {pb:<22s} "
                  f"static={row['static_f1']:.3f} -> {row['f1']:.3f} "
                  f"(gain {row['gain']:+.3f}, labels={row['labels']}, "
                  f"compute={row['compute_units']:.1f})"
                  + (f" ERROR={row['error']}" if row.get("error") else ""))
    return rows


def print_table(rows: list[dict]) -> None:
    print("\n| 场景 | phase | playbook | 静态F1 | 恢复F1 | 增益 | 标签 | 算力 |")
    print("|---|---|---|---|---|---|---|---|")
    for r in rows:
        print(f"| {r['scenario']} | P{r['phase']:02d} | {r['playbook']} | "
              f"{r['static_f1']:.3f} | {r['f1']:.3f} | {r['gain']:+.3f} | "
              f"{r['labels']} | {r['compute_units']:.1f} |")


# --------------------------------------------------------------------------
# 诊断命中率评测
# --------------------------------------------------------------------------
def diagnosis_hit_rate(seed: int = 7, labels_total: int = 2000,
                       scenarios: Optional[list[str]] = None) -> dict:
    scenarios = scenarios or SCENARIOS
    out = {}
    for scenario in scenarios:
        cfg, bench, base_model = make_env(scenario, seed=seed,
                                          cfg_overrides={"budget": {"labels_total": labels_total}})
        budget = BudgetState(labels_total=labels_total, compute_total=400,
                             tokens_total=10 ** 6, updates_total=100)
        oracle = SyntheticOracle(bench.visible_y, budget)
        mon = DriftMonitor(cfg, bench.d, bench.n_classes)
        mon.reset_reference(np.vstack([bench.chunks[t].x for t in range(
            int(cfg["monitor"].get("warmup_chunks", 4)))]))
        diagr = Diagnoser(cfg, bench.d, bench.n_classes,
                          make_feature_groups(bench.d))
        preds = {}
        for t, chunk in enumerate(bench.chunks):
            rep = mon.update(chunk, base_model, oracle)
            preds.setdefault(int(chunk.phase), []).append((t, rep))
        rows, hits, n_trigger, n_trigger_hits = [], 0, 0, 0
        for phase in range(len(bench.phase_names)):
            gt = GT_KINDS[scenario][phase]
            gt_list = gt if isinstance(gt, list) else [gt]
            ev_chunks = [(t, r) for (t, r) in preds[phase] if r.events]
            if ev_chunks:
                # 闭环中真正会发生诊断的窗口：该 phase 内**第一个触发窗口**
                t, rep = ev_chunks[0]
                diag = diagr.diagnose(bench.chunks[t], base_model, rep, [], oracle)
                pred, conf, cost = diag.kind, round(diag.confidence, 3), diag.cost.labels
                ok = pred in gt_list
                n_trigger += 1
                n_trigger_hits += int(ok)
            else:
                t = preds[phase][-1][0]
                diag = diagr.diagnose(bench.chunks[t], base_model, preds[phase][-1][1],
                                      [], oracle)
                pred, conf, cost = "(未触发诊断)", 0.0, diag.cost.labels
                # 未触发：若该 phase 本就不需要动作（benign/repair）则视为正确
                ok = set(gt_list) <= {"benign", "repair"}
            hits += int(ok)
            rows.append({"phase": phase, "name": bench.phase_names[phase],
                         "gt": gt_list, "pred": pred, "t": int(t),
                         "confidence": conf, "labels": int(cost),
                         "triggered": bool(ev_chunks), "hit": ok})
        rate = hits / len(rows)
        trig_rate = n_trigger_hits / max(1, n_trigger)
        out[scenario] = {"hit_rate": rate, "triggered_hit_rate": trig_rate,
                         "n_trigger": n_trigger, "rows": rows,
                         "n_labels": int(budget.labels_used)}
        print(f"\n[{scenario}] 全 phase 命中 {hits}/{len(rows)}={rate:.1%}；"
              f"触发窗口种类命中 {n_trigger_hits}/{n_trigger}={trig_rate:.1%}；"
              f"标签消耗 {budget.labels_used}")
        for r in rows:
            mark = "✓" if r["hit"] else "✗"
            print(f"  P{r['phase']:02d} t={r['t']:2d} {r['name']:<10s} "
                  f"gt={str(r['gt']):<30s} pred={r['pred']:<16s} "
                  f"conf={r['confidence']:.2f} L={r['labels']} {mark}")
    if scenarios:
        avg = float(np.mean([v["hit_rate"] for v in out.values()]))
        avg_trig = float(np.mean([v["triggered_hit_rate"] for v in out.values()]))
        print(f"\n平均全 phase 命中率（{len(out)} 场景）: {avg:.2%}；"
              f"平均触发窗口命中率: {avg_trig:.2%}")
    return out


# --------------------------------------------------------------------------
# 监测器剖面
# --------------------------------------------------------------------------
def monitor_profile(scenario: str = "S7_full_mix", seed: int = 7) -> list[dict]:
    cfg, bench, base_model = make_env(scenario, seed=seed,
                                      cfg_overrides={"budget": {"labels_total": 800}})
    budget = BudgetState(labels_total=800, compute_total=400, tokens_total=10 ** 6,
                         updates_total=100)
    oracle = SyntheticOracle(bench.visible_y, budget)
    mon = DriftMonitor(cfg, bench.d, bench.n_classes)
    mon.reset_reference(np.vstack([bench.chunks[t].x for t in range(
        int(cfg["monitor"].get("warmup_chunks", 4)))]))
    rows = []
    for t, chunk in enumerate(bench.chunks):
        rep = mon.update(chunk, base_model, oracle)
        s = rep.state
        row = {"t": t, "phase": int(chunk.phase),
               "mmd": round(float(s.get("mmd", 0.0)), 4),
               "mmd_baseline": round(float(s.get("mmd_baseline", 0.0)), 4),
               "ks_count": int(s.get("ks_count", 0)),
               "ks_stat": round(float(s.get("ks_stat_topk", 0.0)), 3),
               "entropy_z": round(float(s.get("entropy_z", 0.0)), 2),
               "ph": round(float(s.get("ph_stat", 0.0)), 2),
               "novelty": round(float(s.get("novelty", 0.0)), 3),
               "probe_acc": (None if not np.isfinite(s.get("probe_acc", float("nan")))
                             else round(float(s["probe_acc"]), 3)),
               "tv": round(float(s.get("pred_hist_tv", 0.0)), 3),
               "repair": round(float(s.get("repair_score", 0.0)), 3),
               "events": [e.detector for e in rep.events],
               "rebase": bool(s.get("rebase", False))}
        rows.append(row)
        ev = ",".join(row["events"]) or "-"
        print(f"t={t:2d} P{row['phase']:02d} mmd={row['mmd']:.3f} "
              f"ksN={row['ks_count']:2d} entz={row['entropy_z']:5.2f} "
              f"ph={row['ph']:5.2f} nov={row['novelty']:5.2f} "
              f"acc={str(row['probe_acc']):>5s} tv={row['tv']:.2f} "
              f"rep={row['repair']:.2f} reb={int(row['rebase'])} ev={ev}")
    return rows


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description="引擎离线评测台")
    ap.add_argument("cmd", nargs="?", default="playbooks",
                    choices=["playbooks", "diagnosis", "monitor", "upper", "all"])
    ap.add_argument("--scenario", default="S7_full_mix")
    ap.add_argument("--labels", type=int, default=500)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", default="")
    args = ap.parse_args(argv)
    t0 = time.time()
    if args.cmd in ("playbooks", "all"):
        print(f"== playbook 恢复评测（budget={args.labels} labels, seed={args.seed}）==")
        rows = playbook_table(labels_total=args.labels, seed=args.seed)
        print_table(rows)
        if args.cmd == "all":
            print("\n== 离线可恢复上界（600 条当前域标签重训，仅参考，系统不可见）==")
            for scenario, phase, _ in REPRESENTATIVE:
                ub = upper_bound(scenario, phase, seed=args.seed)
                print(f"  {scenario} P{phase:02d}: upper≈{ub['f1']:.3f} (n_cur={ub['n_cur']})")
        if args.out:
            Path(args.out).write_text(json.dumps(rows, ensure_ascii=False, indent=1),
                                      encoding="utf-8")
    if args.cmd in ("diagnosis", "all"):
        print("\n== 诊断命中率 ==")
        diagnosis_hit_rate(seed=args.seed)
    if args.cmd in ("monitor", "all", "upper"):
        print(f"\n== 监测器剖面 {args.scenario} ==")
        monitor_profile(scenario=args.scenario, seed=args.seed)
    if args.cmd == "upper":
        for scenario, phase, _ in REPRESENTATIVE:
            ub = upper_bound(scenario, phase, seed=args.seed)
            print(f"{scenario} P{phase:02d}: static? upper≈{ub['f1']:.3f}")
    print(f"\n[done] {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main(sys.argv[1:])
