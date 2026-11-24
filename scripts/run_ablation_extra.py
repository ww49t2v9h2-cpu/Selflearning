"""补充消融：A7 关闭 TTA（S7）+ 在 S3 上重跑 A1–A6（更新更频繁，机制差异更明显）。

输出 results/{ab-A7_no_tta, ab-S3-*}/*.json，供 fill_docs 的消融表直接扫描。
用法：$PY scripts/run_ablation_extra.py
"""
from __future__ import annotations
import copy, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.core.config import load_config          # noqa: E402
from src.loop import run_one                     # noqa: E402

ABL = {
    "A1_no_trigger": {"ablation": {"trigger_mode": "always"}},
    "A2_no_diagnosis": {"ablation": {"diagnosis": False}},
    "A3_random_planner": {"ablation": {"planner": "random"}},
    "A4_no_gate": {"ablation": {"gate": False}},
    "A5_no_strategy": {"ablation": {"strategy_learning": False}},
    "A6_no_watchdog": {"ablation": {"watchdog": False}},
    "A7_no_tta": {"playbooks": {"RECALIBRATE_AFFINE": {"blend": 0.0}}},
}


def apply(cfg, ov):
    from src.core.config import _deep_merge
    return _deep_merge(cfg, ov)


def main():
    base = load_config("configs/main.json")
    # 1) A7 关闭 TTA（S7）：证明显性收益来自 TTA
    for seed in (7, 17, 27):
        cfg = apply(base, ABL["A7_no_tta"])
        cfg["benchmark"]["scenario"] = "S7_full_mix"
        run_one(cfg, group="ab-A7_no_tta", seed=seed)
        print("done A7 S7 seed", seed)
    # 2) S3 上重跑 A1-A6（S3 更新次数多，机制差异可观测）
    for name, ov in ABL.items():
        if name == "A7_no_tta":
            continue
        for seed in (7, 17, 27):
            cfg = apply(base, ov)
            cfg["benchmark"]["scenario"] = "S3_condition_switch"
            run_one(cfg, group=f"ab-S3-{name}", seed=seed)
            print("done", name, "S3 seed", seed)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
