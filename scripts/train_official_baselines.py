"""在官方 CPU-A/B/C 公开数据上续训/调参，并汇总验证指标。

数据来源（均已解压到 data/official/）：
  cpu-a-guide-20261110-v1   气体传感器漂移 6 类（classification）
  cpu-b-rul-20261110-v1     XJTU-SY 轴承 RUL 分钟（regression）
  cpu-c-guide-20261110-v1   NASA C-MAPSS 涡扇 RUL 循环（regression）

做法：调用官方 baselines/train.py，从官方 checkpoint 续训（合法协议：只用 public/train.jsonl，
validation 只做选择），跑一个小超参 sweep（epochs × lr），选验证最优，输出：
  results/official/sweeps/<profile>_<tag>/（官方脚本产物：checkpoint/training/history）
  results/official/official_training_summary.json / .md

用法（bundled python，numpy 2.x）：
  PY=.../python3
  $PY scripts/train_official_baselines.py
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "official"
OUT = ROOT / "results" / "official"
PY = sys.executable

PACKAGES = {
    "A": DATA / "cpu-a-guide-20261110-v1",
    "B": DATA / "cpu-b-rul-20261110-v1",
    "C": DATA / "cpu-c-guide-20261110-v1",
}
SWEEP = {
    "A": [(10, 0.003), (20, 0.003), (40, 0.003), (20, 0.001), (40, 0.001)],
    "B": [(10, 0.003), (20, 0.003), (40, 0.003), (20, 0.001), (40, 0.001)],
    "C": [(5, 0.002), (10, 0.002), (20, 0.002), (10, 0.001)],
}
METRIC = {"A": ("macro_f1", "max"), "B": ("rmse", "min"), "C": ("rmse", "min")}


def run_one(profile: str, epochs: int, lr: float) -> dict:
    pkg = PACKAGES[profile]
    tag = f"ep{epochs}_lr{str(lr).replace('.', '')}"
    out = OUT / "sweeps" / f"{profile}_{tag}"
    if out.exists():
        shutil.rmtree(out)
    cmd = [PY, "baselines/train.py", "--profile", profile,
           "--train", "public/train.jsonl", "--validation", "public/validation.jsonl",
           "--from-baseline", "checkpoint.npz", "--epochs", str(epochs),
           "--learning-rate", str(lr), "--output", str(out), "--stage-id", f"sweep-{tag}"]
    if profile == "B":
        cmd += ["--label-provenance", "label_provenance.json"]
    r = subprocess.run(cmd, cwd=str(pkg), capture_output=True, text=True)
    if r.returncode != 0:
        return {"profile": profile, "epochs": epochs, "lr": lr, "ok": False,
                "error": (r.stderr or r.stdout)[-500:]}
    report = json.loads((out / "training.json").read_text(encoding="utf-8"))
    before, after = report.get("validation_before", {}), report.get("validation_after", {})
    return {"profile": profile, "epochs": epochs, "lr": lr, "ok": True,
            "before": before, "after": after,
            "parent": report.get("parent_checkpoint_sha256"),
            "checkpoint": report.get("checkpoint_sha256"),
            "steps": [report.get("initial_steps"), report.get("final_steps")],
            "dir": str(out.relative_to(ROOT))}


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    results = []
    for profile, grid in SWEEP.items():
        for epochs, lr in grid:
            print(f"[{profile}] epochs={epochs} lr={lr} ...")
            res = run_one(profile, epochs, lr)
            results.append(res)
            print("   ", "ok" if res["ok"] else res.get("error", "")[-120:])
    summary = {"source": "official CPU-A/B/C public train/validation",
               "metric": METRIC, "runs": results}
    # 每个 profile 选最优
    best = {}
    for profile, (metric, mode) in METRIC.items():
        ok = [r for r in results if r["profile"] == profile and r["ok"]]
        key = lambda r: r["after"].get(metric, float("-inf") if mode == "max" else float("inf"))
        b = sorted(ok, key=key, reverse=(mode == "max"))[0] if ok else None
        best[profile] = b
        if b:
            tag = f"ep{b['epochs']}_lr{str(b['lr']).replace('.', '')}"
            target = OUT / f"{profile}_best"
            if target.exists():
                shutil.rmtree(target)
            shutil.copytree(ROOT / b["dir"], target)
            b["best_copy"] = str(target.relative_to(ROOT))
    summary["best"] = best
    (OUT / "official_training_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")

    lines = ["# 官方 CPU-A/B/C 数据包训练汇总", "",
             "> 数据：官方 public/train.jsonl（训练）+ public/validation.jsonl（选择），",
             "> 从官方 checkpoint 续训；验证指标仅本地参考，不代表隐藏测试。", "",
             "| Profile | 任务 | 初始验证 | 最优验证 | 提升 | 配置 |", "|---|---|---|---|---|---|"]
    label = {"A": "气体传感器 6 类（Macro-F1）", "B": "XJTU-SY RUL（RMSE 分钟）",
             "C": "C-MAPSS RUL（RMSE 循环）"}
    for p in ("A", "B", "C"):
        bs = [r for r in results if r["profile"] == p and r["ok"]]
        if not bs:
            continue
        b = best[p]
        after = b["after"]; before = b["before"]
        if p == "A":
            lines.append(f"| {p} | {label[p]} | {before.get('macro_f1'):.4f} | "
                         f"{after.get('macro_f1'):.4f} | "
                         f"{after.get('macro_f1')-before.get('macro_f1'):+.4f} | "
                         f"epochs={b['epochs']}, lr={b['lr']} |")
        else:
            lines.append(f"| {p} | {label[p]} | {before.get('rmse'):.2f} | {after.get('rmse'):.2f} | "
                         f"{after.get('rmse')-before.get('rmse'):+.2f} | "
                         f"epochs={b['epochs']}, lr={b['lr']} |")
    lines += ["", "最优 checkpoint 目录：`results/official/{A,B,C}_best/`；全部 sweep：`results/official/sweeps/`。",
              "官方脚本要求：输出目录必须全新；B 需 label_provenance（RUL 定义），本实验已声明为 draft。",
              "说明：本地验证集指标用于选模型，平台隐藏测试由官方独立评测。", ""]
    (OUT / "official_training_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
