"""官方 CPU-A（气体传感器漂移）上的无监督/少标签适应对照实验。

基座：官方 public/train.jsonl（batch1，445 样本）训练；
测试：UCI 原始 batches 3–10（每批时间顺序前 20% 作为可见适应集，后 80% 作为测试集）；
对照：
  - static      : 基座模型不更新
  - TTA          : 零标签仿射自适应（RECALIBRATE_AFFINE 原语）
  - head-FT      : 20% 标签微调末层（LAST_LAYER_FT 近似）
  - full-FT+replay: 20% 标签 + 官方 batch1 回放全量微调（FULL_FINETUNE_REPLAY 近似）
输出：results/official/cpu_a_adapt/summary.{json,md}

用法：PY=.../python3
      $PY scripts/eval_official_cpu_a_adapt.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.metrics import evaluate      # noqa: E402
from src.core.model import MLP             # noqa: E402

OFF = ROOT / "data" / "official" / "cpu-a-guide-20261110-v1"
RAW = ROOT / "data" / "external" / "gas_sensor_drift" / "Dataset"
OUT = ROOT / "results" / "official" / "cpu_a_adapt"
CLASSES = ["Ethanol", "Ethylene", "Ammonia", "Acetaldehyde", "Acetone", "Toluene"]


def load_jsonl(path: Path):
    xs, ys = [], []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        d = json.loads(line)
        xs.append(d["inputs"]["features"])
        ys.append(CLASSES.index(d["answer"]))
    return np.asarray(xs, float), np.asarray(ys, np.int64)


def load_batch(batch: int):
    xs, ys = [], []
    for line in (RAW / f"batch{batch}.dat").read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if not parts:
            continue
        y = int(parts[0]) - 1
        v = np.zeros(128, float)
        for tok in parts[1:]:
            k, val = tok.split(":")
            v[int(k) - 1] = float(val)
        xs.append(v)
        ys.append(y)
    return np.asarray(xs, float), np.asarray(ys, np.int64)


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    base_x, base_y = load_jsonl(OFF / "public" / "train.jsonl")
    val_x, val_y = load_jsonl(OFF / "public" / "validation.jsonl")
    base = MLP(128, 6, hidden=[64, 32], seed=7)
    base.fit(base_x, base_y, epochs=40, lr=0.004, batch=128, seed=7)

    rows = []
    for batch in range(3, 11):
        x, y = load_batch(batch)
        n = len(x)
        cut = max(20, int(n * 0.20))
        ax, ay = x[:cut], y[:cut]
        tx, ty = x[cut:], y[cut:]
        static_f1 = evaluate(ty, base.predict(tx), 6)["macro_f1"]
        # TTA：零标签仿射自适应
        tta = base.clone()
        tta.adaptive = True
        tta.adaptive_blend = 1.0
        tta_f1 = evaluate(ty, tta.predict(tx), 6)["macro_f1"]
        # 末层微调（20% 标签）
        head = base.clone()
        head.fit_head(ax, ay, epochs=80, lr=0.01, batch=min(64, len(ax)), seed=7)
        head_f1 = evaluate(ty, head.predict(tx), 6)["macro_f1"]
        # 全量微调 + 官方 batch1 回放
        full = base.clone()
        rx = np.vstack([base_x, ax])
        ry = np.concatenate([base_y, ay])
        full.fit(rx, ry, epochs=25, lr=0.002, batch=128, seed=7)
        full_f1 = evaluate(ty, full.predict(tx), 6)["macro_f1"]
        rows.append({"batch": batch, "n": n, "adapt_labels": cut, "test_n": len(ty),
                     "static": float(static_f1), "tta": float(tta_f1),
                     "head_ft": float(head_f1), "full_ft_replay": float(full_f1)})
        print(f"batch{batch} n={n} static={static_f1:.4f} tta={tta_f1:.4f} "
              f"head={head_f1:.4f} full={full_f1:.4f}")

    mean = {k: float(np.mean([r[k] for r in rows])) for k in ("static", "tta", "head_ft", "full_ft_replay")}
    summary = {"protocol": "base=official batch1; adapt=first 20% of each test batch; test=remaining 80%",
               "rows": rows, "mean": mean}
    (OUT / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    lines = ["# 官方 CPU-A 数据上的适应对照（气体传感器漂移）", "",
             "> 基座：官方 batch1 训练；每个测试批次按时间前 20% 作为可见适应集（有标签），后 80% 作为测试集。",
             "", "| 批次 | 测试 n | static | TTA(0标签) | 末层FT(20%标签) | 全量FT+回放(20%标签) |",
             "|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| batch{r['batch']} | {r['test_n']} | {r['static']:.4f} | {r['tta']:.4f} | "
                     f"{r['head_ft']:.4f} | {r['full_ft_replay']:.4f} |")
    lines += ["", f"**平均**：static **{mean['static']:.4f}**；TTA **{mean['tta']:.4f}**"
                  f"（{mean['tta']-mean['static']:+.4f}）；末层FT **{mean['head_ft']:.4f}**"
                  f"（{mean['head_ft']-mean['static']:+.4f}）；全量FT **{mean['full_ft_replay']:.4f}**"
                  f"（{mean['full_ft_replay']-mean['static']:+.4f}）", "",
              "说明：本地实验；官方隐藏测试由平台独立评测。", ""]
    (OUT / "summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
