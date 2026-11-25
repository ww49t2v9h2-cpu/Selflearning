"""冒烟验证：数据基准 + 模型 + 静态基线 + 三种低成本适应原语。

运行：
  PY=python3
  $PY scripts/smoke.py                 # 默认 S7
  $PY scripts/smoke.py S2_sensor_gain   # 指定场景
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.core.config import load_config, set_global_seed          # noqa: E402
from src.core.metrics import evaluate              # noqa: E402
from src.core.model import MLP, class_weights_from_labels, expand_model_classes         # noqa: E402
from src.data.synth import make_benchmark                         # noqa: E402


def eval_hidden(model, bench, chunk_idx, n_classes=5):
    hx, hy = bench.hidden_by_chunk[chunk_idx]
    pred = model.predict(hx)
    return evaluate(hy, pred, n_classes)


def main(scenario: str = "S7_full_mix") -> None:
    cfg = load_config(overrides={"benchmark": {"scenario": scenario}})
    set_global_seed(cfg["seed"])
    t0 = time.time()
    bench = make_benchmark(cfg, seed=cfg["seed"])
    print(f"[bench] {bench.name}: {bench.n_chunks} chunks × "
          f"{len(bench.chunks[0].x)} samples, d={bench.d}, phases={len(bench.phase_names)}")
    print(f"[bench] 生成耗时 {time.time() - t0:.2f}s")

    # 1) 基础训练
    set_global_seed(cfg["seed"])
    rng = np.random.default_rng(cfg["seed"])
    model = MLP(bench.d, bench.base_classes, hidden=cfg["model"]["hidden"], seed=cfg["seed"])
    res = model.fit(bench.base_x, bench.base_y, epochs=cfg["model"]["epochs"],
                    lr=cfg["model"]["lr"], batch=cfg["model"]["batch"],
                    seed=cfg["seed"])
    val = evaluate(bench.val_y, model.predict(bench.val_x), bench.base_classes)
    print(f"[base] train updates={res.sample_updates} wall={res.wall_s:.2f}s val={val}")

    # 2) 静态基线逐窗口 + 逐 phase
    rows = []
    for ci, chunk in enumerate(bench.chunks):
        sc = eval_hidden(model, bench, ci)
        rows.append((ci, chunk.phase, bench.phase_names[chunk.phase], sc["macro_f1"]))
    print("\n[static baseline] 每 phase 的 hidden macro-F1：")
    for p in range(len(bench.phase_names)):
        vals = [r[3] for r in rows if r[1] == p]
        print(f"  P{p:02d} {bench.phase_names[p]:<8s} n_chunk={len(vals)} "
              f"macroF1={np.mean(vals):.3f} (±{np.std(vals):.3f})")

    # 3) 适应原语演示：在代表性漂移窗口上验证三种低成本更新
    rng = np.random.default_rng(0)
    demos = {2: "传感器老化→仿射校准", 3: "工况切换→带标签微调", 5: "未知故障→类别扩展"}
    for pidx, why in demos.items():
        ci = [i for i, c in enumerate(bench.chunks) if c.phase == pidx][-1]
        chunk = bench.chunks[ci]
        hx, hy = bench.hidden_by_chunk[ci]
        x = chunk.x
        static_f1 = eval_hidden(model, bench, ci)["macro_f1"]

        # 3a) 无标签仿射校准
        tgt = model.estimate_target_scaler(x)
        f1_calib = evaluate(hy, np.argmax(model.align_predict_proba(hx, tgt), axis=1), 5)["macro_f1"]

        # 3b) 线性头微调（只用当前类别）
        lab_idx = np.random.default_rng(pidx).choice(len(x), size=min(60, len(x)), replace=False)
        labs = bench.visible_y[chunk.ids[lab_idx]]
        keep = labs < model.n_classes
        m_head = model.clone()
        if keep.sum() >= 8:
            m_head.fit_head(x[lab_idx][keep], labs[keep], epochs=60, lr=1e-2, seed=0)
        f1_head = evaluate(hy, m_head.predict(hx), 5)["macro_f1"]

        # 3c) 全量微调 + 回放（60 条新标签 + 800 条回放）
        n_replay = min(800, len(bench.base_x))
        ridx = np.random.default_rng(pidx + 100).choice(len(bench.base_x), n_replay, replace=False)
        x_ft = np.vstack([bench.base_x[ridx], x[lab_idx][keep]])
        y_ft = np.concatenate([bench.base_y[ridx], labs[keep]])
        m_ft = model.clone()
        res_ft = m_ft.fit(x_ft, y_ft, epochs=25, lr=2e-3, batch=128, seed=0, refit_scaler=False)
        f1_ft = evaluate(hy, m_ft.predict(hx), 5)["macro_f1"]

        # 3d) 类别扩展（novel 场景）
        m_ext = expand_model_classes(model, bench.n_classes, seed=0)
        ridx2 = np.random.default_rng(pidx + 200).choice(len(bench.base_x), 800, replace=False)
        lab2 = np.random.default_rng(pidx + 300).choice(len(x), size=min(160, len(x)), replace=False)
        x_ex = np.vstack([bench.base_x[ridx2], x[lab2]])
        y_ex = np.concatenate([bench.base_y[ridx2], bench.visible_y[chunk.ids[lab2]]])
        m_ext.fit(x_ex, y_ex, epochs=30, lr=2e-3, batch=128, seed=0)
        f1_ext = evaluate(hy, m_ext.predict(hx), 5)["macro_f1"]

        print(f"\n[adapt] P{pidx:02d} {why} (static={static_f1:.3f})")
        print(f"  affine recalib : {f1_calib:.3f}")
        print(f"  head FT (60L)  : {f1_head:.3f}")
        print(f"  full FT+replay : {f1_ft:.3f}  (updates={res_ft.sample_updates})")
        print(f"  novel extend   : {f1_ext:.3f}")

    # 4) benchmark 结构自检
    assert bench.n_chunks > 0
    assert len(bench.visible_y) == sum(len(c.ids) for c in bench.chunks)
    assert all(len(bench.hidden_by_chunk[i][0]) > 0 for i in range(bench.n_chunks))
    print("\n[OK] 冒烟验证通过")


def micro_summary_stub():  # pragma: no cover
    pass


if __name__ == "__main__":
    scenario = sys.argv[1] if len(sys.argv) > 1 else "S7_full_mix"
    main(scenario)
