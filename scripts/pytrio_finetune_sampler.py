#!/usr/bin/env python3
"""PyTRIO 最小 LoRA SFT → save_weights_for_sampler 的 trio:// 路径。

前置：
  Python >= 3.10（本机用 /tmp/pytrivenv）
  pip install pytrio
  trio login  或 export PYTRIO_API_KEY=...
用法：
  /tmp/pytrivenv/bin/python scripts/pytrio_finetune_sampler.py --steps 3
输出：
  results/pytrio/finetune_result.json（含 trio_path / base_model / loss / swanlab 待写字段）
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pytrio as trio

ROOT = Path(__file__).resolve().parents[1]
EXAMPLES = [
    ("智训引擎是什么？", "智训引擎是面向工业制造场景的模型自进化系统，覆盖漂移监测、诊断归因、实验优化、评估上线与回滚。"),
    ("智训引擎怎样控制标注成本？", "它用无标签监测与 TTA 优先，必要时才用预算内探针标签，并为适应动作预留标签预算。"),
    ("为什么需要回滚？", "回滚保证候选模型在门禁不通过或线上退化时能恢复上一个稳定版本，形成可审计的安全闭环。"),
    ("Harness 的作用是什么？", "Harness 负责编排工具、技能、预算与实验记录，把 LLM 的开放式决策限制在可复现、可审计的流程内。"),
    ("SwanLab 在这里做什么？", "SwanLab 用于实验记录与指标追踪，每个 run 记录配置、成本、增益和审计字段，支持云端回放。"),
    ("官方 CPU-A 任务是什么？", "官方 CPU-A 是 UCI 气体传感器阵列漂移的 6 类分类任务，Baseline 为 NumPy MLP。"),
    ("怎样评估漂移下的提升？", "用隐藏测试流上的 stream macro-F1 与静态基线比较，并报告标注、算力和 token 成本。"),
    ("TTA 为什么优先？", "TTA 只需当前批次的统计量，不消耗标签，适合强协变量漂移；官方 CPU-A 上平均提升约 23 个百分点。"),
    ("门禁的最低要求是什么？", "综合性能不得低于静态基线，同时检查召回下降、校准与成本；不通过就不部署。"),
    ("如何保证可复现？", "固定随机种子、冻结核心配置、按 run 保存策略文件，并用 CLI repro 校验字节级一致。"),
]


def _datum(tokenizer, prompt: str, completion: str) -> trio.Datum:
    p = tokenizer.encode(prompt, add_special_tokens=True)
    c = tokenizer.encode(completion, add_special_tokens=False)
    tokens = p + c
    weights = [0] * len(p) + [1] * len(c)
    input_tokens = tokens[:-1]
    target_tokens = tokens[1:]
    weights = weights[1:]
    return trio.Datum(
        model_input=trio.ModelInput.from_ints(tokens=input_tokens),
        loss_fn_inputs={
            "weights": np.asarray(weights, dtype=np.float32),
            "target_tokens": np.asarray(target_tokens, dtype=np.int32),
        },
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", default="auto", help="auto 或 Qwen/Qwen3.8-27B")
    ap.add_argument("--rank", type=int, default=32)
    ap.add_argument("--steps", type=int, default=3)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--name", default="zhixun-sampler")
    ap.add_argument("--out", default=str(ROOT / "results/pytrio/finetune_result.json"))
    args = ap.parse_args()

    service = trio.ServiceClient()
    models = service.get_supported_models()
    print("supported models:", models)
    base = args.base_model
    if base == "auto":
        cands = [m for m in models if "Qwen3.8-27B" in m]
        if not cands:
            cands = [m for m in models if "27B" in m]
        if not cands:
            raise SystemExit(f"no 27B model in supported list: {models}")
        base = cands[0]
    print("base_model:", base)

    training = service.create_lora_training_client(base_model=base, rank=args.rank)
    tokenizer = training.get_tokenizer()
    data = [
        _datum(tokenizer, f"用户：{q}\n助手：", f"{a}\n")
        for q, a in EXAMPLES
    ]

    losses = []
    for step in range(1, args.steps + 1):
        fw = training.forward_backward(data, "cross_entropy").result()
        training.optim_step(trio.AdamParams(learning_rate=args.lr)).result()
        m = getattr(fw, "metrics", {}) or {}
        loss = m.get("loss_mean") if isinstance(m, dict) else None
        losses.append(loss)
        print(f"step {step}/{args.steps} loss={loss} metrics={m}")

    name = f"{args.name}-{time.strftime('%Y%m%d-%H%M%S')}"
    result = training.save_weights_for_sampler(name=name).result()
    path = getattr(result, "path", None) or getattr(result, "model_path", None)
    print("save result:", result)
    print("TRIO_PATH:", path)

    out = {
        "trio_path": path,
        "base_model": base,
        "checkpoint_name": name,
        "rank": args.rank,
        "steps": args.steps,
        "learning_rate": args.lr,
        "losses": losses,
        "examples": len(EXAMPLES),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print("WROTE", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
