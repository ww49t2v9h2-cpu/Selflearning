# 官方 CPU-A/B/C 任务预测器与格式自检

本目录是**独立于官方代码**的预测器（纯 NumPy、CPU），可直接读取官方 checkpoint 格式（`contest.cpu-mlp.v1`），
按官方预测协议输出预测文件；同时提供格式自检与官方数值一致性校验。

## 目录

| 文件 | 说明 |
|---|---|
| `predict.py` | 预测器（stdin→stdout，官方协议） |
| `checkpoints/A_best.npz` | 官方 CPU-A 续训最优 checkpoint（气体传感器 6 类） |
| `checkpoints/B_best.npz` | 官方 CPU-B 续训最优 checkpoint（XJTU-SY RUL） |
| `checkpoints/C_best.npz` | 官方 CPU-C 续训最优 checkpoint（C-MAPSS RUL） |
| `selfcheck_A.json` / `_B` / `_C` | 官方公开验证集上的格式自检 + 与官方 `predict.py` 的数值一致性报告 |

## 预测协议（与官方 baselines/predict.py 一致）

**输入**（stdin，每行一个请求；只允许 `inputs` 和可选 `task`）：

```json
{"inputs": {"features": [ ... 128 个数 ... ]}}
```

- A：`inputs.features` 为 128 维数组；
- B：`inputs.features.horizontal` 与 `inputs.features.vertical` 各 17 个字段（34 维）；
- C：`inputs.settings`（3 维）+ `inputs.sensors`（21 维）。

**输出**（stdout，每行一个响应）：

- 分类（A）：`{"outputs":[{"label":"Ethanol","score":0.98}, ... 6 个类别 ...]}`
- 回归（B/C）：`{"outputs":{"value": 非负数值}}`

## 用法

```bash
# 1) 预测（A/B/C 换对应 checkpoint）
python predict.py --checkpoint checkpoints/A_best.npz < requests_A.jsonl > predictions_A.jsonl

# 2) 格式自检 + 与官方 predict.py 数值一致性（--official-package 指向官方包目录）
python predict.py --checkpoint checkpoints/A_best.npz \
  --self-check --data /path/to/public/validation.jsonl \
  --official-package /path/to/cpu-a-guide-20261110-v1 \
  --report selfcheck_A.json

# 3) 只做 schema 自检（不提供官方包时跳过数值比对）
python predict.py --checkpoint checkpoints/B_best.npz \
  --self-check --data /path/to/public/validation.jsonl --limit 200
```

把官方 JSONL 样本转成请求（去掉 id/answer/meta）：

```python
import json
for line in open("public/validation.jsonl", encoding="utf-8"):
    row = json.loads(line)
    print(json.dumps({"inputs": row["inputs"]}, ensure_ascii=False))
```

## 自检结果（官方公开验证集，2026-11-23）

| Profile | 样本数 | schema 错误 | 与官方 predict.py 最大绝对差 | 本地指标 |
|---|---|---|---|---|
| A | 1,244 | 0 | **0.0**（完全一致） | Macro-F1 **0.7258**，Accuracy 0.8842 |
| B | 1,679 | 0 | **0.0**（完全一致） | RMSE **642.63** 分钟，MAE 540.80 |
| C | 23,049 | 0 | **0.0**（完全一致） | RMSE **52.12** 循环，MAE 38.18 |

自检内容：响应仅含 `outputs`；分类输出类别顺序与 checkpoint 一致、分数在 [0,1] 且和为 1；
回归输出仅含 `value` 且有限非负；每条请求与 checkpoint 的特征 schema 完全匹配；输出行数 = 输入行数；
与官方预测器逐样本数值差为 0。

## 说明与边界

- checkpoint 来自官方 baseline 续训（官方 public/train.jsonl 训练、public/validation.jsonl 选模），
  官方 `official_scoring_ready=false`，本地验证指标不代表平台隐藏测试成绩；
- `predict.py` 不包含任何官方测试标签或隐藏数据；只做输入→输出推理；
- 官方协议要求预测响应只含 `outputs`，本实现严格遵循；批量预测请保证输出行与输入行**顺序一致**。
