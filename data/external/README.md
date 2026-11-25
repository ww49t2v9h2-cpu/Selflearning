# data/external —— 公开工业数据（UCI AI4I 2020）

本目录存放**公开真实工业数据**（补充验证），与合成漂移基准（`src/data/synth.py`）互补。
系统运行期**不联网**：数据以静态文件随仓库提供；下载/解压在准备阶段完成一次即可。

## 1. 数据来源与许可

| 项 | 内容 |
|---|---|
| 数据集 | AI4I 2020 Predictive Maintenance Dataset（UCI Machine Learning Repository, ID **601**） |
| 作者/说明 | S. Matzka, *Explainable Artificial Intelligence for Predictive Maintenance: A Framework and a Case Study on a Milling Machine*（2020） |
| 数据页 | https://archive.ics.uci.edu/dataset/601/ai4i+2020+predictive+maintenance+dataset |
| 下载直链 | https://archive.ics.uci.edu/static/public/601/ai4i+2020+predictive+maintenance+dataset.zip |
| 许可 | **CC BY 4.0**（https://creativecommons.org/licenses/by/4.0/）——使用时须注明出处 |
| 规模 | 10,000 行 × 14 列；10000 条为铣床加工记录（数据由作者基于工业机理+随机过程生成，公开可复现） |
| 故障率 | 3.48%（339/10000） |

**引用（答辩/文档请保留）**：Matzka, S. (2020). *AI4I 2020 Predictive Maintenance Dataset*.
UCI Machine Learning Repository. https://doi.org/10.24432/C5HS5C (CC BY 4.0)。

## 2. 本地文件

| 文件 | 说明 | 是否入 git |
|---|---|---|
| `ai4i2020.zip` | 原始官方 zip（522 KB，校验和见 manifest） | 是（保证离线可复现） |
| `ai4i2020.csv` | 解压后的原始 CSV（522 KB） | 仓库 `.gitignore` 忽略 CSV；缺 CSV 时脚本自动从本地 zip 离线解压 |
| `download_manifest.json` | 下载 URL、sha256、字节数、时间、许可 | 是 |
| `README.md` | 本文件（字段映射与生成方式） | 是 |

校验：`csv_sha256 = dc6630cd9b1f0f853922fad78a1b6436570d3f1ec863f1dd5c4340ac56bc8a8e`，
`zip_sha256 = f601f14294bcf190f9d720676b7f0aea46a26cde9ab8ebc7b4f8174d9d26b252`。

## 3. 字段映射（原始 14 列 → Benchmark 11 维特征）

原始列：`UDI, Product ID, Type, Air temperature [K], Process temperature [K],
Rotational speed [rpm], Torque [Nm], Tool wear [min], Machine failure, TWF, HDF, PWF, OSF, RNF`

| 生成特征（`FEATURE_NAMES`） | 来源 | 物理含义 |
|---|---|---|
| `air_temp` | Air temperature | 环境气温 |
| `proc_temp` | Process temperature | 过程温度 |
| `temp_diff` | process − air | 温升（HDF 失效判据之一） |
| `rpm` | Rotational speed | 主轴转速（工况段变量） |
| `torque` | Torque | 扭矩（OSF/PWF 判据之一） |
| `tool_wear` | Tool wear | 刀具磨损（渐变漂移主变量） |
| `power_kw` | torque × rpm × 2π/60 / 1000 | 机械功率（PWF 判据） |
| `torque_per_rpm` | torque / rpm × 1000 | 扭矩-转速比 |
| `wear_ratio` | wear / 253 | 归一化磨损 |
| `wear_x_torque` | wear × torque / 1000 | 磨损-扭矩交互（OSF 判据） |
| `temp_ratio` | process / air | 温度比 |

标签：`label = Machine failure`（0 正常 / 1 故障）。类别名 `["正常","故障"]`。

> 为什么不用 `Type`(L/M/H) 做特征/工况：实测各 Type 的温/速/扭矩/磨损分布均值差 <1%，
> 它只轻微影响故障先验；真正的协变量漂移在**转速-扭矩-功率**方向（低转速重载工况
> 故障率 7.9% vs 中转速 0.7%）。用恒为 0 的 one-hot 列会在漂移域外推造成伪影，故不采用。

## 4. 从原始数据生成漂移流的剧本（`src/data/public_ai4i.py::PHASE_SCRIPT`）

初始基座（base train/val）来自 **重载低转速工况（rpm<1450）& 刀具磨损<80min** 的
"健康工况"段；部署流按"转速工况段 × 刀具磨损段"切分，阶段内按磨损时间排序：

| 阶段 | 名字 | 漂移机制 |
|---|---|---|
| P0 | 基线延续(重载B,磨损80-140) | 同工况、磨损继续累积 |
| P1 | 深度磨损(重载B,磨损≥140) | 渐变磨损加深（故障率 12.7%，静态模型故障召回塌陷） |
| P2 | 工况切换A(中转速,磨损≥80) | 转速/扭矩工况切换 = 协变量漂移（故障率降至 1.0%） |
| P3 | 工况切换C(高转速,全磨损) | 第三种工况（故障率 1.8%），静态模型误报/漏报同时出现 |

- 每阶段随机 34% 作为**隐藏测试**（同分布、系统不可见），其余 66% 为部署流并按
  200 样本/窗口切分；`visible_y` 供 `LabelOracle` 按预算取标注。
- 结果：22 个窗口、d=11、4 个阶段、2 类；故障率 3.48%（高度不平衡）。
- `make_ai4i_benchmark(cfg, seed)` 与 `synth.make_benchmark` 同签名、返回同一 `Benchmark` 结构，
  因此闭环/基线/引擎代码无需任何改动即可在真实数据上运行。
- seed 只控制 base 划分、hidden/visible 划分与标注抽样顺序；数据本身固定。

## 5. 准备与自检命令

```bash
PY=python3
$PY -m src.data.public_ai4i --check                 # 检查文件是否就绪（不联网）
$PY -m src.data.public_ai4i --download              # 准备阶段：下载/解压（允许联网，仅一次）
$PY -m src.data.public_ai4i --download --force      # 强制重新下载（直链失效时换镜像会自动尝试）
$PY scripts/baseline_suite.py --include-ai4i        # 基线套件附带 AI4I 真实数据场景（补充验证）
```

离线/无网环境：本目录已带 `ai4i2020.zip`，脚本会在缺 CSV 时**从本地 zip 解压**，不联网。

## 6. 使用注意

- AI4I 故障率仅 3.48%：macro-F1 对稀有类（故障）召回敏感、阶段内隐藏测试故障样本少
  （每阶段约 7–54 条），**结果波动大于合成基准**（3 seeds static 0.496±0.040，
  periodic 0.610±0.065，oracle 0.630±0.014）；用于"闭环/适配层在真实数据上可运行"的
  补充验证，**不进入 S1-S7 难度主表与 Pareto 主图**；解读时同时看 accuracy 与故障召回
  （见 docs/09 §5）。
- 许可 CC BY 4.0：参赛文档/PPT/代码中引用本数据须保留出处与许可说明；
  再分发本目录 zip/csv 时同样须附带许可与出处。
