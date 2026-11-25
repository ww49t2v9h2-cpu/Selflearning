# 平台提交与镜像说明（2026-11-25）

本页补充 2026-11-25 平台提交所需的 **精简 Harness 镜像**、**PyTRIO 微调权重** 与材料索引。
不包含 API Key、隐藏测试集或本机绝对路径。

## 1. 提交用 Harness 镜像（tar < 50MB）

- 镜像：`zhixun-engine:1.0-linux-amd64-slim`
- 架构：`linux/amd64`；单镜像
- tar：`02_Harness系统镜像.tar`
  - 大小：**42,350,592 bytes（42.35 MB < 50MB）**
  - SHA256：`1ceb2753f570837d12122c7ac61047c73b264d9270085744d94547d1c0e66c3e`
- 入口：`python -m src.cli`；默认 `CMD info`
- 离线运行：内置 numpy + 公开 AI4I 样例数据，无网络依赖；输出 `/app/experiments`、`/app/results`
- 构建：`Dockerfile.alpine` + `scripts/build_harness_slim.sh`

```bash
docker load -i 02_Harness系统镜像.tar
docker run --rm zhixun-engine:1.0-linux-amd64-slim info
docker run --rm zhixun-engine:1.0-linux-amd64-slim run --config configs/quick.json
```

## 2. PyTRIO 微调权重（平台字段）

- 基座：`Qwen/Qwen3.8-27B`
- LoRA SFT：rank=32，3 steps，lr=1e-4；loss 3.6353 → 3.4063 → 3.0056
- Sampler 权重路径：`trio://run_ux2lc7p5v3r2/sampler_weights/zhixun-sampler-20261124-172703`
- 训练记录：https://swanlab.cn/@Migratory_Bird/zhixun-engine/runs/g06cbpn6
- API Key 不写入仓库/镜像，由提交人本人在平台表单填写

## 3. 平台 7 项材料

| 字段 | 文件 |
|---|---|
| 模型配置 | `01_模型配置.txt` |
| Harness 系统镜像 | `02_Harness系统镜像.tar`（42.35MB） |
| 实验记录 | `03_完整云端实验记录_查询依据.pdf` + `03_实验记录链接.txt` |
| 系统运行说明 | `04_系统运行说明.pdf` |
| 项目说明书 | `05_项目说明书.pdf` |
| 答辩 PPT | `06_答辩PPT.pptx` |
| 项目源代码 | `07_项目源代码.zip` |

材料包：`平台材料包_2026-11-25.zip`（44MB）

## 4. 仓库

https://github.com/ww49t2v9h2-cpu/Selflearning
