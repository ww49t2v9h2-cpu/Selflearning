# 平台提交与镜像说明（2026-11-25）

本页补充平台提交所需的 **官方 contest_harness SDK 镜像**、**PyTRIO 微调权重** 与材料索引。
不包含 API Key、隐藏测试集或本机绝对路径。

## 1. 提交用 Harness 镜像（tar < 50MB）

- 镜像：`zhixun-engine:1.0-linux-amd64-slim`
- 架构：`linux/amd64`；单镜像
- tar：`02_Harness系统镜像.tar`
  - 大小：**42,422,784 bytes（42.42 MB < 50MB）**
  - SHA256：`e71a9769ccbebc1f3e3d36f1fcdffeaf19f92f48dd7553bb553bb70ff7eefe21`
- 平台入口（**已内置 SDK**）：
  - `/app/harness.py`（模板来自 `contest_harness` 1.0.4）
  - `/app/contest_submission.py`（选手模块：`train(rnd)` / `predict(sample, model_dir)`）
  - 默认命令：`python3 /app/harness.py describe`
  - 平台固定入口：`describe` / `train` / `predict`
- `describe` 实测输出：
  ```json
  {"protocol":"contest.harness.workspace.v2","model_access":"host_gateway_rpc","sdk":"contest_harness","sdk_version":"1.0.4","tasks":["classification","regression"]}
  ```
- 构建：`Dockerfile.alpine` + `scripts/build_harness_slim.sh`；SDK 源码在 `harness_image/`

```bash
docker load -i 02_Harness系统镜像.tar
docker run --rm zhixun-engine:1.0-linux-amd64-slim describe
docker run --rm -i zhixun-engine:1.0-linux-amd64-slim predict
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
| Harness 系统镜像 | `02_Harness系统镜像.tar`（42.42MB） |
| 实验记录 | `03_完整云端实验记录_查询依据.pdf` + `03_实验记录链接.txt` |
| 系统运行说明 | `04_系统运行说明.pdf` |
| 项目说明书 | `05_项目说明书.pdf` |
| 答辩 PPT | `06_答辩PPT.pptx` |
| 项目源代码 | `07_项目源代码.zip` |

材料包：`平台材料包_2026-11-25.zip`

## 4. 仓库

https://github.com/ww49t2v9h2-cpu/Selflearning
