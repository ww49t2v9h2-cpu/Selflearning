# syntax=docker/dockerfile:1
# 智训引擎 Harness 系统镜像（linux/amd64，CPU-only）
FROM --platform=linux/amd64 python:3.12-slim

LABEL org.opencontainers.image.title="zhixun-engine-harness" \
      org.opencontainers.image.description="智训引擎：工业模型自进化系统（Harness/实验记录/安全审计）" \
      org.opencontainers.image.version="1.0" \
      org.opencontainers.image.architecture="amd64" \
      org.opencontainers.image.os="linux" \
      org.opencontainers.image.license="research-use-only"

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    PIP_NO_CACHE_DIR=1 \
    PYTHONPATH=/app

WORKDIR /app

# 非 root 运行
RUN groupadd --gid 10001 app && useradd --uid 10001 --gid app --create-home app

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY src ./src
COPY configs ./configs
COPY scripts ./scripts
COPY skills ./skills
COPY tests ./tests
COPY docs ./docs
COPY README.md ./
# 可选：内置公开 AI4I 数据（真实数据 real 组离线可跑）
COPY data/external/ai4i2020.zip ./data/external/ai4i2020.zip

RUN chmod -R a+rX /app \
 && find /app -type d -name __pycache__ -prune -exec rm -rf {} + || true
# 运行产物目录（experiments/results）给 app 用户可写，其余代码保持只读
RUN mkdir -p /app/experiments /app/results && chown -R app:app /app/experiments /app/results

USER app

# 默认进 CLI；官方评测可覆盖命令：
#   docker run --rm <image> run --config configs/main.json
ENTRYPOINT ["python", "-m", "src.cli"]
CMD ["info"]
