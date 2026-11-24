#!/usr/bin/env bash
# 在任意 Docker 主机上构建 linux/amd64 单镜像并 docker save 为未压缩 tar。
# 本机（Apple Silicon）使用 buildx 跨构建；Linux/amd64 主机直接构建。
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
PY="${PY:-python3}"
TAG="${TAG:-zhixun-engine:1.0-linux-amd64}"
OUT_DIR="${1:-${ROOT}/deliverables}"
RAW="${OUT_DIR}/智训引擎_Harness_linux_amd64_docker-save-raw.tar"
CLASSIC="${OUT_DIR}/智训引擎_Harness_linux_amd64.tar"

echo "== 1/6 buildx build --platform linux/amd64 --load =="
docker buildx build --platform linux/amd64 --load -t "${TAG}" "${ROOT}"

echo "== 2/6 架构断言 =="
docker image inspect "${TAG}" >/dev/null
ARCH="$(docker image inspect "${TAG}" --format '{{.Architecture}}')"
OS="$(docker image inspect "${TAG}" --format '{{.Os}}')"
echo "image: ${OS}/${ARCH}"
[ "${ARCH}" = "amd64" ] || { echo "ERROR: 镜像架构不是 amd64（${ARCH}）"; exit 1; }

echo "== 3/6 容器冒烟（linux/amd64，QEMU/Rosetta） =="
docker run --rm --platform linux/amd64 "${TAG}" info | head -5

echo "== 4/6 docker save（原始未压缩 tar） =="
mkdir -p "$(dirname "${RAW}")"
docker save -o "${RAW}" "${TAG}"
ls -lh "${RAW}"

echo "== 5/6 转换为经典 docker-archive 布局 =="
"${PY}" "${ROOT}/scripts/docker_save_to_classic.py" "${RAW}" "${CLASSIC}"
ls -lh "${CLASSIC}"

echo "== 6/6 结构与 load 回放校验 =="
"${PY}" "${ROOT}/scripts/verify_docker_archive.py" "${CLASSIC}"
docker load -i "${CLASSIC}" >/dev/null
echo "DONE: ${RAW} (raw) / ${CLASSIC} (classic, 提交用)"
