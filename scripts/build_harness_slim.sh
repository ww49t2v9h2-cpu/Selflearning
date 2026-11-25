#!/usr/bin/env bash
# 构建平台提交用精简 linux/amd64 镜像（Alpine，目标 docker save tar < 50MB）
set -euo pipefail
cd "$(dirname "$0")/.."
ROOT="$(pwd)"
TAG="${TAG:-zhixun-engine:1.0-linux-amd64-slim}"
OUT="${1:-${ROOT}/deliverables/zhixun-engine_linux_amd64_slim.tar}"

docker buildx build --platform linux/amd64 --provenance=false --sbom=false \
  --load -f Dockerfile.alpine -t "${TAG}" "${ROOT}"
docker image inspect "${TAG}" --format 'arch={{.Architecture}} os={{.Os}} id={{.Id}}'
docker run --rm --platform linux/amd64 "${TAG}" info >/dev/null
mkdir -p "$(dirname "${OUT}")"
docker save -o "${OUT}" "${TAG}"
ls -lh "${OUT}"
python3 scripts/verify_docker_archive.py "${OUT}"
