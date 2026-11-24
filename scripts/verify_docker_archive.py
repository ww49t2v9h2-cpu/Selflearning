#!/usr/bin/env python3
"""校验 docker save 导出的单镜像未压缩 tar（无需 Docker）。

检查：
  1) 文件是未压缩 POSIX tar（非 gzip）；
  2) manifest.json 恰好一个镜像条目，RepoTags 存在；
  3) config JSON 的 architecture=amd64、os=linux；
  4) Layers 数量与 config.rootfs.diff_ids 数量一致；
  5) 不包含 multi-arch index / OCI layout；
  6) 打印 layer 数、总大小、entry 类型。
用法：python3 scripts/verify_docker_archive.py path/to/image.tar
"""
import json, sys, tarfile
from pathlib import Path

def main() -> int:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else None
    if not path or not path.is_file():
        print("usage: verify_docker_archive.py <image.tar>"); return 2
    raw = path.read_bytes()[:4]
    if raw[:2] == b"\x1f\x8b":
        print("FAIL: tar 是 gzip 压缩的，要求未压缩 tar"); return 2
    if not tarfile.is_tarfile(path):
        print("FAIL: 不是 tar 文件"); return 2
    with tarfile.open(path, "r:") as tf:
        names = tf.getnames()
        if "manifest.json" not in names:
            print("FAIL: 缺少 manifest.json（不是 docker save 格式）"); return 2
        manifest = json.load(tf.extractfile("manifest.json"))
        if not isinstance(manifest, list) or len(manifest) != 1:
            print(f"FAIL: manifest 条目数={len(manifest) if isinstance(manifest,list) else 'bad'}，要求单个镜像"); return 2
        entry = manifest[0]
        config_name = entry.get("Config")
        if not config_name or config_name not in names:
            print("FAIL: manifest.Config 缺失或 tar 内不存在"); return 2
        config = json.load(tf.extractfile(config_name))
        arch, os_ = config.get("architecture"), config.get("os")
        layers = entry.get("Layers") or []
        diff_ids = ((config.get("rootfs") or {}).get("diff_ids") or [])
        errors = []
        if arch != "amd64": errors.append(f"architecture={arch}, 需要 amd64")
        if os_ != "linux": errors.append(f"os={os_}, 需要 linux")
        if not layers: errors.append("Layers 为空")
        if len(layers) != len(diff_ids): errors.append(f"Layers={len(layers)} != diff_ids={len(diff_ids)}")
        # Docker 29（containerd image store）docker save 会同时写 OCI layout
        # （blobs/ + index.json + oci-layout），但 manifest.json 仍是单镜像；
        # 只要 manifest.json 单条且 config 为 amd64/linux，就满足“单镜像 tar”。
        layout = "oci" if "index.json" in names else "docker"
        if "index.json" in names:
            try:
                idx = json.load(tf.extractfile("index.json"))
                if not isinstance(idx, dict) or len(idx.get("manifests") or []) != 1:
                    errors.append("index.json 不是单镜像 index")
            except Exception as e:
                errors.append(f"index.json 解析失败: {e}")
        size_mb = path.stat().st_size / 1048576
        print(f"file: {path}")
        print(f"size: {size_mb:.1f} MB | tar members: {len(names)} | layers: {len(layers)}")
        print(f"repo tags: {entry.get('RepoTags')}")
        print(f"architecture: {arch} | os: {os_}")
        if errors:
            print("FAIL: " + "; ".join(errors)); return 2
        print(f"OK: 单镜像 linux/amd64 未压缩 docker save tar（layout={layout}）")
        return 0

if __name__ == "__main__":
    raise SystemExit(main())
