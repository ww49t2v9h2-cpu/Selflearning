#!/usr/bin/env python3
"""把 Docker 29/containerd image store 的 docker save tar（OCI hybrid）转换为
经典 docker-archive 布局：manifest.json + config.json + layer_*.tar（层解压为 tar），
外层保持未压缩 tar。输出仍可用 docker load 载入。

用法：python3 scripts/docker_save_to_classic.py <raw.tar> <classic.tar>
"""
import gzip, io, json, sys, tarfile
from pathlib import Path

def main() -> int:
    if len(sys.argv) != 3:
        print("usage: docker_save_to_classic.py <raw.tar> <classic.tar>"); return 2
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    if not src.is_file():
        print("missing", src); return 2
    if dst.exists():
        dst.unlink()
    with tarfile.open(src, "r:") as tf:
        manifest = json.load(tf.extractfile("manifest.json"))
        if not isinstance(manifest, list) or len(manifest) != 1:
            print("FAIL: not single-image docker save"); return 2
        entry = manifest[0]
        config_name = entry["Config"]
        config_bytes = tf.extractfile(config_name).read()
        config = json.loads(config_bytes)
        if config.get("architecture") != "amd64" or config.get("os") != "linux":
            print("FAIL: not linux/amd64"); return 2
        layer_names = entry["Layers"]
        with tarfile.open(dst, "w", format=tarfile.PAX_FORMAT) as out:
            # 1) manifest.json
            new_manifest = [{"Config": "config.json",
                             "RepoTags": entry.get("RepoTags") or [],
                             "Layers": [f"layer_{i}.tar" for i in range(len(layer_names))]}]
            data = json.dumps(new_manifest, ensure_ascii=False).encode()
            ti = tarfile.TarInfo("manifest.json"); ti.size = len(data); ti.mode = 0o644
            out.addfile(ti, io.BytesIO(data))
            # 2) config.json
            ti = tarfile.TarInfo("config.json"); ti.size = len(config_bytes); ti.mode = 0o644
            out.addfile(ti, io.BytesIO(config_bytes))
            # 3) layers: gzip blob -> uncompressed layer tar
            for i, name in enumerate(layer_names):
                f = tf.extractfile(name)
                first = f.read(2); f.seek(0)
                if first == b"\x1f\x8b":
                    raw = gzip.GzipFile(fileobj=f).read()
                else:
                    raw = f.read()
                ti = tarfile.TarInfo(f"layer_{i}.tar"); ti.size = len(raw); ti.mode = 0o644
                out.addfile(ti, io.BytesIO(raw))
    print(f"written {dst} ({dst.stat().st_size/1048576:.1f} MB)")
    return 0

if __name__ == "__main__":
    raise SystemExit(main())
