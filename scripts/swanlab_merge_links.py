"""合并并行分片的 SwanLab 上传结果（links/state），产出总映射与上传摘要。

用法：/tmp/swanenv/bin/python scripts/swanlab_merge_links.py
输出：
  results/swanlab_links.jsonl        合并后的 run_id -> {swanlab_run_id,url,group}
  results/swanlab_upload_state.json  合并后的已完成 run_id
  results/swanlab_upload_summary.json 计数/分组/缺失/额度标记
"""

from __future__ import annotations

import glob
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
LINKS = RESULTS / "swanlab_links.jsonl"
STATE = RESULTS / "swanlab_upload_state.json"
SUMMARY = RESULTS / "swanlab_upload_summary.json"


def main() -> int:
    mapping: dict[str, dict] = {}
    for f in glob.glob(str(RESULTS / "swanlab_links*.jsonl")):
        for line in Path(f).read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if d.get("url"):
                    mapping[d["run_id"]] = d
            except Exception:
                pass
    done: set[str] = set()
    for f in glob.glob(str(RESULTS / "swanlab_state*.json")) + [str(STATE)]:
        p = Path(f)
        if not p.exists():
            continue
        try:
            done |= set(json.loads(p.read_text(encoding="utf-8")).get("done", []))
        except Exception:
            pass
    # 默认 state 里可能只有早期 64 条；合并分片 state 的 done 才是全量
    done |= set(mapping.keys())

    LINKS.write_text("".join(json.dumps(v, ensure_ascii=False) + "\n" for v in mapping.values()),
                     encoding="utf-8")
    STATE.write_text(json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8")
    groups: dict[str, int] = {}
    for d in mapping.values():
        groups[d.get("group", "")] = groups.get(d.get("group", ""), 0) + 1
    blocked = [p.name for p in RESULTS.glob("quota_blocked_part*.txt")]
    summary = {
        "project_url": "https://swanlab.cn/@Migratory_Bird/zhixun-engine",
        "uploaded": len(mapping),
        "groups": groups,
        "quota_blocked": blocked,
        "generated_at": __import__("datetime").datetime.now().astimezone().isoformat(timespec="seconds"),
    }
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
