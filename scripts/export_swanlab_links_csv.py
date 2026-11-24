"""导出 SwanLab 云端 run 链接清单 CSV（供评委批量抽查 / 提交包附件）。

输入：
  experiments/final_records/tracker.sqlite（本地记录）
  results/swanlab_links.jsonl（run_id → 云端 run 链接）
输出：
  results/swanlab_links.csv
  deliverables/提交包_2026-11-20/附_SwanLab云端Run链接清单.csv（UTF-8 BOM，Excel 友好）

用法：/tmp/swanenv/bin/python scripts/export_swanlab_links_csv.py
"""

from __future__ import annotations

import csv
import json
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "experiments" / "final_records" / "tracker.sqlite"
LINKS = ROOT / "results" / "swanlab_links.jsonl"
OUT_RESULTS = ROOT / "results" / "swanlab_links.csv"
OUT_PKG = ROOT / "deliverables" / "提交包_2026-11-20" / "附_SwanLab云端Run链接清单.csv"
PROJECT = "https://swanlab.cn/@Migratory_Bird/zhixun-engine"

FIELDS = ["run_id", "group", "name", "kind", "stage", "status", "scenario", "seed",
          "created_at", "stream_macro_f1", "stream_gain_pt", "labels", "compute_units",
          "tokens", "cost_scalar", "n_windows", "audit_verified",
          "swanlab_run_id", "swanlab_url", "project_url", "cloud_synced"]


def _summary(run: dict, dot: str, default=None):
    try:
        cur = json.loads(run.get("summary_json") or "{}")
        for k in dot.split("."):
            cur = cur[k]
        return cur
    except Exception:
        return default


def main() -> int:
    links = {}
    for line in LINKS.read_text(encoding="utf-8").splitlines():
        try:
            d = json.loads(line)
            if d.get("url"):
                links[d["run_id"]] = d
        except Exception:
            pass

    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    runs = [dict(r) for r in conn.execute("select * from runs order by group_name, created_at, run_id")]
    conn.close()

    rows = []
    for r in runs:
        link = links.get(r["run_id"], {})
        tags = []
        try:
            tags = json.loads(r.get("tags_json") or "[]")
        except Exception:
            pass
        seed = None
        try:
            for t in tags:
                if str(t).startswith("seed="):
                    seed = str(t).split("=", 1)[1]
        except Exception:
            pass
        try:
            created = __import__("datetime").datetime.fromtimestamp(
                float(r["created_at"] or 0)).astimezone().isoformat(timespec="seconds")
        except Exception:
            created = ""
        rows.append({
            "run_id": r["run_id"],
            "group": r["group_name"],
            "name": r["name"],
            "kind": r["kind"],
            "stage": r["stage"],
            "status": r["status"],
            "scenario": tags[0] if tags else "",
            "seed": seed or "",
            "created_at": created,
            "stream_macro_f1": _summary(r, "hidden.stream_macro_f1", ""),
            "stream_gain_pt": _summary(r, "comparison.stream_gain_points", ""),
            "labels": _summary(r, "cost.labels", ""),
            "compute_units": _summary(r, "cost.compute_units", ""),
            "tokens": _summary(r, "cost.tokens", ""),
            "cost_scalar": _summary(r, "cost.scalar", ""),
            "n_windows": _summary(r, "hidden.n_chunks", ""),
            "audit_verified": _summary(r, "audit_verified", ""),
            "swanlab_run_id": link.get("swanlab_run_id", ""),
            "swanlab_url": link.get("url", ""),
            "project_url": PROJECT,
            "cloud_synced": "TRUE" if link.get("url") else "FALSE",
        })

    for p in (OUT_RESULTS, OUT_PKG):
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "w", encoding="utf-8-sig", newline="") as f:
            w = csv.DictWriter(f, fieldnames=FIELDS)
            w.writeheader()
            w.writerows(rows)
    synced = sum(1 for x in rows if x["cloud_synced"] == "TRUE")
    print(f"rows={len(rows)} synced={synced} -> {OUT_RESULTS}")
    print(f"copy -> {OUT_PKG} ({OUT_PKG.stat().st_size/1024:.1f}KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
