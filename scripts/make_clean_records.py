"""生成"只含最终冻结实验"的干净记录镜像（用于交付的《实验记录汇编》）。

问题：开发过程中多次重跑矩阵，tracker.sqlite 里混入了调试/中间版本 run；
交付的记录汇编应只包含最终结果对应的 run（与 results/ 一一对应）。

做法：
1) 从 results/**/*.json 与 results/acceptance.json 收集最终 loop run_id；
2) 复制 tracker.sqlite 到 experiments/final_records/；
3) 删除不属于"最终 loop run 及其子记录（parent_run 指向它们）"的行；
4) 输出统计，供 gen_record_book --dir experiments/final_records 使用。

用法：
  PY=.../python3
  $PY scripts/make_clean_records.py
  $PY scripts/gen_record_book.py --dir experiments/final_records \
      --limit 20000 --title "智训引擎 · 实验记录汇编（最终冻结版）" \
      --out-md deliverables/实验记录汇编.md --out-xlsx deliverables/实验记录汇编.xlsx
"""

from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
SRC_DB = ROOT / "experiments" / "runs" / "tracker.sqlite"
OUT_DIR = ROOT / "experiments" / "final_records"


FINAL_GROUP_PREFIXES = ("main", "scenarios", "ab-", "budget-", "real", "route-")


def is_final_group(path: Path) -> bool:
    group = path.parent.name
    return any(group == g or group.startswith(g) for g in FINAL_GROUP_PREFIXES)


def collect_loop_ids() -> set[str]:
    ids: set[str] = set()
    for p in RESULTS.glob("*/*.json"):
        if not is_final_group(p):
            continue
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            if isinstance(r, dict) and "meta" in r and "run_id" in r["meta"]:
                ids.add(str(r["meta"]["run_id"]))
        except Exception:
            continue
    # acceptance（含 --repro 的两次 run）
    for p in RESULTS.glob("acceptance/*.json"):
        try:
            with open(p, "r", encoding="utf-8") as f:
                r = json.load(f)
            if "meta" in r:
                ids.add(str(r["meta"]["run_id"]))
        except Exception:
            continue
    acc = RESULTS / "acceptance.json"
    if acc.exists():
        try:
            with open(acc, "r", encoding="utf-8") as f:
                a = json.load(f)
            if a.get("run_id"):
                ids.add(str(a["run_id"]))
        except Exception:
            pass
    return ids


def main() -> int:
    loop_ids = collect_loop_ids()
    if not loop_ids:
        print("没有找到最终结果 run（先跑 scripts/run_all.py）")
        return 1
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_db = OUT_DIR / "tracker.sqlite"
    shutil.copy2(SRC_DB, out_db)
    con = sqlite3.connect(str(out_db))
    cur = con.cursor()
    # 子记录：parent_run 指向某个最终 loop run
    rows = cur.execute("SELECT run_id, parent_run FROM runs").fetchall()
    keep = set(loop_ids)
    changed = True
    while changed:  # parent 关系可能多层，迭代到闭包
        changed = False
        for run_id, parent in rows:
            if run_id in keep:
                continue
            if parent and parent in keep:
                keep.add(str(run_id))
                changed = True
    all_ids = [str(r[0]) for r in rows]
    drop = [rid for rid in all_ids if rid not in keep]
    cur.executemany("DELETE FROM runs WHERE run_id=?", [(r,) for r in drop])
    cur.executemany("DELETE FROM metrics WHERE run_id=?", [(r,) for r in drop])
    cur.executemany("DELETE FROM artifacts WHERE run_id=?", [(r,) for r in drop])
    con.commit()
    n = cur.execute("SELECT count(*) FROM runs").fetchone()[0]
    nm = cur.execute("SELECT count(*) FROM metrics").fetchone()[0]
    groups = dict(cur.execute("SELECT group_name, count(*) FROM runs GROUP BY group_name").fetchall())
    con.close()
    print(f"[clean_records] loop runs={len(loop_ids)} → 保留记录 {n} 条 / 指标 {nm} 个")
    print(f"[clean_records] 输出目录：{OUT_DIR.relative_to(ROOT)}")
    print(f"[clean_records] 分组：{json.dumps(groups, ensure_ascii=False)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
