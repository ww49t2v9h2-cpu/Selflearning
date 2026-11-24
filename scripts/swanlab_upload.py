"""把本地 tracker.sqlite 的 2511 条实验记录回放到 SwanLab 云端（范围 B：全部 runs）。

用途：校赛记录是 SwanLab offline/local 兼容镜像（无云额度）；拿到账号后，
本脚本按 run 逐个重建云端记录（summary + 逐 step 指标曲线），并生成云端链接映射。

运行环境：/tmp/swanenv/bin/python（Python 3.9 + swanlab 0.10.6；不要用 DSH 内置 python，
其签名策略会阻止 pip 扩展加载）。登录：先在可见终端执行 `swanlab login` 粘贴 API key。

用法：
  # 先用 local 模式验证脚本（不上云）
  /tmp/swanenv/bin/python scripts/swanlab_upload.py --mode local --limit 2
  # 登录后 pilot 3 条
  /tmp/swanenv/bin/python scripts/swanlab_upload.py --mode online --limit 3 --public
  # 全量 2511 条（可重复执行，断点续传）
  /tmp/swanenv/bin/python scripts/swanlab_upload.py --mode online --public

输出：
  results/swanlab_links.jsonl       每行 {run_id, swanlab_run_id, url, group}
  results/swanlab_upload_state.json 已完成 run_id 集合（断点续传）
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DB = ROOT / "experiments" / "final_records" / "tracker.sqlite"
LINKS = ROOT / "results" / "swanlab_links.jsonl"
STATE = ROOT / "results" / "swanlab_upload_state.json"
PROJECT = "zhixun-engine"
WORKSPACE = "Migratory_Bird"


def load_state(path: Path) -> set[str]:
    if path.exists():
        try:
            return set(json.loads(path.read_text(encoding="utf-8")).get("done", []))
        except Exception:
            return set()
    return set()


def save_state(path: Path, done: set[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"done": sorted(done)}, ensure_ascii=False), encoding="utf-8")


def load_runs() -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    runs = [dict(r) for r in conn.execute(
        "select * from runs order by created_at, run_id")]
    conn.close()
    return runs


def load_metrics(run_id: str) -> list[dict]:
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    rows = [dict(r) for r in conn.execute(
        "select key, step, value from metrics where run_id=? order by key, step", (run_id,))]
    conn.close()
    return rows


def summary_of(run: dict) -> dict:
    try:
        s = json.loads(run.get("summary_json") or "{}")
    except Exception:
        s = {}
    out = {}
    for key, path in {
        "stream_macro_f1": ("hidden", "stream_macro_f1"),
        "final_phase_macro_f1": ("hidden", "final_phase_macro_f1"),
        "stream_accuracy": ("hidden", "stream_accuracy"),
        "gain_points": ("comparison", "stream_gain_points"),
        "win_windows": ("comparison", "win_windows"),
        "lose_windows": ("comparison", "lose_windows"),
        "labels": ("cost", "labels"),
        "compute_units": ("cost", "compute_units"),
        "tokens": ("cost", "tokens"),
        "cost_scalar": ("cost", "scalar"),
        "audit_verified": ("audit_verified",),
    }.items():
        cur = s
        try:
            for p in path:
                cur = cur[p]
            if isinstance(cur, (int, float)):
                out[key] = float(cur)
        except Exception:
            pass
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", default="online", choices=["online", "local", "offline"])
    ap.add_argument("--limit", type=int, default=0, help="只传前 N 条（0=全部）")
    ap.add_argument("--offset", type=int, default=0)
    ap.add_argument("--public", action="store_true", help="项目公开（评委可直接打开链接）")
    ap.add_argument("--sleep", type=float, default=0.0, help="每条 run 之间 sleep 秒数")
    ap.add_argument("--project", default=PROJECT)
    ap.add_argument("--workspace", default=WORKSPACE)
    ap.add_argument("--part", type=int, default=0, help="并行分片编号（0..parts-1）")
    ap.add_argument("--parts", type=int, default=1, help="并行分片数")
    ap.add_argument("--state", default=str(STATE))
    ap.add_argument("--links", default=str(LINKS))
    args = ap.parse_args()

    # swanlab 的本地模式会把日志写到 ./.swanlab；统一放到 results 下
    os.environ.setdefault("SWANLAB_LOG_DIR", str(ROOT / "results" / "swanlab_logs"))
    import swanlab  # 延迟导入，避免无登录时的副作用

    state_path = Path(args.state)
    links_path = Path(args.links)
    legacy_state = load_state(STATE) | load_state(state_path)
    runs = load_runs()
    todo = [r for r in runs if r["run_id"] not in legacy_state][args.offset:]
    if args.parts > 1:
        todo = todo[args.part::args.parts]
    if args.limit:
        todo = todo[:args.limit]
    if not todo:
        print("没有待上传 run（全部已完成或为空）")
        return 0

    done = load_state(STATE) | load_state(state_path)
    print(f"待上传 {len(todo)} / 总计 {len(runs)}；mode={args.mode}；project={args.project}；workspace={args.workspace}")
    links_path.parent.mkdir(parents=True, exist_ok=True)
    ok = fail = 0
    try:
        for i, run in enumerate(todo, 1):
            rid = run["run_id"]
            tags = []
            try:
                tags = json.loads(run.get("tags_json") or "[]")
            except Exception:
                pass
            cfg = {"scenario": tags[0] if tags else "", "seed": None,
                   "group": run.get("group_name", ""), "kind": run.get("kind", ""),
                   "stage": run.get("stage", ""), "run_id": rid}
            desc = f"智训引擎 · {run.get('name') or rid}（由本地 SwanLab 兼容记录回放）"
            last_err = None
            succeeded = False
            for attempt in range(3):
                try:
                    r = swanlab.init(
                        project=args.project,
                        workspace=args.workspace,
                        name=rid,
                        description=desc,
                        group=run.get("group_name") or None,
                        tags=tags[:2],
                        config=cfg,
                        mode=args.mode,
                        public=bool(args.public),
                        reinit=True,
                    )
                    sm = summary_of(run)
                    if sm:
                        r.log({f"summary/{k}": v for k, v in sm.items()})
                    by_step: dict[int, dict[str, float]] = {}
                    for m in load_metrics(rid):
                        try:
                            v = float(m["value"])
                        except (TypeError, ValueError):
                            continue
                        if v != v or v in (float("inf"), float("-inf")):
                            continue
                        by_step.setdefault(int(m["step"]), {})[m["key"]] = v
                    for st in sorted(by_step):
                        r.log(by_step[st], step=st)
                    try:
                        url = str(r.url or "")
                    except Exception:
                        url = ""
                    rid_cloud = str(getattr(r, "id", "") or "")
                    r.finish("success")
                    if args.mode == "online":
                        with open(links_path, "a", encoding="utf-8") as f:
                            f.write(json.dumps({"run_id": rid, "swanlab_run_id": rid_cloud,
                                                "url": url, "group": run.get("group_name", "")},
                                               ensure_ascii=False) + "\n")
                    done.add(rid)
                    save_state(state_path, done)
                    ok += 1
                    succeeded = True
                    if i % 10 == 0 or i <= 5:
                        print(f"[part{args.part} {i}/{len(todo)}] ok={ok} fail={fail} last={rid} url={url}")
                    break
                except Exception as e:  # noqa: BLE001
                    last_err = e
                    if attempt < 2:
                        time.sleep(1.5 * (attempt + 1))
                        continue
                    fail += 1
                    print(f"[part{args.part} {i}/{len(todo)}] FAILED {rid}: {type(e).__name__}: {e}")
                    msg = str(e).lower()
                    if any(k in msg for k in ("quota", "limit exceeded", "forbidden",
                                              "unauthorized", "payment", "plan")):
                        print("疑似额度/权限限制，停止本分片")
                        (links_path.parent / f"quota_blocked_part{args.part}.txt").write_text(
                            f"{now_iso()}\t{rid}\t{e}", encoding="utf-8")
                        return 3
                    break
            if not succeeded and fail >= 5 and ok == 0:
                print("连续失败且无成功，停止（检查登录/额度/网络）")
                break
            if args.sleep:
                time.sleep(args.sleep)
    finally:
        try:
            swanlab.finish("success")
        except Exception:
            pass
    print(f"完成：ok={ok} fail={fail}；链接映射：{LINKS}")
    return 0 if fail == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
