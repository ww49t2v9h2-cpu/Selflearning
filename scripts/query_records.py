#!/usr/bin/env python3
"""实验记录查询 CLI：列表 / 详情 / 指标 / 两次 run 对比 / 导出（CSV/JSON/Markdown）。

数据来源：`src/observability/tracker.py` 的本地镜像（SQLite+JSONL）。
若启用了 SwanLab，本地镜像与 SwanLab run 一一对应（记录里带 swanlab_run_id/url），
本脚本离线即可查询全部迭代实验，无需联网。

用法示例
--------
```bash
PY=python3

# 1) 列出所有 run（按时间倒序）
$PY scripts/query_records.py list

# 2) 过滤：main 组的实验类 run，phase 2，最近 20 条
$PY scripts/query_records.py list --group main --kind experiment --phase 2 --limit 20

# 3) 看某次 run 的详情 + 指标
$PY scripts/query_records.py show --run-id main-20261116-2210-a1b2c3 --metrics

# 4) 对比两次 run（summary + 最新指标 + 差值）
$PY scripts/query_records.py compare --run-a main-20261116-2210-a1b2c3 \
                                    --run-b main-20261116-2230-d4e5f6

# 5) 导出 CSV / Markdown（可带过滤）
$PY scripts/query_records.py export --fmt csv --out results/runs.csv --group main

# 6) 统计
$PY scripts/query_records.py stats
```
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.observability.tracker import LocalTracker  # noqa: E402


# --------------------------------------------------------------------------
def _default_dir(cfg_path: str | None = None) -> Path:
    cfg = load_config(cfg_path)
    return ROOT / str(cfg["tracker"]["dir"])


def _open_tracker(args) -> LocalTracker:
    root = Path(args.dir) if args.dir else _default_dir(args.config)
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        raise SystemExit(f"[query_records] 记录目录不存在: {root}\n"
                         f"提示：先运行闭环（python -m src.cli run）生成实验记录")
    return LocalTracker(root=root)


def _filters_from_args(args) -> dict:
    filters: dict = {}
    for key in ("group", "kind", "stage", "status", "tag", "name_like",
                "parent_run"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    if getattr(args, "phase", None) is not None:
        filters["phase"] = args.phase
    for key in ("since", "until"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    filters["limit"] = getattr(args, "limit", 50)
    filters["order"] = getattr(args, "order", "desc")
    return filters


def _fmt_cost(run: dict) -> str:
    summary = run.get("summary") or {}
    cost = summary.get("cost") if isinstance(summary.get("cost"), dict) else {}
    if not cost:
        return "-"
    parts = []
    for key, label in (("compute_units", "算力"), ("labels", "标签"),
                       ("tokens", "token")):
        if cost.get(key) is not None:
            parts.append(f"{label}{cost[key]}")
    return "/".join(parts) or "-"


def _table(rows: list[dict], columns: list[tuple[str, str]]) -> str:
    if not rows:
        return "（无匹配记录）\n"
    widths = []
    for key, title in columns:
        width = max([len(title)] + [len(str(run.get(key, "-"))) for run in rows])
        widths.append(min(width, 44))
    header = " | ".join(title.ljust(widths[i])
                        for i, (_, title) in enumerate(columns))
    lines = [header, "-+-".join("-" * w for w in widths)]
    for run in rows:
        cells = []
        for i, (key, _) in enumerate(columns):
            text = str(run.get(key, "-")).replace("\n", " ")
            cells.append(text[:widths[i]].ljust(widths[i]))
        lines.append(" | ".join(cells))
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def cmd_list(args) -> int:
    tracker = _open_tracker(args)
    filters = _filters_from_args(args)
    runs = tracker.query(filters)
    if args.json:
        print(json.dumps(runs, ensure_ascii=False, indent=2))
        return 0
    print(f"[query_records] root={tracker.root} 匹配 {len(runs)} 条 "
          f"（filters={json.dumps(filters, ensure_ascii=False)}）\n")
    columns = [
        ("created_at_iso", "创建时间"), ("run_id", "run_id"),
        ("group", "组"), ("kind", "类型"), ("stage", "阶段"),
        ("status", "状态"), ("t", "t"), ("phase", "phase"),
        ("metric_count", "指标数"), ("swanlab_run_id", "SwanLab"),
        ("name", "名称"),
    ]
    print(_table(runs, columns))
    return 0


def cmd_show(args) -> int:
    tracker = _open_tracker(args)
    run = tracker.get_run(args.run_id, include_metrics=True)
    if run is None:
        print(f"[query_records] 未找到 run: {args.run_id}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps(run, ensure_ascii=False, indent=2))
        return 0
    print(f"# run 详情：{run['run_id']}\n")
    for key in ("group", "name", "kind", "stage", "status", "t", "phase",
                "created_at_iso", "finished_at_iso", "parent_run",
                "swanlab_run_id", "swanlab_url", "swanlab_mode", "note"):
        print(f"- {key}: {run.get(key)}")
    print(f"- config: {json.dumps(run.get('config') or {}, ensure_ascii=False)[:500]}")
    print(f"- summary: {json.dumps(run.get('summary') or {}, ensure_ascii=False)[:500]}")
    print(f"- artifacts: {len(run.get('artifacts') or [])}")
    for art in run.get("artifacts") or []:
        print(f"  - step={art['step']} {art['path']} "
              f"({art['size']} bytes, sha256={art['sha256'][:12]}…)")
    if args.metrics:
        metrics = run.get("metrics") or []
        print(f"\n## 指标（{len(metrics)} 条，最近 30 条）")
        print(_table(metrics[-30:], [("step", "step"), ("key", "key"),
                                     ("value", "value"), ("t", "t")]))
        latest = tracker.latest_metrics(run["run_id"])
        print(f"\n## 最新值\n{json.dumps(latest, ensure_ascii=False, indent=2)}")
    return 0


def cmd_metrics(args) -> int:
    tracker = _open_tracker(args)
    run = tracker.get_run(args.run_id)
    if run is None:
        print(f"[query_records] 未找到 run: {args.run_id}", file=sys.stderr)
        return 2
    keys = [k.strip() for k in args.keys.split(",") if k.strip()] if args.keys else None
    rows = tracker.metrics(args.run_id, keys=keys)
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w", encoding="utf-8") as f:
            f.write("run_id,key,step,value,t\n")
            for row in rows:
                f.write(f"{row['run_id']},{row['key']},{row['step']},"
                        f"{row['value']},{row['t']}\n")
        print(f"[query_records] 已导出 {len(rows)} 条指标 → {out}")
        return 0
    print(_table(rows, [("step", "step"), ("key", "key"),
                        ("value", "value"), ("t", "t")]))
    return 0


def cmd_compare(args) -> int:
    tracker = _open_tracker(args)
    a = tracker.get_run(args.run_a, include_metrics=True)
    b = tracker.get_run(args.run_b, include_metrics=True)
    for label, run in (("run-a", a), ("run-b", b)):
        if run is None:
            print(f"[query_records] 未找到 {label}: "
                  f"{args.run_a if label == 'run-a' else args.run_b}",
                  file=sys.stderr)
            return 2
    keys = [k.strip() for k in args.keys.split(",")] if args.keys else None
    latest_a = tracker.latest_metrics(args.run_a, keys=keys)
    latest_b = tracker.latest_metrics(args.run_b, keys=keys)
    all_keys = sorted(set(latest_a) | set(latest_b))
    rows = []
    for key in all_keys:
        va, vb = latest_a.get(key), latest_b.get(key)
        delta = (None if va is None or vb is None else vb - va)
        rows.append({"key": key,
                     "run_a": va if va is not None else "-",
                     "run_b": vb if vb is not None else "-",
                     "delta": (f"{delta:+.6g}" if delta is not None else "-")})
    print(f"# 对比\n- run-a: {args.run_a}\n- run-b: {args.run_b}\n")
    for label, run in (("run-a", a), ("run-b", b)):
        print(f"- {label} summary: "
              f"{json.dumps(run.get('summary') or {}, ensure_ascii=False)[:300]}")
    summary_a = (a.get("summary") or {}).get("metrics") or {}
    summary_b = (b.get("summary") or {}).get("metrics") or {}
    if summary_a or summary_b:
        rows = []
        for key in sorted(set(summary_a) | set(summary_b)):
            va, vb = summary_a.get(key), summary_b.get(key)
            delta = (None if va is None or vb is None else vb - va)
            rows.append({"key": key,
                         "run_a": va if va is not None else "-",
                         "run_b": vb if vb is not None else "-",
                         "delta": (f"{delta:+.6g}" if delta is not None else "-")})
        print("\n## summary.metrics 对比（run_b - run_a）")
        print(_table(rows, [("key", "指标"), ("run_a", "run-a"),
                            ("run_b", "run-b"), ("delta", "差值")]))
    print("\n## 最新指标对比（run_b - run_a）")
    print(_table(rows, [("key", "指标"), ("run_a", "run-a"),
                        ("run_b", "run-b"), ("delta", "差值")]))
    if args.out:
        out = Path(args.out)
        if not out.is_absolute():
            out = ROOT / out
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"run_a": args.run_a, "run_b": args.run_b,
                                   "latest_a": latest_a, "latest_b": latest_b,
                                   "diff": rows}, ensure_ascii=False, indent=2),
                       encoding="utf-8")
        print(f"[query_records] 对比结果 → {out}")
    return 0


def cmd_export(args) -> int:
    tracker = _open_tracker(args)
    filters = _filters_from_args(args)
    out = tracker.export(args.fmt, args.out, filters)
    n = len(tracker.query(filters))
    print(f"[query_records] 已导出 {n} 条记录（{args.fmt}）→ {out}")
    return 0


def cmd_stats(args) -> int:
    tracker = _open_tracker(args)
    stats = tracker.stats()
    if args.json:
        print(json.dumps(stats, ensure_ascii=False, indent=2))
        return 0
    print(f"# 实验记录统计\n")
    print(f"- 根目录: {stats['root']}")
    print(f"- run 数: {stats['n_runs']}  指标点: {stats['n_metrics']}  "
          f"artifact: {stats.get('n_artifacts', 0)}")
    print(f"- 按组: {json.dumps(stats.get('by_group', {}), ensure_ascii=False)}")
    print(f"- 按类型: {json.dumps(stats.get('by_kind', {}), ensure_ascii=False)}")
    health = tracker.health()
    print(f"- 后端: {health.get('backend')}  schema={health.get('schema_version')}")
    if health.get("warnings"):
        print(f"- 告警: {len(health['warnings'])} 条（见 tracker_warnings.jsonl）")
    return 0


# --------------------------------------------------------------------------
def _add_common(p: argparse.ArgumentParser, with_output: bool = False) -> None:
    p.add_argument("--dir", default=None,
                   help="记录根目录（默认 cfg.tracker.dir = experiments/runs）")
    p.add_argument("--config", default=None, help="可选配置文件路径（读 tracker.dir）")
    if with_output:
        p.add_argument("--out", default=None, help="导出路径")
        p.add_argument("--fmt", default="csv", choices=["csv", "json", "md"])


def _add_filters(p: argparse.ArgumentParser) -> None:
    p.add_argument("--group", default=None)
    p.add_argument("--kind", default=None)
    p.add_argument("--stage", default=None)
    p.add_argument("--phase", type=int, default=None)
    p.add_argument("--status", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--name-like", dest="name_like", default=None)
    p.add_argument("--parent-run", dest="parent_run", default=None)
    p.add_argument("--since", default=None, help="创建时间下界（ISO8601/epoch）")
    p.add_argument("--until", default=None, help="创建时间上界")
    p.add_argument("--limit", type=int, default=50)
    p.add_argument("--order", default="desc", choices=["asc", "desc"])
    p.add_argument("--json", action="store_true", help="输出原始 JSON")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="智训引擎 · 实验记录查询工具（SwanLab 兼容本地镜像）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_list = sub.add_parser("list", help="按条件列出 run")
    _add_common(p_list)
    _add_filters(p_list)
    p_list.set_defaults(func=cmd_list)

    p_show = sub.add_parser("show", help="查看单个 run 详情")
    _add_common(p_show)
    p_show.add_argument("--run-id", dest="run_id", required=True)
    p_show.add_argument("--metrics", action="store_true", help="同时打印指标")
    p_show.add_argument("--json", action="store_true")
    p_show.set_defaults(func=cmd_show)

    p_metrics = sub.add_parser("metrics", help="导出/查看某个 run 的指标")
    _add_common(p_metrics)
    p_metrics.add_argument("--run-id", dest="run_id", required=True)
    p_metrics.add_argument("--keys", default=None, help="逗号分隔的指标名")
    p_metrics.add_argument("--out", default=None, help="导出 CSV 路径")
    p_metrics.set_defaults(func=cmd_metrics)

    p_cmp = sub.add_parser("compare", help="对比两次 run 的指标")
    _add_common(p_cmp)
    p_cmp.add_argument("--run-a", dest="run_a", required=True)
    p_cmp.add_argument("--run-b", dest="run_b", required=True)
    p_cmp.add_argument("--keys", default=None)
    p_cmp.add_argument("--out", default=None)
    p_cmp.set_defaults(func=cmd_compare)

    p_export = sub.add_parser("export", help="导出记录（csv/json/md）")
    _add_common(p_export, with_output=True)
    _add_filters(p_export)
    p_export.set_defaults(func=cmd_export)

    p_stats = sub.add_parser("stats", help="统计概览")
    _add_common(p_stats)
    p_stats.add_argument("--json", action="store_true")
    p_stats.set_defaults(func=cmd_stats)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.command == "export" and not args.out:
        print("[query_records] export 需要 --out", file=sys.stderr)
        return 2
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
