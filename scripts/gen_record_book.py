#!/usr/bin/env python3
"""生成《实验记录汇编》（Markdown + xlsx），满足赛题"实验记录规范"要求。

输入：本地 tracker 镜像（SQLite+JSONL，SwanLab 兼容）。输出：
1. Markdown：封面/记录规范说明/总览/成本汇总/分组明细/指标明细/复现与上云附录；
2. xlsx（openpyxl 软依赖）：说明 / 总览 / 指标明细 / 成本汇总 四个 sheet；
   未安装 openpyxl 时自动降级为同目录 CSV 文件并打印提示（不中断）。

用法
----
```bash
PY=python3

# 全量汇编（默认输出 results/record_book/实验记录汇编.md + .xlsx）
$PY scripts/gen_record_book.py

# 只看 main 组、实验类记录
$PY scripts/gen_record_book.py --group main --kind experiment

# 自定义标题与输出路径（提交前把最终数字交给 Lead 复核）
$PY scripts/gen_record_book.py --title "智训引擎·实验记录汇编（校赛版）" \
    --out-md deliverables/实验记录汇编.md \
    --out-xlsx deliverables/实验记录汇编.xlsx
```
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.core.config import load_config  # noqa: E402
from src.observability.tracker import LocalTracker  # noqa: E402

try:  # openpyxl 软依赖
    from openpyxl import Workbook  # type: ignore
    from openpyxl.styles import Alignment, Font, PatternFill  # type: ignore
    _HAS_OPENPYXL = True
except Exception:  # noqa: BLE001
    _HAS_OPENPYXL = False

DEFAULT_OUT_DIR = ROOT / "results" / "record_book"


# --------------------------------------------------------------------------
def _json_brief(obj, limit: int = 600) -> str:
    try:
        text = json.dumps(obj or {}, ensure_ascii=False, sort_keys=True)
    except TypeError:
        text = str(obj)
    return text if len(text) <= limit else text[:limit] + "…"


def _cost_of(run: dict) -> dict:
    summary = run.get("summary") or {}
    cost = summary.get("cost")
    if isinstance(cost, dict):
        return cost
    if isinstance(summary.get("cost_actual"), dict):
        return summary["cost_actual"]
    return {}


def _collect_cost(runs: list[dict]) -> dict:
    total = {"compute_units": 0.0, "labels": 0, "tokens": 0}
    for run in runs:
        cost = _cost_of(run)
        for key in total:
            try:
                total[key] += float(cost.get(key, 0) or 0)
            except (TypeError, ValueError):
                pass
    return total


def _key_metrics(run: dict, limit: int = 8) -> dict:
    latest = run.get("metrics_summary") or {}
    preferred = ["probe_macro_f1", "macro_f1", "pred_stability",
                 "class_recall_min", "accuracy", "loss", "cost_compute",
                 "compute_units", "wall_s"]
    ordered = {k: latest[k] for k in preferred if k in latest}
    for key, value in latest.items():
        if len(ordered) >= limit:
            break
        if key not in ordered:
            ordered[key] = value
    return ordered


# --------------------------------------------------------------------------
def build_markdown(runs: list[dict], stats: dict, title: str,
                   tracker: LocalTracker, filters: dict) -> str:
    now = time.strftime("%Y-%m-%d %H:%M:%S")
    groups: dict[str, list[dict]] = {}
    others: list[dict] = []
    for run in runs:
        stage = str(run.get("stage", ""))
        if stage == "rollback":
            others.append(run)  # 安全类事件单独成章，便于审计
            continue
        groups.setdefault(str(run.get("group", "misc")), []).append(run)
    total_cost = _collect_cost(runs)
    lines: list[str] = []
    add = lines.append

    add(f"# {title}\n")
    add("> 本汇编由 `scripts/gen_record_book.py` 从实验记录镜像自动生成，"
        "与 SwanLab run 一一对应；记录不可手工编辑，保证可审计。\n")
    add(f"- 生成时间：{now}")
    add(f"- 记录后端：`{tracker.backend}`（本地 SQLite+JSONL 镜像，"
        f"根目录 `{tracker.root}`）")
    add(f"- Schema 版本：`{tracker.health().get('schema_version')}`；"
        f"筛选条件：`{_json_brief(filters, 300)}`")
    add(f"- run 总数：**{len(runs)}**；指标点总数：**{stats.get('n_metrics', 0)}**；"
        f"artifact 总数：**{stats.get('n_artifacts', 0)}**")
    add(f"- 累计成本：算力 **{total_cost['compute_units']:.2f} units**，"
        f"标签 **{int(total_cost['labels'])}** 条，"
        f"token **{int(total_cost['tokens'])}**")
    add("")

    add("## 1. 实验记录规范（自动执行）\n")
    add("| 项 | 规范 |")
    add("|---|---|")
    add("| run_id | `{group}-{yyyyMMdd-HHMMSS}-{6hex}`，全链路唯一 |")
    add("| 记录字段 | 与 `RunRecord`/`MetricPoint` 一致（group/name/kind/stage/"
        "config/summary/tags/t/phase/parent_run/created_at） |")
    add("| 目录 | `experiments/runs/{run_id}/record.json` + `metrics.jsonl` + "
        "`artifacts.json`；索引 `tracker.sqlite`；审计 `records.jsonl` |")
    add("| 指标 | 每个 step 一行 JSONL + SQLite 双写，含指标名/步数/时间戳 |")
    add("| 成本 | 每个实验记录 compute_units/labels/tokens/wall_s 实测值 |")
    add("| 审计追溯 | Decision → run_id → 代码 commit + config_used.json + seed |")
    add("| SwanLab | `backend=swanlab` 时一一对应；离线用 `swanlab sync` 上云 |")
    add("")

    add("## 2. 实验总览\n")
    add("| # | run_id | 组 | 名称 | 类型 | 阶段 | 状态 | t/phase | 指标数 | "
        "artifact | SwanLab | 创建时间 |")
    add("|---|---|---|---|---|---|---|---|---|---|---|---|")
    for i, run in enumerate(runs, 1):
        add(f"| {i} | `{run.get('run_id')}` | {run.get('group')} | "
            f"{run.get('name') or '-'} | {run.get('kind')} | {run.get('stage')} | "
            f"{run.get('status')} | {run.get('t')}/{run.get('phase')} | "
            f"{run.get('metric_count', 0)} | "
            f"{len(run.get('artifacts') or [])} | "
            f"{run.get('swanlab_run_id') or '-'} | "
            f"{run.get('created_at_iso') or run.get('created_at')} |")
    add("")

    add("## 3. 成本汇总\n")
    add("| 组 | run 数 | 算力 units | 标签 | token |")
    add("|---|---|---|---|---|")
    for group, group_runs in sorted(groups.items()):
        cost = _collect_cost(group_runs)
        add(f"| {group} | {len(group_runs)} | {cost['compute_units']:.2f} | "
            f"{int(cost['labels'])} | {int(cost['tokens'])} |")
    add(f"| **合计** | {len(runs)} | {total_cost['compute_units']:.2f} | "
        f"{int(total_cost['labels'])} | {int(total_cost['tokens'])} |")
    add("")

    section = 4
    for group, group_runs in sorted(groups.items()):
        add(f"## {section}. 分组明细：`{group}`（{len(group_runs)} 条）\n")
        section += 1
        for i, run in enumerate(group_runs, 1):
            add(f"### {section - 1}.{i} `{run.get('run_id')}` — "
                f"{run.get('name') or run.get('kind')}\n")
            add(f"- 类型/阶段：`{run.get('kind')}` / `{run.get('stage')}`；"
                f"状态 `{run.get('status')}`；t={run.get('t')}, "
                f"phase={run.get('phase')}；父 run："
                f"`{run.get('parent_run') or '-'}`")
            if run.get("tags"):
                add(f"- 标签：{', '.join(map(str, run['tags']))}")
            cost = _cost_of(run)
            if cost:
                add(f"- 成本：{_json_brief(cost)}")
            if run.get("config"):
                add(f"- 配置：`{_json_brief(run['config'], 500)}`")
            if run.get("summary"):
                add(f"- 结论/summary：`{_json_brief(run['summary'], 500)}`")
            key_metrics = _key_metrics(run)
            if key_metrics:
                add(f"- 关键指标（最新值）：`{_json_brief(key_metrics, 400)}`")
            if run.get("artifacts"):
                add("- artifact：")
                for art in run["artifacts"]:
                    add(f"  - `{art.get('path')}`（step={art.get('step')}, "
                        f"sha256={str(art.get('sha256', ''))[:12]}…）")
            if run.get("swanlab_run_id"):
                add(f"- SwanLab：run_id=`{run.get('swanlab_run_id')}` "
                    f"url={run.get('swanlab_url') or '-'}")
            add("")
        add("")

    if others:
        add(f"## {section}. 安全事件（回滚/异常类，审计用）\n")
        section += 1
        add("| run_id | 类型 | 状态 | summary | 创建时间 |")
        add("|---|---|---|---|---|")
        for run in others:
            add(f"| `{run.get('run_id')}` | {run.get('kind')} | "
                f"{run.get('status')} | `{_json_brief(run.get('summary'), 300)}` | "
                f"{run.get('created_at_iso')} |")
        add("")

    add(f"## {section}. 指标明细（每 run 最新值，完整序列见 metrics.jsonl / xlsx）\n")
    add("| run_id | 指标 | 最新值 |")
    add("|---|---|---|")
    for run in runs:
        latest = run.get("metrics_summary") or {}
        for key, value in sorted(latest.items()):
            add(f"| `{run.get('run_id')}` | {key} | {value} |")
    add("")

    add(f"## {section + 1}. 复现与上云\n")
    add("```bash")
    add("# 1) 复现闭环（同一配置 + seed 必须得到同一结果）")
    add("PY=python3")
    add("$PY -m src.cli run --config configs/default.json")
    add("")
    add("# 2) 生成记录汇编（本文件）")
    add("$PY scripts/gen_record_book.py")
    add("")
    add("# 3) 查询/对比/导出")
    add("$PY scripts/query_records.py list --group main")
    add("$PY scripts/query_records.py compare --run-a <RUN_A> --run-b <RUN_B>")
    add("$PY scripts/query_records.py export --fmt csv --out results/runs.csv")
    add("")
    add("# 4) SwanLab 离线记录上云（需 pip install swanlab && swanlab login）")
    add("swanlab sync experiments/runs/swanlog")
    add("swanlab watch experiments/runs/swanlog   # 本地看板")
    add("```")
    add("")
    add("> 说明：校赛阶段在无 PyTRi0 额度、无网络环境完成全部实验记录；"
        "SwanLab 采用 `mode=offline` 本地记录（软依赖，缺失时自动降级为本地镜像），"
        "联网后一条 `swanlab sync` 即可把全部迭代实验上云，run 与记录一一对应。")
    return "\n".join(lines) + "\n"


# --------------------------------------------------------------------------
def _write_xlsx(path: Path, runs: list[dict], stats: dict, title: str,
                filters: dict) -> None:
    wb = Workbook()
    header_font = Font(bold=True, color="FFFFFF")
    header_fill = PatternFill("solid", fgColor="2F5597")

    def _sheet(ws, headers: list[str], rows: list[list]):
        ws.append(headers)
        for cell in ws[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="center")
        for row in rows:
            ws.append(row)
        for i, width in enumerate(_col_widths(headers, rows), 1):
            ws.column_dimensions[chr(64 + i) if i <= 26 else "A"].width = width
        ws.freeze_panes = "A2"

    meta_ws = wb.active
    meta_ws.title = "说明"
    meta_ws["A1"] = title
    meta_ws["A1"].font = Font(bold=True, size=14)
    lines = [
        ("生成时间", time.strftime("%Y-%m-%d %H:%M:%S")),
        ("run 总数", len(runs)),
        ("指标点总数", stats.get("n_metrics", 0)),
        ("artifact 总数", stats.get("n_artifacts", 0)),
        ("筛选条件", _json_brief(filters, 300)),
        ("run_id 规范", "{group}-{yyyyMMdd-HHMMSS}-{6hex}"),
        ("记录字段", "与 src/core/spec.py::RunRecord/MetricPoint 一致"),
        ("审计追溯", "Decision → run_id → config+seed → 代码 commit；"
                     "audit 哈希链 experiments/audit/*.jsonl"),
        ("SwanLab 上云", "swanlab sync experiments/runs/swanlog"),
    ]
    for i, (key, value) in enumerate(lines, 3):
        meta_ws[f"A{i}"] = key
        meta_ws[f"A{i}"].font = Font(bold=True)
        meta_ws[f"B{i}"] = value
    meta_ws.column_dimensions["A"].width = 18
    meta_ws.column_dimensions["B"].width = 80

    overview_headers = ["run_id", "group", "name", "kind", "stage", "status",
                        "t", "phase", "parent_run", "created_at",
                        "metric_count", "artifact_count", "swanlab_run_id",
                        "swanlab_url", "summary"]
    overview_rows = [[run.get("run_id"), run.get("group"), run.get("name"),
                      run.get("kind"), run.get("stage"), run.get("status"),
                      run.get("t"), run.get("phase"), run.get("parent_run") or "",
                      run.get("created_at_iso") or run.get("created_at"),
                      run.get("metric_count", 0),
                      len(run.get("artifacts") or []),
                      run.get("swanlab_run_id") or "",
                      run.get("swanlab_url") or "",
                      _json_brief(run.get("summary"), 500)]
                     for run in runs]
    _sheet(wb.create_sheet("总览"), overview_headers, overview_rows)

    metric_headers = ["run_id", "指标", "最新值"]
    metric_rows = []
    for run in runs:
        for key, value in sorted((run.get("metrics_summary") or {}).items()):
            metric_rows.append([run.get("run_id"), key, value])
    _sheet(wb.create_sheet("指标明细"), metric_headers, metric_rows)

    cost_headers = ["run_id", "kind", "stage", "compute_units", "labels",
                    "tokens", "wall_s", "gain_per_compute"]
    cost_rows = []
    for run in runs:
        cost = _cost_of(run)
        summary = run.get("summary") or {}
        cost_rows.append([run.get("run_id"), run.get("kind"),
                          run.get("stage"),
                          cost.get("compute_units", 0),
                          cost.get("labels", 0), cost.get("tokens", 0),
                          cost.get("wall_s", 0),
                          summary.get("gain_per_compute", "")])
    _sheet(wb.create_sheet("成本汇总"), cost_headers, cost_rows)
    wb.save(str(path))


def _col_widths(headers: list[str], rows: list[list]) -> list[int]:
    widths = []
    for i, header in enumerate(headers):
        values = [len(str(row[i])) for row in rows[:200]] if rows else []
        widths.append(min(max([len(str(header))] + values) + 2, 48))
    return widths


def _write_csv_fallback(out_dir: Path, runs: list[dict]) -> list[Path]:
    """未安装 openpyxl 时的降级：拆成三个 CSV，保证信息不丢。"""
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    overview = out_dir / "record_book_overview.csv"
    keys = ["run_id", "group", "name", "kind", "stage", "status", "t", "phase",
            "created_at", "metric_count", "swanlab_run_id", "swanlab_url"]
    with open(overview, "w", encoding="utf-8") as f:
        f.write(",".join(keys) + "\n")
        for run in runs:
            f.write(",".join(str(run.get(k, "")).replace(",", ";")
                             for k in keys) + "\n")
    paths.append(overview)
    metrics = out_dir / "record_book_metrics.csv"
    with open(metrics, "w", encoding="utf-8") as f:
        f.write("run_id,key,value\n")
        for run in runs:
            for key, value in sorted((run.get("metrics_summary") or {}).items()):
                f.write(f"{run.get('run_id')},{key},{value}\n")
    paths.append(metrics)
    costs = out_dir / "record_book_costs.csv"
    with open(costs, "w", encoding="utf-8") as f:
        f.write("run_id,compute_units,labels,tokens,wall_s\n")
        for run in runs:
            cost = _cost_of(run)
            f.write(f"{run.get('run_id')},{cost.get('compute_units', 0)},"
                    f"{cost.get('labels', 0)},{cost.get('tokens', 0)},"
                    f"{cost.get('wall_s', 0)}\n")
    paths.append(costs)
    return paths


# --------------------------------------------------------------------------
def generate(args) -> dict:
    cfg = load_config(args.config)
    root = Path(args.dir) if args.dir else ROOT / str(cfg["tracker"]["dir"])
    if not root.is_absolute():
        root = ROOT / root
    if not root.exists():
        raise SystemExit(f"[gen_record_book] 记录目录不存在: {root}")
    tracker = LocalTracker(root=root)
    filters: dict = {"limit": args.limit, "order": "asc",
                     "include_metrics": True, "include_artifacts": True}
    for key in ("group", "kind", "stage", "status"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    for key in ("since", "until"):
        value = getattr(args, key, None)
        if value:
            filters[key] = value
    runs = tracker.query(filters)
    stats = tracker.stats()
    out_md = Path(args.out_md) if args.out_md else DEFAULT_OUT_DIR / "实验记录汇编.md"
    out_md = out_md if out_md.is_absolute() else ROOT / out_md
    out_xlsx = Path(args.out_xlsx) if args.out_xlsx else out_md.with_suffix(".xlsx")
    out_xlsx = out_xlsx if out_xlsx.is_absolute() else ROOT / out_xlsx
    out_md.parent.mkdir(parents=True, exist_ok=True)

    markdown = build_markdown(runs, stats, args.title, tracker, filters)
    out_md.write_text(markdown, encoding="utf-8")

    result = {"out_md": str(out_md), "n_runs": len(runs),
              "n_metrics": stats.get("n_metrics", 0), "xlsx": ""}
    if args.no_xlsx:
        print("[gen_record_book] --no-xlsx：跳过 xlsx")
    elif _HAS_OPENPYXL:
        _write_xlsx(out_xlsx, runs, stats, args.title, filters)
        result["xlsx"] = str(out_xlsx)
    else:
        csv_paths = _write_csv_fallback(out_md.parent, runs)
        result["csv_fallback"] = [str(p) for p in csv_paths]
        print("[gen_record_book] 未安装 openpyxl（pip install openpyxl），"
              "已降级输出 CSV：")
        for path in csv_paths:
            print(f"  - {path}")
    tracker.close()
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="生成《实验记录汇编》（Markdown + xlsx）",
        formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    parser.add_argument("--dir", default=None,
                        help="记录根目录（默认 experiments/runs）")
    parser.add_argument("--config", default=None, help="可选配置文件路径")
    parser.add_argument("--group", default=None)
    parser.add_argument("--kind", default=None)
    parser.add_argument("--stage", default=None)
    parser.add_argument("--status", default=None)
    parser.add_argument("--since", default=None)
    parser.add_argument("--until", default=None)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument("--title", default="智训引擎·实验记录汇编（Industrial Model Self-Evolution）")
    parser.add_argument("--out-md", dest="out_md", default=None)
    parser.add_argument("--out-xlsx", dest="out_xlsx", default=None)
    parser.add_argument("--no-xlsx", dest="no_xlsx", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = generate(args)
    print(f"[gen_record_book] Markdown → {result['out_md']}")
    if result.get("xlsx"):
        print(f"[gen_record_book] xlsx     → {result['xlsx']}")
    print(f"[gen_record_book] 收录 {result['n_runs']} 条 run / "
          f"{result['n_metrics']} 个指标点")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
