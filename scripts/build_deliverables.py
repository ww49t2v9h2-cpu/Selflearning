"""构建交付物：填充参赛方案（真实数字/表格/图）→ docx；并汇总其他交付物状态。

流程：
  1) 从 results/ 聚合真实数字，填充 deliverables/参赛方案.md 的占位符
     （<<GAIN_ALL>> <<S7_GAIN>> <<S7_COST>> <<TABLE_*>> <<ROLLBACK_EVIDENCE>> ...）
  2) 复制图表到 deliverables/figures/
  3) 调用 scripts/md_to_docx.py 生成 deliverables/智训引擎_参赛方案.docx
  4) （可选）调用 scripts/build_pptx.py 生成答辩 PPT
  5) 打印其余交付物（PDF/记录汇编/系统镜像）的构建指令

用法：
  PY=.../python3
  $PY scripts/build_deliverables.py [--pptx] [--zip]
  # docx → PDF 的转换由 DSH office_convert 工具完成（或本机 Word/LibreOffice）
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEAM = "agent说的都队"
SCHOOL = "清华大学"
ADVISOR = "龙明盛"
RESULTS = ROOT / "results"
DELIV = ROOT / "deliverables"
FIGS = DELIV / "figures"


def load(p):
    try:
        with open(p, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def runs(group: str) -> list[dict]:
    d = RESULTS / group
    if not d.exists():
        return []
    out = []
    for p in sorted(d.glob("*.json")):
        r = load(p)
        if r and "meta" in r:
            out.append(r)
    return out


def mean(xs):
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else 0.0


def fmt(x, d=3):
    return f"{x:.{d}f}" if isinstance(x, (int, float)) else str(x)


def build_tables():
    main = runs("main")
    scen = runs("scenarios")
    baselines = load(RESULTS / "baselines" / "summary.json") or {}
    out = {}

    # ---- 主结果表 ----
    rows = ["| 方法 | 流式 macro-F1 | 相对静态增益 | 标注(条) | 算力(units) | 部署/回滚 |",
            "|---|---|---|---|---|---|"]
    if main:
        m = mean([r["hidden"]["stream_macro_f1"] for r in main])
        s = mean([r["static_baseline_hidden"]["stream_macro_f1"] for r in main])
        labels = mean([r["cost"]["labels"] for r in main])
        comp = mean([r["cost"]["compute_units"] for r in main])
        dep = mean([sum(1 for x in r["timeline"] if x.get("deploy")) for r in main])
        rb = mean([sum(1 for x in r["timeline"] if x.get("rollback")) for r in main])
        rows.append(f"| B0 静态基线 | {s:.4f} | — | 0 | 0 | 0/0 |")
        rows.append(f"| **Ours（自进化系统）** | **{m:.4f}** | **{(m - s) * 100:+.2f} pt** | "
                    f"{labels:.0f} | {comp:.1f} | {dep:.1f}/{rb:.1f} |")
    if baselines.get("results", {}).get("S7_full_mix"):
        b7 = baselines["results"]["S7_full_mix"]
        for key, name in [("periodic", "B1 周期重训"), ("oracle", "B2 Oracle 上界")]:
            v = b7.get(key) or {}
            if v:
                rows.append(f"| {name} | {v.get('mean_f1_mean', 0):.4f} | — | "
                            f"{v.get('labels_mean', 0):.0f} | {v.get('compute_units_mean', 0):.0f} | — |")
    out["TABLE_MAIN"] = "\n".join(rows)

    # ---- 分场景表 ----
    per = {}
    for r in scen + main:
        scn = r["meta"]["scenario"]
        per.setdefault(scn, {"m": [], "s": [], "c": []})
        per[scn]["m"].append(r["hidden"]["stream_macro_f1"])
        per[scn]["s"].append(r["static_baseline_hidden"]["stream_macro_f1"])
        per[scn]["c"].append(r["cost"]["scalar"])
    rows = ["| 场景 | 静态基线 F1 | 自进化 F1 | 增益(pt) | 成本(scalar) |", "|---|---|---|---|---|"]
    for scn in sorted(per):
        v = per[scn]
        m, s = mean(v["m"]), mean(v["s"])
        rows.append(f"| {scn} | {s:.4f} | {m:.4f} | {(m - s) * 100:+.2f} | {mean(v['c']):.1f} |")
    out["TABLE_SCENARIOS"] = "\n".join(rows)

    # ---- 消融表 ----
    rows = ["| 消融 | 流式 F1 | 增益(pt) | 成本(scalar) | 部署 | 回滚 |", "|---|---|---|---|---|---|"]
    for d in sorted(RESULTS.glob("ab-*")):
        rs = runs(d.name)
        if not rs:
            continue
        name = d.name.replace("ab-", "")
        m = mean([r["hidden"]["stream_macro_f1"] for r in rs])
        g = mean([r["comparison"]["stream_gain_points"] for r in rs])
        c = mean([r["cost"]["scalar"] for r in rs])
        dep = mean([sum(1 for x in r["timeline"] if x.get("deploy")) for r in rs])
        rb = mean([sum(1 for x in r["timeline"] if x.get("rollback")) for r in rs])
        rows.append(f"| {name} | {m:.4f} | {g:+.2f} | {c:.1f} | {dep:.1f} | {rb:.1f} |")
    out["TABLE_ABLATION"] = "\n".join(rows)

    # ---- 预算扫描表 ----
    rows = ["| 标注预算 | 场景 | 增益(pt) | 流式 F1 | 成本 |", "|---|---|---|---|---|"]
    for d in sorted(RESULTS.glob("budget-*"), key=lambda p: int(p.name.split("-")[1])):
        rs = runs(d.name)
        if not rs:
            continue
        budget = d.name.split("-")[1]
        for scn in sorted({r["meta"]["scenario"] for r in rs}):
            sub = [r for r in rs if r["meta"]["scenario"] == scn]
            rows.append(f"| {budget} | {scn} | "
                        f"{mean([r['comparison']['stream_gain_points'] for r in sub]):+.2f} | "
                        f"{mean([r['hidden']['stream_macro_f1'] for r in sub]):.4f} | "
                        f"{mean([r['cost']['scalar'] for r in sub]):.1f} |")
    out["TABLE_BUDGET"] = "\n".join(rows)

    # ---- Pareto 表 ----
    rows = ["| 方法（S7） | 标注 | 算力(units) | 流式 F1 |", "|---|---|---|---|"]
    if baselines.get("results", {}).get("S7_full_mix"):
        b7 = baselines["results"]["S7_full_mix"]
        rows.append(f"| B0 静态基线 | 0 | 0 | {b7.get('static', {}).get('mean_f1_mean', 0):.4f} |")
        for key, name in [("periodic", "B1 周期重训"), ("oracle", "B2 Oracle 上界")]:
            v = b7.get(key) or {}
            if v:
                rows.append(f"| {name} | {v.get('labels_mean', 0):.0f} | "
                            f"{v.get('compute_units_mean', 0):.0f} | {v.get('mean_f1_mean', 0):.4f} |")
    if main:
        rows.append(f"| **Ours** | {mean([r['cost']['labels'] for r in main]):.0f} | "
                    f"{mean([r['cost']['compute_units'] for r in main]):.1f} | "
                    f"{mean([r['hidden']['stream_macro_f1'] for r in main]):.4f} |")
    out["TABLE_PARETO"] = "\n".join(rows)

    # ---- 回滚/安全证据 ----
    n_rollback = sum(sum(1 for x in r["timeline"] if x.get("rollback")) for r in main)
    n_deploy = sum(sum(1 for x in r["timeline"] if x.get("deploy")) for r in main)
    audit_ok = all(r["audit"]["verified"] for r in main) if main else False
    lines = [
        f"- 主实验 {len(main)} 次 run 共 **{n_deploy} 次上线、{n_rollback} 次回滚**；"
        f"审计链校验 **{'全部通过' if audit_ok else '存在失败'}**。",
        "- 看门狗使用**无标签/探针信号**（绝不使用隐藏测试分数）判断是否回滚；"
        "维修复位（S5）与有害更新注入测试均触发自动回退。",
        "- 负向测试覆盖：越权动作、超预算、审计篡改、kill-switch、LLM 解析失败降级。",
    ]
    for r in main:
        for row in r["timeline"]:
            if row.get("rollback"):
                lines.append(f"- `{r['meta']['run_id']}` 在 t={row['t']}（{row['phase_name']}）"
                             f"触发回滚：{row.get('watchdog', {}).get('reason', '')}")
    out["ROLLBACK_EVIDENCE"] = "\n".join(lines)
    return out


def fill_report() -> Path:
    """用 fill_docs 的单一映射源回填参赛方案（避免两套占位符映射不一致）。"""
    sys.path.insert(0, str(ROOT / "scripts"))
    from fill_docs import build_mapping, summarize

    src = DELIV / "参赛方案.md"
    text = src.read_text(encoding="utf-8")
    d = summarize()
    mapping = build_mapping(d)
    # 额外兼容（PPT 的 OURS_* 占位符不再出现在方案里，但保留兜底）
    mapping.setdefault("<<OURS_LABELS>>", f"{d['s7_labels']:.0f}")
    mapping.setdefault("<<OURS_COMPUTE>>", f"{d['s7_compute']:.0f}")
    mapping.setdefault("<<OURS_F1>>", f"{d['s7_method']:.3f}")
    for k, v in mapping.items():
        text = text.replace(k, str(v))
    # 图路径：复制到 deliverables/figures
    FIGS.mkdir(parents=True, exist_ok=True)
    for p in sorted((RESULTS / "figures").glob("*.png")):
        shutil.copy2(p, FIGS / p.name)
    out = DELIV / "参赛方案_filled.md"
    out.write_text(text, encoding="utf-8")
    return out


def build_docx(filled: Path) -> Path:
    sys.path.insert(0, str(ROOT / "scripts"))
    from md_to_docx import build

    out = DELIV / "智训引擎_参赛方案.docx"
    build(filled, out,
          cover_title="智训引擎：工业模型自进化系统",
          cover_subtitle="第二届北京市大学生“人工智能+”创新大赛 · “智训引擎”赛题参赛方案",
          cover_meta=f"参赛团队：{TEAM} | 学校：{SCHOOL} | 指导教师：{ADVISOR} | 日期：2026-11",
          with_toc=True)
    return out


def make_zip() -> Path:
    import zipfile

    out = DELIV / "zhixun_engine_system.zip"
    include = ["src", "configs", "scripts", "skills", "docs", "tests", "notes/lit",
               "refs/官方速览视频_关键帧", "notes/视频分析_官方5分钟速览_2026-11-16.md",
               "notes/J1-J3_在项目中的角色.md",
               "README.md", "requirements.txt"]
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        for item in include:
            p = ROOT / item
            if p.is_dir():
                for f in p.rglob("*"):
                    if f.is_file() and "__pycache__" not in str(f):
                        z.write(f, f.relative_to(ROOT))
            elif p.exists():
                z.write(p, p.relative_to(ROOT))
        for f in (RESULTS / "SUMMARY.md", RESULTS / "SUMMARY.json", RESULTS / "acceptance.json",
                  RESULTS / "tables" / "official_rubric.md", RESULTS / "tables" / "official_rubric.json",
                  RESULTS / "tables" / "gate_operating_curve.md", RESULTS / "tables" / "gate_operating_curve.json",
                  RESULTS / "tables" / "invariant_check.md", RESULTS / "tables" / "invariant_check.json",
                  RESULTS / "tables" / "negative_inband_test.md", RESULTS / "tables" / "negative_inband_test.json",
                  RESULTS / "dirty" / "summary.md", RESULTS / "dirty" / "stress_summary.json",
                  ROOT / "deliverables" / "figures" / "fig9_gate_curve.png"):
            if f.exists():
                z.write(f, f.relative_to(ROOT))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pptx", action="store_true")
    ap.add_argument("--zip", action="store_true")
    args = ap.parse_args()
    filled = fill_report()
    print(f"[1/4] 报告已填充：{filled.relative_to(ROOT)}")
    docx = build_docx(filled)
    print(f"[2/4] docx：{docx.relative_to(ROOT)}（下一步用 office_convert 转 PDF）")
    # 系统运行说明（docx，人工转 PDF）
    try:
        sys.path.insert(0, str(ROOT / "scripts"))
        from md_to_docx import build as build_docx_md
        runbook_md = ROOT / "docs" / "12_系统运行说明.md"
        if runbook_md.exists():
            out_rb = build_docx_md(runbook_md, DELIV / "智训引擎_系统运行说明.docx",
                                   cover_title="智训引擎 · 系统运行说明",
                                   cover_subtitle="工业模型自进化系统 · 环境/命令/配置/FAQ",
                                   cover_meta=f"参赛团队：{TEAM} | 学校：{SCHOOL} | 指导教师：{ADVISOR} | 2026-11",
                                   with_toc=False)
            print(f"[3/5] 运行说明：{out_rb.relative_to(ROOT)}（下一步 office_convert 转 PDF）")
    except Exception as e:
        print(f"[3/5] 运行说明生成失败：{e}")

    if args.pptx:
        sys.path.insert(0, str(ROOT / "scripts"))
        from build_pptx import build as build_ppt
        ppt = build_ppt(DELIV / "智训引擎_答辩PPT.pptx", team=TEAM, school=SCHOOL, advisor=ADVISOR)
        print(f"[3/4] PPT：{ppt.relative_to(ROOT)}")
    if args.zip:
        z = make_zip()
        print(f"[4/4] 系统镜像：{z.relative_to(ROOT)} ({z.stat().st_size / 1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
