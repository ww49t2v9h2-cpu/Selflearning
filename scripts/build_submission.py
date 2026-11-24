"""生成比赛提交包（按赛事要求的格式）。

产出目录：deliverables/提交包_2026-11-20/
  01_模型配置.txt
  02_Harness系统镜像.tar
  03_SwanLab实验记录_查询依据.pdf（由本脚本生成 .md → 另行转 PDF）
  03_SwanLab实验记录_完整查询依据.txt
  04_系统运行说明.pdf / .txt
  05_项目说明书.pdf
  06_答辩PPT.pptx
  07_项目源代码.zip
  附_实验记录汇编.xlsx
  README_提交说明.txt / SHA256SUMS.txt

本脚本只生成 txt/tar/zip 与 pdf 的 markdown 源；PDF 由 DSH office_convert 生成，
已有 PDF（运行说明/项目说明书）直接复制。
"""

from __future__ import annotations

import hashlib
import json
import shutil
import sqlite3
import sys
import tarfile
import zipfile
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "deliverables" / "提交包_2026-11-20"
BUILD = OUT / "_build"
DB = ROOT / "experiments" / "final_records" / "tracker.sqlite"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def human(n: int) -> str:
    for unit in ("B", "KB", "MB", "GB"):
        if n < 1024 or unit == "GB":
            return f"{n:.1f}{unit}" if unit != "B" else f"{n}{unit}"
        n /= 1024
    return f"{n:.1f}GB"


# ---------------------------------------------------------------------------
# 01 模型配置
# ---------------------------------------------------------------------------
def build_model_config() -> Path:
    cfg_path = ROOT / "configs" / "main.json"
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    eng_defaults = ROOT / "src" / "engine" / "defaults.py"
    model = {"hidden": [64, 32], "activation": "relu", "lr": 0.004, "epochs": 40,
             "batch": 128, "l2": 1e-4, "seed": 0}
    try:
        sys.path.insert(0, str(ROOT))
        from src.engine.defaults import ENGINE_DEFAULTS  # type: ignore
        model = dict(ENGINE_DEFAULTS.get("model", model))
    except Exception:
        pass

    lines = [
        "=" * 78,
        "智训引擎 · 模型配置（提交格式：txt）",
        "第二届北京市大学生“人工智能+”创新大赛 ·「智训引擎」赛题",
        "命题企业：软通计算机有限公司 × 华为技术有限公司",
        f"生成时间：{datetime.now(timezone.utc).astimezone().strftime('%Y-%m-%d %H:%M:%S %Z')}",
        "说明：本文件包含模型结构/训练超参、预算与成本模型、监测/诊断/门禁/规划器/安全阈值，",
        "      以及 configs/main.json、configs/official_template.json、configs/llm.json 全文。",
        "=" * 78,
        "",
        "一、模型结构（纯 numpy MLP，CPU 可跑）",
        "-" * 78,
        f"输入维度            : 48（8 个传感器 × 6 个统计量；由 benchmark.d_features 决定）",
        f"可见类别数          : 4（隐藏测试含新类时为 5；见 benchmark.n_classes）",
        f"隐藏层              : {model.get('hidden')}",
        f"激活函数            : {model.get('activation')}",
        f"输出                : softmax（4/5 类），Adam 优化",
        f"学习率              : {model.get('lr')}",
        f"训练轮数            : {model.get('epochs')}（基础模型）",
        f"batch size          : {model.get('batch')}",
        f"L2 正则             : {model.get('l2')}",
        f"随机种子            : {model.get('seed')}（run seed 见 configs/main.json / run_all.py）",
        "",
        "二、训练与预算（configs/main.json）",
        "-" * 78,
        json.dumps(cfg.get("budget", {}), ensure_ascii=False, indent=2),
        "",
        "三、数据与基准",
        "-" * 78,
        json.dumps(cfg.get("benchmark", {}), ensure_ascii=False, indent=2),
        "数据适配：合成漂移基准（S1–S7）+ 公开工业数据 AI4I 2020（UCI，CC BY 4.0）；",
        "官方数据包发放后通过 DatasetAdapter 接入（configs/official_template.json）。",
        "",
        "四、Harness / LLM 配置",
        "-" * 78,
        json.dumps(cfg.get("harness", {}), ensure_ascii=False, indent=2),
        "默认 deterministic（离线、0 token、可复现）；企业赛可切 openai_compat 指向 PyTRi0。",
        "",
        "五、监测 / 诊断 / 门禁 / 规划器阈值（引擎默认值摘要）",
        "-" * 78,
    ]
    summary_keys = [
        ("monitor.window", "monitor", "window"), ("monitor.mmd_threshold", "monitor", "mmd_threshold"),
        ("monitor.ks_alpha", "monitor", "ks_alpha"), ("monitor.ks_topk", "monitor", "ks_topk"),
        ("monitor.probe_every", "monitor", "probe_every"),
        ("engine.monitor.ph_threshold", "engine", "ph_threshold"),
        ("engine.monitor.novelty_threshold", "engine", "novelty_threshold"),
        ("engine.diagnosis.affine_min_gain", "engine", "affine_min_gain"),
        ("gates.min_gain_points", "gates", "min_gain_points"),
        ("gates.max_recall_drop", "gates", "max_recall_drop"),
        ("gates.calibration_slack", "gates", "calibration_slack"),
        ("gates.max_cost_per_update", "gates", "max_cost_per_update"),
        ("planner.ucb_c", "planner", "ucb_c"),
        ("planner.exploration", "planner", "exploration"),
        ("planner.cheapest_first", "planner", "cheapest_first"),
        ("watchdog.rollback_drop_points", "watchdog", "rollback_drop_points"),
        ("watchdog.rollback_window", "watchdog", "rollback_window"),
        ("safety.auto_approve_high_risk", "safety", "auto_approve_high_risk"),
        ("safety.human_gate", "safety", "human_gate"),
    ]
    try:
        from src.engine.defaults import ENGINE_DEFAULTS  # type: ignore
        for label, sect, key in summary_keys:
            v = (ENGINE_DEFAULTS.get(sect) or {}).get(key)
            if v is not None:
                lines.append(f"{label:42s}: {v}")
    except Exception:
        pass
    lines += [
        "",
        "六、九个 playbook 的成本/标注配置（完整值见 configs/main.json 与 src/engine/defaults.py）",
        "-" * 78,
        "NOOP                 : 0 标注 / 0 算力（合法动作，优先）",
        "RECALIBRATE_AFFINE   : 0 标注 / ~0 算力（零标签 TTA，仿射校准）",
        "PRIOR_CORRECTION     : 0 标注 / ~0 算力（标签先验纠正，EM）",
        "LAST_LAYER_FT        : 72 标注（上限 110）/ 低算力（末层微调）",
        "FULL_FINETUNE_REPLAY : 110 标注（上限 150）/ 中算力（回放微调）",
        "RETRAIN_MIXED        : 120 标注（上限 150）/ 中高算力（混合重训）",
        "NOVEL_CLASS_EXTEND   : 140 标注（上限 150）/ 中高算力（新类扩展，高风险）",
        "SHADOW_ONLY          : 0 标注 / 低算力（影子评估，不改变 champion）",
        "ROLLBACK             : 0 标注 / 0 算力（回退到上一稳定版本）",
        "",
        "七、配置全文（JSON）",
        "-" * 78,
        "### configs/main.json（主实验，sanitize=false 以匹配冻结结果）",
        cfg_path.read_text(encoding="utf-8"),
        "",
        "### configs/official_template.json（官方/决赛数据包模板，sanitize=true）",
        (ROOT / "configs" / "official_template.json").read_text(encoding="utf-8"),
        "",
        "### configs/llm.json（LLM 驱动配置示例，mock provider）",
        (ROOT / "configs" / "llm.json").read_text(encoding="utf-8"),
        "",
        "### engine defaults（src/engine/defaults.py 的 ENGINE_DEFAULTS，节选）",
    ]
    try:
        from src.engine.defaults import ENGINE_DEFAULTS  # type: ignore
        lines.append(json.dumps(ENGINE_DEFAULTS, ensure_ascii=False, indent=1))
    except Exception as e:
        lines.append(f"（无法导入默认配置：{e}）")
    lines += ["", "=" * 78, "文件结束", ""]
    p = OUT / "01_模型配置.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# 02 Harness 系统镜像 tar + 07 源代码 zip
# ---------------------------------------------------------------------------
SYSTEM_INCLUDE = ["src", "configs", "scripts", "skills", "tests", "docs",
                  "requirements.txt", "README.md"]
DATA_INCLUDE = ["data/external/ai4i2020.zip"]


def build_tar() -> Path:
    p = OUT / "02_Harness系统镜像.tar"
    manifest = []
    note = [
        "智训引擎 · Harness 系统镜像（tar）",
        "=====================================",
        "内容：src（系统源码）、configs（配置）、scripts（一键复现脚本）、skills（7 个技能）、",
        "      tests（161 项测试）、docs（15 份文档）、requirements.txt、README.md、",
        "      data/external/ai4i2020.zip（公开 AI4I 数据，离线可跑真实数据组）。",
        "运行环境：Python >= 3.10（本机 3.12），numpy；可选 matplotlib/openpyxl/swanlab。",
        "快速开始：",
        "  tar -xf 02_Harness系统镜像.tar && cd harness_system",
        "  python -m unittest discover -s tests",
        "  python scripts/smoke.py",
        "  python -m src.cli run --config configs/main.json",
        "  # 一键完整复现（约 15–20 分钟）：bash scripts/build_all.sh",
        "说明：镜像不含 experiments/ 与 results/（运行产物），也不含官方隐藏测试集。",
        "论文/答辩材料见另附的 05_项目说明书.pdf 与 06_答辩PPT.pptx。",
        "",
    ]
    with tarfile.open(p, "w") as tf:
        note_b = "\n".join(note).encode("utf-8")
        ti = tarfile.TarInfo("harness_system/系统镜像说明.txt")
        ti.size = len(note_b)
        ti.mtime = int(datetime.now().timestamp())
        tf.addfile(ti, BytesIO(note_b))

        def add_path(rel: str):
            src = ROOT / rel
            if not src.exists():
                return
            arc = f"harness_system/{rel}"
            if src.is_dir():
                for f in sorted(src.rglob("*")):
                    if not f.is_file() or "__pycache__" in str(f):
                        continue
                    if f.suffix == ".pyc":
                        continue
                    tf.add(str(f), arcname=f"harness_system/{f.relative_to(ROOT)}")
                    manifest.append((f.relative_to(ROOT).as_posix(), sha256(f)))
            else:
                tf.add(str(src), arcname=arc)
                manifest.append((rel, sha256(src)))

        for rel in SYSTEM_INCLUDE + DATA_INCLUDE:
            add_path(rel)
        man_lines = [f"{h}  {rel}" for rel, h in sorted(manifest)]
        man_b = ("\n".join(man_lines) + "\n").encode("utf-8")
        ti = tarfile.TarInfo("harness_system/MANIFEST.sha256")
        ti.size = len(man_b)
        ti.mtime = int(datetime.now().timestamp())
        tf.addfile(ti, BytesIO(man_b))
    return p


def build_source_zip() -> Path:
    p = OUT / "07_项目源代码.zip"
    include = ["src", "configs", "scripts", "skills", "tests", "requirements.txt", "README.md",
               "Dockerfile", ".dockerignore"]
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as z:
        for rel in include:
            src = ROOT / rel
            if src.is_dir():
                for f in sorted(src.rglob("*")):
                    if f.is_file() and "__pycache__" not in str(f) and f.suffix != ".pyc":
                        z.write(f, f.relative_to(ROOT))
            elif src.exists():
                z.write(src, rel)
    return p


# ---------------------------------------------------------------------------
# 03 SwanLab 实验记录导出（完整 TXT + 查询依据 md）
# ---------------------------------------------------------------------------
RUN_COLS = ["run_id", "group_name", "name", "kind", "stage", "status",
            "created_at", "swanlab_mode", "swanlab_run_id", "swanlab_url", "config_json", "summary_json"]


def _load_db():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row
    runs = [dict(r) for r in conn.execute("select * from runs order by created_at, run_id")]
    metrics = [dict(r) for r in conn.execute("select run_id,key,step,value from metrics order by run_id,key,step")]
    conn.close()
    return runs, metrics


def build_swanlab_export() -> tuple[Path, Path]:
    runs, metrics = _load_db()
    now = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    cloud_map: dict[str, dict] = {}
    links_path = ROOT / "results" / "swanlab_links.jsonl"
    if links_path.exists():
        for line in links_path.read_text(encoding="utf-8").splitlines():
            try:
                d = json.loads(line)
                if d.get("url"):
                    cloud_map[d["run_id"]] = d
            except Exception:
                pass
    project_url = "https://swanlab.cn/@Migratory_Bird/zhixun-engine" if cloud_map else ""

    groups, scen = {}, {}
    for r in runs:
        groups[r["group_name"]] = groups.get(r["group_name"], 0) + 1
        tags = json.loads(r["tags_json"] or "[]")
        key = tags[0] if tags else "unknown"
        scen[key] = scen.get(key, 0) + 1

    def rs(r, dot, default=None):
        try:
            cur = json.loads(r["summary_json"] or "{}")
            for k in dot.split("."):
                cur = cur[k]
            return cur
        except Exception:
            return default

    # ---------------- TXT：完整逐条导出 ----------------
    lines = [
        "# 智训引擎 · SwanLab 实验记录完整查询依据（txt）",
        "# 生成时间：" + now,
        "# 项目：zhixun-engine（SwanLab 兼容本地镜像 + 云端回放）",
        "# 数据源：experiments/final_records/tracker.sqlite（SQLite 镜像）",
        f"# 规模：{len(runs)} runs / {len(metrics)} metric points",
        f"# sha256(tracker.sqlite)={sha256(DB) if DB.exists() else 'n/a'}",
        f"# 云端项目：{project_url or '（未上传）'}",
        f"# 已上云 run 数：{len(cloud_map)} / {len(runs)}",
        "",
        "# 组分布：",
    ]
    for g, n in sorted(groups.items()):
        lines.append(f"#   {g}: {n}")
    lines += ["", "# 场景/标签分布："]
    for sk, n in sorted(scen.items()):
        lines.append(f"#   {sk}: {n}")
    lines += [
        "",
        "# 本地查询：",
        "#   $PY scripts/query_records.py stats",
        "#   $PY scripts/query_records.py list --group main --kind deploy",
        "#   $PY scripts/query_records.py compare <run_id_a> <run_id_b>",
        "#   $PY scripts/query_records.py export --fmt csv --out results/records_all.csv",
        "# 云端续传：/tmp/swanenv/bin/python scripts/swanlab_upload.py --mode online",
        "",
        "# ===== RUNS（\t 分隔） =====",
        "# 字段：run_id\tgroup\tname\tkind\tstage\tstatus\tcreated_iso\tmode\tswanlab_run_id\tswanlab_url"
        "\tstream_macro_f1\tstream_gain_pt\tlabels\tcompute_units\ttokens\tcost_scalar\tn_windows\taudit_verified\ttags",
    ]
    for r in runs:
        tags = json.loads(r["tags_json"] or "[]")
        created = (datetime.fromtimestamp(float(r["created_at"] or 0)).astimezone().isoformat(timespec="seconds")
                   if r["created_at"] else "")
        link = cloud_map.get(r["run_id"]) or {}
        vals = [
            r["run_id"], r["group_name"], r["name"], r["kind"], r["stage"], r["status"], created,
            r["swanlab_mode"] or "", link.get("swanlab_run_id") or r["swanlab_run_id"] or "",
            link.get("url") or r["swanlab_url"] or "",
            str(rs(r, "hidden.stream_macro_f1", "")), str(rs(r, "comparison.stream_gain_points", "")),
            str(rs(r, "cost.labels", "")), str(rs(r, "cost.compute_units", "")),
            str(rs(r, "cost.tokens", "")), str(rs(r, "cost.scalar", "")),
            str(rs(r, "hidden.n_chunks", "")), str(rs(r, "audit_verified", "")),
            ",".join(tags),
        ]
        lines.append("\t".join(str(v) for v in vals))
    lines += ["", "# ===== METRICS（run_id\tkey\tstep\tvalue） ====="]
    for m in metrics:
        lines.append(f"{m['run_id']}\t{m['key']}\t{m['step']}\t{m['value']}")
    lines += ["", "# 文件结束", ""]
    txt = OUT / "03_SwanLab实验记录_完整查询依据.txt"
    txt.write_text("\n".join(lines), encoding="utf-8")

    # ---------------- PDF 的 markdown：查询依据 ----------------
    md = [
        "# SwanLab 实验记录查询依据（校赛提交版）",
        "",
        "> 赛题要求：提供「云端实验记录链接或查询依据」（PDF/TXT，≤50MB）。",
        "> 本项目以 SwanLab 兼容的本地记录（SQLite 镜像 + JSONL）完整留痕，并已用账号",
        "> `Migratory_Bird` 将记录回放到 SwanLab 云端（公开项目）；逐条链接见 §1.5。",
        "",
        "## 1. 记录规模与来源",
        "",
        "- 记录文件：`experiments/final_records/tracker.sqlite`（SwanLab 兼容镜像；另有 JSONL 原生日志）",
        f"- 规模：**{len(runs)} 条 run、{len(metrics)} 个指标点**",
        f"- 生成时间：{now}",
        f"- tracker.sqlite SHA256：`{sha256(DB) if DB.exists() else 'n/a'}`",
        "",
        "| 组 | run 数 |",
        "|---|---|",
    ]
    for g, n in sorted(groups.items()):
        md.append(f"| {g} | {n} |")
    md += ["", "| 场景/标签 | run 数 |", "|---|---|"]
    for sk, n in sorted(scen.items()):
        md.append(f"| {sk} | {n} |")
    md += [
        "",
        "## 1.5 云端链接（SwanLab）",
        "",
        f"- **项目主页**：{project_url or '（未上传）'}",
        f"- **已同步 run**：**{len(cloud_map)} / {len(runs)}**（逐条链接见 TXT 的 swanlab_url 列）",
        "",
        "| 示例 run | 云端链接 |",
        "|---|---|",
    ]
    shown = 0
    for r in runs:
        c = cloud_map.get(r["run_id"])
        if c and shown < 10:
            md.append(f"| {r['run_id']} | {c['url']} |")
            shown += 1
    md += [
        "",
        "> 上传方式：`scripts/swanlab_upload.py --mode online` 将本地 SQLite 记录逐条回放为",
        "> SwanLab run（summary + 逐 step 指标曲线），断点续传；本地记录与云端一一对应。",
        "",
        "## 2. 记录字段（每条 run）",
        "",
        "run_id、group、name、kind、stage、status、created_at、swanlab_mode、swanlab_run_id、swanlab_url、",
        "stream_macro_f1、stream_gain_pt、labels、compute_units、tokens、cost_scalar、n_windows、audit_verified、tags。",
        "指标明细（metric key / step / value）逐条见 TXT。",
        "",
        "## 3. 本地查询命令（可复跑）",
        "",
        "```bash",
        "PY=/Users/mb/.dsh/dsh-runtimes/dsh-primary-runtime/dependencies/python/bin/python3",
        "$PY scripts/query_records.py stats",
        "$PY scripts/query_records.py list --group main --kind deploy",
        "$PY scripts/query_records.py compare <run_id_a> <run_id_b>",
        "$PY scripts/query_records.py export --fmt csv --out results/records_all.csv",
        "```",
        "",
        "## 4. 完整性校验",
        "",
        "| 文件 | 大小 | SHA256 |",
        "|---|---|---|",
    ]
    for f in [txt, OUT / "附_实验记录汇编.xlsx"]:
        if f.exists():
            md.append(f"| {f.name} | {human(f.stat().st_size)} | `{sha256(f)}` |")
    md += ["", "> 本 PDF 与同目录 TXT 共同构成赛题「云端实验记录链接或查询依据」提交物。", ""]
    md_path = BUILD / "SwanLab_实验记录查询依据.md"
    md_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.write_text("\n".join(md), encoding="utf-8")
    return txt, md_path


# ---------------------------------------------------------------------------
# 04 运行说明 txt
# ---------------------------------------------------------------------------
def build_runbook_txt() -> Path:
    import re
    md = (ROOT / "docs" / "12_系统运行说明.md").read_text(encoding="utf-8")
    out = []
    for line in md.splitlines():
        if line.startswith("```"):
            out.append("")  # 代码块边界用空行
            continue
        line = re.sub(r"^#{1,6}\s*", "", line)
        line = re.sub(r"\*\*(.*?)\*\*", r"\1", line)
        line = re.sub(r"`([^`]*)`", r"\1", line)
        out.append(line)
    p = OUT / "04_系统运行说明.txt"
    p.write_text("\n".join(out) + "\n", encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# README / checksums
# ---------------------------------------------------------------------------
def build_readme() -> Path:
    files = sorted([f for f in OUT.iterdir() if f.is_file()])
    lines = [
        "=" * 78,
        "智训引擎 · 校赛提交包（2026-11-20）",
        "参赛团队：agent说的都队　学校：清华大学　指导教师：龙明盛",
        "第二届北京市大学生“人工智能+”创新大赛 ·「智训引擎——工业制造场景下的模型训练与超参数优化」",
        "命题企业：软通计算机有限公司 × 华为技术有限公司",
        "=" * 78,
        "文件清单（格式与赛题要求的对应关系）：",
        "",
        "  [模型配置]     01_模型配置.txt",
        "  [系统镜像]     02_Harness系统镜像.tar（linux/amd64，经典 docker-archive：manifest.json + config.json + layer_*.tar；docker load -i 可直接载入）",
        "  [源码镜像]     02b_Harness系统源码.tar（源码+文档+测试的 tar，便于无 Docker 环境查看）",
        "  [实验记录]     03_SwanLab实验记录_查询依据.pdf + 03_SwanLab实验记录_完整查询依据.txt（≤50MB；另附 xlsx）",
        "  [运行说明]     04_系统运行说明.pdf + 04_系统运行说明.txt（PDF ≤10MB）",
        "  [项目说明书]   05_项目说明书.pdf",
        "  [答辩PPT]      06_答辩PPT.pptx",
        "  [项目源代码]   07_项目源代码.zip",
        "  [校验]         SHA256SUMS.txt",
        "",
        "说明：",
        "1) 实验记录以 SwanLab 兼容的本地记录（SQLite 镜像 + JSONL）完整留痕，并已回放到 SwanLab 云端：",
        "   https://swanlab.cn/@Migratory_Bird/zhixun-engine （公开项目；逐条 run 链接见 03 的 §1.5 与 TXT）。",
        "   企业赛可继续用 `scripts/swanlab_upload.py --mode online` 增量续传。",
        "2) 所有数字可由 02 镜像内 `bash scripts/build_all.sh` 一键复现；未包含官方隐藏测试集。",
        "3) 项目说明书为 18 页参赛方案；答辩 PPT 为 17 页（含讲稿备注）。",
        "4) 镜像用法（经典布局）：",
        "   docker load -i 02_Harness系统镜像.tar",
        "   docker run --rm zhixun-engine:1.0-linux-amd64 info",
        "   docker run --rm zhixun-engine:1.0-linux-amd64 run --config configs/quick.json",
        "   镜像为 linux/amd64、单镜像、未压缩 tar；入口 python -m src.cli；/app/experiments、/app/results 可写。",
        "5) 文件大小：",
        "",
    ]
    for f in files:
        lines.append(f"   {f.name:48s} {human(f.stat().st_size)}")
    lines += ["", "SHA256 校验：`shasum -a 256 -c SHA256SUMS.txt`（macOS）或 `sha256sum -c SHA256SUMS.txt`。", ""]
    p = OUT / "README_提交说明.txt"
    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def build_checksums() -> Path:
    p = OUT / "SHA256SUMS.txt"
    lines = []
    for f in sorted(OUT.iterdir()):
        if f.is_file() and f.name != "SHA256SUMS.txt":
            lines.append(f"{sha256(f)}  {f.name}")
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> int:
    if OUT.exists():
        shutil.rmtree(OUT)
    OUT.mkdir(parents=True)
    BUILD.mkdir()
    print("生成 01 模型配置 ...")
    build_model_config()
    print("生成 02 Harness 系统镜像 tar ...")
    build_tar()
    print("生成 03 SwanLab 记录导出 ...")
    build_swanlab_export()
    print("生成 04 运行说明 txt ...")
    build_runbook_txt()
    print("复制 04/05/06 与附件 ...")
    shutil.copy2(ROOT / "deliverables" / "智训引擎_系统运行说明.pdf", OUT / "04_系统运行说明.pdf")
    shutil.copy2(ROOT / "deliverables" / "智训引擎_参赛方案.pdf", OUT / "05_项目说明书.pdf")
    shutil.copy2(ROOT / "deliverables" / "智训引擎_答辩PPT.pptx", OUT / "06_答辩PPT.pptx")
    shutil.copy2(ROOT / "deliverables" / "实验记录汇编.xlsx", OUT / "附_实验记录汇编.xlsx")
    print("生成 07 项目源代码 zip ...")
    build_source_zip()
    build_readme()
    print("生成 SHA256SUMS ...")
    build_checksums()
    print("完成，文件清单：")
    total = 0
    for f in sorted(OUT.iterdir()):
        if f.is_file():
            total += f.stat().st_size
            print(f"  {f.name:46s} {human(f.stat().st_size)}")
    print(f"  合计：{human(total)}")
    print(f"  PDF 源（待转）：{BUILD/'SwanLab_实验记录查询依据.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
