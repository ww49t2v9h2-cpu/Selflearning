"""生成答辩 PPT（16:9，主题化，数据驱动，带讲稿备注）。

依赖：python-pptx（bundled Python 自带）、读取 results/SUMMARY.json 与 results/figures/*.png。
运行：
  PY=.../python3
  $PY scripts/build_pptx.py --out deliverables/智训引擎_答辩PPT.pptx
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.util import Emu, Inches, Pt
from pptx.oxml.ns import qn

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
FIGS = RESULTS / "figures"
OUT = ROOT / "deliverables"

# ---------------- 主题 ----------------
PRIMARY = RGBColor(0x1F, 0x4E, 0x79)
ACCENT = RGBColor(0x2E, 0x74, 0xB5)
ACCENT_LIGHT = RGBColor(0xDE, 0xEB, 0xF7)
GREEN = RGBColor(0x2D, 0xA4, 0x4E)
ORANGE = RGBColor(0xD2, 0x99, 0x22)
RED = RGBColor(0xCF, 0x22, 0x2E)
DARK = RGBColor(0x24, 0x29, 0x2F)
GRAY = RGBColor(0x57, 0x60, 0x6A)
LIGHT_BG = RGBColor(0xF4, 0xF7, 0xFA)
WHITE = RGBColor(0xFF, 0xFF, 0xFF)

CJK = "Microsoft YaHei"
LATIN = "Arial"

W, H = Inches(13.333), Inches(7.5)


def set_font(run, size=16, bold=False, color=DARK, name=LATIN, cjk=CJK):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = color
    run.font.name = name
    r = run._r
    rPr = r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", cjk)


def add_text(slide, text, left, top, width, height, size=16, bold=False,
             color=DARK, align=PP_ALIGN.LEFT, anchor=MSO_ANCHOR.TOP, wrap=True):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = wrap
    tf.vertical_anchor = anchor
    tf.margin_left = tf.margin_right = Emu(0)
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        set_font(run, size=size, bold=bold, color=color)
    return tb


def add_bullets(slide, items, left, top, width, height, size=17, color=DARK,
                line_spacing=1.25, bullet="▸ "):
    tb = slide.shapes.add_textbox(left, top, width, height)
    tf = tb.text_frame
    tf.word_wrap = True
    tf.margin_left = tf.margin_right = Emu(0)
    for i, item in enumerate(items):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.space_after = Pt(int(10 * line_spacing))
        if isinstance(item, tuple):
            head, body = item
            r1 = p.add_run()
            r1.text = bullet + head
            set_font(r1, size=size, bold=True, color=ACCENT)
            if body:
                r2 = p.add_run()
                r2.text = body
                set_font(r2, size=size, color=color)
        else:
            r = p.add_run()
            r.text = bullet + item
            set_font(r, size=size, color=color)
    return tb


def add_rect(slide, left, top, width, height, fill=LIGHT_BG, line=None,
             shape=MSO_SHAPE.ROUNDED_RECTANGLE, radius=0.06):
    shp = slide.shapes.add_shape(shape, left, top, width, height)
    shp.fill.solid()
    shp.fill.fore_color.rgb = fill
    if line is None:
        shp.line.fill.background()
    else:
        shp.line.color.rgb = line
        shp.line.width = Pt(1.0)
    if shape == MSO_SHAPE.ROUNDED_RECTANGLE:
        try:
            shp.adjustments[0] = radius
        except Exception:
            pass
    shp.shadow.inherit = False
    return shp


def add_kicker(slide, text, color=ACCENT):
    add_rect(slide, Inches(0.55), Inches(0.52), Inches(0.09), Inches(0.42),
             fill=color, shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, text, Inches(0.78), Inches(0.5), Inches(10.5), Inches(0.45),
             size=13, bold=True, color=color)


def new_slide(prs, title, kicker=None):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    # 顶部装饰
    add_rect(slide, 0, 0, W, Inches(0.09), fill=PRIMARY, shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, title, Inches(0.55), Inches(1.02), Inches(12.2), Inches(0.75),
             size=26, bold=True, color=PRIMARY)
    if kicker:
        add_text(slide, kicker, Inches(0.57), Inches(1.86), Inches(12.2), Inches(0.45),
                 size=13, color=GRAY)
    # 页脚
    add_text(slide, "智训引擎 · 工业模型自进化系统", Inches(0.55), Inches(7.12),
             Inches(6), Inches(0.3), size=9, color=GRAY)
    add_text(slide, "第二届北京市大学生“人工智能+”创新大赛", Inches(7.3), Inches(7.12),
             Inches(5.5), Inches(0.3), size=9, color=GRAY, align=PP_ALIGN.RIGHT)
    return slide


def add_page_number(slide, n):
    add_text(slide, str(n), Inches(12.7), Inches(7.1), Inches(0.5), Inches(0.3),
             size=10, color=GRAY, align=PP_ALIGN.RIGHT)


def add_picture_fit(slide, path: Path, left, top, max_w, max_h, caption=None):
    if not path.exists():
        add_text(slide, f"[缺少图：{path.name}]", left, top + Inches(1), max_w, Inches(0.5),
                 size=14, color=RED)
        return
    from PIL import Image

    with Image.open(path) as im:
        iw, ih = im.size
    ratio = min(max_w / iw, max_h / ih)
    w, h = int(iw * ratio), int(ih * ratio)
    l = left + int((max_w - w) / 2)
    t = top + int((max_h - h) / 2)
    slide.shapes.add_picture(str(path), l, t, width=w, height=h)
    if caption:
        add_text(slide, caption, left, top + max_h + Inches(0.05), max_w, Inches(0.3),
                 size=10, color=GRAY, align=PP_ALIGN.CENTER)


def add_table(slide, header, rows, left, top, width, height, col_widths=None,
              font_size=12, header_fill=PRIMARY, zebra=True):
    nrow, ncol = len(rows) + 1, len(header)
    shape = slide.shapes.add_table(nrow, ncol, left, top, width, height)
    table = shape.table
    if col_widths:
        total = sum(col_widths)
        for j, cw in enumerate(col_widths):
            table.columns[j].width = int(width * cw / total)
    for j, h in enumerate(header):
        cell = table.cell(0, j)
        cell.fill.solid()
        cell.fill.fore_color.rgb = header_fill
        cell.margin_left = cell.margin_right = Inches(0.06)
        tf = cell.text_frame
        tf.word_wrap = True
        p = tf.paragraphs[0]
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = str(h)
        set_font(run, size=font_size, bold=True, color=WHITE)
    for i, row in enumerate(rows):
        for j in range(ncol):
            cell = table.cell(i + 1, j)
            cell.fill.solid()
            cell.fill.fore_color.rgb = (LIGHT_BG if (i % 2 == 1 and zebra) else WHITE)
            cell.margin_left = cell.margin_right = Inches(0.06)
            cell.margin_top = cell.margin_bottom = Inches(0.02)
            tf = cell.text_frame
            tf.word_wrap = True
            p = tf.paragraphs[0]
            p.alignment = PP_ALIGN.CENTER if j > 0 else PP_ALIGN.LEFT
            run = p.add_run()
            run.text = str(row[j]) if j < len(row) else ""
            color = DARK
            if isinstance(run.text, str) and run.text.startswith("+"):
                color = GREEN
            set_font(run, size=font_size, color=color)
    return table


def add_notes(slide, text):
    slide.notes_slide.notes_text_frame.text = text


def metric_cards(slide, cards, top=Inches(2.5), height=Inches(2.1)):
    n = len(cards)
    gap = Inches(0.35)
    total_w = W - Inches(1.1)
    card_w = int((total_w - gap * (n - 1)) / n)
    for i, (value, label, color) in enumerate(cards):
        left = Inches(0.55) + i * (card_w + gap)
        add_rect(slide, left, top, card_w, height, fill=LIGHT_BG, line=RGBColor(0xDC, 0xE3, 0xEA))
        add_text(slide, value, left, top + Inches(0.28), card_w, Inches(0.85),
                 size=40, bold=True, color=color, align=PP_ALIGN.CENTER)
        add_text(slide, label, left + Inches(0.15), top + Inches(1.15),
                 card_w - Inches(0.3), Inches(0.8), size=13, color=GRAY,
                 align=PP_ALIGN.CENTER)


# --------------------------------------------------------------------------
# 数据
# --------------------------------------------------------------------------
def load_summary() -> dict:
    p = RESULTS / "SUMMARY.json"
    if not p.exists():
        return {}
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)


def fmt_pct(x, digits=1):
    return f"{x:+.{digits}f}"



def slide_landscape(prs, n):
    """2024–2026 研究版图与我们位置（相关工作页）。"""
    slide = new_slide(prs, "2024–2026 研究版图：我们的位置")
    rows = [
        ["Google / DeepMind", "AlphaEvolve 评估器驱动自改进；Titans/Nested Learning 参数级持续学习；FSF v3 治理",
         "同向：评估器+治理；缺口：工业无标签漂移闭环"],
        ["OpenAI", "Tax AI 自改进飞轮；AgentKit Evals；Preparedness v2 把 AI Self-improvement 列为 Tracked Category",
         "同向：评测飞轮/风险分级；缺口：生产版本自动回滚"],
        ["Anthropic", "受控 RSI 议程（《When AI builds itself》）、AAR 自动对齐、Claude Code/Skills、RSP",
         "概念已被占位；不涉及工业漂移与标注预算"],
        ["DeepSeek", "R1（Nature）训练期自我验证；V3.2→V4 快速版本迭代；昇腾适配",
         "模型层版本滚动先例；无部署后漂移闭环"],
        ["华为 / 软通", "盘古预测大模型（万华 70%→90%）、天璇 MaaS/数智本体、Token 工厂 LoadGen 2.0",
         "工业基座已有；缺“模型质量自愈”控制回路"],
    ]
    add_table(slide, ["机构", "2024–2026 公开方向（官方自述）", "与我们的关系"], rows,
              Inches(0.5), Inches(1.65), Inches(12.35), Inches(3.6),
              col_widths=[1.9, 7.2, 3.3], font_size=10.5)
    cards = [
        ("N1 双层受控 RSI 闭环", "有界动作空间 + 预算硬门 + 探针门禁 + 版本回滚 + 哈希链审计"),
        ("N2 诊断→假设→探针门禁", "无标签漂移流上用预算内探针标签构造临时验证器"),
        ("N3 可复现/可审计评测协议", "run_id + 配置快照 + SQLite 镜像 + 七场景基准 + A1–A7 验收"),
    ]
    cw = Inches(3.95)
    for i, (head, body) in enumerate(cards):
        left = Inches(0.5) + i * (cw + Inches(0.22))
        add_rect(slide, left, Inches(5.45), cw, Inches(1.05), fill=ACCENT_LIGHT)
        add_text(slide, head, left + Inches(0.15), Inches(5.52), cw - Inches(0.3), Inches(0.35),
                 size=11.5, bold=True, color=PRIMARY)
        add_text(slide, body, left + Inches(0.15), Inches(5.85), cw - Inches(0.3), Inches(0.55),
                 size=9.2, color=GRAY)
    add_text(slide, "我们不声称首创门禁/回滚（2026 已有 VaG/Bilevel/Gated）；定位=应用层·无标签漂移流·预算受控·可审计闭环（docs/16）",
             Inches(0.5), Inches(6.65), Inches(12.3), Inches(0.35), size=10, color=RED, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "相关工作页：先讲巨头都在做自改进（Anthropic 已把它变成公司议程），再讲 2026 年门禁式自进化已出现，"
                     "所以我们明确不声称首创；我们的位置是无标签工业漂移流的应用层控制回路 + 可复现审计。"
                     "引用口径：公司自述、未独立复现；VaG/Bilevel/Gated 为 2026 预印本。")

# --------------------------------------------------------------------------
# 幻灯片
# --------------------------------------------------------------------------
def slide_cover(prs, team: str = "＿＿＿＿", school: str = "＿＿＿＿", advisor: str = "＿＿＿＿"):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, W, H, fill=RGBColor(0x0B, 0x1F, 0x3A), shape=MSO_SHAPE.RECTANGLE)
    add_rect(slide, 0, Inches(1.1), W, Inches(0.06), fill=ACCENT, shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, "智训引擎", Inches(0.9), Inches(2.1), Inches(11.5), Inches(1.1),
             size=52, bold=True, color=WHITE)
    add_text(slide, "工业模型自进化系统", Inches(0.95), Inches(3.15), Inches(11.5), Inches(0.8),
             size=30, bold=True, color=RGBColor(0x9E, 0xC5, 0xF0))
    add_text(slide, "漂移监测 → 诊断归因 → 实验优化 → 评估上线 → 回滚", Inches(0.95), Inches(4.1),
             Inches(11.5), Inches(0.5), size=17, color=RGBColor(0xC9, 0xD8, 0xEA))
    add_text(slide, "第二届北京市大学生“人工智能+”创新大赛 · “智训引擎”赛题",
             Inches(0.95), Inches(5.6), Inches(11.5), Inches(0.4), size=14,
             color=RGBColor(0xB0, 0xC4, 0xDA))
    add_text(slide, f"参赛团队：{team}　　学校：{school}　　指导教师：{advisor}　　日期：2026-11",
             Inches(0.95), Inches(6.15), Inches(11.5), Inches(0.4), size=12.5,
             color=RGBColor(0x9E, 0xC5, 0xF0))
    add_notes(slide, "开场：一句话说明我们做了什么——在无标签漂移流上让工业模型自己发现失准、自己归因、自己实验、"
                     "自己上线和回滚，且每一步都记账、可审计、可复现。")


def slide_summary(prs, summary, n):
    slide = new_slide(prs, "一句话结论", "把「模型自进化」做成一个预算受控、可审计、可复现的工程系统")
    g = summary.get("groups", {})
    main = g.get("main", {})
    scens = g.get("scenarios", {})
    main_gain = main.get("gain_points_mean", 0.0)
    main_cost = main.get("cost_scalar_mean", 0.0)
    all_gain = scens.get("gain_points_mean", 0.0)
    cards = [
        (fmt_pct(main_gain) + " pt", "S7 综合漂移：\n流式 F1 相对静态基线提升", GREEN),
        (fmt_pct(all_gain) + " pt", "七场景 × 3 seeds：\n平均流式增益", GREEN),
        (f"{main_cost:.1f}", "等效成本（scalar）：\n标注+算力+token 全记账", ACCENT),
    ]
    metric_cards(slide, cards, top=Inches(2.35), height=Inches(2.0))
    add_bullets(slide, [
        ("不是所有漂移都值得重训。", "诊断决定最小充分动作：零标签校准 / 少量标注微调 / 类别扩展 / 回滚。"),
        ("LLM 不进高频回路。", "确定性策略 + 在线 bandit 做每窗口决策（0 token）；LLM 只做低频开放式决策。"),
        ("受控 RSI。", "只允许「模型版本滚动 + 策略先验更新」，白名单/预算/审计/回滚四道边界。"),
    ], Inches(0.55), Inches(4.75), Inches(12.2), Inches(2.2), size=15)
    add_page_number(slide, n)
    add_notes(slide, "三个大数字是整份材料的总纲：S7 主场景增益、七场景平均增益、成本。"
                     "接下来每一页都在证明这三个数字的来源与可信度。")


def slide_problem(prs, n):
    slide = new_slide(prs, "官方评分：35 客观 + 65 主观 + 进阶加分<10",
                      "评分只看效果、成本与论证质量；不因路线选择加减分（官方速览视频）")
    # 标题区：title 1.02–1.77，kicker 1.86–2.31；表格从 2.40 起，避免遮挡
    rows = [
        ["漂移流综合性能", "15", "G +11.39pt（G_norm +36.9%）"],
        ["最差阶段表现", "5", "最差阶段 F1 0.505（+6.31pt）"],
        ["恢复速度", "5", "平均 1.04 阶段恢复至漂移前 90%"],
        ["稳定性与抗遗忘", "5", "回测最大下降 0.54pt（≤5%）"],
        ["资源效率", "5", "196 标注 / 0 训练算力 / 0 token"],
    ]
    add_table(slide, ["客观分（35）", "分值", "我们的实测"], rows,
              Inches(0.45), Inches(2.40), Inches(6.15), Inches(2.55),
              col_widths=[2.1, 0.8, 3.25], font_size=11.5)
    rows2 = [
        ["方法设计与技术路线", "30", "docs/02 + docs/16"],
        ["实验规范与分析深度", "15", "记录汇编 + docs/08"],
        ["安全可控与工业落地", "10", "docs/06 + 161 tests"],
        ["报告与答辩", "10", "方案 PDF + PPT + 讲稿"],
    ]
    add_table(slide, ["主观分（65）", "分值", "对应材料"], rows2,
              Inches(6.75), Inches(2.40), Inches(6.0), Inches(2.10),
              col_widths=[2.2, 0.8, 3.0], font_size=11.5)
    add_text(slide, "进阶加分 <10：对照实验证明的创新", Inches(0.45), Inches(5.12),
             Inches(6.15), Inches(0.32), size=12, bold=True, color=PRIMARY)
    add_text(slide, "A2 关诊断 +0.00 / A7 关 TTA −0.56 / AI4I 真实数据 +5.63pt / 预算扫描 60·120·220",
             Inches(0.45), Inches(5.42), Inches(6.15), Inches(0.45), size=10, color=GRAY)
    add_text(slide, "数据包-B 预演（决赛含脏数据）", Inches(6.75), Inches(4.62),
             Inches(6.0), Inches(0.32), size=12, bold=True, color=PRIMARY)
    add_text(slide, "重度脏数据：清洗后 0.704（+1.02pt vs 静态）／关清洗 0.159，闭环仍完成。",
             Inches(6.75), Inches(4.92), Inches(6.0), Inches(0.45), size=10, color=GRAY)
    add_text(slide, "路线 A/B 不加减分 → 混合双层是效果与成本的选择；决赛换数据包-B（更大/更复杂/含脏数据），已补抗脏数据清洗与压力测试。",
             Inches(0.45), Inches(5.95), Inches(12.3), Inches(0.6), size=10.5, color=RED, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "官方评分口径来自 5 分钟速览视频（2026-11-16 获取）：客观 35=综合性能15+最差阶段5+恢复速度5+"
                     "抗遗忘5+资源效率5；主观 65=方法路线30+实验规范15+安全10+答辩10；加分≤10 须对照实验证明。"
                     "路线选择不加减分。决赛换数据包-B（含脏数据），我们已做 RobustSanitizer 清洗与压力测试。")


def slide_insight(prs, n):
    slide = new_slide(prs, "三个核心判断", "它们直接决定了系统长什么样")
    cards = [
        ("判断一", "漂移分型，动作分阶",
         "轴承磨损要微调；传感器老化零标签即可校准；\n工况切换要带标签微调；新故障要扩类别；\n维修复位要能回滚。NOOP 也是合法动作。", ACCENT),
        ("判断二", "LLM 不进高频回路",
         "每窗口数值决策用确定性策略+在线 bandit（0 token）；\nLLM 只在每漂移一次的归因解释/实验设计/\n报告生成出现。按 PyTRi0 定价每 run < ¥0.3。", GREEN),
        ("判断三", "受控 RSI = 数值层 + 不可变版本",
         "策略层只改先验数值，不改代码/门禁/预算；\n模型版本不可变、lineage 完整、随时回滚；\n所有决策进 sha256 哈希链审计。", ORANGE),
    ]
    card_w = Inches(3.95)
    for i, (tag, title, body, color) in enumerate(cards):
        left = Inches(0.55) + i * (card_w + Inches(0.2))
        add_rect(slide, left, Inches(2.25), card_w, Inches(4.15), fill=LIGHT_BG,
                 line=RGBColor(0xDC, 0xE3, 0xEA))
        add_rect(slide, left, Inches(2.25), Inches(0.1), Inches(4.15), fill=color,
                 shape=MSO_SHAPE.RECTANGLE)
        add_text(slide, tag, left + Inches(0.3), Inches(2.45), card_w - Inches(0.5), Inches(0.4),
                 size=12, bold=True, color=color)
        add_text(slide, title, left + Inches(0.3), Inches(2.85), card_w - Inches(0.5), Inches(0.8),
                 size=18, bold=True, color=PRIMARY)
        add_text(slide, body, left + Inches(0.3), Inches(3.75), card_w - Inches(0.55), Inches(2.4),
                 size=12.5, color=GRAY)
    add_page_number(slide, n)
    add_notes(slide, "如果只能记住一页，就是这一页：诊断驱动的成本阶梯、LLM 的低频定位、受控 RSI 的可验证边界。")


def slide_route(prs, n):
    slide = new_slide(prs, "技术路线：为什么选混合双层", "不是「能不能用大模型」，而是「在哪一层用、花多少」")
    rows = [
        ["单位成本增益", "低（每漂移都重训）", "中低（token 计入成本）", "高（诊断决定动作）"],
        ["归因能力", "无", "有（依赖 LLM 质量）", "有（统计证据可审计）"],
        ["开集新故障", "不支持", "支持（不稳定）", "支持（类别扩展）"],
        ["可复现", "高", "低", "高（确定性默认）"],
        ["离线无网", "满足", "需本地模型", "满足"],
        ["策略层自改进", "无", "隐含在 prompt", "显式 bandit 策略表"],
        ["受控 RSI 可验证", "弱", "弱", "强（白名单/预算/审计/回滚）"],
    ]
    add_table(slide, ["维度", "R1 纯微调", "R2 纯 LLM Harness", "R3 混合双层（本方案）"], rows,
              Inches(0.55), Inches(2.32), Inches(12.2), Inches(3.6),
              col_widths=[2.6, 3.0, 3.2, 3.4], font_size=12.5)
    add_rect(slide, Inches(0.55), Inches(6.0), Inches(12.2), Inches(0.8), fill=ACCENT_LIGHT)
    add_text(slide, "LLM 成本论据：Qwen3.8-27B 一次 4k+600 token 决策 ≈ ¥0.049；60 窗口×3–5 次将成主要成本项。"
                    "→ 高频用确定性策略，低频用 LLM。",
             Inches(0.8), Inches(6.12), Inches(11.8), Inches(0.6), size=13, color=PRIMARY, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "路线对比是主观分核心。要点：R1 省钱但笨，R2 聪明但贵且不稳定，R3 把贵的部分放到低频。")


def slide_arch(prs, n):
    slide = new_slide(prs, "系统架构")
    add_picture_fit(slide, FIGS / "fig0_architecture.png", Inches(0.35), Inches(2.0),
                    Inches(12.6), Inches(4.9))
    add_page_number(slide, n)
    add_notes(slide, "沿着箭头讲一遍闭环；强调两条旁路：平台侧隐藏评测（系统不可见）与可观测性/安全（所有决策留痕）。")


def slide_playbooks(prs, n, summary):
    slide = new_slide(prs, "诊断驱动的最小充分动作", "9 个 playbook 组成成本阶梯，诊断结果决定花多少钱")
    rows = [
        ["NOOP", "0", "0", "benign / 预算不足", "low"],
        ["RECALIBRATE_AFFINE", "0", "≈0", "传感器老化（仿射）", "low"],
        ["PRIOR_CORRECTION", "0", "≈0", "标签先验漂移", "low"],
        ["LAST_LAYER_FT", "20–60", "低", "轻度概念漂移", "low"],
        ["FULL_FINETUNE_REPLAY", "60–150", "中", "工况切换", "medium"],
        ["RETRAIN_MIXED", "120–250", "中高", "严重/混合漂移", "medium"],
        ["NOVEL_CLASS_EXTEND", "100–200", "中高", "开集新故障", "high"],
        ["SHADOW_ONLY / ROLLBACK", "0", "低/0", "门禁边缘 / 修复回退", "low/medium"],
    ]
    add_table(slide, ["playbook", "标注", "算力", "适用诊断", "风险"], rows,
              Inches(0.55), Inches(2.38), Inches(8.0), Inches(4.4),
              col_widths=[3.3, 0.9, 0.8, 2.4, 1.1], font_size=12)
    add_rect(slide, Inches(8.85), Inches(2.38), Inches(3.9), Inches(4.4), fill=LIGHT_BG,
             line=RGBColor(0xDC, 0xE3, 0xEA))
    add_text(slide, "关键创新：仿射可恢复性实测", Inches(9.1), Inches(2.3), Inches(3.4), Inches(0.7),
             size=14, bold=True, color=ACCENT)
    add_bullets(slide, [
        "用预算内探针标签，实测「目标域统计量对齐后」的准确率回升",
        "回升显著 → covariate_affine：零标签校准，不花标注",
        "不回升 → 概念漂移：才升级到带标签微调",
        "每类漂移都有可测证据，而不是拍脑袋",
    ], Inches(9.1), Inches(3.05), Inches(3.5), Inches(3.2), size=12, bullet="· ")
    add_page_number(slide, n)
    add_notes(slide, "重点讲「仿射可恢复性」：这是把诊断从定性做成定量的关键，也是省钱的核心机制。"
                     "S2 场景零标签即可从 0.95 恢复到接近上界。")


def slide_strategy(prs, n):
    slide = new_slide(prs, "策略层：把经验写回策略表", "受控 RSI 的「递归」发生在这里，而且可 diff、可审计")
    add_bullets(slide, [
        ("状态：", "诊断类型 × 严重度分箱 × 剩余预算分箱"),
        ("动作：", "白名单内的 playbook（成本阶梯）"),
        ("奖励：", "门禁通过的实测增益 − λ·成本"),
        ("更新：", "Thompson 采样 / UCB，每个 (状态, 动作) 维护后验"),
        ("落盘：", "experiments/strategy_policy.json（旧先验→新先验→依据 run_id）"),
    ], Inches(0.55), Inches(2.15), Inches(6.3), Inches(3.6), size=15)
    add_rect(slide, Inches(7.1), Inches(2.1), Inches(5.65), Inches(4.4), fill=LIGHT_BG,
             line=RGBColor(0xDC, 0xE3, 0xEA))
    add_text(slide, "受控边界（只改数值，不改行为）", Inches(7.35), Inches(2.3), Inches(5.2), Inches(0.5),
             size=14, bold=True, color=RED)
    add_bullets(slide, [
        "不能新增 playbook / 改门禁阈值 / 改预算",
        "不能访问网络、不能执行任意代码",
        "高风险动作（类别扩展/重训）记录升级",
        "预算耗尽 / 连续失败 / STOP 文件 → 熔断",
        "审计链 sha256(prev_hash + record)，可校验",
    ], Inches(7.35), Inches(2.9), Inches(5.2), Inches(3.4), size=12.5, bullet="✕ ")
    add_page_number(slide, n)
    add_notes(slide, "评委容易质疑 RSI 安全性：这一页就是答案——递归只发生在数值先验，且每次变更都可追溯。")


def slide_main_result(prs, n, summary):
    slide = new_slide(prs, "主实验：S7 综合漂移（静态 0.701 → 自身 0.800）")
    add_picture_fit(slide, FIGS / "fig1_main_curve.png", Inches(0.4), Inches(2.0),
                    Inches(8.4), Inches(4.7))
    main = summary.get("groups", {}).get("main", {})
    cards = [
        (f"{main.get('static_f1_mean', 0.0):.3f}", "静态基线\n流式 F1", GRAY),
        (f"{main.get('method_f1_mean', 0.0):.3f}", "自进化\n流式 F1", GREEN),
        (f"{main.get('labels_mean', 0):.0f}", "标注消耗\n（条）", ACCENT),
        (f"{main.get('compute_mean', 0):.0f}", "算力消耗\n（units）", ACCENT),
    ]
    x = Inches(9.05)
    for i, (v, label, color) in enumerate(cards):
        top = Inches(2.05) + i * Inches(1.2)
        add_rect(slide, x, top, Inches(3.7), Inches(1.05), fill=LIGHT_BG,
                 line=RGBColor(0xDC, 0xE3, 0xEA))
        add_text(slide, v, x + Inches(0.15), top + Inches(0.12), Inches(1.5), Inches(0.7),
                 size=22, bold=True, color=color)
        add_text(slide, label, x + Inches(1.6), top + Inches(0.2), Inches(2.0), Inches(0.7),
                 size=11, color=GRAY)
    add_page_number(slide, n)
    add_notes(slide, "看点：每次漂移（竖线）后系统在 1-2 个窗口内恢复；成本曲线阶梯上升，说明只在需要时花钱。")


def slide_scenarios(prs, n, summary):
    slide = new_slide(prs, "七种漂移场景：全部不劣于静态基线")
    add_picture_fit(slide, FIGS / "fig2_scenarios.png", Inches(0.4), Inches(2.0),
                    Inches(12.5), Inches(4.3))
    add_text(slide, "S6 是负对照：静态基线已经很高（宏平均对先验漂移不敏感），系统识别为低成本修复，几乎不花钱。",
             Inches(0.55), Inches(6.45), Inches(12.2), Inches(0.4), size=12.5, color=ACCENT, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "按场景讲：S2 靠零标签校准、S3 靠带标签微调、S4 靠类别扩展、S5 靠回滚，S6 证明我们不乱花钱。")


def slide_ablation(prs, n, summary):
    slide = new_slide(prs, "消融：主收益来自「诊断 + 零标签 TTA」")
    add_picture_fit(slide, FIGS / "fig3_ablation.png", Inches(0.3), Inches(1.95),
                    Inches(7.9), Inches(4.4))
    add_picture_fit(slide, FIGS / "fig4_budget.png", Inches(8.25), Inches(1.95),
                    Inches(4.8), Inches(4.4))
    add_text(slide, "左：消融（A1 关闭触发 / A2 关闭诊断 / A3 随机规划 / A4 关闭门禁 / A5 关闭策略学习 / A6 关闭回滚）",
             Inches(0.4), Inches(6.5), Inches(7.7), Inches(0.4), size=10.5, color=GRAY)
    add_text(slide, "右：标注预算 60/120/220 条下的增益",
             Inches(8.25), Inches(6.5), Inches(4.8), Inches(0.4), size=10.5, color=GRAY)
    add_page_number(slide, n)
    add_notes(slide, "诚实解读：A2（关闭诊断）增益归零、A7（关闭TTA）变负，证明主收益来源；"
                     "A1/A3/A4/A5/A6 在 S7 上与主实验相同，因为 260 标注预算下只发生 1 次更新，"
                     "这些机制的价值是安全性（门禁/看门狗/审计），由 docs/06 的负向测试与回滚演练证明。"
                     "预算扫描：60 条时多数场景保守 NOOP，120 条起恢复。")


def slide_pareto(prs, n, summary):
    slide = new_slide(prs, "成本-增益：帕累托最优")
    add_picture_fit(slide, FIGS / "fig6_pareto.png", Inches(0.4), Inches(2.0),
                    Inches(7.6), Inches(4.6))
    mg = summary.get("groups", {}).get("main", {})
    ours_labels = f"{mg.get('labels_mean', 0):.0f}"
    ours_compute = f"{mg.get('compute_mean', 0):.0f}"
    ours_f1 = f"{mg.get('method_f1_mean', 0):.3f}"
    rows = [
        ["B0 静态基线", "0", "0", f"{mg.get('static_f1_mean', 0):.3f}", "—"],
        ["B1 周期重训", "600–1200", "433–926", "0.80", "高成本低增益"],
        ["B2 Oracle 上界", "3600–7200", "276–552", "0.91", "标注不可行"],
        ["Ours", ours_labels, ours_compute, ours_f1, "Pareto 前沿"],
    ]
    add_table(slide, ["方法（S7）", "标注", "算力", "流式 F1", "评价"], rows,
              Inches(8.25), Inches(2.2), Inches(4.7), Inches(2.6),
              col_widths=[1.6, 1.0, 1.0, 1.0, 1.4], font_size=11.5)
    add_text(slide, "结论：0 训练算力 + 约 196 条标注即超过周期重训（1200 标注/926 units），并接近 oracle 上界。",
             Inches(8.25), Inches(5.2), Inches(4.7), Inches(1.0), size=13, color=PRIMARY, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "这是客观分的核心证据：单位增益成本。评委看的是「花了多少、赚了多少」。")


def slide_safety(prs, n):
    slide = new_slide(prs, "安全可控：回滚不是口号")
    add_picture_fit(slide, FIGS / "fig5_timeline.png", Inches(0.4), Inches(2.0),
                    Inches(8.3), Inches(4.6))
    add_bullets(slide, [
        ("门禁四判：", "增益、类召回、校准、成本"),
        ("看门狗：", "连续 3 窗口低于参考 → 自动回滚"),
        ("审计链：", "sha256(prev_hash+record)，可 verify"),
        ("负向测试：", "越权动作/超预算/篡改日志全部拒绝"),
        ("无人值守：", "一条命令，失败自动降级不中断"),
        ("kill-switch：", "STOP 文件出现即停止更新"),
    ], Inches(9.0), Inches(2.2), Inches(3.9), Inches(4.3), size=13.5, bullet="· ")
    add_page_number(slide, n)
    add_notes(slide, "安全是主观分项。强调：回滚有实测证据（S5 维修复位/有害更新注入），审计链有单测。")


def slide_engineering(prs, n):
    slide = new_slide(prs, "工程与可复现", "一条命令跑完闭环；每条数字都能反查到 run_id")
    add_bullets(slide, [
        ("纯 numpy + CPU：0 网络依赖", "，官方环境「普通 CPU 即可」直接适配"),
        ("实验记录：", "每次触发/实验/上线/回滚 = 一个 run；SQLite 镜像 + SwanLab 兼容（online/local/offline）"),
        ("复现：", "配置快照 + seed + run_id；`python -m src.cli repro` 逐窗口比对"),
        ("审计：", "`python -m src.cli audit-verify` 校验哈希链完整性"),
        ("官方数据：", "DatasetAdapter 隔离，换 adapter 即接入企业赛数据"),
        ("LLM 接线：", "PyTRi0 / OpenAI 兼容，环境变量配置，失败自动降级"),
    ], Inches(0.55), Inches(2.2), Inches(12.2), Inches(3.6), size=15)
    add_rect(slide, Inches(0.55), Inches(5.65), Inches(12.2), Inches(1.2), fill=RGBColor(0x24, 0x29, 0x2F))
    add_text(slide, "$ python -m src.cli run --config configs/main.json\n"
                    "$ python -m src.cli suite --groups main,scenarios,ablations,budget --seeds 7,17,27",
             Inches(0.8), Inches(5.82), Inches(11.7), Inches(0.9), size=13,
             color=RGBColor(0x9E, 0xC5, 0xF0))
    add_page_number(slide, n)
    add_notes(slide, "工程完整性：这是「可复现、可审计」的落地证据。企业赛只需换配置与数据 adapter。")


def slide_limitations(prs, n):
    slide = new_slide(prs, "局限与下一步", "诚实说明边界，比过度承诺更可信")
    add_bullets(slide, [
        ("合成基准的边界：", "已用公开工业数据佐证，并给出官方数据接入路径；真实分布差异是最大风险"),
        ("门禁阈值启发式：", "平台口径未知 → 做权重敏感性分析，结论用 Pareto 而非单点分数"),
        ("LLM 路线仅 mock 验证：", "校赛无 PyTRi0 额度 → 已完成接线/计量/降级（mock 闭环 +7.84pt，token 为估算），企业赛补真实额度对比"),
        ("小模型边界：", "预提取特征任务足够；原始长序列端到端需换 adapter/模型"),
    ], Inches(0.55), Inches(2.2), Inches(12.2), Inches(3.3), size=15)
    add_rect(slide, Inches(0.55), Inches(5.4), Inches(12.2), Inches(1.35), fill=ACCENT_LIGHT)
    add_text(slide, "企业赛路线图：① 官方数据 adapter 接入验证（1 天）→ ② PyTRi0 Qwen3.8-27B 驱动接入（2 天）"
                    "→ ③ 隐藏流封闭评测演练（1 天）→ ④ 预算-增益调优（持续）",
             Inches(0.8), Inches(5.55), Inches(11.8), Inches(1.0), size=13.5, color=PRIMARY, bold=True)
    add_page_number(slide, n)
    add_notes(slide, "主动讲局限，避免被问倒；同时给出清晰的下一步，显示工程成熟度。")


def slide_end(prs):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_rect(slide, 0, 0, W, H, fill=RGBColor(0x0B, 0x1F, 0x3A), shape=MSO_SHAPE.RECTANGLE)
    add_text(slide, "让模型在实际使用中自我进化", Inches(0.9), Inches(2.6), Inches(11.5), Inches(1.0),
             size=36, bold=True, color=WHITE)
    add_text(slide, "可复现 · 可审计 · 安全可控 · 预算受控", Inches(0.95), Inches(3.7), Inches(11.5),
             Inches(0.6), size=20, color=RGBColor(0x9E, 0xC5, 0xF0))
    add_text(slide, "谢谢，请评委指正", Inches(0.95), Inches(5.4), Inches(11.5), Inches(0.5),
             size=16, color=RGBColor(0xB0, 0xC4, 0xDA))
    add_notes(slide, "收尾：回到赛题原话——让模型在实际使用中自我进化；我们给了一个可运行、可审计、受控的答案。")


def build(out: Path, team: str = "＿＿＿＿", school: str = "＿＿＿＿", advisor: str = "＿＿＿＿"):
    summary = load_summary()
    if not summary:
        raise SystemExit(
            "缺少 results/SUMMARY.json：请先运行 scripts/run_all.py 生成正式实验汇总，"
            "再构建答辩 PPT（禁止用写死的占位数字出正式材料）。")
    prs = Presentation()
    prs.slide_width, prs.slide_height = W, H
    slide_cover(prs, team=team, school=school, advisor=advisor)
    n = 2
    slide_summary(prs, summary, n); n += 1
    slide_problem(prs, n); n += 1
    slide_insight(prs, n); n += 1
    slide_route(prs, n); n += 1
    slide_landscape(prs, n); n += 1
    slide_arch(prs, n); n += 1
    slide_playbooks(prs, n, summary); n += 1
    slide_strategy(prs, n); n += 1
    slide_main_result(prs, n, summary); n += 1
    slide_scenarios(prs, n, summary); n += 1
    slide_ablation(prs, n, summary); n += 1
    slide_pareto(prs, n, summary); n += 1
    slide_safety(prs, n); n += 1
    slide_engineering(prs, n); n += 1
    slide_limitations(prs, n); n += 1
    slide_end(prs)
    out.parent.mkdir(parents=True, exist_ok=True)
    prs.save(str(out))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(OUT / "智训引擎_答辩PPT.pptx"))
    ap.add_argument("--team", default="参赛团队")
    ap.add_argument("--school", default="XX 大学")
    ap.add_argument("--advisor", default="＿＿＿＿")
    args = ap.parse_args()
    out = build(Path(args.out), args.team, args.school, args.advisor)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
