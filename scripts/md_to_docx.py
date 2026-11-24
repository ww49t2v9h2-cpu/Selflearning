"""Markdown → 排版精美的 docx（用于方案文档/运行说明的交付版）。

依赖 python-docx（bundled Python 自带）。之后用 office_convert 转 PDF。

用法：
  PY=.../python3
  $PY scripts/md_to_docx.py --input deliverables/参赛方案.md \
      --output deliverables/智训引擎_参赛方案.docx \
      --cover-title "智训引擎：工业模型自进化系统" \
      --cover-subtitle "第二届北京市大学生“人工智能+”创新大赛 · 赛题参赛方案" \
      --cover-meta "参赛团队：XXX（3–5 人）| 学校：XXX | 日期：2026-11"
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

from docx import Document
from docx.enum.section import WD_SECTION
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.enum.text import WD_ALIGN_PARAGRAPH, WD_BREAK
from docx.oxml import OxmlElement
from docx.oxml.ns import qn
from docx.shared import Cm, Pt, RGBColor
from docx.enum.text import WD_LINE_SPACING

ROOT = Path(__file__).resolve().parents[1]

ACCENT = RGBColor(0x1F, 0x4E, 0x79)
ACCENT2 = RGBColor(0x2E, 0x74, 0xB5)
GRAY = RGBColor(0x59, 0x59, 0x59)
LIGHT = RGBColor(0x7F, 0x7F, 0x7F)

CJK_FONT = "Microsoft YaHei"
LATIN_FONT = "Calibri"
MONO_FONT = "Consolas"


# --------------------------------------------------------------------------
# 基础排版辅助
# --------------------------------------------------------------------------
def set_run_font(run, size=None, bold=None, color=None, mono=False, italic=None):
    run.font.name = MONO_FONT if mono else LATIN_FONT
    rpr = run._element.get_or_add_rPr()
    rfonts = rpr.find(qn("w:rFonts"))
    if rfonts is None:
        rfonts = OxmlElement("w:rFonts")
        rpr.append(rfonts)
    rfonts.set(qn("w:ascii"), MONO_FONT if mono else LATIN_FONT)
    rfonts.set(qn("w:hAnsi"), MONO_FONT if mono else LATIN_FONT)
    rfonts.set(qn("w:eastAsia"), CJK_FONT)
    if size is not None:
        run.font.size = Pt(size)
    if bold is not None:
        run.font.bold = bold
    if italic is not None:
        run.font.italic = italic
    if color is not None:
        run.font.color.rgb = color


def shade(element, fill: str):
    """给段落/单元格加背景色。"""
    pPr = element._element.get_or_add_pPr() if hasattr(element, "_element") else element
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    pPr.append(shd)


def shade_cell(cell, fill: str):
    tcPr = cell._tc.get_or_add_tcPr()
    shd = OxmlElement("w:shd")
    shd.set(qn("w:val"), "clear")
    shd.set(qn("w:color"), "auto")
    shd.set(qn("w:fill"), fill)
    tcPr.append(shd)


def add_page_number_footer(section):
    footer = section.footer
    p = footer.paragraphs[0] if footer.paragraphs else footer.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run()
    set_run_font(run, size=8, color=LIGHT)
    fld_begin = OxmlElement("w:fldChar")
    fld_begin.set(qn("w:fldCharType"), "begin")
    instr = OxmlElement("w:instrText")
    instr.set(qn("xml:space"), "preserve")
    instr.text = " PAGE "
    fld_end = OxmlElement("w:fldChar")
    fld_end.set(qn("w:fldCharType"), "end")
    run._r.append(fld_begin)
    run._r.append(instr)
    run._r.append(fld_end)
    run2 = p.add_run("  /  智训引擎·工业模型自进化系统")
    set_run_font(run2, size=8, color=LIGHT)


# --------------------------------------------------------------------------
# 中文标点与 LaTeX 简化
# --------------------------------------------------------------------------
GREEK = {
    r"\alpha": "α", r"\beta": "β", r"\gamma": "γ", r"\delta": "δ", r"\epsilon": "ε",
    r"\lambda": "λ", r"\mu": "μ", r"\pi": "π", r"\sigma": "σ", r"\tau": "τ",
    r"\theta": "θ", r"\Delta": "Δ", r"\Sigma": "Σ", r"\Phi": "Φ", r"\Omega": "Ω",
    r"\times": "×", r"\cdot": "·", r"\le": "≤", r"\ge": "≥", r"\ne": "≠",
    r"\approx": "≈", r"\to": "→", r"\rightarrow": "→", r"\infty": "∞",
    r"\sum": "Σ", r"\int": "∫", r"\pm": "±", r"\in": "∈", r"\subset": "⊂",
    r"\quad": "  ", r"\qquad": "    ",
}


def latex_to_text(s: str) -> str:
    s = re.sub(r"\$\$(.*?)\$\$", lambda m: " " + _one_formula(m.group(1)) + " ", s, flags=re.S)
    s = re.sub(r"\$(.*?)\$", lambda m: _one_formula(m.group(1)), s, flags=re.S)
    return s


def _one_formula(s: str) -> str:
    s = s.strip().replace("\\\\", "  ")
    for k, v in GREEK.items():
        s = s.replace(k, v)
    s = re.sub(r"\\(?:text|mathrm|mathbf|mathcal|operatorname)\{([^{}]*)\}", r"\1", s)
    s = re.sub(r"\\frac\{([^{}]*)\}\{([^{}]*)\}", r"(\1)/(\2)", s)
    s = re.sub(r"\\(?:big|Big|bigg|Bigg)?[\[\]()]", " ", s)
    s = re.sub(r"\\[a-zA-Z]+", "", s)
    s = s.replace("{", "").replace("}", "").replace("&", " ")
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def clean_inline(s: str) -> tuple[str, list[tuple[str, bool, bool]]]:
    """把 **bold** / `code` / [text](url) 拆成 (纯文本, [(片段, bold, code)])。"""
    s = latex_to_text(s)
    s = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", s)      # 链接保留文字
    s = re.sub(r"!\[([^\]]*)\]\([^)]+\)", r"\1", s)      # 图片文字
    tokens: list[tuple[str, bool, bool]] = []
    pattern = re.compile(r"(\*\*.+?\*\*|`[^`]+`|\*[^*]+\*)")
    pos = 0
    for m in pattern.finditer(s):
        if m.start() > pos:
            tokens.append((s[pos:m.start()], False, False))
        seg = m.group(0)
        if seg.startswith("**"):
            tokens.append((seg[2:-2], True, False))
        elif seg.startswith("`"):
            tokens.append((seg[1:-1], False, True))
        else:
            tokens.append((seg[1:-1], False, False))
        pos = m.end()
    if pos < len(s):
        tokens.append((s[pos:], False, False))
    return s, tokens


# --------------------------------------------------------------------------
# Markdown 解析
# --------------------------------------------------------------------------
def parse_markdown(text: str) -> list[tuple]:
    blocks: list[tuple] = []
    lines = text.splitlines()
    i = 0
    while i < len(lines):
        line = lines[i]
        stripped = line.strip()
        if not stripped:
            i += 1
            continue
        # 代码块
        if stripped.startswith("```"):
            lang = stripped[3:].strip()
            j = i + 1
            code = []
            while j < len(lines) and not lines[j].strip().startswith("```"):
                code.append(lines[j])
                j += 1
            blocks.append(("code", "\n".join(code), lang))
            i = j + 1
            continue
        # 图片
        m = re.match(r"!\[([^\]]*)\]\(([^)]+)\)", stripped)
        if m:
            blocks.append(("image", m.group(2), m.group(1)))
            i += 1
            continue
        # 标题
        m = re.match(r"^(#{1,6})\s+(.*)$", stripped)
        if m:
            blocks.append(("heading", len(m.group(1)), m.group(2).strip()))
            i += 1
            continue
        # 表格
        if stripped.startswith("|") and i + 1 < len(lines) and re.match(r"^\|[\s:\-|]+\|$", lines[i + 1].strip()):
            header = [c.strip() for c in stripped.strip("|").split("|")]
            rows = []
            j = i + 2
            while j < len(lines) and lines[j].strip().startswith("|"):
                rows.append([c.strip() for c in lines[j].strip().strip("|").split("|")])
                j += 1
            blocks.append(("table", header, rows))
            i = j
            continue
        # 分隔线
        if re.match(r"^-{3,}$", stripped):
            blocks.append(("hr",))
            i += 1
            continue
        # 引用
        if stripped.startswith(">"):
            content = stripped.lstrip("> ").strip()
            blocks.append(("quote", content))
            i += 1
            continue
        # 列表
        if re.match(r"^[-*+]\s+", stripped) or re.match(r"^\d+[.、]\s+", stripped):
            ordered = bool(re.match(r"^\d+[.、]\s+", stripped))
            items = []
            while i < len(lines):
                s = lines[i].strip()
                m2 = re.match(r"^(?:[-*+]|\d+[.、])\s+(.*)$", s)
                if not m2:
                    break
                items.append(m2.group(1))
                i += 1
            blocks.append(("list", items, ordered))
            continue
        # 普通段落（合并到空行）
        para = [stripped]
        j = i + 1
        while j < len(lines) and lines[j].strip() and not re.match(
                r"^(#|\||```|!\[|>|[-*+]\s|\d+[.、]\s|-{3,}$)", lines[j].strip()):
            para.append(lines[j].strip())
            j += 1
        blocks.append(("para", " ".join(para)))
        i = j
    return blocks


# --------------------------------------------------------------------------
# docx 渲染
# --------------------------------------------------------------------------
def setup_document() -> Document:
    doc = Document()
    section = doc.sections[0]
    section.page_width = Cm(21.0)
    section.page_height = Cm(29.7)
    section.left_margin = Cm(2.4)
    section.right_margin = Cm(2.4)
    section.top_margin = Cm(2.2)
    section.bottom_margin = Cm(2.0)

    styles = doc.styles
    normal = styles["Normal"]
    normal.font.size = Pt(10.5)
    normal.font.name = LATIN_FONT
    normal.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
    normal.paragraph_format.line_spacing = 1.35
    normal.paragraph_format.space_after = Pt(4)

    for name, size, color in [("Heading 1", 17, ACCENT), ("Heading 2", 13.5, ACCENT2),
                              ("Heading 3", 11.5, ACCENT2)]:
        st = styles[name]
        st.font.name = LATIN_FONT
        st.font.size = Pt(size)
        st.font.bold = True
        st.font.color.rgb = color
        st.element.rPr.rFonts.set(qn("w:eastAsia"), CJK_FONT)
        st.paragraph_format.space_before = Pt(14 if name == "Heading 1" else 10)
        st.paragraph_format.space_after = Pt(6)
        st.paragraph_format.keep_with_next = True
    add_page_number_footer(section)
    return doc


def add_cover(doc: Document, title: str, subtitle: str, meta: str):
    for _ in range(4):
        doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(title)
    set_run_font(run, size=28, bold=True, color=ACCENT)
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run(subtitle)
    set_run_font(run, size=13, color=GRAY)
    doc.add_paragraph()
    p = doc.add_paragraph()
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    run = p.add_run("—" * 12)
    set_run_font(run, size=12, color=ACCENT2)
    for _ in range(8):
        doc.add_paragraph()
    for line in meta.split("|"):
        p = doc.add_paragraph()
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER
        run = p.add_run(line.strip())
        set_run_font(run, size=11, color=GRAY)
    p = doc.add_paragraph()
    p.add_run().add_break(WD_BREAK.PAGE)


def add_toc_placeholder(doc: Document, blocks: list[tuple]):
    doc.add_paragraph()
    p = doc.add_paragraph()
    run = p.add_run("目  录")
    set_run_font(run, size=16, bold=True, color=ACCENT)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER
    for b in blocks:
        if b[0] == "heading" and b[1] <= 2:
            depth = b[1] - 1
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.8 * depth)
            run = p.add_run(b[2])
            set_run_font(run, size=11 if depth == 0 else 10.5,
                         bold=(depth == 0), color=ACCENT if depth == 0 else GRAY)
    doc.add_paragraph().add_run().add_break(WD_BREAK.PAGE)


def render_blocks(doc: Document, blocks: list[tuple], base_dir: Path):
    for b in blocks:
        kind = b[0]
        if kind == "heading":
            level, text = b[1], b[2]
            if level == 1:
                continue  # 标题已在封面/文档标题
            h = doc.add_heading(level=min(level - 1, 3))
            run = h.add_run(text)
            set_run_font(run, size=None, bold=True,
                         color=ACCENT if level == 2 else ACCENT2)
        elif kind == "para":
            p = doc.add_paragraph()
            p.paragraph_format.first_line_indent = Cm(0.0)
            _, tokens = clean_inline(b[1])
            for text, bold, code in tokens:
                run = p.add_run(text)
                is_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in text)
                set_run_font(run, size=10.5, bold=bold, mono=(code and not is_cjk),
                             color=RGBColor(0x1F, 0x1F, 0x1F) if not code else RGBColor(0x8B, 0x1A, 0x1A))
        elif kind == "list":
            items, ordered = b[1], b[2]
            # 有序列表：手动编号（Word "List Number" 会跨列表连续编号，导致 1,2 后出现 6,7）
            for idx, item in enumerate(items, 1):
                try:
                    p = doc.add_paragraph(style="List Bullet" if not ordered else "Normal")
                except KeyError:
                    p = doc.add_paragraph()
                if ordered:
                    p.paragraph_format.left_indent = Cm(0.55)
                    num = p.add_run(f"{idx}. ")
                    set_run_font(num, size=10.5)
                _, tokens = clean_inline(item)
                for text, bold, code in tokens:
                    run = p.add_run(text)
                    set_run_font(run, size=10.5, bold=bold, mono=code)
                p.paragraph_format.space_after = Pt(2)
        elif kind == "code":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.4)
            p.paragraph_format.space_before = Pt(4)
            p.paragraph_format.space_after = Pt(6)
            shade(p, "F2F4F6")
            run = p.add_run(b[1])
            set_run_font(run, size=8.5, mono=True, color=RGBColor(0x24, 0x29, 0x2F))
        elif kind == "table":
            header, rows = b[1], b[2]
            ncol = max(len(header), max((len(r) for r in rows), default=0))
            table = doc.add_table(rows=1, cols=ncol)
            table.style = "Table Grid"
            table.alignment = WD_TABLE_ALIGNMENT.CENTER
            hdr = table.rows[0].cells
            for j, txt in enumerate(header + [""] * (ncol - len(header))):
                hdr[j].text = ""
                p = hdr[j].paragraphs[0]
                _, tokens = clean_inline(txt)
                for text, bold, code in tokens:
                    run = p.add_run(text)
                    set_run_font(run, size=8.8, bold=True, mono=code,
                                 color=RGBColor(0xFF, 0xFF, 0xFF))
                shade_cell(hdr[j], "1F4E79")
            for ri, row in enumerate(rows):
                cells = table.add_row().cells
                for j in range(ncol):
                    txt = row[j] if j < len(row) else ""
                    p = cells[j].paragraphs[0]
                    _, tokens = clean_inline(txt)
                    for text, bold, code in tokens:
                        run = p.add_run(text)
                        set_run_font(run, size=8.5, bold=bold, mono=code)
                    if ri % 2 == 1:
                        shade_cell(cells[j], "F2F4F6")
            # 表格后留白
            doc.add_paragraph().paragraph_format.space_after = Pt(2)
        elif kind == "image":
            path = Path(b[1])
            if not path.is_absolute():
                path = (base_dir / path).resolve()
            if path.exists():
                p = doc.add_paragraph()
                p.alignment = WD_ALIGN_PARAGRAPH.CENTER
                run = p.add_run()
                run.add_picture(str(path), width=Cm(16.0))
                if b[2]:
                    cap = doc.add_paragraph()
                    cap.alignment = WD_ALIGN_PARAGRAPH.CENTER
                    r2 = cap.add_run(b[2])
                    set_run_font(r2, size=8.5, color=LIGHT)
            else:
                p = doc.add_paragraph()
                run = p.add_run(f"[缺失图片: {b[1]}]")
                set_run_font(run, size=9, color=RGBColor(0xC0, 0x39, 0x2B))
        elif kind == "quote":
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Cm(0.5)
            p.paragraph_format.right_indent = Cm(0.5)
            shade(p, "EAF1F8")
            _, tokens = clean_inline(b[1])
            for text, bold, code in tokens:
                run = p.add_run(text)
                set_run_font(run, size=9.5, italic=False, mono=code, color=GRAY)
        elif kind == "hr":
            p = doc.add_paragraph()
            pPr = p._element.get_or_add_pPr()
            pbdr = OxmlElement("w:pBdr")
            bottom = OxmlElement("w:bottom")
            bottom.set(qn("w:val"), "single")
            bottom.set(qn("w:sz"), "6")
            bottom.set(qn("w:space"), "1")
            bottom.set(qn("w:color"), "D0D7DE")
            pbdr.append(bottom)
            pPr.append(pbdr)


def build(input_md: Path, output_docx: Path, cover_title: str, cover_subtitle: str,
          cover_meta: str, with_toc: bool = True) -> Path:
    text = input_md.read_text(encoding="utf-8")
    blocks = parse_markdown(text)
    doc = setup_document()
    if cover_title:
        add_cover(doc, cover_title, cover_subtitle, cover_meta)
    if with_toc:
        add_toc_placeholder(doc, blocks)
    render_blocks(doc, blocks, input_md.parent)
    output_docx.parent.mkdir(parents=True, exist_ok=True)
    doc.save(str(output_docx))
    return output_docx


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--cover-title", default="")
    ap.add_argument("--cover-subtitle", default="")
    ap.add_argument("--cover-meta", default="")
    ap.add_argument("--no-toc", action="store_true")
    args = ap.parse_args()
    out = build(Path(args.input), Path(args.output), args.cover_title,
                args.cover_subtitle, args.cover_meta, with_toc=not args.no_toc)
    print(f"written {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
