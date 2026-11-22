"""Render the report Markdown subset to PDF with ReportLab.

One renderer serves all three report types. It understands headings, paragraphs,
bullet and numbered lists, Markdown tables, blockquotes (callouts), fenced code,
and a small figure directive written as an HTML comment so the Markdown stays
readable:

    <!-- figure: key | Caption text -->

ReportLab is optional; ``pdf_available`` lets callers fall back to Markdown.
"""

from __future__ import annotations

import importlib.util
import re
from io import BytesIO
from pathlib import Path

A4 = (595.27, 841.89)
ACCENT = "#2F5D8A"
MUTED = "#6B7280"
LIGHT = "#F1F5F9"

_UNICODE_MAP = {
    "≥": ">=", "≤": "<=", "−": "-", "×": "x", "→": "->", "≈": "~",
    "–": "-", "—": "-", "“": '"', "”": '"', "‘": "'", "’": "'",
    "…": "...", "·": "-", "\u00a0": " ",
}


POINTS_PER_INCH = 72.0
# A table this wide (or wider) uses a smaller font and a tighter minimum column
# width, so every column of a wide subgroup table stays readable on A4.
WIDE_TABLE_COLUMNS = 9
WIDE_TABLE_MIN_COLUMN = 0.42 * POINTS_PER_INCH
NARROW_TABLE_MIN_COLUMN = 0.7 * POINTS_PER_INCH
_WIDE_CELL_LENGTH_CAP = 26


def table_column_widths(rows: list[list[str]], available: float) -> list[float]:
    """Column widths for a Markdown table, scaled to the available width.

    Widths follow the header and body text, but a long text column is capped so
    it cannot squeeze every short numeric column to nothing. The returned widths
    always sum to ``available``.
    """
    columns = len(rows[0])
    wide = columns >= WIDE_TABLE_COLUMNS
    cap = _WIDE_CELL_LENGTH_CAP if wide else 10_000
    minimum = WIDE_TABLE_MIN_COLUMN if wide else NARROW_TABLE_MIN_COLUMN
    lengths = []
    for column in range(columns):
        header = len(str(rows[0][column]))
        body = max((len(str(row[column])) for row in rows[1:]), default=0)
        lengths.append(max(header, min(body, cap)))
    total = sum(lengths) or 1
    widths = [max(minimum, available * length / total) for length in lengths]
    scale = available / sum(widths)
    return [width * scale for width in widths]


def pdf_available() -> bool:
    return importlib.util.find_spec("reportlab") is not None


def _safe(text: str) -> str:
    for source, target in _UNICODE_MAP.items():
        text = text.replace(source, target)
    return text.encode("cp1252", "replace").decode("cp1252")


def _inline(text: str) -> str:
    """Escape XML, then apply the tiny inline subset (bold, code)."""
    text = _safe(str(text))
    text = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    text = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", text)
    text = re.sub(r"`([^`]+)`", r'<font face="Courier" size="8">\1</font>', text)
    return text


# --------------------------------------------------------------------------- #
# Markdown subset parser                                                      #
# --------------------------------------------------------------------------- #

def _is_separator(cells: list[str]) -> bool:
    return bool(cells) and all(re.fullmatch(r":?-{2,}:?", cell.strip())
                               for cell in cells)


def _split_row(line: str) -> list[str]:
    line = line.strip()
    if line.startswith("|"):
        line = line[1:]
    if line.endswith("|"):
        line = line[:-1]
    return [cell.strip() for cell in line.split("|")]


def _parse_table(lines: list[str]) -> list[list[str]]:
    rows = [_split_row(line) for line in lines]
    rows = [row for row in rows if not _is_separator(row)]
    width = max((len(row) for row in rows), default=0)
    return [row + [""] * (width - len(row)) for row in rows]


def _starts_block(line: str) -> bool:
    stripped = line.strip()
    return bool(
        stripped.startswith(("```", ">", "|", "#"))
        or re.match(r"^<!--\s*figure:", stripped)
        or re.match(r"^([-*]|\d+\.)\s+", stripped)
    )


def parse_blocks(markdown: str) -> list[tuple[str, object]]:
    lines = markdown.splitlines()
    blocks: list[tuple[str, object]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not stripped:
            index += 1
            continue
        if stripped.startswith("```"):
            index += 1
            code: list[str] = []
            while index < len(lines) and not lines[index].strip().startswith("```"):
                code.append(lines[index])
                index += 1
            index += 1
            blocks.append(("code", code))
        elif re.match(r"^#{1,4}\s", stripped):
            match = re.match(r"^(#{1,4})\s+(.*)$", stripped)
            blocks.append(("heading", (len(match.group(1)), match.group(2).strip())))
            index += 1
        elif re.match(r"^<!--\s*figure:", stripped):
            match = re.match(r"^<!--\s*figure:\s*([^|]+?)\s*\|\s*(.*?)\s*-->$",
                             stripped)
            if match:
                blocks.append(("figure", (match.group(1).strip(),
                                          match.group(2).strip())))
            index += 1
        elif stripped.startswith(">"):
            quote: list[str] = []
            while index < len(lines) and lines[index].strip().startswith(">"):
                quote.append(lines[index].strip().lstrip(">").strip())
                index += 1
            blocks.append(("quote", " ".join(part for part in quote if part)))
        elif stripped.startswith("|"):
            table: list[str] = []
            while index < len(lines) and lines[index].strip().startswith("|"):
                table.append(lines[index])
                index += 1
            blocks.append(("table", _parse_table(table)))
        elif re.match(r"^([-*]|\d+\.)\s+", stripped):
            items: list[tuple[int, str]] = []
            while index < len(lines) and re.match(r"^\s*([-*]|\d+\.)\s+", lines[index]):
                raw = lines[index]
                indent = len(raw) - len(raw.lstrip())
                text = re.sub(r"^\s*([-*]|\d+\.)\s+", "", raw)
                items.append((indent, text))
                index += 1
            blocks.append(("list", items))
        else:
            paragraph = [stripped]
            index += 1
            while (index < len(lines) and lines[index].strip()
                   and not _starts_block(lines[index])):
                paragraph.append(lines[index].strip())
                index += 1
            blocks.append(("paragraph", " ".join(paragraph)))
    return blocks


# --------------------------------------------------------------------------- #
# Rendering                                                                   #
# --------------------------------------------------------------------------- #

def markdown_to_pdf(markdown: str, out_path: str | Path, *,
                    title: str = "", footer: str = "", hypotheses: bool = False,
                    figures: dict[str, bytes] | None = None) -> Path:
    if not pdf_available():
        raise RuntimeError(
            "PDF export needs ReportLab. Install it with `uv sync --extra pdf`.")

    from reportlab.lib import colors
    from reportlab.lib.enums import TA_JUSTIFY, TA_LEFT
    from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
    from reportlab.lib.units import inch
    from reportlab.lib.utils import ImageReader
    from reportlab.platypus import (
        HRFlowable, Image, LongTable, PageBreak, Paragraph, SimpleDocTemplate,
        Spacer, Table, TableStyle,
    )

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    figures = figures or {}
    blocks = parse_blocks(markdown)

    styles = getSampleStyleSheet()
    body = ParagraphStyle("body", parent=styles["BodyText"], fontName="Helvetica",
                          fontSize=9.5, leading=12.5, alignment=TA_JUSTIFY,
                          spaceAfter=4)
    title_style = ParagraphStyle("title", parent=styles["Title"],
                                 fontName="Helvetica-Bold", fontSize=22,
                                 leading=26, textColor=colors.HexColor(ACCENT),
                                 alignment=TA_LEFT, spaceAfter=10)
    h2 = ParagraphStyle("h2", parent=styles["Heading2"], fontName="Helvetica-Bold",
                        fontSize=13.5, leading=16.5, textColor=colors.HexColor(ACCENT),
                        spaceBefore=10, spaceAfter=4)
    h3 = ParagraphStyle("h3", parent=styles["Heading3"], fontName="Helvetica-Bold",
                        fontSize=11, leading=13.5, spaceBefore=8, spaceAfter=3)
    h4 = ParagraphStyle("h4", parent=styles["Heading4"], fontName="Helvetica-Bold",
                        fontSize=10, leading=13, spaceBefore=6, spaceAfter=2)
    bullet = ParagraphStyle("bullet", parent=body, alignment=TA_LEFT,
                            leftIndent=14, bulletIndent=4, spaceAfter=3)
    code_style = ParagraphStyle("code", parent=body, fontName="Courier",
                                fontSize=8.2, leading=10.5, alignment=TA_LEFT,
                                backColor=colors.HexColor(LIGHT),
                                borderPadding=5, spaceBefore=2, spaceAfter=6)
    callout = ParagraphStyle("callout", parent=body, alignment=TA_LEFT,
                             backColor=colors.HexColor(LIGHT),
                             borderColor=colors.HexColor(ACCENT), borderWidth=0.6,
                             borderPadding=9, spaceBefore=10, spaceAfter=12)
    cell = ParagraphStyle("cell", parent=body, fontSize=7.8, leading=9.6,
                          alignment=TA_LEFT, spaceAfter=0)
    cell_header = ParagraphStyle("cellh", parent=cell, fontName="Helvetica-Bold",
                                 textColor=colors.white)
    cell_compact = ParagraphStyle("cellc", parent=cell, fontSize=6.5, leading=8.0)
    cell_header_compact = ParagraphStyle("cellhc", parent=cell_compact,
                                         fontName="Helvetica-Bold",
                                         textColor=colors.white)
    caption = ParagraphStyle("caption", parent=body, fontSize=8.4, leading=10.5,
                             textColor=colors.HexColor(MUTED), alignment=TA_LEFT,
                             spaceBefore=1, spaceAfter=6)
    toc = ParagraphStyle("toc", parent=body, fontSize=10, leading=13,
                         alignment=TA_LEFT)

    doc = SimpleDocTemplate(
        str(out_path), pagesize=A4, title=title or "Report",
        leftMargin=51, rightMargin=51, topMargin=54, bottomMargin=54,  # 18 mm
        author="general-agent",
    )
    available = doc.width

    def _table_flowable(rows: list[list[str]]):
        if not rows:
            return Spacer(1, 0)
        wide = len(rows[0]) >= WIDE_TABLE_COLUMNS
        body_cell = cell_compact if wide else cell
        head_cell = cell_header_compact if wide else cell_header
        data = []
        for row_index, row in enumerate(rows):
            style = head_cell if row_index == 0 else body_cell
            data.append([Paragraph(_inline(value), style) for value in row])
        widths = table_column_widths(rows, available)
        table = LongTable(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
        table.setStyle(TableStyle([
            ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor(ACCENT)),
            ("ROWBACKGROUNDS", (0, 1), (-1, -1),
             [colors.white, colors.HexColor("#F8FAFC")]),
            ("GRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#CBD5E1")),
            ("VALIGN", (0, 0), (-1, -1), "TOP"),
            ("LEFTPADDING", (0, 0), (-1, -1), 4),
            ("RIGHTPADDING", (0, 0), (-1, -1), 4),
            ("TOPPADDING", (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ]))
        return table

    def _figure_flowable(key: str, text: str):
        data = figures.get(key)
        flowables = []
        if data:
            reader = ImageReader(BytesIO(data))
            width, height = reader.getSize()
            max_width = available
            max_height = 4.6 * inch
            scale = min(max_width / width, max_height / height, 1.0)
            image = Image(BytesIO(data), width=width * scale, height=height * scale)
            image.hAlign = "CENTER"
            flowables.append(image)
        else:
            flowables.append(Paragraph(
                _inline(f"[{text}: figure unavailable]"), caption))
        if text:
            flowables.append(Paragraph(_inline(f"Figure. {text}"), caption))
        return flowables

    def render(block):
        kind, payload = block
        if kind == "heading":
            level, text = payload
            if level == 1:
                return [Paragraph(_inline(text), title_style),
                        HRFlowable(width="100%", thickness=1.1,
                                   color=colors.HexColor(ACCENT), spaceAfter=10)]
            return [Paragraph(_inline(text), (h2, h3, h4)[min(level - 2, 2)])]
        if kind == "paragraph":
            return [Paragraph(_inline(payload), body)]
        if kind == "quote":
            return [Paragraph(_inline(payload), callout)]
        if kind == "code":
            return [Paragraph("<br/>".join(_inline(line) for line in payload),
                              code_style)]
        if kind == "table":
            return [_table_flowable(payload), Spacer(1, 4)]
        if kind == "list":
            flowables = []
            for indent, text in payload:
                style = bullet if indent < 2 else ParagraphStyle(
                    "sub", parent=bullet, leftIndent=28)
                flowables.append(Paragraph(_inline(text), style, bulletText="•"))
            return flowables
        if kind == "figure":
            return _figure_flowable(payload[0], payload[1])
        return []

    story = []
    heading_index = next((i for i, (kind, payload) in enumerate(blocks)
                          if kind == "heading" and payload[0] == 2), None)
    if heading_index is None:
        for block in blocks:
            story.extend(render(block))
    else:
        for block in blocks[:heading_index]:
            story.extend(render(block))
        entries = [payload[1] for kind, payload in blocks
                   if kind == "heading" and payload[0] == 2]
        if len(entries) >= 3:
            # Keep the contents right after the title matter to avoid a near-empty
            # page; it flows onto page 2 only when the first page is already full.
            # The extra spacer keeps it clear of the executive-summary callout.
            story.append(Spacer(1, 16))
            story.append(Paragraph("Contents", h2))
            story.append(Spacer(1, 2))
            for entry in entries:
                story.append(Paragraph(_inline(entry), toc))
            story.append(PageBreak())
        else:
            story.append(Spacer(1, 10))
        for block in blocks[heading_index:]:
            story.extend(render(block))

    def _footer(canvas, document):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CBD5E1"))
        canvas.setLineWidth(0.4)
        canvas.line(51, 42, A4[0] - 51, 42)
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawString(51, 31, _safe(title)[:90])
        canvas.drawRightString(A4[0] - 51, 31, f"Page {document.page}")
        if footer:
            canvas.drawCentredString(A4[0] / 2, 31, _safe(footer)[:80])
        if hypotheses:
            canvas.setFillColor(colors.HexColor("#9A3412"))
            canvas.drawRightString(A4[0] - 51, 20, "qualitative content = hypotheses")
        canvas.restoreState()

    def _first_page(canvas, document):
        canvas.saveState()
        canvas.setFont("Helvetica", 7.5)
        canvas.setFillColor(colors.HexColor(MUTED))
        canvas.drawRightString(A4[0] - 51, 31, f"Page {document.page}")
        canvas.restoreState()

    doc.build(story, onFirstPage=_first_page, onLaterPages=_footer)
    return out_path
