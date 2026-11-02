"""Markdown → ReportLab flowables (tables, bullets, quotes, headings).

Drop-in for report/weekly_report.py:
    from report.pdf_md import md_to_flowables
"""

import re
from reportlab.lib import colors
from reportlab.lib.colors import HexColor
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

# Characters AppleGothic lacks → ASCII equivalents (keeps minus signs!)
_SUBS = {"\u2212": "-", "\u2013": "-", "\u2014": "-", "\u2192": "->", "\u2022": "·",
         "\u00d7": "x", "\u2264": "<=", "\u2265": ">="}


def _clean(s):
    for k, v in _SUBS.items():
        s = s.replace(k, v)
    s = re.sub(r"\*\*(.+?)\*\*", r"<b>\1</b>", s)
    s = re.sub(r"(?<!\*)\*(?!\*)(.+?)(?<!\*)\*(?!\*)", r"<i>\1</i>", s)
    s = s.replace("&", "&amp;").replace("<b>", "\x00b").replace("</b>", "\x00/b") \
         .replace("<i>", "\x00i").replace("</i>", "\x00/i")
    s = s.replace("<", "&lt;").replace(">", "&gt;")
    return s.replace("\x00b", "<b>").replace("\x00/b", "</b>").replace("\x00i", "<i>").replace("\x00/i", "</i>")


def _table(rows, font, body_style):
    cells = [[c.strip() for c in r.strip().strip("|").split("|")] for r in rows]
    cells = [r for r in cells if not all(re.fullmatch(r":?-{2,}:?", c) for c in r)]
    data = [[Paragraph(_clean(c), body_style) for c in r] for r in cells]
    t = Table(data, hAlign="LEFT")
    t.setStyle(TableStyle([
        ("FONTNAME", (0, 0), (-1, -1), font),
        ("BACKGROUND", (0, 0), (-1, 0), HexColor("#f3efe0")),
        ("GRID", (0, 0), (-1, -1), 0.4, HexColor("#cfcfcf")),
        ("VALIGN", (0, 0), (-1, -1), "MIDDLE"),
        ("TOPPADDING", (0, 0), (-1, -1), 3), ("BOTTOMPADDING", (0, 0), (-1, -1), 3),
        ("LEFTPADDING", (0, 0), (-1, -1), 5), ("RIGHTPADDING", (0, 0), (-1, -1), 5),
    ]))
    return t


def md_to_flowables(md, font, styles):
    """styles: dict with keys h, body, quote."""
    out, table_buf = [], []
    lines = md.replace("\r", "").split("\n")

    def flush_table():
        nonlocal table_buf
        if table_buf:
            out.append(_table(table_buf, font, styles["body"]))
            out.append(Spacer(1, 4))
            table_buf = []

    for raw in lines:
        line = raw.rstrip()
        if line.lstrip().startswith("|"):
            table_buf.append(line)
            continue
        flush_table()
        s = line.strip()
        if not s or re.fullmatch(r"-{3,}", s):
            continue
        if s.startswith("#"):
            out.append(Paragraph(_clean(s.lstrip("#").strip()), styles["h"]))
        elif re.match(r"^\d+\.\s", s) and len(s) < 60:
            out.append(Paragraph(_clean(s), styles["h"]))
        elif s.startswith(">"):
            out.append(Paragraph(_clean(s.lstrip("> ").strip()), styles["quote"]))
        elif s.startswith(("- ", "• ", "* ")):
            out.append(Paragraph("• " + _clean(s[2:]), styles["body"]))
        elif s.startswith("→") or s.startswith("->"):
            out.append(Paragraph("<i>" + _clean(s) + "</i>", styles["body"]))
        else:
            out.append(Paragraph(_clean(s), styles["body"]))
    flush_table()
    return out
