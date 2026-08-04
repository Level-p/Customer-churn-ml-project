# -*- coding: utf-8 -*-
"""Render content.py into the IEEE two-column Word template.

Successor to build.py, which assumed a POSIX box: it shelled out to `zip`, read
its template from /tmp, and could place exactly one figure. This one runs on
Windows, uses Python's own zipfile, takes the template from the .docx sitting
next to it, and handles an arbitrary number of figures.

Two figures in this paper are dashboard screenshots. A screenshot reduced to the
3.15 in column of a two-column layout puts its body text at roughly four points,
so those are marked ``wide`` in content.py and are emitted inside a one-column
continuous section that spans the page. A section break in OOXML applies to
everything *before* it, so a wide figure is written as: close the running
two-column section on the preceding paragraph, emit the figure and its caption,
then close that one-column section on the caption itself.

    python build_paper.py
"""

from __future__ import annotations

import os
import re
import shutil
import zipfile
from pathlib import Path

from PIL import Image

from content import (ABSTRACT, AFFIL, AUTHOR, CONTENT, FIGURES, KEYWORDS, REFS,
                     SUBTITLE, TABLES, TITLE)

HERE = Path(__file__).resolve().parent
TEMPLATE = HERE / "ziCvcsAL"                       # the blank IEEE A4 template
STAGE = HERE / "_build"
OUT = HERE.parent / "Onoja_Churn_Prediction_Interpretable_ML.docx"

EMU_PER_INCH = 914400
COL_WIDTH_IN = 2.92      # slightly inside the 3.15 in column: at eight pages the
                         # plots stay legible a little smaller, and five figures
                         # of slack adds up to most of a column of text
PAGE_WIDTH_IN = 6.10     # both columns, inset slightly: a page-spanning figure
                         # that is a shade narrower than the measure fits its
                         # caption on the same page instead of pushing itself
                         # over and leaving half a page blank behind it
MAX_HEIGHT_IN = 8.20     # leave room for the caption on a full page

# Page geometry, shared by every section: A4, narrow margins, as the template.
PAGE = ('<w:pgSz w:w="11906" w:h="16838"/>'
        '<w:pgMar w:left="734" w:right="734" w:gutter="0" w:header="0" '
        'w:top="1080" w:footer="0" w:bottom="2434"/>')
TAIL = ('<w:formProt w:val="false"/><w:textDirection w:val="lrTb"/>'
        '<w:docGrid w:type="default" w:linePitch="360" w:charSpace="0"/>')
TWO_COLUMN = '<w:cols w:num="2" w:space="360" w:equalWidth="true" w:sep="false"></w:cols>'
ONE_COLUMN = '<w:cols w:num="1" w:space="360"></w:cols>'


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


# --------------------------------------------------------------------------- #
# Citations: {{key}} and {{key1,key2}} become [n] in order of first appearance
# --------------------------------------------------------------------------- #

order: list[str] = []


def cite_repl(match: re.Match) -> str:
    out = []
    for key in (k.strip() for k in match.group(1).split(",")):
        if key not in REFS:
            raise KeyError(f"unknown ref key: {key}")
        if key not in order:
            order.append(key)
        out.append(f"[{order.index(key) + 1}]")
    return ", ".join(out)


def resolve(text: str) -> str:
    return re.sub(r"\{\{([^}]+)\}\}", cite_repl, text)


# --------------------------------------------------------------------------- #
# Paragraph, run, equation and table builders
# --------------------------------------------------------------------------- #

TNR = "Times New Roman"


def run(text: str, bold=False, italic=False, vert=None, font=None) -> str:
    rpr = ""
    if font:
        rpr += f'<w:rFonts w:ascii="{font}" w:hAnsi="{font}" w:cs="{font}"/>'
    if bold:
        rpr += "<w:b/>"
    if italic:
        rpr += "<w:i/>"
    if vert:
        rpr += f'<w:vertAlign w:val="{vert}"/>'
    rpr = f"<w:rPr>{rpr}</w:rPr>" if rpr else ""
    return f'<w:r>{rpr}<w:t xml:space="preserve">{esc(text)}</w:t></w:r>'


def para(style: str, runs_xml: str, extra_ppr: str = "") -> str:
    return (f'<w:p><w:pPr><w:pStyle w:val="{style}"/>{extra_ppr}</w:pPr>'
            f"{runs_xml}</w:p>")


def simple(style: str, text: str, extra_ppr: str = "") -> str:
    return para(style, run(text), extra_ppr)


def section(columns: str) -> str:
    """A continuous section break with the given column count."""
    return (f'<w:sectPr><w:type w:val="continuous"/>{PAGE}{columns}{TAIL}'
            f"</w:sectPr>")


def break_para(columns: str) -> str:
    """An all-but-invisible paragraph whose only job is to carry a section break.

    The break could be hung on the preceding paragraph instead, and the first
    version of this script did that. Word then treats that paragraph's last line
    as though the text continued, and justifies it: the closing line of a
    paragraph, or of a figure caption, comes out stretched across the measure
    with word-sized gaps. Hanging the break on a two-point empty paragraph
    costs a fraction of a line and leaves the real text alone.
    """
    return ('<w:p><w:pPr><w:spacing w:before="0" w:after="0" w:line="20" '
            'w:lineRule="exact"/><w:ind w:firstLine="0"/>'
            '<w:rPr><w:sz w:val="2"/><w:szCs w:val="2"/></w:rPr>'
            f"{section(columns)}</w:pPr></w:p>")


def eq_para(parts, number: int) -> str:
    runs = [f'<w:r><w:rPr><w:rFonts w:ascii="{TNR}" w:hAnsi="{TNR}" '
            f'w:cs="{TNR}"/></w:rPr><w:tab/></w:r>']
    for text, fmt in parts:
        italic = fmt in ("i", "isub", "isup")
        vert = ("subscript" if fmt in ("sub", "isub")
                else "superscript" if fmt in ("sup", "isup") else None)
        runs.append(run(text, italic=italic, vert=vert, font=TNR))
    runs.append(f'<w:r><w:rPr><w:rFonts w:ascii="{TNR}" w:hAnsi="{TNR}" '
                f'w:cs="{TNR}"/></w:rPr><w:tab/>'
                f'<w:t xml:space="preserve">({number})</w:t></w:r>')
    return para("equation", "".join(runs))


BORDERS = ("<w:tblBorders>"
           + "".join(f'<w:{edge} w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
                     for edge in ("top", "start", "bottom", "end", "insideH", "insideV"))
           + "</w:tblBorders>")


def table_xml(spec: dict) -> str:
    widths = spec["widths"]
    grid = "".join(f'<w:gridCol w:w="{w}"/>' for w in widths)
    out = [f'<w:tbl><w:tblPr><w:tblW w:w="{sum(widths)}" w:type="dxa"/>'
           f'<w:jc w:val="center"/>{BORDERS}<w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="28" w:type="dxa"/>'
           '<w:start w:w="57" w:type="dxa"/><w:bottom w:w="28" w:type="dxa"/>'
           '<w:end w:w="57" w:type="dxa"/></w:tblCellMar></w:tblPr>'
           f"<w:tblGrid>{grid}</w:tblGrid>"]

    def row(cells, header=False) -> str:
        parts = ["<w:tr>"]
        if header:
            parts.append("<w:trPr><w:tblHeader/></w:trPr>")
        for width, cell in zip(widths, cells):
            style = "tablecolhead" if header else "tablecopy"
            ppr = '<w:jc w:val="center"/>' if header else ""
            parts.append(f'<w:tc><w:tcPr><w:tcW w:w="{width}" w:type="dxa"/>'
                         f'<w:vAlign w:val="center"/></w:tcPr>'
                         f"{para(style, run(cell), ppr)}</w:tc>")
        parts.append("</w:tr>")
        return "".join(parts)

    out.append(row(spec["header"], header=True))
    out.extend(row(cells) for cells in spec["rows"])
    out.append("</w:tbl>")
    return "".join(out)


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #

media: list[tuple[str, Path]] = []   # (relationship id, source path)


def drawing(rel_id: str, name: str, cx: int, cy: int, doc_id: int) -> str:
    # keepNext binds the image to its caption, so a figure near a page boundary
    # moves as a unit instead of leaving its caption stranded overleaf.
    return (
        '<w:p><w:pPr><w:keepNext/><w:spacing w:lineRule="auto" w:line="240" '
        'w:before="120" w:after="80"/><w:ind w:firstLine="0"/>'
        '<w:jc w:val="center"/></w:pPr>'
        '<w:r><w:drawing xmlns:wp="http://schemas.openxmlformats.org/drawingml/'
        '2006/wordprocessingDrawing"><wp:inline distT="0" distB="0" distL="0" '
        f'distR="0"><wp:extent cx="{cx}" cy="{cy}"/>'
        '<wp:effectExtent l="0" t="0" r="0" b="0"/>'
        f'<wp:docPr id="{doc_id}" name="{name}"/>'
        '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
        '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
        f'<pic:nvPicPr><pic:cNvPr id="{doc_id}" name="{name}"/><pic:cNvPicPr/></pic:nvPicPr>'
        f'<pic:blipFill><a:blip r:embed="{rel_id}" xmlns:r="http://schemas.'
        'openxmlformats.org/officeDocument/2006/relationships"/>'
        '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
        f'<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="{cx}" cy="{cy}"/></a:xfrm>'
        '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
        "</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>"
    )


def figure_block(key: str, spec: dict, index: int) -> tuple[str, bool]:
    """Return (xml, is_wide) for one figure and its caption."""
    source = HERE / spec["file"]
    if not source.exists():
        raise FileNotFoundError(source)
    with Image.open(source) as image:
        width_px, height_px = image.size

    wide = bool(spec.get("wide"))
    target_in = PAGE_WIDTH_IN if wide else COL_WIDTH_IN
    # Never let a figure grow taller than the text block it has to sit in.
    target_in = min(target_in, MAX_HEIGHT_IN * width_px / height_px)
    cx = int(round(target_in * EMU_PER_INCH))
    cy = int(round(cx * height_px / width_px))

    rel_id = f"rIdFig{index}"
    media.append((rel_id, source))
    xml = drawing(rel_id, f"Figure{index}", cx, cy, 100 + index)
    # No "Fig. n." prefix here: the template's figurecaption style carries a
    # numbering definition whose level text is literally "Fig. %1.", so Word
    # supplies the number. Adding one by hand would print it twice.
    # keepLines stops a caption breaking over a column or page boundary; with
    # keepNext on the image above, the pair migrates together.
    xml += simple("figurecaption", spec["caption"], "<w:keepLines/>")
    return xml, wide


# --------------------------------------------------------------------------- #
# Assemble the document body
# --------------------------------------------------------------------------- #


def build_body() -> str:
    body: list[str] = []

    # Title block: its own single-column section on the first page.
    body.append(simple("papertitle", TITLE))
    body.append(simple("papersubtitle", SUBTITLE))
    body.append(simple("Author", AUTHOR))
    for line in AFFIL:
        body.append(simple("Affiliation", line))
    body.append(
        '<w:p><w:pPr><w:pStyle w:val="Affiliation"/><w:sectPr>'
        f'<w:type w:val="nextPage"/>{PAGE}'
        f'<w:pgNumType w:fmt="decimal"/>{TAIL}</w:sectPr></w:pPr></w:p>'
    )

    body.append(simple("Abstract", resolve(ABSTRACT)))
    body.append(simple("keywords", KEYWORDS))

    equation_no = 0
    figure_no = 0
    for item in CONTENT:
        kind = item[0]
        if kind == "h1":
            body.append(simple("Heading1", item[1]))
        elif kind == "h2":
            body.append(simple("Heading2", item[1]))
        elif kind == "h5":
            body.append(simple("Heading5", item[1]))
        elif kind == "body":
            body.append(simple("BodyText", resolve(item[1])))
        elif kind == "bullet":
            body.append(simple("bulletlist", resolve(item[1])))
        elif kind == "eq":
            equation_no += 1
            body.append(eq_para(item[1], equation_no))
        elif kind in ("table", "tablewide"):
            spec = TABLES[item[1]]
            block = (simple("tablehead", spec["caption"],
                            "<w:keepNext/><w:keepLines/>") + table_xml(spec))
            body.append(wrap_wide(block) if kind == "tablewide" else block)
        elif kind in ("fig", "figwide"):
            figure_no += 1
            spec = dict(FIGURES[item[1]])
            if kind == "figwide":
                spec["wide"] = True
            xml, wide = figure_block(item[1], spec, figure_no)
            body.append(wrap_wide(xml) if wide else xml)
        else:
            raise ValueError(f"unknown content item: {kind}")

    body.append(simple("Heading5", "References"))
    for key in order:
        body.append(simple("references", REFS[key]))
    return "".join(body)


def wrap_wide(block: str) -> str:
    """Put one block of content in a one-column section that spans the page.

    A section break in OOXML applies to everything *before* it, so spanning the
    page means: end the running two-column section, emit the block, end the
    one-column section that now contains it. Whatever follows falls into the
    next section, which is the two-column one closing the document.
    """
    return break_para(TWO_COLUMN) + block + break_para(ONE_COLUMN)


# --------------------------------------------------------------------------- #
# Package
# --------------------------------------------------------------------------- #


def write_package(document_xml: str) -> None:
    if STAGE.exists():
        shutil.rmtree(STAGE)
    STAGE.mkdir()
    with zipfile.ZipFile(TEMPLATE) as template:
        template.extractall(STAGE)

    original = (STAGE / "word/document.xml").read_text(encoding="utf-8")
    head = original[: original.index("<w:body>") + len("<w:body>")]
    full = head + document_xml + section(TWO_COLUMN) + "</w:body></w:document>"
    (STAGE / "word/document.xml").write_text(full, encoding="utf-8")

    media_dir = STAGE / "word/media"
    media_dir.mkdir(parents=True, exist_ok=True)
    rels_path = STAGE / "word/_rels/document.xml.rels"
    rels = rels_path.read_text(encoding="utf-8")
    additions = []
    for rel_id, source in media:
        target = f"fig_{rel_id}{source.suffix}"
        shutil.copy(source, media_dir / target)
        additions.append(
            f'<Relationship Id="{rel_id}" Type="http://schemas.openxmlformats.'
            f'org/officeDocument/2006/relationships/image" '
            f'Target="media/{target}"/>'
        )
    rels_path.write_text(
        rels.replace("</Relationships>", "".join(additions) + "</Relationships>"),
        encoding="utf-8",
    )

    ct_path = STAGE / "[Content_Types].xml"
    content_types = ct_path.read_text(encoding="utf-8")
    if 'Extension="png"' not in content_types:
        content_types = content_types.replace(
            "</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
        ct_path.write_text(content_types, encoding="utf-8")

    if OUT.exists():
        OUT.unlink()
    with zipfile.ZipFile(OUT, "w", zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(STAGE.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(STAGE).as_posix())
    shutil.rmtree(STAGE)


def main() -> None:
    document = build_body()
    write_package(document)
    print(f"wrote {OUT}  ({OUT.stat().st_size / 1024:.0f} KB)")
    print(f"figures: {len(media)}   tables: {len(TABLES)}   references: {len(order)}")
    unused = sorted(set(REFS) - set(order))
    if unused:
        print("uncited references:", ", ".join(unused))


if __name__ == "__main__":
    main()
