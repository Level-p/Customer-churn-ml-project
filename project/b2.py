# -*- coding: utf-8 -*-
import re, os, sys, shutil, subprocess
sys.path.insert(0, "/tmp")
from content import REFS, TITLE, SUBTITLE, AUTHOR, AFFIL, ABSTRACT, KEYWORDS, CONTENT, TABLES

TPL = "/tmp/tplx"
OUT_DIR = "/tmp/paperx"
OUT_DOCX = "/tmp/Onoja_Churn_Prediction_Interpretable_ML.docx"

def esc(t):
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")

# ---- citation numbering ----
order = []
def cite_repl(m):
    keys = [k.strip() for k in m.group(1).split(",")]
    outs = []
    for k in keys:
        if k not in REFS:
            raise KeyError("unknown ref key: " + k)
        if k not in order:
            order.append(k)
        outs.append("[%d]" % (order.index(k) + 1))
    return ", ".join(outs)

def resolve(text):
    return re.sub(r"\{\{([^}]+)\}\}", cite_repl, text)

# ---- run/paragraph builders ----
def run(text, bold=False, italic=False, vert=None, font=None):
    rpr = ""
    if font:
        rpr += '<w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s"/>' % (font, font, font)
    if bold: rpr += "<w:b/>"
    if italic: rpr += "<w:i/>"
    if vert: rpr += '<w:vertAlign w:val="%s"/>' % vert
    rpr = "<w:rPr>%s</w:rPr>" % rpr if rpr else ""
    return '<w:r>%s<w:t xml:space="preserve">%s</w:t></w:r>' % (rpr, esc(text))

def para(style, runs_xml, extra_ppr=""):
    return '<w:p><w:pPr><w:pStyle w:val="%s"/>%s</w:pPr>%s</w:p>' % (style, extra_ppr, runs_xml)

def simple(style, text):
    return para(style, run(text))

TNR = "Times New Roman"
def eq_para(parts, num):
    runs = ['<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s"/></w:rPr><w:tab/></w:r>' % (TNR, TNR, TNR)]
    for text, fmt in parts:
        italic = fmt in ("i", "isub", "isup")
        vert = "subscript" if fmt in ("sub", "isub") else ("superscript" if fmt in ("sup", "isup") else None)
        runs.append(run(text, italic=italic, vert=vert, font=TNR))
    runs.append('<w:r><w:rPr><w:rFonts w:ascii="%s" w:hAnsi="%s" w:cs="%s"/></w:rPr><w:tab/><w:t xml:space="preserve">(%d)</w:t></w:r>' % (TNR, TNR, TNR, num))
    return para("equation", "".join(runs))

def table_xml(spec):
    widths = spec["widths"]
    total = sum(widths)
    borders = ('<w:tblBorders><w:top w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '<w:start w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '<w:bottom w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '<w:end w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '<w:insideH w:val="single" w:sz="4" w:space="0" w:color="000000"/>'
               '<w:insideV w:val="single" w:sz="4" w:space="0" w:color="000000"/></w:tblBorders>')
    out = ['<w:tbl><w:tblPr><w:tblW w:w="%d" w:type="dxa"/><w:jc w:val="center"/>%s'
           '<w:tblLayout w:type="fixed"/>'
           '<w:tblCellMar><w:top w:w="28" w:type="dxa"/><w:start w:w="57" w:type="dxa"/>'
           '<w:bottom w:w="28" w:type="dxa"/><w:end w:w="57" w:type="dxa"/></w:tblCellMar>'
           '</w:tblPr><w:tblGrid>%s</w:tblGrid>'
           % (total, borders, "".join('<w:gridCol w:w="%d"/>' % w for w in widths))]
    def row(cells, header=False):
        tr = ["<w:tr>"]
        if header:
            tr.append("<w:trPr><w:tblHeader/></w:trPr>")
        for w, c in zip(widths, cells):
            style = "tablecolhead" if header else "tablecopy"
            ppr = '<w:jc w:val="center"/>' if header else ""
            tr.append('<w:tc><w:tcPr><w:tcW w:w="%d" w:type="dxa"/><w:vAlign w:val="center"/></w:tcPr>%s</w:tc>'
                      % (w, para(style, run(c), ppr)))
        tr.append("</w:tr>")
        return "".join(tr)
    out.append(row(spec["header"], header=True))
    for r in spec["rows"]:
        out.append(row(r))
    out.append("</w:tbl>")
    return "".join(out)

# ---- assemble body ----
body = []

# Title block (section 1, single column)
body.append(simple("papertitle", TITLE))
body.append(simple("papersubtitle", SUBTITLE))
body.append(simple("Author", AUTHOR))
for i, line in enumerate(AFFIL):
    body.append(simple("Affiliation", line))
sect1 = ('<w:sectPr><w:type w:val="nextPage"/><w:pgSz w:w="11906" w:h="16838"/>'
         '<w:pgMar w:left="734" w:right="734" w:gutter="0" w:header="0" w:top="1080" w:footer="0" w:bottom="2434"/>'
         '<w:pgNumType w:fmt="decimal"/><w:formProt w:val="false"/><w:textDirection w:val="lrTb"/>'
         '<w:docGrid w:type="default" w:linePitch="360" w:charSpace="0"/></w:sectPr>')
body.append('<w:p><w:pPr><w:pStyle w:val="Affiliation"/>%s</w:pPr></w:p>' % sect1)

# Abstract + keywords
body.append(simple("Abstract", resolve(ABSTRACT)))
body.append(simple("keywords", KEYWORDS))

eq_no = 0
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
        eq_no += 1
        body.append(eq_para(item[1], eq_no))
    elif kind == "table":
        spec = TABLES[item[1]]
        body.append(simple("tablehead", spec["caption"]))
        body.append(table_xml(spec))
    elif kind == "fig":
        CX = 2880360  # 3.15 in
        CY = CX * 1080 // 990
        drawing = (
            '<w:p><w:pPr><w:spacing w:lineRule="auto" w:line="240" w:before="120" w:after="80"/><w:ind w:firstLine="0"/><w:jc w:val="center"/></w:pPr>'
            '<w:r><w:drawing xmlns:wp="http://schemas.openxmlformats.org/drawingml/2006/wordprocessingDrawing">'
            '<wp:inline distT="0" distB="0" distL="0" distR="0">'
            '<wp:extent cx="%d" cy="%d"/><wp:effectExtent l="0" t="0" r="0" b="0"/>'
            '<wp:docPr id="10" name="Figure1"/>'
            '<a:graphic xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main">'
            '<a:graphicData uri="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:pic xmlns:pic="http://schemas.openxmlformats.org/drawingml/2006/picture">'
            '<pic:nvPicPr><pic:cNvPr id="10" name="Figure1"/><pic:cNvPicPr/></pic:nvPicPr>'
            '<pic:blipFill><a:blip r:embed="rIdFig1" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships"/>'
            '<a:stretch><a:fillRect/></a:stretch></pic:blipFill>'
            '<pic:spPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="%d" cy="%d"/></a:xfrm>'
            '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom></pic:spPr>'
            '</pic:pic></a:graphicData></a:graphic></wp:inline></w:drawing></w:r></w:p>'
            % (CX, CY, CX, CY))
        body.append(drawing)
        body.append(simple("figurecaption",
            "System architecture of the proposed framework. Each layer consumes the outputs of the layer above it, "
            "ending in prioritised retention recommendations delivered through the dashboard."))
    else:
        raise ValueError(kind)

# References
body.append(simple("Heading5", "References"))
for k in order:
    body.append(simple("references", REFS[k]))

sect_body = ('<w:sectPr><w:type w:val="continuous"/><w:pgSz w:w="11906" w:h="16838"/>'
             '<w:pgMar w:left="734" w:right="734" w:gutter="0" w:header="0" w:top="1080" w:footer="0" w:bottom="2434"/>'
             '<w:cols w:num="2" w:space="360" w:equalWidth="true" w:sep="false"></w:cols>'
             '<w:formProt w:val="false"/><w:textDirection w:val="lrTb"/>'
             '<w:docGrid w:type="default" w:linePitch="360" w:charSpace="0"/></w:sectPr>')

# ---- write package ----
orig = open(os.path.join(TPL, "word/document.xml"), encoding="utf8").read()
head = orig[: orig.index("<w:body>") + len("<w:body>")]
new_xml = head + "".join(body) + sect_body + "</w:body></w:document>"

if os.path.exists(OUT_DIR):
    shutil.rmtree(OUT_DIR)
shutil.copytree(TPL, OUT_DIR)
open(os.path.join(OUT_DIR, "word/document.xml"), "w", encoding="utf8").write(new_xml)

# figure media + relationship + content type
os.makedirs(os.path.join(OUT_DIR, "word/media"), exist_ok=True)
shutil.copy("/tmp/framework.png", os.path.join(OUT_DIR, "word/media/framework.png"))
relp = os.path.join(OUT_DIR, "word/_rels/document.xml.rels")
rels = open(relp, encoding="utf8").read()
if "rIdFig1" not in rels:
    rels = rels.replace("</Relationships>",
        '<Relationship Id="rIdFig1" Type="http://schemas.openxmlformats.org/officeDocument/2006/relationships/image" '
        'Target="media/framework.png"/></Relationships>')
    open(relp, "w", encoding="utf8").write(rels)
ctp = os.path.join(OUT_DIR, "[Content_Types].xml")
ct = open(ctp, encoding="utf8").read()
if 'Extension="png"' not in ct:
    ct = ct.replace("</Types>", '<Default Extension="png" ContentType="image/png"/></Types>')
    open(ctp, "w", encoding="utf8").write(ct)

if os.path.exists(OUT_DOCX):
    os.remove(OUT_DOCX)
subprocess.run(["zip", "-Xrq", OUT_DOCX, "."], cwd=OUT_DIR, check=True)
print("written:", OUT_DOCX)
print("refs:", len(order))
