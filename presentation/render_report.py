"""Render the shared report content as a readable PDF and standalone HTML."""

import html
import json
from pathlib import Path

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (
    SimpleDocTemplate,
    Paragraph,
    Spacer,
    PageBreak,
    Table,
    TableStyle,
    Preformatted,
    Image,
    Flowable,
    KeepTogether,
)
from PIL import Image as PILImage
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parent
PROJECT = ROOT.parent
WIDTH, HEIGHT = A4
CONTENT_WIDTH = WIDTH - 88
INK = colors.HexColor("#22323B")
TEAL = colors.HexColor("#087F7A")
LIGHT = colors.HexColor("#F0F5F6")


def main():
    pdfmetrics.registerFont(TTFont("Inter", str(ROOT / "data/Inter-Regular.ttf")))
    pdfmetrics.registerFont(TTFont("InterBold", str(ROOT / "data/Inter-SemiBold.ttf")))
    pdfmetrics.registerFontFamily(
        "Inter",
        normal="Inter",
        bold="InterBold",
        italic="Inter",
        boldItalic="InterBold",
    )
    styles = {
        "title": ParagraphStyle(
            "title",
            fontName="InterBold",
            fontSize=22,
            leading=27,
            textColor=INK,
            spaceAfter=15,
        ),
        "subtitle": ParagraphStyle(
            "subtitle",
            fontName="Inter",
            fontSize=9,
            leading=13,
            textColor=TEAL,
            spaceAfter=12,
        ),
        "paragraph": ParagraphStyle(
            "paragraph",
            fontName="Inter",
            fontSize=9.6,
            leading=13.8,
            textColor=INK,
            spaceAfter=9,
            allowOrphans=0,
            allowWidows=0,
        ),
        "heading": ParagraphStyle(
            "heading",
            fontName="InterBold",
            fontSize=12,
            leading=16,
            textColor=TEAL,
            spaceBefore=7,
            spaceAfter=8,
            keepWithNext=True,
        ),
        "caption": ParagraphStyle(
            "caption",
            fontName="Inter",
            fontSize=8,
            leading=11,
            textColor=colors.HexColor("#596B75"),
            spaceAfter=12,
        ),
        "cell": ParagraphStyle(
            "cell", fontName="Inter", fontSize=8.1, leading=11, textColor=INK
        ),
        "tablehead": ParagraphStyle(
            "tablehead",
            fontName="InterBold",
            fontSize=8.1,
            leading=11,
            textColor=colors.white,
        ),
        "code": ParagraphStyle(
            "code", fontName="Courier", fontSize=8, leading=11, textColor=INK
        ),
    }
    document = json.loads((PROJECT / "report_content.json").read_text())
    flow = []
    sections = []
    placements = []

    class VectorFigure(Flowable):
        def __init__(self, path, width, height):
            super().__init__()
            self.path = path
            self.width = width
            self.height = height
            self.hAlign = "CENTER"

        def draw(self):
            x, y = self.canv.absolutePosition(0, 0)
            placements.append(
                (
                    self.canv.getPageNumber() - 1,
                    self.path,
                    x,
                    y,
                    self.width,
                    self.height,
                )
            )

    for index, section in enumerate(document["pages"]):
        if index:
            flow.append(PageBreak())
        flow.append(Paragraph(f"RESEARCH REPORT / {index+1:02d}", styles["subtitle"]))
        flow.append(Paragraph(section["title"], styles["title"]))
        if section["subtitle"]:
            flow.append(Paragraph(section["subtitle"], styles["subtitle"]))
        markup = [
            f'<section><h1>{html.escape(section["title"])}</h1>',
            f'<p class="subtitle">{html.escape(section["subtitle"])}</p>',
        ]
        for block in section["blocks"]:
            kind = block["type"]
            if kind in ["paragraph", "heading"]:
                flow.append(Paragraph(block["text"], styles[kind]))
                tag = "h2" if kind == "heading" else "p"
                markup.append(f'<{tag}>{block["text"]}</{tag}>')
            elif kind == "table":
                data = [
                    [
                        Paragraph(html.escape(str(v)), styles["tablehead"])
                        for v in block["headers"]
                    ]
                ]
                data += [
                    [Paragraph(html.escape(str(v)), styles["cell"]) for v in row]
                    for row in block["rows"]
                ]
                n = len(block["headers"])
                widths = [CONTENT_WIDTH / n] * n
                if n == 4:
                    widths = [
                        CONTENT_WIDTH * 0.28,
                        CONTENT_WIDTH * 0.21,
                        CONTENT_WIDTH * 0.21,
                        CONTENT_WIDTH * 0.30,
                    ]
                table = Table(data, colWidths=widths, repeatRows=1, hAlign="LEFT")
                table.setStyle(
                    TableStyle(
                        [
                            ("BACKGROUND", (0, 0), (-1, 0), INK),
                            ("VALIGN", (0, 0), (-1, -1), "TOP"),
                            ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.white, LIGHT]),
                            ("LEFTPADDING", (0, 0), (-1, -1), 7),
                            ("RIGHTPADDING", (0, 0), (-1, -1), 7),
                            ("TOPPADDING", (0, 0), (-1, -1), 4),
                            ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
                            (
                                "LINEBELOW",
                                (0, -1),
                                (-1, -1),
                                0.5,
                                colors.HexColor("#CAD7DC"),
                            ),
                        ]
                    )
                )
                flow.extend([table, Spacer(1, 12)])
                markup.append(
                    "<table><thead><tr>"
                    + "".join(
                        f"<th>{html.escape(str(v))}</th>" for v in block["headers"]
                    )
                    + "</tr></thead><tbody>"
                )
                markup += [
                    "<tr>"
                    + "".join(f"<td>{html.escape(str(v))}</td>" for v in row)
                    + "</tr>"
                    for row in block["rows"]
                ]
                markup.append("</tbody></table>")
            elif kind == "figure":
                path = PROJECT / block["file"]
                vector = path.with_suffix(".pdf")
                source = PdfReader(vector).pages[0]
                w, h = float(source.mediabox.width), float(source.mediabox.height)
                scale = min(CONTENT_WIDTH / w, block["height"] / h)
                flow.append(
                    KeepTogether(
                        [
                            VectorFigure(vector, w * scale, h * scale),
                            Paragraph(html.escape(block["caption"]), styles["caption"]),
                        ]
                    )
                )
                markup.append(
                    f'<figure><img src="{block["file"]}" alt="{html.escape(block["caption"],quote=True)}"><figcaption>{html.escape(block["caption"])}</figcaption></figure>'
                )
            elif kind == "code":
                flow.extend(
                    [Preformatted(block["text"], styles["code"]), Spacer(1, 10)]
                )
                markup.append(f'<pre>{html.escape(block["text"])}</pre>')
        markup.append("</section>")
        sections.append("\n".join(markup))

    def footer(canvas, doc):
        canvas.saveState()
        canvas.setStrokeColor(colors.HexColor("#CAD7DC"))
        canvas.line(44, 38, WIDTH - 44, 38)
        canvas.setFont("Inter", 8)
        canvas.setFillColor(colors.HexColor("#596B75"))
        canvas.drawString(
            44, 25, "M2 GENIOMHE-AI  |  Chemical perturbation response prediction"
        )
        canvas.drawRightString(WIDTH - 44, 25, str(doc.page))
        canvas.restoreState()

    SimpleDocTemplate(
        str(PROJECT / "Research_Report.pdf"),
        pagesize=A4,
        leftMargin=44,
        rightMargin=44,
        topMargin=42,
        bottomMargin=53,
        title=document["title"],
        author="Yernur Kenzhegazin; Alan Berdigaliyev; Hugo Telles",
    ).build(flow, onFirstPage=footer, onLaterPages=footer)
    report_path = PROJECT / "Research_Report.pdf"
    reader = PdfReader(report_path)
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    for page_no, path, x, y, width, height in placements:
        source = PdfReader(path).pages[0]
        factor = width / float(source.mediabox.width)
        writer.pages[page_no].merge_transformed_page(
            source, Transformation().scale(factor).translate(x, y)
        )
    writer.add_metadata(
        {
            "/Title": document["title"],
            "/Author": "Yernur Kenzhegazin; Alan Berdigaliyev; Hugo Telles",
        }
    )
    with report_path.open("wb") as stream:
        writer.write(stream)
    css = """@font-face{font-family:Inter;src:url(presentation/data/Inter-Regular.ttf)}body{font:16px/1.6 Inter,Arial,sans-serif;color:#22323b;background:#edf2f4;margin:0}main{max-width:1000px;margin:auto;background:white;padding:48px}section{padding:12px 0 40px;border-bottom:1px solid #cad7dc}h1{font-size:30px;line-height:1.25}h2{font-size:20px;color:#087f7a}table{border-collapse:collapse;width:100%;font-size:14px;margin:20px 0}th,td{text-align:left;padding:10px;vertical-align:top}th{background:#22323b;color:white}tr:nth-child(even){background:#f0f5f6}img{max-width:100%}figure{margin:20px 0}figcaption,.subtitle{font-size:13px;color:#596b75}pre{background:#f0f5f6;padding:16px;overflow:auto}@media print{body{background:white}main{padding:0}section{break-before:page}h1,h2{break-after:avoid}table,figure{break-inside:avoid}}"""
    htmltext = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>'
        + html.escape(document["title"])
        + "</title><style>"
        + css
        + "</style><main>"
        + "".join(sections)
        + "</main></html>"
    )
    (PROJECT / "Research_Report.html").write_text(htmltext)
    print(PROJECT / "Research_Report.pdf")


if __name__ == "__main__":
    main()
