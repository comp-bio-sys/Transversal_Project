"""Export the full research report in GitHub-readable Markdown."""

import html
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parent.parent


def prose(value):
    text = str(value)
    text = re.sub(r'<a href="([^"]+)">(.+?)</a>', r"[\2](\1)", text)
    text = re.sub(r"<br\s*/?>", "  \n", text)
    for tag in ("b", "strong"):
        text = text.replace(f"<{tag}>", "**").replace(f"</{tag}>", "**")
    for tag in ("i", "em"):
        text = text.replace(f"<{tag}>", "*").replace(f"</{tag}>", "*")
    return html.unescape(text)


def cell(value):
    return prose(value).replace("|", "\\|").replace("\n", "<br>")


def main():
    document = json.loads((ROOT / "report_content.json").read_text())
    lines = [
        f'# {document["title"]}',
        "[Project overview](../README.md) · [PDF report](../Research_Report.pdf) · [Notebook](../Analysis.ipynb)",
    ]
    for page in document["pages"]:
        lines.append(f'## {page["title"]}')
        if page["subtitle"]:
            lines.append(prose(page["subtitle"]))
        for block in page["blocks"]:
            kind = block["type"]
            if kind == "paragraph":
                lines.append(prose(block["text"]))
            elif kind == "heading":
                lines.append("### " + prose(block["text"]))
            elif kind == "table":
                rows = ["| " + " | ".join(map(cell, block["headers"])) + " |"]
                rows.append("| " + " | ".join("---" for _ in block["headers"]) + " |")
                rows.extend(
                    "| " + " | ".join(map(cell, row)) + " |" for row in block["rows"]
                )
                lines.append("\n".join(rows))
            elif kind == "figure":
                lines.append(
                    f'![{Path(block["file"]).stem.replace("_", " ")}](../{block["file"]})'
                )
                lines.append(prose(block["caption"]))
            elif kind == "code":
                lines.append(f'```sh\n{block["text"]}\n```')
            else:
                raise ValueError(f"Unknown report block: {kind}")
    destination = ROOT / "docs/REPORT.md"
    destination.write_text("\n\n".join(lines) + "\n")
    print(destination)


if __name__ == "__main__":
    main()
