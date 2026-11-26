"""A0 poster presenting the shared pipeline and completed benchmark."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figures
from workflow import draw_workflow

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1189, 841
INK = "#22272D"
MUTED = "#596A73"
GRID = "#D5DFE2"


def main():
    figures.setup()
    data = figures.load()
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=65, facecolor="white")
    page = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    page.axis("off")

    def box(x, y, w, h, color):
        page.add_patch(Rectangle((x, y), w, h, fc=color, ec="none"))

    def text(x, y, value, size=24, color=INK, bold=False, ha="left", lines=1.2):
        return page.text(x, y, value, fontsize=size, color=color,
                         weight="semibold" if bold else "normal",
                         va="top", ha=ha, linespacing=lines)

    def axes(x, y, w, h):
        return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])

    def section(x, y, width, label):
        box(x, y, width, 19, INK)
        text(x + 7, y + 4, label, 26, "white", True)

    box(0, 0, WIDTH, 120, INK)
    text(24, 14, "Predicting gene-expression responses", 67, "white", True)
    text(24, 44, "to unseen chemical compounds", 67, "white", True)
    text(26, 82, "Yernur Kenzhegazin  ·  Alan Berdigaliyev  ·  Hugo Telles", 29, "white", True)
    text(26, 101, "M2 GENIOMHE-AI     |     Supervisor: Amaury Giard, IBISC", 23, "#E0E9EB")
    logo = plt.imread(ROOT / "data/ibisc.png")
    box(947, 23, 218, 69, "white")
    page.imshow(logo, extent=(955, 1157, 33 + 202 * logo.shape[0] / logo.shape[1], 33),
                aspect="auto", zorder=2)
    text(1159, 101, "Bulk LINCS L1000 + independent GEO studies", 20, "#E0E9EB", ha="right")

    text(26, 132, "Does molecular structure improve prediction beyond the measured control state?", 31, INK, True)
    text(26, 149, "One common dataset, three chemical encodings and a retrained PRnet comparator.", 23, MUTED)

    workflow_ax = axes(24, 166, 1141, 418.4)
    draw_workflow(workflow_ax, scale=2.49, title=False)

    section(24, 595, 366, "A  |  Linear baseline leads")
    section(410, 595, 366, "B  |  Mean response transfers best")
    section(796, 595, 369, "C  |  External evidence is limited")
    text(30, 620, "Distant compounds · 474 compounds / 16,531 profiles", 19, MUTED)
    text(416, 620, "Unseen SKBR3 · 137 compounds / 574 profiles", 19, MUTED)
    text(802, 620, "Seven GEO studies · six gene directions per study", 19, MUTED)

    a = axes(146, 642, 232, 108)
    figures.benchmark_axis(a, data, "internal_low_similarity", baselines=True, size=1.8, annotate=True)
    b = axes(532, 642, 232, 108)
    figures.benchmark_axis(b, data, "skbr3", baselines=True, size=1.8, annotate=True)
    c = axes(920, 642, 245, 108)
    figures.study_axis(c, data, size=1.8)

    text(30, 776, "A–B: lower MSE is better. Circles: seeds; diamond: mean; line: seed range. Dashed: unchanged control.", 18, MUTED)
    text(802, 776, "C: mean correct directions / 6; three training seeds.", 18, MUTED)
    box(24, 792, 1141, 29, "#EEF3F4")
    text(32, 797, "Chemical MLPs have lower MSE than PRnet here; simple baselines remain strongest.", 28, INK, True)
    text(32, 810, "External differences remain uncertain; only 15/42 labels show clear changes. Reused test sets make these findings exploratory.", 20, MUTED)
    text(24, 829, "Qi et al., Nature Communications (2024), doi:10.1038/s41467-024-53457-1  ·  Wei et al., Nature Methods (2026), doi:10.1038/s41592-025-02980-0", 15, MUTED)
    text(1165, 829, "Culture dishes are illustrative; all result plots use project measurements.", 15, MUTED, ha="right")

    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bad = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bb = artist.get_window_extent(renderer)
        if bb.x0 < -2 or bb.y0 < -2 or bb.x1 > fig.bbox.width + 2 or bb.y1 > fig.bbox.height + 2:
            bad.append(artist.get_text())
    if bad:
        raise RuntimeError("Text outside page: " + repr(bad))
    for ext in ["pdf", "svg"]:
        fig.savefig(ROOT / f"Conference_Poster.{ext}", facecolor="white", dpi=220)
    fig.savefig(ROOT / "Conference_Poster_preview.png", facecolor="white", dpi=65)
    plt.close(fig)
    print(ROOT / "Conference_Poster.pdf")


if __name__ == "__main__":
    main()
