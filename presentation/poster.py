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

    box(0, 0, WIDTH, 101, INK)
    text(24, 10, "Predicting gene-expression responses", 64, "white", True)
    text(24, 37, "to unseen chemical compounds", 64, "white", True)
    text(26, 70, "Yernur Kenzhegazin  ·  Alan Berdigaliyev  ·  Hugo Telles", 28, "white", True)
    text(26, 87, "M2 GENIOMHE-AI     |     Supervisor: Amaury Giard, IBISC", 21, "#E0E9EB")
    logo = plt.imread(ROOT / "data/ibisc.png")
    box(947, 17, 218, 69, "white")
    page.imshow(logo, extent=(955, 1157, 27 + 202 * logo.shape[0] / logo.shape[1], 27),
                aspect="auto", zorder=2)

    text(24, 111, "Introduction", 28, INK, True)
    text(24, 128,
         "Chemical compounds can change a cell’s gene expression.\n"
         "Predicting this response could help prioritize experiments.\n"
         "However, low error may reflect the control state rather than\n"
         "a learned chemical effect, making simple baselines essential [2].",
         23, INK, lines=1.25)
    text(620, 111, "Does chemical structure add predictive value?", 28, INK, True)
    text(620, 128,
         "We use LINCS L1000 bulk expression data to predict 978 genes\n"
         "from chemical structure and matched control measurements.\n"
         "Three multilayer perceptrons (MLPs), retrained PRnet [1] and\n"
         "baselines are tested on new compounds, SKBR3 and GEO studies.",
         23, INK, lines=1.25)

    workflow_ax = axes(83, 178, 1023, 375.1)
    draw_workflow(workflow_ax, scale=2.23, title=False, schematic_note=False)

    section(24, 565, 366, "A  |  Linear baseline leads")
    section(410, 565, 366, "B  |  Mean response transfers best")
    section(796, 565, 369, "C  |  External evidence is limited")
    text(30, 591, "Distant compounds · 474 compounds / 16,531 profiles", 19, MUTED)
    text(416, 591, "Unseen SKBR3 · 137 compounds / 574 profiles", 19, MUTED)
    text(802, 591, "Seven GEO studies · six gene directions per study", 19, MUTED)

    a = axes(146, 611, 232, 100)
    figures.benchmark_axis(a, data, "internal_low_similarity", baselines=True, size=1.8, annotate=True)
    b = axes(532, 611, 232, 100)
    figures.benchmark_axis(b, data, "skbr3", baselines=True, size=1.8, annotate=True)
    c = axes(920, 611, 245, 100)
    figures.study_axis(c, data, size=1.8)

    text(30, 740,
         "MSE measures expression-prediction error: lower is better.\n"
         "The linear baseline leads (0.635); all chemical MLPs have\n"
         "lower error than PRnet, but do not outperform this baseline.",
         20, INK, lines=1.25)
    text(416, 740,
         "SKBR3 was excluded from training. Adding the training\n"
         "mean response to measured controls gives the lowest\n"
         "error (0.551), below every neural network in this test.",
         20, INK, lines=1.25)
    text(802, 740,
         "Each cell shows correct up/down predictions out of six,\n"
         "averaged over three seeds; darker means more correct.\n"
         "These studies show no clear advantage over PRnet.",
         20, INK, lines=1.25)
    text(30, 778, "A–B: circles = seeds; diamond = mean; line = seed range. Dashed: unchanged control.", 17, MUTED)
    text(802, 778, "Only 15 of 42 observed gene directions show clear changes.", 17, MUTED)
    box(24, 792, 1141, 29, "#EEF3F4")
    text(32, 797, "Chemical MLPs have lower MSE than PRnet here; simple baselines remain strongest.", 28, INK, True)
    text(32, 810, "Beating PRnet here does not establish added value from chemistry. One held-out cell line, a small external panel and reused tests limit generalization.", 20, MUTED)
    text(24, 829, "[1] Qi et al., Nature Communications (2024), doi:10.1038/s41467-024-53457-1", 16, MUTED)
    text(620, 829, "[2] Wei et al., Nature Methods (2026), doi:10.1038/s41592-025-02980-0", 16, MUTED)

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
