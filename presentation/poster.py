"""A0 conference poster with illustrated methods and separate result panels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figures
from poster_methods import draw_pairing, draw_encoding, draw_training

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1189, 841
INK, MUTED, RULE = "#242426", "#54595D", "#D2D2D2"


def noise_axis(ax, data):
    rows = data["noise"].copy()
    rows["increase"] = rows.mse - rows.clean_mse
    for model in figures.MODELS:
        summary = rows.loc[rows.model.eq(model)].groupby("level").increase.agg(
            ["mean", "min", "max"])
        ax.plot(summary.index, summary["mean"], "o-", lw=2.3, ms=6,
                color=figures.COLORS[model], label=figures.NAMES[model])
        ax.fill_between(summary.index, summary["min"], summary["max"],
                        color=figures.COLORS[model], alpha=.12)
    ax.axhline(0, color="#999999", lw=1)
    ax.set_xticks([0, .25, .5, 1])
    ax.set_xlabel("Added noise / control replicate SD", fontsize=21)
    ax.set_ylabel("Increase in SKBR3 MSE", fontsize=21)
    ax.tick_params(labelsize=20)
    ax.grid(color="#E8E8E8", lw=.8)
    ax.legend(frameon=False, loc="upper left", fontsize=18, ncol=2,
              columnspacing=.8, handlelength=1.5)


def main():
    figures.setup()
    figures.COLORS.update(zip(figures.MODELS,
                              ["#3F7895", "#458579", "#8B729E", "#B47C53", "#53575A"]))
    data = figures.load()
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=65, facecolor="white")
    page = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    page.axis("off")

    def box(x, y, w, h, color):
        page.add_patch(Rectangle((x, y), w, h, fc=color, ec="none"))

    def text(x, y, value, size=24, color=INK, bold=False, ha="left", lines=1.17):
        return page.text(x, y, value, fontsize=size, color=color,
                         weight="semibold" if bold else "normal",
                         va="top", ha=ha, linespacing=lines)

    def axes(x, y, w, h):
        return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])

    def section(x, y, w, label):
        box(x, y, w, 21, INK)
        text(x + w / 2, y + 4, label, 29, "white", True, "center")

    def subtitle(x, y, w, label):
        box(x, y, w, 16, INK)
        text(x + 5, y + 2.5, label, 23, "white", True)

    box(0, 0, WIDTH, 101, INK)
    text(24, 10, "Predicting gene-expression responses", 64, "white", True)
    text(24, 37, "to unseen chemical compounds", 64, "white", True)
    text(26, 70, "Yernur Kenzhegazin  ·  Alan Berdigaliyev  ·  Hugo Telles", 28, "white", True)
    text(26, 87, "M2 GENIOMHE-AI     |     Supervisor: Amaury Giard, IBISC", 21, "#ECECEC")
    logo = plt.imread(ROOT / "data/ibisc.png")
    box(947, 17, 218, 69, "white")
    page.imshow(logo, extent=(955, 1157, 27 + 202 * logo.shape[0] / logo.shape[1], 27),
                aspect="auto", zorder=2)
    page.plot([384, 384], [119, 798], color=RULE, lw=1)
    page.plot([782, 782], [150, 798], color=RULE, lw=1)

    section(24, 119, 346, "BACKGROUND")
    text(30, 151,
         "Predicting gene-expression responses could help\n"
         "prioritize compounds for experimental testing.\n"
         "We compare three chemical multilayer perceptrons\n"
         "(MLPs) with retrained PRnet [1] and simple baselines,\n"
         "asking whether chemistry adds predictive value [2].", 24)
    cultures = plt.imread(ROOT / "data/cultures.png")
    page.imshow(cultures, extent=(77, 317, 325, 205), aspect="auto", zorder=2)
    text(137, 326, "DMSO control", 24, "#3F7895", True, "center")
    text(137, 338, "Baseline input", 21, MUTED, ha="center")
    text(257, 326, "Treated cells", 24, "#B47C53", True, "center")
    text(257, 338, "Measured target", 21, MUTED, ha="center")

    section(24, 360, 346, "METHODS")
    subtitle(24, 391, 346, "1  Pair treatments with measured controls")
    text(30, 415,
         "LINCS L1000: 978 measured genes, 10 µM, 24 h.\n"
         "Match controls by plate, cell, time and bead batch;\n"
         "average replicate expression for each gene.", 23)
    draw_pairing(axes(30, 451, 335, 50))

    subtitle(24, 512, 346, "2  Compare three chemical representations")
    text(30, 536,
         "FCFP4: functional classes; ECFP4: atom environments.\n"
         "ChemicalVAE: frozen SMILES features. Each MLP also\n"
         "receives controls; PRnet uses dose-scaled FCFP4.", 23)
    draw_encoding(axes(30, 575, 335, 100))

    subtitle(24, 695, 346, "3  Fit on training data; select on validation")
    text(30, 719,
         "Use compound-disjoint splits. Select settings and\n"
         "checkpoints by validation negative log likelihood\n"
         "(NLL), then repeat with three training seeds.", 23)
    draw_training(axes(30, 754, 335, 43))

    section(398, 119, 767, "RESULTS")
    subtitle(398, 151, 371, "A  Distant compounds: a linear model leads")
    text(404, 177, "474 compounds · 16,531 profiles", 23, MUTED)
    text(404, 190, "Nearest-training ECFP4 Tanimoto ≤ 0.7", 21, MUTED)
    a = axes(528, 215, 231, 154)
    figures.benchmark_axis(a, data, "internal_low_similarity", baselines=True,
                           size=2.2, annotate=True)
    text(404, 397,
         "Lower MSE means closer predicted expression.\n"
         "The linear baseline has the lowest error (0.635).\n"
         "All chemical MLPs have lower MSE than PRnet, but none\n"
         "improves on this simple baseline.", 24)

    subtitle(398, 454, 371, "B  Unseen cell line: the mean response leads")
    text(404, 480, "SKBR3 · 574 profiles · 137 compounds", 23, MUTED)
    text(404, 493, "Cell line excluded from model training", 21, MUTED)
    b = axes(528, 518, 231, 154)
    figures.benchmark_axis(b, data, "skbr3", baselines=True, size=2.2, annotate=True)
    text(404, 700,
         "Adding the training mean response to measured\n"
         "controls gives the lowest error (0.551), below every\n"
         "neural network. Accurate compound prediction\n"
         "does not guarantee transfer to a new cell context.", 24)
    text(404, 755,
         "A–B: circles = training seeds; diamond = mean;\n"
         "line = seed range. Dashed line = unchanged control.", 20, MUTED)

    subtitle(796, 151, 369, "C  External studies: evidence remains limited")
    text(802, 177, "Seven GEO studies · six gene directions per study", 23, MUTED)
    c = axes(922, 203, 243, 146)
    figures.study_axis(c, data, size=2.2)
    text(802, 380,
         "Values count correct up/down predictions out of six,\n"
         "averaged over three seeds; darker means more correct.\n"
         "The chemical MLPs show no clear advantage over PRnet.\n"
         "Only 15 of 42 observed directions show clear changes.", 23)

    subtitle(796, 431, 369, "D  Control noise increases prediction error")
    d = axes(837, 463, 319, 127)
    noise_axis(d, data)
    text(802, 618,
         "Gaussian noise is added to the same SKBR3 controls;\n"
         "models and treated targets stay fixed. Curves show\n"
         "the increase over each model’s clean-input MSE.\n"
         "At 1 replicate SD, chemical-MLP error rises by 0.33–0.34.", 23)
    text(802, 665,
         "20 draws per nonzero noise level; 3 seeds. Shading: seed range.\n"
         "PRnet changes less, but its clean-input error is higher (B).", 19, MUTED)

    section(796, 695, 369, "CONCLUSIONS")
    text(802, 727,
         "Chemical MLPs have lower MSE than PRnet here,\n"
         "but simple baselines remain strongest.\n"
         "Added predictive value from chemistry is not established.", 24, bold=True)
    text(802, 768,
         "One held-out cell line, a small external panel and\n"
         "reused test sets make these findings exploratory.", 21, MUTED)

    page.plot([24, 1165], [810, 810], color=RULE, lw=1)
    text(24, 820, "REFERENCES", 19, INK, True)
    text(106, 820, "[1] Qi et al. Nature Communications (2024). doi:10.1038/s41467-024-53457-1", 17, MUTED)
    text(676, 820, "[2] Wei et al. Nature Methods (2026). doi:10.1038/s41592-025-02980-0", 17, MUTED)

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
