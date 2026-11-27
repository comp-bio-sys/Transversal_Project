"""A0 conference poster with illustrated methods and separate result panels."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import figures
from focus_figures import mse_bar_axis, external_bar_axis, noise_axis, selection_axis
from poster_methods import draw_pairing, draw_encoding, draw_training

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1189, 841
INK, MUTED, RULE = "#242426", "#54595D", "#D2D2D2"


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
         "We compare three chemical MLPs, retain ECFP4 by\n"
         "validation, then compare it with retrained PRnet [1].\n"
         "DMSO measurements anchor the response tests [2].", 24)
    cultures = plt.imread(ROOT / "data/cultures.png")
    page.imshow(cultures, extent=(77, 317, 325, 205), aspect="auto", zorder=2)
    text(137, 326, "DMSO control", 24, "#3F7895", True, "center")
    text(137, 338, "Baseline input", 21, MUTED, ha="center")
    text(257, 326, "Treated cells", 24, "#B47C53", True, "center")
    text(257, 338, "Measured target", 21, MUTED, ha="center")
    text(30, 351, "Figure 1. DMSO provides the reference; treated expression is the target.", 16, MUTED)

    section(24, 360, 346, "METHODS")
    subtitle(24, 391, 346, "1  Pair treatments with measured controls")
    text(30, 415,
         "LINCS L1000: 978 measured genes, 10 µM, 24 h.\n"
         "Match controls by plate, cell, time and bead batch;\n"
         "average replicate expression for each gene.", 23)
    draw_pairing(axes(30, 451, 335, 50))
    text(30, 503, "Figure 2a. Averaging matched DMSO replicates gives 978 input values.", 16, MUTED)

    subtitle(24, 512, 346, "2  Compare three chemical representations")
    text(30, 536,
         "FCFP4: functional classes; ECFP4: atom environments.\n"
         "ChemicalVAE: frozen SMILES features. All models use\n"
         "DMSO expression; PRnet uses dose-scaled FCFP4.", 23)
    draw_encoding(axes(30, 575, 335, 100))
    text(30, 680, "Figure 2b. Each chemical representation enters a separate predictor.", 16, MUTED)

    subtitle(24, 695, 346, "3  Select ECFP4; compare with PRnet")
    text(30, 719,
         "Use compound-disjoint splits and validation NLL.\n"
         "Select ECFP4; adjust its learning rate using validation.\n"
         "Compare the fitted models across three training seeds.", 23)
    draw_training(axes(30, 754, 335, 43))
    text(30, 800, "Figure 2c. Validation-based selection and ECFP4 learning-rate adjustment.", 16, MUTED)

    section(398, 119, 767, "RESULTS")
    subtitle(398, 151, 371, "A  Validation selects the ECFP4 MLP")
    text(404, 177, "Three default MLPs · seed 42 · lower NLL is better", 22, MUTED)
    a = axes(431, 205, 324, 120)
    selection_axis(a, data, size=2.2)
    text(404, 354,
         "Figure 3. Lowest validation NLL for each default-setting MLP.\n"
         "Each bar is one training run; no seed range is estimated.", 18, MUTED)
    text(404, 381, "ECFP4 learning rate: 0.001 → 0.0003", 24, bold=True)
    text(404, 397, "Validation NLL: 0.08465 → 0.08372", 23)
    text(404, 413, "SKBR3 MSE rises: 0.580 → 0.654 (three seeds).", 20)
    text(404, 429,
         "All encodings were tuned in parallel. The validation ranking\n"
         "identifies ECFP4 for this focused comparison.", 18, MUTED)

    subtitle(398, 454, 371, "B  Test the selected MLP against PRnet")
    text(431, 481, "Distant compounds", 23, bold=True)
    text(612, 481, "Unseen SKBR3", 23, bold=True)
    text(431, 496, "474 compounds / 16,531 profiles", 17, MUTED)
    text(612, 496, "137 compounds / 574 profiles", 17, MUTED)
    b1 = axes(431, 522, 141, 156)
    mse_bar_axis(b1, data, "internal_low_similarity", size=1.85)
    b2 = axes(612, 522, 141, 156)
    mse_bar_axis(b2, data, "skbr3", size=1.85)
    text(404, 709,
         "Figure 4. Treated-expression MSE; lower is better.\n"
         "Model bars: three-seed means; dots: runs; whiskers: seed range.\n"
         "The DMSO reference predicts the measured control expression.\n"
         "Distant compounds have nearest-training Tanimoto ≤ 0.7.", 18, MUTED)
    text(404, 751,
         "ECFP4 has lower MSE than PRnet in both tests.\n"
         "Unchanged DMSO is strongest on SKBR3.", 23)
    text(404, 781,
         "A validation gain alone does not establish better transfer.", 20, MUTED)

    subtitle(796, 151, 369, "C  External studies: evidence remains limited")
    text(802, 177, "Seven GEO studies · six gene directions per study", 23, MUTED)
    c = axes(832, 203, 324, 146)
    external_bar_axis(c, data, size=2.2)
    text(802, 380,
         "Figure 5. Correct directions across seven studies × six genes.\n"
         "Bars: mean; points: three runs; whiskers: run range.\n"
         "DMSO is the reference and does not make up/down predictions.\n"
         "Only 15 of the 42 observed directions show clear changes.", 18, MUTED)
    text(802, 415, "No clear external advantage over PRnet is established.", 23)

    subtitle(796, 431, 369, "D  Control noise: ECFP4 stays below PRnet")
    d = axes(837, 463, 319, 127)
    noise_axis(d, data, size=2.1)
    text(802, 618,
         "Figure 6. SKBR3 MSE after adding Gaussian noise to DMSO inputs.\n"
         "20 draws per nonzero level; noise scaled by control-replicate SD.\n"
         "Model curves: three-seed means; shading: seed ranges.\n"
         "The DMSO reference predicts the noise-perturbed control values.", 18, MUTED)
    text(802, 657,
         "At 1 replicate SD: ECFP4 MSE rises from 0.654 to 0.993;\n"
         "PRnet rises from 2.176 to 2.229. ECFP4 remains lower.", 22)

    section(796, 695, 369, "CONCLUSIONS")
    text(802, 727,
         "Selected ECFP4 has lower MSE than PRnet here.\n"
         "Unchanged DMSO is stronger on unseen SKBR3.\n"
         "Reliable external regulation prediction is not established.", 24, bold=True)
    text(802, 768,
         "One held-out cell line, a small external panel and\n"
         "reused test sets make these findings exploratory.", 21, MUTED)
    text(802, 795, "Additional linear/mean baselines perform better; see report supplement.", 16, MUTED)

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
