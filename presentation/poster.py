"""A0 landscape conference poster from frozen scientific results."""

from pathlib import Path
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrowPatch, Circle
import numpy as np
import pandas as pd
import figures
from pypdf import PdfReader, PdfWriter, Transformation

ROOT = Path(__file__).resolve().parent
WIDTH, HEIGHT = 1189, 841
INK = "#22272D"
MUTED = "#596A73"
TEAL = "#087F7A"
GRID = "#D5DFE2"


def main():
    figures.setup()
    data = figures.load()
    fig = plt.figure(figsize=(WIDTH / 25.4, HEIGHT / 25.4), dpi=65, facecolor="white")
    page = fig.add_axes([0, 0, 1, 1], xlim=(0, WIDTH), ylim=(HEIGHT, 0))
    page.axis("off")

    def box(x, y, w, h, c, edge="none", lw=1):
        patch = Rectangle((x, y), w, h, fc=c, ec=edge, lw=lw)
        page.add_patch(patch)
        return patch

    def text(x, y, t, size=24, color=INK, bold=False, ha="left", lines=1.15):
        return page.text(
            x,
            y,
            t,
            fontsize=size,
            color=color,
            weight="semibold" if bold else "normal",
            va="top",
            ha=ha,
            linespacing=lines,
        )

    def line(x, y, xx, yy, c=GRID, lw=1):
        page.plot([x, xx], [y, yy], color=c, lw=lw)

    def arrow(x, y, xx, yy, c=MUTED, lw=1.7):
        page.add_patch(
            FancyArrowPatch(
                (x, y), (xx, yy), arrowstyle="-|>", mutation_scale=17, color=c, lw=lw
            )
        )

    def section(x, y, w, label):
        box(x, y, w, 18, INK)
        text(x + 9, y + 4, label, 29, "white", True)

    def axes(x, y, w, h):
        return fig.add_axes([x / WIDTH, 1 - (y + h) / HEIGHT, w / WIDTH, h / HEIGHT])

    def network(x, y, color):
        coords = [
            [(x, y + 6), (x, y + 17)],
            [(x + 8, y + 2), (x + 8, y + 11.5), (x + 8, y + 21)],
            [(x + 16, y + 6), (x + 16, y + 17)],
        ]
        for a, b in zip(coords, coords[1:]):
            for aa in a:
                for bb in b:
                    line(*aa, *bb, c=color, lw=0.7)
        for layer in coords:
            for xy in layer:
                page.add_patch(Circle(xy, 1.5, fc="white", ec=color, lw=1))

    def expression(x, y, color):
        vals = [0.2, 0.8, 0.35, 0.65, 0.45, 0.9, 0.1, 0.7]
        base = np.array(plt.matplotlib.colors.to_rgb(color))
        for j, v in enumerate(vals):
            col = base * v + np.ones(3) * (1 - v)
            box(x + j * 3.3, y, 2.7, 16, col)

    box(0, 0, WIDTH, 137, INK)
    text(24, 18, "Predicting gene-expression responses", 76, "white", True)
    text(24, 53, "to unseen chemical compounds", 76, "white", True)
    text(
        26,
        98,
        "Yernur Kenzhegazin  ·  Alan Berdigaliyev  ·  Hugo Telles",
        31,
        "white",
        True,
    )
    text(
        26,
        116,
        "M2 GENIOMHE-AI     |     Supervisor: Amaury Giard, IBISC",
        25,
        "#E0E9EB",
    )
    logo = plt.imread(ROOT / "data/ibisc.png")
    box(947, 24, 218, 72, "white")
    page.imshow(
        logo,
        extent=(955, 1157, 36 + 202 * logo.shape[0] / logo.shape[1], 36),
        aspect="auto",
        zorder=2,
    )
    text(
        1159,
        112,
        "Bulk LINCS L1000 + independent GEO studies",
        22,
        "#E0E9EB",
        ha="right",
    )

    # Left column: context, original schematic and shared evaluation protocol.
    section(24, 156, 300, "BACKGROUND  |  What does chemistry add?")
    text(
        30,
        187,
        "We test whether molecular structure adds predictive\nvalue beyond measured control expression.",
        23,
        lines=1.2,
    )
    text(30, 211, "PUBLISHED MOTIVATION  |  Single-cell benchmark", 19, TEAL, True)
    lit_x, lit_y, lit_w = 94, 222, 160
    lit_h = lit_w * 95 / 122
    published = plt.imread(ROOT / "data/wei_fig5e_chemical.png")
    literature_image = page.imshow(
        published, extent=(lit_x, lit_x + lit_w, lit_y + lit_h, lit_y), aspect="auto"
    )
    text(
        30,
        349,
        "Wei et al. (2026), Fig. 5e (right), cropped.\nPublished sci-Plex3/A549 results; separate from our data.",
        15,
        MUTED,
        lines=1.15,
    )
    section(24, 365, 300, "METHODS  |  Three encodings, one task")
    text(
        30,
        393,
        "Three MLPs share controls, hidden layers and targets;\nthe chemical representation and input width can differ.",
        22,
        lines=1.15,
    )
    text(154, 416, "DMSO expression", 17, MUTED, ha="center")
    text(263, 416, "978 genes", 17, MUTED, ha="center")
    line(154, 422, 154, 612, c="#8799A1", lw=1.4)
    for i, (label, detail, key) in enumerate(
        [
            ("FCFP4", "1,024 bits", "ResidualFCFP4"),
            ("ECFP4", "1,024 bits", "Residual"),
            ("ChemicalVAE", "196 values", "ResidualChemicalVAE"),
        ]
    ):
        y = 432 + i * 43
        c = figures.COLORS[key]
        box(31, y, 107, 31, "#F5F7F8", c, 1.2)
        text(38, y + 4, label, 27, c, True)
        text(38, y + 18, detail, 18, MUTED)
        arrow(138, y + 15.5, 164, y + 15.5, c)
        arrow(154, y + 15.5, 164, y + 15.5, "#8799A1", 1)
        box(165, y, 65, 31, "white", c, 1.2)
        network(172, y + 3, c)
        text(195, y + 9, "MLP", 26, c, True)
        arrow(231, y + 15.5, 247, y + 15.5, c)
        expression(250, y + 7, c)
    box(30, 560, 285, 17, "white")
    text(31, 563, "512 → 256 hidden units · predicted change + control", 20, MUTED)
    y = 588
    c = figures.COLORS["PRnet"]
    box(31, y, 107, 31, "#F5F7F8", c, 1.2)
    text(38, y + 5, "FCFP4 × dose", 25, c, True)
    arrow(138, y + 15, 164, y + 15, c)
    box(165, y, 65, 31, "white", c, 1.2)
    text(197.5, y + 8, "PRnet", 28, c, True, ha="center")
    arrow(231, y + 15, 247, y + 15, c)
    expression(250, y + 7, c)
    text(
        31, 625, "PRnet adds Gaussian noise. All predict 978 treated genes.", 19, MUTED
    )
    section(24, 649, 300, "EVALUATION  |  Shared held-out data")
    text(
        30,
        677,
        "210,438 train · 25,190 validation · 27,765 test\n10 µM · 24 h · 978 measured landmark genes\nCompound-disjoint splits; ECFP4 similarity ≤0.7\nto external compounds (Tanimoto)\nValidation NLL selects settings and checkpoints",
        21,
        lines=1.25,
    )
    for x, label, scenario in [
        (24, "Unseen compounds", "compound"),
        (178, "Unseen cell line", "cell"),
    ]:
        box(x, 741, 146, 66, "#F1F5F6")
        text(x + 7, 748, label, 21, INK, True)
        for row in range(3):
            for col in range(5):
                held = (col >= 3) if scenario == "compound" else (row == 2)
                box(
                    x + 8 + col * 8,
                    765 + row * 8,
                    7,
                    7,
                    "#B46F48" if held else "#A8C6D9",
                    "white",
                    0.4,
                )
        text(
            x + 53,
            766,
            (
                "New chemical\nstructures"
                if scenario == "compound"
                else "SKBR3 excluded\nfrom training"
            ),
            17,
            MUTED,
            lines=1.35,
        )
        text(
            x + 8,
            794,
            "Training data" if scenario == "compound" else "Held-out data",
            16,
            MUTED,
        )
        box(
            x + 65 if scenario == "compound" else x + 68,
            795,
            6,
            4,
            "#A8C6D9" if scenario == "compound" else "#B46F48",
        )
    text(28, 810, "Schematic: rows = cell lines; columns = compounds.", 15, MUTED)
    line(335, 156, 335, 807, c=GRID, lw=1.2)

    # The primary comparison uses validation-selected settings and all three seeds.
    section(
        347,
        156,
        818,
        "A  |  Simple baselines remain stronger than the chemical networks",
    )
    text(
        353,
        186,
        "Equal tuning budget · three seeds · low-similarity test: 474 compounds / 16,531 profiles",
        23,
        MUTED,
    )
    a = axes(474, 221, 262, 161)
    figures.benchmark_axis(
        a, data, "internal_low_similarity", baselines=True, size=2.0, annotate=True
    )
    a.set_title(
        "Low-similarity compounds", fontsize=26, loc="left", pad=14, weight="semibold"
    )
    b = axes(901, 221, 252, 161)
    figures.benchmark_axis(b, data, "skbr3", baselines=True, size=2.0, annotate=True)
    b.set_title(
        "Unseen cell line: SKBR3", fontsize=26, loc="left", pad=14, weight="semibold"
    )
    text(
        353,
        410,
        "Circles: training seeds; diamonds: mean; lines: seed range. Dashed lines: unchanged controls.",
        21,
        MUTED,
    )
    text(
        353,
        426,
        "Left: nearest training ECFP4 similarity ≤0.7. Right: 574 profiles / 137 compounds. Lower MSE is better.",
        21,
        MUTED,
    )

    section(347, 454, 397, "B  |  Test chemical dependence directly")
    text(353, 483, "SKBR3 · 20 within-control-group shuffles / seed", 21, MUTED)
    c = axes(483, 524, 248, 166)
    figures.shuffle_axis(c, data, "skbr3", size=1.95)
    text(
        353,
        719,
        "Positive values show a shuffling penalty. Input dependence\nis distinct from beating a trained control-only model.",
        21,
        lines=1.2,
    )

    section(765, 454, 400, "C  |  Control-only leads this small panel")
    text(771, 483, "10 µM · 24 h · correct directions out of six genes", 21, MUTED)
    d = axes(895, 522, 257, 176)
    figures.study_axis(d, data, size=1.95)
    text(
        771,
        719,
        "Seven studies; each value averages three training seeds.\nOnly 15 of 42 labels meet the clear-change sensitivity rule.",
        20,
        lines=1.2,
    )

    section(
        347,
        762,
        818,
        "CONCLUSION  |  Chemical dependence alone does not establish useful drug ranking",
    )
    text(
        353,
        790,
        "The molecular MLPs have lower MSE than PRnet, but simple baselines remain stronger on average.\nExternal direction differences remain uncertain. Reused test sets make this evidence exploratory.",
        23,
        lines=1.14,
    )
    line(24, 820, 1165, 820, c=INK, lw=1)
    text(
        24,
        827,
        "Wei et al., Nat Methods (2026) · doi:10.1038/s41592-025-02980-0 · Published inset: Fig. 5e, right",
        16,
        MUTED,
    )
    text(
        1165,
        827,
        "Project panels A–C: bulk LINCS / GEO · Full methods, noise tests and uncertainty in the report and notebook",
        15,
        MUTED,
        ha="right",
    )

    for ext in ["pdf", "svg"]:
        literature_image.set_visible(ext != "pdf")
        fig.savefig(ROOT / f"Conference_Poster.{ext}", facecolor="white", dpi=220)
    literature_image.set_visible(True)
    fig.savefig(ROOT / "Conference_Poster_preview.png", facecolor="white", dpi=65)
    # Preserve the published chart's original vector marks in the print PDF.
    pdf_path = ROOT / "Conference_Poster.pdf"
    reader = PdfReader(pdf_path)
    source = PdfReader(ROOT / "data/wei_fig5e_chemical.pdf").pages[0]
    factor = (lit_w * 72 / 25.4) / float(source.mediabox.width)
    transform = (
        Transformation()
        .scale(factor)
        .translate(lit_x * 72 / 25.4, (HEIGHT - lit_y - lit_h) * 72 / 25.4)
    )
    writer = PdfWriter()
    writer.add_page(reader.pages[0])
    writer.pages[0].merge_transformed_page(source, transform)
    writer.add_metadata(
        {
            "/Title": "Chemical perturbation response prediction",
            "/Subject": "Project results with attributed published motivation inset",
        }
    )
    with pdf_path.open("wb") as f:
        writer.write(f)
    # Check all drawn text stays inside the physical page.
    fig.canvas.draw()
    r = fig.canvas.get_renderer()
    bad = []
    for artist in fig.findobj(match=plt.Text):
        if not artist.get_visible() or not artist.get_text():
            continue
        bb = artist.get_window_extent(r)
        if (
            bb.x0 < -2
            or bb.y0 < -2
            or bb.x1 > fig.bbox.width + 2
            or bb.y1 > fig.bbox.height + 2
        ):
            bad.append(artist.get_text())
    if bad:
        raise RuntimeError("Text outside page: " + repr(bad))
    plt.close(fig)
    print(ROOT / "Conference_Poster.pdf")


if __name__ == "__main__":
    main()
