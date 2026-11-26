"""Compact procedure illustrations for the conference poster."""

import json
from pathlib import Path

import matplotlib.image as mimage
from matplotlib.colors import to_rgb
from matplotlib.patches import Circle, FancyArrowPatch, Polygon, Rectangle
import numpy as np

DATA = Path(__file__).resolve().parent / "data"
INK, MUTED = "#22272D", "#66717A"
BLUE, TEAL, PURPLE, ORANGE = "#3F7895", "#458579", "#8B729E", "#B47C53"


def _canvas(ax, height):
    ax.set(xlim=(0, 335), ylim=(height, 0), aspect="equal")
    ax.set_facecolor("white")
    ax.axis("off")


def _text(ax, x, y, value, size=20, color=INK, ha="left", va="center", bold=False):
    return ax.text(x, y, value, fontsize=size, color=color, ha=ha, va=va,
                   weight="semibold" if bold else "normal", linespacing=1.15)


def _line(ax, points, color=MUTED, width=1):
    points = np.asarray(points)
    ax.plot(points[:, 0], points[:, 1], color=color, lw=width, zorder=2)


def _arrow(ax, start, end, color=MUTED, width=1.3):
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=13,
                                lw=width, color=color, shrinkA=1, shrinkB=1,
                                zorder=3))


def _rect(ax, x, y, width, height, color="white", edge="none", lw=1):
    ax.add_patch(Rectangle((x, y), width, height, fc=color, ec=edge, lw=lw))


def _molecule(ax, x, y, scale=1):
    compound = json.loads((DATA / "caffeine.json").read_text())
    coords = np.asarray(compound["xy"]) * [scale, -scale] + [x, y]
    for start, end, order in compound["bonds"]:
        a, b = coords[start], coords[end]
        unit = (b - a) / np.linalg.norm(b - a)
        offset = .06 * scale * np.array([-unit[1], unit[0]])
        shifts = [-offset, offset] if order == 2 else [np.zeros(2)]
        for shift in shifts:
            _line(ax, [a + shift, b + shift], INK, 1.1)
    for (xx, yy), symbol in zip(coords, compound["symbols"]):
        if symbol in ("N", "O"):
            ax.text(xx, yy, symbol, fontsize=16, color=INK, ha="center", va="center",
                    bbox={"fc": "white", "ec": "none", "pad": .15}, zorder=5)


def _network(ax, x, y, color):
    layers = [[(x, y - 3), (x, y + 3)],
              [(x + 10, y - 6), (x + 10, y), (x + 10, y + 6)],
              [(x + 20, y - 3), (x + 20, y + 3)]]
    for first, second in zip(layers, layers[1:]):
        for a in first:
            for b in second:
                _line(ax, [a, b], color, .7)
    for layer in layers:
        for point in layer:
            ax.add_patch(Circle(point, 1.35, fc="white", ec=color, lw=1, zorder=4))


def _gene_row(ax, x, y, values, color, width=76, height=7):
    rgb = np.asarray(to_rgb(color))
    step = width / len(values)
    for j, value in enumerate(values):
        shade = rgb * value + (1 - value) * np.ones(3)
        _rect(ax, x + j * step, y, step - 1.1, height, shade)


def draw_pairing(ax):
    """Draw replicate averaging on a 335 × 50 mm canvas."""
    _canvas(ax, 50)
    image = mimage.imread(DATA / "cultures.png")
    h, w = image.shape[:2]
    control = image[int(h * .15):int(h * .97), :w // 2]
    image_width = 50
    image_height = image_width * control.shape[0] / control.shape[1]
    ax.imshow(control, extent=(10, 10 + image_width, 3 + image_height, 3),
              interpolation="antialiased", zorder=1)
    _text(ax, 36, 45, "DMSO controls", 20, ha="center")
    _arrow(ax, (73, 29), (101, 29), TEAL)
    _text(ax, 150, 4, "Replicate profiles", 20, ha="center")
    values = np.array([[.25, .75, .5, .3], [.35, .65, .6, .45], [.3, .7, .4, .3]])
    for i, row in enumerate(values):
        _text(ax, 108, 17 + i * 9, f"r{i + 1}", 17, MUTED, ha="right")
        _gene_row(ax, 114, 13 + i * 9, row, TEAL, width=75, height=7)
    _arrow(ax, (194, 29), (249, 29), TEAL)
    _text(ax, 221, 19, "Mean", 20, TEAL, ha="center")
    _text(ax, 293, 7, "Mean DMSO\nexpression", 20, ha="center", bold=True)
    _gene_row(ax, 256, 25, values.mean(axis=0), TEAL, width=76, height=8)
    for x, width in [(114, 75), (256, 76)]:
        for j, label in enumerate(["g₁", "g₂", "…", "g₉₇₈"]):
            _text(ax, x + (j + .5) * width / 4, 44, label, 17, MUTED, ha="center")


def draw_encoding(ax):
    """Draw four separate predictors on a 335 × 100 mm canvas."""
    _canvas(ax, 100)
    _molecule(ax, 25, 47, scale=5.4)
    _text(ax, 25, 71, "SMILES", 20, ha="center")
    _text(ax, 25, 81, "Caffeine", 17, MUTED, ha="center")
    _text(ax, 25, 88, "(example)", 16, MUTED, ha="center")
    _arrow(ax, (46, 47), (62, 47))
    _line(ax, [(62, 21), (62, 84)], MUTED)
    _text(ax, 196, 3, "DMSO expression", 19, TEAL, ha="center")
    _line(ax, [(196, 10), (196, 87)], TEAL, 1.2)
    _text(ax, 307, 3, "978 genes", 18, MUTED, ha="center")
    rows = [(21, BLUE, "FCFP4", "1,024 feature bits", "MLP"),
            (42, TEAL, "ECFP4", "1,024 atom bits", "MLP"),
            (63, PURPLE, "ChemicalVAE", "196 frozen values", "MLP"),
            (84, ORANGE, "FCFP4 × dose", "Dose-scaled bits", "PRnet")]
    _rect(ax, 76, 32, 259, 20, "#F0F7F5", TEAL, .8)
    for y, color, name, detail, model in rows:
        _arrow(ax, (62, y), (76, y), color)
        vals = [0, 1, 1, 0, 1, 0]
        if name == "ChemicalVAE":
            vals = [.3, .8, .45, .6, .25, .7]
        if model == "PRnet":
            vals = [.0, .65, .65, .0, .65, .0]
        _gene_row(ax, 79, y - 5, vals[:3], color, width=17, height=4.4)
        _gene_row(ax, 79, y + .5, vals[3:], color, width=17, height=4.4)
        _text(ax, 101, y - 3.6, name, 21, color, bold=True)
        _text(ax, 101, y + 4.4, detail, 18, MUTED)
        _arrow(ax, (174, y), (208, y), color)
        ax.add_patch(Circle((196, y), 1.3, fc=TEAL, ec="white", lw=.4, zorder=5))
        if model == "PRnet":
            ax.add_patch(Polygon([(212, y - 6), (221, y - 2), (221, y + 2), (212, y + 6)],
                                 fc="#F4ECE6", ec=color, lw=1.2))
            ax.add_patch(Polygon([(225, y - 2), (234, y - 6), (234, y + 6), (225, y + 2)],
                                 fc="#F4ECE6", ec=color, lw=1.2))
            _arrow(ax, (221, y), (225, y), color, .8)
        else:
            _network(ax, 212, y, color)
        _text(ax, 241, y, model, 20, color, bold=True)
        _arrow(ax, (269, y), (285, y), color)
        for j, label in enumerate(["ŷ₁", "…", "ŷ₉₇₈"]):
            _rect(ax, 287 + 15 * j, y - 5, 14, 10, edge=color, lw=1)
            _text(ax, 294 + 15 * j, y, label, 17, color, ha="center")

    _text(ax, 166, 96, "ECFP4 selected by recorded validation; all four models fitted", 17, MUTED, ha="center")

def draw_training(ax):
    """Summarize representation selection and tuning from recorded validation."""
    _canvas(ax, 50)
    _text(ax, 167.5, 4, "Recorded validation results; same training cohort", 18, MUTED, ha="center")
    boxes = [(4, 91, "Three MLPs", BLUE),
             (121, 94, "Select ECFP4", TEAL),
             (248, 83, "Lower learning rate", INK)]
    for x, width, label, color in boxes:
        _rect(ax, x, 13, width, 15, edge=color, lw=1.2)
        _text(ax, x + width / 2, 20.5, label, 20, color, ha="center", bold=True)
    _arrow(ax, (96, 20.5), (119, 20.5))
    _arrow(ax, (216, 20.5), (246, 20.5))
    _text(ax, 49, 34, "Default validation NLL", 17, MUTED, ha="center")
    _text(ax, 169, 34, "Lowest NLL: 0.0846", 17, MUTED, ha="center")
    _text(ax, 289, 34, "0.001 → 0.0003", 17, MUTED, ha="center")
    _text(ax, 167.5, 46, "Tuned ECFP4 NLL: 0.0837; compare with retrained PRnet", 17, MUTED, ha="center")
