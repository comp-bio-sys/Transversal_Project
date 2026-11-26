"""Publication figures drawn from the completed benchmark tables."""

from pathlib import Path
import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Circle, Rectangle
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT.parent / "extension/balanced"
OUT = ROOT / "figures"
MODELS = ["ResidualFCFP4", "Residual", "ResidualChemicalVAE", "PRnet", "ControlOnly"]
NAMES = {
    "ResidualFCFP4": "FCFP4 MLP",
    "Residual": "ECFP4 MLP",
    "ResidualChemicalVAE": "ChemicalVAE MLP",
    "PRnet": "PRnet",
    "ControlOnly": "Control-only MLP",
    "DMSO": "Unchanged control",
    "GlobalMeanResponse": "Global mean response",
    "CellMeanResponse": "Cell mean response",
    "RidgeECFP4": "Linear ECFP4 + control",
    "Direct": "Direct / random",
    "DirectZeroInit": "Direct / zero",
    "ResidualRandomInit": "Residual / random",
}
COLORS = dict(zip(MODELS, ["#2878A5", "#078478", "#9268A1", "#BE6C37", "#4B545C"]))
COLORS.update(
    DMSO="#99A4AA",
    GlobalMeanResponse="#94A56C",
    CellMeanResponse="#627C39",
    RidgeECFP4="#B48A20",
)
INK = "#202A31"
MUTED = "#5B6A73"


def setup():
    for f in (ROOT / "data").glob("Inter-*.ttf"):
        font_manager.fontManager.addfont(f)
    plt.rcParams.update(
        {
            "font.family": "Inter",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.labelcolor": INK,
            "text.color": INK,
            "axes.edgecolor": "#BDCACF",
            "xtick.color": MUTED,
            "ytick.color": MUTED,
            "svg.fonttype": "none",
            "pdf.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )
    OUT.mkdir(exist_ok=True)


def save(fig, name):
    for ext in ["pdf", "svg", "png"]:
        fig.savefig(OUT / f"{name}.{ext}", dpi=220, bbox_inches="tight")
    plt.close(fig)


def load():
    suffix = "fixed_tuned_group_holdout_baselines"
    return {
        "summary": pd.read_csv(RESULTS / "summary/model_summary.csv"),
        "strict": pd.read_csv(RESULTS / "summary/low_similarity_seed_scores.csv"),
        "raw": pd.read_csv(RESULTS / f"results/summary_{suffix}.csv"),
        "external": pd.read_csv(RESULTS / "summary/external_seed_counts.csv"),
        "studies": pd.read_csv(RESULTS / "summary/external_by_study.csv"),
        "paired": pd.read_csv(RESULTS / "summary/paired_mse_intervals.csv"),
        "shuffle": pd.read_csv(RESULTS / "diagnostics/permutation_seed_means.csv"),
        "noise": pd.read_csv(RESULTS / "diagnostics/noise_seed_means.csv"),
    }


def benchmark_axis(
    ax, data, dataset, external=False, baselines=True, size=1.0, annotate=False
):
    models = MODELS + (
        ["GlobalMeanResponse", "CellMeanResponse", "RidgeECFP4"] if baselines else []
    )
    table = (
        data["external"]
        if external
        else (data["strict"] if dataset == "internal_low_similarity" else data["raw"])
    )
    table = table.loc[table.phase.isin(["tuned", "baseline"])]
    if not external:
        table = table.loc[table.dataset.eq(dataset)]
    for y, model in enumerate(models):
        values = table.loc[
            table.model.eq(model), "mean" if external else "mse"
        ].to_numpy()
        if not len(values):
            continue
        ax.plot([min(values), max(values)], [y, y], color=COLORS[model], lw=1.6 * size)
        ax.scatter(
            values,
            np.full(len(values), y),
            s=21 * size,
            facecolors="white",
            edgecolors=COLORS[model],
            zorder=3,
        )
        ax.scatter(
            values.mean(), y, marker="D", s=30 * size, color=COLORS[model], zorder=4
        )
    if not external:
        dmso = table.loc[table.model.eq("DMSO"), "mse"].iloc[0]
        ax.axvline(dmso, color=COLORS["DMSO"], ls="--", lw=1.2 * size)
    else:
        ax.set_xlim(0, 1)
    if annotate:
        lower, upper = ax.get_xlim()
        ax.set_xlim(lower, upper + 0.3 * (upper - lower))
        for y, model in enumerate(models):
            values = table.loc[table.model.eq(model), "mean" if external else "mse"]
            if len(values):
                ax.text(
                    0.98,
                    y,
                    f"{values.mean():.3f}",
                    transform=ax.get_yaxis_transform(),
                    ha="right",
                    va="center",
                    fontsize=9 * size,
                    color=COLORS[model],
                    bbox={"facecolor": "white", "edgecolor": "none", "pad": 1},
                )
    ax.set_yticks(range(len(models)), [NAMES[m] for m in models], fontsize=9.5 * size)
    ax.set_ylim(len(models) - 0.5, -0.5)
    ax.set_xlabel(
        "Correct directions / 42" if external else "Treated-expression MSE",
        fontsize=10 * size,
    )
    ax.tick_params(axis="x", labelsize=9 * size)
    ax.grid(axis="x", color="#E5EBED", zorder=0)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)


def study_axis(ax, data, size=1.0):
    studies = pd.read_csv(ROOT.parent / "dose_matched/study_manifest.csv")
    studies = studies.loc[studies.chemical_split.eq("eligible")]
    table = data["studies"].loc[data["studies"].phase.eq("tuned")]
    matrix = (
        table.groupby(["gse_id", "model"])
        .correct.mean()
        .unstack()
        .reindex(index=studies.gse_id, columns=MODELS)
        * 6
    )
    ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=6, aspect="auto")
    for i in range(len(matrix)):
        for j in range(len(MODELS)):
            value = matrix.iloc[i, j]
            ax.text(
                j,
                i,
                f"{value:.1f}",
                ha="center",
                va="center",
                color="white" if value > 3.7 else INK,
                fontsize=10 * size,
            )
    ax.set_xticks(
        range(5),
        ["FCFP4", "ECFP4", "Chemical\nVAE", "PRnet", "Control\nonly"],
        fontsize=9 * size,
    )
    ax.set_yticks(
        range(len(studies)),
        [f"{r.drug_name} / {r.cell}" for r in studies.itertuples()],
        fontsize=9 * size,
    )
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)


def shuffle_axis(ax, data, dataset="skbr3", size=1.0):
    rows = data["shuffle"].loc[data["shuffle"].dataset.eq(dataset)]
    for y, model in enumerate(MODELS):
        values = rows.loc[rows.model.eq(model), "mse_increase"]
        ax.scatter(
            values, np.full(len(values), y), color=COLORS[model], s=28 * size, alpha=0.8
        )
        ax.plot(
            [values.min(), values.max()], [y, y], color=COLORS[model], lw=1.4 * size
        )
    ax.axvline(0, color=MUTED, lw=1)
    ax.set_yticks(range(5), [NAMES[m] for m in MODELS], fontsize=9 * size)
    ax.set_ylim(4.5, -0.5)
    ax.set_xlabel("MSE increase after chemical shuffling", fontsize=10 * size)
    ax.tick_params(axis="x", labelsize=9 * size)
    ax.grid(axis="x", color="#E5EBED")


def noise_axis(ax, data, size=1.0):
    for model in MODELS + ["DMSO"]:
        rows = data["noise"].loc[data["noise"].model.eq(model)]
        means = rows.groupby("level").mse.mean()
        low = rows.groupby("level").mse.min()
        high = rows.groupby("level").mse.max()
        ax.plot(
            means.index,
            means,
            ".-",
            label=NAMES[model],
            color=COLORS[model],
            lw=1.5 * size,
            markersize=6 * size,
        )
        ax.fill_between(means.index, low, high, color=COLORS[model], alpha=0.10)
    ax.set_xlabel("Added noise / within-plate control SD", fontsize=10 * size)
    ax.set_ylabel("SKBR3 MSE", fontsize=10 * size)
    ax.set_xticks([0, 0.25, 0.5, 1])
    ax.tick_params(labelsize=9 * size)
    ax.grid(color="#E5EBED")


def pipeline():
    fig, ax = plt.subplots(figsize=(14, 7.8))
    ax.set(xlim=(0, 14), ylim=(0, 7.8))
    ax.axis("off")

    def box(x, y, w, h, title, detail="", color="#087F7A"):
        ax.add_patch(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                boxstyle="round,pad=0.05,rounding_size=.08",
                facecolor="#F3F6F7",
                edgecolor=color,
                lw=1.1,
            )
        )
        ax.text(
            x + 0.12,
            y + h - 0.18,
            title,
            fontsize=12,
            weight="semibold",
            color=color,
            va="top",
        )
        if detail:
            ax.text(
                x + 0.12, y + h - 0.49, detail, fontsize=9.5, va="top", linespacing=1.4
            )

    def arrow(x, y, xx, yy, color=MUTED):
        ax.add_patch(
            FancyArrowPatch(
                (x, y),
                (xx, yy),
                arrowstyle="-|>",
                mutation_scale=14,
                lw=1.3,
                color=color,
            )
        )

    ax.text(
        0.1,
        7.55,
        "From chemical structure to a testable cellular response",
        fontsize=19,
        weight="semibold",
    )
    box(
        0.1,
        5.95,
        3.5,
        1.05,
        "1  Observed experiments",
        "LINCS Level 3 · 978 measured genes\n10 µM · 24 h · treated + DMSO profiles",
    )
    box(
        4.1,
        5.95,
        4.6,
        1.05,
        "2  Fixed training / validation / test sets",
        "Keep supplied compound assignments; match measured controls.\nExclude external analogues >0.7 Tanimoto; exclude SKBR3.",
    )
    box(
        9.2,
        5.95,
        4.6,
        1.05,
        "3  Fair fitting and selection",
        "Three training seeds; equal three-setting tuning budget.\nChoose settings and checkpoints using validation NLL only.",
    )
    arrow(3.62, 6.47, 4.05, 6.47)
    arrow(8.73, 6.47, 9.15, 6.47)
    box(
        0.1,
        3.9,
        2.4,
        1.25,
        "Compound structure",
        "Same molecule for each\nrepresentation (schematic)",
    )
    vertices = np.array(
        [
            (2.13 + 0.16 * np.cos(t), 4.12 + 0.16 * np.sin(t))
            for t in np.linspace(0, 2 * np.pi, 7)
        ]
    )
    ax.plot(vertices[:, 0], vertices[:, 1], color=MUTED, lw=1.3)
    box(
        0.1,
        2.25,
        2.4,
        1.15,
        "Measured DMSO",
        "Control vector c: 978 genes\nSupplies the cell context",
    )
    ax.plot([2.55, 2.85, 2.85], [2.8, 2.8, 5.1], color=MUTED, lw=1.1)
    for i, (model, title, description) in enumerate(
        [
            ("ResidualFCFP4", "FCFP4", "1,024 functional-class bits"),
            ("Residual", "ECFP4", "1,024 atom-environment bits"),
            ("ResidualChemicalVAE", "ChemicalVAE", "196 frozen learned values"),
            ("PRnet", "FCFP4 × dose", "Published PRnet architecture"),
        ]
    ):
        y = 4.7 - i * 0.86
        color = COLORS[model]
        box(3.2, y, 3.25, 0.67, title, description, color)
        arrow(2.5, 4.5, 3.15, y + 0.33, color)
        box(
            7.05,
            y,
            2.35,
            0.67,
            "PRnet" if model == "PRnet" else "Residual MLP",
            "Noise-conditioned" if model == "PRnet" else "512 → 256 hidden units",
            color,
        )
        arrow(6.5, y + 0.33, 7.0, y + 0.33, color)
        ax.plot(
            [2.85, 2.99, 2.99, 6.75, 6.75],
            [2.8, 2.8, y + 0.76, y + 0.76, y + 0.33],
            color="#AAB8BE",
            lw=0.75,
        )
        arrow(6.75, y + 0.33, 7.0, y + 0.33, "#AAB8BE")
        arrow(9.45, y + 0.33, 9.84, y + 0.33, color)
    box(
        10.05,
        2.8,
        3.75,
        1.72,
        "Predicted treated expression",
        "MLPs: ŷ = c + predicted change\nPRnet: average 32 decoder draws\nOutput: 978 means + uncertainty\nCompare against measured targets",
    )
    ax.plot([9.85, 9.85], [2.45, 5.03], color=MUTED, lw=1)
    arrow(9.85, 3.65, 10.03, 3.65)
    box(
        0.1,
        0.25,
        4.0,
        1.33,
        "4  Shared held-out evaluations",
        "New compounds · new cell line SKBR3\nSeven dose-matched GEO studies / six genes\nStricter compound + control-group holdout",
    )
    box(
        4.65,
        0.25,
        4.05,
        1.33,
        "5  Test what chemistry adds",
        "Unchanged control, mean response and ridge\nControl-only network; shuffled chemical inputs\nResidual × initialization ablation",
    )
    box(
        9.25,
        0.25,
        4.55,
        1.33,
        "6  Assess reliability",
        "MSE, response correlation, direction accuracy\nStudy-cluster intervals and label sensitivity\nControl noise and predictive interval coverage",
    )
    arrow(4.15, 0.9, 4.6, 0.9)
    arrow(8.75, 0.9, 9.2, 0.9)
    ax.text(
        0.1,
        1.8,
        "Question: does chemistry improve prediction beyond the control state, and does that benefit transfer?",
        fontsize=12,
        weight="semibold",
    )
    fig.tight_layout()
    save(fig, "pipeline")


def main():
    setup()
    data = load()
    pipeline()
    selected = json.loads((RESULTS / "selected_settings.json").read_text())
    fig, axes = plt.subplots(2, 3, figsize=(12, 6), layout="constrained")
    for ax, model in zip(axes.flat, MODELS):
        setting = selected[model]
        folder = (
            RESULTS / "weights"
            if setting == "default"
            else RESULTS / "tuning" / setting
        )
        positive = True
        for seed, color in zip([42, 43, 44], ["#2878A5", "#078478", "#BE6C37"]):
            history = pd.read_csv(folder / f"{model}_{seed}.csv")
            best = history.loc[history.val_nll.idxmin()]
            positive &= history.val_nll.gt(0).all()
            ax.plot(
                history.epoch,
                history.val_nll,
                color=color,
                lw=1.1,
                label=f"Seed {seed}",
            )
            ax.scatter(
                best.epoch, best.val_nll, marker="*", color=color, s=65, zorder=3
            )
        if positive:
            ax.set_yscale("log")
        ax.set_title(NAMES[model], loc="left", weight="semibold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Validation NLL" + (" (log scale)" if positive else ""))
        ax.grid(alpha=0.2)
    axes.flat[-1].axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    axes.flat[-1].legend(handles, labels, loc="center", frameon=False)
    axes.flat[-1].text(
        0.5,
        0.22,
        "Stars mark selected checkpoints.\nEach panel has its own y-axis.\nNo test score selects an epoch.",
        ha="center",
        fontsize=10,
    )
    save(fig, "training_convergence")
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5), layout="constrained")
    for ax, dataset, title in zip(
        axes,
        ["internal", "skbr3", None],
        [
            "Original compound test · 27,765 profiles",
            "Unseen SKBR3 · 574 profiles",
            "External GEO · 7 studies / 42 directions",
        ],
    ):
        benchmark_axis(ax, data, dataset, external=dataset is None)
        ax.set_title(title, loc="left", fontsize=11, weight="semibold")
    fig.suptitle(
        "Circles: individual seeds · diamonds: mean · line: seed range · dashed: unchanged control",
        fontsize=10,
        color=MUTED,
    )
    save(fig, "benchmark")
    fig, ax = plt.subplots(figsize=(8, 4.3), layout="constrained")
    study_axis(ax, data)
    ax.set_title(
        "Correct directions out of six per study, averaged over three seeds",
        loc="left",
        fontsize=11,
    )
    save(fig, "external_studies")
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.5), layout="constrained")
    for ax, dataset in zip(axes, ["internal_low_similarity", "skbr3"]):
        shuffle_axis(ax, data, dataset)
        ax.set_title(
            (
                "Low-similarity internal targets"
                if dataset == "internal_low_similarity"
                else "Unseen SKBR3"
            ),
            loc="left",
        )
    fig.suptitle(
        "Chemical inputs shuffled within control groups · each dot averages 20 permutations for one training seed",
        fontsize=10,
    )
    save(fig, "chemistry_shuffle")
    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    noise_axis(ax, data)
    ax.legend(frameon=False, fontsize=9, ncol=2)
    ax.set_title(
        "Sensitivity to synthetic noise in measured control inputs",
        loc="left",
        weight="semibold",
    )
    save(fig, "control_noise")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), layout="constrained")
    for ax, dataset in zip(axes, ["internal_low_similarity", "skbr3"]):
        rows = (
            data["paired"]
            .loc[
                data["paired"].dataset.eq(dataset)
                & data["paired"].reference.eq("ControlOnly")
                & data["paired"].cluster.eq("compound")
            ]
            .set_index("model")
        )
        for i, model in enumerate(MODELS[:3]):
            r = rows.loc[model]
            ax.plot([r.lower, r.upper], [i, i], color=COLORS[model], lw=2)
            ax.scatter(r.difference, i, color=COLORS[model], s=35)
        ax.axvline(0, color=MUTED, lw=1)
        ax.set_yticks(range(3), [NAMES[m] for m in MODELS[:3]])
        ax.invert_yaxis()
        ax.set_xlabel("MSE difference from control-only (negative is better)")
        ax.set_title(
            (
                "Low-similarity internal targets"
                if dataset == "internal_low_similarity"
                else dataset
            ),
            loc="left",
        )
    fig.suptitle(
        "Paired compound-bootstrap 95% intervals · seeds averaged before resampling",
        fontsize=11,
    )
    save(fig, "chemical_added_value")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    rows = data["raw"].loc[data["raw"].dataset.eq("joint_low_similarity")]
    for j, phase in enumerate(["shared_groups_same_test", "group_holdout"]):
        means = (
            rows.loc[rows.phase.eq(phase)].groupby("model").mse.mean().reindex(MODELS)
        )
        ax.plot(
            range(5),
            means,
            "o-",
            color=["#8B9EA8", "#087F7A"][j],
            label=[
                "Original training; same test rows",
                "Control groups excluded from training",
            ][j],
        )
    ax.set_xticks(range(5), [NAMES[m].replace(" ", "\n", 1) for m in MODELS])
    ax.set_ylabel("MSE")
    ax.legend(frameon=False)
    ax.set_title(
        "Low-similarity compounds and unseen control groups",
        loc="left",
        weight="semibold",
    )
    save(fig, "control_group_holdout")
    ablation = pd.read_csv(RESULTS / "summary/factorial_ablation.csv")
    order = ["Direct", "DirectZeroInit", "ResidualRandomInit", "Residual"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")
    for ax, dataset in zip(axes, ["internal", "skbr3"]):
        for j, model in enumerate(order):
            vals = ablation.loc[
                ablation.model.eq(model) & ablation.dataset.eq(dataset), "mse"
            ]
            ax.scatter(
                np.full(len(vals), j),
                vals,
                color=["#77878D", "#A8B7BA", "#59A8A0", "#078478"][j],
            )
            ax.plot([j - 0.18, j + 0.18], [vals.mean(), vals.mean()], color=INK, lw=2)
        ax.set_xticks(
            range(4),
            ["Direct\nrandom", "Direct\nzero", "Residual\nrandom", "Residual\nzero"],
        )
        ax.set_ylabel("MSE")
        ax.set_title(
            (
                "Low-similarity internal targets"
                if dataset == "internal_low_similarity"
                else dataset
            ),
            loc="left",
        )
    fig.suptitle(
        "Separating residual prediction from mean-head initialization · three seeds / setting",
        fontsize=11,
    )
    save(fig, "residual_ablation")
    compounds = pd.read_csv(
        RESULTS / "results/compounds_fixed_tuned_group_holdout_baselines.csv"
    )
    sk = compounds.loc[compounds.phase.eq("tuned") & compounds.dataset.eq("skbr3")]
    exposure = pd.read_csv(RESULTS / "skbr3_chemical_exposure.csv")
    fig, axes = plt.subplots(2, 2, figsize=(8.3, 6.6), layout="constrained")
    for ax, model in zip(axes.flat, MODELS[:-1]):
        points = (
            sk.loc[sk.model.eq(model)]
            .groupby("structure")
            .agg(mse=("mse", "mean"), dmso=("dmso_mse", "first"))
            .reset_index()
            .merge(exposure, on="structure", validate="one_to_one")
        )
        limit = max(points.mse.max(), points.dmso.max()) * 1.05
        for seen, label, color in [
            (True, "Compound seen in training", "#8799A1"),
            (False, "Compound absent from training", "#087F7A"),
        ]:
            part = points.loc[points.seen_in_training.eq(seen)]
            ax.scatter(part.dmso, part.mse, c=color, s=24, alpha=0.75, label=label)
        ax.plot([0, limit], [0, limit], ls="--", c=MUTED, lw=1)
        ax.set(
            xlim=(0, limit),
            ylim=(0, limit),
            xlabel="Unchanged-control MSE",
            ylabel="Model MSE",
        )
        ax.set_title(NAMES[model], loc="left", weight="semibold")
        ax.text(
            0.03,
            0.96,
            f"{int((points.mse<points.dmso).sum())}/137 compounds below equality",
            transform=ax.transAxes,
            va="top",
            fontsize=9,
        )
    axes[1, 1].legend(frameon=False, fontsize=8, loc="lower right")
    fig.suptitle(
        "SKBR3: one point per compound; model loss averaged over three seeds",
        fontsize=11,
    )
    save(fig, "compound_response")
    calibration = pd.read_csv(RESULTS / "results/calibration.csv")
    fig, ax = plt.subplots(figsize=(8, 4), layout="constrained")
    for dataset, marker in [("internal", "o"), ("skbr3", "s")]:
        values = (
            calibration.loc[calibration.dataset.eq(dataset)]
            .groupby("model")
            .covered95.mean()
            .reindex(MODELS)
        )
        ax.plot(range(5), values, marker + "-", label=dataset)
    ax.axhline(0.95, color=MUTED, ls="--")
    ax.set_ylim(0, 1)
    ax.set_xticks(range(5), [NAMES[m].replace(" ", "\n", 1) for m in MODELS])
    ax.set_ylabel("Observed marginal 95% interval coverage")
    ax.legend(frameon=False)
    save(fig, "uncertainty_coverage")
    print("Saved vector PDF/SVG and PNG figures to", OUT)


if __name__ == "__main__":
    main()
