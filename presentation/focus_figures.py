"""Validation selection and comparison of ECFP4 MLP with PRnet."""

import matplotlib.pyplot as plt
from matplotlib.colors import to_rgb
import numpy as np
import pandas as pd

import figures

MAIN_MODELS = ["ResidualFCFP4", "Residual", "ResidualChemicalVAE", "PRnet"]
ALL_DISPLAY_MODELS = MAIN_MODELS + ["DMSO"]
SELECTED_MODELS = ["Residual", "PRnet"]
DISPLAY_MODELS = SELECTED_MODELS + ["DMSO"]
NAMES = {model: figures.NAMES[model] for model in MAIN_MODELS}
NAMES["DMSO"] = "DMSO reference"
COLORS = dict(zip(ALL_DISPLAY_MODELS,
                  ["#3F7895", "#458579", "#8B729E", "#B47C53", "#92999D"]))
SHORT_NAMES = dict(zip(ALL_DISPLAY_MODELS, ["FCFP4\nMLP", "ECFP4\nMLP", "ChemicalVAE\nMLP", "PRnet", "DMSO\nreference"]))


def _seed_values(table, model, column):
    phase = "baseline" if model == "DMSO" else "tuned"
    rows = table.loc[table.model.eq(model) & table.phase.eq(phase)].sort_values("seed")
    expected = [0] if model == "DMSO" else [42, 43, 44]
    if rows.seed.tolist() != expected:
        raise ValueError(f"Unexpected runs for {model}: {rows.seed.tolist()}")
    values = rows[column].to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise ValueError(f"Non-finite {column} for {model}")
    return values


def _bars(ax, values, models, size, annotate, upper=None):
    maximum = max(float(v.max()) for v in values)
    limit = float(upper) if upper is not None else maximum * 1.22
    for position, (model, runs) in enumerate(zip(models, values)):
        mean = float(runs.mean())
        color = COLORS[model]
        fill = np.asarray(to_rgb(color)) * .6 + .4
        ax.bar(position, mean, width=.62, color=fill, edgecolor="#303030",
               linewidth=1 * size, zorder=2)
        if model != "DMSO":
            ax.errorbar(position, mean,
                        yerr=[[mean - runs.min()], [runs.max() - mean]],
                        fmt="none", ecolor=figures.INK, elinewidth=1.1 * size,
                        capsize=3.5 * size, capthick=1 * size, zorder=4)
            ax.scatter(position + np.array([-.15, 0, .15]), runs,
                       s=23 * size, facecolors="white", edgecolors="#303030",
                       linewidths=1 * size, zorder=5)
        if annotate:
            label = f"{mean:.1f}" if upper is not None else f"{mean:.3f}"
            ax.text(position, runs.max() + .035 * limit, label, ha="center",
                    va="bottom", fontsize=9 * size, color=figures.INK)
    ax.set_ylim(0, limit)
    ax.set_xlim(-.6, len(models) - .4)
    ax.set_xticks(range(len(models)), [SHORT_NAMES[m] for m in models], fontsize=9 * size)
    ax.tick_params(axis="x", length=0, pad=6 * size, colors="#303030")
    ax.tick_params(axis="y", labelsize=9 * size, colors="#303030")
    ax.set_axisbelow(True)
    ax.grid(False)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#303030")
        ax.spines[side].set_linewidth(1 * size)
    return {model: runs.tolist() for model, runs in zip(models, values)}


def mse_bar_axis(ax, data, dataset, size=1.0, annotate=True):
    """Plot seed means and observed seed ranges; DMSO has one fixed score."""
    table = data["strict"] if dataset == "internal_low_similarity" else data["raw"]
    table = table.loc[table.dataset.eq(dataset)]
    values = [_seed_values(table, model, "mse") for model in DISPLAY_MODELS]
    scores = _bars(ax, values, DISPLAY_MODELS, size, annotate)
    ax.set_ylabel("Treated-expression MSE", fontsize=10 * size)
    return scores


def external_bar_axis(ax, data, size=1.0, annotate=True):
    """Plot correct signs; unchanged DMSO is not an up/down predictor."""
    table = data["external"]
    rows = table.loc[table.phase.eq("tuned") & table.model.isin(SELECTED_MODELS)]
    if not rows["size"].eq(42).all():
        raise ValueError("External comparison must contain 42 directions per run")
    values = [_seed_values(table, model, "sum") for model in SELECTED_MODELS]
    scores = _bars(ax, values, SELECTED_MODELS, size, annotate, upper=42)
    ax.set_ylabel("Correct directions / 42", fontsize=10 * size)
    ax.set_yticks(np.arange(0, 43, 7))
    return scores


def study_axis(ax, data, size=1.0):
    """Show the mean number of correct signs for six genes per GEO study."""
    studies = pd.read_csv(figures.ROOT.parent / "dose_matched/study_manifest.csv")
    studies = studies.loc[studies.chemical_split.eq("eligible")]
    table = data["studies"].loc[
        data["studies"].phase.eq("tuned") & data["studies"].model.isin(SELECTED_MODELS)]
    counts = table.groupby(["gse_id", "model"]).seed.agg(list)
    if len(counts) != len(studies) * len(SELECTED_MODELS) or not counts.map(
            lambda seeds: sorted(seeds) == [42, 43, 44]).all():
        raise ValueError("Expected three runs for every study and chemical model")
    matrix = (table.groupby(["gse_id", "model"]).correct.mean().unstack()
              .reindex(index=studies.gse_id, columns=SELECTED_MODELS) * 6)
    if matrix.isna().any().any() or not ((matrix >= 0) & (matrix <= 6)).all().all():
        raise ValueError("Invalid external study scores")
    ax.imshow(matrix, cmap="YlGnBu", vmin=0, vmax=6, aspect="auto")
    for i in range(len(matrix)):
        for j in range(len(SELECTED_MODELS)):
            value = matrix.iloc[i, j]
            ax.text(j, i, f"{value:.1f}", ha="center", va="center",
                    color="white" if value > 3.7 else figures.INK,
                    fontsize=10 * size)
    ax.set_xticks(range(2), ["ECFP4 MLP", "PRnet"],
                  fontsize=9 * size)
    ax.set_yticks(range(len(studies)),
                  [f"{r.drug_name} / {r.cell}" for r in studies.itertuples()],
                  fontsize=9 * size)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    return matrix


def noise_axis(ax, data, size=1.0):
    """Plot absolute prediction error while adding noise to control inputs."""
    rows = data["noise"].copy()
    if rows.duplicated(["model", "seed", "level"]).any():
        raise ValueError("Expected one technical-draw mean per model, seed and level")
    rows["increase"] = rows.mse - rows.clean_mse
    output = {}
    for model in DISPLAY_MODELS:
        selected = rows.loc[rows.model.eq(model)].sort_values(["level", "seed"])
        expected = [0] if model == "DMSO" else [42, 43, 44]
        if selected.level.unique().tolist() != [0, .25, .5, 1]:
            raise ValueError(f"Unexpected control-noise levels for {model}")
        for _, group in selected.groupby("level"):
            if group.seed.tolist() != expected:
                raise ValueError(f"Unexpected noise runs for {model}")
        np.testing.assert_allclose(selected.loc[selected.level.eq(0), "increase"], 0)
        summary = selected.groupby("level").mse.agg(["mean", "min", "max"])
        ax.plot(summary.index, summary["mean"], "o--" if model == "DMSO" else "o-",
                lw=1.5 * size, ms=4 * size, color=COLORS[model], label=NAMES[model])
        if model != "DMSO":
            ax.fill_between(summary.index, summary["min"], summary["max"],
                            color=COLORS[model], alpha=.12)
        output[model] = summary
    ax.axhline(0, color="#999999", lw=.8 * size)
    ax.set_xticks([0, .25, .5, 1])
    ax.set_xlabel("Added noise / within-plate control SD", fontsize=10 * size)
    ax.set_ylabel("SKBR3 treated-expression MSE", fontsize=10 * size)
    ax.set_ylim(bottom=0)
    ax.tick_params(labelsize=9 * size)
    ax.grid(color="#E5EBED", lw=.7 * size)
    ax.legend(frameon=False, loc="upper left", fontsize=8.5 * size, ncol=2,
              columnspacing=.8, handlelength=1.5)
    return output



def _validation_axis(ax, models, values, labels, size, annotate):
    for i, (model, value) in enumerate(zip(models, values)):
        fill = np.asarray(to_rgb(COLORS[model])) * .6 + .4
        ax.bar(i, value, width=.58, color=fill, edgecolor="#303030", linewidth=size)
        if annotate:
            ax.text(i, value + .035 * max(values), f"{value:.5f}", ha="center",
                    va="bottom", fontsize=9 * size, color=figures.INK)
    ax.set_xticks(range(len(models)), labels, fontsize=9 * size)
    ax.set_ylim(0, max(values) * 1.22)
    ax.set_ylabel("Validation NLL", fontsize=10 * size)
    ax.tick_params(axis="x", length=0, pad=6 * size)
    ax.tick_params(axis="y", labelsize=9 * size)
    for side in ["left", "bottom"]:
        ax.spines[side].set_color("#303030")
        ax.spines[side].set_linewidth(size)
    ax.grid(False)


def _validation_table():
    table = pd.read_csv(figures.ROOT.parent / "extension/balanced/tuning_validation.csv")
    if table.duplicated(["model", "setting"]).any():
        raise ValueError("Duplicate model/setting validation records")
    return table.set_index(["model", "setting"])


def selection_axis(ax, data=None, size=1.0, annotate=True):
    """Compare three representations under the same default setting, seed 42."""
    table = _validation_table()
    models = MAIN_MODELS[:3]
    values = [float(table.loc[(model, "default"), "val_nll"]) for model in models]
    if models[int(np.argmin(values))] != "Residual":
        raise ValueError("Recorded validation winner is no longer ECFP4")
    _validation_axis(ax, models, values, [SHORT_NAMES[m] for m in models], size, annotate)
    return dict(zip(models, values))


def improvement_axis(ax, data=None, size=1.0, annotate=True):
    """Show the selected learning-rate change without implying test improvement."""
    table = _validation_table()
    values = [float(table.loc[("Residual", setting), "val_nll"])
              for setting in ["default", "lower_lr"]]
    _validation_axis(ax, ["Residual", "Residual"], values,
                     ["Default\nLR 0.001", "Selected\nLR 0.0003"], size, annotate)
    return values


def build_main_figures():
    """Export the main figures after supplementary figures are generated."""
    figures.setup()
    data = figures.load()
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2), layout="constrained")
    selection_axis(axes[0], data)
    improvement_axis(axes[1], data)
    axes[0].set_title("a  Select representation · default setting", loc="left", fontsize=11, pad=12)
    axes[1].set_title("b  Tune ECFP4 · validation score", loc="left", fontsize=11, pad=12)
    figures.save(fig, "model_selection")

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), layout="constrained")
    for ax, dataset, title in zip(
            axes, ["internal_low_similarity", "skbr3"],
            ["Distant compounds · 474 compounds / 16,531 profiles",
             "Unseen SKBR3 · 137 compounds / 574 profiles"]):
        mse_bar_axis(ax, data, dataset)
        ax.set_title(title, loc="left", fontsize=10, weight="semibold", pad=12)
    figures.save(fig, "benchmark")

    fig, ax = plt.subplots(figsize=(9, 4.6), layout="constrained")
    study_axis(ax, data)
    ax.set_title("Correct directions out of six per study; mean of three seeds",
                 loc="left", fontsize=11, pad=12)
    figures.save(fig, "external_studies")

    fig, ax = plt.subplots(figsize=(7, 4.5), layout="constrained")
    external_bar_axis(ax, data)
    ax.set_title("Independent GEO studies · 42 gene directions",
                 loc="left", fontsize=11, weight="semibold", pad=12)
    figures.save(fig, "external_comparison")

    fig, ax = plt.subplots(figsize=(8, 4.5), layout="constrained")
    noise_axis(ax, data)
    ax.set_title("Control-input noise: error on the same SKBR3 test profiles",
                 loc="left", fontsize=11, weight="semibold", pad=12)
    figures.save(fig, "control_noise")


def main():
    build_main_figures()


if __name__ == "__main__":
    main()
