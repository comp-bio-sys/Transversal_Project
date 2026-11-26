"""Summarize the completed experiments without selecting models on test scores."""

import json
from pathlib import Path

import numpy as np
import pandas as pd

from benchmark import ROOT

SUFFIX = "fixed_tuned_group_holdout_baselines"
MODELS = ["ResidualFCFP4", "Residual", "ResidualChemicalVAE", "PRnet", "ControlOnly"]
N_BOOTSTRAP = 4000


def bootstrap_difference(differences, weights=None, seed=26026):
    """Resample paired clusters; preserve profile weighting within each draw."""
    differences = np.asarray(differences, dtype=float)
    weights = (
        np.ones(len(differences))
        if weights is None
        else np.asarray(weights, dtype=float)
    )
    assert len(differences) and np.isfinite(differences).all()
    rng = np.random.default_rng(seed)
    estimates = []
    for _ in range(N_BOOTSTRAP):
        ix = rng.integers(len(differences), size=len(differences))
        estimates.append(np.average(differences[ix], weights=weights[ix]))
    lower, upper = np.quantile(estimates, [0.025, 0.975])
    return dict(
        difference=float(np.average(differences, weights=weights)),
        lower=float(lower),
        upper=float(upper),
        clusters=len(differences),
    )


def main():
    root = ROOT / "balanced"
    results = root / "results"
    out = root / "summary"
    out.mkdir(exist_ok=True)
    scores = pd.read_csv(results / f"summary_{SUFFIX}.csv")
    external = pd.read_csv(results / f"external_{SUFFIX}.csv")
    external = external.loc[
        external.chemical_split.eq("eligible") & external.measured_change.ne(0)
    ].copy()
    compounds = pd.read_csv(results / f"compounds_{SUFFIX}.csv")
    groups = pd.read_csv(results / f"groups_{SUFFIX}.csv")
    aggregate = (
        scores.groupby(["phase", "dataset", "model"])
        .agg(
            seeds=("seed", "size"),
            profiles=("profiles", "first"),
            compounds=("compounds", "first"),
            mse_mean=("mse", "mean"),
            mse_seed_sd=("mse", "std"),
            macro_compound_mse=("macro_compound_mse", "mean"),
            response_pearson=("mean_response_pearson", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            skill_vs_dmso=("mse_skill_vs_dmso", "mean"),
        )
        .reset_index()
    )
    aggregate.to_csv(out / "model_summary.csv", index=False)

    # Average losses across seeds before resampling data clusters. This describes
    # the expected single-run loss over the three runs, not an ensemble's loss.
    comparisons = []
    for dataset in ["internal", "skbr3"]:
        for cluster, table, key in [
            ("compound", compounds, "structure"),
            ("control_group", groups, "group"),
        ]:
            rows = table.loc[table.dataset.eq(dataset) & table.phase.eq("tuned")]
            averaged = (
                rows.groupby(["model", key])
                .agg(mse=("mse", "mean"), profiles=("profiles", "first"))
                .reset_index()
            )
            reference = averaged.loc[averaged.model.eq("ControlOnly")].set_index(key)
            for model in MODELS[:-1]:
                pair = (
                    averaged.loc[averaged.model.eq(model)]
                    .set_index(key)
                    .join(reference, rsuffix="_reference", validate="one_to_one")
                )
                assert pair.mse_reference.notna().all()
                comparisons.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "reference": "ControlOnly",
                        "cluster": cluster,
                        **bootstrap_difference(
                            pair.mse - pair.mse_reference, pair.profiles
                        ),
                    }
                )
            for model in MODELS:
                means = (
                    rows.loc[rows.model.eq(model)]
                    .groupby(key)
                    .agg(
                        mse=("mse", "mean"),
                        dmso=("dmso_mse", "first"),
                        profiles=("profiles", "first"),
                    )
                )
                comparisons.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "reference": "DMSO",
                        "cluster": cluster,
                        **bootstrap_difference(means.mse - means.dmso, means.profiles),
                    }
                )
    pd.DataFrame(comparisons).to_csv(out / "paired_mse_intervals.csv", index=False)

    counts = (
        external.groupby(["phase", "model", "seed"])
        .correct.agg(["sum", "size", "mean"])
        .reset_index()
    )
    counts.to_csv(out / "external_seed_counts.csv", index=False)
    per_study = (
        external.groupby(["phase", "model", "seed", "gse_id"])
        .correct.mean()
        .reset_index()
    )
    per_study.to_csv(out / "external_by_study.csv", index=False)
    external.groupby(
        ["phase", "model", "seed", "genes"]
    ).correct.mean().reset_index().to_csv(out / "external_by_gene.csv", index=False)
    sensitivity = []
    for label, mask in [
        ("all_measured", np.ones(len(external), dtype=bool)),
        ("replicates_separated", external.separated_replicates),
        ("clear_change", external.clear_change),
    ]:
        z = (
            external.loc[mask]
            .groupby(["phase", "model", "seed"])
            .correct.agg(["sum", "size", "mean"])
            .reset_index()
        )
        z["subset"] = label
        sensitivity.append(z)
    pd.concat(sensitivity).to_csv(out / "external_label_sensitivity.csv", index=False)
    ext_intervals = []
    pivot = (
        per_study.loc[per_study.phase.eq("tuned")]
        .groupby(["model", "gse_id"])
        .correct.mean()
        .unstack("model")
    )
    for reference in ["PRnet", "ControlOnly"]:
        for model in MODELS:
            if model != reference:
                ext_intervals.append(
                    {
                        "model": model,
                        "reference": reference,
                        "cluster": "GEO study",
                        **bootstrap_difference(pivot[model] - pivot[reference]),
                    }
                )
    pd.DataFrame(ext_intervals).to_csv(
        out / "paired_external_intervals.csv", index=False
    )

    # A descriptive chemical-distance analysis of the original fixed test set.
    separation = pd.read_csv(root / "internal_chemical_separation.csv")
    selected = compounds.loc[
        compounds.dataset.eq("internal") & compounds.phase.isin(["tuned", "baseline"])
    ]
    selected = selected.merge(separation, on="structure", validate="many_to_one")
    selected["similarity_stratum"] = np.where(
        selected.max_training_tanimoto > 0.7, ">0.7", "<=0.7"
    )
    strata = []
    for (phase, model, seed, stratum), d in selected.groupby(
        ["phase", "model", "seed", "similarity_stratum"]
    ):
        strata.append(
            {
                "phase": phase,
                "model": model,
                "seed": seed,
                "stratum": stratum,
                "compounds": len(d),
                "profiles": int(d.profiles.sum()),
                "mse": float(np.average(d.mse, weights=d.profiles)),
                "macro_compound_mse": float(d.mse.mean()),
                "dmso_mse": float(np.average(d.dmso_mse, weights=d.profiles)),
            }
        )
    pd.DataFrame(strata).to_csv(out / "chemical_distance_results.csv", index=False)

    exposure = pd.read_csv(root / "skbr3_chemical_exposure.csv")
    sk = compounds.loc[
        compounds.dataset.eq("skbr3") & compounds.phase.isin(["tuned", "baseline"])
    ]
    sk = sk.merge(exposure, on="structure", validate="many_to_one")
    exposed = []
    for (phase, model, seed, seen), d in sk.groupby(
        ["phase", "model", "seed", "seen_in_training"]
    ):
        exposed.append(
            {
                "phase": phase,
                "model": model,
                "seed": seed,
                "compound_seen_in_training": bool(seen),
                "compounds": len(d),
                "profiles": int(d.profiles.sum()),
                "mse": float(np.average(d.mse, weights=d.profiles)),
                "dmso_mse": float(np.average(d.dmso_mse, weights=d.profiles)),
            }
        )
    pd.DataFrame(exposed).to_csv(
        out / "skbr3_compound_exposure_results.csv", index=False
    )

    strict = selected.loc[selected.max_training_tanimoto.le(0.7)]
    strict_records = []
    for (phase, model, seed), d in strict.groupby(["phase", "model", "seed"]):
        valid = d.response_pearson.notna()
        error = float(np.average(d.mse, weights=d.profiles))
        baseline = float(np.average(d.dmso_mse, weights=d.profiles))
        strict_records.append(
            {
                "phase": phase,
                "model": model,
                "seed": seed,
                "dataset": "internal_low_similarity",
                "profiles": int(d.profiles.sum()),
                "compounds": len(d),
                "mse": error,
                "macro_compound_mse": float(d.mse.mean()),
                "mse_skill_vs_dmso": 1 - error / baseline,
                "mean_response_pearson": (
                    float(
                        np.average(
                            d.loc[valid, "response_pearson"],
                            weights=d.loc[valid, "profiles"],
                        )
                    )
                    if valid.any()
                    else np.nan
                ),
                "direction_accuracy": (
                    float(d.direction_correct.sum() / d.direction_scored.sum())
                    if model != "DMSO"
                    else np.nan
                ),
            }
        )
    strict_scores = pd.DataFrame(strict_records)
    strict_scores.to_csv(out / "low_similarity_seed_scores.csv", index=False)
    strict_aggregate = (
        strict_scores.groupby(["phase", "dataset", "model"])
        .agg(
            seeds=("seed", "size"),
            profiles=("profiles", "first"),
            compounds=("compounds", "first"),
            mse_mean=("mse", "mean"),
            mse_seed_sd=("mse", "std"),
            macro_compound_mse=("macro_compound_mse", "mean"),
            response_pearson=("mean_response_pearson", "mean"),
            direction_accuracy=("direction_accuracy", "mean"),
            skill_vs_dmso=("mse_skill_vs_dmso", "mean"),
        )
        .reset_index()
    )
    aggregate = pd.concat([aggregate, strict_aggregate], ignore_index=True)
    aggregate.to_csv(out / "model_summary.csv", index=False)
    reference = (
        strict.loc[strict.phase.eq("tuned") & strict.model.eq("ControlOnly")]
        .groupby("structure")
        .mse.mean()
    )
    for model in MODELS:
        d = (
            strict.loc[strict.phase.eq("tuned") & strict.model.eq(model)]
            .groupby("structure")
            .agg(
                mse=("mse", "mean"),
                dmso=("dmso_mse", "first"),
                profiles=("profiles", "first"),
            )
        )
        for baseline in ["ControlOnly", "DMSO"]:
            if model == baseline:
                continue
            target = d.dmso if baseline == "DMSO" else reference.reindex(d.index)
            comparisons.append(
                {
                    "dataset": "internal_low_similarity",
                    "model": model,
                    "reference": baseline,
                    "cluster": "compound",
                    **bootstrap_difference(d.mse - target, d.profiles),
                }
            )
    pd.DataFrame(comparisons).to_csv(out / "paired_mse_intervals.csv", index=False)

    # Factorial comparison: isolate residual connection from mean-head initialization.
    ablations = scores.loc[
        scores.phase.eq("fixed")
        & scores.model.isin(
            ["Residual", "Direct", "DirectZeroInit", "ResidualRandomInit"]
        )
    ].copy()
    ablations["residual_connection"] = ablations.model.isin(
        ["Residual", "ResidualRandomInit"]
    )
    ablations["zero_mean_initialization"] = ablations.model.isin(
        ["Residual", "DirectZeroInit"]
    )
    ablations.to_csv(out / "factorial_ablation.csv", index=False)
    (out / "uncertainty_protocol.json").write_text(
        json.dumps(
            {
                "bootstrap_draws": N_BOOTSTRAP,
                "seed": 26026,
                "seed_handling": "Mean of the three per-run losses or accuracies before paired cluster resampling. Not ensemble predictions.",
                "internal_and_skbr3": "Profile-weighted paired losses, resampling compounds or control groups separately.",
                "external": "Seven whole GEO studies; six genes stay together. All studies have six labels.",
                "limits": "Exploratory descriptive percentile intervals; no multiplicity correction. Compound and control-group dependence is crossed and not jointly modeled. Seeds are not biological replicates. These are previously examined test sets.",
            },
            indent=2,
        )
        + "\n"
    )
    print(
        aggregate.loc[aggregate.phase.isin(["tuned", "baseline"])].to_string(
            index=False
        )
    )


if __name__ == "__main__":
    main()
