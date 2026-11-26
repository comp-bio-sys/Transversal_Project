"""Evaluate matched runs, tuning, baselines and joint compound/control holdout."""

import argparse, json
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, BASE, build
from balanced_train import MAIN, ABLATIONS
from scientific_data import (
    load_internal,
    load_skbr3,
    load_external,
    infer,
    metric_rows,
    mean_prediction,
    ridge_prediction,
    REP,
)
from evaluate_project import GENES


def main(phases):
    torch.set_num_threads(4)
    root = ROOT / "balanced"
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    internal = load_internal()
    sk = load_skbr3()
    external = load_external()
    gene_order = json.loads((BASE / "data/feature_spec.json").read_text())["gene_order"]
    cols = [gene_order.index(GENES[g]) for g in GENES]
    summaries = []
    compounds = []
    groups = []
    external_rows = []
    separation = pd.read_csv(root / "internal_chemical_separation.csv")
    low_similarity = set(
        separation.loc[separation.max_training_tanimoto.le(0.7), "structure"]
    )

    def record(phase, model, seed, d, mu):
        summary, frame = metric_rows(d, mu)
        if model == "DMSO":
            summary["direction_accuracy"] = None
        tag = {"phase": phase, "model": model, "seed": seed, "dataset": d["name"]}
        summaries.append({**tag, **summary})
        for key, destination in [("structure", compounds), ("group", groups)]:
            z = (
                frame.groupby(key)
                .agg(
                    mse=("mse", "mean"),
                    dmso_mse=("dmso_mse", "mean"),
                    profiles=("mse", "size"),
                    response_pearson=("response_pearson", "mean"),
                    direction_correct=("direction_correct", "sum"),
                    direction_scored=("direction_scored", "sum"),
                )
                .reset_index()
            )
            destination.append(z.assign(**tag))
        if d["name"] == "joint_compound_control_holdout":
            mask = np.array([s in low_similarity for s in d["structure"]])
            restricted = {
                k: (v[mask] if isinstance(v, np.ndarray) and len(v) == len(mask) else v)
                for k, v in d.items()
            }
            restricted["name"] = "joint_low_similarity"
            record(phase, model, seed, restricted, mu[mask])

    def record_external(phase, model, seed, mu):
        changes = (mu - external["control"])[:, cols]
        frame = pd.DataFrame(changes, columns=list(GENES)).assign(
            drug_id=external["studies"].drug_id.to_numpy()
        )
        long = frame.melt(
            id_vars="drug_id", var_name="genes", value_name="predicted_change"
        ).merge(external["measured"], on=["drug_id", "genes"], validate="one_to_one")
        long["correct"] = np.sign(long.predicted_change) == np.sign(
            long.measured_change
        )
        external_rows.append(long.assign(phase=phase, model=model, seed=seed))

    def evaluate_checkpoint(
        phase, variant, seed, path, datasets, scaling_path=None, score_external=True
    ):
        model = build(variant)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.eval()
        rep = REP.get(variant, "ECFP4")
        for d in datasets:
            mu = infer(model, variant, d["features"][rep], d["control"], scaling_path)
            record(phase, variant, seed, d, mu)
        if score_external:
            record_external(
                phase,
                variant,
                seed,
                infer(
                    model,
                    variant,
                    external["features"][rep],
                    external["control"],
                    scaling_path,
                ),
            )
        print("Evaluated", phase, variant, seed, flush=True)

    if "fixed" in phases:
        for variant in MAIN + ABLATIONS:
            for seed in [42, 43, 44]:
                evaluate_checkpoint(
                    "fixed",
                    variant,
                    seed,
                    root / "weights" / f"{variant}_{seed}.pt",
                    [internal, sk],
                )
    if "tuned" in phases:
        selected = json.loads((root / "selected_settings.json").read_text())
        for variant, setting in selected.items():
            folder = (
                root / "weights" if setting == "default" else root / "tuning" / setting
            )
            for seed in [42, 43, 44]:
                evaluate_checkpoint(
                    "tuned",
                    variant,
                    seed,
                    folder / f"{variant}_{seed}.pt",
                    [internal, sk],
                )
    if "group_holdout" in phases:
        folder = root / "group_holdout"
        ix = np.load(folder / "split_indices.npz")["test"]
        joint = load_internal(ix)
        joint["name"] = "joint_compound_control_holdout"
        # Original fixed models evaluated on identical rows show the changed test population.
        for variant in MAIN:
            for seed in [42, 43, 44]:
                evaluate_checkpoint(
                    "group_holdout",
                    variant,
                    seed,
                    folder / "weights" / f"{variant}_{seed}.pt",
                    [joint],
                    folder / "scaling.npz",
                    False,
                )
                evaluate_checkpoint(
                    "shared_groups_same_test",
                    variant,
                    seed,
                    root / "weights" / f"{variant}_{seed}.pt",
                    [joint],
                    None,
                    False,
                )
        record("baseline", "DMSO", 0, joint, joint["control"])
    if "baselines" in phases:
        functions = {
            "DMSO": lambda d: d["control"],
            "GlobalMeanResponse": lambda d: mean_prediction(d),
            "CellMeanResponse": lambda d: mean_prediction(d, True),
            "RidgeECFP4": ridge_prediction,
        }
        for name, func in functions.items():
            for d in [internal, sk]:
                record("baseline", name, 0, d, func(d))
            if name != "DMSO":
                record_external("baseline", name, 0, func(external))
        train = np.load(ROOT / "split_indices.npz")["train"]
        ids = np.load(BASE / "prepared/indices.npz")
        controls = np.load(BASE / "data/control_means.npy")
        targets = np.load(BASE / "prepared/targets.npy", mmap_mode="r")
        votes = np.zeros(len(cols))
        for start in range(0, len(train), 1024):
            ix = train[start : start + 1024]
            votes += np.sign(
                targets[ix][:, cols] - controls[ids["control_ids"][ix]][:, cols]
            ).sum(0)
        voted = external["control"].copy()
        voted[:, cols] += np.sign(votes)
        record_external("baseline", "TrainingGeneDirection", 0, voted)
        true_delta = sk["y"] - sk["control"]
        loo = np.empty_like(true_delta)
        eligible = np.zeros(len(loo), bool)
        for indices in (
            pd.Series(np.arange(len(loo)))
            .groupby(sk["structure"])
            .apply(lambda x: x.to_numpy())
        ):
            if len(indices) > 1:
                loo[indices] = (true_delta[indices].sum(0) - true_delta[indices]) / (
                    len(indices) - 1
                )
                eligible[indices] = True
        subset = {
            k: (
                v[eligible]
                if isinstance(v, np.ndarray) and len(v) == len(eligible)
                else v
            )
            for k, v in sk.items()
        }
        subset["name"] = "skbr3_replicate_reference"
        record(
            "reference",
            "LeaveOneProfileOutResponse",
            0,
            subset,
            sk["control"][eligible] + loo[eligible],
        )
        record("reference", "DMSO", 0, subset, sk["control"][eligible])
    suffix = "_".join(phases)
    pd.DataFrame(summaries).to_csv(out / f"summary_{suffix}.csv", index=False)
    pd.concat(compounds, ignore_index=True).to_csv(
        out / f"compounds_{suffix}.csv", index=False
    )
    pd.concat(groups, ignore_index=True).to_csv(
        out / f"groups_{suffix}.csv", index=False
    )
    if external_rows:
        rows = pd.concat(external_rows, ignore_index=True)
        rows.to_csv(out / f"external_{suffix}.csv", index=False)
        eligible = rows.loc[
            rows.chemical_split.eq("eligible") & rows.measured_change.ne(0)
        ]
        assert eligible.groupby(["phase", "model", "seed"]).size().eq(42).all()
        eligible.groupby(["phase", "model", "seed"]).correct.agg(
            ["sum", "size", "mean"]
        ).to_csv(out / f"external_scores_{suffix}.csv")
    (out / "evaluation_protocol.json").write_text(
        json.dumps(
            {
                "PRnet_noise_draws": 32,
                "PRnet_noise_seed_sequence": "1042 through 1073, one RNG per draw",
                "direction_threshold": ".5 absolute Level 3 units for internal/SKBR3; sign of measured change for GEO",
                "replicate_reference": "Leave-one-profile-out mean response of same compound in SKBR3, across plates. Excludes own target; diagnostic reference, not deployable model or guaranteed ceiling.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument(
        "--phases", nargs="+", default=["fixed", "tuned", "group_holdout", "baselines"]
    )
    a = p.parse_args()
    main(a.phases)
