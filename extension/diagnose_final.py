"""Input-shuffling and synthetic-noise diagnostics for all selected model seeds."""

import json
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, build
from scientific_data import load_internal, load_skbr3, infer, REP


def main():
    torch.set_num_threads(4)
    root = ROOT / "balanced"
    out = root / "diagnostics"
    out.mkdir(exist_ok=True)
    selected = json.loads((root / "selected_settings.json").read_text())
    datasets = [load_internal(), load_skbr3()]
    permutations = []
    noise_results = []
    empirical = []
    draws = 20
    protocol = {
        "model_selection": "validation NLL only; selected settings reused unchanged",
        "seeds": [42, 43, 44],
        "permutation_draws": draws,
        "permutation": "Chemical feature rows permuted within matched control group, same draws for every model and seed. Low-similarity scores restrict target rows; donor pool is the full internal test group.",
        "empirical_control_resampling": "Twenty draws using half the measured DMSO replicates per plate (minimum one), without replacement.",
        "PRnet_decoder_noise_draws": 32,
        "noise_levels": [0, 0.25, 0.5, 1.0],
        "synthetic_noise": "Independent Gaussian gene noise, scaled by within-plate DMSO replicate SD; a shared draw is added to all profiles on the same plate.",
        "limits": "Already examined test sets; exploratory. Noise/permutation draws are technical repeats, not new biological replicates.",
    }
    (out / "protocol.json").write_text(json.dumps(protocol, indent=2) + "\n")
    for d in datasets:
        grouped = (
            pd.Series(np.arange(len(d["control"])))
            .groupby(d["group"])
            .apply(lambda x: x.to_numpy())
            .tolist()
        )
        if d["name"] == "internal":
            similarity = (
                pd.read_csv(root / "internal_chemical_separation.csv")
                .set_index("structure")
                .max_training_tanimoto
            )
            low_similarity = (
                pd.Series(d["structure"]).map(similarity).le(0.7).to_numpy()
            )
        ix_draws = []
        for draw in range(draws):
            rng = np.random.default_rng(26000 + draw)
            ix = np.arange(len(d["control"]))
            for group in grouped:
                ix[group] = rng.permutation(group)
            ix_draws.append(ix)
        if d["name"] == "skbr3":
            cm = d["control_metadata"]
            raw = d["raw_controls"]
            batches = cm.groupby("det_plate").indices
            names = sorted(batches)
            lookup = {p: i for i, p in enumerate(names)}
            sd = np.stack([raw[batches[p]].std(0, ddof=1) for p in names])
            plate_ix = np.array([lookup[p] for p in d["group"]])
            assert min(len(batches[p]) for p in names) > 1
            half_controls = []
            for draw in range(draws):
                rng = np.random.default_rng(28000 + draw)
                means = np.stack(
                    [
                        raw[
                            rng.choice(
                                batches[p], max(1, len(batches[p]) // 2), replace=False
                            )
                        ].mean(0)
                        for p in names
                    ]
                )
                half_controls.append(means[plate_ix])
        for model, setting in selected.items():
            folder = (
                root / "weights" if setting == "default" else root / "tuning" / setting
            )
            for seed in [42, 43, 44]:
                network = build(model)
                network.load_state_dict(
                    torch.load(
                        folder / f"{model}_{seed}.pt",
                        map_location="cpu",
                        weights_only=True,
                    )
                )
                network.eval()
                fp = d["features"][REP.get(model, "ECFP4")]
                c = d["control"]
                y = d["y"]
                mu = infer(network, model, fp, c)
                clean = float(np.square(y - mu).mean(dtype="float64"))
                for draw, ix in enumerate(ix_draws):
                    perturbed = infer(network, model, fp[ix], c)
                    mse = float(np.square(y - perturbed).mean(dtype="float64"))
                    permutations.append(
                        {
                            "dataset": d["name"],
                            "model": model,
                            "seed": seed,
                            "draw": draw,
                            "changed_fraction": float(
                                np.mean(d["structure"][ix] != d["structure"])
                            ),
                            "clean_mse": clean,
                            "shuffled_mse": mse,
                            "mse_increase": mse - clean,
                            "prediction_rms_change": float(
                                np.sqrt(np.square(perturbed - mu).mean(dtype="float64"))
                            ),
                        }
                    )
                    if d["name"] == "internal":
                        low_clean = float(
                            np.square(y[low_similarity] - mu[low_similarity]).mean(
                                dtype="float64"
                            )
                        )
                        low_mse = float(
                            np.square(
                                y[low_similarity] - perturbed[low_similarity]
                            ).mean(dtype="float64")
                        )
                        permutations.append(
                            {
                                "dataset": "internal_low_similarity",
                                "model": model,
                                "seed": seed,
                                "draw": draw,
                                "changed_fraction": float(
                                    np.mean(
                                        d["structure"][ix][low_similarity]
                                        != d["structure"][low_similarity]
                                    )
                                ),
                                "clean_mse": low_clean,
                                "shuffled_mse": low_mse,
                                "mse_increase": low_mse - low_clean,
                                "prediction_rms_change": float(
                                    np.sqrt(
                                        np.square(
                                            perturbed[low_similarity]
                                            - mu[low_similarity]
                                        ).mean(dtype="float64")
                                    )
                                ),
                            }
                        )
                if d["name"] == "skbr3":
                    for level in protocol["noise_levels"]:
                        for draw in range(draws if level else 1):
                            rng = np.random.default_rng(27000 + draw)
                            noisy = (
                                c
                                + (
                                    rng.standard_normal(sd.shape).astype("float32")
                                    * sd
                                    * level
                                )[plate_ix]
                            )
                            pred = (
                                mu if level == 0 else infer(network, model, fp, noisy)
                            )
                            noise_results.append(
                                {
                                    "model": model,
                                    "seed": seed,
                                    "level": level,
                                    "draw": draw,
                                    "mse": float(
                                        np.square(y - pred).mean(dtype="float64")
                                    ),
                                    "clean_mse": clean,
                                    "prediction_rms_change": float(
                                        np.sqrt(
                                            np.square(pred - mu).mean(dtype="float64")
                                        )
                                    ),
                                }
                            )
                if d["name"] == "skbr3":
                    for draw, noisy in enumerate(half_controls):
                        pred = infer(network, model, fp, noisy)
                        empirical.append(
                            {
                                "model": model,
                                "seed": seed,
                                "draw": draw,
                                "clean_mse": clean,
                                "half_control_mse": float(
                                    np.square(y - pred).mean(dtype="float64")
                                ),
                                "prediction_rms_change": float(
                                    np.sqrt(np.square(pred - mu).mean(dtype="float64"))
                                ),
                            }
                        )
                pd.DataFrame(permutations).to_csv(
                    out / "permutation_draws.csv", index=False
                )
                pd.DataFrame(noise_results).to_csv(out / "noise_draws.csv", index=False)
                print("Diagnosed", d["name"], model, seed, flush=True)
        if d["name"] == "skbr3":
            for level in protocol["noise_levels"]:
                for draw in range(draws if level else 1):
                    rng = np.random.default_rng(27000 + draw)
                    noisy = (
                        d["control"]
                        + (
                            rng.standard_normal(sd.shape).astype("float32") * sd * level
                        )[plate_ix]
                    )
                    noise_results.append(
                        {
                            "model": "DMSO",
                            "seed": 0,
                            "level": level,
                            "draw": draw,
                            "mse": float(np.square(y - noisy).mean(dtype="float64")),
                            "clean_mse": float(
                                np.square(y - d["control"]).mean(dtype="float64")
                            ),
                            "prediction_rms_change": float(
                                np.sqrt(
                                    np.square(noisy - d["control"]).mean(
                                        dtype="float64"
                                    )
                                )
                            ),
                        }
                    )
            for draw, noisy in enumerate(half_controls):
                empirical.append(
                    {
                        "model": "DMSO",
                        "seed": 0,
                        "draw": draw,
                        "clean_mse": float(
                            np.square(d["y"] - d["control"]).mean(dtype="float64")
                        ),
                        "half_control_mse": float(
                            np.square(d["y"] - noisy).mean(dtype="float64")
                        ),
                        "prediction_rms_change": float(
                            np.sqrt(
                                np.square(noisy - d["control"]).mean(dtype="float64")
                            )
                        ),
                    }
                )
    pd.DataFrame(empirical).to_csv(
        out / "empirical_control_resampling.csv", index=False
    )
    p = pd.DataFrame(permutations)
    n = pd.DataFrame(noise_results)
    p.to_csv(out / "permutation_draws.csv", index=False)
    n.to_csv(out / "noise_draws.csv", index=False)
    assert p.loc[p.model.eq("ControlOnly"), "mse_increase"].eq(0).all()
    p.groupby(["dataset", "model", "seed"]).agg(
        clean_mse=("clean_mse", "first"),
        shuffled_mse=("shuffled_mse", "mean"),
        mse_increase=("mse_increase", "mean"),
        prediction_rms_change=("prediction_rms_change", "mean"),
    ).to_csv(out / "permutation_seed_means.csv")
    n.groupby(["model", "seed", "level"]).agg(
        mse=("mse", "mean"), clean_mse=("clean_mse", "first")
    ).to_csv(out / "noise_seed_means.csv")


if __name__ == "__main__":
    main()
