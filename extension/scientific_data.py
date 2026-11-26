"""Shared data loading and deterministic prediction for the final benchmark."""

import json
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, BASE, predict
from evaluate_project import encode_panel

REP = {"ResidualFCFP4": "FCFP4", "ResidualChemicalVAE": "ChemicalVAE", "PRnet": "FCFP4"}


def load_internal(indices=None):
    ix = np.load(ROOT / "split_indices.npz")["test"] if indices is None else indices
    m = (
        pd.read_feather(BASE / "data/sample_manifest.feather")
        .iloc[ix]
        .reset_index(drop=True)
    )
    ids = np.load(BASE / "prepared/indices.npz")
    fp = np.load(BASE / "data/chemical_features.npz")
    return {
        "name": "internal",
        "y": np.load(BASE / "prepared/targets.npy", mmap_mode="r")[ix],
        "control": np.load(BASE / "data/control_means.npy")[ids["control_ids"][ix]],
        "features": {k: fp[k][ids["compound_ids"][ix]] for k in fp.files},
        "structure": m.structure.to_numpy(),
        "cell": m.cell_iname.to_numpy(),
        "group": m.control_group.to_numpy(),
        "sample_id": m.sample_id.to_numpy(),
    }


def load_skbr3():
    m = pd.read_csv(ROOT / "skbr3_treated.csv").merge(
        pd.read_csv(ROOT / "skbr3_structures.csv"),
        on="pert_id",
        how="left",
        validate="many_to_one",
        sort=False,
    )
    keep = m.canonical_smiles.notna() & m.canonical_smiles.ne("restricted")
    y = np.load(ROOT / "skbr3_treated.npy")[keep.to_numpy()]
    m = m.loc[keep].reset_index(drop=True)
    cm = pd.read_csv(ROOT / "skbr3_control.csv")
    raw = np.load(ROOT / "skbr3_control.npy")
    groups = cm.groupby("det_plate").indices
    means = {g: raw[ix].mean(0) for g, ix in groups.items()}
    features = {}
    for v in ["Residual", "ResidualFCFP4", "ResidualChemicalVAE"]:
        f, ok = encode_panel(m.canonical_smiles, v)
        assert ok.all()
        features[REP.get(v, "ECFP4")] = f
    return {
        "name": "skbr3",
        "y": y,
        "control": np.stack([means[g] for g in m.det_plate]),
        "features": features,
        "structure": m.canonical_smiles.to_numpy(),
        "cell": m.cell_iname.to_numpy(),
        "group": m.det_plate.to_numpy(),
        "sample_id": m.sample_id.to_numpy(),
        "control_metadata": cm,
        "raw_controls": raw,
    }


def infer(model, variant, fp, control, scaling_path=None, noise_draws=32):
    s = np.load(scaling_path or ROOT / "scaling.npz")
    center, scale = [
        torch.from_numpy(s[k].astype("float32")) for k in ["center", "scale"]
    ]
    model.eval()
    result = []
    draws = noise_draws if variant == "PRnet" else 1
    rngs = [np.random.default_rng(1042 + j) for j in range(draws)]
    with torch.no_grad():
        for start in range(0, len(control), 256):
            c = torch.from_numpy(control[start : start + 256])
            f = torch.from_numpy(fp[start : start + 256])
            total = None
            for rng in rngs:
                z = torch.from_numpy(
                    rng.standard_normal((len(c), 10)).astype("float32")
                )
                mu, _ = predict(model, variant, f, c, center, scale, z)
                total = mu if total is None else total + mu
            result.append((total / draws).numpy())
    return np.concatenate(result)


def metric_rows(data, mu):
    y = data["y"]
    c = data["control"]
    true = y - c
    change = mu - c
    mask = np.abs(true) >= 0.5
    mse = np.square(y - mu).mean(1, dtype="float64")
    base = np.square(true).mean(1, dtype="float64")
    tx = true - true.mean(1, keepdims=True)
    px = change - change.mean(1, keepdims=True)
    den = np.sqrt(np.square(tx).sum(1) * np.square(px).sum(1))
    corr = np.divide(
        (tx * px).sum(1), den, out=np.full(len(y), np.nan), where=den > 1e-12
    )
    frame = pd.DataFrame(
        {
            "sample_id": data["sample_id"],
            "structure": data["structure"],
            "group": data["group"],
            "mse": mse,
            "dmso_mse": base,
            "response_pearson": corr,
            "direction_correct": ((np.sign(change) == np.sign(true)) & mask).sum(1),
            "direction_scored": mask.sum(1),
        }
    )
    summary = {
        "profiles": len(y),
        "compounds": frame.structure.nunique(),
        "control_groups": frame.group.nunique(),
        "mse": float(mse.mean()),
        "macro_compound_mse": float(frame.groupby("structure").mse.mean().mean()),
        "mse_skill_vs_dmso": float(1 - mse.mean() / base.mean()),
        "mean_response_pearson": (
            float(np.nanmean(corr)) if np.isfinite(corr).any() else None
        ),
        "direction_accuracy": (
            float(frame.direction_correct.sum() / frame.direction_scored.sum())
            if frame.direction_scored.sum()
            else None
        ),
    }
    return summary, frame


def mean_prediction(data, cell_specific=False):
    z = np.load(ROOT / "balanced/mean_baselines.npz")
    global_mean = z["global_mean"]
    if not cell_specific:
        return data["control"] + global_mean
    lookup = dict(zip(z["cells"], z["cell_means"]))
    return data["control"] + np.stack(
        [lookup.get(c, global_mean) for c in data["cell"]]
    )


def ridge_prediction(data):
    z = np.load(ROOT / "balanced/ridge.npz")
    s = np.load(ROOT / "scaling.npz")
    result = []
    coef = torch.from_numpy(z["coef"])
    intercept = torch.from_numpy(z["intercept"])
    with torch.no_grad():
        for start in range(0, len(data["control"]), 512):
            c = data["control"][start : start + 512]
            x = np.concatenate(
                [
                    data["features"]["ECFP4"][start : start + 512],
                    (c - s["center"]) / s["scale"],
                ],
                axis=1,
            ).astype("float32")
            result.append((torch.from_numpy(x) @ coef + intercept).numpy() + c)
    return np.concatenate(result)


def load_external():
    here = ROOT.parent / "dose_matched"
    studies = pd.read_csv(here / "study_manifest.csv")
    m = pd.read_feather(BASE / "data/sample_manifest.feather")
    ids = np.load(BASE / "prepared/indices.npz")
    means = np.load(BASE / "data/control_means.npy")
    groups = m[["cell_iname"]].assign(group=ids["control_ids"]).drop_duplicates()
    refs = {c: means[g.group].mean(0) for c, g in groups.groupby("cell_iname")}
    features = {}
    for variant in ["Residual", "ResidualFCFP4", "ResidualChemicalVAE"]:
        fp, keep = encode_panel(studies.structure, variant)
        assert keep.all()
        features[REP.get(variant, "ECFP4")] = fp
    return {
        "control": np.stack([refs[c] for c in studies.cell]),
        "features": features,
        "cell": studies.cell.to_numpy(),
        "studies": studies,
        "measured": pd.read_csv(here / "measured_effects.csv"),
    }
