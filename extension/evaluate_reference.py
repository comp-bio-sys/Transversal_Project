"""Evaluate the separate PRnet source-recipe checkpoint on dose-matched studies."""

import hashlib
import json

import numpy as np
import pandas as pd
import torch

from benchmark import BASE, ROOT, build
from evaluate_project import GENES, encode_panel

NOISE_SEED = 1042
NOISE_DRAWS = 32
ORIGINAL_PREDICTIONS_SHA256 = (
    "e7301c26d8c5259ff9230d46e853fe8a5bae1001accef84f98b4dda02079b790"
)


def main():
    torch.set_num_threads(2)
    source = ROOT.parent / "dose_matched"
    output = ROOT / "reference"
    checkpoint = output / "PRnet.pt"
    studies = pd.read_csv(source / "study_manifest.csv")
    measured = pd.read_csv(source / "measured_effects.csv")
    published = pd.read_csv(source / "published_effects.csv")
    measured = measured.merge(
        published[["drug_id", "genes", "logFC"]],
        on=["drug_id", "genes"],
        how="left",
        validate="one_to_one",
    )

    manifest = pd.read_feather(BASE / "data/sample_manifest.feather")
    indices = np.load(BASE / "prepared/indices.npz")
    means = np.load(BASE / "data/control_means.npy")
    groups = (
        manifest[["cell_iname"]].assign(group=indices["control_ids"]).drop_duplicates()
    )
    reference = {
        cell: means[group.group].mean(0) for cell, group in groups.groupby("cell_iname")
    }
    controls = np.stack([reference[cell] for cell in studies.cell])

    training = json.loads((output / "summary.json").read_text())
    total = training["training_median_total"]
    controls = np.log1p(controls * (total / controls.sum(1, keepdims=True))).astype(
        "float32"
    )
    features, supported = encode_panel(studies.structure, "PRnet")
    assert supported.all()
    control = torch.from_numpy(controls)
    features = torch.from_numpy(features)
    noise = torch.from_numpy(
        np.random.default_rng(NOISE_SEED)
        .standard_normal((NOISE_DRAWS, len(studies), 10))
        .astype("float32")
    )

    model = build("PRnet")
    model.load_state_dict(torch.load(checkpoint, map_location="cpu", weights_only=True))
    model.eval()
    with torch.no_grad():
        prediction = torch.stack(
            [model(features, None, control, draw)[0] for draw in noise]
        ).mean(0)

    gene_order = json.loads((BASE / "data/feature_spec.json").read_text())["gene_order"]
    genes = list(GENES)
    columns = [gene_order.index(GENES[gene]) for gene in genes]
    change = (prediction - control).numpy()[:, columns].astype("float64")
    rows = pd.DataFrame(change, columns=genes).assign(drug_id=studies.drug_id)
    rows = rows.melt(
        id_vars="drug_id", var_name="genes", value_name="predicted_change"
    ).assign(model="PRnet_source", seed=42)
    rows = rows.merge(measured, on=["drug_id", "genes"], validate="many_to_one")
    rows["correct"] = np.sign(rows.measured_change) == np.sign(rows.predicted_change)
    eligible = rows.loc[rows.chemical_split.eq("eligible") & rows.measured_change.ne(0)]
    assert np.isfinite(rows.predicted_change).all()
    assert len(rows) == 47 and len(eligible) == 42

    rows.to_csv(output / "dose_matched_predictions.csv", index=False)
    summary = {
        "model": "PRnet_source",
        "checkpoint": str(checkpoint.relative_to(ROOT.parent)),
        "checkpoint_sha256": hashlib.sha256(checkpoint.read_bytes()).hexdigest(),
        "training_seed": 42,
        "decoder_noise_seed": NOISE_SEED,
        "decoder_draws": NOISE_DRAWS,
        "normalization": "Cell-matched mean controls are rescaled to the training median total and transformed with natural log1p.",
        "training_median_total": total,
        "predicted_response": "Mean of decoder predictions minus the transformed control.",
        "all_measured_rows": len(rows),
        "eligible_filter": "chemical_split == 'eligible' and measured_change != 0",
        "eligible_directions": len(eligible),
        "eligible_studies": int(eligible.gse_id.nunique()),
        "correct_directions": int(eligible.correct.sum()),
        "accuracy": float(eligible.correct.mean()),
        "original_predictions_sha256": ORIGINAL_PREDICTIONS_SHA256,
        "scope": "Separate source-recipe sensitivity analysis; not pooled with the shared-budget benchmark.",
    }
    (output / "dose_matched_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
