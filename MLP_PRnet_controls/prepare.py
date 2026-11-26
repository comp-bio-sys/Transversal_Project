"""Prepare common targets and matched-control inputs for all four models."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def digest(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def prepare(expression):
    data = ROOT / "data"
    out = ROOT / "prepared"
    out.mkdir(exist_ok=True)
    manifest = pd.read_feather(data / "sample_manifest.feather")
    spec = json.loads((data / "feature_spec.json").read_text())
    genes = spec["gene_order"]
    expression = Path(expression)
    assert (
        digest(expression) == spec["expression_sha256"]
    ), "Expression file checksum mismatch."
    frame = pd.read_feather(expression, columns=["sample_id", "cell_iname", *genes])
    assert frame.sample_id.is_unique and manifest.sample_id.is_unique
    positions = pd.Index(frame.sample_id).get_indexer(manifest.sample_id)
    assert (positions >= 0).all()
    assert np.array_equal(frame.cell_iname.to_numpy()[positions], manifest.cell_iname)
    y = frame[genes].to_numpy(dtype=np.float32)[positions]
    del frame
    assert np.isfinite(y).all()
    np.save(out / "targets.npy", y)
    controls = np.load(data / "control_means.npy")
    group_names = json.loads((data / "control_groups.json").read_text())
    groups = pd.Index(group_names).get_indexer(manifest.control_group)
    assert (groups >= 0).all() and np.isfinite(controls).all()
    train = manifest.split.eq("train").to_numpy()
    total = np.zeros(978, dtype=np.float64)
    squared = np.zeros(978, dtype=np.float64)
    train_groups = groups[train]
    for start in range(0, len(train_groups), 1024):
        block = controls[train_groups[start : start + 1024]].astype("float64")
        total += block.sum(0)
        squared += (block * block).sum(0)
    center = total / len(train_groups)
    scale = np.sqrt(np.maximum(squared / len(train_groups) - center * center, 0))
    np.savez(
        out / "control_scaler.npz",
        mean=center.astype("float32"),
        std=np.maximum(scale, 1e-3).astype("float32"),
    )
    cells = sorted(manifest.loc[train, "cell_iname"].unique())
    cell_ids = pd.Index(cells).get_indexer(manifest.cell_iname)
    assert (cell_ids >= 0).all()
    compound_ids = pd.Index(spec["structures"]).get_indexer(manifest.structure)
    assert (compound_ids >= 0).all()
    assert manifest.groupby("structure").split.nunique().max() == 1
    baseline = np.stack(
        [
            y[train & (cell_ids == i)].mean(0, dtype=np.float64)
            for i in range(len(cells))
        ]
    ).astype("float32")
    np.savez(
        out / "indices.npz",
        control_ids=groups,
        cell_ids=cell_ids,
        compound_ids=compound_ids,
    )
    np.save(out / "cell_mean_baseline.npy", baseline)
    for split in ["train", "val", "test"]:
        np.save(out / f"{split}_indices.npy", np.flatnonzero(manifest.split.eq(split)))
    audit = {
        "n_cells": len(cells),
        "cells": cells,
        "gene_order": genes,
        "n_genes": len(genes),
        "n_samples": len(y),
        "samples_by_split": manifest.split.value_counts().to_dict(),
        "input_sha256": {
            str(p.relative_to(ROOT)): digest(p)
            for p in [
                data / "sample_manifest.feather",
                data / "control_means.npy",
                data / "chemical_features.npz",
                out / "control_scaler.npz",
            ]
        },
        "expression_sha256": digest(expression),
        "target_scale": "original Level 3; no extra transformation",
        "control_input": "mean of QC-passed DMSO replicates on the same detection plate",
        "control_scaling": "per-gene mean/std fitted on training-paired controls only; std floor 1e-3",
        "test_targets_used_for_training_or_selection": False,
    }
    (out / "audit.json").write_text(json.dumps(audit, indent=2))
    print(
        json.dumps(
            {k: v for k, v in audit.items() if k not in ["cells", "gene_order"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--expression", required=True)
    prepare(parser.parse_args().expression)
