"""Secondary stress test: hold out control groups and compounds simultaneously."""

import json
import fcntl
import time
import numpy as np
import pandas as pd
from benchmark import ROOT, BASE
from balanced_train import main as fit, MAIN


def run_holdout():
    out = ROOT / "balanced/group_holdout"
    out.mkdir(parents=True, exist_ok=True)
    manifest = pd.read_feather(BASE / "data/sample_manifest.feather")
    original = np.load(ROOT / "split_indices.npz")
    rng = np.random.default_rng(26026)
    held = []
    for cell, rows in manifest.iloc[original["train"]].groupby("cell_iname"):
        groups = np.array(sorted(rows.control_group.unique()))
        if len(groups) > 1:
            held.extend(
                rng.choice(groups, size=max(1, int(0.2 * len(groups))), replace=False)
            )
    held = np.array(held)
    mask = manifest.control_group.isin(held).to_numpy()
    splits = {
        k: original[k][mask[original[k]] if k == "test" else ~mask[original[k]]]
        for k in ["train", "val", "test"]
    }
    assert not set(manifest.iloc[splits["train"]].control_group) & set(
        manifest.iloc[splits["test"]].control_group
    )
    assert not set(manifest.iloc[splits["train"]].structure) & set(
        manifest.iloc[splits["test"]].structure
    )
    np.savez(out / "split_indices.npz", **splits)
    pd.Series(held, name="held_out_control_group").to_csv(
        out / "held_out_groups.csv", index=False
    )
    # Recompute standardization from this smaller training set only.
    ids = np.load(BASE / "prepared/indices.npz")
    means = np.load(BASE / "data/control_means.npy")
    counts = np.bincount(ids["control_ids"][splits["train"]], minlength=len(means))
    center = np.average(means, axis=0, weights=counts)
    scale = np.maximum(
        np.sqrt(np.average((means - center) ** 2, axis=0, weights=counts)), 0.001
    )
    np.savez(out / "scaling.npz", center=center, scale=scale)
    (out / "protocol.json").write_text(
        json.dumps(
            {
                "control_group_fraction_target": 0.2,
                "random_seed": 26026,
                "sampling": "Within each training cell line; floor(20%) groups, minimum one when at least two groups exist; at least one training group retained.",
                "samples": {k: len(v) for k, v in splits.items()},
                "same_original_compound_assignments": True,
                "test_cells_absent_from_training": sorted(
                    set(manifest.iloc[splits["test"]].cell_iname)
                    - set(manifest.iloc[splits["train"]].cell_iname)
                ),
                "selection": "Predeclared shared default settings; validation NLL; no test-based setting selection",
                "interpretation": "Both compounds and matched control-group identities are absent from fitting; groups need not equal whole detection plates.",
            },
            indent=2,
        )
        + "\n"
    )
    fit(MAIN, [42, 43, 44], out / "weights", 0.001, 0.01, out / "split_indices.npz")


def main():
    while (ROOT / "balanced/.convergence_pending").exists():
        time.sleep(5)
    lock_path = ROOT / "balanced/group_holdout.lock"
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        run_holdout()


if __name__ == "__main__":
    main()
