"""Average measured DMSO replicates within each matched control group."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    data = ROOT / "data"
    manifest = pd.read_feather(data / "control_manifest.feather")
    audit = json.loads((data / "control_download_audit.json").read_text())
    assert audit["all_ranges_verified"]
    values = np.load(data / "control_expression.npy", mmap_mode="r")
    assert values.shape == (len(manifest), 978) and np.isfinite(values).all()
    groups = sorted(manifest.control_group.unique())
    means = np.stack(
        [
            values[ix].mean(0, dtype=np.float64)
            for _, ix in manifest.groupby("control_group", sort=True).groups.items()
        ]
    ).astype("float32")
    assert len(means) == len(groups)
    np.save(data / "control_means.npy", means)
    (data / "control_groups.json").write_text(json.dumps(groups, indent=2))
    print(f"{len(groups)} control means, each with 978 genes")


if __name__ == "__main__":
    main()
