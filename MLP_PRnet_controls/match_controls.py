"""Match DMSO controls to the original selected treatment samples."""

import json
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    source = ROOT / "data/source"
    out = ROOT / "data"
    usage = pd.read_feather(out / "sample_usage.feather")
    selected = usage.loc[usage.used].reset_index(drop=True)
    split = pd.read_csv(out / "compound_split.csv").set_index("structure").split
    assert selected.structure.map(split).eq(selected.split).all()
    cols = [
        "sample_id",
        "pert_id",
        "pert_type",
        "cell_iname",
        "pert_time",
        "det_plate",
        "rna_plate",
        "bead_batch",
        "qc_pass",
        "pert_dose",
        "pert_dose_unit",
    ]
    parts = []
    for chunk in pd.read_csv(
        source / "instinfo_beta.txt",
        sep="\t",
        usecols=cols,
        chunksize=200000,
        low_memory=False,
    ):
        keep = chunk.sample_id.isin(selected.sample_id) | chunk.pert_type.eq(
            "ctl_vehicle"
        )
        parts.append(chunk.loc[keep])
    metadata = pd.concat(parts, ignore_index=True)
    assert metadata.sample_id.is_unique
    treated = selected[["sample_id"]].merge(
        metadata, on="sample_id", how="left", validate="one_to_one"
    )
    assert treated.pert_type.eq("trt_cp").all()
    assert treated.cell_iname.eq(selected.cell_iname).all()
    assert treated.pert_time.eq(24).all()
    index = np.load(source / "control_matrix_index.npz")
    controls = metadata.loc[
        metadata.pert_type.eq("ctl_vehicle")
        & metadata.pert_id.eq("DMSO")
        & metadata.qc_pass.eq(1)
        & metadata.pert_time.eq(24)
        & metadata.sample_id.isin(index["sample_id"])
    ].copy()
    keys = ["det_plate", "cell_iname", "pert_time", "bead_batch"]
    controls["control_group"] = controls[keys].astype(str).agg("|".join, axis=1)
    treated["control_group"] = treated[keys].astype(str).agg("|".join, axis=1)
    counts = controls.groupby("control_group").size()
    selected["control_group"] = treated.control_group
    selected["n_controls"] = selected.control_group.map(counts).fillna(0).astype(int)
    selected["control_eligible"] = selected.n_controls.gt(0)
    selected["original_row"] = np.arange(len(selected))
    train_cells = set(
        selected.loc[
            selected.control_eligible & selected.split.eq("train"), "cell_iname"
        ]
    )
    selected["control_eligible"] &= selected.cell_iname.isin(train_cells)
    selected["exclusion_reason"] = np.where(
        selected.n_controls.eq(0),
        "no_QC_passed_DMSO_on_same_detection_plate",
        np.where(
            selected.cell_iname.isin(train_cells),
            "",
            "cell_absent_from_retained_training",
        ),
    )
    selected.to_feather(out / "cohort_audit.feather")
    retained = selected.loc[selected.control_eligible].reset_index(drop=True)
    assert retained.groupby("structure").split.nunique().max() == 1
    used_controls = controls.loc[
        controls.control_group.isin(retained.control_group)
    ].copy()
    used_controls["matrix_row"] = pd.Index(index["sample_id"]).get_indexer(
        used_controls.sample_id
    )
    used_controls = used_controls.sort_values("matrix_row")
    used_controls.reset_index(drop=True).to_feather(out / "control_manifest.feather")
    retained.to_feather(out / "sample_manifest.feather")
    report = {
        "matching": keys,
        "control_type": "QC-passed DMSO vehicle, 24 h",
        "original_samples": len(selected),
        "retained_samples": len(retained),
        "unique_controls": len(used_controls),
        "control_groups": retained.control_group.nunique(),
        "samples_by_split": retained.split.value_counts().to_dict(),
        "compounds_by_split": retained.groupby("split").structure.nunique().to_dict(),
        "excluded_by_reason": selected.loc[
            ~selected.control_eligible, "exclusion_reason"
        ]
        .value_counts()
        .to_dict(),
        "control_count_distribution": retained.n_controls.value_counts()
        .sort_index()
        .to_dict(),
        "split_assignments_changed": False,
        "target_scale": "original Level 3",
        "matched_control_available_at_prediction": True,
    }
    (out / "matching_audit.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
