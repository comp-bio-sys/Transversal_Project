"""Describe separation of compounds, cell lines and control groups."""

import json
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator
from benchmark import ROOT, BASE


def main():
    out = ROOT / "balanced"
    out.mkdir(exist_ok=True)
    m = pd.read_feather(BASE / "data/sample_manifest.feather")
    splits = np.load(ROOT / "split_indices.npz")
    rows = {k: m.iloc[ix] for k, ix in splits.items()}
    train = rows["train"]
    test = rows["test"]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    structures = train.structure.unique()
    fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in structures]
    detail = []
    for s in test.structure.unique():
        values = DataStructs.BulkTanimotoSimilarity(
            gen.GetFingerprint(Chem.MolFromSmiles(s)), fps
        )
        nearest = int(np.argmax(values))
        detail.append(
            {
                "structure": s,
                "max_training_tanimoto": values[nearest],
                "nearest_training_structure": structures[nearest],
            }
        )
    d = pd.DataFrame(detail)
    d.to_csv(out / "internal_chemical_separation.csv", index=False)
    connectivity = {
        Chem.MolToSmiles(Chem.MolFromSmiles(t), isomericSmiles=False)
        for t in structures
    }
    report = {
        "samples": {k: len(v) for k, v in rows.items()},
        "compounds": {k: v.structure.nunique() for k, v in rows.items()},
        "test_profiles_with_control_group_seen_in_train": int(
            test.control_group.isin(set(train.control_group)).sum()
        ),
        "test_control_groups_seen_in_train": len(
            set(test.control_group) & set(train.control_group)
        ),
        "test_control_groups": test.control_group.nunique(),
        "test_cells_absent_from_training": sorted(
            set(test.cell_iname) - set(train.cell_iname)
        ),
        "test_compounds_above_0_7_similarity_to_train": int(
            d.max_training_tanimoto.gt(0.7).sum()
        ),
        "test_compounds_with_identical_ECFP4_to_training": int(
            d.max_training_tanimoto.eq(1).sum()
        ),
        "test_compounds_with_matching_connectivity_ignoring_stereochemistry": int(
            sum(
                Chem.MolToSmiles(Chem.MolFromSmiles(s), isomericSmiles=False)
                in connectivity
                for s in d.structure
            )
        ),
        "interpretation": "Compound-disjoint split. Shared measured controls do not by themselves imply target leakage; batch and scaffold separation are distinct tests.",
    }
    sk = pd.read_csv(ROOT / "skbr3_treated.csv").merge(
        pd.read_csv(ROOT / "skbr3_structures.csv"), on="pert_id", validate="many_to_one"
    )
    sk = sk.loc[sk.canonical_smiles.notna() & sk.canonical_smiles.ne("restricted")]
    exposure = []
    for structure in sk.canonical_smiles.unique():
        canonical = Chem.MolToSmiles(Chem.MolFromSmiles(structure))
        exposure.append(
            {
                "structure": structure,
                "seen_in_training": canonical in set(structures),
                "max_training_tanimoto": max(
                    DataStructs.BulkTanimotoSimilarity(
                        gen.GetFingerprint(Chem.MolFromSmiles(structure)), fps
                    )
                ),
            }
        )
    exposure = pd.DataFrame(exposure)
    exposure.to_csv(out / "skbr3_chemical_exposure.csv", index=False)
    report["skbr3_compounds_seen_in_training"] = int(exposure.seen_in_training.sum())
    report["skbr3_compounds_not_seen_in_training"] = int(
        (~exposure.seen_in_training).sum()
    )
    (out / "generalization_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
