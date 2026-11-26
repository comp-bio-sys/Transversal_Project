"""Check the final comparison against sample assignments, measurements and logs."""

import ast
import hashlib
import json

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import rdFingerprintGenerator

from benchmark import ROOT, BASE
from balanced_train import MAIN, ABLATIONS


def main():
    root = ROOT / "balanced"
    published = ast.parse((BASE / "vendor/PRnet/models/PRnet.py").read_text())
    implemented = ast.parse((BASE / "prnet_core.py").read_text())
    classes = lambda tree: {
        n.name: ast.dump(n, include_attributes=False)
        for n in tree.body
        if isinstance(n, ast.ClassDef) and n.name != "PRnet"
    }
    assert classes(published) == classes(implemented)
    m = pd.read_feather(BASE / "data/sample_manifest.feather")
    original = pd.read_feather(BASE / "data/sample_usage.feather")
    original = original.loc[original.used].set_index("sample_id")
    assert m.sample_id.is_unique
    assert original.loc[m.sample_id, "split"].to_numpy().tolist() == m.split.tolist()
    splits = np.load(ROOT / "split_indices.npz")
    compound_sets = []
    for key, ix in splits.items():
        rows = m.iloc[ix]
        assert rows.split.eq(key).all()
        assert not rows.cell_iname.isin(["SKBR3", "SKB"]).any()
        assert len(np.unique(ix)) == len(ix)
        compound_sets.append(set(rows.structure))
    assert all(
        compound_sets[i].isdisjoint(compound_sets[j])
        for i in range(3)
        for j in range(i)
    )
    metadata = pd.read_feather(
        BASE / "l1000_level3_filtered_train_clean.feather",
        columns=["sample_id", "pert_id"],
    ).set_index("sample_id")
    perturbagen = m.sample_id.map(metadata.pert_id)
    assert perturbagen.notna().all()
    perturbagen_sets = [
        set(perturbagen.iloc[splits[k]]) for k in ["train", "val", "test"]
    ]
    assert all(
        perturbagen_sets[i].isdisjoint(perturbagen_sets[j])
        for i in range(3)
        for j in range(i)
    )
    features = np.load(BASE / "data/chemical_features.npz")
    assert features["FCFP4"].shape[1] == features["ECFP4"].shape[1] == 1024
    assert features["ChemicalVAE"].shape[1] == 196
    assert not np.array_equal(features["FCFP4"], features["ECFP4"])
    assert all(np.isfinite(features[k]).all() for k in features.files)
    assert np.load(BASE / "prepared/targets.npy", mmap_mode="r").shape == (len(m), 978)
    sk = pd.read_csv(ROOT / "skbr3_treated.csv")
    sk_controls = pd.read_csv(ROOT / "skbr3_control.csv")
    assert sk.cell_iname.eq("SKBR3").all() and sk_controls.cell_iname.eq("SKBR3").all()
    assert set(sk.sample_id).isdisjoint(set(m.sample_id))
    assert sk.pert_idose.eq("10 uM").all() and sk.pert_itime.eq("24 h").all()
    assert sk_controls.pert_id.eq("DMSO").all()
    assert sk.det_plate.isin(sk_controls.det_plate).all()
    assert len(sk) == len(np.load(ROOT / "skbr3_treated.npy")) == 580
    assert len(sk_controls) == len(np.load(ROOT / "skbr3_control.npy")) == 226
    studies = pd.read_csv(ROOT.parent / "dose_matched/study_manifest.csv")
    eligible = studies.loc[studies.chemical_split.eq("eligible")]
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    fp = lambda s: gen.GetFingerprint(Chem.MolFromSmiles(s))
    trainfp = [fp(s) for s in m.iloc[splits["train"]].structure.unique()]
    maximum = [
        max(DataStructs.BulkTanimotoSimilarity(fp(s), trainfp))
        for s in eligible.structure
    ]
    assert max(maximum) <= 0.7
    np.testing.assert_allclose(maximum, eligible.max_train_similarity, atol=1e-12)
    measured = pd.read_csv(ROOT.parent / "dose_matched/measured_effects.csv")
    measured = measured.loc[measured.chemical_split.eq("eligible")]
    assert (
        len(measured) == 42
        and measured.gse_id.nunique() == 7
        and measured.genes.nunique() == 6
    )
    assert measured.groupby("gse_id").size().eq(6).all()
    assert measured.clear_change.equals(
        measured.measured_change.abs().ge(0.5) & measured.separated_replicates
    )
    joint = np.load(root / "group_holdout/split_indices.npz")
    for k in joint.files:
        assert set(joint[k]) <= set(splits[k])
    held = set(m.iloc[joint["test"]].control_group)
    assert held.isdisjoint(set(m.iloc[joint["train"]].control_group))
    assert held.isdisjoint(set(m.iloc[joint["val"]].control_group))
    ids = np.load(BASE / "prepared/indices.npz")
    controls = np.load(BASE / "data/control_means.npy")
    counts = np.bincount(ids["control_ids"][joint["train"]], minlength=len(controls))
    center = np.average(controls, axis=0, weights=counts)
    scale = np.maximum(
        np.sqrt(np.average((controls - center) ** 2, axis=0, weights=counts)), 0.001
    )
    actual = np.load(root / "group_holdout/scaling.npz")
    np.testing.assert_allclose(actual["center"], center)
    np.testing.assert_allclose(actual["scale"], scale)
    full_counts = np.bincount(
        ids["control_ids"][splits["train"]], minlength=len(controls)
    )
    full_center = np.average(controls, axis=0, weights=full_counts)
    full_scale = np.maximum(
        np.sqrt(np.average((controls - full_center) ** 2, axis=0, weights=full_counts)),
        0.001,
    )
    full_scaling = np.load(ROOT / "scaling.npz")
    np.testing.assert_allclose(full_scaling["center"], full_center)
    np.testing.assert_allclose(full_scaling["scale"], full_scale)

    selected = json.loads((root / "selected_settings.json").read_text())
    tuning = pd.read_csv(root / "tuning_validation.csv")
    for model in MAIN:
        assert (
            selected[model]
            == tuning.loc[tuning.model.eq(model)]
            .sort_values(["val_nll", "setting"])
            .iloc[0]
            .setting
        )
    checked = []
    paths = (
        list((root / "weights").glob("*_*.pt"))
        + list((root / "tuning").glob("*/*_*.pt"))
        + list((root / "group_holdout/weights").glob("*_*.pt"))
    )
    for path in paths:
        meta = json.loads(path.with_suffix(".json").read_text())
        history = pd.read_csv(path.with_suffix(".csv"))
        assert np.isfinite(history.val_nll).all()
        best = history.loc[history.val_nll.idxmin()]
        assert int(best.epoch) == meta["best_epoch"]
        np.testing.assert_allclose(best.val_nll, meta["val_nll"], atol=1e-12)
        last_epoch = int(history.epoch.max())
        assert last_epoch - meta["best_epoch"] >= 7 or last_epoch == 150
        checked.append(
            {
                "model": path.stem.rsplit("_", 1)[0],
                "seed": int(path.stem.rsplit("_", 1)[1]),
                "path": str(path.relative_to(ROOT.parent)),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "best_epoch": meta["best_epoch"],
                "val_nll": meta["val_nll"],
            }
        )
    suffix = "fixed_tuned_group_holdout_baselines"
    scores = pd.read_csv(root / f"results/summary_{suffix}.csv")
    assert not scores.duplicated(["phase", "dataset", "model", "seed"]).any()
    assert np.isfinite(scores.mse).all() and scores.mse.ge(0).all()
    for model in MAIN:
        for dataset in ["internal", "skbr3"]:
            assert set(
                scores.loc[
                    scores.phase.eq("tuned")
                    & scores.model.eq(model)
                    & scores.dataset.eq(dataset),
                    "seed",
                ]
            ) == {42, 43, 44}
    predictions = pd.read_csv(root / f"results/external_{suffix}.csv")
    assert np.array_equal(
        np.sign(predictions.predicted_change) == np.sign(predictions.measured_change),
        predictions.correct,
    )
    e = predictions.loc[predictions.chemical_split.eq("eligible")]
    assert e.groupby(["phase", "model", "seed"]).size().eq(42).all()
    p = pd.read_csv(root / "diagnostics/permutation_draws.csv")
    n = pd.read_csv(root / "diagnostics/noise_draws.csv")
    assert p.groupby(["dataset", "model", "seed"]).size().eq(20).all()
    assert p.loc[p.model.eq("ControlOnly"), "mse_increase"].eq(0).all()
    assert p.loc[p.model.eq("ControlOnly"), "prediction_rms_change"].eq(0).all()
    strict_scores = pd.read_csv(root / "summary/low_similarity_seed_scores.csv")
    assert (
        strict_scores.profiles.eq(16531).all() and strict_scores.compounds.eq(474).all()
    )
    expected = pd.concat([scores, strict_scores], ignore_index=True)
    clean = p.groupby(["dataset", "model", "seed"]).clean_mse.first().reset_index()
    joined = clean.merge(
        expected.loc[expected.phase.eq("tuned"), ["dataset", "model", "seed", "mse"]],
        on=["dataset", "model", "seed"],
        validate="one_to_one",
    )
    assert len(joined) == len(clean)
    np.testing.assert_allclose(joined.clean_mse, joined.mse, atol=1e-7, rtol=1e-7)

    assert np.isfinite(n.mse).all()
    assert n.loc[n.level.eq(0), "prediction_rms_change"].eq(0).all()
    empirical = pd.read_csv(root / "diagnostics/empirical_control_resampling.csv")
    assert empirical.groupby(["model", "seed"]).size().eq(20).all()
    assert np.isfinite(empirical.half_control_mse).all()
    cal = pd.read_csv(root / "results/calibration.csv")
    assert len(cal) == 30 and cal.covered95.between(0, 1).all()
    report = {
        "status": "passed",
        "published_PRnet_network_classes_match": True,
        "training_compounds": len(trainfp),
        "max_similarity_to_eligible_external": max(maximum),
        "external_studies": 7,
        "external_directions": 42,
        "clear_directions": int(measured.clear_change.sum()),
        "joint_split_rows": {k: len(joint[k]) for k in joint.files},
        "checked_checkpoints": len(checked),
        "historical_tests_reused": True,
        "interpretation": "Verified implementation and stored outcomes; this is not independent replication.",
    }
    (root / "verification.json").write_text(json.dumps(report, indent=2) + "\n")
    pd.DataFrame(checked).to_csv(root / "checkpoint_manifest.csv", index=False)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
