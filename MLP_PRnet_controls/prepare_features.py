"""Verify or regenerate the three fixed chemical representations."""

import argparse
import json
from pathlib import Path
import numpy as np
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from encoder import encode

ROOT = Path(__file__).resolve().parent


def main(write=False):
    torch.set_num_threads(4)
    data = ROOT / "data"
    structures = json.loads((data / "feature_spec.json").read_text())["structures"]
    molecules = [Chem.MolFromSmiles(s) for s in structures]
    assert all(mol is not None for mol in molecules)
    feature = rdFingerprintGenerator.GetMorganGenerator(
        radius=2,
        fpSize=1024,
        atomInvariantsGenerator=rdFingerprintGenerator.GetMorganFeatureAtomInvGen(),
    )
    atom = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
    values = {
        "FCFP4": np.stack([feature.GetFingerprintAsNumPy(m) for m in molecules]).astype(
            "float32"
        ),
        "ECFP4": np.stack([atom.GetFingerprintAsNumPy(m) for m in molecules]).astype(
            "float32"
        ),
        "ChemicalVAE": encode(structures).numpy(),
    }
    path = data / "chemical_features.npz"
    if write:
        np.savez_compressed(path, **values)
    else:
        original = np.load(path)
        report = {}
        for name, value in values.items():
            np.testing.assert_allclose(value, original[name], atol=1e-5, rtol=1e-5)
            report[name] = {
                "compounds": len(value),
                "max_abs_difference": float(np.abs(value - original[name]).max()),
            }
        (data / "feature_verification.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--write", action="store_true")
    main(parser.parse_args().write)
