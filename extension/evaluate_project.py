"""Gene identifiers and molecular features for external evaluation."""

import json

import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

from benchmark import BASE

GENES = {
    "AKT1": "207",
    "AURKB": "9212",
    "EGFR": "1956",
    "PIK3CA": "5290",
    "SMAD3": "4088",
    "TP53": "7157",
}


def fingerprints(structures, feature):
    args = (
        {"atomInvariantsGenerator": rdFingerprintGenerator.GetMorganFeatureAtomInvGen()}
        if feature
        else {}
    )
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024, **args)
    return np.stack(
        [gen.GetFingerprintAsNumPy(Chem.MolFromSmiles(s)) for s in structures]
    ).astype("float32")


def encode_panel(series, variant):
    structures = [Chem.MolToSmiles(Chem.MolFromSmiles(s)) for s in series]
    if variant == "ResidualChemicalVAE":
        from encoder import encode

        vocab = set(json.loads((BASE / "encoder/zinc.json").read_text()))
        keep = np.array([len(s) <= 120 and set(s) <= vocab for s in structures])
        unique, ix = np.unique(np.array(structures)[keep], return_inverse=True)
        return encode(unique.tolist()).numpy()[ix], keep
    return (
        fingerprints(structures, variant in ["PRnet", "ResidualFCFP4"]),
        np.ones(len(series), dtype=bool),
    )
