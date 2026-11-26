"""Curate 24-hour PharmGEO comparisons before examining model predictions."""

import json
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit import DataStructs
import numpy as np

out = Path(__file__).resolve().parent
raw = Path("tmp/pharm")
base = out.parent / "MLP_PRnet_controls/data"
m = pd.read_csv(raw / "metadata.csv", skiprows=14)
genes = pd.read_csv(raw / "six.csv")
m = m.loc[m.organism.eq("Homo sapiens")].copy()
dose = (
    m.dose.str.lower()
    .str.replace(r"<u\+00b5>|<u\+03bc>|μ|µ", "u", regex=True)
    .str.replace(" ", "")
)
duration = m.duration.str.lower().str.replace(" ", "")
m["exact_dose"] = dose.isin(["10um", "10.0um", "10000nm", "0.01mm"])
m = m.loc[duration.isin(["24h", "24hr", "24hrs", "24hours", "1day"])]
cells = pd.read_feather(
    base / "sample_manifest.feather"
).cell_iname.unique().tolist() + ["SKBR3"]


def norm(s):
    return "".join(c for c in str(s).upper() if c.isalnum())


cell_map = {norm(c): c for c in cells}
m["cell"] = m.cell_type.map(lambda c: cell_map.get(norm(c)))
drugs = (
    pd.read_csv(raw / "structures.csv")
    .dropna(subset=["InChI"])
    .drop_duplicates(["Drug", "InChI"])
)
drugs = drugs.loc[~drugs.Drug.duplicated(keep=False)]
m = m.merge(
    drugs[["Drug", "InChI", "CID"]],
    left_on="drug_name",
    right_on="Drug",
    how="left",
    validate="many_to_one",
)


def structure(value):
    if pd.isna(value):
        return None
    molecule = Chem.MolFromInchi(value)
    return Chem.MolToSmiles(molecule) if molecule else None


m["structure"] = m.InChI.map(structure)
m.to_csv(out / "external_condition_audit.csv", index=False)
external = m.dropna(subset=["cell", "structure"]).merge(genes, on="drug_id")
external = external.loc[external.logFC.ne(0) & external.logFC.notna()].drop_duplicates()
external.to_csv(out / "external_labels.csv", index=False)
print(external[["cell", "drug_name", "genes"]].value_counts().to_string())
gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=1024)
structures = json.loads((base / "feature_spec.json").read_text())["structures"]
fps = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in structures]
ref = [gen.GetFingerprint(Chem.MolFromSmiles(s)) for s in external.structure.unique()]
assert ref
scores = np.maximum.reduce([DataStructs.BulkTanimotoSimilarity(fp, fps) for fp in ref])
pd.DataFrame({"structure": structures, "max_external_tanimoto": scores}).to_csv(
    out / "similarities.csv", index=False
)
