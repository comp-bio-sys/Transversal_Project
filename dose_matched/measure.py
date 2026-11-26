"""Reconstruct treatment-minus-control expression independently of model outputs."""

import csv
import gzip
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "source"
GENES = {
    "AKT1": 207,
    "AURKB": 9212,
    "EGFR": 1956,
    "PIK3CA": 5290,
    "SMAD3": 4088,
    "TP53": 7157,
}


def marked_table(path, marker, end):
    with gzip.open(path, "rt", errors="replace") as stream:
        for line in stream:
            if line.lower().startswith(marker.lower()):
                break
        else:
            raise ValueError(f"Missing {marker} in {path}")
        import io

        rows = []
        for line in stream:
            if line.lower().startswith(end.lower()):
                break
            rows.append(line)
    return pd.read_csv(io.StringIO("".join(rows)), sep="\t", dtype=str)


def measure():
    experiments = pd.read_csv(ROOT / "study_manifest.csv")
    observations = []
    effects = []
    evidence = []
    for study in experiments.itertuples():
        controls = study.ctrl_ids.split("|")
        treated = study.pert_ids.split("|")
        ids = controls + treated
        series = SOURCE / f"{study.gse_id}_series_matrix.txt.gz"
        with gzip.open(series, "rt") as stream:
            metadata = []
            for row in csv.reader(stream, delimiter="\t"):
                if row and row[0] == "!series_matrix_table_begin":
                    break
                if row:
                    metadata.append(row)
        samples = next(r[1:] for r in metadata if r[0] == "!Sample_geo_accession")
        for sample in ids:
            col = 1 + samples.index(sample)
            for row in metadata:
                if row[0] in [
                    "!Sample_title",
                    "!Sample_source_name_ch1",
                    "!Sample_characteristics_ch1",
                    "!Sample_treatment_protocol_ch1",
                    "!Sample_description",
                ]:
                    evidence.append(
                        (
                            study.gse_id,
                            sample,
                            "control" if sample in controls else "treated",
                            row[0],
                            row[col],
                        )
                    )
        count_path = SOURCE / f"{study.gse_id}_counts.gz"
        if count_path.exists():
            raw = pd.read_csv(count_path, sep="\t", index_col=0)[ids]
            assert raw.index.is_unique and (raw >= 0).all().all()
            positive = raw.loc[raw.min(axis=1) > 0]
            geom = np.exp(np.log(positive).mean(axis=1))
            sizes = positive.div(geom, axis=0).median(axis=0)
            assert sizes.gt(0).all()
            transformed = np.log2(raw.div(sizes, axis=1) + 0.5)
            values = {
                g: transformed.loc[i]
                for g, i in GENES.items()
                if i in raw.index and raw.loc[i].sum() > 0
            }
            scale = "log2 size-factor-normalized gene count + pseudocount 0.5"
        else:
            probes = (
                marked_table(
                    series, "!series_matrix_table_begin", "!series_matrix_table_end"
                )
                .set_index("ID_REF")[ids]
                .astype(float)
            )
            if study.gse_id == "GSE2450":
                probes = np.log2(probes.clip(lower=1))
            annotation = marked_table(
                SOURCE / f"{study.GPL}.annot.gz",
                "!platform_table_begin",
                "!platform_table_end",
            )
            values = {}
            for gene in GENES:
                rows = annotation.loc[annotation["Gene symbol"].eq(gene), "ID"]
                matches = probes.index.intersection(rows)
                if len(matches):
                    values[gene] = probes.loc[matches].median()
            scale = "per-sample median of uniquely mapped log2 probe intensities"
        for gene, v in values.items():
            assert np.isfinite(v).all()
            delta = float(v[treated].mean() - v[controls].mean())
            contrasts = v[treated].to_numpy()[:, None] - v[controls].to_numpy()
            consistent = bool((contrasts * np.sign(delta) > 0).all())
            effects.append(
                (
                    study.drug_id,
                    study.gse_id,
                    study.cell,
                    study.drug_name,
                    gene,
                    len(controls),
                    len(treated),
                    delta,
                    consistent,
                    abs(delta) >= 0.5 and consistent,
                    study.chemical_split,
                    scale,
                )
            )
            for sample in ids:
                observations.append(
                    (
                        study.drug_id,
                        gene,
                        sample,
                        "treated" if sample in treated else "control",
                        v[sample],
                    )
                )
    pd.DataFrame(
        effects,
        columns="drug_id gse_id cell drug genes n_control n_treated measured_change separated_replicates clear_change chemical_split expression_scale".split(),
    ).to_csv(ROOT / "measured_effects.csv", index=False)
    pd.DataFrame(
        observations, columns=["drug_id", "genes", "sample_id", "arm", "log_expression"]
    ).to_csv(ROOT / "replicate_values.csv", index=False)
    pd.DataFrame(
        evidence, columns=["gse_id", "sample_id", "arm", "field", "value"]
    ).to_csv(ROOT / "geo_metadata.csv", index=False)


if __name__ == "__main__":
    measure()
