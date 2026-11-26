"""Extract QC-passed SKBR3 profiles from the two Broad Level 3 matrices."""

from pathlib import Path
import fsspec
import h5py
import json
import concurrent.futures
import urllib.request
import numpy as np
import pandas as pd

out = Path(__file__).resolve().parent
source = out.parent / "MLP_PRnet_controls/data/source"
spec = json.loads((source.parent / "feature_spec.json").read_text())
# The base package's download_metadata.py supplies instinfo_beta.txt.
for label, name in [
    ("treated", "level3_beta_trt_cp_n1805898x12328.gctx"),
    ("control", "level3_beta_ctl_n188708x12328.gctx"),
]:
    index_path = source / f"{label}_matrix_index.npz"
    if not index_path.exists():
        with fsspec.open(
            "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/"
            + name,
            "rb",
        ) as remote:
            with h5py.File(remote, "r") as matrix:
                np.savez(
                    index_path,
                    sample_id=matrix["0/META/COL/id"][:].astype(str),
                    gene_id=matrix["0/META/ROW/id"][:].astype(str),
                )
parts = []
for chunk in pd.read_csv(
    source / "instinfo_beta.txt", sep="\t", chunksize=200000, low_memory=False
):
    parts.append(
        chunk.loc[
            chunk.cell_iname.eq("SKBR3")
            & chunk.qc_pass.eq(1)
            & chunk.pert_itime.eq("24 h")
        ]
    )
meta = pd.concat(parts)
ctrl = meta.loc[meta.pert_type.eq("ctl_vehicle") & meta.pert_id.eq("DMSO")]
trt = meta.loc[
    meta.pert_type.eq("trt_cp")
    & meta.pert_idose.eq("10 uM")
    & meta.det_plate.isin(ctrl.det_plate)
]
for name, frame, filename, dtype, offset in [
    ("treated", trt, "level3_beta_trt_cp_n1805898x12328.gctx", "<f4", 6144),
    (
        "control",
        ctrl[ctrl.det_plate.isin(trt.det_plate)],
        "level3_beta_ctl_n188708x12328.gctx",
        "<f8",
        6003,
    ),
]:
    index = np.load(source / f"{name}_matrix_index.npz")
    frame = frame.copy()
    frame["matrix_row"] = pd.Index(index["sample_id"]).get_indexer(frame.sample_id)
    frame = (
        frame.loc[frame.matrix_row.ge(0)]
        .sort_values("matrix_row")
        .reset_index(drop=True)
    )
    cols = pd.Index(index["gene_id"]).get_indexer(spec["gene_order"])
    assert min(cols) >= 0
    rows = frame.matrix_row.to_numpy()
    width = len(index["gene_id"])
    stride = width * np.dtype(dtype).itemsize
    batches = np.split(np.arange(len(rows)), np.where(np.diff(rows) > 8)[0] + 1)
    url = (
        "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/level3/" + filename
    )

    def read(batch):
        start, end = rows[batch[0]], rows[batch[-1]] + 1
        lo, hi = offset + int(start) * stride, offset + int(end) * stride - 1
        req = urllib.request.Request(url, headers={"Range": f"bytes={lo}-{hi}"})
        with urllib.request.urlopen(req, timeout=90) as response:
            assert response.headers["Content-Range"].startswith(f"bytes {lo}-{hi}/")
            vals = np.frombuffer(response.read(), dtype=dtype).reshape(-1, width)
        return vals[rows[batch] - start][:, cols].astype("float32")

    with concurrent.futures.ThreadPoolExecutor(5) as pool:
        values = np.concatenate(list(pool.map(read, batches)))
    assert np.isfinite(values).all()
    np.save(out / f"skbr3_{name}.npy", values)
    frame[
        [
            "sample_id",
            "pert_id",
            "cmap_name",
            "det_plate",
            "cell_iname",
            "pert_itime",
            "pert_idose",
        ]
    ].to_csv(out / f"skbr3_{name}.csv", index=False)
    print(name, values.shape, flush=True)

public_compounds = (
    "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/compoundinfo_beta.txt"
)
compounds = pd.read_csv(public_compounds, sep="\t")
used = pd.read_csv(out / "skbr3_treated.csv").pert_id
structures = (
    compounds.loc[compounds.pert_id.isin(used), ["pert_id", "canonical_smiles"]]
    .dropna()
    .drop_duplicates()
)
assert structures.pert_id.is_unique
structures.to_csv(out / "skbr3_structures.csv", index=False)
