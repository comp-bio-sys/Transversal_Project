"""Download public LINCS 2020 metadata and the control matrix index."""

import json
import urllib.request
from pathlib import Path
import fsspec
import h5py
import numpy as np

ROOT = Path(__file__).resolve().parent
BASE = "https://s3.amazonaws.com/macchiato.clue.io/builds/LINCS2020/"


def main():
    source = ROOT / "data/source"
    source.mkdir(parents=True, exist_ok=True)
    destination = source / "instinfo_beta.txt"
    if not destination.exists():
        temporary = destination.with_suffix(".partial")
        urllib.request.urlretrieve(BASE + "instinfo_beta.txt", temporary)
        temporary.rename(destination)
    url = BASE + "level3/level3_beta_ctl_n188708x12328.gctx"
    with fsspec.open(
        url,
        "rb",
        block_size=1024 * 1024,
        cache_type="blockcache",
        cache_options={"maxblocks": 32},
    ) as remote:
        with h5py.File(remote, "r") as h:
            np.savez(
                source / "control_matrix_index.npz",
                sample_id=h["0/META/COL/id"][:].astype(str),
                gene_id=h["0/META/ROW/id"][:].astype(str),
            )
            matrix = h["0/DATA/0/matrix"]
            layout = {
                "url": url,
                "shape": matrix.shape,
                "dtype": matrix.dtype.str,
                "offset": matrix.id.get_offset(),
                "chunks": matrix.chunks,
            }
            (source / "control_matrix_layout.json").write_text(
                json.dumps(layout, indent=2)
            )
    print("Metadata and control matrix index saved.")


if __name__ == "__main__":
    main()
