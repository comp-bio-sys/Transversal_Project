"""Extract matched landmark-gene control profiles using HTTP byte ranges."""

import concurrent.futures
import hashlib
import json
import time
import urllib.request
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent


def main():
    data = ROOT / "data"
    source = data / "source"
    layout = json.loads((source / "control_matrix_layout.json").read_text())
    assert layout["chunks"] is None
    index = np.load(source / "control_matrix_index.npz")
    genes = json.loads((data / "feature_spec.json").read_text())["gene_order"]
    gene_rows = pd.Index(index["gene_id"]).get_indexer(genes)
    assert (gene_rows >= 0).all() and len(genes) == 978
    manifest = pd.read_feather(data / "control_manifest.feather")
    rows = manifest.matrix_row.to_numpy()
    assert np.array_equal(index["sample_id"][rows], manifest.sample_id)
    plan = {"layout": layout, "rows": rows.tolist(), "genes": genes}
    plan_path = data / "download_plan.json"
    if plan_path.exists():
        assert (
            json.loads(plan_path.read_text()) == plan
        ), "Download selection changed; use a fresh data directory."
    elif (data / "download_completed.json").exists():
        raise ValueError(
            "Download progress has no matching plan; use a fresh download directory."
        )
    else:
        plan_path.write_text(json.dumps(plan))
    path = data / "control_expression.npy"
    shape = (len(rows), len(genes))
    output = np.lib.format.open_memmap(
        path, mode="r+" if path.exists() else "w+", dtype="float32", shape=shape
    )
    assert output.shape == shape
    blocks = []
    start = 0
    for i in range(1, len(rows) + 1):
        if i == len(rows) or rows[i] - rows[i - 1] > 4 or rows[i] - rows[start] >= 128:
            blocks.append((start, i))
            start = i
    completed_path = data / "download_completed.json"
    completed = (
        set(json.loads(completed_path.read_text()))
        if completed_path.exists()
        else set()
    )
    dtype = np.dtype(layout["dtype"])
    width = layout["shape"][1]
    row_bytes = width * dtype.itemsize

    def fetch(block):
        a, b = block
        lo = layout["offset"] + int(rows[a]) * row_bytes
        hi = layout["offset"] + (int(rows[b - 1]) + 1) * row_bytes - 1
        request = urllib.request.Request(
            layout["url"], headers={"Range": f"bytes={lo}-{hi}"}
        )
        for attempt in range(5):
            try:
                with urllib.request.urlopen(request, timeout=90) as response:
                    assert response.status == 206
                    assert response.headers["Content-Range"].startswith(
                        f"bytes {lo}-{hi}/"
                    )
                    raw = response.read()
                assert len(raw) == hi - lo + 1
                values = np.frombuffer(raw, dtype=dtype).reshape(-1, width)
                values = values[rows[a:b] - rows[a]][:, gene_rows].astype("float32")
                assert np.isfinite(values).all()
                return a, b, values
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2**attempt)

    print(f"{len(rows)} controls; {len(blocks)} download ranges", flush=True)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        futures = [
            pool.submit(fetch, block) for block in blocks if block[0] not in completed
        ]
        for future in concurrent.futures.as_completed(futures):
            a, b, values = future.result()
            output[a:b] = values
            output.flush()
            completed.add(a)
            completed_path.write_text(json.dumps(sorted(completed)))
            if len(completed) % 25 == 0 or len(completed) == len(blocks):
                print(f"Completed {len(completed)}/{len(blocks)} ranges", flush=True)
    assert len(completed) == len(blocks)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    (data / "control_download_audit.json").write_text(
        json.dumps(
            {
                "url": layout["url"],
                "samples": len(rows),
                "genes": genes,
                "sha256": digest.hexdigest(),
                "min": float(output.min()),
                "max": float(output.max()),
                "all_ranges_verified": True,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
