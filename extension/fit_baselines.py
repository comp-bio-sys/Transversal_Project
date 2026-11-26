"""Training-only mean responses and a validation-selected linear ECFP4 model."""

import json
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, BASE


def main():
    torch.set_num_threads(2)
    out = ROOT / "balanced"
    out.mkdir(exist_ok=True)
    m = pd.read_feather(BASE / "data/sample_manifest.feather")
    ids = np.load(BASE / "prepared/indices.npz")
    splits = np.load(ROOT / "split_indices.npz")
    train = splits["train"]
    val = splits["val"]
    y = np.load(BASE / "prepared/targets.npy", mmap_mode="r")
    control = np.load(BASE / "data/control_means.npy")
    scaler = np.load(ROOT / "scaling.npz")
    center = scaler["center"]
    scale = scaler["scale"]
    fp = np.load(BASE / "data/chemical_features.npz")["ECFP4"]
    global_sum = np.zeros(978)
    cell_names = sorted(m.iloc[train].cell_iname.unique())
    cell_means = []
    for cell in cell_names:
        ix = train[m.iloc[train].cell_iname.eq(cell).to_numpy()]
        response = np.asarray(y[ix]) - control[ids["control_ids"][ix]]
        cell_means.append(response.mean(0, dtype="float64"))
        global_sum += response.sum(0, dtype="float64")
    np.savez(
        out / "mean_baselines.npz",
        cells=np.array(cell_names),
        cell_means=np.array(cell_means, dtype="float32"),
        global_mean=(global_sum / len(train)).astype("float32"),
    )
    width = 2002
    xx = torch.zeros(width, width)
    xy = torch.zeros(width, 978)
    sx = torch.zeros(width)
    sy = torch.zeros(978)

    def features(ix):
        c = control[ids["control_ids"][ix]]
        return (
            np.concatenate(
                [fp[ids["compound_ids"][ix]], (c - center) / scale], axis=1
            ).astype("float32"),
            c,
        )

    for start in range(0, len(train), 2048):
        ix = train[start : start + 2048]
        x, c = features(ix)
        x = torch.from_numpy(x)
        dy = torch.from_numpy(np.asarray(y[ix]) - c)
        xx += x.T @ x
        xy += x.T @ dy
        sx += x.sum(0)
        sy += dy.sum(0)
    mx = sx.double() / len(train)
    my = sy.double() / len(train)
    covariance = xx.double() / len(train) - mx[:, None] * mx[None, :]
    cross = xy.double() / len(train) - mx[:, None] * my[None, :]
    records = []
    best = float("inf")
    for alpha in [1e-4, 1e-3, 0.01, 0.1]:
        coef = torch.linalg.solve(
            covariance + alpha * torch.eye(width, dtype=torch.float64), cross
        )
        intercept = my - mx @ coef
        coef = coef.float()
        intercept = intercept.float()
        total = 0.0
        for start in range(0, len(val), 512):
            ix = val[start : start + 512]
            x, c = features(ix)
            pred = torch.from_numpy(x) @ coef + intercept + torch.from_numpy(c)
            total += float(
                (pred - torch.from_numpy(np.asarray(y[ix]))).square().double().sum()
            )
        mse = total / (len(val) * 978)
        records.append({"alpha": alpha, "validation_mse": mse})
        if mse < best:
            best = mse
            np.savez(
                out / "ridge.npz",
                coef=coef.numpy(),
                intercept=intercept.numpy(),
                alpha=alpha,
            )
        print("Ridge", alpha, mse, flush=True)
    pd.DataFrame(records).to_csv(out / "ridge_validation.csv", index=False)
    (out / "baseline_protocol.json").write_text(
        json.dumps(
            {
                "training_rows": len(train),
                "alpha_candidates": [1e-4, 1e-3, 0.01, 0.1],
                "selection": "validation MSE only",
                "ridge_objective": "mean squared residual error plus alpha times coefficient squared norm",
                "inputs": "ECFP4 + train-standardized control expression; unpenalized intercept",
                "cell_mean_unseen_cell_fallback": "global training mean response",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
