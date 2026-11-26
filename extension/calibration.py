"""Check held-out likelihood and marginal predictive interval coverage."""

import json
import math

import numpy as np
import pandas as pd
import torch

from benchmark import ROOT, build, predict
from scientific_data import load_internal, load_skbr3, REP


def evaluate(model, variant, data):
    scaling = np.load(ROOT / "scaling.npz")
    center, scale = [
        torch.from_numpy(scaling[k].astype("float32")) for k in ["center", "scale"]
    ]
    draws = 32 if variant == "PRnet" else 1
    rngs = [np.random.default_rng(1042 + j) for j in range(draws)]
    counts = dict(
        n=0, nll=0.0, covered95=0.0, width95=0.0, squared_standardized_error=0.0
    )
    model.eval()
    fp = data["features"][REP.get(variant, "ECFP4")]
    with torch.no_grad():
        for start in range(0, len(data["y"]), 256):
            c = torch.from_numpy(data["control"][start : start + 256])
            f = torch.from_numpy(fp[start : start + 256])
            y = torch.from_numpy(data["y"][start : start + 256])
            means, variances, logp = [], [], []
            for rng in rngs:
                noise = torch.from_numpy(
                    rng.standard_normal((len(c), 10)).astype("float32")
                )
                mu, var = predict(model, variant, f, c, center, scale, noise)
                var = var.clamp_min(1e-6)
                means.append(mu)
                variances.append(var)
                logp.append(-0.5 * (torch.log(var) + (y - mu).square() / var))
            means, variances = torch.stack(means), torch.stack(variances)
            mean = means.mean(0)
            variance = (variances + means.square()).mean(0) - mean.square()
            sigma = variance.clamp_min(1e-6).sqrt()
            nll = -(torch.logsumexp(torch.stack(logp), dim=0) - math.log(draws))
            counts["n"] += y.numel()
            counts["nll"] += nll.double().sum().item()
            counts["covered95"] += ((y - mean).abs() <= 1.96 * sigma).sum().item()
            counts["width95"] += (3.92 * sigma).double().sum().item()
            counts["squared_standardized_error"] += (
                ((y - mean).square() / sigma.square()).double().sum().item()
            )
    n = counts.pop("n")
    return {k: v / n for k, v in counts.items()}


def main():
    torch.set_num_threads(4)
    root = ROOT / "balanced"
    selected = json.loads((root / "selected_settings.json").read_text())
    datasets = [load_internal(), load_skbr3()]
    rows = []
    for variant, setting in selected.items():
        folder = root / "weights" if setting == "default" else root / "tuning" / setting
        for seed in [42, 43, 44]:
            model = build(variant)
            model.load_state_dict(
                torch.load(
                    folder / f"{variant}_{seed}.pt",
                    map_location="cpu",
                    weights_only=True,
                )
            )
            for data in datasets:
                rows.append(
                    dict(
                        model=variant,
                        seed=seed,
                        dataset=data["name"],
                        **evaluate(model, variant, data),
                    )
                )
            print("Calibration:", variant, seed, flush=True)
    pd.DataFrame(rows).to_csv(root / "results/calibration.csv", index=False)
    (root / "results/calibration_protocol.json").write_text(
        json.dumps(
            {
                "NLL": "Gaussian marginal negative log likelihood, omitting 0.5*log(2*pi); for PRnet, log-average density across 32 decoder draws per gene.",
                "interval": "Mean +/- 1.96 marginal standard deviations. Exact Gaussian interval for MLP; moment-matched Gaussian approximation to PRnet mixture.",
                "limits": "Marginal profile-gene coverage, not joint 978-gene coverage or confidence in a biological effect. Descriptive on previously examined sets; no post-hoc calibration.",
            },
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
