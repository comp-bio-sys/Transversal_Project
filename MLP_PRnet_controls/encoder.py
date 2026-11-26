"""Frozen ChemicalVAE posterior means using the supplied ZINC weights."""

import json
from pathlib import Path
import h5py
import torch
from torch import nn

ROOT = Path(__file__).resolve().parent / "encoder"


@torch.no_grad()
def encode(structures):
    """Compute the frozen ZINC encoder's posterior mean for each canonical SMILES."""
    vocab = json.loads((ROOT / "zinc.json").read_text())
    idx = {char: i for i, char in enumerate(vocab)}
    eye = torch.eye(len(vocab))
    weights = {}
    with h5py.File(ROOT / "zinc_encoder.h5", "r") as f:
        for name, group in f["model_weights"].items():
            if name in group:
                weights[name] = {
                    k.split(":")[0]: torch.from_numpy(v[:])
                    for k, v in group[name].items()
                }

    def normalize(x, name):
        w = weights[name]
        scale = w["gamma"] * torch.rsqrt(w["moving_variance"] + 0.001)
        return (x - w["moving_mean"]) * scale + w["beta"]

    def dense(x, name):
        w = weights[name]
        return x @ w["kernel"] + w["bias"]

    batches = []
    for start in range(0, len(structures), 128):
        ids = [[idx[c] for c in s.ljust(120)] for s in structures[start : start + 128]]
        x = eye[torch.tensor(ids)]
        for i in range(3):
            w = weights[f"encoder_conv{i}"]
            x = nn.functional.conv1d(
                x.transpose(1, 2), w["kernel"].permute(2, 1, 0), w["bias"]
            )
            x = normalize(x.transpose(1, 2).tanh(), f"encoder_norm{i}")
        x = normalize(
            dense(x.flatten(1), "encoder_dense0").tanh(), "encoder_dense0_norm"
        )
        batches.append(dense(x, "z_mean_sample"))
    return torch.cat(batches)
