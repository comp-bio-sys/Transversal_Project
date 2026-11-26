"""PRnet source-recipe sensitivity analysis on the screened LINCS split."""

import argparse
from pathlib import Path
import json
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, BASE, build
from evaluate_project import GENES, fingerprints


def run(base, device):
    torch.set_num_threads(4)
    torch.manual_seed(42)
    rng = np.random.default_rng(42)
    data = base / "data"
    m = pd.read_feather(data / "sample_manifest.feather")
    individual = pd.read_feather(data / "control_manifest.feather")
    ctrl_lookup = individual.groupby("control_group").indices
    # Select a measured vehicle replicate once per treatment, without using targets.
    chosen = np.empty(len(m), dtype=int)
    for group, ix in m.groupby("control_group").indices.items():
        chosen[ix] = rng.choice(ctrl_lookup[group], size=len(ix))
    rows = np.load(ROOT / "split_indices.npz")
    targets = np.load(base / "prepared/targets.npy")
    raw_controls = np.load(data / "control_expression.npy")
    median = float(
        np.median(
            np.r_[
                targets[rows["train"]].sum(1),
                raw_controls[np.unique(chosen[rows["train"]])].sum(1),
            ]
        )
    )

    def transform(x):
        return np.log1p(x * (median / x.sum(axis=1, keepdims=True))).astype("float32")

    def tensor(x):
        return torch.as_tensor(x, device=device)

    y = tensor(transform(targets))
    controls = tensor(transform(raw_controls))
    paired = tensor(chosen)
    fps = tensor(np.load(data / "chemical_features.npz")["FCFP4"])
    compound = tensor(np.load(base / "prepared/indices.npz")["compound_ids"])
    model = build("PRnet").to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-8)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, factor=0.5, patience=10, min_lr=1e-8
    )
    out = ROOT / "reference"
    out.mkdir(exist_ok=True)
    best = float("inf")
    history = []
    for epoch in range(1, 501):
        for split in ["train", "val"]:
            train = split == "train"
            model.train(train)
            seq = rng.permutation(rows[split]) if train else rows[split]
            losses = []
            with torch.set_grad_enabled(train):
                for start in range(0, len(seq), 512):
                    ix = tensor(seq[start : start + 512])
                    mu, var = model(fps[compound[ix]], None, controls[paired[ix]])
                    if train:
                        loss = torch.nn.functional.gaussian_nll_loss(mu, y[ix], var)
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                    else:
                        sample = torch.normal(
                            mu.clamp(0.001, 1000), var.sqrt().clamp(0.001, 1000)
                        )
                        losses.append((sample - y[ix]).square().mean().detach())
        score = float(torch.stack(losses).mean().cpu())
        scheduler.step(score)
        assert np.isfinite(score)
        history.append({"epoch": epoch, "validation_sampled_mse": score})
        if score < best:
            best = score
            best_epoch = epoch
            torch.save(
                {k: v.detach().cpu() for k, v in model.state_dict().items()},
                out / "PRnet.pt",
            )
        if epoch - best_epoch >= 22:
            break
    pd.DataFrame(history).to_csv(out / "history.csv", index=False)
    model.load_state_dict(
        torch.load(out / "PRnet.pt", map_location=device, weights_only=True)
    )
    model.eval()
    labels = pd.read_csv(ROOT / "external_labels.csv")
    group = pd.DataFrame(
        {
            "cell": m.cell_iname,
            "id": np.load(base / "prepared/indices.npz")["control_ids"],
        }
    ).drop_duplicates()
    means = np.load(data / "control_means.npy")
    refs = {c: means[g.id].mean(0) for c, g in group.groupby("cell")}
    ct = transform(np.stack([refs[c] for c in labels.cell]))
    fp = tensor(fingerprints(labels.structure, True))
    torch.manual_seed(1042)
    with torch.no_grad():
        mu, _ = model(fp, None, tensor(ct))
    genes = json.loads((data / "feature_spec.json").read_text())["gene_order"]
    cols = [genes.index(GENES[g]) for g in labels.genes]
    delta = (mu.cpu().numpy() - ct)[np.arange(len(labels)), cols]
    labels.assign(
        predicted_change=delta, correct=np.sign(delta) == np.sign(labels.logFC)
    ).to_csv(out / "pharmgeo_predictions.csv", index=False)
    (out / "summary.json").write_text(
        json.dumps(
            dict(
                epochs=epoch,
                best_epoch=best_epoch,
                best_val_sampled_mse=best,
                training_median_total=median,
                external_correct=int((np.sign(delta) == np.sign(labels.logFC)).sum()),
                external_pairs=len(labels),
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--base", type=Path, default=BASE)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    run(a.base, a.device)
