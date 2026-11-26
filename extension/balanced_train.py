"""Matched CPU reruns and a factorial residual/initialization ablation."""

from pathlib import Path
import argparse, json, time
import numpy as np
import pandas as pd
import torch
from benchmark import ROOT, BASE, build, predict

MAIN = ["ResidualFCFP4", "Residual", "ResidualChemicalVAE", "PRnet", "ControlOnly"]
ABLATIONS = ["Direct", "DirectZeroInit", "ResidualRandomInit"]
REP = {"ResidualFCFP4": "FCFP4", "ResidualChemicalVAE": "ChemicalVAE", "PRnet": "FCFP4"}


def main(variants, seeds, out, lr, wd, split_path=None):
    torch.set_num_threads(4)
    out.mkdir(exist_ok=True, parents=True)
    config = {
        "variants": variants,
        "seeds": seeds,
        "learning_rate": lr,
        "weight_decay": wd,
        "batch_size": 512,
        "max_epochs": 150,
        "patience": 7,
        "selection": "validation Gaussian NLL only",
        "device": "cpu",
        "threads": 4,
        "torch": torch.__version__,
        "numpy": np.__version__,
        "split": (
            str(split_path.relative_to(ROOT.parent))
            if split_path is not None
            else "extension/split_indices.npz"
        ),
    }
    (out / "config.json").write_text(json.dumps(config, indent=2) + "\n")
    split = np.load(split_path or ROOT / "split_indices.npz")
    ids = np.load(BASE / "prepared/indices.npz")
    control = torch.from_numpy(np.load(BASE / "data/control_means.npy"))
    targets = torch.from_numpy(np.load(BASE / "prepared/targets.npy"))
    cid = torch.from_numpy(ids["control_ids"])
    fid = torch.from_numpy(ids["compound_ids"])
    fp = {
        k: torch.from_numpy(v)
        for k, v in np.load(BASE / "data/chemical_features.npz").items()
    }
    scaling = np.load(
        split_path.parent / "scaling.npz"
        if split_path is not None
        else ROOT / "scaling.npz"
    )
    center, scale = [
        torch.from_numpy(scaling[k].astype("float32")) for k in ["center", "scale"]
    ]
    noise = torch.from_numpy(
        np.random.default_rng(1042)
        .standard_normal((len(split["val"]), 10))
        .astype("float32")
    )
    for seed in seeds:
        for variant in variants:
            path = out / f"{variant}_{seed}.pt"
            if path.exists() and path.with_suffix(".json").exists():
                continue
            torch.manual_seed(seed)
            model = build(variant)
            if variant == "PRnet":
                model.control_center.copy_(center)
                model.control_scale.copy_(scale)
            optim = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=wd)
            chem = fp[REP.get(variant, "ECFP4")]
            rng = np.random.default_rng(seed)
            history = []
            best = float("inf")
            best_epoch = 0
            started = time.monotonic()
            for epoch in range(1, 151):
                tick = time.monotonic()
                for name in ["train", "val"]:
                    training = name == "train"
                    model.train(training)
                    rows = rng.permutation(split[name]) if training else split[name]
                    total = 0.0
                    with torch.set_grad_enabled(training):
                        for start in range(0, len(rows), 512):
                            ix = torch.from_numpy(rows[start : start + 512])
                            c = control[cid[ix]]
                            z = None if training else noise[start : start + len(ix)]
                            mu, var = predict(
                                model, variant, chem[fid[ix]], c, center, scale, z
                            )
                            loss = torch.nn.functional.gaussian_nll_loss(
                                mu, targets[ix], var
                            )
                            if training:
                                optim.zero_grad()
                                loss.backward()
                                optim.step()
                            else:
                                total += float(loss) * len(ix)
                    if not training:
                        val = total / len(rows)
                assert np.isfinite(val)
                history.append(
                    {"epoch": epoch, "val_nll": val, "seconds": time.monotonic() - tick}
                )
                if val < best:
                    best = val
                    best_epoch = epoch
                    torch.save(model.state_dict(), path.with_suffix(".partial"))
                pd.DataFrame(history).to_csv(path.with_suffix(".csv"), index=False)
                print(
                    f"{variant} seed={seed} epoch={epoch} val={val:.6f} seconds={time.monotonic()-tick:.1f}",
                    flush=True,
                )
                if epoch - best_epoch >= 7:
                    break
            path.with_suffix(".partial").replace(path)
            metadata = {
                "best_epoch": best_epoch,
                "val_nll": best,
                "elapsed_seconds": time.monotonic() - started,
                "parameters": sum(p.numel() for p in model.parameters()),
                **config,
            }
            path.with_suffix(".json").write_text(json.dumps(metadata, indent=2) + "\n")
            print("Saved", path.name, flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--variants", nargs="+", default=MAIN + ABLATIONS)
    p.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44])
    p.add_argument("--out", type=Path, default=ROOT / "balanced/weights")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--wd", type=float, default=0.01)
    p.add_argument("--split", type=Path)
    a = p.parse_args()
    main(a.variants, a.seeds, a.out, a.lr, a.wd, a.split)
