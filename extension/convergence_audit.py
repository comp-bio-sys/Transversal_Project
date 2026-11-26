"""Extend any trial truncated before patience stopping, then reselect on validation."""

import json
import shutil
import subprocess
import sys
import time

import numpy as np
import pandas as pd

from benchmark import ROOT
from balanced_train import main as fit, MAIN


def main():
    root = ROOT / "balanced"
    gate = root / ".convergence_pending"
    gate.write_text("Validation-only convergence audit in progress.\n")
    initial = json.loads((root / "selected_settings.json").read_text())
    expected = []
    for model, setting in initial.items():
        folder = root / "weights" if setting == "default" else root / "tuning" / setting
        expected.extend(folder / f"{model}_{seed}.json" for seed in [42, 43, 44])
    started = time.monotonic()
    while not all(p.exists() for p in expected):
        if time.monotonic() - started > 7200:
            raise TimeoutError("Selected model runs did not finish.")
        time.sleep(5)
    candidates = (
        list((root / "weights").glob("*_*.json"))
        + list((root / "tuning").glob("*/*_*.json"))
        + list((root / "group_holdout/weights").glob("*_*.json"))
    )
    extended = []
    for path in candidates:
        metadata = json.loads(path.read_text())
        history = pd.read_csv(path.with_suffix(".csv"))
        last = int(history.epoch.max())
        if last != 50 or last - metadata["best_epoch"] >= 7:
            continue
        backup = root / "epoch_cap_50" / path.relative_to(root)
        backup.parent.mkdir(parents=True, exist_ok=True)
        for suffix in [".json", ".csv", ".pt"]:
            shutil.move(path.with_suffix(suffix), backup.with_suffix(suffix))
        variant, seed = path.stem.rsplit("_", 1)
        split = (
            root / "group_holdout/split_indices.npz"
            if "group_holdout" in path.parts
            else None
        )
        print("Extending validation-limited trial:", path.relative_to(root), flush=True)
        fit(
            [variant],
            [int(seed)],
            path.parent,
            metadata["learning_rate"],
            metadata["weight_decay"],
            split,
        )
        new_history = pd.read_csv(path.with_suffix(".csv"))
        difference = float(
            np.max(
                np.abs(
                    new_history.val_nll.iloc[: len(history)].to_numpy()
                    - history.val_nll.to_numpy()
                )
            )
        )
        extended.append(
            {
                "checkpoint": str(path.relative_to(root)),
                "old_epochs": last,
                "new_epochs": len(new_history),
                "first_50_epoch_max_nll_difference": difference,
            }
        )
    subprocess.run([sys.executable, str(ROOT / "tune_models.py")], check=True)
    report = {
        "initial_epoch_cap": 50,
        "extended_epoch_cap": 150,
        "patience": 7,
        "rule": "Every completed trial reaching 50 epochs without patience stopping is refitted from its original seed with a cap of 150. Trials that already stopped are unchanged.",
        "selection": "The same three settings are re-ranked by seed-42 validation NLL. Any newly selected setting is then fitted for seeds 43 and 44.",
        "test_results_used": False,
        "extended_trials": extended,
    }
    (root / "convergence_audit.json").write_text(json.dumps(report, indent=2) + "\n")
    gate.unlink()
    print("Convergence audit complete; evaluation may proceed.", flush=True)


if __name__ == "__main__":
    main()
