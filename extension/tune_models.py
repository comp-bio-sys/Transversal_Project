"""Validation-only tuning for four chemical models and the control-only comparator."""

from pathlib import Path
import json
import pandas as pd
from benchmark import ROOT
from balanced_train import main as fit, MAIN


def main():
    root = ROOT / "balanced"
    root.mkdir(exist_ok=True)
    settings = {
        "default": (0.001, 0.01),
        "lower_lr": (0.0003, 0.01),
        "lower_decay": (0.001, 0.0001),
    }
    (root / "tuning_protocol.json").write_text(
        json.dumps(
            {
                "settings": settings,
                "selection_seed": 42,
                "selection_metric": "validation Gaussian NLL",
                "evaluation_seeds": [42, 43, 44],
                "test_data_used_for_selection": False,
                "models": MAIN,
                "scope": "Within-grid setting and checkpoint selection use validation only. Earlier test inspection informed the exploratory revision design.",
                "test_sets_previously_examined": True,
                "epoch_cap": 150,
            },
            indent=2,
        )
        + "\n"
    )
    for name, (lr, wd) in settings.items():
        directory = root / "weights" if name == "default" else root / "tuning" / name
        fit(MAIN, [42], directory, lr, wd)
    rows = []
    selected = {}
    for model in MAIN:
        scores = []
        for name in settings:
            directory = (
                root / "weights" if name == "default" else root / "tuning" / name
            )
            meta = json.loads((directory / f"{model}_42.json").read_text())
            rows.append(
                {
                    "model": model,
                    "setting": name,
                    "val_nll": meta["val_nll"],
                    "best_epoch": meta["best_epoch"],
                }
            )
            scores.append((meta["val_nll"], name))
        best = min(scores)[1]
        selected[model] = best
    pd.DataFrame(rows).to_csv(root / "tuning_validation.csv", index=False)
    (root / "selected_settings.json").write_text(json.dumps(selected, indent=2) + "\n")
    for model, name in selected.items():
        directory = root / "weights" if name == "default" else root / "tuning" / name
        fit([model], [43, 44], directory, *settings[name])
    print("Validation-only selected settings:", selected, flush=True)


if __name__ == "__main__":
    main()
