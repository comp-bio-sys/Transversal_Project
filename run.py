"""Run the analysis, presentation builders or model training from one entry point."""

import argparse
from datetime import datetime
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "MLP_PRnet_controls"


def run(script):
    print(f"\nRunning {script}", flush=True)
    subprocess.run([sys.executable, str(ROOT / script)], cwd=ROOT, check=True)


def require_assets(checkpoints=False):
    required = [
        "MLP_PRnet_controls/l1000_level3_filtered_train_clean.feather",
        "MLP_PRnet_controls/data/sample_manifest.feather",
        "MLP_PRnet_controls/data/sample_usage.feather",
        "MLP_PRnet_controls/data/chemical_features.npz",
        "MLP_PRnet_controls/data/control_means.npy",
        "MLP_PRnet_controls/prepared/targets.npy",
        "MLP_PRnet_controls/prepared/indices.npz",
        "MLP_PRnet_controls/encoder/zinc_encoder.h5",
        "extension/split_indices.npz",
        "extension/scaling.npz",
        "extension/skbr3_treated.npy",
        "extension/skbr3_control.npy",
    ]
    if checkpoints:
        required += [
            "extension/balanced/weights/Residual_42.pt",
            "extension/balanced/group_holdout/split_indices.npz",
            "extension/balanced/mean_baselines.npz",
            "extension/balanced/ridge.npz",
        ]
        # Every saved checkpoint has a corresponding metadata file.
        for directory in ["weights", "tuning", "group_holdout/weights"]:
            required.extend(
                str(p.with_suffix(".pt").relative_to(ROOT))
                for p in (ROOT / "extension/balanced" / directory).rglob(
                    "*_[0-9][0-9].json"
                )
            )
    missing = sorted({name for name in required if not (ROOT / name).is_file()})
    if missing:
        names = "\n".join(f"  {name}" for name in missing[:8])
        more = f"\n  ... and {len(missing) - 8} more" if len(missing) > 8 else ""
        raise SystemExit(
            "This command needs the data and model files from the submission archive.\n"
            "See docs/DATA.md for their locations. Missing files:\n" + names + more
        )


def figures():
    run("presentation/figures.py")
    run("presentation/poster.py")


def report():
    run("presentation/build_report.py")
    run("presentation/render_report.py")
    run("presentation/export_markdown.py")


def presentation():
    figures()
    report()
    run("presentation/build_notebook.py")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser(
        "check", help="verify splits, model metadata and stored results"
    )
    commands.add_parser(
        "figures", help="rebuild figures and the poster from result tables"
    )
    commands.add_parser("report", help="rebuild the PDF, HTML and GitHub reports")
    commands.add_parser("notebook", help="rebuild and execute the analysis notebook")
    commands.add_parser(
        "evaluate",
        help="reevaluate saved models and rebuild all outputs; may take hours",
    )
    train = commands.add_parser(
        "train", help="fit the benchmark and rebuild all outputs; may take days"
    )
    train.add_argument(
        "--fresh",
        action="store_true",
        help="back up existing experiment outputs before fitting",
    )
    args = parser.parse_args()

    cache = ROOT / ".cache"
    cache.mkdir(exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(cache / "matplotlib"))
    os.environ.setdefault("IPYTHONDIR", str(cache / "ipython"))
    os.environ.setdefault("MPLBACKEND", "Agg")

    if args.command in {"check", "evaluate", "train"}:
        require_assets(checkpoints=args.command != "train")
    if args.command == "check":
        run("extension/verify_revision.py")
    elif args.command == "figures":
        figures()
    elif args.command == "report":
        report()
    elif args.command == "notebook":
        run("presentation/build_notebook.py")
    elif args.command == "evaluate":
        for script in [
            "evaluate_balanced",
            "diagnose_final",
            "calibration",
            "summarize_revision",
            "verify_revision",
        ]:
            run(f"extension/{script}.py")
        presentation()
    elif args.command == "train":
        existing = ROOT / "extension/balanced"
        if args.fresh and existing.exists():
            backup = (
                ROOT / "runs" / datetime.now().strftime("%Y%m%d_%H%M%S_%f") / "balanced"
            )
            backup.parent.mkdir(parents=True)
            shutil.move(str(existing), backup)
            print(
                f"Previous experiment saved in {backup.relative_to(ROOT)}", flush=True
            )
        elif existing.exists():
            print(
                "Resuming the saved experiment. Completed fits are reused. Use --fresh to refit all models.",
                flush=True,
            )
        run("extension/run_scientific_revision.py")
        run("extension/verify_revision.py")
        presentation()


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as error:
        raise SystemExit(
            f"Stopped because a stage failed (exit code {error.returncode}). See the message above."
        ) from error
