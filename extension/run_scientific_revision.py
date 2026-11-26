"""Run benchmark training and evaluation in dependency order."""

from pathlib import Path
import argparse, json, subprocess, sys, time
from benchmark import ROOT
from balanced_train import MAIN, ABLATIONS


def main(wait):
    if wait:
        needed = [
            ROOT / "balanced/weights" / f"{m}_{s}.json"
            for m in MAIN + ABLATIONS
            for s in [42, 43, 44]
        ]
        started = time.monotonic()
        while not all(p.exists() for p in needed):
            if time.monotonic() - started > 7200:
                raise TimeoutError(
                    "Initial balanced training did not finish within two hours."
                )
            time.sleep(10)
    stages = [] if wait else [("balanced_train.py", [])]
    stages += [
        ("fit_baselines.py", []),
        ("audit_generalization.py", []),
        ("tune_models.py", []),
        ("group_holdout.py", []),
        ("evaluate_balanced.py", []),
        ("diagnose_final.py", []),
        ("calibration.py", []),
        ("summarize_revision.py", []),
    ]
    for script, args in stages:
        print("Starting", script, flush=True)
        subprocess.run([sys.executable, str(ROOT / script), *args], check=True)
        (ROOT / "balanced/latest_completed_stage.json").write_text(
            json.dumps({"stage": script}) + "\n"
        )
    print("Scientific experiment stages complete.", flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--wait-for-initial", action="store_true")
    a = p.parse_args()
    main(a.wait_for_initial)
