"""Full training (100 epochs, seeds 0-2) for uCBOptAdaptCurv / LeNet / MNIST across gamma values.

Edit LR, WD, HESS_INIT, and BETA3 below to the best values found from
hyperparameter tuning before running this script.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/scratch") / os.environ.get("USER", "USER") / "cbo_results"
OUTPUT_ROOT = Path(os.environ.get("CBO_OUTPUT_ROOT", SCRATCH_ROOT / "final"))
DATA_DIR = Path(os.environ.get("CBO_DATA_DIR", OUTPUT_ROOT / "data"))
DATASET = "mnist"
MODEL = "lenet"
OPTIMIZER = "ucbopt_adaptcurv"
EPOCHS = "100"
BATCH = "128"
VAL_BATCH = "256"
TVSPLIT = "0.9"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")
SEEDS = ["0", "1", "2"]

# --- Fill in best values from tuning ---
LR = "5e-2"
WD = "1e-4"
HESS_INIT = "0.1"
BETA1 = "0.9"
BETA2 = "0.99999"
BETA3 = "0.99999"
EPS = "1e-8"
CLIP_RADIUS = "inf"
RESCALE_LR = True
# ---------------------------------------

GAMMA_SWEEP = ["0.0", "1e-4", "5e-4", "1e-3", "5e-3", "1e-2", "5e-2", "1e-1", "2e-1"]


def hyperparam_dir(gamma: str) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{LR}_wd_{WD}_h0_{HESS_INIT}_gamma_{gamma}_b3_{BETA3}_ep_{EPOCHS}"
    )


def run_dir(gamma: str, seed: str, timestamp: str) -> Path:
    return hyperparam_dir(gamma) / f"seed={seed}" / timestamp


def completed_val_csv(val_csv: Path) -> bool:
    if not val_csv.exists():
        return False
    try:
        with val_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return bool(rows) and max(int(float(row["epoch"])) for row in rows) >= int(EPOCHS) - 1
    except (KeyError, ValueError):
        return False


def train_command(gamma: str, seed: str, save_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_ucbopt_adaptcurv",
        MODEL,
        DATASET,
        "-s",
        seed,
        "-d",
        DEVICE,
        "-dd",
        str(DATA_DIR),
        "-sd",
        str(save_dir),
        "-lr",
        LR,
        "--wd",
        WD,
        "--hess_init",
        HESS_INIT,
        "--gamma",
        gamma,
        "--beta1",
        BETA1,
        "--beta2",
        BETA2,
        "--beta3",
        BETA3,
        "--eps",
        EPS,
        "--clip-radius",
        CLIP_RADIUS,
        *(["--rescale_lr"] if RESCALE_LR else ["--no-rescale_lr"]),
        "-e",
        EPOCHS,
        "-tb",
        BATCH,
        "-vb",
        VAL_BATCH,
        "-sp",
        TVSPLIT,
        "-j",
        WORKERS,
    ]


def run_gamma(gamma: str, dry_run: bool = False) -> None:
    """Launch all seeds for one gamma value in parallel; wait for all to finish."""
    if not dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    procs: list[tuple[str, subprocess.Popen[str], object]] = []
    for seed in SEEDS:
        save_dir = run_dir(gamma, seed, timestamp)
        val_csv = save_dir / "val.csv"
        cmd = train_command(gamma, seed, save_dir)

        if dry_run:
            print(" ".join(cmd))
            continue

        if completed_val_csv(val_csv):
            print(f"Skipping completed run: {save_dir}")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Launching gamma={gamma}, seed={seed} -> {save_dir}")
        log = (save_dir / "stdout.log").open("w", encoding="utf-8")
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
        procs.append((seed, proc, log))

    failed = []
    for seed, proc, log in procs:
        ret = proc.wait()
        log.close()
        status = "OK" if ret == 0 else f"FAILED (exit={ret})"
        print(f"  seed={seed}: {status}", flush=True)
        if ret != 0:
            failed.append(seed)

    if failed:
        raise RuntimeError(
            f"Training failed for gamma={gamma}, seed(s)={failed}. "
            f"See stdout.log in each seed subfolder under {hyperparam_dir(gamma)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    for gamma in GAMMA_SWEEP:
        print(f"\n=== gamma={gamma} ===")
        run_gamma(gamma, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nAll runs complete.")
        print(f"Results written under: {OUTPUT_ROOT / OPTIMIZER / f'{DATASET}_{MODEL}'}")


if __name__ == "__main__":
    main()
