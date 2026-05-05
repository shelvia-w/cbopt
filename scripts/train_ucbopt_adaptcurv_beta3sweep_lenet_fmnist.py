"""Full training (100 epochs, seeds 0-2) for uCBOptAdaptCurv / LeNet / Fashion-MNIST across beta3 values.

Edit LR, WD, HESS_INIT, and GAMMA below to the best values found from
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
DATASET = "fmnist"
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
LR = "1e-2"
WD = "2e-3"
HESS_INIT = "0.05"
GAMMA = "5e-1"
BETA1 = "0.9"
BETA2 = "0.99999"
EPS = "1e-8"
CLIP_RADIUS = "inf"
RESCALE_LR = False
# ---------------------------------------

BETA3_SWEEP = [
    "1.00001",
    "1.0001",
    "1.01",
    "1.1",
]


def hyperparam_dir(beta3: str) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{LR}_wd_{WD}_h0_{HESS_INIT}_gamma_{GAMMA}_b3_{beta3}_ep_{EPOCHS}"
    )


def run_dir(beta3: str, seed: str, timestamp: str) -> Path:
    return hyperparam_dir(beta3) / f"seed={seed}" / timestamp


def completed_val_csv(val_csv: Path) -> bool:
    if not val_csv.exists():
        return False
    try:
        with val_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return bool(rows) and max(int(float(row["epoch"])) for row in rows) >= int(EPOCHS) - 1
    except (KeyError, ValueError):
        return False


def train_command(beta3: str, seed: str, save_dir: Path) -> list[str]:
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
        GAMMA,
        "--beta1",
        BETA1,
        "--beta2",
        BETA2,
        "--beta3",
        beta3,
        "--eps",
        EPS,
        "--clip-radius",
        CLIP_RADIUS,
        *(["--rescale_lr"] if RESCALE_LR else []),
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


def run_beta3(beta3: str, dry_run: bool = False) -> None:
    """Launch all seeds for one beta3 value in parallel; wait for all to finish."""
    if not dry_run:
        DATA_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")

    procs: list[tuple[str, subprocess.Popen[str], object]] = []
    for seed in SEEDS:
        save_dir = run_dir(beta3, seed, timestamp)
        val_csv = save_dir / "val.csv"
        cmd = train_command(beta3, seed, save_dir)

        if dry_run:
            print(" ".join(cmd))
            continue

        if completed_val_csv(val_csv):
            print(f"Skipping completed run: {save_dir}")
            continue

        save_dir.mkdir(parents=True, exist_ok=True)
        print(f"Launching beta3={beta3}, seed={seed} -> {save_dir}")
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
            f"Training failed for beta3={beta3}, seed(s)={failed}. "
            f"See stdout.log in each seed subfolder under {hyperparam_dir(beta3)}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    for beta3 in BETA3_SWEEP:
        print(f"\n=== beta3={beta3} ===")
        run_beta3(beta3, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nAll runs complete.")
        print(f"Results written under: {OUTPUT_ROOT / OPTIMIZER / f'{DATASET}_{MODEL}'}")


if __name__ == "__main__":
    main()
