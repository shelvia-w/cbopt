"""Full training (100 epochs, seeds 0-2) for uCBOpt / LeNet / MNIST across 6 curvature values.

Edit LR, WD, and HESS_INIT below to the best values found from hyperparameter tuning
before running this script.
"""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/scratch") / os.environ.get("USER", "USER") / "cbo_results"
OUTPUT_ROOT = Path(os.environ.get("CBO_OUTPUT_ROOT", SCRATCH_ROOT / "final"))
DATA_DIR = Path(os.environ.get("CBO_DATA_DIR", OUTPUT_ROOT / "data"))
DATASET = "mnist"
MODEL = "lenet"
OPTIMIZER = "ucbopt"
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
HESS_INIT = "0.5"
BETA1 = "0.9"
BETA2 = "0.99999"
RESCALE_LR = False
# ---------------------------------------

CURVATURE_SWEEP = ["0.0", "1e-8", "1e-7", "1e-6", "1e-5", "1e-4"]


def run_dir(cand_curvature: str, seed: str) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{LR}_wd_{WD}_h0_{HESS_INIT}_curv_{cand_curvature}_rslr_{RESCALE_LR}_ep_{EPOCHS}_s{seed}"
    )


def completed_val_csv(val_csv: Path) -> bool:
    if not val_csv.exists():
        return False
    try:
        with val_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return bool(rows) and max(int(float(row["epoch"])) for row in rows) >= int(EPOCHS) - 1
    except (KeyError, ValueError):
        return False


def train_command(cand_curvature: str, seed: str, save_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_ucbopt",
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
        "--cand_curvature",
        cand_curvature,
        "--beta1",
        BETA1,
        "--beta2",
        BETA2,
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


def run_training(cand_curvature: str, seed: str, dry_run: bool = False) -> Path:
    save_dir = run_dir(cand_curvature, seed)
    val_csv = save_dir / "val.csv"

    cmd = train_command(cand_curvature, seed, save_dir)
    if dry_run:
        print(" ".join(cmd))
        return save_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if completed_val_csv(val_csv):
        print(f"Skipping completed run: {save_dir}")
        return save_dir

    print(f"Running cand_curvature={cand_curvature}, seed={seed}")
    with (save_dir / "stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
            check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for cand_curvature={cand_curvature}, seed={seed}. "
            f"See {save_dir / 'stdout.log'}"
        )
    return save_dir


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    for cand_curvature in CURVATURE_SWEEP:
        print(f"\n=== cand_curvature={cand_curvature} ===")
        for seed in SEEDS:
            run_training(cand_curvature, seed, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nAll runs complete.")
        print(f"Results written under: {OUTPUT_ROOT / OPTIMIZER / f'{DATASET}_{MODEL}'}")


if __name__ == "__main__":
    main()
