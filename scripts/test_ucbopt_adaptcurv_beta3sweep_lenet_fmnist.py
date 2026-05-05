"""Evaluate saved checkpoints for uCBOptAdaptCurv / LeNet / Fashion-MNIST across beta3 values.

Mirrors the hyperparameter layout of train_ucbopt_adaptcurv_beta3sweep_lenet_fmnist.py.
Edit LR, WD, HESS_INIT, and GAMMA to match the values used during training.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRATCH_ROOT = Path("/scratch") / os.environ.get("USER", "USER") / "cbo_results"
OUTPUT_ROOT = Path(os.environ.get("CBO_OUTPUT_ROOT", SCRATCH_ROOT / "final"))
DATA_DIR = Path(os.environ.get("CBO_DATA_DIR", OUTPUT_ROOT / "data"))
DATASET = "fmnist"
MODEL = "lenet"
OPTIMIZER = "ucbopt_adaptcurv"
EPOCHS = "100"
BATCH = "256"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")
SEED_START = 0
SEED_END = 2

# --- Must match values used in training ---
LR = "1e-2"
WD = "2e-3"
HESS_INIT = "0.05"
GAMMA = "5e-1"
# -----------------------------------------

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


def eval_command(beta3: str, save_dir: Path) -> list[str]:
    train_dir = hyperparam_dir(beta3)
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.test",
        str(train_dir),
        DATASET,
        "-dd",
        str(DATA_DIR),
        "-sd",
        str(save_dir),
        "-d",
        DEVICE,
        "-ss",
        str(SEED_START),
        "-se",
        str(SEED_END),
        "-b",
        BATCH,
        "-j",
        WORKERS,
        "--tvsplit",
        "1.0",
        "--checkpoint",
        "latest",
    ]


def eval_beta3(beta3: str, dry_run: bool = False) -> None:
    train_dir = hyperparam_dir(beta3)
    if not train_dir.exists():
        print(f"Skipping missing traindir: {train_dir}")
        return

    save_dir = train_dir / "eval_final"
    cmd = eval_command(beta3, save_dir)

    if dry_run:
        print(" ".join(cmd))
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating beta3={beta3} -> {save_dir}")
    log = (save_dir / "stdout.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    ret = proc.wait()
    log.close()

    status = "OK" if ret == 0 else f"FAILED (exit={ret})"
    print(f"  beta3={beta3}: {status}", flush=True)
    if ret != 0:
        raise RuntimeError(
            f"Evaluation failed for beta3={beta3}. "
            f"See {save_dir / 'stdout.log'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running evaluation."
    )
    args = parser.parse_args()

    for beta3 in BETA3_SWEEP:
        print(f"\n=== beta3={beta3} ===")
        eval_beta3(beta3, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nAll evaluations complete.")
        print(f"Results written under: {OUTPUT_ROOT / OPTIMIZER / f'{DATASET}_{MODEL}'}")


if __name__ == "__main__":
    main()
