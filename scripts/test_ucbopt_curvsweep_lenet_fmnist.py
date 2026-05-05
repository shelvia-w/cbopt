"""Evaluate saved checkpoints for uCBOpt / LeNet / Fashion MNIST across 6 curvature values.

Mirrors the hyperparameter layout of train_ucbopt_curvsweep_lenet_fmnist.py.
Edit LR, WD, and HESS_INIT to match the values used during training.
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
OPTIMIZER = "ucbopt"
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
RESCALE_LR = False
# -----------------------------------------

CURVATURE_SWEEP = [
    "0.0",    
    "1e-7",
    "2e-7",
    "5e-7",
    "8e-7",
    "1e-6",
    "2e-6",
    "5e-6",
    "8e-6",
    "1e-5",
]


def hyperparam_dir(cand_curvature: str) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{LR}_wd_{WD}_h0_{HESS_INIT}_curv_{cand_curvature}_ep_{EPOCHS}"
    )


def eval_command(cand_curvature: str, save_dir: Path) -> list[str]:
    train_dir = hyperparam_dir(cand_curvature)
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
        "best",
    ]


def eval_curvature(cand_curvature: str, dry_run: bool = False) -> None:
    train_dir = hyperparam_dir(cand_curvature)
    if not train_dir.exists():
        print(f"Skipping missing traindir: {train_dir}")
        return

    save_dir = train_dir / "eval"
    cmd = eval_command(cand_curvature, save_dir)

    if dry_run:
        print(" ".join(cmd))
        return

    save_dir.mkdir(parents=True, exist_ok=True)
    print(f"Evaluating cand_curvature={cand_curvature} -> {save_dir}")
    log = (save_dir / "stdout.log").open("w", encoding="utf-8")
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True)
    ret = proc.wait()
    log.close()

    status = "OK" if ret == 0 else f"FAILED (exit={ret})"
    print(f"  cand_curvature={cand_curvature}: {status}", flush=True)
    if ret != 0:
        raise RuntimeError(
            f"Evaluation failed for cand_curvature={cand_curvature}. "
            f"See {save_dir / 'stdout.log'}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running evaluation."
    )
    args = parser.parse_args()

    for cand_curvature in CURVATURE_SWEEP:
        print(f"\n=== cand_curvature={cand_curvature} ===")
        eval_curvature(cand_curvature, dry_run=args.dry_run)

    if not args.dry_run:
        print("\nAll evaluations complete.")
        print(f"Results written under: {OUTPUT_ROOT / OPTIMIZER / f'{DATASET}_{MODEL}'}")


if __name__ == "__main__":
    main()
