"""HPC/local three-stage uCBOpt tuning sweep for LeNet on Fashion-MNIST.

Stages:
  1. Learning-rate sweep  (fixed wd=1e-4, cand_curvature=0.0)
  2. Weight-decay sweep   (fixed best lr, cand_curvature=0.0)
  3. Curvature sweep      (fixed best lr and best wd)
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
OUTPUT_ROOT = Path(os.environ.get("CBO_OUTPUT_ROOT", SCRATCH_ROOT / "tuning"))
DATA_DIR = Path(os.environ.get("CBO_DATA_DIR", OUTPUT_ROOT / "data"))
DATASET = "fmnist"
MODEL = "lenet"
OPTIMIZER = "ucbopt"
SEED = "0"
EPOCHS = "30"
BATCH = "128"
VAL_BATCH = "256"
TVSPLIT = "0.9"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")

HESS_INIT = "1.0"
BETA1 = "0.9"
BETA2 = "0.99999"

LR_SWEEP = ["1e-1", "5e-2", "1e-2", "5e-3", "1e-3"]
LR_SWEEP_WD = "1e-4"
LR_SWEEP_CURVATURE = "0.0"

WD_SWEEP = ["1e-5", "1e-4", "1e-3"]
WD_SWEEP_CURVATURE = "0.0"

CURVATURE_SWEEP = [
    "0.0",
    "1e-7",
    "5e-7",
    "1e-6",
    "5e-6",
    "1e-5",
    "5e-5",
    "1e-4",
]

CURVATURE_SWEEP_REFINED = [
    "1e-8",
    "2e-8",
    "5e-8",
    "2e-7",
    "3e-7",
]


def run_dir(lr: str, weight_decay: str, cand_curvature: str, epochs: str = EPOCHS) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{lr}_wd_{weight_decay}_hi_{HESS_INIT}_curv_{cand_curvature}_ep_{epochs}"
    )


def completed_val_csv(val_csv: Path, epochs: str = EPOCHS) -> bool:
    if not val_csv.exists():
        return False
    try:
        with val_csv.open(newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        return bool(rows) and max(int(float(row["epoch"])) for row in rows) >= int(epochs) - 1
    except (KeyError, ValueError):
        return False


def train_command(lr: str, weight_decay: str, cand_curvature: str, save_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_ucbopt",
        MODEL,
        DATASET,
        "-s",
        SEED,
        "-d",
        DEVICE,
        "-dd",
        str(DATA_DIR),
        "-sd",
        str(save_dir),
        "-lr",
        lr,
        "--wd",
        weight_decay,
        "--hess_init",
        HESS_INIT,
        "--cand_curvature",
        cand_curvature,
        "--beta1",
        BETA1,
        "--beta2",
        BETA2,
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


def run_training(lr: str, weight_decay: str, cand_curvature: str, dry_run: bool = False) -> Path:
    save_dir = run_dir(lr, weight_decay, cand_curvature)
    val_csv = save_dir / "val.csv"

    cmd = train_command(lr, weight_decay, cand_curvature, save_dir)
    if dry_run:
        print(" ".join(cmd))
        return save_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if completed_val_csv(val_csv):
        print(f"Skipping completed run: {save_dir}")
        return save_dir

    print(f"Running lr={lr}, weight-decay={weight_decay}, cand_curvature={cand_curvature}")
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
            f"Training failed for lr={lr}, weight-decay={weight_decay}, "
            f"cand_curvature={cand_curvature}. See {save_dir / 'stdout.log'}"
        )
    return save_dir


def best_val_metrics(
    save_dir: Path, lr: str, weight_decay: str, cand_curvature: str
) -> dict[str, str | float | int]:
    val_csv = save_dir / "val.csv"
    if not val_csv.exists():
        raise FileNotFoundError(f"Missing validation metrics: {val_csv}")

    with val_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No validation rows found in {val_csv}")

    best = min(rows, key=lambda row: float(row["nll"]))
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "hess_init": HESS_INIT,
        "cand_curvature": cand_curvature,
        "beta1": BETA1,
        "beta2": BETA2,
        "best_val_nll": float(best["nll"]),
        "best_val_accuracy": float(best["acc"]),
        "best_val_ece": float(best["ece"]),
        "epoch": int(float(best["epoch"])),
        "save_dir": str(save_dir),
    }


def markdown_table(rows: list[dict[str, str | float | int]]) -> str:
    header = (
        "| lr | weight-decay | cand_curvature | best val NLL | best val accuracy | best val ECE | epoch | stage |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            "| {lr} | {weight_decay} | {cand_curvature} | {best_val_nll:.6f} | "
            "{best_val_accuracy:.6f} | {best_val_ece:.6f} | {epoch} | {stage} |".format(**row)
        )
    return "\n".join(lines)


def write_summary(
    rows: list[dict[str, str | float | int]], best: dict[str, str | float | int]
) -> None:
    summary_dir = OUTPUT_ROOT / OPTIMIZER / f"{DATASET}_{MODEL}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "tuning_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr",
                "weight_decay",
                "hess_init",
                "cand_curvature",
                "beta1",
                "beta2",
                "best_val_nll",
                "best_val_accuracy",
                "best_val_ece",
                "epoch",
                "save_dir",
                "stage",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    recommendation = (
        "optimizer: ucbopt\n"
        f"dataset: {DATASET}\n"
        f"model: {MODEL}\n"
        "seeds: [0, 1, 2]\n"
        f"device: {DEVICE}\n"
        f"traindir: \"runs/final/ucbopt/{DATASET}_{MODEL}/"
        f"lr_{best['lr']}_wd_{best['weight_decay']}_hi_1.0_curv_{best['cand_curvature']}_ep_100\"\n"
        "train_args:\n"
        f"  lr: \"{best['lr']}\"\n"
        "  e: \"100\"\n"
        f"  weight-decay: \"{best['weight_decay']}\"\n"
        "  hess_init: \"1.0\"\n"
        f"  cand_curvature: \"{best['cand_curvature']}\"\n"
        "  beta1: \"0.9\"\n"
        "  beta2: \"0.99999\"\n"
        "  tbatch: \"128\"\n"
        "  vbatch: \"256\"\n"
        "  tvsplit: \"0.9\"\n"
        f"  j: \"{WORKERS}\"\n"
        "train_flags: []\n"
    )
    (summary_dir / "recommended_final_config.yaml").write_text(recommendation, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    rows: list[dict[str, str | float | int]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_row(row: dict[str, str | float | int], stage: str) -> None:
        key = (str(row["lr"]), str(row["weight_decay"]), str(row["cand_curvature"]))
        if key not in seen:
            seen.add(key)
            rows.append({**row, "stage": stage})

    print("Stage 1: learning-rate sweep")
    for lr in LR_SWEEP:
        save_dir = run_training(lr, LR_SWEEP_WD, LR_SWEEP_CURVATURE, dry_run=args.dry_run)
        if not args.dry_run:
            add_row(best_val_metrics(save_dir, lr, LR_SWEEP_WD, LR_SWEEP_CURVATURE), "lr_sweep")

    if args.dry_run:
        best_lr = "<best_lr_from_stage_1>"
    else:
        lr_rows = [r for r in rows if r["stage"] == "lr_sweep"]
        best_lr = str(min(lr_rows, key=lambda r: float(r["best_val_nll"]))["lr"])
        print(f"Best learning rate from stage 1: {best_lr}")

    print("Stage 2: weight-decay sweep")
    for weight_decay in WD_SWEEP:
        save_dir = run_training(best_lr, weight_decay, WD_SWEEP_CURVATURE, dry_run=args.dry_run)
        if not args.dry_run:
            add_row(
                best_val_metrics(save_dir, best_lr, weight_decay, WD_SWEEP_CURVATURE), "wd_sweep"
            )

    if args.dry_run:
        best_wd = "<best_wd_from_stage_2>"
    else:
        wd_rows = [r for r in rows if r["stage"] in ("lr_sweep", "wd_sweep")]
        best_row = min(wd_rows, key=lambda r: float(r["best_val_nll"]))
        best_lr = str(best_row["lr"])
        best_wd = str(best_row["weight_decay"])
        print(f"Best config from stage 2: lr={best_lr}, weight-decay={best_wd}")

    print("Stage 3: candidate-curvature sweep")
    for cand_curvature in CURVATURE_SWEEP:
        if not args.dry_run and float(cand_curvature) >= float(best_wd):
            print(
                f"Skipping cand_curvature={cand_curvature} because it is >= weight_decay={best_wd}"
            )
            continue
        save_dir = run_training(best_lr, best_wd, cand_curvature, dry_run=args.dry_run)
        if not args.dry_run:
            add_row(
                best_val_metrics(save_dir, best_lr, best_wd, cand_curvature), "curvature_sweep"
            )

    print("Stage 4: refined candidate-curvature sweep")
    summary_csv = OUTPUT_ROOT / OPTIMIZER / f"{DATASET}_{MODEL}" / "tuning_summary.csv"
    if args.dry_run:
        best_lr_s4 = "<best_lr_from_summary>"
        best_wd_s4 = "<best_wd_from_summary>"
    elif summary_csv.exists():
        with summary_csv.open(newline="", encoding="utf-8") as f:
            prior_rows = list(csv.DictReader(f))
        if not prior_rows:
            raise RuntimeError(f"tuning_summary.csv is empty: {summary_csv}")
        best_prior = min(prior_rows, key=lambda r: float(r["best_val_nll"]))
        best_lr_s4 = str(best_prior["lr"])
        best_wd_s4 = str(best_prior["weight_decay"])
        print(f"Stage 4 best config from summary CSV: lr={best_lr_s4}, weight-decay={best_wd_s4}")
    elif rows:
        stage13_rows = [r for r in rows if r["stage"] in ("lr_sweep", "wd_sweep", "curvature_sweep")]
        best_row = min(stage13_rows, key=lambda r: float(r["best_val_nll"]))
        best_lr_s4 = str(best_row["lr"])
        best_wd_s4 = str(best_row["weight_decay"])
        print(f"Stage 4 best config from in-session rows: lr={best_lr_s4}, weight-decay={best_wd_s4}")
    else:
        raise RuntimeError(
            f"No tuning_summary.csv found at {summary_csv} and no in-session rows. "
            "Run stages 1-3 first."
        )

    for cand_curvature in CURVATURE_SWEEP_REFINED:
        if not args.dry_run and float(cand_curvature) >= float(best_wd_s4):
            print(
                f"Skipping cand_curvature={cand_curvature} because it is >= weight_decay={best_wd_s4}"
            )
            continue
        save_dir = run_training(best_lr_s4, best_wd_s4, cand_curvature, dry_run=args.dry_run)
        if not args.dry_run:
            add_row(
                best_val_metrics(save_dir, best_lr_s4, best_wd_s4, cand_curvature),
                "curvature_sweep_refined",
            )

    if args.dry_run:
        return

    rows = sorted(rows, key=lambda r: (str(r["lr"]), str(r["weight_decay"]), str(r["cand_curvature"])))
    best = min(rows, key=lambda r: float(r["best_val_nll"]))
    write_summary(rows, best)

    print("\nValidation summary")
    print(markdown_table(rows))
    print("\nSelected best config by validation NLL:")
    print(
        f"lr={best['lr']}, weight-decay={best['weight_decay']}, "
        f"hess_init={HESS_INIT}, cand_curvature={best['cand_curvature']}"
    )
    print("\nRecommended final uCBOpt run config:")
    print(f"lr: {best['lr']}")
    print(f"weight-decay: {best['weight_decay']}")
    print(f"hess_init: {HESS_INIT}")
    print(f"cand_curvature: {best['cand_curvature']}")
    print(f"beta1: {BETA1}")
    print(f"beta2: {BETA2}")
    print("epochs: 100")
    print("seeds: [0, 1, 2]")
    print(
        "final output directory pattern: "
        f"runs/final/ucbopt/{DATASET}_{MODEL}/"
        f"lr_{best['lr']}_wd_{best['weight_decay']}_hi_1.0_curv_{best['cand_curvature']}_ep_100"
    )


if __name__ == "__main__":
    main()
