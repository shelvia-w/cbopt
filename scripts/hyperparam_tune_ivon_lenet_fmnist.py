"""HPC IVON grid-search tuning sweep for LeNet on Fashion-MNIST.

Runs every combination of HESS_INIT_SWEEP, LR_SWEEP, and WD_SWEEP; the
overall best config is selected by validation NLL.
"""

from __future__ import annotations

import argparse
import csv
import math
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
OPTIMIZER = "ivon"
SEED = "0"
EPOCHS = "30"
BATCH = "128"
VAL_BATCH = "256"
TVSPLIT = "0.9"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")
ESS = "54000"
HESS_INIT_SWEEP = ["0.05", "0.1", "0.5"]
TRAIN_SAMPLES = "1"
RESCALE_LR = True
LR_SWEEP = ["1e-1", "5e-2", "1e-2"]
WD_SWEEP = ["5e-4", "1e-3", "2e-3"]


def run_dir(lr: str, weight_decay: str, hess_init: str, epochs: str = EPOCHS) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{lr}_wd_{weight_decay}_ess_{ESS}_h0_{hess_init}_ep_{epochs}"
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


def train_command(lr: str, weight_decay: str, hess_init: str, save_dir: Path) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_ivon",
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
        "--ess",
        ESS,
        "--hess_init",
        hess_init,
        "--mc_samples",
        TRAIN_SAMPLES,
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


def run_training(lr: str, weight_decay: str, hess_init: str, dry_run: bool = False) -> Path | None:
    save_dir = run_dir(lr, weight_decay, hess_init)
    val_csv = save_dir / "val.csv"

    cmd = train_command(lr, weight_decay, hess_init, save_dir)
    if dry_run:
        print(" ".join(cmd))
        return save_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if completed_val_csv(val_csv):
        print(f"Skipping completed run: {save_dir}")
        return save_dir

    print(f"Running lr={lr}, weight-decay={weight_decay}, hess_init={hess_init}")
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
        print(
            f"Training failed (skipping) for lr={lr}, weight-decay={weight_decay}, "
            f"hess_init={hess_init}. See {save_dir / 'stdout.log'}"
        )
        return None
    return save_dir


def best_val_metrics(save_dir: Path, lr: str, weight_decay: str, hess_init: str) -> dict[str, str | float | int] | None:
    val_csv = save_dir / "val.csv"
    if not val_csv.exists():
        return None

    with val_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    finite_rows = [r for r in rows if math.isfinite(float(r["nll"]))]
    if not finite_rows:
        print(f"No finite NLL in {val_csv} (all NaN/Inf) — skipping")
        return None

    best = min(finite_rows, key=lambda row: float(row["nll"]))
    return {
        "lr": lr,
        "weight_decay": weight_decay,
        "hess_init": hess_init,
        "best_val_nll": float(best["nll"]),
        "best_val_accuracy": float(best["acc"]),
        "best_val_ece": float(best["ece"]),
        "epoch": int(float(best["epoch"])),
        "save_dir": str(save_dir),
    }


def markdown_table(rows: list[dict[str, str | float | int]]) -> str:
    header = "| lr | weight-decay | h0 | best val NLL | best val accuracy | best val ECE | epoch of best val NLL | stage |"
    divider = "|---|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for row in rows:
        lines.append(
            "| {lr} | {weight_decay} | {hess_init} | {best_val_nll:.6f} | "
            "{best_val_accuracy:.6f} | {best_val_ece:.6f} | {epoch} | {stage} |".format(**row)
        )
    return "\n".join(lines)


def write_summary(rows: list[dict[str, str | float | int]], best: dict[str, str | float | int]) -> None:
    summary_dir = OUTPUT_ROOT / OPTIMIZER / f"{DATASET}_{MODEL}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "tuning_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "lr",
                "weight_decay",
                "hess_init",
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
        "optimizer: ivon\n"
        f"dataset: {DATASET}\n"
        f"model: {MODEL}\n"
        "seeds: [0, 1, 2]\n"
        f"device: {DEVICE}\n"
        f"traindir: \"runs/final/ivon/{DATASET}_{MODEL}/"
        f"lr_{best['lr']}_wd_{best['weight_decay']}_ess_{ESS}_h0_{best['hess_init']}_ep_100\"\n"
        "train_args:\n"
        f"  lr: \"{best['lr']}\"\n"
        "  e: \"100\"\n"
        f"  weight-decay: \"{best['weight_decay']}\"\n"
        f"  ess: \"{ESS}\"\n"
        f"  hess_init: \"{best['hess_init']}\"\n"
        "  mc_samples: \"1\"\n"
        "  tbatch: \"128\"\n"
        "  vbatch: \"256\"\n"
        "  tvsplit: \"0.9\"\n"
        f"  j: \"{WORKERS}\"\n"
        f"train_flags: {['rescale_lr'] if RESCALE_LR else []}\n"
    )
    (summary_dir / "recommended_final_config.yaml").write_text(recommendation, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Print generated commands without running training.")
    args = parser.parse_args()

    rows: list[dict[str, str | float | int]] = []
    seen: set[tuple[str, str, str]] = set()

    def add_row(row: dict[str, str | float | int], stage: str) -> None:
        key = (str(row["hess_init"]), str(row["lr"]), str(row["weight_decay"]))
        if key not in seen:
            seen.add(key)
            rows.append({**row, "stage": stage})

    for hess_init in HESS_INIT_SWEEP:
        print(f"\n=== hess_init={hess_init} ===")
        for lr in LR_SWEEP:
            for weight_decay in WD_SWEEP:
                save_dir = run_training(lr, weight_decay, hess_init, dry_run=args.dry_run)
                if not args.dry_run and save_dir is not None:
                    metrics = best_val_metrics(save_dir, lr, weight_decay, hess_init)
                    if metrics is not None:
                        add_row(metrics, "grid_search")

    if args.dry_run:
        return

    rows = sorted(rows, key=lambda r: (str(r["hess_init"]), str(r["lr"]), str(r["weight_decay"])))
    if not rows:
        raise RuntimeError("No successful IVON tuning runs produced finite validation NLL.")
    best = min(rows, key=lambda r: float(r["best_val_nll"]))
    write_summary(rows, best)

    print("\nValidation summary")
    print(markdown_table(rows))
    print("\nSelected best config by validation NLL:")
    print(f"lr={best['lr']}, weight-decay={best['weight_decay']}, hess_init={best['hess_init']}")
    print("\nRecommended final IVON run config:")
    print(f"lr: {best['lr']}")
    print(f"weight-decay: {best['weight_decay']}")
    print(f"ess: {ESS}")
    print(f"hess_init: {best['hess_init']}")
    print(f"mc_samples: {TRAIN_SAMPLES}")
    print("epochs: 100")
    print("seeds: [0, 1, 2]")
    print(
        "final output directory pattern: "
        f"runs/final/ivon/{DATASET}_{MODEL}/lr_{best['lr']}_wd_{best['weight_decay']}_ess_{ESS}_h0_{best['hess_init']}_ep_100"
    )


if __name__ == "__main__":
    main()
