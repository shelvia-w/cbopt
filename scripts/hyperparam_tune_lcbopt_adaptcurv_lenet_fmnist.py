"""HPC/local lCBOptAdaptCurv tuning sweep for LeNet on Fashion-MNIST.

For each value in HESS_INIT_SWEEP:
  1. LR sweep    (fixed wd=WD_DEFAULT, gamma=GAMMA_DEFAULT, beta3=BETA3_DEFAULT)
  2. WD sweep    (fixed best lr, gamma=GAMMA_DEFAULT, beta3=BETA3_DEFAULT)
  3. Gamma sweep (fixed best lr, best wd, beta3=BETA3_DEFAULT)
  4. Beta3 sweep (fixed best lr, best wd, best gamma)

The overall best across all h0 values determines the recommended final config.
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
OPTIMIZER = "lcbopt_adaptcurv"
SEED = "0"
EPOCHS = "30"
BATCH = "128"
VAL_BATCH = "256"
TVSPLIT = "0.9"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")

LR_DEFAULT = "1e-3"
LR_SWEEP = ["1e-1", "5e-2", "1e-2", "5e-3", "1e-3", "5e-4", "1e-4"]
WD_DEFAULT = "1e-4"
WD_SWEEP = ["1e-5", "1e-4", "5e-4", "1e-3", "2e-3"]
HESS_INIT_SWEEP = ["2.0", "1.0", "0.5", "0.1"]
BETA1 = "0.9"
BETA2 = "0.99999"
BETA3_DEFAULT = "0.999"
GAMMA_DEFAULT = "1e-3"
EPS = "1e-8"
CLIP_RADIUS = "inf"

GAMMA_SWEEP = [
    "0.0",
    "1e-4",
    "5e-4",
    "1e-3",
    "5e-3",
    "1e-2",
    "5e-2",
    "1e-1",
    "4e-1",
]

BETA3_SWEEP = [
    "0.9",
    "0.99",
    "0.999",
    "0.9999",
    "0.99999",
]


def run_dir(gamma: str, beta3: str, hess_init: str, weight_decay: str = WD_DEFAULT, lr: str = LR_DEFAULT, epochs: str = EPOCHS) -> Path:
    return (
        OUTPUT_ROOT
        / OPTIMIZER
        / f"{DATASET}_{MODEL}"
        / f"lr_{lr}_wd_{weight_decay}_h0_{hess_init}_gamma_{gamma}_b3_{beta3}_ep_{epochs}"
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


def train_command(gamma: str, beta3: str, hess_init: str, save_dir: Path, weight_decay: str = WD_DEFAULT, lr: str = LR_DEFAULT) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_lcbopt_adaptcurv",
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
        hess_init,
        "--gamma",
        gamma,
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


def run_training(gamma: str, beta3: str, hess_init: str, weight_decay: str = WD_DEFAULT, lr: str = LR_DEFAULT, dry_run: bool = False) -> Path:
    save_dir = run_dir(gamma, beta3, hess_init, weight_decay, lr)
    val_csv = save_dir / "val.csv"

    cmd = train_command(gamma, beta3, hess_init, save_dir, weight_decay, lr)
    if dry_run:
        print(" ".join(cmd))
        return save_dir

    DATA_DIR.mkdir(parents=True, exist_ok=True)
    save_dir.mkdir(parents=True, exist_ok=True)

    if completed_val_csv(val_csv):
        print(f"Skipping completed run: {save_dir}")
        return save_dir

    print(f"Running lr={lr}, wd={weight_decay}, gamma={gamma}, beta3={beta3}, hess_init={hess_init}")
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
            f"Training failed for lr={lr}, wd={weight_decay}, gamma={gamma}, beta3={beta3}, "
            f"hess_init={hess_init}. See {save_dir / 'stdout.log'}"
        )
    return save_dir


def best_val_metrics(
    save_dir: Path, gamma: str, beta3: str, hess_init: str, weight_decay: str = WD_DEFAULT, lr: str = LR_DEFAULT
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
        "hess_init": hess_init,
        "gamma": gamma,
        "beta1": BETA1,
        "beta2": BETA2,
        "beta3": beta3,
        "best_val_nll": float(best["nll"]),
        "best_val_accuracy": float(best["acc"]),
        "best_val_ece": float(best["ece"]),
        "epoch": int(float(best["epoch"])),
        "save_dir": str(save_dir),
    }


def markdown_table(rows: list[dict[str, str | float | int]]) -> str:
    header = (
        "| lr | weight-decay | h0 | gamma | beta3 | best val NLL | best val accuracy | best val ECE | epoch | stage |"
    )
    divider = "|---|---:|---:|---:|---:|---:|---:|---:|---:|---|"
    lines = [header, divider]
    for row in sorted(rows, key=lambda r: float(r["best_val_nll"])):
        lines.append(
            "| {lr} | {weight_decay} | {hess_init} | {gamma} | {beta3} | {best_val_nll:.6f} | "
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
                "gamma",
                "beta1",
                "beta2",
                "beta3",
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
        "optimizer: lcbopt_adaptcurv\n"
        f"dataset: {DATASET}\n"
        f"model: {MODEL}\n"
        "seeds: [0, 1, 2]\n"
        f"device: {DEVICE}\n"
        f"traindir: \"final/lcbopt_adaptcurv/{DATASET}_{MODEL}/"
        f"lr_{best['lr']}_wd_{best['weight_decay']}_h0_{best['hess_init']}"
        f"_gamma_{best['gamma']}_b3_{best['beta3']}_ep_100\"\n"
        "train_args:\n"
        f"  lr: \"{best['lr']}\"\n"
        "  e: \"100\"\n"
        f"  weight-decay: \"{best['weight_decay']}\"\n"
        f"  hess_init: \"{best['hess_init']}\"\n"
        f"  gamma: \"{best['gamma']}\"\n"
        f"  beta1: \"{best['beta1']}\"\n"
        f"  beta2: \"{best['beta2']}\"\n"
        f"  beta3: \"{best['beta3']}\"\n"
        f"  eps: \"{EPS}\"\n"
        f"  clip_radius: \"{CLIP_RADIUS}\"\n"
        "  tvsplit: \"0.9\"\n"
        "  tbatch: \"128\"\n"
        "  vbatch: \"256\"\n"
        f"  j: \"{WORKERS}\"\n"
    )
    (summary_dir / "recommended_final_config.yaml").write_text(recommendation, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    rows: list[dict[str, str | float | int]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def add_row(row: dict[str, str | float | int], stage: str) -> None:
        key = (str(row["hess_init"]), str(row["lr"]), str(row["weight_decay"]), str(row["gamma"]), str(row["beta3"]))
        if key not in seen:
            seen.add(key)
            rows.append({**row, "stage": stage})

    for hess_init in HESS_INIT_SWEEP:
        print(f"\n=== hess_init={hess_init} ===")

        print("Step 1: learning-rate sweep")
        for lr in LR_SWEEP:
            save_dir = run_training(GAMMA_DEFAULT, BETA3_DEFAULT, hess_init, WD_DEFAULT, lr, dry_run=args.dry_run)
            if not args.dry_run:
                add_row(best_val_metrics(save_dir, GAMMA_DEFAULT, BETA3_DEFAULT, hess_init, WD_DEFAULT, lr), "lr_sweep")

        if args.dry_run:
            best_lr = "<best_lr_from_step_1>"
        else:
            h0_lr_rows = [r for r in rows if str(r["hess_init"]) == hess_init and r["stage"] == "lr_sweep"]
            best_lr = str(min(h0_lr_rows, key=lambda r: float(r["best_val_nll"]))["lr"])
            print(f"Best learning rate from step 1 (h0={hess_init}): {best_lr}")

        print("Step 2: weight-decay sweep")
        for weight_decay in WD_SWEEP:
            save_dir = run_training(GAMMA_DEFAULT, BETA3_DEFAULT, hess_init, weight_decay, best_lr, dry_run=args.dry_run)
            if not args.dry_run:
                add_row(best_val_metrics(save_dir, GAMMA_DEFAULT, BETA3_DEFAULT, hess_init, weight_decay, best_lr), "wd_sweep")

        if args.dry_run:
            best_wd = "<best_wd_from_step_2>"
        else:
            h0_wd_rows = [
                r for r in rows if str(r["hess_init"]) == hess_init and r["stage"] in ("lr_sweep", "wd_sweep")
            ]
            best_row = min(h0_wd_rows, key=lambda r: float(r["best_val_nll"]))
            best_lr = str(best_row["lr"])
            best_wd = str(best_row["weight_decay"])
            print(f"Best config from step 2 (h0={hess_init}): lr={best_lr}, wd={best_wd}")

        print("Step 3: gamma sweep")
        for gamma in GAMMA_SWEEP:
            save_dir = run_training(gamma, BETA3_DEFAULT, hess_init, best_wd, best_lr, dry_run=args.dry_run)
            if not args.dry_run:
                add_row(best_val_metrics(save_dir, gamma, BETA3_DEFAULT, hess_init, best_wd, best_lr), "gamma_sweep")

        if args.dry_run:
            best_gamma = "<best_gamma_from_step_3>"
        else:
            h0_gamma_rows = [
                r for r in rows
                if str(r["hess_init"]) == hess_init and r["stage"] in ("lr_sweep", "wd_sweep", "gamma_sweep")
            ]
            best_row = min(h0_gamma_rows, key=lambda r: float(r["best_val_nll"]))
            best_lr = str(best_row["lr"])
            best_wd = str(best_row["weight_decay"])
            best_gamma = str(best_row["gamma"])
            print(f"Best config from step 3 (h0={hess_init}): lr={best_lr}, wd={best_wd}, gamma={best_gamma}")

        print("Step 4: beta3 sweep")
        for beta3 in BETA3_SWEEP:
            save_dir = run_training(best_gamma, beta3, hess_init, best_wd, best_lr, dry_run=args.dry_run)
            if not args.dry_run:
                add_row(best_val_metrics(save_dir, best_gamma, beta3, hess_init, best_wd, best_lr), "beta3_sweep")

    if args.dry_run:
        return

    rows = sorted(rows, key=lambda r: float(r["best_val_nll"]))
    best = rows[0]
    write_summary(rows, best)

    print("\nValidation summary (sorted by best val NLL)")
    print(markdown_table(rows))
    print("\nSelected best config by validation NLL:")
    print(
        f"lr={best['lr']}, weight-decay={best['weight_decay']}, "
        f"hess_init={best['hess_init']}, gamma={best['gamma']}, beta3={best['beta3']}"
    )
    print("\nRecommended final lCBOptAdaptCurv run config:")
    print(f"lr: {best['lr']}")
    print(f"weight-decay: {best['weight_decay']}")
    print(f"hess_init: {best['hess_init']}")
    print(f"gamma: {best['gamma']}")
    print(f"beta1: {BETA1}")
    print(f"beta2: {BETA2}")
    print(f"beta3: {best['beta3']}")
    print(f"eps: {EPS}")
    print(f"clip_radius: {CLIP_RADIUS}")
    print("epochs: 100")
    print("seeds: [0, 1, 2]")
    print(
        "final output directory pattern: "
        f"final/lcbopt_adaptcurv/{DATASET}_{MODEL}/"
        f"lr_{best['lr']}_wd_{best['weight_decay']}_h0_{best['hess_init']}"
        f"_gamma_{best['gamma']}_b3_{best['beta3']}_ep_100"
    )


if __name__ == "__main__":
    main()
