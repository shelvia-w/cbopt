"""Two-stage uCBOptAdaptCurv tuning sweep for LeNet on Fashion-MNIST.

Stage 1 — gamma/beta3 sweep:
  Sweep all (gamma, beta3) combinations with h0, lr, wd held fixed.
  Select the top 3 pairs by best validation NLL.
  Rows tagged: stage = "gamma_beta3_sweep".

Stage 2 — h0/lr/wd sweep:
  For each of the top 3 (gamma, beta3) pairs from Stage 1,
  sweep a small grid of hess_init, learning rate, and weight decay.
  Select the overall best config by best validation NLL.
  Rows tagged: stage = "h0_lr_wd_sweep".

Outputs (written to OUTPUT_ROOT/ucbopt_adaptcurv/fmnist_lenet/):
  tuning_summary.csv          — all Stage 1–2 results
  recommended_final_config.yaml — best config from Stage 2
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
OPTIMIZER = "ucbopt_adaptcurv"
SEED = "0"
EPOCHS = "30"
BATCH = "128"
VAL_BATCH = "256"
TVSPLIT = "0.9"
WORKERS = os.environ.get("CBO_WORKERS", os.environ.get("PBS_NCPUS", "32"))
DEVICE = os.environ.get("CBO_DEVICE", "cuda")

BETA1 = "0.9"
BETA2 = "0.99999"
EPS = "1e-8"
CLIP_RADIUS = "inf"

# Stage 1: fixed values while sweeping gamma x beta3
H0_DEFAULT = "0.05"
LR_DEFAULT = "1e-2"
WD_DEFAULT = "2e-3"

GAMMA_SWEEP = [
    "0.0",
    "5e-3",
    "1e-2",
    "5e-2",
    "1e-1",
    "2e-1",
    "3e-1",
    "4e-1",
    "5e-1",
]

BETA3_SWEEP = [
    "1.00001",
    "1.0001",
    "1.001",
    "1.01",
]

# Stage 2: small grids swept for each top-3 (gamma, beta3) pair
TOP_K = 3
H0_STAGE2 = ["0.05", "0.1", "0.2"]
LR_STAGE2 = ["5e-3", "1e-2", "2e-2"]
WD_STAGE2 = ["5e-4", "2e-3", "5e-3"]

def run_dir(
    gamma: str,
    beta3: str,
    hess_init: str,
    weight_decay: str = WD_DEFAULT,
    lr: str = LR_DEFAULT,
    epochs: str = EPOCHS,
) -> Path:
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


def train_command(
    gamma: str,
    beta3: str,
    hess_init: str,
    save_dir: Path,
    weight_decay: str = WD_DEFAULT,
    lr: str = LR_DEFAULT,
) -> list[str]:
    return [
        sys.executable,
        "-u",
        "-m",
        "experiments.train_ucbopt_adaptcurv",
        MODEL,
        DATASET,
        "-s", SEED,
        "-d", DEVICE,
        "-dd", str(DATA_DIR),
        "-sd", str(save_dir),
        "-lr", lr,
        "--wd", weight_decay,
        "--hess_init", hess_init,
        "--gamma", gamma,
        "--beta1", BETA1,
        "--beta2", BETA2,
        "--beta3", beta3,
        "--eps", EPS,
        "--clip-radius", CLIP_RADIUS,
        "--no-rescale_lr",
        "-e", EPOCHS,
        "-tb", BATCH,
        "-vb", VAL_BATCH,
        "-sp", TVSPLIT,
        "-j", WORKERS,
    ]


def run_training(
    gamma: str,
    beta3: str,
    hess_init: str,
    weight_decay: str = WD_DEFAULT,
    lr: str = LR_DEFAULT,
    dry_run: bool = False,
) -> Path:
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

    print(f"Running lr={lr}, wd={weight_decay}, h0={hess_init}, gamma={gamma}, beta3={beta3}")
    with (save_dir / "stdout.log").open("w", encoding="utf-8") as log:
        result = subprocess.run(
            cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT, text=True, check=False,
        )
    if result.returncode != 0:
        raise RuntimeError(
            f"Training failed for lr={lr}, wd={weight_decay}, h0={hess_init}, "
            f"gamma={gamma}, beta3={beta3}. See {save_dir / 'stdout.log'}"
        )
    return save_dir


def best_val_metrics(
    save_dir: Path,
    gamma: str,
    beta3: str,
    hess_init: str,
    weight_decay: str = WD_DEFAULT,
    lr: str = LR_DEFAULT,
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
    rows: list[dict[str, str | float | int]],
    best: dict[str, str | float | int],
) -> None:
    summary_dir = OUTPUT_ROOT / OPTIMIZER / f"{DATASET}_{MODEL}"
    summary_dir.mkdir(parents=True, exist_ok=True)

    with (summary_dir / "tuning_summary.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "stage",
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
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    recommendation = (
        "optimizer: ucbopt_adaptcurv\n"
        f"dataset: {DATASET}\n"
        f"model: {MODEL}\n"
        "seeds: [0, 1, 2]\n"
        f"device: {DEVICE}\n"
        f"traindir: \"final/ucbopt_adaptcurv/{DATASET}_{MODEL}/"
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
        "train_flags:\n"
        "  - --no-rescale_lr\n"
    )
    (summary_dir / "recommended_final_config.yaml").write_text(recommendation, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true", help="Print generated commands without running training."
    )
    args = parser.parse_args()

    all_rows: list[dict[str, str | float | int]] = []
    seen: set[tuple[str, str, str, str, str]] = set()

    def add_row(row: dict[str, str | float | int], stage: str) -> None:
        key = (
            str(row["hess_init"]), str(row["lr"]), str(row["weight_decay"]),
            str(row["gamma"]), str(row["beta3"]),
        )
        if key not in seen:
            seen.add(key)
            all_rows.append({**row, "stage": stage})

    # ------------------------------------------------------------------
    # Stage 1: sweep gamma x beta3, fixed h0 / lr / wd
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print(f"Stage 1: gamma x beta3 sweep  (h0={H0_DEFAULT}, lr={LR_DEFAULT}, wd={WD_DEFAULT})")
    print("=" * 60)

    for gamma in GAMMA_SWEEP:
        for beta3 in BETA3_SWEEP:
            save_dir = run_training(gamma, beta3, H0_DEFAULT, WD_DEFAULT, LR_DEFAULT, dry_run=args.dry_run)
            if not args.dry_run:
                add_row(
                    best_val_metrics(save_dir, gamma, beta3, H0_DEFAULT, WD_DEFAULT, LR_DEFAULT),
                    "gamma_beta3_sweep",
                )

    if args.dry_run:
        print("\n[dry-run] Stage 2 would sweep h0/lr/wd for the top-3 (gamma, beta3) pairs.")
        return

    stage1_rows = [r for r in all_rows if r["stage"] == "gamma_beta3_sweep"]
    stage1_sorted = sorted(stage1_rows, key=lambda r: float(r["best_val_nll"]))
    top_pairs = [(str(r["gamma"]), str(r["beta3"])) for r in stage1_sorted[:TOP_K]]

    print(f"\nTop {TOP_K} (gamma, beta3) pairs from Stage 1:")
    for i, (g, b3) in enumerate(top_pairs, 1):
        nll = stage1_sorted[i - 1]["best_val_nll"]
        print(f"  {i}. gamma={g}, beta3={b3}  (val NLL={nll:.6f})")

    # ------------------------------------------------------------------
    # Stage 2: h0 x lr x wd sweep for each top-(gamma, beta3) pair
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("Stage 2: h0 x lr x wd sweep for top-3 (gamma, beta3) pairs")
    print("=" * 60)

    for gamma, beta3 in top_pairs:
        print(f"\n  --- gamma={gamma}, beta3={beta3} ---")
        for hess_init in H0_STAGE2:
            for lr in LR_STAGE2:
                for wd in WD_STAGE2:
                    save_dir = run_training(gamma, beta3, hess_init, wd, lr)
                    add_row(
                        best_val_metrics(save_dir, gamma, beta3, hess_init, wd, lr),
                        "h0_lr_wd_sweep",
                    )

    stage2_rows = [r for r in all_rows if r["stage"] == "h0_lr_wd_sweep"]
    best_overall = min(stage2_rows, key=lambda r: float(r["best_val_nll"]))

    write_summary(all_rows, best_overall)

    print("\n" + "=" * 60)
    print("Validation summary (sorted by best val NLL)")
    print("=" * 60)
    print(markdown_table(all_rows))

    print("\nBest overall config (Stage 2):")
    print(
        f"  lr={best_overall['lr']}, wd={best_overall['weight_decay']}, h0={best_overall['hess_init']}, "
        f"gamma={best_overall['gamma']}, beta3={best_overall['beta3']}  "
        f"(val NLL={best_overall['best_val_nll']:.6f})"
    )
    print(
        "\nRecommended final output directory:\n"
        f"  final/ucbopt_adaptcurv/{DATASET}_{MODEL}/"
        f"lr_{best_overall['lr']}_wd_{best_overall['weight_decay']}_h0_{best_overall['hess_init']}"
        f"_gamma_{best_overall['gamma']}_b3_{best_overall['beta3']}_ep_100"
    )


if __name__ == "__main__":
    main()
