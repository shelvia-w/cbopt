"""Summarize metrics across independent random seeds."""

import argparse
import csv
import math
import statistics
from collections import defaultdict
from pathlib import Path

GROUP_KEYS = ("experiment", "dataset", "model", "method", "mc_dropout", "checkpoint", "prediction", "split")
METRICS = ("acc", "acc@5", "nll", "ece", "brier", "auroc", "mean_confidence", "entropy",
           "aupr-in", "aupr-out", "fpr95", "dterr")


def summarize(root, output, expected_seeds=None):
    groups = defaultdict(dict)
    for path in sorted(Path(root).glob("**/seed*/metrics.csv")):
        with path.open(newline="", encoding="utf-8") as f:
            for row in csv.DictReader(f):
                key = tuple(row[k] for k in GROUP_KEYS)
                seed = int(row["seed"])
                if seed in groups[key]:
                    raise ValueError(f"Duplicate seed {seed} for {key}; refusing to combine repeated runs")
                groups[key][seed] = row
    if not groups:
        raise FileNotFoundError(f"No per-seed metrics.csv files under {root}")
    rows = []
    for key, by_seed in sorted(groups.items()):
        if expected_seeds is not None and set(by_seed) != set(expected_seeds):
            raise ValueError(f"Expected seeds {list(expected_seeds)}, found {sorted(by_seed)} for {key}")
        result = dict(zip(GROUP_KEYS, key))
        result.update(n_runs=len(by_seed), seeds=",".join(map(str, sorted(by_seed))))
        for metric in METRICS:
            present = [row.get(metric, "") for row in by_seed.values()]
            if not any(value != "" for value in present):
                continue
            values = [float(value) for value in present if value != "" and math.isfinite(float(value))]
            result[f"{metric}_n"] = len(values)
            result[f"{metric}_mean"] = statistics.mean(values) if values else float("nan")
            result[f"{metric}_std"] = statistics.stdev(values) if len(values) > 1 else float("nan")
        rows.append(result)
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("outputs"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--require-seeds", type=int, nargs="+")
    args = parser.parse_args()
    summarize(args.root, args.output or args.root / "summary.csv", args.require_seeds)


if __name__ == "__main__":
    main()
