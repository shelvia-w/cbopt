"""Compute evaluation metrics from saved predictions."""

import argparse
import csv
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from core.metrics import classification_metrics
from datasets.ood_utils import OODMetrics


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--indomain", required=True, type=Path)
    parser.add_argument("--ood", type=Path)
    parser.add_argument("--labels", type=Path)
    parser.add_argument("--bins", type=int, default=20)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.ood is None and args.labels is None:
        parser.error("Provide --ood and/or --labels")
    probs = np.load(args.indomain)
    rows = []
    if args.labels:
        rows.append({"split": "indomain", **classification_metrics(probs, np.load(args.labels), args.bins)})
    if args.ood:
        out_probs = np.load(args.ood)
        rows.append({"split": "ood", **OODMetrics(probs.max(axis=1), out_probs.max(axis=1)).get_all()})
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({key for row in rows for key in row}))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
