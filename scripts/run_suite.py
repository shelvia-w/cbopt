"""Run the main, combined-dropout, or ablation experiments."""

import argparse
import json
import shlex
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def commands(suite, forwarded=()):
    manifest = json.loads((ROOT / "configs" / "suites.json").read_text(encoding="utf-8"))
    groups = manifest if suite == "all" else {suite: manifest[suite]}
    for group, paths in groups.items():
        script = "run_sweep.py" if group == "ablations" else "run_experiment.py"
        for path in paths:
            config = ROOT / path
            if not config.is_file():
                raise FileNotFoundError(config)
            yield [sys.executable, str(ROOT / "scripts" / script), "--config", str(config), *forwarded]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", choices=("main", "combined", "ablations", "all"), default="main")
    parser.add_argument("--list", action="store_true", help="Print commands only; never construct models or load data")
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--evaluate-only", action="store_true")
    args = parser.parse_args()
    forwarded = []
    for key in ("device", "data_dir"):
        if getattr(args, key):
            forwarded.extend(["--" + key.replace("_", "-"), getattr(args, key)])
    if args.seeds:
        forwarded.extend(["--seeds", *map(str, args.seeds)])
    if args.evaluate_only:
        forwarded.append("--evaluate-only")
    for command in commands(args.suite, forwarded):
        print(shlex.join(command), flush=True)
        if not args.list:
            subprocess.run(command, cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
