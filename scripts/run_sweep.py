"""Run a parameter sweep from a YAML specification."""

import argparse
import copy
import subprocess
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def expand(spec_path):
    spec_path = Path(spec_path)
    spec = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    base = yaml.safe_load((spec_path.parent / spec["base_config"]).resolve().read_text(encoding="utf-8"))
    for value in spec["values"]:
        cfg = copy.deepcopy(base)
        container = cfg
        keys = spec["parameter"].split(".")
        for key in keys[:-1]:
            container = container[key]
        container[keys[-1]] = value
        cfg["experiment"] = f"ablations/{spec_path.stem}/{value}"
        cfg["output_dir"] = f"outputs/{cfg['experiment']}"
        yield cfg


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--list", action="store_true")
    args, forwarded = parser.parse_known_args()
    for cfg in expand(args.config):
        if args.list:
            print(cfg["experiment"])
            continue
        output = ROOT / cfg["output_dir"]
        output.mkdir(parents=True, exist_ok=True)
        expanded = output / "expanded.yaml"
        expanded.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "run_experiment.py"),
                        "--config", str(expanded), *forwarded], cwd=ROOT, check=True)


if __name__ == "__main__":
    main()
