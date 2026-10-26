import argparse
from pathlib import Path

from .specs import DATASET_NAMES, normalize_numbers, validate_config

ROOT = Path(__file__).resolve().parents[1]


def main(experiment, argv=None):
    parser = argparse.ArgumentParser(description=f"Run the {experiment} regression experiment")
    parser.add_argument("--config", type=Path, default=ROOT / "configs" / "regression" / f"{experiment}.yaml")
    parser.add_argument("--datasets", nargs="+", choices=DATASET_NAMES)
    parser.add_argument("--repetitions", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--data-dir", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--summary-only", action="store_true", help="Read existing per-dataset CSVs and write tables")
    parser.add_argument("--list", action="store_true", help="List supported datasets without loading dependencies or data")
    args = parser.parse_args(argv)
    if args.list:
        print("\n".join(DATASET_NAMES))
        return

    import yaml

    config = normalize_numbers(yaml.safe_load(args.config.read_text(encoding="utf-8")))
    for key in ("datasets", "seed", "device", "data_dir", "output_dir"):
        value = getattr(args, key)
        if value is not None:
            config[key] = str(value) if isinstance(value, Path) else value
    if args.repetitions is not None:
        config["parameters"]["n_repetitions"] = args.repetitions
    validate_config(config, experiment)
    if args.summary_only:
        from .reporting import summarize_directory

        summarize_directory(config)
    else:
        from .runner import run

        run(config)
