import gc
import hashlib
import importlib.metadata
import json
import platform
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import dataset_digest, load_dataset
from .reporting import completed_repetitions, load_results, summarize_directory, write_csv
from .specs import DATASET_NAMES, METHODS, METRIC_COLUMNS, dataset_slug, repetition_seed


def _write_json(value, path):
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def _version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


def _manifest(config, device):
    folder = Path(__file__).resolve().parent
    source_hashes = {name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
                     for name in (f"{config['experiment']}.py", "data.py", "specs.py", "runner.py")}
    signature = {"experiment": config["experiment"], "seed": config["seed"], "parameters": config["parameters"],
                 "dataset_order": list(DATASET_NAMES), "methods": list(METHODS), "dtype": "float64",
                 "device": str(device), "source_hashes": source_hashes}
    environment = {"python": platform.python_version(),
                   **{package: _version(package) for package in ("torch", "numpy", "scipy", "pandas", "scikit-learn", "ucimlrepo")}}
    return {"signature": signature, "environment": environment, "data_hashes": {}}


def run(config):
    experiment = config["experiment"]
    selected_device = config["device"]
    if selected_device == "auto":
        selected_device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(selected_device)
    if device.type not in {"cpu", "cuda"}:
        raise ValueError("These float64 experiments support CPU or CUDA devices")
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; select --device cpu")
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "experiment.json"
    manifest = _manifest(config, device)
    if manifest_path.exists():
        saved = json.loads(manifest_path.read_text(encoding="utf-8"))
        if saved["signature"] != manifest["signature"] or saved["environment"] != manifest["environment"]:
            raise ValueError("Existing results have different settings, source, or package versions. Select a new --output-dir")
        manifest = saved
    elif any((output / f"{dataset_slug(name)}_all_runs.csv").exists() for name in DATASET_NAMES):
        raise ValueError("Existing CSVs have no experiment manifest. Use --summary-only or select a new --output-dir")
    _write_json(manifest, manifest_path)
    _write_json(config, output / "config.json")

    previous_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    engine = None
    primary = METRIC_COLUMNS[experiment]
    try:
        for name in config["datasets"]:
            path = output / f"{dataset_slug(name)}_all_runs.csv"
            frame = load_results(path, name, experiment, config["parameters"]["n_repetitions"])
            completed = completed_repetitions(frame)
            missing = sorted(set(range(config["parameters"]["n_repetitions"])) - completed)
            if not missing:
                print(f"{name}: all repetitions already complete", flush=True)
                continue
            raw_x, raw_y = load_dataset(name, config["data_dir"])
            digest = dataset_digest(raw_x, raw_y)
            previous_hash = manifest["data_hashes"].get(name)
            if previous_hash is not None and digest != previous_hash:
                raise ValueError(f"Cached data changed for {name}; refusing to combine results")
            manifest["data_hashes"][name] = digest
            _write_json(manifest, manifest_path)
            if engine is None:
                if experiment == "contamination":
                    from .contamination import ContaminationExperiment

                    engine = ContaminationExperiment(config["parameters"], device, config["seed"])
                else:
                    from .underrepresentation import UnderrepresentationExperiment

                    engine = UnderrepresentationExperiment(config["parameters"], device, config["seed"])
            for repetition in missing:
                seed = repetition_seed(config["seed"], name, repetition)
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if device.type == "cuda":
                    torch.cuda.manual_seed_all(seed)
                data = engine.build_data(raw_x, raw_y, seed)
                fitted = engine.fit_all_methods(data, seed)
                result = engine.evaluate(fitted, data)
                if set(result["Method"]) != set(METHODS) or len(result) != len(METHODS):
                    raise RuntimeError("A repetition must produce all six methods")
                if not np.isfinite(result[primary].to_numpy(dtype=float)).all():
                    raise RuntimeError(f"Nonfinite {primary} in {name}, repetition {repetition}")
                result["Dataset"], result["Repetition"] = name, repetition
                result["Experiment seed"] = seed
                result = result[["Dataset", "Repetition", "Experiment seed", "Method", primary]]
                frame = pd.concat([frame, result], ignore_index=True)
                frame = frame.drop_duplicates(["Repetition", "Method"], keep="last").sort_values(["Repetition", "Method"])
                write_csv(frame, path)
                print(f"{name}: repetition {repetition + 1}/{config['parameters']['n_repetitions']}, "
                      f"seed={seed}, train={data['n_train']}, best {primary}={result[primary].min():.3f}", flush=True)
                del data, fitted, result
                if device.type == "cuda":
                    torch.cuda.empty_cache()
                gc.collect()
            del raw_x, raw_y
    finally:
        torch.set_default_dtype(previous_dtype)
    summarize_directory(config)
