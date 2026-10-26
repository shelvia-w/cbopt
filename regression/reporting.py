from pathlib import Path

import numpy as np
import pandas as pd

from .specs import DATASET_LABELS, METHODS, METHOD_LABELS, METRIC_COLUMNS, dataset_slug


def write_csv(frame, path, index=False):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    frame.to_csv(temporary, index=index)
    temporary.replace(path)


def load_results(path, dataset_name, experiment, repetitions):
    path = Path(path)
    if not path.exists():
        return pd.DataFrame()
    frame = pd.read_csv(path)
    metric = METRIC_COLUMNS[experiment]
    if experiment == "contamination" and "Test RMSE" in frame:
        if metric not in frame:
            frame = frame.rename(columns={"Test RMSE": metric})
        elif not np.array_equal(frame[metric].to_numpy(), frame["Test RMSE"].to_numpy(), equal_nan=True):
            raise ValueError(f"{path} contains conflicting RMSE and Test RMSE columns")
    required = {"Dataset", "Repetition", "Method", metric}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(frame.columns))}")
    if not frame["Dataset"].eq(dataset_name).all():
        raise ValueError(f"{path} contains another dataset")
    repetition = pd.to_numeric(frame["Repetition"], errors="raise")
    if not np.isfinite(repetition).all() or not np.equal(repetition, np.floor(repetition)).all():
        raise ValueError(f"{path} contains invalid repetition IDs")
    frame["Repetition"] = repetition.astype(int)
    frame = frame[repetition.between(0, repetitions - 1) & frame["Method"].isin(METHODS)].copy()
    if frame.duplicated(["Repetition", "Method"]).any():
        raise ValueError(f"{path} contains duplicate method/repetition rows")
    frame[metric] = pd.to_numeric(frame[metric], errors="raise")
    if not np.isfinite(frame[metric].to_numpy(dtype=float)).all() or (frame[metric] < 0).any():
        raise ValueError(f"{path} contains invalid {metric} values")
    columns = ["Dataset", "Repetition"]
    if "Experiment seed" in frame:
        columns.append("Experiment seed")
    return frame[columns + ["Method", metric]]


def completed_repetitions(frame):
    if frame.empty:
        return set()
    counts = frame.groupby("Repetition")["Method"].nunique()
    return set(counts[counts == len(METHODS)].index.astype(int))


def write_reports(results, config):
    metric = METRIC_COLUMNS[config["experiment"]]
    datasets = config["datasets"]
    repetitions = config["parameters"]["n_repetitions"]
    expected = pd.MultiIndex.from_product([datasets, range(repetitions), METHODS],
                                         names=["Dataset", "Repetition", "Method"])
    actual = pd.MultiIndex.from_frame(results[["Dataset", "Repetition", "Method"]])
    if actual.has_duplicates or len(actual) != len(expected) or not expected.isin(actual).all():
        raise ValueError("Tables require every configured dataset, repetition, and method exactly once")
    if not np.isfinite(results[metric].to_numpy(dtype=float)).all() or (results[metric] < 0).any():
        raise ValueError(f"Tables require finite, nonnegative {metric} values")

    stats = results.groupby(["Dataset", "Method"])[metric].agg(["mean", "std"])
    means = stats["mean"].unstack("Method").reindex(index=datasets, columns=METHODS)
    stds = stats["std"].unstack("Method").reindex(index=datasets, columns=METHODS)
    run_ranks = results.groupby(["Dataset", "Repetition"])[metric].rank(method="average", ascending=True)
    ranks = results.assign(Rank=run_ranks).groupby("Method")["Rank"].mean().reindex(METHODS)

    table = pd.DataFrame(index=datasets, columns=METHODS)
    for dataset in datasets:
        for method in METHODS:
            table.loc[dataset, method] = f"{means.loc[dataset, method]:.2f} ± {stds.loc[dataset, method]:.2f}"
    table.index = [DATASET_LABELS.get(name, name) for name in datasets]
    table.loc["Mean Rank"] = [f"{ranks.loc[method]:.2f}" for method in METHODS]
    table = table.rename(columns=METHOD_LABELS)
    table.index.name = "Dataset"
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    write_csv(table, output / "table.csv", index=True)
    return table


def summarize_directory(config):
    output = Path(config["output_dir"])
    repetitions = config["parameters"]["n_repetitions"]
    frames, incomplete = [], []
    for name in config["datasets"]:
        frame = load_results(output / f"{dataset_slug(name)}_all_runs.csv", name, config["experiment"], repetitions)
        missing = sorted(set(range(repetitions)) - completed_repetitions(frame))
        if missing:
            incomplete.append(f"{name}: missing complete repetitions {missing}")
        else:
            frames.append(frame)
    if incomplete:
        raise ValueError("Cannot produce a complete table:\n" + "\n".join(incomplete))
    results = pd.concat(frames, ignore_index=True)
    write_reports(results, config)
    print(f"Wrote {output / 'table.csv'}")
    return results
