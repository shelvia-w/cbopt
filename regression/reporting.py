from html import escape
from pathlib import Path

import numpy as np
import pandas as pd

from .specs import METHODS, METRICS, RMSE_COLUMNS, dataset_slug


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
    required = {"Dataset", "Repetition", "Method", *METRICS[experiment]}
    if not required.issubset(frame.columns):
        raise ValueError(f"{path} is missing columns: {sorted(required - set(frame.columns))}")
    if not frame["Dataset"].eq(dataset_name).all():
        raise ValueError(f"{path} contains another dataset")
    repetition = pd.to_numeric(frame["Repetition"], errors="raise")
    if not np.equal(repetition, np.floor(repetition)).all():
        raise ValueError(f"{path} contains noninteger repetition IDs")
    frame = frame[repetition.between(0, repetitions - 1) & frame["Method"].isin(METHODS)].copy()
    frame["Repetition"] = frame["Repetition"].astype(int)
    frame = frame.drop_duplicates(["Repetition", "Method"], keep="last")
    if not np.isfinite(frame[RMSE_COLUMNS[experiment]].to_numpy(dtype=float)).all():
        raise ValueError(f"{path} contains nonfinite RMSE values")
    return frame


def completed_repetitions(frame):
    if frame.empty:
        return set()
    counts = frame.groupby("Repetition")["Method"].nunique()
    return set(counts[counts == len(METHODS)].index.astype(int))


def _metric_table(results, datasets, metric, path):
    stats = results.groupby(["Dataset", "Method"])[metric].agg(["mean", "std"]).reset_index()
    stats["formatted"] = stats.apply(lambda row: f"{row['mean']:.3f} ± {row['std']:.3f}", axis=1)
    table = stats.pivot(index="Dataset", columns="Method", values="formatted").reindex(index=datasets, columns=METHODS)
    means = stats.pivot(index="Dataset", columns="Method", values="mean").reindex(index=datasets, columns=METHODS)
    write_csv(table, path, index=True)
    html = ["<!doctype html><html><head><meta charset=\"utf-8\"><title>" + escape(metric) + "</title>",
            "<style>body{font-family:serif}table{border-collapse:collapse}th,td{padding:.4em .7em;text-align:right}"
            "thead{border-bottom:1px solid}tbody{border-bottom:1px solid}th:first-child{text-align:left}</style>",
            "</head><body><table><caption>" + escape(metric) + " (mean ± sample standard deviation)</caption><thead><tr>",
            "<th>Dataset</th>", *["<th>" + escape(method) + "</th>" for method in METHODS], "</tr></thead><tbody>"]
    for dataset in datasets:
        order = sorted(means.loc[dataset].dropna().unique())
        html.append("<tr><th>" + escape(dataset) + "</th>")
        for method in METHODS:
            value = escape(str(table.loc[dataset, method]))
            mean = means.loc[dataset, method]
            if order and mean == order[0]:
                value = "<strong>" + value + "</strong>"
            elif len(order) > 1 and mean == order[1]:
                value = "<u>" + value + "</u>"
            html.append("<td>" + value + "</td>")
        html.append("</tr>")
    html.append("</tbody></table></body></html>")
    path.with_suffix(".html").write_text("\n".join(html), encoding="utf-8")


def write_reports(results, config):
    output = Path(config["output_dir"])
    output.mkdir(parents=True, exist_ok=True)
    experiment = config["experiment"]
    primary = RMSE_COLUMNS[experiment]
    datasets = [name for name in config["datasets"] if name in set(results["Dataset"])]
    write_csv(results, output / "all_datasets_all_runs.csv")
    aggregate = results.groupby(["Dataset", "Method"])[list(METRICS[experiment])].agg(["mean", "std", "count"]).reset_index()
    aggregate.columns = [" ".join(column).strip() for column in aggregate.columns]
    write_csv(aggregate, output / "aggregate_metrics.csv")
    means = results.groupby(["Dataset", "Method"])[primary].mean().reset_index()
    winners = means.loc[means.groupby("Dataset")[primary].idxmin()].sort_values("Dataset")
    winners = winners.rename(columns={"Method": "Winner", primary: "Mean RMSE"})
    write_csv(winners, output / "winner_by_dataset.csv")

    for dataset in datasets:
        ranking = results[results["Dataset"].eq(dataset)].groupby("Method")[primary].agg(["mean", "std", "count"]).sort_values("mean")
        write_csv(ranking, output / f"{dataset_slug(dataset)}_rmse_ranking.csv", index=True)
    _metric_table(results, datasets, primary, output / "rmse_mean_std_table.csv")
    if experiment == "underrepresentation":
        for metric in ("RMSE", "High-risk MAE", "Asymmetric cost"):
            name = metric.lower().replace(" ", "_").replace("-", "_")
            _metric_table(results, datasets, metric, output / f"{name}_mean_std.csv")

    ranked = results.copy()
    summary = pd.DataFrame(index=pd.Index(METHODS, name="Method"))
    rank_metrics = {"RMSE": primary}
    if experiment == "underrepresentation":
        rank_metrics["MAE"] = "MAE"
    for label, metric in rank_metrics.items():
        column = f"{label} Rank"
        ranked[column] = ranked.groupby(["Dataset", "Repetition"])[metric].rank(method="average")
        by_dataset = ranked.groupby(["Dataset", "Method"])[column].mean()
        summary[f"Mean {label} Rank"] = by_dataset.groupby("Method").mean().reindex(METHODS)
        means = results.groupby(["Dataset", "Method"])[metric].mean().reset_index()
        wins = means[metric].eq(means.groupby("Dataset")[metric].transform("min"))
        counts = means.assign(win=wins).groupby("Method")["win"].sum().reindex(METHODS, fill_value=0)
        summary[f"{label} Dataset Wins"] = counts.map(lambda count: f"{int(count)}/{len(datasets)}")
    summary = summary.sort_values("Mean RMSE Rank").reset_index()
    write_csv(summary, output / "rank_summary.csv")
    return summary


def summarize_directory(config):
    output = Path(config["output_dir"])
    frames = []
    for name in config["datasets"]:
        frame = load_results(output / f"{dataset_slug(name)}_all_runs.csv", name, config["experiment"], config["parameters"]["n_repetitions"])
        if not frame.empty:
            frame = frame[frame["Repetition"].isin(completed_repetitions(frame))]
            if not frame.empty:
                frames.append(frame)
    if not frames:
        raise FileNotFoundError(f"No complete per-dataset repetitions in {output}")
    results = pd.concat(frames, ignore_index=True)
    write_reports(results, config)
    print(f"Wrote summaries for {results['Dataset'].nunique()} datasets to {output}")
    return results
