import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.datasets import load_diabetes

from .specs import DATASET_NAMES, dataset_slug

UCI_DATASETS = {
    "Concrete": 165, "Airfoil": 291, "Real Estate Valuation": 477,
    "Power Plant": 294, "Energy Heating": 242, "Auto MPG": 9, "Automobile": 10,
}
WINE_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/wine-quality/winequality-red.csv"


def _clean_xy(x, y):
    x = pd.DataFrame(x).copy()
    y = pd.DataFrame(y).copy()
    x = pd.get_dummies(x, dummy_na=True)
    x = x.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    x = x.fillna(x.median(numeric_only=True)).fillna(0.0)
    x = x.loc[:, x.nunique(dropna=False) > 1]
    y = y.apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    valid = y.notna().all(axis=1).to_numpy()
    x, y = x.loc[valid].reset_index(drop=True), y.loc[valid].reset_index(drop=True)
    return x.to_numpy(dtype=np.float64), y.to_numpy(dtype=np.float64)


def _uci_xy(dataset_id, target_column=None):
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=dataset_id)
    x, y = _clean_xy(dataset.data.features, dataset.data.targets)
    if target_column is None:
        if y.shape[1] != 1:
            raise ValueError(f"UCI dataset {dataset_id} has {y.shape[1]} targets; specify target_column")
        y = y[:, 0]
    else:
        y = y[:, int(target_column)]
    return x, y


def _load_computer_hardware():
    from ucimlrepo import fetch_ucirepo

    dataset = fetch_ucirepo(id=29)
    frame = dataset.data.original.copy()
    frame.columns = frame.columns.astype(str).str.strip()
    target = next(column for column in frame.columns if column.lower() == "prp")
    x = frame.drop(columns=[target, "ERP", "erp", "VendorName", "vendor", "ModelName", "model"], errors="ignore")
    x, y = _clean_xy(x, frame[[target]])
    return x, y[:, 0]


def load_dataset(name, cache_dir):
    if name not in DATASET_NAMES:
        raise ValueError(f"Unknown dataset: {name}")
    cache_dir = Path(cache_dir)
    cache = cache_dir / f"{dataset_slug(name)}.npz"
    if cache.exists():
        with np.load(cache, allow_pickle=False) as saved:
            return saved["x"], saved["y"]
    if name in UCI_DATASETS:
        x, y = _uci_xy(UCI_DATASETS[name], 0 if name == "Energy Heating" else None)
    elif name == "Computer Hardware":
        x, y = _load_computer_hardware()
    elif name == "Wine Red":
        frame = pd.read_csv(WINE_URL, sep=";")
        x, y = _clean_xy(frame.drop(columns="quality"), frame[["quality"]])
        y = y[:, 0]
    else:
        dataset = load_diabetes()
        x, y = np.asarray(dataset.data, float), np.asarray(dataset.target, float)
    cache_dir.mkdir(parents=True, exist_ok=True)
    temporary = cache.with_suffix(".tmp")
    with temporary.open("wb") as f:
        np.savez_compressed(f, x=x, y=y)
    temporary.replace(cache)
    return x, y


def dataset_digest(x, y):
    digest = hashlib.sha256()
    for array in (x, y):
        array = np.ascontiguousarray(array)
        digest.update(json.dumps({"shape": array.shape, "dtype": str(array.dtype)}).encode())
        digest.update(array.tobytes())
    return digest.hexdigest()
