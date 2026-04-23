"""General-purpose utilities."""

import csv
import os
import pathlib
import statistics
from os.path import join as pjoin

import numpy as np
import torch

from .coroutines import coro_npybatchgatherer
from data.data_utils import corrupt_labels


def deterministic_run(seed=0):
    """Configure libraries for reproducible execution."""
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.use_deterministic_algorithms(True)


def check_cuda() -> None:
    """Print available CUDA devices and the active device."""
    if not torch.cuda.is_available():
        raise Exception("No CUDA device available")
    cuda_count = torch.cuda.device_count()
    print("{0} CUDA device(s) available:".format(cuda_count))
    for i in range(cuda_count):
        print(
            "- {0}: {1} ({2})".format(
                i,
                torch.cuda.get_device_name(i),
                torch.cuda.get_device_capability(i),
            )
        )
    curr_idx = torch.cuda.current_device()
    print("Currently using device {0}".format(curr_idx))


def mkdir(dirpath: str, parents: bool = False, exist_ok: bool = False) -> None:
    """Create a directory."""
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirp(dirpath: str) -> None:
    """Create a directory and any missing parents."""
    mkdir(dirpath, parents=True, exist_ok=True)


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    """Create a saver coroutine for batched prediction outputs."""
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile),
        ndata,
        (outclass,),
        True,
        str(torch.get_default_dtype())[6:],
    )


def summarize_csv(csvfile):
    """Print mean and standard deviation for CSV metrics."""
    with open(csvfile, "r") as csvfp:
        reader = csv.DictReader(csvfp)
        criteria = [k for k in reader.fieldnames if k != "epoch"]
        maxlen = max(len(k) for k in criteria)
        values = {k: [] for k in criteria}
        for row in reader:
            for k, v in row.items():
                if k != "epoch":
                    values[k].append(float(v))
        for k, vals in values.items():
            print(
                f"{k:>{maxlen}}:\tmean {statistics.mean(vals):.4f}, "
                f"std={statistics.stdev(vals):.4f}" if len(vals) > 1 else "std=NaN"
            )
