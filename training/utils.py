import os
import pathlib
import warnings
import statistics
import csv
import random
import zipfile
from datetime import datetime
from itertools import zip_longest
from typing import Any, Iterable, Mapping
from os.path import join as pjoin

import numpy as np
import torch
from torch import nn
from torch.optim import SGD, AdamW, Optimizer
from torch.optim.lr_scheduler import LRScheduler, LinearLR, CosineAnnealingLR

import models as _models
from methods.ucbopt import uCBOpt
from methods.baselines.ivon import IVON
from methods.baselines.adahessian import AdaHessian
from methods.baselines.vogn import VOGN
from models.uncertainty.duq import DUQ


def autoinitcoroutine(coro):
    """Wrap a coroutine so it is automatically primed."""
    def initcoro(*args, **kwargs):
        cr = coro(*args, **kwargs)
        next(cr)
        return cr
    return initcoro


def coro_timer():
    """Yield the current time, then successive time deltas."""
    now = datetime.now()
    yield now
    while True:
        now, past = datetime.now(), now
        yield (now, now - past)

def div0(a, b):
    """Return a / b, or 0.0 when b is zero."""
    return 0.0 if b == 0 else a / b


@autoinitcoroutine
def coro_trackavg_weighted():
    """Track and yield the running weighted average."""
    total, totalweights = 0.0, 0.0
    try:
        val, weight = yield
        while True:
            total, totalweights = total + val, totalweights + weight
            val, weight = yield div0(total, totalweights)
    except StopIteration:
        return div0(total, totalweights)


@autoinitcoroutine
def coro_dict2csv(to, fieldnames, append: bool = False, **kwargs):
    """Write streamed dictionaries to a CSV file."""
    with open(to, "a+" if append else "w", buffering=1) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, **kwargs)
        if not append:
            writer.writeheader()
        d = yield
        while True:
            writer.writerow(d)
            d = yield


def mkdir(dirpath: str, parents: bool = False, exist_ok: bool = False) -> None:
    """Create a directory."""
    pathlib.Path(dirpath).mkdir(parents=parents, exist_ok=exist_ok)


def mkdirp(dirpath: str) -> None:
    """Create a directory and any missing parents."""
    mkdir(dirpath, parents=True, exist_ok=True)


def unzip(zipped: str, unzipto: str = ".") -> None:
    """Extract a zip archive."""
    print(f"unzipping {zipped} to {unzipto} ...")
    with zipfile.ZipFile(zipped, "r") as zipfp:
        zipfp.extractall(unzipto)
    print("done.")


def asnpbatchiter(entryiter, batchsize: int, droplast: bool = False):
    """Yield NumPy batches from an entry iterator."""
    if droplast:
        yield from (np.stack(b) for b in zip(*[entryiter] * batchsize))
    else:
        yield from (
            np.concatenate(b).reshape(-1, *b[0].shape)
            for b in zip_longest(
                *[entryiter] * batchsize, fillvalue=np.array(())
            )
        )


@autoinitcoroutine
def coro_npybatchgatherer(filepath, entrycount: int, entryshape=(), overwrites=False, dtype="float"):
    """Stream batches into a memory-mapped .npy file."""
    if (not overwrites) and os.path.exists(filepath):
        raise FileExistsError(f"{filepath} already exists.")
    array = np.empty((entrycount, *entryshape), dtype)
    np.save(filepath, array)
    pos = 0
    try:
        batch = yield
        while pos < entrycount:
            ll = len(batch)
            mm = np.load(filepath, mmap_mode="r+")
            mm[pos : pos + ll] = batch
            pos += ll
            batch = yield
    except StopIteration:
        return pos


def npyiterator(filepath, transform=None, cache=1000000):
    """Iterate over entries in a .npy file, optionally transforming them."""
    ll = np.load(filepath, mmap_mode="r").shape[0]
    pos = 0
    while pos * cache <= ll:
        chunk = np.load(filepath, mmap_mode="r")[
            pos * cache : (pos + 1) * cache
        ]
        if transform is None:
            yield from chunk
        else:
            yield from (transform(d) for d in chunk)
        pos += 1


def npybatchiterator(filepath, batchsize: int, droplast: bool = False, transform=None, cache=1000000):
    """Yield batches from a .npy file iterator."""
    yield from asnpbatchiter(
        npyiterator(filepath, transform, cache), batchsize, droplast
    )

def savecheckpoint(
    to,
    modelname: str,
    modelargs: Iterable[Any],
    modelkwargs: Mapping[str, Any],
    model: nn.Module,
    optimizer: Optimizer,
    scheduler,
    **kwargs,
) -> None:
    """Save a model checkpoint with optimizer and scheduler state."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        _models.savemodel(
            to,
            modelname,
            modelargs,
            modelkwargs,
            model,
            **{
                "optimname": type(optimizer).__name__,
                "optimargs": optimizer.defaults,
                "optimstates": optimizer.state_dict(),
                "schedulername": type(scheduler).__name__ if scheduler is not None else None,
                "schedulerstates": scheduler.state_dict() if scheduler is not None else None,
            },
            **kwargs,
        )


def loadcheckpoint(fromfile, device=torch.device("cpu"), epochs=200):
    """Load a checkpoint and rebuild model, optimizer, and scheduler."""
    model, dic = _models.loadmodel(fromfile, device)
    optimname = dic["optimname"]
    optimargs = dict(dic.pop("optimargs"))

    if optimname == "AdamW":
        optimargs.pop("decoupled_weight_decay", None)

    optimizer = {
        "SGD": SGD,
        "AdamW": AdamW,
        "VOGN": VOGN,
        "AdaHessian": AdaHessian,
        "IVON": IVON,
        "uCBOpt": uCBOpt,
        "DUQ": DUQ,
    }[optimname](model.parameters(), **optimargs)
    optimizer.load_state_dict(dic.pop("optimstates"))
    schedulername = dic.get("schedulername")
    if schedulername in (None, "NoneType"):
        dic.pop("schedulerstates", None)
        return 0, model, optimizer, None, dic
    elif schedulername == "LinearLR":
        scheduler = LinearLR(optimizer)
    elif schedulername == "CosineAnnealingLR":
        scheduler = CosineAnnealingLR(optimizer, eta_min=0.0, T_max=epochs)
    else:
        raise NotImplementedError(f"Unknown scheduler: {schedulername}")
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", category=UserWarning)
        scheduler.load_state_dict(dic.pop("schedulerstates"))
    startepoch = scheduler.last_epoch
    return startepoch, model, optimizer, scheduler, dic


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


def corrupt_labels(dataset, noise_rate, seed=None, indices=None):
    """Randomly corrupt dataset labels at the given noise rate."""
    if seed is not None:
        random.seed(seed)

    targets = dataset.targets
    n_classes = max(targets) + 1
    idx_to_corrupt = indices if indices is not None else range(len(targets))

    n_corrupted = 0
    for i in idx_to_corrupt:
        if random.random() < noise_rate:
            wrong_labels = [c for c in range(n_classes) if c != targets[i]]
            targets[i] = random.choice(wrong_labels)
            n_corrupted += 1

    print(f"Corrupted {n_corrupted}/{len(idx_to_corrupt)} labels ({100*noise_rate:.0f}% noise)")
