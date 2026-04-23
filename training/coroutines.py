"""Coroutine helpers used across training and evaluation."""

import csv
import os
from datetime import datetime

import numpy as np


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


def _div0(a, b):
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
            val, weight = yield _div0(total, totalweights)
    except StopIteration:
        return _div0(total, totalweights)


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


@autoinitcoroutine
def coro_npybatchgatherer(filepath, entrycount: int, entryshape=(), overwrites=False, dtype="float"):
    """Stream batches into a memory-mapped `.npy` file."""
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
