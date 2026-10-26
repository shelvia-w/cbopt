import csv
import os
from datetime import datetime

import numpy as np


def autoinitcoroutine(coro):
    """Prime a coroutine before returning it."""
    def initcoro(*args, **kwargs):
        cr = coro(*args, **kwargs)
        next(cr)
        return cr

    return initcoro


def coro_timer():
    """Yield the initial timestamp, then (timestamp, elapsed) pairs."""
    now = datetime.now()
    yield now
    while True:
        now, past = datetime.now(), now
        yield (now, now - past)


def _div0(a, b):
    return 0.0 if b == 0 else a / b


@autoinitcoroutine
def coro_trackavg_weighted():
    """Yield sum(values) / sum(weights) as pairs arrive."""
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
    with open(to, "a+" if append else "w", buffering=1) as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames, **kwargs)
        if not append:
            writer.writeheader()
        d = yield
        while True:
            writer.writerow(d)
            d = yield


@autoinitcoroutine
def coro_npybatchgatherer(
    filepath,
    entrycount: int,
    entryshape=(),
    overwrites=False,
    dtype="float",
):
    """Write incoming batches to a memory-mapped NumPy array."""
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
