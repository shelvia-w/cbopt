"""Out-of-domain confidence helpers and logging coroutines."""

from os.path import join as pjoin

import numpy as np

from .coroutines import autoinitcoroutine, coro_dict2csv, coro_trackavg_weighted
from .metrics import cumconfidence, cumentropy


def confidence_from_prediction_npy(npyfile: str) -> np.ndarray:
    """Load saved probabilities and return each sample's max confidence."""
    probas = np.load(npyfile)
    return np.amax(probas, axis=1)


def coro_epochlog_ood(total: int, logfreq: int = 100, outputsaver=None):
    """Track running OOD confidence and entropy over one epoch."""
    conftracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    conf, ent = float("nan"), float("nan")
    try:
        yield
        while True:
            outprobas, i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            ent = enttracker.send((cumentropy(outprobas), bs))
            conf = conftracker.send((cumconfidence(outprobas), bs))
            if i % logfreq == 0:
                print(f"  {i}/{total}: conf={conf:.4f}, entropy={ent:.4f}")
    except StopIteration:
        return conf, ent


@autoinitcoroutine
def coro_log_ood(sw=None, logfreq: int = 100, save_dir=""):
    """Log OOD confidence and entropy to stdout, CSV, and TensorBoard."""
    ent, conf = float("nan"), float("nan")
    csvhead = ("epoch", "confidence", "entropy")
    csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            conf, ent = yield from coro_epochlog_ood(total, logfreq, outputsaver)
            print(f"\nEpoch {epoch}: conf={conf:.4f}, entropy={ent:.4f};\n")
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        pjoin(save_dir, f"{prefix}.csv"), csvhead
                    )
                csvcorologs[prefix].send(
                    {"epoch": epoch, "confidence": conf, "entropy": ent}
                )
            if sw is not None:
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.flush()
            epoch, prefix, total, outputsaver = yield (conf, ent)
    except StopIteration:
        return conf, ent
