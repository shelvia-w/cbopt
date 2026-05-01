"""Epoch-level logging coroutines."""

import torch
from torch import LongTensor

from timeit import default_timer as timer

from .calibration import bins2acc, bins2conf, bins2ece, coro_binsmerger, data2bins
from .coroutines import autoinitcoroutine, coro_dict2csv, coro_trackavg_weighted
from .metrics import AUROC, cumbrier, cumentropy, cumnll, top5corrects


def onehot(t: LongTensor, nclasses: int, dtype=torch.long):
    """Convert class indices to one-hot vectors."""
    if torch.numel(t) == 0:
        return torch.empty(0, nclasses, device=t.device)
    t_onehot = torch.zeros(*t.size(), nclasses, device=t.device, dtype=dtype)
    return t_onehot.scatter(t.dim(), t.unsqueeze(-1), 1)


# =============================
# Epoch metric collection
# =============================

def coro_epochlog(
    total: int,
    logfreq: int = 100,
    nbin: int = 10,
    outputsaver=None,
    global_rank=None,
):
    """Track epoch metrics from batch predictions."""
    losstracker = coro_trackavg_weighted()
    nlltracker = coro_trackavg_weighted()
    briertracker = coro_trackavg_weighted()
    binsmerger = coro_binsmerger()
    top5tracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    auroctracker = AUROC()
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    try:
        yield
        while True:
            (outprobas, gt, loss), i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            probas, preds = outprobas.max(dim=1)
            bins = binsmerger.send(
                data2bins(zip((preds == gt).tolist(), probas.tolist()), nbin)
            )
            auroctracker.collect((preds == gt).tolist(), probas.tolist())
            loss = losstracker.send((loss * bs, bs))
            nll = nlltracker.send((cumnll(outprobas, gt), bs))
            brier = briertracker.send(
                (cumbrier(outprobas, onehot(gt, outprobas.size(1), outprobas.dtype)), bs)
            )
            acc5 = top5tracker.send((top5corrects(outprobas, gt), bs))
            ent = enttracker.send((cumentropy(outprobas), bs))
            if i % logfreq == 0 and (not global_rank):
                print(
                    f"  {i}/{total}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"acc={bins2acc(bins):.4f}, acc@5={acc5:.4f}, "
                    f"brier={brier:.4f}, ece={bins2ece(bins):.4f}, "
                    f"conf={bins2conf(bins):.4f}, entropy={ent:.4f}"
                )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroctracker.compute()


# =============================
# Standard epoch logging
# =============================

@autoinitcoroutine
def coro_log_metrics(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
):
    """Log per-epoch metrics to stdout, CSV, and TensorBoard."""
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    if save_dir:
        csvhead = ("epoch", "loss", "nll", "acc", "acc@5",
                   "brier", "ece", "confidence", "entropy", "auroc")
        csvcorologs = {}
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            bins, loss, nll, brier, acc5, ent, auroc = yield from coro_epochlog(
                total, logfreq, nbin, outputsaver
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            print(
                f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                f"acc={acc:.4f}, acc@5={acc5:.4f}, "
                f"brier={brier:.4f}, ece={ece:.4f}, "
                f"conf={conf:.4f}, entropy={ent:.4f}, auroc={auroc:.4f};\n"
            )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        f"{save_dir}/{prefix}.csv", csvhead
                    )
                csvcorologs[prefix].send(
                    {"epoch": epoch, "loss": loss, "nll": nll, "acc": acc, "acc@5": acc5,
                     "brier": brier, "ece": ece, "confidence": conf, "entropy": ent, "auroc": auroc}
                )
            if sw is not None:
                sw.add_scalar(f"{prefix}/loss", loss, epoch)
                sw.add_scalar(f"{prefix}/nll", nll, epoch)
                sw.add_scalar(f"{prefix}/error", 1 - acc, epoch)
                sw.add_scalar(f"{prefix}/error@5", 1 - acc5, epoch)
                sw.add_scalar(f"{prefix}/brier", brier, epoch)
                sw.add_scalar(f"{prefix}/ece", ece, epoch)
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.add_scalar(f"{prefix}/auroc", auroc, epoch)
                sw.flush()
            epoch, prefix, total, outputsaver = yield (bins, loss, nll, brier, acc5, ent, auroc)
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc


# =============================
# Epoch logging with time
# =============================

@autoinitcoroutine
def coro_log_timed(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
    global_rank=None,
    append: bool = False,
):
    """Log per-epoch metrics together with elapsed time."""
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    start = timer()
    if save_dir:
        csvhead = ("time", "epoch", "loss", "nll", "acc", "acc@5",
                   "brier", "ece", "confidence", "entropy", "auroc")
        csvcorologs = {}
    else:
        csvcorologs = None
        csvhead = None
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            if not global_rank:
                print(f"*** Epoch {epoch} {prefix} ***\n")

            bins, loss, nll, brier, acc5, ent, auroc = yield from coro_epochlog(
                total, logfreq, nbin, outputsaver, global_rank
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            duration = timer() - start

            if not global_rank:
                print(
                    f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"acc={acc:.4f}, acc@5={acc5:.4f}, "
                    f"brier={brier:.4f}, ece={ece:.4f}, "
                    f"conf={conf:.4f}, entropy={ent:.4f}, auroc={auroc:.4f};\n"
                    f"Current elapsed time: {duration:.2f} s\n"
                )

            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        f"{save_dir}/{prefix}.csv", csvhead, append=append
                    )
                csvcorologs[prefix].send(
                    {"time": duration, "epoch": epoch, "loss": loss, "nll": nll,
                     "acc": acc, "acc@5": acc5, "brier": brier, "ece": ece,
                      "confidence": conf, "entropy": ent, "auroc": auroc}
                )
            if sw is not None:
                sw.add_scalar(f"{prefix}/loss", loss, epoch)
                sw.add_scalar(f"{prefix}/nll", nll, epoch)
                sw.add_scalar(f"{prefix}/error", 1 - acc, epoch)
                sw.add_scalar(f"{prefix}/error@5", 1 - acc5, epoch)
                sw.add_scalar(f"{prefix}/brier", brier, epoch)
                sw.add_scalar(f"{prefix}/ece", ece, epoch)
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.add_scalar(f"{prefix}/auroc", auroc, epoch)
                sw.flush()
            epoch, prefix, total, outputsaver = yield (bins, loss, nll, brier, acc5, ent, auroc)
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc
