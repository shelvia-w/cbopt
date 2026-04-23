from typing import Callable, Any, Optional
import collections
import os
import random
from timeit import default_timer as timer

import numpy as np
from sklearn.metrics import roc_auc_score
import torch
from torch import Tensor, LongTensor, nn
import torch.nn.functional as nnf
from torch.utils.data import DataLoader

from .calibration import (
    data2bins,
    coro_binsmerger,
    bins2acc,
    bins2ece,
    bins2conf,
)
from .utils import (
    coro_trackavg_weighted,
    coro_dict2csv,
    coro_npybatchgatherer,
    autoinitcoroutine,
)

SummaryWriter = None


def deteministic_run(seed=0):
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


def avgdups(t: Tensor, dups: int) -> Tensor:
    return torch.mean(t.view(dups, -1, *t.size()[1:]), dim=0)


def apply_batch(batch, fn: Callable[[Tensor], Any]):
    if isinstance(batch, Tensor):
        return fn(batch)
    elif isinstance(batch, collections.abc.Mapping):
        return {k: apply_batch(sample, fn) for k, sample in batch.items()}
    elif isinstance(batch, collections.abc.Sequence):
        return [apply_batch(sample, fn) for sample in batch]
    else:
        return batch


def top5corrects(outprobas: Tensor, gt: LongTensor) -> int:
    preds = outprobas.topk(5)[1].t()
    return torch.sum(preds.eq(gt.view(1, -1)), dtype=torch.long).item()


def cumentropy(probas: Tensor) -> float:
    return torch.sum(
        -probas * torch.log(probas + torch.finfo(probas.dtype).tiny)
    ).item()


def cumnll(probas: Tensor, gts: LongTensor) -> float:
    nlls = -torch.log(torch.gather(probas, -1, gts.unsqueeze(-1)).squeeze(-1))
    return torch.sum(nlls).item()


def onehot(t: LongTensor, nclasses: int, dtype=torch.long):
    if torch.numel(t) == 0:
        return torch.empty(0, nclasses, device=t.device)
    t_onehot = torch.zeros(*t.size(), nclasses, device=t.device, dtype=dtype)
    return t_onehot.scatter(t.dim(), t.unsqueeze(-1), 1)


def cumbrier(probas: Tensor, onehotgts: Tensor) -> float:
    return torch.sum((probas - onehotgts) ** 2).item()


def cumnll_logprob(logprob: Tensor, gts: LongTensor) -> float:
    nlls = -torch.gather(logprob, -1, gts.unsqueeze(-1)).squeeze(-1)
    return torch.sum(nlls).item()


def coro_epochlog(
    total: int, logfreq: int = 100, nbin: int = 10, outputsaver=None, global_rank=None
):
    losstracker = coro_trackavg_weighted()
    nlltracker = coro_trackavg_weighted()
    briertracker = coro_trackavg_weighted()
    binsmerger = coro_binsmerger()
    top5tracker = coro_trackavg_weighted()
    enttracker = coro_trackavg_weighted()
    bins, loss, nll, brier, acc5, ent = (None,) + (float("nan"),) * 5
    try:
        yield
        while True:
            (outprobas, gt, loss), i = yield
            if outputsaver is not None:
                outputsaver.send(outprobas.cpu().numpy())
            bs = outprobas.size(0)
            probas, preds = torch.max(outprobas, dim=1)
            bins = binsmerger.send(
                data2bins(zip((preds == gt).tolist(), probas.tolist()), nbin)
            )
            loss = losstracker.send((loss * bs, bs))
            nll = nlltracker.send((cumnll(outprobas, gt), bs))
            brier = briertracker.send(
                (cumbrier(outprobas, onehot(gt, outprobas.size(1), outprobas.dtype)), bs)
            )
            acc5 = top5tracker.send((top5corrects(outprobas, gt), bs))
            ent = enttracker.send((cumentropy(outprobas), bs))
            if (not global_rank) and (i % logfreq == 0):
                print(
                    f"  {i}/{total}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={bins2acc(bins):.4f}, "
                    f"conf={bins2conf(bins):.4f}, ece={bins2ece(bins):.4f}, "
                    f"acc@5={acc5:.4f}, entropy={ent:.4f}"
                )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent


@autoinitcoroutine
def coro_log(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
    global_rank=None,
):
    bins, loss, nll, brier, acc5, ent = (None,) + (float("nan"),) * 5
    if save_dir:
        csvhead = ("epoch", "loss", "nll", "brier", "acc", "confidence", "ece", "acc@5", "entropy")
        csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            (bins, loss, nll, brier, acc5, ent) = yield from coro_epochlog(
                total, logfreq, nbin, outputsaver, global_rank
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            if not global_rank:
                print(
                    f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                    f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f};\n"
                )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        f"{save_dir}/{prefix}.csv", csvhead
                    )
                csvcorologs[prefix].send(
                    {"epoch": epoch, "loss": loss, "nll": nll, "brier": brier,
                     "acc": acc, "confidence": conf, "ece": ece, "acc@5": acc5, "entropy": ent}
                )
            (epoch, prefix, total, outputsaver) = yield (bins, loss, nll, brier, acc5, ent)
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent


def coro_epochlog_auroc(
    total: int, logfreq: int = 100, nbin: int = 10, outputsaver=None, global_rank=None,
):
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
            probas, preds = torch.max(outprobas, dim=1)
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
                    f"brier={brier:.4f}, acc={bins2acc(bins):.4f}, "
                    f"conf={bins2conf(bins):.4f}, ece={bins2ece(bins):.4f}, "
                    f"acc@5={acc5:.4f}, entropy={ent:.4f}"
                )
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroctracker.compute()


@autoinitcoroutine
def coro_log_metrics(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
):
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6
    if save_dir:
        csvhead = ("epoch", "loss", "nll", "brier", "acc", "confidence", "ece", "acc@5", "entropy", "auroc")
        csvcorologs = dict()
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            print(f"*** Epoch {epoch} {prefix} ***\n")
            (bins, loss, nll, brier, acc5, ent, auroc) = yield from coro_epochlog_auroc(
                total, logfreq, nbin, outputsaver
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            print(
                f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f}, "
                f"auroc={auroc:.4f};\n"
            )
            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        f"{save_dir}/{prefix}.csv", csvhead
                    )
                csvcorologs[prefix].send(
                    {"epoch": epoch, "loss": loss, "nll": nll, "brier": brier,
                     "acc": acc, "confidence": conf, "ece": ece, "acc@5": acc5,
                     "entropy": ent, "auroc": auroc}
                )
            if sw is not None:
                sw.add_scalar(f"{prefix}/loss", loss, epoch)
                sw.add_scalar(f"{prefix}/nll", nll, epoch)
                sw.add_scalar(f"{prefix}/brier", brier, epoch)
                sw.add_scalar(f"{prefix}/error", 1 - acc, epoch)
                sw.add_scalar(f"{prefix}/error@5", 1 - acc5, epoch)
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.add_scalar(f"{prefix}/auroc", auroc, epoch)
                sw.add_scalar(f"{prefix}/ece", ece, epoch)
                sw.flush()
            (epoch, prefix, total, outputsaver) = yield (bins, loss, nll, brier, acc5, ent, auroc)
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc


@autoinitcoroutine
def coro_log_timed(
    sw=None,
    logfreq: int = 100,
    nbin: int = 10,
    save_dir="",
    global_rank=None,
    append: bool = False,
):
    bins, loss, nll, brier, acc5, ent, auroc = (None,) + (float("nan"),) * 6

    if save_dir:
        csvhead = ("time", "epoch", "loss", "nll", "brier", "acc", "confidence", "ece", "acc@5", "entropy", "auroc")
        csvcorologs = dict()
        start = timer()
    else:
        csvcorologs = None
        csvhead = None
        start = None
    try:
        epoch, prefix, total, outputsaver = yield
        while True:
            if not global_rank:
                print(f"*** Epoch {epoch} {prefix} ***\n")

            (bins, loss, nll, brier, acc5, ent, auroc) = yield from coro_epochlog_auroc(
                total, logfreq, nbin, outputsaver, global_rank
            )
            acc, conf, ece = bins2acc(bins), bins2conf(bins), bins2ece(bins)
            duration = timer() - start

            if not global_rank:
                print(
                    f"\nEpoch {epoch}: loss={loss:.4f}, nll={nll:.4f}, "
                    f"brier={brier:.4f}, acc={acc:.4f}, conf={conf:.4f}, "
                    f"ece={ece:.4f}, acc@5={acc5:.4f}, entropy={ent:.4f}, "
                    f"auroc={auroc:.4f};\nCurrent elapsed time: {duration:.2f} s\n"
                )

            if save_dir:
                if prefix not in csvcorologs:
                    csvcorologs[prefix] = coro_dict2csv(
                        f"{save_dir}/{prefix}.csv", csvhead, append=append
                    )
                csvcorologs[prefix].send(
                    {"time": duration, "epoch": epoch, "loss": loss, "nll": nll,
                     "brier": brier, "acc": acc, "confidence": conf, "ece": ece,
                     "acc@5": acc5, "entropy": ent, "auroc": auroc}
                )
            if sw is not None:
                sw.add_scalar(f"{prefix}/loss", loss, epoch)
                sw.add_scalar(f"{prefix}/nll", nll, epoch)
                sw.add_scalar(f"{prefix}/brier", brier, epoch)
                sw.add_scalar(f"{prefix}/error", 1 - acc, epoch)
                sw.add_scalar(f"{prefix}/error@5", 1 - acc5, epoch)
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.add_scalar(f"{prefix}/auroc", auroc, epoch)
                sw.add_scalar(f"{prefix}/ece", ece, epoch)
                sw.flush()
            (epoch, prefix, total, outputsaver) = yield (bins, loss, nll, brier, acc5, ent, auroc)
    except StopIteration:
        return bins, loss, nll, brier, acc5, ent, auroc


def do_epoch(
    loader: DataLoader,
    compbatch,
    corolog,
    device=torch.device("cpu"),
    **comp_kwargs,
):
    i = -1
    batchoutput = None
    for i, batchinput in enumerate(loader):
        batchinput = apply_batch(
            batchinput, lambda t: t.to(device, non_blocking=True)
        )
        corolog.send((batchoutput, i))
        batchoutput = compbatch(batchinput, **comp_kwargs)
    corolog.send((batchoutput, i + 1))


def do_trainbatch(batchinput, model, optimizer, dups: int = 1, repeat: int = 1):
    optimizer.zero_grad(set_to_none=True)
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        ll = avgdups(ll, dups) if dups > 1 else ll
        loss = nnf.nll_loss(ll, gt) / repeat
        loss.backward()
        cumloss += loss.item()
        prob = nnf.softmax(output.detach(), 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    optimizer.step()
    return cumprob, gt, cumloss


def do_evalbatch(batchinput, model, dups: int = 1, repeat: int = 1):
    inputs, gt = batchinput[:-1], batchinput[-1]
    cumloss = 0.0
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        ll = nnf.log_softmax(output, 1)
        ll = avgdups(ll, dups) if dups > 1 else ll
        loss = nnf.nll_loss(ll, gt) / repeat
        cumloss += loss.item()
        prob = nnf.softmax(output, 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    return cumprob, gt, cumloss


# --- BatchNorm utilities (from Izmailov et al. BSD 2-Clause) ---

BatchNorm = nn.modules.batchnorm._BatchNorm


def _check_bn(module, flag):
    if isinstance(module, BatchNorm):
        flag[0] = True


def check_bn(model):
    flag = [False]
    model.apply(lambda module: _check_bn(module, flag))
    return flag[0]


def reset_bn(module):
    if isinstance(module, BatchNorm):
        module.running_mean = torch.zeros_like(module.running_mean)
        module.running_var = torch.ones_like(module.running_var)


def _get_momenta(module, momenta):
    if isinstance(module, BatchNorm):
        momenta[module] = module.momentum


def _set_momenta(module, momenta):
    if isinstance(module, BatchNorm):
        module.momentum = momenta[module]


def bn_update(loader, model, device=None, **kwargs):
    if not check_bn(model):
        return
    model.train()
    momenta = {}
    model.apply(reset_bn)
    model.apply(lambda module: _get_momenta(module, momenta))
    n = 0
    with torch.no_grad():
        for t, _ in loader:
            b = t.size(0)
            t = t.to(device=device, non_blocking=True)
            momentum = float(b) / (n + b)
            for module in momenta.keys():
                module.momentum = momentum
            model(t, **kwargs)
            n += b
    model.apply(lambda module: _set_momenta(module, momenta))


class AUROC:
    def __init__(self):
        self.positive = []
        self.confidence = []

    def collect(self, positives, confidences):
        self.positive += positives
        self.confidence += confidences

    def compute(self) -> float:
        try:
            auroc = roc_auc_score(
                np.asarray(self.positive), np.asarray(self.confidence)
            )
        except ValueError:
            auroc = float("nan")
        return auroc
