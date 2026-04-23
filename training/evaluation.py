"""Evaluation helpers for in-domain and out-of-domain uncertainty workflows."""

from os.path import join as pjoin
import numpy as np
import torch
import torch.nn.functional as nnf
from torch import Tensor
from torch.utils.data import DataLoader

from .coroutines import autoinitcoroutine, coro_dict2csv, coro_npybatchgatherer, coro_trackavg_weighted
from .engine import avgdups
from .metrics import cumentropy


def confidence_from_prediction_npy(npyfile: str) -> np.ndarray:
    probas = np.load(npyfile)
    return np.amax(probas, axis=1)


def cumconfidence(probas: Tensor) -> float:
    return torch.sum(torch.max(probas, dim=1)[0]).item()


def do_evalbatch_ood(batchinput, model, dups: int = 1, repeat: int = 1):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        prob = nnf.softmax(output, 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    return cumprob


def do_evalbatch_von(batchinput, model, optimizer, repeat: int = 1):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / repeat
    return cumprob


def do_evalbatch_swag(batchinput, models):
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob


def do_evalbatch_duq(batchinput, model):
    inputs = batchinput[:-1]
    scores = model(*inputs)
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return prob


def do_evalbatch_sngp(batchinput, model):
    inputs = batchinput[:-1]
    logits = model(*inputs, mean_field=True, return_gp_cov=False)
    prob = torch.softmax(logits, dim=1)
    return prob


def dup_collate_fn(dups: int):
    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,) * (t.ndim - 1)), torch.as_tensor(gts)
    return collate_fn


def get_outputsaver(save_dir, ndata, outclass, predictionfile):
    return coro_npybatchgatherer(
        pjoin(save_dir, predictionfile),
        ndata,
        (outclass,),
        True,
        str(torch.get_default_dtype())[6:],
    )


def coro_epochlog_ood(total: int, logfreq: int = 100, outputsaver=None):
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
                csvcorologs[prefix].send({"epoch": epoch, "confidence": conf, "entropy": ent})
            if sw is not None:
                sw.add_scalar(f"{prefix}/uncertainty", 1 - conf, epoch)
                sw.add_scalar(f"{prefix}/entropy", ent, epoch)
                sw.flush()
            epoch, prefix, total, outputsaver = yield (conf, ent)
    except StopIteration:
        return conf, ent
