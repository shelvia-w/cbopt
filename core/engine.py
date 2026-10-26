import collections
from typing import Any, Callable

import torch
import torch.nn.functional as nnf
from torch import Tensor
from torch.utils.data import DataLoader

SummaryWriter = None


def avgdups(t: Tensor, dups: int) -> Tensor:
    """Average repeated copies arranged as (duplicates, batch, ...)."""
    return torch.mean(t.view(dups, -1, *t.size()[1:]), dim=0)


def apply_batch(batch, fn: Callable[[Tensor], Any]):
    if isinstance(batch, Tensor):
        return fn(batch)
    if isinstance(batch, collections.abc.Mapping):
        return {k: apply_batch(sample, fn) for k, sample in batch.items()}
    if isinstance(batch, collections.abc.Sequence):
        return [apply_batch(sample, fn) for sample in batch]
    return batch


def do_epoch(
    loader: DataLoader,
    compbatch,
    corolog,
    device=torch.device("cpu"),
    **comp_kwargs,
):
    """Stream batches to the logger.

    End the epoch with corolog.throw(StopIteration) to obtain its metrics.
    """
    i = -1
    batchoutput = None
    metrics = None
    for i, batchinput in enumerate(loader):
        batchinput = apply_batch(
            batchinput, lambda t: t.to(device, non_blocking=True)
        )
        corolog.send((batchoutput, i))
        batchoutput = compbatch(batchinput, **comp_kwargs)
    metrics = corolog.send((batchoutput, i + 1))
    return metrics


def do_trainbatch(
    batchinput,
    model,
    optimizer,
    dups: int = 1,
    repeat: int = 1,
):
    """Return mean probabilities, labels, and loss after one optimizer step."""
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


def do_evalbatch(
    batchinput,
    model,
    dups: int = 1,
    repeat: int = 1,
):
    """Return mean probabilities, labels, and mean loss across repeated passes."""
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
