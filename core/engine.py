"""Core batch execution helpers."""

import collections
from typing import Any, Callable

import torch
import torch.nn.functional as nnf
from torch import Tensor
from torch.utils.data import DataLoader

SummaryWriter = None


# =============================
# Batch tensor utilities
# =============================

def avgdups(t: Tensor, dups: int) -> Tensor:
    """Average predictions from duplicated copies of the same samples."""
    return torch.mean(t.view(dups, -1, *t.size()[1:]), dim=0)


def apply_batch(batch, fn: Callable[[Tensor], Any]):
    """Recursively apply a function to tensors inside a batch."""
    if isinstance(batch, Tensor):
        return fn(batch)
    if isinstance(batch, collections.abc.Mapping):
        return {k: apply_batch(sample, fn) for k, sample in batch.items()}
    if isinstance(batch, collections.abc.Sequence):
        return [apply_batch(sample, fn) for sample in batch]
    return batch


# =============================
# Epoch execution
# =============================

def do_epoch(
    loader: DataLoader,
    compbatch,
    corolog,
    device=torch.device("cpu"),
    **comp_kwargs,
):
    """Run one full pass over a dataloader and stream outputs to a logger coroutine."""
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


# =============================
# Batch execution
# =============================

def do_trainbatch(
    batchinput,
    model,
    optimizer,
    dups: int = 1,
    repeat: int = 1,
):
    """Run one training batch and return averaged probabilities, labels, and loss."""
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
    """Run one evaluation batch and return averaged probabilities, labels, and loss."""
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
