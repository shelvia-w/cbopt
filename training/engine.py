"""Core batch execution helpers."""

import collections
from typing import Any, Callable

import torch
import torch.nn.functional as nnf
from torch import LongTensor, Tensor
from torch.utils.data import DataLoader

from .utils import check_cuda, deterministic_run

SummaryWriter = None


def avgdups(t: Tensor, dups: int) -> Tensor:
    """Average duplicated samples along the leading batch dimension."""
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


def onehot(t: LongTensor, nclasses: int, dtype=torch.long):
    """Convert class indices to one-hot vectors."""
    if torch.numel(t) == 0:
        return torch.empty(0, nclasses, device=t.device)
    t_onehot = torch.zeros(*t.size(), nclasses, device=t.device, dtype=dtype)
    return t_onehot.scatter(t.dim(), t.unsqueeze(-1), 1)


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
    for i, batchinput in enumerate(loader):
        batchinput = apply_batch(
            batchinput, lambda t: t.to(device, non_blocking=True)
        )
        corolog.send((batchoutput, i))
        batchoutput = compbatch(batchinput, **comp_kwargs)
    corolog.send((batchoutput, i + 1))


def do_trainbatch(batchinput, model, optimizer, dups: int = 1, repeat: int = 1):
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


def do_evalbatch(batchinput, model, dups: int = 1, repeat: int = 1):
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


def coro_log_metrics(*args, **kwargs):
    """Compatibility wrapper for the logging module."""
    from .logging import coro_log_metrics as _coro_log_metrics

    return _coro_log_metrics(*args, **kwargs)


def coro_log_timed(*args, **kwargs):
    """Compatibility wrapper for the logging module."""
    from .logging import coro_log_timed as _coro_log_timed

    return _coro_log_timed(*args, **kwargs)
