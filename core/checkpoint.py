"""Checkpoint save/load helpers."""

import inspect
import warnings
from importlib import import_module
from typing import Any, Iterable, Mapping

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR

import models as _models


# =============================
# Class/function resolution
# =============================

def _resolve_qualified_attr(module_name: str, qualname: str):
    """Resolve a qualified attribute from an importable module."""
    obj = import_module(module_name)
    for part in qualname.split("."):
        obj = getattr(obj, part)
    return obj


def _filter_constructor_kwargs(cls, kwargs: Mapping[str, Any]) -> dict[str, Any]:
    """Keep only kwargs accepted by the class constructor unless it already accepts `**kwargs`."""
    signature = inspect.signature(cls.__init__)
    if any(param.kind == inspect.Parameter.VAR_KEYWORD for param in signature.parameters.values()):
        return dict(kwargs)

    accepted = {
        name
        for name, param in signature.parameters.items()
        if name not in {"self", "params"}
        and param.kind in {inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.KEYWORD_ONLY}
    }
    return {key: value for key, value in kwargs.items() if key in accepted}


def _resolve_optimizer_class(checkpoint: Mapping[str, Any]):
    """Resolve the optimizer class from checkpoint metadata, with name-based fallback for older checkpoints."""
    module_name = checkpoint.get("optimmodule")
    qualname = checkpoint.get("optimqualname")
    if module_name and qualname:
        return _resolve_qualified_attr(module_name, qualname)

    optimname = checkpoint["optimname"]
    if hasattr(torch.optim, optimname):
        return getattr(torch.optim, optimname)

    raise NotImplementedError(
        f"Unknown optimizer: {optimname}. Re-save the checkpoint with optimizer module metadata."
    )


# =============================
# Checkpoint saving & loading
# =============================

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
                "optimmodule": type(optimizer).__module__,
                "optimqualname": type(optimizer).__qualname__,
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
    optimargs = dict(dic.pop("optimargs"))
    optimizer_cls = _resolve_optimizer_class(dic)
    optimizer = optimizer_cls(model.parameters(), **_filter_constructor_kwargs(optimizer_cls, optimargs))
    optimizer.load_state_dict(dic.pop("optimstates"))
    schedulername = dic.get("schedulername")
    if schedulername in (None, "NoneType"):
        dic.pop("schedulerstates", None)
        return 0, model, optimizer, None, dic
    if schedulername == "LinearLR":
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
