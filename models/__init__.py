"""Model constructors and checkpoint serialization helpers."""

from typing import Any, Iterable, Mapping

import torch
from torch import nn

from .backbones.frn import FilterResponseNorm
from .backbones.lenet import LeNet
from .backbones.models32 import get_model
from .uncertainty.mcdropout import MCDropout
from .uncertainty.swag import SWAG


def savemodel(
    to,
    modelname: str,
    modelargs: Iterable[Any],
    modelkwargs: Mapping[str, Any],
    model: nn.Module,
    **kwargs,
) -> None:
    dic = {
        "modelname": modelname,
        "modelargs": tuple(modelargs),
        "modelkwargs": dict(modelkwargs),
        "modelstates": model.state_dict(),
        **kwargs,
    }
    torch.save(dic, to)


def loadmodel(fromfile, device=torch.device("cpu")):
    dic = torch.load(fromfile, map_location=device)
    model = MODELS[dic["modelname"]](*dic["modelargs"], **dic.get("modelkwargs", {})).to(device)
    model.load_state_dict(dic.pop("modelstates"))
    return model, dic


def lenet(outclass: int, input_size: int = 28) -> torch.nn.Module:
    return LeNet(outclass, input_size)


def resnet20(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
        norm_layer=FilterResponseNorm,
    )


def densenet101(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model("densenet101", data_info={"num_classes": outclass, "input_size": input_size})


def lenet_mcdrop(outclass: int, input_size: int = 28, p: float = 0.05) -> torch.nn.Module:
    return LeNet(outclass, input_size, dropout_p=p)


def resnet20_mcdrop(outclass: int, input_size: int = 32, p: float = 0.05) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=lambda: MCDropout(p),
        norm_layer=FilterResponseNorm,
    )


def densenet101_mcdrop(outclass: int, input_size: int = 32, p: float = 0.05) -> torch.nn.Module:
    model = densenet101(outclass, input_size)
    model.linear = nn.Sequential(MCDropout(p), model.linear)
    return model


def lenet_swag(outclass: int, input_size: int = 28, max_rank: int = 20) -> SWAG:
    return SWAG(lenet(outclass, input_size), max_rank)


def resnet20_swag(outclass: int, input_size: int = 32, max_rank: int = 20) -> SWAG:
    return SWAG(resnet20(outclass, input_size), max_rank)


def densenet101_swag(outclass: int, input_size: int = 32, max_rank: int = 20) -> SWAG:
    return SWAG(densenet101(outclass, input_size), max_rank)


STANDARDMODELS = {
    "lenet": lenet,
    "resnet20": resnet20,
    "densenet101": densenet101,
}

MCDROPMODELS = {
    "lenet_mcdrop": lenet_mcdrop,
    "resnet20_mcdrop": resnet20_mcdrop,
    "densenet101_mcdrop": densenet101_mcdrop,
}

SWAGMODELS = {
    "lenet_swag": lenet_swag,
    "resnet20_swag": resnet20_swag,
    "densenet101_swag": densenet101_swag,
}

MODELS = {**STANDARDMODELS, **MCDROPMODELS, **SWAGMODELS}
