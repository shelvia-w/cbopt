"""Model constructors, serialization helpers, and public model registries."""

from typing import Any, Iterable, Mapping
from functools import partial
import numpy as np
import torch
from torch import nn

from .backbones.models32 import get_model
from .backbones.grudense import GRUDense
from .backbones.lenet import LeNet
from .backbones.frn import FilterResponseNorm, FilterResponseNormLipschitz
from .backbones.resnet224 import resnet50
from .uncertainty.swag import SWAG
from .uncertainty.mcdropout import MCDropout
from .uncertainty.sngp import SNGPConfig, build_sngp_from_standard_model


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
        "modelkwargs": {k: modelkwargs[k] for k in modelkwargs},
        "modelstates": model.state_dict(),
        **kwargs,
    }
    torch.save(dic, to)


_LENET_IDENTITY_ERA_REMAP = {
    # Checkpoints saved when nn.Identity() was always inserted into the classifier
    # had linear-layer keys at .4 and .7; current layout (no-op dropout omitted) uses .3 and .5.
    "classifier.4.": "classifier.3.",
    "classifier.7.": "classifier.5.",
}


def _coerce_state_dict(state_dict: dict, model: nn.Module) -> dict:
    model_keys = set(model.state_dict().keys())
    if set(state_dict.keys()) == model_keys:
        return state_dict
    remapped = {}
    for k, v in state_dict.items():
        new_k = k
        for old, new in _LENET_IDENTITY_ERA_REMAP.items():
            if k.startswith(old):
                new_k = new + k[len(old):]
                break
        remapped[new_k] = v
    if set(remapped.keys()) == model_keys:
        return remapped
    return state_dict


def loadmodel(fromfile, device=torch.device("cpu")):
    dic = torch.load(fromfile, map_location=device)
    model = globals()[dic["modelname"]](*dic["modelargs"], **dic.get("modelkwargs", {})).to(device)
    model.load_state_dict(_coerce_state_dict(dic.pop("modelstates"), model))
    return model, dic


def resnet20(outclass: int, input_size: int = 32, norm_layer=FilterResponseNorm) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
        norm_layer=norm_layer,
    )


def lenet(outclass: int, input_size: int = 28) -> torch.nn.Module:
    return LeNet(outclass, input_size)


def resnet20_sngp(
    outclass: int,
    input_size: int = 32,
    use_spec_norm: bool = True,
    spec_norm_iteration: int = 1,
    spec_norm_bound: float = 0.95,
    gp_input_dim: int = -1,
    gp_hidden_dim: int = 1024,
    gp_scale: float = 1.0,
    gp_bias: float = 0.0,
    gp_input_normalization: bool = False,
    gp_random_feature_type: str = "orf",
    gp_cov_discount_factor: float = -1.0,
    gp_cov_ridge_penalty: float = 1.0,
    gp_mean_field_factor: float = np.pi / 8.0,
    gp_output_init_std: float = 0.01,
) -> torch.nn.Module:
    base = resnet20(
        outclass,
        input_size,
        norm_layer=partial(FilterResponseNormLipschitz, gamma_bound=spec_norm_bound),
    )
    cfg = SNGPConfig(
        use_spec_norm=use_spec_norm,
        spec_norm_iteration=spec_norm_iteration,
        spec_norm_bound=spec_norm_bound,
        gp_input_dim=gp_input_dim,
        gp_hidden_dim=gp_hidden_dim,
        gp_scale=gp_scale,
        gp_bias=gp_bias,
        gp_input_normalization=gp_input_normalization,
        gp_random_feature_type=gp_random_feature_type,
        gp_cov_discount_factor=gp_cov_discount_factor,
        gp_cov_ridge_penalty=gp_cov_ridge_penalty,
        gp_mean_field_factor=gp_mean_field_factor,
        gp_output_init_std=gp_output_init_std,
    )
    return build_sngp_from_standard_model(base, outclass, cfg)


def resnet20_mcdrop(outclass: int, input_size: int = 32, p: float = 0.05) -> torch.nn.Module:
    return get_model(
        "resnet20_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=lambda: MCDropout(p),
    )


def lenet_mcdrop(outclass: int, input_size: int = 28, p: float = 0.05) -> torch.nn.Module:
    return LeNet(outclass, input_size, dropout_p=p)


def softplus_inv(x: float) -> float:
    return x + np.log(-np.expm1(-x))


def resnet20_bbb(
    outclass: int,
    input_size: int = 32,
    prior_precision: float = 1.0,
    std_init: float = 0.05,
    bnn_type: str = "Reparameterization",
) -> torch.nn.Module:
    from bayesian_torch.models.dnn_to_bnn import dnn_to_bnn
    bnn_options = {
        "prior_mu": 0.0,
        "prior_sigma": 1.0 / np.sqrt(prior_precision),
        "posterior_mu_init": 0.0,
        "posterior_rho_init": softplus_inv(std_init),
        "type": bnn_type,
        "moped_enable": False,
    }
    model = resnet20(outclass, input_size)
    dnn_to_bnn(model, bnn_options)
    return model


def resnet20_swag(outclass: int, input_size: int = 32, max_rank: int = 20) -> SWAG:
    return SWAG(resnet20(outclass, input_size), max_rank)


def lenet_swag(outclass: int, input_size: int = 28, max_rank: int = 20) -> SWAG:
    return SWAG(lenet(outclass, input_size), max_rank)


def preresnet110(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model(
        "preresnet110_frn",
        data_info={"num_classes": outclass, "input_size": input_size},
        activation=torch.nn.Identity,
    )


def resnet18wide(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model("resnet18", data_info={"num_classes": outclass, "input_size": input_size})


def densenet121(outclass: int, input_size: int = 32) -> torch.nn.Module:
    return get_model("densenet121", data_info={"num_classes": outclass, "input_size": input_size})


def densenet121_mcdrop(outclass: int, input_size: int = 32, p: float = 0.05) -> torch.nn.Module:
    model = densenet121(outclass, input_size)
    model.linear = nn.Sequential(MCDropout(p), model.linear)
    return model


def gru_dense(vocab_size: int, num_classes: int, padding_idx: int) -> GRUDense:
    return GRUDense(vocab_size, num_classes, padding_idx)


def resnet50_imagenet(outclass: int, input_size: int = 224) -> torch.nn.Module:
    return resnet50(activation=nn.Identity, norm_layer=FilterResponseNorm, num_classes=outclass)


STANDARDMODELS = {
    "lenet": lenet,
    "resnet20": resnet20,
    "resnet18wide": resnet18wide,
    "preresnet110": preresnet110,
    "densenet121": densenet121,
    "resnet50_imagenet": resnet50_imagenet,
}
MCDROPMODELS = {
    "resnet20_mcdrop": resnet20_mcdrop,
    "lenet_mcdrop": lenet_mcdrop,
    "densenet121_mcdrop": densenet121_mcdrop,
}
BBBMODELS = {"resnet20_bbb": resnet20_bbb}
SWAGMODELS = {"resnet20_swag": resnet20_swag, "lenet_swag": lenet_swag}
SNGPMODELS = {"resnet20_sngp": resnet20_sngp}
MODELS = {**STANDARDMODELS, **MCDROPMODELS, **BBBMODELS, **SWAGMODELS, **SNGPMODELS}
