"""Spectral-normalized neural Gaussian process model components."""

from __future__ import annotations

import math
from dataclasses import dataclass, asdict
from typing import Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F
from torch.nn.utils import parametrize
from torch.nn.utils.parametrizations import spectral_norm


class BoundedSpectralNorm(nn.Module):
    def __init__(
        self,
        bound: float = 1.0,
        n_power_iterations: int = 1,
        eps: float = 1e-12,
        dim: int = 0,
    ):
        super().__init__()
        if bound <= 0:
            raise ValueError(f"bound must be > 0, got {bound}")
        if n_power_iterations < 1:
            raise ValueError(
                f"n_power_iterations must be >= 1, got {n_power_iterations}"
            )
        self.bound = float(bound)
        self.n_power_iterations = int(n_power_iterations)
        self.eps = float(eps)
        self.dim = int(dim)
        self.register_buffer("u", torch.empty(0), persistent=False)
        self.register_buffer("v", torch.empty(0), persistent=False)

    def _reshape_weight_to_matrix(self, weight: Tensor) -> Tensor:
        if self.dim != 0:
            dims = list(range(weight.ndim))
            dims[0], dims[self.dim] = dims[self.dim], dims[0]
            weight = weight.permute(*dims)
        return weight.flatten(1)

    def _maybe_init_uv(self, weight_mat: Tensor) -> None:
        if self.u.numel() != weight_mat.size(0):
            self.u = F.normalize(
                weight_mat.new_empty(weight_mat.size(0)).normal_(0, 1),
                dim=0,
                eps=self.eps,
            )
        if self.v.numel() != weight_mat.size(1):
            self.v = F.normalize(
                weight_mat.new_empty(weight_mat.size(1)).normal_(0, 1),
                dim=0,
                eps=self.eps,
            )

    def _power_iteration(self, weight_mat: Tensor) -> Tuple[Tensor, Tensor]:
        u = self.u
        v = self.v
        for _ in range(self.n_power_iterations):
            v = F.normalize(torch.mv(weight_mat.t(), u), dim=0, eps=self.eps)
            u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps)
        return u, v

    def forward(self, weight: Tensor) -> Tensor:
        weight_mat = self._reshape_weight_to_matrix(weight)
        self._maybe_init_uv(weight_mat)

        u, v = self.u, self.v

        if self.training:
            with torch.no_grad():
                u, v = self._power_iteration(weight_mat.detach())
                u = F.normalize(u, dim=0, eps=self.eps)
                v = F.normalize(v, dim=0, eps=self.eps)
                self.u.copy_(u)
                self.v.copy_(v)
        else:
            if u.numel() == 0 or v.numel() == 0:
                u, v = self._power_iteration(weight_mat.detach())
                u = F.normalize(u, dim=0, eps=self.eps)
                v = F.normalize(v, dim=0, eps=self.eps)

        sigma = torch.dot(u, torch.mv(weight_mat, v)).abs().clamp_min(self.eps)
        scale = torch.clamp(sigma / self.bound, min=1.0)
        return weight / scale


class RandomFourierFeatures(nn.Module):
    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        kernel_scale: float = 1.0,
        random_feature_type: str = "orf",
        scale_features: bool = False,
        normalize_input: bool = False,
    ):
        super().__init__()
        if input_dim <= 0 or output_dim <= 0:
            raise ValueError("input_dim and output_dim must be positive")
        if kernel_scale <= 0:
            raise ValueError("kernel_scale must be positive")

        self.input_dim = int(input_dim)
        self.output_dim = int(output_dim)
        self.kernel_scale = float(kernel_scale)
        self.random_feature_type = random_feature_type.lower()

        self.input_norm = (
            nn.LayerNorm(self.input_dim) if normalize_input else nn.Identity()
        )
        weight = self._sample_weight(
            self.output_dim, self.input_dim, self.random_feature_type
        )
        bias = torch.empty(self.output_dim).uniform_(0.0, 2.0 * math.pi)
        self.register_buffer("weight", weight)
        self.register_buffer("bias", bias)

        self.feature_scale = (
            math.sqrt(2.0 / float(self.output_dim)) if scale_features else 1.0
        )
        self.input_scale = 1.0 / math.sqrt(self.kernel_scale)

    @staticmethod
    def _sample_weight(rows: int, cols: int, kind: str) -> Tensor:
        if kind == "rff":
            return torch.randn(rows, cols)
        if kind == "orf":
            blocks = []
            remaining = rows
            while remaining > 0:
                q, _ = torch.linalg.qr(torch.randn(cols, cols), mode="reduced")
                take = min(remaining, cols)
                blocks.append(q[:take, :])
                remaining -= take
            return torch.cat(blocks, dim=0)
        raise ValueError(f"Unsupported random_feature_type: {kind}")

    def forward(self, x: Tensor) -> Tensor:
        x = self.input_norm(x) * self.input_scale
        proj = F.linear(x, self.weight, self.bias)
        return torch.cos(proj) * self.feature_scale


class LaplaceRandomFeatureCovariance(nn.Module):
    def __init__(
        self,
        feature_dim: int,
        num_classes: int,
        momentum: float = -1.0,
        ridge_penalty: float = 1.0,
    ):
        super().__init__()
        if ridge_penalty <= 0:
            raise ValueError("ridge_penalty must be positive")
        self.feature_dim = int(feature_dim)
        self.num_classes = int(num_classes)
        self.momentum = float(momentum)
        self.ridge_penalty = float(ridge_penalty)

        eye = torch.eye(self.feature_dim)
        init = eye.unsqueeze(0).repeat(self.num_classes, 1, 1) * self.ridge_penalty
        self.register_buffer("initial_precision", init)
        self.register_buffer("precision_matrix", init.clone())
        self._covariance_cache: Optional[Tensor] = None
        self.covariance_dirty = True

    @torch.no_grad()
    def reset_precision_matrix(self) -> None:
        self.precision_matrix.copy_(self.initial_precision)
        self._covariance_cache = None
        self.covariance_dirty = True

    @torch.no_grad()
    def update_precision_matrix(
        self, gp_feature: Tensor, logits: Optional[Tensor] = None
    ) -> None:
        if logits is None or logits.ndim != 2 or logits.size(-1) != self.num_classes:
            raise ValueError(
                f"multiclass_logistic requires logits with shape [B, {self.num_classes}]"
            )
        prob = torch.softmax(logits, dim=-1)
        weights = (prob * (1.0 - prob)).transpose(0, 1)  # [K, B]
        scaled_feature = gp_feature.unsqueeze(0) * torch.sqrt(weights.clamp_min(1e-12)).unsqueeze(-1)
        minibatch_precision = torch.matmul(
            scaled_feature.transpose(1, 2), scaled_feature
        )  # [K, D, D]

        if self.momentum > 0.0:
            minibatch_precision = minibatch_precision / max(int(gp_feature.size(0)), 1)
            self.precision_matrix.mul_(self.momentum).add_(
                minibatch_precision, alpha=1.0 - self.momentum
            )
        else:
            self.precision_matrix.add_(minibatch_precision)

        self._covariance_cache = None
        self.covariance_dirty = True

    def compute_predictive_covariance(self, gp_feature: Tensor) -> Tensor:
        if self.covariance_dirty or self._covariance_cache is None:
            self._covariance_cache = torch.linalg.inv(self.precision_matrix)
            self.covariance_dirty = False

        # var_k(x) = phi(x)^T Sigma_k phi(x)
        cov = self._covariance_cache  # [K, D, D]
        feature = gp_feature.unsqueeze(0).expand(self.num_classes, -1, -1)  # [K, B, D]
        tmp = torch.matmul(feature, cov)  # [K, B, D]
        var = (tmp * feature).sum(dim=-1).transpose(0, 1)  # [B, K]
        return var * self.ridge_penalty


class RandomFeatureGaussianProcess(nn.Module):
    def __init__(
        self,
        input_dim: int,
        num_classes: int,
        num_inducing: int = 1024,
        gp_kernel_scale: float = 1.0,
        gp_output_bias: float = 0.0,
        gp_input_normalization: bool = False,
        gp_random_feature_type: str = "orf",
        gp_cov_discount_factor: float = -1.0,
        gp_cov_ridge_penalty: float = 1.0,
        mean_field_factor: float = math.pi / 8.0,
        output_init_std: float = 0.01,
    ):
        super().__init__()
        self.mean_field_factor = float(mean_field_factor)
        self.num_classes = int(num_classes)
        self.random_feature = RandomFourierFeatures(
            input_dim=input_dim,
            output_dim=num_inducing,
            kernel_scale=gp_kernel_scale,
            random_feature_type=gp_random_feature_type,
            scale_features=False,
            normalize_input=gp_input_normalization,
        )
        self.gp_output = nn.Linear(num_inducing, num_classes, bias=False)
        nn.init.normal_(self.gp_output.weight, mean=0.0, std=output_init_std)
        self.gp_output_bias = nn.Parameter(
            torch.full((num_classes,), float(gp_output_bias))
        )
        self.gp_cov = LaplaceRandomFeatureCovariance(
            feature_dim=num_inducing,
            num_classes=num_classes,
            momentum=gp_cov_discount_factor,
            ridge_penalty=gp_cov_ridge_penalty,
        )

    @torch.no_grad()
    def reset_covariance_matrix(self) -> None:
        self.gp_cov.reset_precision_matrix()

    def forward(
        self,
        inputs: Tensor,
        update_covariance: bool = False,
        return_gp_cov: bool = False,
        mean_field: bool = False,
    ):
        gp_feature = self.random_feature(inputs)
        logits = self.gp_output(gp_feature) + self.gp_output_bias

        if self.training and update_covariance:
            self.gp_cov.update_precision_matrix(gp_feature.detach(), logits.detach())

        if not return_gp_cov and not mean_field:
            return logits

        gp_cov = self.gp_cov.compute_predictive_covariance(gp_feature)
        if mean_field:
            logits = mean_field_logits(logits, gp_cov, self.mean_field_factor)

        if return_gp_cov:
            return logits, gp_cov
        return logits


@torch.no_grad()
def mean_field_logits(
    logits: Tensor,
    covariance_matrix: Optional[Tensor] = None,
    mean_field_factor: float = math.pi / 8.0,
) -> Tensor:
    if covariance_matrix is None:
        variances = torch.ones_like(logits)
    elif covariance_matrix.ndim == 2 and covariance_matrix.shape == logits.shape:
        variances = covariance_matrix
    else:
        variances = torch.diagonal(covariance_matrix, dim1=-2, dim2=-1).unsqueeze(-1)
        variances = variances.expand_as(logits)
    scale = torch.sqrt(1.0 + mean_field_factor * variances)
    return logits / scale


@dataclass
class SNGPConfig:
    use_spec_norm: bool = True
    spec_norm_iteration: int = 1
    spec_norm_bound: float = 0.95
    gp_input_dim: int = -1
    gp_hidden_dim: int = 1024
    gp_scale: float = 1.0
    gp_bias: float = 0.0
    gp_input_normalization: bool = False
    gp_random_feature_type: str = "orf"
    gp_cov_discount_factor: float = -1.0
    gp_cov_ridge_penalty: float = 1.0
    gp_mean_field_factor: float = math.pi / 8.0
    gp_output_init_std: float = 0.01

    def to_kwargs(self):
        return asdict(self)


class SNGPModel(nn.Module):
    def __init__(self, backbone: nn.Module, num_classes: int, config: SNGPConfig):
        super().__init__()
        self.config = config
        self.backbone = backbone
        self.num_classes = int(num_classes)
        self._update_gp_cov = False

        classifier_name, classifier = self._find_last_linear(self.backbone)
        if classifier is None:
            raise ValueError("Could not find final nn.Linear classifier to replace.")
        self.classifier_name = classifier_name
        self.feature_dim = int(classifier.in_features)
        self._replace_module(self.backbone, classifier_name, nn.Identity())

        if self.config.use_spec_norm:
            self._apply_spectral_norm(self.backbone, skip_module_name=self.classifier_name)

        if self.config.gp_input_dim > 0:
            self.random_projection = nn.Linear(
                self.feature_dim, self.config.gp_input_dim, bias=False
            )
            nn.init.normal_(self.random_projection.weight, mean=0.0, std=0.05)
            self.random_projection.weight.requires_grad_(False)
            gp_input_dim = self.config.gp_input_dim
        else:
            self.random_projection = nn.Identity()
            gp_input_dim = self.feature_dim

        self.gp_layer = RandomFeatureGaussianProcess(
            input_dim=gp_input_dim,
            num_classes=self.num_classes,
            num_inducing=self.config.gp_hidden_dim,
            gp_kernel_scale=self.config.gp_scale,
            gp_output_bias=self.config.gp_bias,
            gp_input_normalization=self.config.gp_input_normalization,
            gp_random_feature_type=self.config.gp_random_feature_type,
            gp_cov_discount_factor=self.config.gp_cov_discount_factor,
            gp_cov_ridge_penalty=self.config.gp_cov_ridge_penalty,
            mean_field_factor=self.config.gp_mean_field_factor,
            output_init_std=self.config.gp_output_init_std,
        )

    def set_update_covariance(self, value: bool) -> None:
        self._update_gp_cov = bool(value)

    @torch.no_grad()
    def reset_covariance_matrix(self) -> None:
        self.gp_layer.reset_covariance_matrix()

    def forward(
        self,
        x: Tensor,
        mean_field: Optional[bool] = None,
        return_gp_cov: bool = False,
    ):
        features = self.backbone(x)
        if isinstance(features, (tuple, list)):
            features = features[0]
        if features.ndim > 2:
            features = torch.flatten(features, 1)
        features = self.random_projection(features)
        use_mean_field = (not self.training) if mean_field is None else bool(mean_field)
        return self.gp_layer(
            features,
            update_covariance=self._update_gp_cov,
            return_gp_cov=return_gp_cov,
            mean_field=use_mean_field,
        )

    @staticmethod
    def _find_last_linear(module: nn.Module) -> Tuple[Optional[str], Optional[nn.Linear]]:
        last_name = None
        last_module = None
        for name, child in module.named_modules():
            if isinstance(child, nn.Linear):
                last_name = name
                last_module = child
        return last_name, last_module

    @staticmethod
    def _replace_module(root: nn.Module, qualified_name: str, new_module: nn.Module) -> None:
        parts = qualified_name.split(".")
        parent = root
        for part in parts[:-1]:
            parent = getattr(parent, part)
        setattr(parent, parts[-1], new_module)

    def _apply_spectral_norm(self, module: nn.Module, skip_module_name: str) -> None:
        for name, child in module.named_modules():
            if name == "" or name == skip_module_name:
                continue
            if isinstance(child, nn.Linear) and hasattr(child, "weight"):
                if not parametrize.is_parametrized(child, "weight"):
                    spectral_norm(
                        child,
                        name="weight",
                        n_power_iterations=self.config.spec_norm_iteration,
                        eps=1e-12,
                    )


def build_sngp_from_standard_model(
    base_model: nn.Module,
    num_classes: int,
    config: Optional[SNGPConfig] = None,
) -> SNGPModel:
    if config is None:
        config = SNGPConfig()
    return SNGPModel(base_model, num_classes=num_classes, config=config)
