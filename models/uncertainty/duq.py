"""Deep Uncertainty Quantification model components."""

import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, base_model: nn.Module):
        super().__init__()
        self.base_model = base_model

        if hasattr(base_model, "fc") and isinstance(base_model.fc, nn.Linear):
            self.feature_dim = base_model.fc.in_features
            base_model.fc = nn.Identity()
        elif hasattr(base_model, "linear") and isinstance(base_model.linear, nn.Linear):
            self.feature_dim = base_model.linear.in_features
            base_model.linear = nn.Identity()
        elif hasattr(base_model, "linear1") and isinstance(base_model.linear1, nn.Linear):
            self.feature_dim = base_model.linear1.in_features
            base_model.linear1 = nn.Identity()
        elif hasattr(base_model, "classifier") and isinstance(base_model.classifier, nn.Linear):
            self.feature_dim = base_model.classifier.in_features
            base_model.classifier = nn.Identity()
        elif hasattr(base_model, "classifier") and isinstance(base_model.classifier, nn.Sequential):
            last = None
            for m in reversed(base_model.classifier):
                if isinstance(m, nn.Linear):
                    last = m
                    break
            if last is None:
                raise ValueError("Could not infer feature dimension from classifier.")
            self.feature_dim = last.in_features
            base_model.classifier = nn.Identity()
        else:
            raise ValueError(
                "Unsupported backbone for DUQ. Expected final classifier as .fc, .linear, or .classifier."
            )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        z = self.base_model(x)
        if z.ndim > 2:
            z = torch.flatten(z, 1)
        return z


class DUQ(nn.Module):
    def __init__(
        self,
        feature_extractor: nn.Module,
        num_classes: int,
        centroid_dim: int,
        model_output_size: int,
        length_scale: float,
        beta: float,
    ):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.num_classes = num_classes
        self.centroid_dim = centroid_dim
        self.model_output_size = model_output_size
        self.length_scale = float(length_scale)
        self.beta = float(beta)

        self.W = nn.Parameter(
            torch.normal(
                mean=0.0,
                std=0.05,
                size=(num_classes, centroid_dim, model_output_size),
            )
        )

        self.register_buffer("N", torch.ones(num_classes) * 12.0)
        self.register_buffer("m", torch.zeros(num_classes, centroid_dim))

    def embed(self, x: torch.Tensor) -> torch.Tensor:
        feats = self.feature_extractor(x)
        z = torch.einsum("cdf,bf->bcd", self.W, feats)
        return z

    def centroids(self) -> torch.Tensor:
        return self.m / self.N.unsqueeze(1)

    def bilinear(self, x: torch.Tensor) -> torch.Tensor:
        z = self.embed(x)
        centroids = self.centroids()
        diff = z - centroids.unsqueeze(0)
        dist = (diff ** 2).mean(dim=2)
        return torch.exp(-dist / (2 * self.length_scale ** 2))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bilinear(x)

    @torch.no_grad()
    def update_embeddings(self, x: torch.Tensor, y_onehot: torch.Tensor) -> None:
        z = self.embed(x)
        batch_N = y_onehot.sum(dim=0)
        batch_m = torch.einsum("bc,bcd->cd", y_onehot, z)

        self.N.mul_(self.beta).add_((1.0 - self.beta) * batch_N)
        self.m.mul_(self.beta).add_((1.0 - self.beta) * batch_m)


class DUQModel(nn.Module):
    def __init__(
        self,
        base_model: nn.Module,
        num_classes: int,
        centroid_dim: int,
        length_scale: float,
        beta: float,
    ):
        super().__init__()
        feature_extractor = FeatureExtractor(base_model)
        self.duq = DUQ(
            feature_extractor=feature_extractor,
            num_classes=num_classes,
            centroid_dim=centroid_dim,
            model_output_size=feature_extractor.feature_dim,
            length_scale=length_scale,
            beta=beta,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.duq(x)

    @torch.no_grad()
    def update_embeddings(self, x: torch.Tensor, y_onehot: torch.Tensor) -> None:
        self.duq.update_embeddings(x, y_onehot)

    @property
    def centroid_dim(self) -> int:
        return self.duq.centroid_dim


def calc_gradients_input(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    gradients = torch.autograd.grad(
        outputs=y_pred,
        inputs=x,
        grad_outputs=torch.ones_like(y_pred),
        create_graph=True,
    )[0]
    return gradients.flatten(start_dim=1)


def calc_gradient_penalty(x: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
    gradients = calc_gradients_input(x, y_pred)
    grad_norm = gradients.norm(2, dim=1)
    return ((grad_norm - 1.0) ** 2).mean()
