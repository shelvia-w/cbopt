"""LeNet-style convolutional classifier for small grayscale images."""

import torch
from torch import nn
from torch.nn import functional as nnf  # used by _AlwaysDrop


class _AlwaysDrop(nn.Module):
    """Dropout active at both train and eval time (MC Dropout)."""

    def __init__(self, p: float) -> None:
        super().__init__()
        self.p = p

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return nnf.dropout(x, self.p, training=True)


class LeNet(nn.Module):
    def __init__(self, outclass: int, input_size: int = 28, in_channels: int = 1, dropout_p: float = 0.0):
        super().__init__()
        feature_size = input_size
        for _ in range(2):
            feature_size = (feature_size - 4) // 2
        if feature_size <= 0:
            raise ValueError(f"input_size={input_size} is too small for LeNet")

        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 6, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(6, 16, kernel_size=5),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        classifier_layers: list[nn.Module] = [
            nn.Flatten(),
            nn.Linear(16 * feature_size * feature_size, 120),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            classifier_layers.append(_AlwaysDrop(dropout_p))
        classifier_layers += [
            nn.Linear(120, 84),
            nn.ReLU(inplace=True),
        ]
        if dropout_p > 0.0:
            classifier_layers.append(_AlwaysDrop(dropout_p))
        classifier_layers.append(nn.Linear(84, outclass))
        self.classifier = nn.Sequential(*classifier_layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(self.features(x))
