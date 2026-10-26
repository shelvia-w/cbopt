from torch import nn, Tensor
import torch.nn.functional as nnf


class MCDropout(nn.Module):
    """Dropout active during training and evaluation."""

    def __init__(self, p: float = 0.5):
        super().__init__()
        self.p = p

    def forward(self, t: Tensor) -> Tensor:
        return nnf.dropout(t, self.p, training=True)
