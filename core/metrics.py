"""Metric helpers."""

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from torch import LongTensor, Tensor


def top5corrects(outprobas: Tensor, gt: LongTensor) -> int:
    """Count samples whose label appears in the top-5 predictions."""
    preds = outprobas.topk(5)[1].t()
    return torch.sum(preds.eq(gt.view(1, -1)), dtype=torch.long).item()


def cumconfidence(probas: Tensor) -> float:
    """Return the summed max confidence over a batch."""
    return torch.sum(torch.max(probas, dim=1)[0]).item()


def cumentropy(probas: Tensor) -> float:
    """Return the summed predictive entropy over a batch."""
    return torch.sum(
        -probas * torch.log(probas + torch.finfo(probas.dtype).tiny)
    ).item()


def cumnll(probas: Tensor, gts: LongTensor) -> float:
    """Return the summed negative log-likelihood over a batch."""
    true_class_probas = torch.gather(probas, -1, gts.unsqueeze(-1)).squeeze(-1)
    nlls = -torch.log(true_class_probas.clamp_min(torch.finfo(probas.dtype).tiny))
    return torch.sum(nlls).item()


def cumbrier(probas: Tensor, onehotgts: Tensor) -> float:
    """Return the summed Brier score over a batch."""
    return torch.sum((probas - onehotgts) ** 2).item()


class AUROC:
    """Incrementally collect confidence labels and compute AUROC."""

    def __init__(self):
        self.positive = []
        self.confidence = []

    def collect(self, positives, confidences):
        """Append a batch of correctness labels and confidences."""
        self.positive += positives
        self.confidence += confidences

    def compute(self) -> float:
        """Compute AUROC, returning `nan` when undefined."""
        try:
            auroc = roc_auc_score(
                np.asarray(self.positive), np.asarray(self.confidence)
            )
        except ValueError:
            auroc = float("nan")
        return auroc
