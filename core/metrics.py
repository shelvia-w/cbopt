import numpy as np
import torch
from torch import LongTensor, Tensor


def top5corrects(outprobas: Tensor, gt: LongTensor) -> int:
    preds = outprobas.topk(5)[1].t()
    return torch.sum(preds.eq(gt.view(1, -1)), dtype=torch.long).item()


def cumconfidence(probas: Tensor) -> float:
    return torch.sum(torch.max(probas, dim=1)[0]).item()


def cumentropy(probas: Tensor) -> float:
    return torch.sum(
        -probas * torch.log(probas + torch.finfo(probas.dtype).tiny)
    ).item()


def cumnll(probas: Tensor, gts: LongTensor) -> float:
    true_class_probas = torch.gather(probas, -1, gts.unsqueeze(-1)).squeeze(-1)
    nlls = -torch.log(true_class_probas.clamp_min(torch.finfo(probas.dtype).tiny))
    return torch.sum(nlls).item()


def cumbrier(probas: Tensor, onehotgts: Tensor) -> float:
    return torch.sum((probas - onehotgts) ** 2).item()


class AUROC:
    """AUROC of correctness against maximum class probability."""

    def __init__(self):
        self.positive = []
        self.confidence = []

    def collect(self, positives, confidences):
        self.positive += positives
        self.confidence += confidences

    def compute(self) -> float:
        """Return NaN when AUROC is undefined."""
        from sklearn.metrics import roc_auc_score

        try:
            auroc = roc_auc_score(
                np.asarray(self.positive), np.asarray(self.confidence)
            )
        except ValueError:
            auroc = float("nan")
        return auroc


def classification_metrics(probas: np.ndarray, targets: np.ndarray, bins: int = 20):
    """Classification metrics for N-by-C probabilities and N labels."""
    from sklearn.metrics import roc_auc_score

    if probas.ndim != 2 or len(probas) != len(targets) or not len(targets):
        raise ValueError("Expected a nonempty [examples, classes] array and matching labels")
    if bins < 1:
        raise ValueError("bins must be positive")
    if not np.isfinite(probas).all() or (probas < 0).any():
        raise ValueError("Predictions must be finite, nonnegative probabilities")
    if not np.allclose(probas.sum(axis=1), 1.0, atol=1e-5):
        raise ValueError("Class probabilities must sum to one")
    targets = np.asarray(targets, dtype=np.int64)
    if (targets < 0).any() or (targets >= probas.shape[1]).any():
        raise ValueError("Target class is outside the model's output range")
    confidence = probas.max(axis=1)
    correct = probas.argmax(axis=1) == targets
    bin_ids = np.minimum((confidence * bins).astype(int), bins - 1)
    ece = sum(
        abs(correct[bin_ids == i].sum() - confidence[bin_ids == i].sum())
        for i in range(bins)
    ) / len(targets)
    top = np.argsort(probas, axis=1)[:, -min(5, probas.shape[1]):]
    tiny = np.finfo(probas.dtype).tiny
    onehot = np.eye(probas.shape[1], dtype=probas.dtype)[targets]
    auroc = float(roc_auc_score(correct, confidence)) if np.unique(correct).size == 2 else float("nan")
    return {
        "acc": float(correct.mean()),
        "acc@5": float((top == targets[:, None]).any(axis=1).mean()),
        "nll": float(-np.log(probas[np.arange(len(targets)), targets].clip(min=tiny)).mean()),
        "ece": float(ece),
        "brier": float(((probas - onehot) ** 2).sum(axis=1).mean()),
        "auroc": auroc,
        "mean_confidence": float(confidence.mean()),
        "entropy": float(-(probas * np.log(probas.clip(min=tiny))).sum(axis=1).mean()),
    }
