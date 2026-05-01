"""Out-of-domain dataset loaders and OOD metric helpers."""

from typing import Tuple, Dict
from os.path import join as pjoin
import statistics
from functools import cached_property, lru_cache
import numpy as np
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms

from .data_utils import dup_collate_fn


class SVHNInfo:
    outclass = 10
    split = ("train", "test", "extra")
    count = {"train": 73257, "test": 26032, "extra": 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    split: str = "test",
    dups: int = 1,
):
    assert split in SVHNInfo.split
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    dataset = datasets.SVHN(
        root=svhn_dir,
        split=split,
        download=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
    )

    loader = (
        DataLoader(
            dataset,
            batch_size=batch,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            dataset,
            batch_size=batch,
            num_workers=workers,
            pin_memory=pin_memory,
        )
    )
    return loader


class Flowers102Info:
    outclass = 102
    split = ("train", "val", "test")
    count = {"train": 1020, "val": 1020, "test": 6149}
    mean = (0.50390434, 0.4516826, 0.494936)
    std = (0.23261614, 0.20974728, 0.2668646)


def get_flowers102_loader(
    data_dir: str,
    workers: int,
    pin_memory: bool,
    batch: int,
    split: str = "test",
    dups: int = 1,
):
    assert split in Flowers102Info.split
    flowers102_dir = pjoin(data_dir, "flowers102")
    normalize = transforms.Normalize(Flowers102Info.mean, Flowers102Info.std)

    dataset = datasets.Flowers102(
        root=flowers102_dir,
        split=split,
        download=True,
        transform=transforms.Compose([
            transforms.Resize(32),
            transforms.CenterCrop(32),
            transforms.ToTensor(),
            normalize,
        ]),
    )

    loader = (
        DataLoader(
            dataset,
            batch_size=batch,
            num_workers=workers,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            dataset,
            batch_size=batch,
            num_workers=workers,
            pin_memory=pin_memory,
        )
    )
    return loader


OOD_LOADERS = {
    "svhn": get_svhn_loader,
    "flowers102": get_flowers102_loader,
}


def auroc(indomain_confidence: np.ndarray, ood_confidence: np.ndarray) -> float:
    confidence = np.concatenate((indomain_confidence, ood_confidence))
    is_indomain = np.concatenate(
        (np.ones_like(indomain_confidence), np.zeros_like(ood_confidence))
    )
    return roc_auc_score(is_indomain, confidence)


class OODMetrics:
    metric_names = ("auroc", "aupr-in", "aupr-out", "fpr95", "dterr")

    def __init__(
        self,
        indomain_confidence: np.ndarray,
        ood_confidence: np.ndarray,
        eps: float = 0.0005,
    ):
        self.indomain_confidence = indomain_confidence
        self.ood_confidence = ood_confidence
        self.eps = eps

    @cached_property
    def _confidence(self) -> np.ndarray:
        return np.concatenate((self.indomain_confidence, self.ood_confidence))

    @cached_property
    def _is_indomain(self) -> np.ndarray:
        return np.concatenate(
            (np.ones_like(self.indomain_confidence), np.zeros_like(self.ood_confidence))
        )

    @cached_property
    def _is_ood(self) -> np.ndarray:
        return np.concatenate(
            (np.zeros_like(self.indomain_confidence), np.ones_like(self.ood_confidence))
        )

    @cached_property
    def _fpr_tpr(self) -> Tuple[np.ndarray, np.ndarray]:
        return roc_curve(self._is_indomain, self._confidence)[:2]

    @cached_property
    def auroc(self) -> float:
        return roc_auc_score(self._is_indomain, self._confidence)

    @cached_property
    def aupr_in(self) -> float:
        return average_precision_score(self._is_indomain, self._confidence)

    @cached_property
    def aupr_out(self) -> float:
        return average_precision_score(self._is_ood, -self._confidence)

    @cached_property
    def fpr_at_tpr95(self) -> float:
        fpr, tpr = self._fpr_tpr
        eps = self.eps
        idx_tpr95 = (tpr <= (0.95 + eps)) >= (0.95 - eps)
        if not np.any(idx_tpr95):
            raise ValueError(
                f"no tpr between [{0.95 - eps}, {0.95 + eps}], increase eps!"
            )
        return fpr[idx_tpr95].mean()

    @cached_property
    def detection_error(self) -> float:
        fpr, tpr = self._fpr_tpr
        detection_error = (fpr - tpr + 1.0) / 2.0
        return detection_error.min()

    @lru_cache(maxsize=None)
    def get_all(self) -> Dict[str, float]:
        return {
            "auroc": self.auroc,
            "aupr-in": self.aupr_in,
            "aupr-out": self.aupr_out,
            "fpr95": self.fpr_at_tpr95,
            "dterr": self.detection_error,
        }
