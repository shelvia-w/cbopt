"""Dataset loader factories for the reproducible experiments."""

from os.path import join as pjoin
from typing import Tuple
import random

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets
import torchvision.transforms as transforms


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def dup_collate_fn(dups: int):
    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,) * (t.ndim - 1)), torch.as_tensor(gts)

    return collate_fn


GENERATOR = torch.Generator()
GENERATOR.manual_seed(0)


class FashionMNISTInfo:
    outclass = 10
    imgshape = (1, 28, 28)
    counts = {"train": 60000, "test": 10000}
    mean = (0.2860,)
    std = (0.3530,)


class CIFAR10Info:
    outclass = 10
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


class CIFAR100Info:
    outclass = 100
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def _loader(dataset, batch, workers, pin_memory, shuffle=False, dups=1):
    kwargs = {
        "batch_size": batch,
        "num_workers": workers,
        "worker_init_fn": seed_worker,
        "generator": GENERATOR,
        "shuffle": shuffle,
        "pin_memory": pin_memory,
    }
    if dups > 1:
        kwargs["collate_fn"] = dup_collate_fn(dups)
    return DataLoader(dataset, **kwargs)


def _split_train_val(train_data, val_data, split, tbatch, vbatch, workers, pin_memory):
    nb_train = int(len(train_data) * split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))
    train_loader = _loader(
        Subset(train_data, train_indices), tbatch, workers, pin_memory, shuffle=True
    )
    val_loader = _loader(
        Subset(val_data, val_indices), vbatch, workers, pin_memory, shuffle=False
    )
    return train_loader, val_loader


def get_fmnist_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
) -> Tuple[DataLoader, DataLoader]:
    root = pjoin(data_dir, "fmnist")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(FashionMNISTInfo.mean, FashionMNISTInfo.std)]
    )
    train_data = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    val_data = datasets.FashionMNIST(root=root, train=True, transform=transform, download=True)
    return _split_train_val(train_data, val_data, train_val_split, tbatch, vbatch, workers, pin_memory)


def get_fmnist_test_loader(data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1):
    root = pjoin(data_dir, "fmnist")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(FashionMNISTInfo.mean, FashionMNISTInfo.std)]
    )
    dataset = datasets.FashionMNIST(root=root, train=False, transform=transform, download=True)
    return _loader(dataset, batch, workers, pin_memory, dups=dups)


def get_cifar10_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
) -> Tuple[DataLoader, DataLoader]:
    root = pjoin(data_dir, "cifar10")
    normalize = transforms.Normalize(CIFAR10Info.mean, CIFAR10Info.std)
    train_data = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
        ),
        download=True,
    )
    val_data = datasets.CIFAR10(
        root=root,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return _split_train_val(train_data, val_data, train_val_split, tbatch, vbatch, workers, pin_memory)


def get_cifar10_test_loader(data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1):
    root = pjoin(data_dir, "cifar10")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR10Info.mean, CIFAR10Info.std)]
    )
    dataset = datasets.CIFAR10(root=root, train=False, transform=transform, download=True)
    return _loader(dataset, batch, workers, pin_memory, dups=dups)


def get_cifar100_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
) -> Tuple[DataLoader, DataLoader]:
    root = pjoin(data_dir, "cifar100")
    normalize = transforms.Normalize(CIFAR100Info.mean, CIFAR100Info.std)
    train_data = datasets.CIFAR100(
        root=root,
        train=True,
        transform=transforms.Compose(
            [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, 4), transforms.ToTensor(), normalize]
        ),
        download=True,
    )
    val_data = datasets.CIFAR100(
        root=root,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    return _split_train_val(train_data, val_data, train_val_split, tbatch, vbatch, workers, pin_memory)


def get_cifar100_test_loader(data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1):
    root = pjoin(data_dir, "cifar100")
    transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(CIFAR100Info.mean, CIFAR100Info.std)]
    )
    dataset = datasets.CIFAR100(root=root, train=False, transform=transform, download=True)
    return _loader(dataset, batch, workers, pin_memory, dups=dups)


TRAINDATALOADERS = {
    "fmnist": get_fmnist_train_loaders,
    "cifar10": get_cifar10_train_loaders,
    "cifar100": get_cifar100_train_loaders,
}

TESTDATALOADER = {
    "fmnist": get_fmnist_test_loader,
    "cifar10": get_cifar10_test_loader,
    "cifar100": get_cifar100_test_loader,
}

DATASET_INFO = {
    "fmnist": FashionMNISTInfo,
    "cifar10": CIFAR10Info,
    "cifar100": CIFAR100Info,
}

INSIZE = {name: info.imgshape[-1] for name, info in DATASET_INFO.items()}
OUTCLASS = {name: info.outclass for name, info in DATASET_INFO.items()}
