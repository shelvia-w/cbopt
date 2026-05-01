"""Dataset loader factories and dataset metadata for training and evaluation."""

from typing import Tuple
from os.path import join as pjoin
import random
import numpy as np
import torch
from torch.utils.data import DataLoader, Subset
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from .tinyimagenet import TinyImageNet

def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

# autobatch collate function for dataloader to duplicate batch
def dup_collate_fn(dups: int):
    # assume a list of img, result pairs
    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,) * (t.ndim - 1)), torch.as_tensor(gts)

    return collate_fn


class CIFAR10Info:
    outclass = 10
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def get_cifar10_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    cifar10_dir = pjoin(data_dir, "cifar10")
    normalize = transforms.Normalize(
        mean=CIFAR10Info.mean, std=CIFAR10Info.std
    )

    train_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    val_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )

    return train_loader, val_loader


def get_cifar10_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    cifar10_dir = pjoin(data_dir, "cifar10")
    normalize = transforms.Normalize(
        mean=CIFAR10Info.mean, std=CIFAR10Info.std
    )

    test_data = datasets.CIFAR10(
        root=cifar10_dir,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            enerator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )
    return test_loader


class CIFAR100Info:
    outclass = 100
    imgshape = (3, 32, 32)
    counts = {"train": 50000, "test": 10000}
    mean = (0.485, 0.456, 0.406)
    std = (0.229, 0.224, 0.225)


def get_cifar100_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    cifar100_dir = pjoin(data_dir, "cifar100")
    normalize = transforms.Normalize(
        mean=CIFAR100Info.mean, std=CIFAR100Info.std
    )

    train_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    val_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )

    return train_loader, val_loader


def get_cifar100_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    cifar100_dir = pjoin(data_dir, "cifar100")
    normalize = transforms.Normalize(
        mean=CIFAR100Info.mean, std=CIFAR100Info.std
    )

    test_data = datasets.CIFAR100(
        root=cifar100_dir,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )

    return test_loader


class MNISTInfo:
    outclass = 10
    imgshape = (1, 28, 28)
    counts = {"train": 60000, "test": 10000}
    mean = (0.1307,)
    std = (0.3081,)


def get_mnist_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    mnist_dir = pjoin(data_dir, "mnist")
    normalize = transforms.Normalize(MNISTInfo.mean, MNISTInfo.std)

    train_data = datasets.MNIST(
        root=mnist_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    val_data = datasets.MNIST(
        root=mnist_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_mnist_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    mnist_dir = pjoin(data_dir, "mnist")
    normalize = transforms.Normalize(MNISTInfo.mean, MNISTInfo.std)

    test_data = datasets.MNIST(
        root=mnist_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )
    return test_loader


class FashionMNISTInfo:
    outclass = 10
    imgshape = (1, 28, 28)
    counts = {"train": 60000, "test": 10000}
    mean = (0.2860,)
    std = (0.3530,)


def get_fmnist_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    fmnist_dir = pjoin(data_dir, "fmnist")
    normalize = transforms.Normalize(FashionMNISTInfo.mean, FashionMNISTInfo.std)

    train_data = datasets.FashionMNIST(
        root=fmnist_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    val_data = datasets.FashionMNIST(
        root=fmnist_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )
    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )
    return train_loader, val_loader


def get_fmnist_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    fmnist_dir = pjoin(data_dir, "fmnist")
    normalize = transforms.Normalize(FashionMNISTInfo.mean, FashionMNISTInfo.std)

    test_data = datasets.FashionMNIST(
        root=fmnist_dir,
        train=False,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )
    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )
    return test_loader


class SVHNInfo:
    outclass = 10
    imgshape = (3, 32, 32)
    split = ("train", "test", "extra")
    counts = {"train": 73257, "test": 26032, "extra": 531131}
    mean = (0.4376821, 0.4437697, 0.47280442)
    std = (0.19803012, 0.20101562, 0.19703614)


def get_svhn_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    train_data = datasets.SVHN(
        root=svhn_dir,
        split="train",
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, 4),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    val_data = datasets.SVHN(
        root=svhn_dir,
        split="train",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )

    return train_loader, val_loader


def get_svhn_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    svhn_dir = pjoin(data_dir, "svhn")
    normalize = transforms.Normalize(SVHNInfo.mean, SVHNInfo.std)

    test_data = datasets.SVHN(
        root=svhn_dir,
        split="test",
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )
    return test_loader


class TinyImageNetInfo:
    outclass = 200
    imgshape = (3, 64, 64)
    counts = {"train": 100000, "test": 10000}
    mean = (0.48024865984916687, 0.4480723738670349, 0.3975464701652527)
    std = (0.23022247850894928, 0.22650277614593506, 0.2261698693037033)


def get_tinyimagenet_train_loaders(
    data_dir: str,
    train_val_split: float,
    workers: int,
    pin_memory: bool,
    tbatch: int,
    vbatch: int,
    tdups: int = 1,
    vdups: int = 1,
) -> Tuple[DataLoader, DataLoader]:
    tinyimagenet_dir = pjoin(data_dir, "tinyimagenet")
    normalize = transforms.Normalize(
        mean=TinyImageNetInfo.mean, std=TinyImageNetInfo.std
    )

    train_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=True,
        transform=transforms.Compose(
            [
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(64, 8),
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    val_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=True,
        transform=transforms.Compose([transforms.ToTensor(), normalize]),
        download=True,
    )

    nb_train = int(len(train_data) * train_val_split)
    train_indices = list(range(nb_train))
    val_indices = list(range(nb_train, len(train_data)))

    train_loader = (
        DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
            collate_fn=dup_collate_fn(tdups),
        )
        if tdups > 1
        else DataLoader(
            Subset(train_data, train_indices),
            batch_size=tbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=True,
        )
    )

    val_loader = (
        DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
            collate_fn=dup_collate_fn(vdups),
        )
        if vdups > 1
        else DataLoader(
            Subset(val_data, val_indices),
            batch_size=vbatch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=pin_memory,
            shuffle=False,
        )
    )

    return train_loader, val_loader


def get_tinyimagenet_test_loader(
    data_dir: str, workers: int, pin_memory: bool, batch: int, dups: int = 1
) -> DataLoader:
    tinyimagenet_dir = pjoin(data_dir, "tinyimagenet")
    normalize = transforms.Normalize(
        mean=TinyImageNetInfo.mean, std=TinyImageNetInfo.std
    )

    test_data = TinyImageNet(
        root=tinyimagenet_dir,
        train=False,
        transform=transforms.Compose(
            [
                transforms.ToTensor(),
                normalize,
            ]
        ),
        download=True,
    )

    test_loader = (
        DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
            collate_fn=dup_collate_fn(dups),
        )
        if dups > 1
        else DataLoader(
            test_data,
            batch_size=batch,
            num_workers=workers,
            worker_init_fn=seed_worker,
            generator=g,
            shuffle=False,
            pin_memory=pin_memory,
        )
    )
    return test_loader


class ImageNetInfo:
    outclass = 1000
    imgshape = (3, 224, 224)
    counts = {"train": 1281167, "test": 50000}
    mean = np.array([0.485, 0.456, 0.406]) * 255
    std = np.array([0.229, 0.224, 0.225]) * 255


# cf. https://github.com/libffcv/ffcv-imagenet/blob/main/train_imagenet.py

def get_imagenet_train_loader(
    imagenet_dir: str,
    workers: int,
    tbatch: int,
    device: torch.device,
    dtype: torch.dtype = np.float32,
    distributed: bool = True,
    noaugment: bool = False,
    shuffle: bool = True 
):
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        ToTorchImage,
        RandomHorizontalFlip,
        NormalizeImage,
        Squeeze
    )
    from ffcv.fields.decoders import (
        IntDecoder,
        RandomResizedCropRGBImageDecoder,
        CenterCropRGBImageDecoder,
    )

    if noaugment == True: 
        cropper = CenterCropRGBImageDecoder((224, 224), ratio=224.0/256.0)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(
                np.asarray(ImageNetInfo.mean),
                np.asarray(ImageNetInfo.std),
                dtype
            ),
        ]
        droplast = False 
    else: 
        # Random resized crop
        decoder = RandomResizedCropRGBImageDecoder((224, 224))

        # Data decoding and augmentation
        image_pipeline = [
            decoder,
            RandomHorizontalFlip(),
            ToTensor(),
            ToDevice(device, non_blocking=True),
            ToTorchImage(),
            NormalizeImage(
                ImageNetInfo.mean,
                ImageNetInfo.std,
                dtype
            ),
        ]
        droplast=True

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    # Pipeline for each data field
    pipelines = {
        'image': image_pipeline,
        'label': label_pipeline
    }

    loader = Loader(
        pjoin(imagenet_dir, 'train.ffcv'),
        batch_size=tbatch,
        num_workers=workers,
        order=OrderOption.RANDOM if shuffle else OrderOption.SEQUENTIAL,
        os_cache=True,
        drop_last=droplast,
        pipelines=pipelines, 
        distributed=distributed)

    return loader


def get_imagenet_test_loader(
    imagenet_dir: str,
    workers: int,
    batch: int,
    device: torch.device,
    dtype: torch.dtype = np.float32,
    distributed: bool = True,
):
    from ffcv.loader import Loader, OrderOption
    from ffcv.transforms import (
        ToTensor,
        ToDevice,
        ToTorchImage,
        NormalizeImage,
        Squeeze
    )
    from ffcv.fields.decoders import (
        IntDecoder,
        CenterCropRGBImageDecoder,
    )

    cropper = CenterCropRGBImageDecoder((224, 224), ratio=224.0/256.0)
    image_pipeline = [
        cropper,
        ToTensor(),
        ToDevice(device, non_blocking=True),
        ToTorchImage(),
        NormalizeImage(
            np.asarray(ImageNetInfo.mean),
            np.asarray(ImageNetInfo.std),
            dtype
        ),
    ]

    label_pipeline = [
        IntDecoder(),
        ToTensor(),
        Squeeze(),
        ToDevice(device, non_blocking=True)
    ]

    loader = Loader(pjoin(imagenet_dir, 'val.ffcv'),
                    batch_size=batch,
                    num_workers=workers,
                    order=OrderOption.SEQUENTIAL,
                    drop_last=False,
                    pipelines={
                        'image': image_pipeline,
                        'label': label_pipeline
                    },
                    distributed=distributed,
                    )
    return loader


# available datasets and corresponding train/val loaders
TRAINDATALOADERS = {
    "cifar10": get_cifar10_train_loaders,
    "cifar100": get_cifar100_train_loaders,
    "mnist": get_mnist_train_loaders,
    "fmnist": get_fmnist_train_loaders,
    "tinyimagenet": get_tinyimagenet_train_loaders,
}
# available datasets and corresponding test loader
TESTDATALOADER = {
    "cifar10": get_cifar10_test_loader,
    "cifar100": get_cifar100_test_loader,
    "mnist": get_mnist_test_loader,
    "fmnist": get_fmnist_test_loader,
    "tinyimagenet": get_tinyimagenet_test_loader,
}
# number of training data
NTRAIN = {
    "cifar10": CIFAR10Info.counts["train"],
    "cifar100": CIFAR100Info.counts["train"],
    "mnist": MNISTInfo.counts["train"],
    "fmnist": FashionMNISTInfo.counts["train"],
    "tinyimagenet": TinyImageNetInfo.counts["train"],
}
# number of test data
NTEST = {
    "cifar10": CIFAR10Info.counts["test"],
    "cifar100": CIFAR100Info.counts["test"],
    "mnist": MNISTInfo.counts["test"],
    "fmnist": FashionMNISTInfo.counts["test"],
    "tinyimagenet": TinyImageNetInfo.counts["test"],
}
# input image size
INSIZE = {
    "cifar10": CIFAR10Info.imgshape[-1],
    "cifar100": CIFAR100Info.imgshape[-1],
    "mnist": MNISTInfo.imgshape[-1],
    "fmnist": FashionMNISTInfo.imgshape[-1],
    "tinyimagenet": TinyImageNetInfo.imgshape[-1],
}
# number of classes
OUTCLASS = {
    "cifar10": CIFAR10Info.outclass,
    "cifar100": CIFAR100Info.outclass,
    "mnist": MNISTInfo.outclass,
    "fmnist": FashionMNISTInfo.outclass,
    "tinyimagenet": TinyImageNetInfo.outclass,
}
