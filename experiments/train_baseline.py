"""Unified baseline training script for SGD, AdamW, IVON, and AdaHessian."""
import argparse
import os
import sys
from os.path import join as pjoin

import torch
import torch.nn.functional as nnf
from torch.utils.data import Subset, DataLoader

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.data_utils import corrupt_labels
from training.checkpoint import loadcheckpoint, savecheckpoint
from training.coroutines import coro_timer
from training.logging import coro_log_timed
from training.utils import check_cuda, deterministic_run, mkdirp
from models import STANDARDMODELS, MCDROPMODELS
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, NTRAIN, OUTCLASS, INSIZE
from training.engine import SummaryWriter, do_epoch, do_evalbatch, do_trainbatch

ALL_MODELS = {**STANDARDMODELS, **MCDROPMODELS}
BASELINE_OPTIMIZERS = ("sgd", "adamw", "ivon", "adahessian", "mcdrop")


def get_args():
    p = argparse.ArgumentParser(description="Baseline training (SGD / AdamW / IVON / AdaHessian / MCDrop)")

    p.add_argument("arch", choices=ALL_MODELS)
    p.add_argument("--optimizer", required=True, choices=BASELINE_OPTIMIZERS)
    p.add_argument("dataset", choices=TRAINDATALOADERS)
    p.add_argument("-j", "--workers", default=0, type=int)
    p.add_argument("-tb", "--tbatch", default=512, type=int)
    p.add_argument("-vb", "--vbatch", default=512, type=int)
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float)
    p.add_argument("-tf", "--train_fraction", default=1.0, type=float)
    p.add_argument("-nr", "--noise_rate", default=0.0, type=float)
    p.add_argument("-e", "--epochs", default=200, type=int)
    p.add_argument("-d", "--device", default="cpu", type=str)
    p.add_argument("-s", "--seed", type=int)
    p.add_argument("-r", "--resume", default="", type=str)

    p.add_argument("-pf", "--printfreq", default=200, type=int)
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    p.add_argument("-dd", "--data_dir", default="../data", type=str)
    p.add_argument("-tbd", "--tensorboard_dir", default="", type=str)
    p.add_argument("-nb", "--bins", default=20, type=int)
    p.add_argument("-pd", "--plotdiagram", action="store_true")

    p.add_argument("-lr", "--learning_rate", default=0.1, type=float)
    p.add_argument("--lr_final", default=0.0, type=float)
    p.add_argument("--warmup", default=5, type=int)
    p.add_argument("--rescale_lr", action="store_true")

    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.99999, type=float)
    p.add_argument("--hess_init", default=0.5, type=float)

    # IVON-specific
    p.add_argument("--hess_approx", default="price", type=str)
    p.add_argument("--mc_samples", default=1, type=int)
    p.add_argument("--ess", default=5e4, type=float)
    p.add_argument("--clip_radius", default=float("inf"), type=float)

    return p.parse_args()


def build_optimizer(args, model):
    if args.optimizer == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=args.learning_rate,
            momentum=args.beta1,
            weight_decay=args.weight_decay,
        )
    elif args.optimizer in ("adamw", "mcdrop"):
        return torch.optim.AdamW(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    elif args.optimizer == "ivon":
        from methods.baselines.ivon import IVON
        return IVON(
            model.parameters(),
            lr=args.learning_rate,
            mc_samples=args.mc_samples,
            beta1=args.beta1,
            beta2=args.beta2,
            weight_decay=args.weight_decay,
            hess_approx=args.hess_approx,
            hess_init=args.hess_init,
            ess=args.ess,
            clip_radius=args.clip_radius,
            rescale_lr=args.rescale_lr,
        )
    elif args.optimizer == "adahessian":
        from methods.baselines.adahessian import AdaHessian
        return AdaHessian(
            model.parameters(),
            lr=args.learning_rate,
            betas=(args.beta1, args.beta2),
            weight_decay=args.weight_decay,
        )
    else:
        raise ValueError(f"Unknown optimizer: {args.optimizer}")


def build_scheduler(args, optimizer):
    if args.warmup > 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / args.warmup,
            end_factor=1.0,
            total_iters=args.warmup,
        )
    return None


def do_trainbatch_ivon(batchinput, model, optimizer, mc_samples=1):
    images, target = batchinput
    loss_samples = []
    prob_samples = []
    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            loss.backward()
        loss_samples.append(loss.detach())
        prob_samples.append(nnf.softmax(output.detach(), -1))
    optimizer.step()
    loss = torch.mean(torch.stack(loss_samples, dim=0), dim=0)
    prob = torch.mean(torch.stack(prob_samples, dim=0), dim=0)
    return prob, target, loss.item()


if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Training initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deterministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)

    if args.resume:
        startepoch, model, optimizer, scheduler, dic = loadcheckpoint(args.resume, device)
        modelargs, modelkwargs = dic["modelargs"], dic["modelkwargs"]
        print(f"resumed from {args.resume}\n")
    else:
        startepoch = 0
        modelargs = (OUTCLASS[args.dataset], INSIZE[args.dataset])
        modelkwargs = {}
        model = ALL_MODELS[args.arch](*modelargs, **modelkwargs).to(device)
        optimizer = build_optimizer(args, model)
        scheduler = build_scheduler(args, optimizer)

    sw = None
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)

    train_loader, val_loader = TRAINDATALOADERS[args.dataset](
        args.data_dir,
        args.tvsplit,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
        args.vbatch,
    )

    if args.train_fraction < 1.0:
        n_train = len(train_loader.dataset)
        n_keep = max(1, int(n_train * args.train_fraction))
        g = torch.Generator()
        if args.seed is not None:
            g.manual_seed(args.seed)
        perm = torch.randperm(n_train, generator=g).tolist()
        train_loader = DataLoader(
            Subset(train_loader.dataset, perm[:n_keep]),
            batch_size=args.tbatch,
            shuffle=True,
            num_workers=args.workers,
            pin_memory=(device != torch.device("cpu")),
        )

    if args.noise_rate > 0.0:
        base_dataset = train_loader.dataset
        indices = None
        while isinstance(base_dataset, torch.utils.data.Subset):
            if indices is None:
                indices = list(base_dataset.indices)
            else:
                indices = [base_dataset.indices[i] for i in indices]
            base_dataset = base_dataset.dataset
        corrupt_labels(base_dataset, args.noise_rate, seed=args.seed, indices=indices)

    test_loader = TESTDATALOADER[args.dataset](
        args.data_dir,
        args.workers,
        (device != torch.device("cpu")),
        args.tbatch,
    )

    data_size = len(train_loader.dataset)
    print(
        f"datasize {data_size}, paramsize "
        f"{sum(p.nelement() for p in model.parameters())}"
    )

    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    log_ece = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)
    checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    # pick the right training batch function
    if args.optimizer == "ivon":
        def _trainbatch(batchinput, model, optimizer):
            return do_trainbatch_ivon(batchinput, model, optimizer, args.mc_samples)
    elif args.optimizer == "adahessian":
        def _trainbatch(batchinput, model, optimizer):
            optimizer.zero_grad(set_to_none=True)
            inputs, gt = batchinput[:-1], batchinput[-1]
            output = model(*inputs)
            loss = nnf.cross_entropy(output, gt)
            loss.backward(create_graph=True)
            optimizer.step()
            return nnf.softmax(output.detach(), 1), gt, loss.item()
    else:
        _trainbatch = do_trainbatch

    for e in range(startepoch, args.epochs):
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=args.epochs
            )

        model.train()
        log_ece.send((e, "train", len(train_loader), None))
        do_epoch(train_loader, _trainbatch, log_ece, device, model=model, optimizer=optimizer)
        log_ece.throw(StopIteration)

        if scheduler is not None:
            scheduler.step()

        savecheckpoint(
            pjoin(args.save_dir, "checkpoint.pt"),
            args.arch, modelargs, modelkwargs, model, optimizer, scheduler,
        )
        if e in checkpoint_epochs:
            savecheckpoint(
                pjoin(args.save_dir, "checkpoint%03d.pt" % (e + 1)),
                args.arch, modelargs, modelkwargs, model, optimizer, scheduler,
            )

        if device.type != "cpu":
            print(f"Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Peak reserved:  {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")

        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")

        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        log_ece.send((e, "test", len(test_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
        log_ece.throw(StopIteration)

        if len(val_loader) == 0:
            continue

        log_ece.send((e, "val", len(val_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
        log_ece.throw(StopIteration)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
