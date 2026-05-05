"""Train a model with the uCBOpt Adaptive Curvature optimizer."""

import argparse
import math
import os
import sys
import warnings
from os.path import join as pjoin

import torch
from torch.utils.data import Subset, DataLoader

warnings.filterwarnings(
    "ignore",
    message=r"gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.linear",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.data_utils import corrupt_labels
from methods.ucbopt_adaptcurv import uCBOptAdaptCurv
from core.checkpoint import loadcheckpoint, savecheckpoint
from core.coroutines import coro_timer
from core.logging import coro_log_timed
from core.utils import check_cuda, deterministic_run, mkdirp
from models import STANDARDMODELS
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, NTRAIN, OUTCLASS, INSIZE
from core.engine import SummaryWriter, do_epoch, do_evalbatch, do_trainbatch


def get_args():
    p = argparse.ArgumentParser(description="uCBOpt Adaptive Curvature training")

    p.add_argument("arch", choices=STANDARDMODELS, help="model architecture")
    p.add_argument("dataset", choices=TRAINDATALOADERS, help="dataset")
    p.add_argument("-j", "--workers", default=0, type=int, help="data loader workers")
    p.add_argument("-tb", "--tbatch", default=512, type=int, help="train batch size")
    p.add_argument("-vb", "--vbatch", default=512, type=int, help="eval batch size")
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float, help="train split ratio")
    p.add_argument("-tf", "--train_fraction", default=1.0, type=float, help="fraction of training data to keep")
    p.add_argument("-nr", "--noise_rate", default=0.0, type=float, help="label noise rate")
    p.add_argument("-e", "--epochs", default=400, type=int, help="epochs")
    p.add_argument("-d", "--device", default="cpu", type=str)
    p.add_argument("-s", "--seed", type=int)
    p.add_argument("-r", "--resume", default="", type=str)

    p.add_argument("-pf", "--printfreq", default=200, type=int)
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    p.add_argument("-dd", "--data_dir", default="../data", type=str)
    p.add_argument("-tbd", "--tensorboard_dir", default="", type=str)
    p.add_argument("-nb", "--bins", default=20, type=int)
    p.add_argument("-pd", "--plotdiagram", action="store_true")

    p.add_argument("-lr", "--learning_rate", default=0.01, type=float)
    p.add_argument("--lr_final", default=0.0, type=float)
    p.add_argument("--warmup", default=5, type=int)

    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.999, type=float)
    p.add_argument("--beta3", default=0.999, type=float)
    p.add_argument("--hess_init", default=0.5, type=float)
    p.add_argument("--no_hess_init", action="store_true", default=False,
                   help="initialize exp_avg_sq to 0 instead of hess_init")
    p.add_argument("--gamma", default=0.1, type=float, help="adaptive curvature weight")
    p.add_argument("--eps", default=1e-8, type=float)
    p.add_argument("--rescale_lr", action="store_true", default=False,
                   help="scale lr by (hess_init + weight_decay) as in uCBOpt")
    p.add_argument("--no-rescale_lr", dest="rescale_lr", action="store_false")
    p.add_argument("--clip-radius", default=float("inf"), type=float,
                   help="elementwise update clipping radius (inf = disabled)")
    p.add_argument("--decoupled-wd", action="store_true", default=False,
                   help="apply weight decay decoupled from the preconditioner")
    p.add_argument("--maximize", action="store_true", default=False)

    return p.parse_args()


def build_optimizer(args, model):
    return uCBOptAdaptCurv(
        model.parameters(),
        lr=args.learning_rate,
        betas=(args.beta1, args.beta2, args.beta3),
        weight_decay=args.weight_decay,
        hess_init=args.hess_init,
        gamma=args.gamma,
        eps=args.eps,
        rescale_lr=args.rescale_lr,
        clip_radius=args.clip_radius,
        maximize=args.maximize,
        use_hess_init=not args.no_hess_init,
    )


def build_scheduler(args, optimizer):
    if args.warmup > 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1.0 / args.warmup,
            end_factor=1.0,
            total_iters=args.warmup,
        )
    return None


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
        model = STANDARDMODELS[args.arch](*modelargs, **modelkwargs).to(device)
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
    has_validation = len(val_loader) > 0
    best_val_metric = float("inf")
    best_val_epoch = None
    best_checkpoint_path = pjoin(args.save_dir, "best_checkpoint.pt")
    if has_validation and os.path.exists(best_checkpoint_path):
        _, _, _, _, best_checkpoint_meta = loadcheckpoint(best_checkpoint_path, device)
        best_val_metric = best_checkpoint_meta.get("best_val_metric", best_val_metric)
        best_val_epoch = best_checkpoint_meta.get("best_val_epoch", best_val_epoch)

    for e in range(startepoch, args.epochs):
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=max(args.epochs - args.warmup, 1)
            )

        model.train()
        log_ece.send((e, "train", len(train_loader), None))
        do_epoch(train_loader, do_trainbatch, log_ece, device, model=model, optimizer=optimizer)
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

        if not has_validation:
            log_ece.send((e, "test", len(test_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
            log_ece.throw(StopIteration)
            continue

        log_ece.send((e, "val", len(val_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
        val_metrics = log_ece.throw(StopIteration)

        _, val_loss, *_ = val_metrics
        if math.isfinite(val_loss) and val_loss < best_val_metric:
            best_val_metric = val_loss
            best_val_epoch = e
            savecheckpoint(
                best_checkpoint_path,
                args.arch, modelargs, modelkwargs, model, optimizer, scheduler,
                epoch=e, best_val_metric_name="val_loss",
                best_val_metric=best_val_metric, best_val_epoch=best_val_epoch,
            )
            print(f"New best checkpoint saved at epoch {e} with val_loss={best_val_metric:.4f}")

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
