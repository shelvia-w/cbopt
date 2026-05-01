"""Train a DUQ model and evaluate it during training."""

import argparse
import os
import sys
from os.path import join as pjoin

import torch
import torch.nn.functional as nnf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.coroutines import coro_timer
from core.logging import coro_log_timed
from core.utils import check_cuda, deterministic_run, mkdirp
from models.uncertainty.duq import DUQModel, FeatureExtractor, calc_gradient_penalty
from models import STANDARDMODELS
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, NTRAIN, OUTCLASS, INSIZE
from core.engine import SummaryWriter, do_epoch


def get_args():
    p = argparse.ArgumentParser(description="DUQ training")
    p.add_argument("arch", choices=STANDARDMODELS)
    p.add_argument("dataset", choices=TRAINDATALOADERS)
    p.add_argument("-j", "--workers", default=0, type=int)
    p.add_argument("-tb", "--tbatch", default=128, type=int)
    p.add_argument("-vb", "--vbatch", default=512, type=int)
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float)
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
    p.add_argument("-lr", "--learning_rate", default=0.05, type=float)
    p.add_argument("--lr_final", default=0.0, type=float)
    p.add_argument("--warmup", default=5, type=int)
    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=5e-4, type=float)
    p.add_argument("--centroid_dim", default=0, type=int)
    p.add_argument("--length_scale", default=0.1, type=float)
    p.add_argument("--beta", default=0.99, type=float)
    p.add_argument("--lambda_gp", default=0.5, type=float)
    return p.parse_args()


def build_base_model(args):
    return STANDARDMODELS[args.arch](OUTCLASS[args.dataset], INSIZE[args.dataset])


def build_model(args, device):
    num_classes = OUTCLASS[args.dataset]
    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        base_model = build_base_model(args)
        model = DUQModel(
            base_model=base_model,
            num_classes=ckpt["modelkwargs"]["num_classes"],
            centroid_dim=ckpt["modelkwargs"]["centroid_dim"],
            length_scale=ckpt["modelkwargs"]["length_scale"],
            beta=ckpt["modelkwargs"]["beta"],
        ).to(device)
        model.load_state_dict(ckpt["model_state_dict"])
        return model, ckpt["epoch"] + 1
    else:
        temp_base = build_base_model(args)
        temp_feat = FeatureExtractor(temp_base)
        centroid_dim = temp_feat.feature_dim if args.centroid_dim <= 0 else args.centroid_dim
        base_model = build_base_model(args)
        model = DUQModel(
            base_model=base_model,
            num_classes=num_classes,
            centroid_dim=centroid_dim,
            length_scale=args.length_scale,
            beta=args.beta,
        ).to(device)
        return model, 0


def build_optimizer(args, model):
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=0.9,
        weight_decay=args.weight_decay,
    )


def build_scheduler(args, optimizer):
    if args.warmup > 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / args.warmup, end_factor=1.0, total_iters=args.warmup,
        )
    return None


def do_trainbatch_duq(batchinput, model, optimizer, lambda_gp):
    images, target = batchinput
    images.requires_grad_(True)
    optimizer.zero_grad(set_to_none=True)
    scores = model(images)
    y_onehot = nnf.one_hot(target, scores.size(1)).float()
    loss_bce = nnf.binary_cross_entropy(scores, y_onehot)
    gp = calc_gradient_penalty(images, scores) if lambda_gp > 0 else scores.new_zeros(())
    loss = loss_bce + lambda_gp * gp
    loss.backward()
    optimizer.step()
    images.requires_grad_(False)
    with torch.no_grad():
        model.update_embeddings(images, y_onehot)
    prob = scores.detach() / scores.detach().sum(dim=1, keepdim=True).clamp_min(1e-12)
    return prob, target, loss.item()


@torch.no_grad()
def predict_proba_duq(batchinput, model):
    images, target = batchinput
    scores = model(images)
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss = nnf.nll_loss(prob.clamp_min(1e-12).log(), target)
    return prob, target, loss.item()


def save_duq_checkpoint(path, epoch, args, model, optimizer, scheduler):
    torch.save({
        "epoch": epoch,
        "modelkwargs": {
            "num_classes": OUTCLASS[args.dataset],
            "centroid_dim": model.centroid_dim,
            "length_scale": args.length_scale,
            "beta": args.beta,
        },
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": None if scheduler is None else scheduler.state_dict(),
        "args": vars(args),
    }, path)


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

    model, startepoch = build_model(args, device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

    if args.resume:
        ckpt = torch.load(args.resume, map_location=device)
        optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scheduler is not None and ckpt.get("scheduler_state_dict") is not None:
            scheduler.load_state_dict(ckpt["scheduler_state_dict"])
        print(f"resumed from {args.resume}\n")

    sw = None
    if args.tensorboard_dir:
        mkdirp(args.tensorboard_dir)
        sw = SummaryWriter(args.tensorboard_dir)

    train_loader, val_loader = TRAINDATALOADERS[args.dataset](
        args.data_dir, args.tvsplit, args.workers,
        (device != torch.device("cpu")), args.tbatch, args.vbatch,
    )
    test_loader = TESTDATALOADER[args.dataset](
        args.data_dir, args.workers, (device != torch.device("cpu")), args.tbatch,
    )

    data_size = int(NTRAIN[args.dataset] * args.tvsplit)
    print(
        f"datasize {data_size}, paramsize "
        f"{sum(p.nelement() for p in model.parameters())}"
    )
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    log_ece = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)
    checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    for e in range(startepoch, args.epochs):
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=max(args.epochs - args.warmup, 1),
            )

        model.train()
        log_ece.send((e, "train", len(train_loader), None))
        do_epoch(
            train_loader,
            lambda batch, model=model, optimizer=optimizer: do_trainbatch_duq(
                batch, model, optimizer, args.lambda_gp
            ),
            log_ece, device, model=model, optimizer=optimizer,
        )
        log_ece.throw(StopIteration)

        if scheduler is not None:
            scheduler.step()

        save_duq_checkpoint(pjoin(args.save_dir, "checkpoint.pt"), e, args, model, optimizer, scheduler)
        if e in checkpoint_epochs:
            save_duq_checkpoint(pjoin(args.save_dir, "checkpoint%03d.pt" % (e + 1)), e, args, model, optimizer, scheduler)

        if device.type != "cpu":
            print(f"Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Peak reserved:  {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")

        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")
        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        log_ece.send((e, "test", len(test_loader), None))
        model.eval()
        do_epoch(test_loader, predict_proba_duq, log_ece, device, model=model)
        log_ece.throw(StopIteration)

        if len(val_loader) != 0:
            log_ece.send((e, "val", len(val_loader), None))
            model.eval()
            do_epoch(val_loader, predict_proba_duq, log_ece, device, model=model)
            log_ece.throw(StopIteration)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")

