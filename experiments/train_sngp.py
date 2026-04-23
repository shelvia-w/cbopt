import argparse
import math
import os
import sys
from os.path import join as pjoin

import torch
import torch.nn.functional as nnf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from training.utils import coro_timer, mkdirp, savecheckpoint
from models import SNGPMODELS
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, NTRAIN, OUTCLASS, INSIZE
from training.engine import (
    coro_log_timed, do_epoch, do_evalbatch,
    SummaryWriter, check_cuda, deteministic_run,
)


def get_args():
    p = argparse.ArgumentParser(description="SNGP training")
    p.add_argument("arch", choices=SNGPMODELS)
    p.add_argument("dataset", choices=TRAINDATALOADERS)
    p.add_argument("-j", "--workers", default=0, type=int)
    p.add_argument("-tb", "--tbatch", default=512, type=int)
    p.add_argument("-vb", "--vbatch", default=512, type=int)
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float)
    p.add_argument("-e", "--epochs", default=200, type=int)
    p.add_argument("-d", "--device", default="cpu", type=str)
    p.add_argument("-s", "--seed", type=int)
    p.add_argument("-pf", "--printfreq", default=200, type=int)
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    p.add_argument("-dd", "--data_dir", default="../data", type=str)
    p.add_argument("-tbd", "--tensorboard_dir", default="", type=str)
    p.add_argument("-nb", "--bins", default=20, type=int)
    p.add_argument("-lr", "--learning_rate", default=0.08, type=float)
    p.add_argument("--lr_final", default=0.0, type=float)
    p.add_argument("--warmup", default=5, type=int)
    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=2e-4, type=float)
    p.add_argument("--momentum", default=0.9, type=float)
    p.add_argument("--use_spec_norm", action="store_true")
    p.add_argument("--spec_norm_iteration", default=1, type=int)
    p.add_argument("--spec_norm_bound", default=0.95, type=float)
    p.add_argument("--gp_input_dim", default=-1, type=int)
    p.add_argument("--gp_hidden_dim", default=1024, type=int)
    p.add_argument("--gp_scale", default=1.0, type=float)
    p.add_argument("--gp_bias", default=0.0, type=float)
    p.add_argument("--gp_input_normalization", action="store_true")
    p.add_argument("--gp_random_feature_type", default="orf", choices=["orf", "rff"])
    p.add_argument("--gp_cov_discount_factor", default=-1.0, type=float)
    p.add_argument("--gp_cov_ridge_penalty", default=1.0, type=float)
    p.add_argument("--gp_mean_field_factor", default=math.pi / 8.0, type=float)
    p.add_argument("--gp_output_init_std", default=0.01, type=float)
    return p.parse_args()


class TrainBatchSNGP:
    def __init__(self, final_epoch: bool):
        self.final_epoch = final_epoch

    def __call__(self, batchinput, model, optimizer):
        images, target = batchinput
        if hasattr(model, "set_update_covariance"):
            model.set_update_covariance(self.final_epoch)
        optimizer.zero_grad(set_to_none=True)
        output = model(images, mean_field=False)
        loss = nnf.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        return nnf.softmax(output.detach(), dim=-1), target, loss.item()


def build_optimizer(args, model):
    return torch.optim.SGD(
        model.parameters(),
        lr=args.learning_rate,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
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


def build_model_kwargs(args):
    return {
        "use_spec_norm": args.use_spec_norm,
        "spec_norm_iteration": args.spec_norm_iteration,
        "spec_norm_bound": args.spec_norm_bound,
        "gp_input_dim": args.gp_input_dim,
        "gp_hidden_dim": args.gp_hidden_dim,
        "gp_scale": args.gp_scale,
        "gp_bias": args.gp_bias,
        "gp_input_normalization": args.gp_input_normalization,
        "gp_random_feature_type": args.gp_random_feature_type,
        "gp_cov_discount_factor": args.gp_cov_discount_factor,
        "gp_cov_ridge_penalty": args.gp_cov_ridge_penalty,
        "gp_mean_field_factor": args.gp_mean_field_factor,
        "gp_output_init_std": args.gp_output_init_std,
    }


if __name__ == "__main__":
    timer = coro_timer()
    print(f">>> Training initiated at {next(timer).isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deteministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)

    modelargs = (OUTCLASS[args.dataset], INSIZE[args.dataset])
    modelkwargs = build_model_kwargs(args)
    model = SNGPMODELS[args.arch](*modelargs, **modelkwargs).to(device)
    optimizer = build_optimizer(args, model)
    scheduler = build_scheduler(args, optimizer)

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

    print(
        f"datasize {int(NTRAIN[args.dataset] * args.tvsplit)}, "
        f"paramsize {sum(p.nelement() for p in model.parameters())}"
    )
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    log_ece = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)
    checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    for e in range(args.epochs):
        final_epoch = (e == args.epochs - 1)
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final, T_max=args.epochs,
            )

        if final_epoch and hasattr(model, "reset_covariance_matrix"):
            gp_cov = getattr(getattr(model, "gp_layer", None), "gp_cov", None)
            if gp_cov is not None and gp_cov.momentum < 0:
                print("Resetting GP precision matrix for exact final-epoch accumulation")
                model.reset_covariance_matrix()

        model.train()
        log_ece.send((e, "train", len(train_loader), None))
        do_epoch(
            train_loader, TrainBatchSNGP(final_epoch), log_ece, device,
            model=model, optimizer=optimizer,
        )
        log_ece.throw(StopIteration)

        if scheduler is not None:
            scheduler.step()

        savecheckpoint(pjoin(args.save_dir, "checkpoint.pt"), args.arch, modelargs, modelkwargs, model, optimizer, scheduler)
        if e in checkpoint_epochs:
            savecheckpoint(pjoin(args.save_dir, f"checkpoint{e+1:03d}.pt"), args.arch, modelargs, modelkwargs, model, optimizer, scheduler)

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
            if hasattr(model, "set_update_covariance"):
                model.set_update_covariance(False)
            do_epoch(test_loader, do_evalbatch, log_ece, device, model=model)
        log_ece.throw(StopIteration)

        if len(val_loader) > 0:
            log_ece.send((e, "val", len(val_loader), None))
            with torch.no_grad():
                model.eval()
                if hasattr(model, "set_update_covariance"):
                    model.set_update_covariance(False)
                do_epoch(val_loader, do_evalbatch, log_ece, device, model=model)
            log_ece.throw(StopIteration)

    log_ece.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
