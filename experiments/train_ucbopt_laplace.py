"""Train with uCBOpt then fit a post-hoc Laplace approximation."""
import argparse
import os
import sys
import warnings
from os.path import join as pjoin

import torch
import torch.nn.functional as nnf
from torch.utils.data import DataLoader, Subset
from laplace import Laplace

warnings.filterwarnings(
    "ignore",
    message=r"gemm_and_bias error: CUBLAS_STATUS_NOT_INITIALIZED.*",
    category=UserWarning,
    module=r"torch\.nn\.modules\.linear",
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from data.data_utils import corrupt_labels
from methods.ucbopt import uCBOpt
from core.checkpoint import loadcheckpoint, savecheckpoint
from core.coroutines import coro_timer
from core.logging import coro_log_timed
from core.utils import check_cuda, deterministic_run, mkdirp
from models import STANDARDMODELS
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, OUTCLASS, INSIZE
from core.engine import SummaryWriter, do_epoch, do_evalbatch, do_trainbatch


def get_args():
    p = argparse.ArgumentParser(description="uCBOpt + post-hoc Laplace approximation")
    p.add_argument("arch", choices=STANDARDMODELS)
    p.add_argument("dataset", choices=TRAINDATALOADERS)
    p.add_argument("-j", "--workers", default=0, type=int)
    p.add_argument("-tb", "--tbatch", default=128, type=int)
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

    # uCBOpt
    p.add_argument("-lr", "--learning_rate", default=1.0, type=float)
    p.add_argument("--lr_final", default=0.0, type=float)
    p.add_argument("--warmup", default=5, type=int)
    p.add_argument("--rescale_lr", action="store_true")
    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=1e-4, type=float)
    p.add_argument("--beta1", default=0.9, type=float)
    p.add_argument("--beta2", default=0.99999, type=float)
    p.add_argument("--hess_init", default=0.5, type=float)
    p.add_argument("--no_hess_init", action="store_true", default=False,
                   help="initialize exp_avg_sq to 0 instead of hess_init")
    p.add_argument("--cand_curvature", default=4e-6, type=float)

    # Laplace
    p.add_argument("--subset_of_weights", default="last_layer",
                   choices=["last_layer", "all", "subnetwork"])
    p.add_argument("--hessian_structure", default="kron",
                   choices=["diag", "kron", "full", "lowrank"])
    p.add_argument("--pred_type", default="glm",
                   choices=["glm", "nn"])
    p.add_argument("--link_approx", default="probit",
                   choices=["mc", "probit", "bridge", "bridge_norm"])
    p.add_argument("--n_samples", default=100, type=int)
    p.add_argument("--prior_precision", default=1.0, type=float)
    p.add_argument("--optimize_prior_precision", action=argparse.BooleanOptionalAction,
                   default=True)
    p.add_argument("--prior_precision_method", default="marglik",
                   choices=["gridsearch", "marglik"])

    args = p.parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {args.train_fraction}")
    return args


def build_optimizer(args, model):
    return uCBOpt(
        model.parameters(),
        lr=args.learning_rate,
        beta1=args.beta1,
        beta2=args.beta2,
        hess_init=args.hess_init,
        weight_decay=args.weight_decay,
        cand_curvature=args.cand_curvature,
        rescale_lr=args.rescale_lr,
        use_hess_init=not args.no_hess_init,
    )


def build_scheduler(args, optimizer):
    if args.warmup > 0:
        return torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0 / args.warmup,
            end_factor=1.0, total_iters=args.warmup,
        )
    return None



class LaplaceEvalWrapper:
    def __init__(self, la, pred_type="glm", link_approx="probit", n_samples=100):
        self.la = la
        self.pred_type = pred_type
        self.link_approx = link_approx
        self.n_samples = n_samples

    def __call__(self, batchinput):
        images, target = batchinput
        kwargs = {"pred_type": self.pred_type}
        if self.pred_type == "glm":
            kwargs["link_approx"] = self.link_approx
        if self.link_approx == "mc" or self.pred_type == "nn":
            kwargs["n_samples"] = self.n_samples
        prob = self.la(images, **kwargs)
        loss = nnf.nll_loss(prob.clamp_min(1e-12).log(), target)
        return prob, target, loss.item()


def fit_laplace(args, model, train_loader, val_loader):
    model.eval()
    la = Laplace(model, likelihood="classification",
                 subset_of_weights=args.subset_of_weights,
                 hessian_structure=args.hessian_structure,
                 prior_precision=args.prior_precision)
    print(f"Fitting Laplace: subset={args.subset_of_weights}, "
          f"hessian={args.hessian_structure}, prior={args.prior_precision}")
    la.fit(train_loader)

    if args.optimize_prior_precision:
        print(f"Optimizing prior precision via {args.prior_precision_method}")
        optimize_kwargs = {
            "method": args.prior_precision_method,
            "pred_type": args.pred_type,
            "val_loader": val_loader,
        }
        if args.pred_type == "glm":
            optimize_kwargs["link_approx"] = args.link_approx
        if args.link_approx == "mc" or args.pred_type == "nn":
            optimize_kwargs["n_samples"] = args.n_samples
        if args.prior_precision_method == "gridsearch" and len(val_loader) == 0:
            print("Skipping gridsearch prior optimization: validation loader is empty")
        else:
            la.optimize_prior_precision(**optimize_kwargs)

    prior_prec = la.prior_precision
    prior_prec_val = (float(prior_prec.detach().cpu().item())
                      if torch.is_tensor(prior_prec) else float(prior_prec))
    laplace_ckpt = {
        "state_dict": la.state_dict(),
        "config": {
            "likelihood": "classification",
            "subset_of_weights": args.subset_of_weights,
            "hessian_structure": args.hessian_structure,
            "pred_type": args.pred_type,
            "link_approx": args.link_approx,
            "n_samples": args.n_samples,
            "prior_precision": prior_prec_val,
        },
    }
    torch.save(laplace_ckpt, pjoin(args.save_dir, "laplace_state.pt"))
    print(f"Saved Laplace state to {pjoin(args.save_dir, 'laplace_state.pt')}")
    return la


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
        args.data_dir, args.tvsplit, args.workers,
        (device != torch.device("cpu")), args.tbatch, args.vbatch,
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
            batch_size=args.tbatch, shuffle=True,
            num_workers=args.workers, pin_memory=(device != torch.device("cpu")),
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
        args.data_dir, args.workers, (device != torch.device("cpu")), args.tbatch,
    )

    print(f"datasize {len(train_loader.dataset)}, "
          f"paramsize {sum(p.nelement() for p in model.parameters())}")
    print(f">>> Training starts at {next(timer)[0].isoformat()} <<<\n")

    log_map = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)
    checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    for e in range(startepoch, args.epochs):
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, eta_min=args.lr_final,
                T_max=max(args.epochs - args.warmup, 1),
            )

        model.train()
        log_map.send((e, "train", len(train_loader), None))
        do_epoch(train_loader, do_trainbatch, log_map, device,
                 model=model, optimizer=optimizer)
        log_map.throw(StopIteration)

        if scheduler is not None:
            scheduler.step()

        savecheckpoint(pjoin(args.save_dir, "checkpoint.pt"),
                       args.arch, modelargs, modelkwargs, model, optimizer, scheduler)
        if e in checkpoint_epochs:
            savecheckpoint(pjoin(args.save_dir, f"checkpoint{e + 1:03d}.pt"),
                           args.arch, modelargs, modelkwargs, model, optimizer, scheduler)

        if device.type != "cpu":
            print(f"Peak allocated: {torch.cuda.max_memory_allocated(device) / 1024**3:.2f} GB")
            print(f"Peak reserved:  {torch.cuda.max_memory_reserved(device) / 1024**3:.2f} GB")

        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")
        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        if len(val_loader) == 0:
            log_map.send((e, "test", len(test_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(test_loader, do_evalbatch, log_map, device, model=model)
            log_map.throw(StopIteration)
        else:
            log_map.send((e, "val", len(val_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(val_loader, do_evalbatch, log_map, device, model=model)
            log_map.throw(StopIteration)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    log_map.close()

    print(f">>> Fitting Laplace starts at {next(timer)[0].isoformat()} <<<\n")
    la = fit_laplace(args, model, train_loader, val_loader)

    log_la = coro_log_timed(None, args.printfreq, args.bins, args.save_dir)
    eval_laplace = LaplaceEvalWrapper(la, pred_type=args.pred_type,
                                      link_approx=args.link_approx,
                                      n_samples=args.n_samples)

    if len(val_loader) == 0:
        log_la.send((args.epochs, "test_laplace", len(test_loader), None))
        with torch.no_grad():
            do_epoch(test_loader, eval_laplace, log_la, device)
        log_la.throw(StopIteration)
    else:
        log_la.send((args.epochs, "val_laplace", len(val_loader), None))
        with torch.no_grad():
            do_epoch(val_loader, eval_laplace, log_la, device)
        log_la.throw(StopIteration)

    log_la.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")
