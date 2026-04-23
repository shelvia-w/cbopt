import argparse
from os.path import join as pjoin
import sys

import torch
import torch.nn.functional as nnf
from torch.optim import SGD
from torch.utils.data import DataLoader, Subset
from laplace import Laplace

sys.path.append("..")

from common.utils import coro_timer, mkdirp
from common.models import STANDARDMODELS
from common.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    NTRAIN,
    OUTCLASS,
    INSIZE,
    seed_worker,
)
from common.trainutils import (
    coro_log_timed,
    do_epoch,
    do_evalbatch,
    SummaryWriter,
    check_cuda,
    deteministic_run,
    savecheckpoint,
    loadcheckpoint,
    corrupt_labels
)


def get_args():
    p = argparse.ArgumentParser(description="Laplace-Redux training with laplace-torch")

    p.add_argument("arch", choices=STANDARDMODELS, help="model architecture")
    p.add_argument("dataset", choices=TRAINDATALOADERS, help="dataset")
    p.add_argument("-j", "--workers", default=0, type=int, help="data loader workers")
    p.add_argument("-tb", "--tbatch", default=128, type=int, help="train batch size")
    p.add_argument("-vb", "--vbatch", default=512, type=int, help="eval batch size")
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float, help="train split ratio")
    p.add_argument("-tf", "--train_fraction", default=1.0, type=float, help="fraction of training data to keep")
    p.add_argument("-nr", "--noise_rate", default=0.0, type=float, help="fraction of training labels to randomly corrupt")
    p.add_argument("-e", "--epochs", default=200, type=int, help="epochs")
    p.add_argument("-d", "--device", default="cpu", type=str, help="cpu/cuda")
    p.add_argument("-s", "--seed", type=int, help="seed for reproducibility")
    p.add_argument("-r", "--resume", default="", type=str, help="checkpoint path")

    p.add_argument("-pf", "--printfreq", default=200, type=int, help="print frequency")
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str, help="directory used to save results")
    p.add_argument("-dd", "--data_dir", default="../data", type=str, help="directory to find/store dataset")
    p.add_argument("-tbd", "--tensorboard_dir", default="", type=str, help="tensorboard directory")
    p.add_argument("-nb", "--bins", default=20, type=int, help="number of bins for ece & reliability diagram")

    # MAP training hyperparameters
    p.add_argument("-lr", "--learning_rate", default=0.05, type=float, help="initial learning rate")
    p.add_argument("--lr_final", default=0.0, type=float, help="final learning rate")
    p.add_argument("--warmup", default=5, type=int, help="number of learning rate warmup epochs")
    p.add_argument("--wd", "--weight-decay", dest="weight_decay", default=2e-4, type=float, help="weight decay")
    p.add_argument("--momentum", default=0.9, type=float, help="SGD momentum")

    # Laplace hyperparameters
    p.add_argument("--subset_of_weights", default="last_layer", choices=["last_layer", "all", "subnetwork"], help="Laplace subset of weights")
    p.add_argument("--hessian_structure", default="diag", choices=["diag", "kron", "full", "lowrank"], help="Laplace Hessian structure")
    p.add_argument("--pred_type", default="glm", choices=["glm", "nn", "linear_sampling", "mc"], help="predictive type used for prior tuning and evaluation")
    p.add_argument("--link_approx", default="probit", choices=["mc", "probit", "bridge", "bridge_norm"], help="classification link approximation")
    p.add_argument("--n_samples", default=100, type=int, help="number of predictive samples for MC-based predictions")
    p.add_argument("--prior_precision", default=1.0, type=float, help="initial prior precision")
    p.add_argument("--optimize_prior_precision", action="store_true", help="tune prior precision on validation data")
    p.add_argument("--prior_precision_method", default="gridsearch", choices=["gridsearch", "marglik"], help="prior precision optimization method")

    args = p.parse_args()
    if not (0.0 < args.train_fraction <= 1.0):
        raise ValueError(f"train_fraction must be in (0, 1], got {args.train_fraction}")
    return args


def build_optimizer(args, model):
    return SGD(
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


def do_trainbatch_sgd(batchinput, model, optimizer):
    images, target = batchinput
    optimizer.zero_grad(set_to_none=True)
    output = model(images)
    loss = nnf.cross_entropy(output, target)
    loss.backward()
    optimizer.step()
    prob = nnf.softmax(output.detach(), dim=1)
    return prob, target, loss.item()


class LaplaceEvalWrapper:
    def __init__(self, la, pred_type="glm", link_approx="probit", n_samples=100):
        self.la = la
        self.pred_type = pred_type
        self.link_approx = link_approx
        self.n_samples = n_samples

    def __call__(self, batchinput):
        images, target = batchinput
        kwargs = {"pred_type": self.pred_type}
        if self.pred_type in {"glm", "mc"}:
            kwargs["link_approx"] = self.link_approx
        if self.pred_type in {"mc", "linear_sampling"}:
            kwargs["n_samples"] = self.n_samples
        prob = self.la(images, **kwargs)
        loss = nnf.nll_loss(prob.clamp_min(1e-12).log(), target)
        return prob, target, loss.item()

def fit_laplace(args, model, train_loader, val_loader):
    model.eval()
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights=args.subset_of_weights,
        hessian_structure=args.hessian_structure,
        prior_precision=args.prior_precision,
    )

    print(
        f"Fitting Laplace: subset={args.subset_of_weights}, "
        f"hessian_structure={args.hessian_structure}, prior_precision={args.prior_precision}"
    )
    la.fit(train_loader)

    if args.optimize_prior_precision and len(val_loader) > 0:
        print(
            f"Optimizing prior precision with method={args.prior_precision_method}, "
            f"pred_type={args.pred_type}, link_approx={args.link_approx}"
        )
        optimize_kwargs = {
            "method": args.prior_precision_method,
            "pred_type": args.pred_type,
            "val_loader": val_loader,
        }
        if args.pred_type in {"glm", "mc"}:
            optimize_kwargs["link_approx"] = args.link_approx
        if args.pred_type in {"mc", "linear_sampling"}:
            optimize_kwargs["n_samples"] = args.n_samples
        la.optimize_prior_precision(**optimize_kwargs)

    laplace_ckpt = {
        "state_dict": la.state_dict(),
        "config": {
            "likelihood": "classification",
            "subset_of_weights": args.subset_of_weights,
            "hessian_structure": args.hessian_structure,
            "pred_type": args.pred_type,
            "link_approx": args.link_approx,
            "n_samples": args.n_samples,
            "prior_precision": float(la.prior_precision.detach().cpu().item())
            if torch.is_tensor(la.prior_precision)
            else float(la.prior_precision),
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
        deteministic_run(seed=args.seed)

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
        modelargs, modelkwargs = (OUTCLASS[args.dataset], INSIZE[args.dataset]), {}
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
        keep_idx = perm[:n_keep]

        train_loader = DataLoader(
            Subset(train_loader.dataset, keep_idx),
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
    log_map = coro_log_timed(sw, args.printfreq, args.bins, args.save_dir)
    checkpoint_epochs = [0, 1, 2, 5, 10, 20, 30, 40, 50, 75, 100, 150, 200]

    for e in range(startepoch, args.epochs):
        if args.warmup > 0 and e == args.warmup:
            print("End of warmup epochs, starting cosine annealing")
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                eta_min=args.lr_final,
                T_max=max(args.epochs - args.warmup, 1),
            )

        model.train()
        log_map.send((e, "train", len(train_loader), None))
        do_epoch(
            train_loader,
            do_trainbatch_sgd,
            log_map,
            device,
            model=model,
            optimizer=optimizer,
        )
        log_map.throw(StopIteration)

        if scheduler is not None:
            scheduler.step()

        savecheckpoint(
            pjoin(args.save_dir, "checkpoint.pt"),
            args.arch,
            modelargs,
            modelkwargs,
            model,
            optimizer,
            scheduler,
        )
        if e in checkpoint_epochs:
            savecheckpoint(
                pjoin(args.save_dir, f"checkpoint{e + 1:03d}.pt"),
                args.arch,
                modelargs,
                modelkwargs,
                model,
                optimizer,
                scheduler,
            )

        if device.type != "cpu":
            print(f"Max memory usage {torch.cuda.max_memory_allocated()}")

        time_per_epoch = next(timer)[1]
        print(f">>> Time elapsed: {time_per_epoch} <<<\n")
        with open(pjoin(args.save_dir, "time.csv"), "a+") as file:
            file.write("%d,%f\n" % (e, time_per_epoch.total_seconds()))

        log_map.send((e, "test", len(test_loader), None))
        with torch.no_grad():
            model.eval()
            do_epoch(test_loader, do_evalbatch, log_map, device, model=model)
        log_map.throw(StopIteration)

        if len(val_loader) > 0:
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
    eval_laplace = LaplaceEvalWrapper(
        la,
        pred_type=args.pred_type,
        link_approx=args.link_approx,
        n_samples=args.n_samples,
    )

    log_la.send((args.epochs, "test_laplace", len(test_loader), None))
    with torch.no_grad():
        do_epoch(test_loader, eval_laplace, log_la, device)
    log_la.throw(StopIteration)

    if len(val_loader) > 0:
        log_la.send((args.epochs, "val_laplace", len(val_loader), None))
        with torch.no_grad():
            do_epoch(val_loader, eval_laplace, log_la, device)
        log_la.throw(StopIteration)

    log_la.close()
    print(f">>> Training completed at {next(timer)[0].isoformat()} <<<\n")