"""Evaluate saved in-domain checkpoints and write prediction summaries."""

import argparse
import os
import sys
from os.path import join as pjoin, exists
from glob import glob

import torch
import torch.nn.functional as nnf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.checkpoint import loadcheckpoint
from core.coroutines import coro_timer
from core.logging import coro_log_metrics
from core.utils import check_cuda, deterministic_run, get_outputsaver, mkdirp, summarize_csv
from core.engine import do_epoch, do_evalbatch
from core.calibration import bins2diagram
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, OUTCLASS, NTRAIN, NTEST
from methods.baselines.ivon import IVON
from models.uncertainty.swag import SWAG

try:
    from laplace import Laplace
except ImportError:
    Laplace = None


def get_args():
    p = argparse.ArgumentParser(description="Test (all standard-checkpoint models)")
    p.add_argument("traindir", type=str, help="directory containing seed= subdirectories")
    p.add_argument("dataset", type=str, choices=TRAINDATALOADERS)
    p.add_argument("-j", "--workers", default=1, type=int)
    p.add_argument("-b", "--batch", default=512, type=int)
    p.add_argument("-vd", "--valdata", action="store_true")
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float)
    p.add_argument("-d", "--device", default="cpu", type=str)
    p.add_argument("-s", "--seed", default=0, type=int)
    p.add_argument("-ss", "--seed_start", default=0, type=int)
    p.add_argument("-se", "--seed_end", default=4, type=int)
    p.add_argument("-pf", "--printfreq", default=10, type=int)
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    p.add_argument("-so", "--saveoutput", action="store_true")
    p.add_argument("-dd", "--data_dir", default="../data", type=str)
    p.add_argument("-nb", "--bins", default=20, type=int)
    p.add_argument("-pd", "--plotdiagram", action="store_true")
    p.add_argument("-tr", "--testrepeat", default=1, type=int,
                   help="Number of stochastic forward passes to average (e.g. 64 for MC Dropout).")
    p.add_argument("--checkpoint", default="best", choices=("best", "latest"))
    p.add_argument("-sms", "--swag_modelsamples", type=int, default=None)
    p.add_argument("-ssm", "--swag_samplemode", default="modelwise", choices=SWAG.sample_mode)
    p.add_argument("--pred_type", default="glm", choices=["glm", "nn"])
    p.add_argument("--link_approx", default="probit", choices=["mc", "probit", "bridge", "bridge_norm"])
    p.add_argument("--n_samples", default=100, type=int)
    return p.parse_args()


def get_dataloader(args, device):
    if args.valdata:
        _, data_loader = TRAINDATALOADERS[args.dataset](
            args.data_dir,
            args.tvsplit,
            args.workers,
            (device != torch.device("cpu")),
            args.batch,
            args.batch,
        )
    else:
        data_loader = TESTDATALOADER[args.dataset](
            args.data_dir,
            args.workers,
            (device != torch.device("cpu")),
            args.batch,
        )
    return data_loader


def _nll_from_prob(prob, gt):
    return nnf.nll_loss(prob.clamp_min(1e-12).log(), gt).item()


@torch.no_grad()
def do_evalbatch_ivon(batchinput, model, optimizer, repeat: int = 1):
    inputs, gt = batchinput[:-1], batchinput[-1]
    repeat = max(1, repeat)
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    cumloss = 0.0
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        cumprob = cumprob + nnf.softmax(output, 1) / repeat
        cumloss += nnf.nll_loss(nnf.log_softmax(output, 1), gt).item() / repeat
    return cumprob, gt, cumloss


@torch.no_grad()
def do_evalbatch_swag(batchinput, models):
    inputs = batchinput[:-1]
    gt = batchinput[-1]
    prob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for model in models:
        prob = prob + nnf.softmax(model(*inputs), 1) / len(models)
    return prob, gt, _nll_from_prob(prob, gt)


@torch.no_grad()
def do_evalbatch_laplace(batchinput, la, pred_type="glm", link_approx="probit", n_samples=100):
    inputs, gt = batchinput[:-1], batchinput[-1]
    kwargs = {"pred_type": pred_type}
    if pred_type == "glm":
        kwargs["link_approx"] = link_approx
    if link_approx == "mc" or pred_type == "nn":
        kwargs["n_samples"] = n_samples
    prob = la(inputs[0], **kwargs)
    return prob, gt, _nll_from_prob(prob, gt)


def run_eval(data_loader, model, optimizer, la, args, log_metrics, device):
    if la is not None:
        do_epoch(
            data_loader,
            do_evalbatch_laplace,
            log_metrics,
            device,
            la=la,
            pred_type=args.pred_type,
            link_approx=args.link_approx,
            n_samples=args.n_samples,
        )
    elif isinstance(optimizer, IVON) and args.testrepeat > 0:
        model.eval()
        do_epoch(
            data_loader,
            do_evalbatch_ivon,
            log_metrics,
            device,
            model=model,
            optimizer=optimizer,
            repeat=args.testrepeat,
        )
    elif isinstance(model, SWAG):
        nsamples = args.swag_modelsamples or max(1, args.testrepeat)
        sampled_models = [model.sampled_model(mode=args.swag_samplemode) for _ in range(nsamples)]
        for sampled_model in sampled_models:
            sampled_model.eval()
        do_epoch(data_loader, do_evalbatch_swag, log_metrics, device, models=sampled_models)
    else:
        model.eval()
        do_epoch(
            data_loader,
            do_evalbatch,
            log_metrics,
            device,
            model=model,
            repeat=max(1, args.testrepeat),
        )


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Test initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deterministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)

    log_metrics = coro_log_metrics(None, args.printfreq, args.bins, args.save_dir)
    prefix = "val" if args.valdata else "test"

    for seed in range(args.seed_start, args.seed_end + 1):
        runfolder = f"seed={seed}"
        seed_dir = pjoin(args.traindir, runfolder)
        checkpoint_name = "best_checkpoint.pt" if args.checkpoint == "best" else "checkpoint.pt"
        model_paths = sorted(glob(pjoin(seed_dir, "*", checkpoint_name)))

        if not model_paths:
            hint = " (use --checkpoint latest to load checkpoint.pt)" if args.checkpoint == "best" else ""
            print(f"skipping {seed_dir}: no {checkpoint_name} found{hint}\n")
            continue

        model_path = model_paths[-1]
        laplace_path = pjoin(os.path.dirname(model_path), "laplace_state.pt")
        print(f"loading model from {model_path} ...\n")
        _, model, optimizer, _, _ = loadcheckpoint(model_path, device)
        print(optimizer.defaults)

        la = None
        if exists(laplace_path):
            if Laplace is None:
                raise ImportError("laplace-torch is not installed, but laplace_state.pt was found.")
            lap_ckpt = torch.load(laplace_path, map_location=device)
            lap_cfg = lap_ckpt["config"]
            la = Laplace(
                model,
                likelihood="classification",
                subset_of_weights=lap_cfg["subset_of_weights"],
                hessian_structure=lap_cfg["hessian_structure"],
                prior_precision=lap_cfg["prior_precision"],
            )
            la.load_state_dict(lap_ckpt["state_dict"])

        data_loader = get_dataloader(args, device)
        dataset = args.dataset
        ndata = (
            NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
            if args.valdata
            else NTEST[dataset]
        )

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        outputsaver = (
            get_outputsaver(args.save_dir, ndata, OUTCLASS[dataset],
                            f"predictions_{prefix}_{runfolder}.npy")
            if args.saveoutput else None
        )

        log_metrics.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            run_eval(data_loader, model, optimizer, la, args, log_metrics, device)

        bins, _, avgvloss = log_metrics.throw(StopIteration)[:3]
        if args.saveoutput:
            outputsaver.close()
        del model
        if la is not None:
            del la

        if args.plotdiagram:
            bins2diagram(
                bins,
                False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))
    log_metrics.close()
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
