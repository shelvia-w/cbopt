"""Evaluate saved DUQ checkpoints and summarize their predictions."""

import argparse
import os
import sys
from os.path import join as pjoin
from glob import glob

import torch
import torch.nn.functional as nnf

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from core.coroutines import coro_timer
from core.logging import coro_log_metrics
from core.utils import check_cuda, deterministic_run, get_outputsaver, mkdirp, summarize_csv
from core.engine import do_epoch
from core.calibration import bins2diagram
from models import STANDARDMODELS
from models.uncertainty.duq import DUQModel
from data.dataloaders import TRAINDATALOADERS, TESTDATALOADER, OUTCLASS, NTRAIN, NTEST, INSIZE


def get_args():
    p = argparse.ArgumentParser(description="DUQ test")
    p.add_argument("traindir", type=str)
    p.add_argument("arch", type=str, choices=STANDARDMODELS)
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
    return p.parse_args()


def get_dataloader(args, device):
    if args.valdata:
        _, data_loader = TRAINDATALOADERS[args.dataset](
            args.data_dir, args.tvsplit, args.workers,
            (device != torch.device("cpu")), args.batch, args.batch,
        )
    else:
        data_loader = TESTDATALOADER[args.dataset](
            args.data_dir, args.workers, (device != torch.device("cpu")), args.batch,
        )
    return data_loader


def load_duq_checkpoint(model_path, args, device):
    ckpt = torch.load(model_path, map_location=device)
    base_model = STANDARDMODELS[args.arch](OUTCLASS[args.dataset], INSIZE[args.dataset])
    modelkwargs = ckpt["modelkwargs"]
    model = DUQModel(
        base_model=base_model,
        num_classes=modelkwargs["num_classes"],
        centroid_dim=modelkwargs["centroid_dim"],
        length_scale=modelkwargs["length_scale"],
        beta=modelkwargs["beta"],
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    return ckpt, model


@torch.no_grad()
def predict_proba_duq(batchinput, model):
    images, target = batchinput
    scores = model(images)
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    loss = nnf.nll_loss(prob.clamp_min(1e-12).log(), target)
    return prob, target, loss.item()


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
        model_paths = sorted(glob(pjoin(seed_dir, "*", "checkpoint.pt")))

        if not model_paths:
            print(f"skipping {seed_dir}\n")
            continue

        model_path = model_paths[-1]
        print(f"loading model from {model_path} ...\n")
        ckpt, model = load_duq_checkpoint(model_path, args, device)

        data_loader = get_dataloader(args, device)
        dataset = args.dataset
        ndata = (
            NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
            if args.valdata else NTEST[dataset]
        )

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        outputsaver = (
            get_outputsaver(args.save_dir, ndata, OUTCLASS[dataset],
                            f"predictions_{prefix}_{runfolder}.npy")
            if args.saveoutput else None
        )

        log_metrics.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            do_epoch(data_loader, predict_proba_duq, log_metrics, device, model=model)

        bins, _, avgvloss = log_metrics.throw(StopIteration)[:3]
        if args.saveoutput:
            outputsaver.close()
        del model

        if args.plotdiagram:
            bins2diagram(
                bins, False,
                pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"),
            )

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))
    log_metrics.close()
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")

