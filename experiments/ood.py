"""Run out-of-domain evaluation across saved uncertainty model checkpoints."""

import argparse
import os
import sys
from os.path import join as pjoin, exists
from glob import glob

import torch
import torch.nn as nn

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models import STANDARDMODELS
from methods.baselines.ivon import IVON
from methods.baselines.vogn import VOGN
from models.uncertainty.swag import SWAG
from models.uncertainty.duq import DUQModel
from models.uncertainty.sngp import SNGPModel
from training.checkpoint import loadcheckpoint
from training.coroutines import coro_dict2csv, coro_timer
from training.utils import check_cuda, deterministic_run, mkdirp
from training.engine import do_epoch
from training.evaluation import (
    do_evalbatch_ood, do_evalbatch_von, do_evalbatch_swag,
    do_evalbatch_duq, do_evalbatch_sngp,
    get_outputsaver, coro_log_ood, confidence_from_prediction_npy,
)
from data.dataloaders import get_cifar10_test_loader
from data.ood_utils import (
    get_svhn_loader, get_flowers102_loader, auroc, OODMetrics,
    SVHNInfo, Flowers102Info,
)

try:
    from laplace import Laplace
except ImportError:
    Laplace = None


OODInfo = {
    "svhn": SVHNInfo,
    "flowers102": Flowers102Info,
}


def get_in_out_confs(test_folder: str, idx, starts_with="predictions"):
    in_name = pjoin(test_folder, f"{starts_with}_indomain_test_{idx}.npy")
    out_name = pjoin(test_folder, f"{starts_with}_ood_test_{idx}.npy")
    return confidence_from_prediction_npy(in_name), confidence_from_prediction_npy(out_name)


def compute_and_save_metrics(test_folder: str, wamode: str = "", runs=()):
    starts_with = "predictions" if not wamode else f"predictions_{wamode}"
    csv_name = "metrics_test.csv" if not wamode else f"metrics_{wamode}_test.csv"
    csvcorolog = coro_dict2csv(
        pjoin(test_folder, csv_name), ("epoch",) + OODMetrics.metric_names
    )
    for e in runs:
        metrics = OODMetrics(*get_in_out_confs(test_folder, e, starts_with)).get_all()
        print(", ".join([f"epoch: {e}"] + [f"{k}: {v:.4f}" for k, v in metrics.items()]))
        csvcorolog.send({"epoch": e, **metrics})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("traindir", type=str)
    parser.add_argument("--ood_dataset", default="svhn", choices=OODInfo)
    parser.add_argument("-j", "--workers", default=1, type=int)
    parser.add_argument("-b", "--batch", default=512, type=int)
    parser.add_argument("-ts", "--testsamples", default=1, type=int)
    parser.add_argument("-tr", "--testrepeat", default=1, type=int)
    parser.add_argument("-vd", "--valdata", action="store_true")
    parser.add_argument("-pf", "--printfreq", default=10, type=int)
    parser.add_argument("-d", "--device", default="cpu", type=str)
    parser.add_argument("-s", "--seed", type=int, default=0)
    parser.add_argument("-sd", "--save_dir", default="save_temp", type=str)
    parser.add_argument("-so", "--saveoutput", action="store_true")
    parser.add_argument("-dd", "--data_dir", default="../data", type=str)
    parser.add_argument("-sms", "--swag_modelsamples", type=int, default=1)
    parser.add_argument("-ssm", "--swag_samplemode", default="modelwise", choices=SWAG.sample_mode)
    parser.add_argument("--pred_type", default="glm", choices=["glm", "nn", "linear_sampling", "mc"])
    parser.add_argument("--link_approx", default="probit", choices=["mc", "probit", "bridge", "bridge_norm"])
    parser.add_argument("--n_samples", default=100, type=int)
    return parser.parse_args()


def get_ood_loader(args):
    if args.ood_dataset == "svhn":
        return get_svhn_loader(args.data_dir, args.workers, (args.device != "cpu"), args.batch, "test", args.testsamples)
    elif args.ood_dataset == "flowers102":
        return get_flowers102_loader(args.data_dir, args.workers, (args.device != "cpu"), args.batch, "test", args.testsamples)


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            module.train()


@torch.no_grad()
def do_evalbatch_laplace(batchinput, la, pred_type="glm", link_approx="probit", n_samples=100):
    inputs = batchinput[:-1]
    x = inputs[0]
    kwargs = {"pred_type": pred_type}
    if pred_type in {"glm", "mc"}:
        kwargs["link_approx"] = link_approx
    if pred_type in {"mc", "linear_sampling"}:
        kwargs["n_samples"] = n_samples
    return la(x, **kwargs)


def _run_eval(loader, model, optimizer, la, modelname, args, log_ece, device):
    if la is not None:
        do_epoch(loader, do_evalbatch_laplace, log_ece, device,
                 la=la, pred_type=args.pred_type, link_approx=args.link_approx, n_samples=args.n_samples)
    elif isinstance(optimizer, (IVON, VOGN)):
        model.eval()
        do_epoch(loader, do_evalbatch_von, log_ece, device, model=model, optimizer=optimizer, repeat=args.testrepeat)
    elif isinstance(model, SWAG):
        sampledmodels = [model.sampled_model(mode=args.swag_samplemode) for _ in range(args.swag_modelsamples)]
        for m in sampledmodels:
            m.eval()
        do_epoch(loader, do_evalbatch_swag, log_ece, device, models=sampledmodels)
    elif isinstance(model, DUQModel):
        model.eval()
        do_epoch(loader, do_evalbatch_duq, log_ece, device, model=model)
    elif isinstance(model, SNGPModel):
        model.eval()
        if hasattr(model, "set_update_covariance"):
            model.set_update_covariance(False)
        do_epoch(loader, do_evalbatch_sngp, log_ece, device, model=model)
    elif modelname == "resnet20_mcdrop":
        enable_mc_dropout(model)
        do_epoch(loader, do_evalbatch_ood, log_ece, device, model=model, dups=args.testsamples, repeat=args.testrepeat)
    else:
        model.eval()
        do_epoch(loader, do_evalbatch_ood, log_ece, device, model=model, dups=args.testsamples, repeat=args.testrepeat)


if __name__ == "__main__":
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
    log_ece = coro_log_ood(None, args.printfreq, args.save_dir)

    indomain_loader = get_cifar10_test_loader(
        args.data_dir, args.workers, (device != torch.device("cpu")), args.batch, args.testsamples,
    )
    ood_loader = get_ood_loader(args)
    valid_runs = []

    for runfolder in glob(f"{args.traindir}/seed=*/*"):
        save_name = os.path.relpath(runfolder, args.traindir).replace(os.sep, "_")
        model_path = pjoin(runfolder, "checkpoint.pt")
        laplace_path = pjoin(runfolder, "laplace_state.pt")

        if not exists(model_path):
            print(f"skipping {runfolder}\n")
            continue
        valid_runs.append(save_name)

        ckpt = torch.load(model_path, map_location=device)
        la = None
        modelname = ckpt.get("modelname", "")

        if exists(laplace_path):
            if Laplace is None:
                raise ImportError("laplace-torch is not installed, but laplace_state.pt was found.")
            _, model, optimizer, _, ddat = loadcheckpoint(model_path, device)
            lap_ckpt = torch.load(laplace_path, map_location=device)
            lap_cfg = lap_ckpt["config"]
            la = Laplace(
                model, likelihood="classification",
                subset_of_weights=lap_cfg["subset_of_weights"],
                hessian_structure=lap_cfg["hessian_structure"],
                prior_precision=lap_cfg["prior_precision"],
            )
            la.load_state_dict(lap_ckpt["state_dict"])
            model.eval()
        elif "modelname" in ckpt:
            _, model, optimizer, _, ddat = loadcheckpoint(model_path, device)
            modelname = ddat.get("modelname", modelname)
        elif "model_state_dict" in ckpt and "optimizer_state_dict" in ckpt:
            base_model = STANDARDMODELS["resnet20"](10, 32)
            model = DUQModel(
                base_model=base_model,
                num_classes=ckpt["modelkwargs"]["num_classes"],
                centroid_dim=ckpt["modelkwargs"]["centroid_dim"],
                length_scale=ckpt["modelkwargs"]["length_scale"],
                beta=ckpt["modelkwargs"]["beta"],
            ).to(device)
            model.load_state_dict(ckpt["model_state_dict"])
            optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
            ddat = ckpt
            modelname = ckpt.get("modelname", "")
        else:
            raise ValueError(f"Unknown checkpoint format: {model_path}")

        if la is None and hasattr(optimizer, "mc_samples"):
            optimizer.mc_samples = args.testrepeat

        outclass = 10
        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        # in-domain pass
        outputsaver = (
            get_outputsaver(args.save_dir, 10000, outclass, f"predictions_indomain_test_{save_name}.npy")
            if args.saveoutput else None
        )
        log_ece.send((runfolder, "indomain_test", len(indomain_loader), outputsaver))
        with torch.no_grad():
            _run_eval(indomain_loader, model if la is None else None, optimizer if la is None else None, la, modelname, args, log_ece, device)
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        # OOD pass
        outputsaver = (
            get_outputsaver(args.save_dir, OODInfo[args.ood_dataset].count["test"], outclass,
                            f"predictions_ood_test_{save_name}.npy")
            if args.saveoutput else None
        )
        log_ece.send((runfolder, "ood_test", len(ood_loader), outputsaver))
        with torch.no_grad():
            _run_eval(ood_loader, model if la is None else None, optimizer if la is None else None, la, modelname, args, log_ece, device)
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        if la is not None:
            del la
        else:
            del model

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir, f"predictions_indomain_test_{save_name}.npy")
        )
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir, f"predictions_ood_test_{save_name}.npy")
        )
        print(f"AUC-ROC score: {auroc(indomain_conf, ood_conf):.4f}")
        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    compute_and_save_metrics(args.save_dir, "", valid_runs)

    from training.utils import summarize_csv
    summarize_csv(pjoin(args.save_dir, "metrics_test.csv"))
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
    log_ece.close()
