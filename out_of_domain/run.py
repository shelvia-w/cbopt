import argparse
from os.path import join as pjoin, exists
from glob import glob
import torch
import torch.nn as nn
import sys
import os

sys.path.append("..")
from common.models import STANDARDMODELS
from common.baselines.ivon import IVON
from common.vogn import VOGN
from common.baselines.swag import SWAG
from common.baselines.duq import DUQModel
from common.baselines.sngp import SNGPModel
from common.utils import coro_timer, mkdirp, coro_dict2csv
from common.trainutils import (
    do_epoch,
    check_cuda,
    deteministic_run,
    loadcheckpoint,
)
from common.dataloaders import get_cifar10_test_loader
from common.ood_utils import (
    get_svhn_loader,
    get_flowers102_loader,
    do_evalbatch,
    do_evalbatch_von,
    do_evalbatch_swag,
    do_evalbatch_duq,
    do_evalbatch_sngp,
    auroc,
    get_outputsaver,
    summarize_csv,
    coro_log,
    SVHNInfo,
    Flowers102Info,
    confidence_from_prediction_npy,
    OODMetrics,
)

try:
    from laplace import Laplace
except ImportError:
    Laplace = None


OODInfo = {
    "svhn": SVHNInfo,
    "flowers102": Flowers102Info,
}


def get_in_out_confs(test_folder: str, idx: int, starts_with="predictions"):
    in_name = pjoin(test_folder, f"{starts_with}_indomain_test_{idx}.npy")
    out_name = pjoin(test_folder, f"{starts_with}_ood_test_{idx}.npy")
    in_conf = confidence_from_prediction_npy(in_name)
    out_conf = confidence_from_prediction_npy(out_name)
    return in_conf, out_conf


def compute_and_save_metrics(test_folder: str, wamode: str = "", runs=()):
    starts_with = "predictions" if not wamode else f"predictions_{wamode}"
    csv_name = "metrics_test.csv" if not wamode else f"metrics_{wamode}_test.csv"
    csvcorolog = coro_dict2csv(
        pjoin(test_folder, csv_name), ("epoch",) + OODMetrics.metric_names
    )
    for e in runs:
        metrics = OODMetrics(*get_in_out_confs(test_folder, e, starts_with)).get_all()
        print(
            ", ".join(
                [f"epoch: {e}"] + [f"{k}: {v:.4f}" for k, v in metrics.items()]
            )
        )
        csvcorolog.send({"epoch": e, **metrics})


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("traindir", type=str, help="path that collects all trained runs.")
    parser.add_argument("--ood_dataset", default="svhn", choices=OODInfo)
    parser.add_argument("-j", "--workers", default=1, type=int, metavar="N", help="number of data loading workers")
    parser.add_argument("-b", "--batch", default=512, type=int, metavar="N", help="test mini-batch size")
    parser.add_argument("-ts", "--testsamples", default=1, type=int, help="create test samples via duplicating batch")
    parser.add_argument("-tr", "--testrepeat", default=1, type=int, help="create test samples via process repeat")
    parser.add_argument("-vd", "--valdata", action="store_true", help="use validation instead of test data")
    parser.add_argument("-pf", "--printfreq", default=10, type=int, metavar="N", help="print frequency")
    parser.add_argument("-d", "--device", default="cpu", type=str, metavar="DEV", help="run on cpu/cuda")
    parser.add_argument("-s", "--seed", type=int, default=0, help="fixes seed for reproducibility")
    parser.add_argument("-sd", "--save_dir", help="The directory used to save test results", default="save_temp", type=str)
    parser.add_argument("-so", "--saveoutput", action="store_true", help="save output probability")
    parser.add_argument("-dd", "--data_dir", help="The directory to find/store dataset", default="../data", type=str)

    # SWAG
    parser.add_argument("-sms", "--swag_modelsamples", type=int, default=1, help="number of swag model samples")
    parser.add_argument("-ssm", "--swag_samplemode", default="modelwise", choices=SWAG.sample_mode, help="specify at which level sampling will happen")

    # Laplace
    parser.add_argument("--pred_type", default="glm", choices=["glm", "nn", "linear_sampling", "mc"])
    parser.add_argument("--link_approx", default="probit", choices=["mc", "probit", "bridge", "bridge_norm"])
    parser.add_argument("--n_samples", default=100, type=int)

    return parser.parse_args()


def get_ood_loader(args):
    if args.ood_dataset == "svhn":
        return get_svhn_loader(
            args.data_dir,
            args.workers,
            (args.device != "cpu"),
            args.batch,
            "test",
            args.testsamples,
        )
    elif args.ood_dataset == "flowers102":
        return get_flowers102_loader(
            args.data_dir,
            args.workers,
            (args.device != "cpu"),
            args.batch,
            "test",
            args.testsamples,
        )


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(
            module,
            (
                nn.Dropout,
                nn.Dropout1d,
                nn.Dropout2d,
                nn.Dropout3d,
                nn.AlphaDropout,
            ),
        ):
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

    prob = la(x, **kwargs)
    return prob


if __name__ == "__main__":
    timer = coro_timer()
    t_init = next(timer)
    print(f">>> Test initiated at {t_init.isoformat()} <<<\n")

    args = get_args()
    print(args, end="\n\n")

    if args.seed is not None:
        deteministic_run(seed=args.seed)

    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()

    mkdirp(args.save_dir)
    log_ece = coro_log(None, args.printfreq, args.save_dir)

    indomain_prefix = "indomain_test"
    indomain_loader = get_cifar10_test_loader(
        args.data_dir,
        args.workers,
        (device != torch.device("cpu")),
        args.batch,
        args.testsamples,
    )
    ood_prefix = "ood_test"
    ood_loader = get_ood_loader(args)
    aucroc_scores = []
    valid_runs = []

    for runfolder in glob(f"{args.traindir}/seed=*/*"):
        save_name = os.path.relpath(runfolder, args.traindir).replace(os.sep, "_")
        model_path = pjoin(runfolder, "checkpoint.pt")
        laplace_path = pjoin(runfolder, "laplace_state.pt")

        if not exists(model_path):
            print(f"skipping {runfolder}\n")
            continue
        else:
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
                model,
                likelihood="classification",
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

        # in-domain
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                10000,
                outclass,
                f"predictions_{indomain_prefix}_{save_name}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, indomain_prefix, len(indomain_loader), outputsaver))
        with torch.no_grad():
            if la is not None:
                do_epoch(
                    indomain_loader,
                    do_evalbatch_laplace,
                    log_ece,
                    device,
                    la=la,
                    pred_type=args.pred_type,
                    link_approx=args.link_approx,
                    n_samples=args.n_samples,
                )
            elif isinstance(optimizer, IVON) or isinstance(optimizer, VOGN):
                model.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch_von,
                    log_ece,
                    device,
                    model=model,
                    optimizer=optimizer,
                    repeat=args.testrepeat,
                )
            elif isinstance(model, SWAG):
                sampledmodels = [
                    model.sampled_model(mode=args.swag_samplemode)
                    for _ in range(args.swag_modelsamples)
                ]
                for m in sampledmodels:
                    m.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch_swag,
                    log_ece,
                    device,
                    models=sampledmodels,
                )
            elif isinstance(model, DUQModel):
                model.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch_duq,
                    log_ece,
                    device,
                    model=model,
                )
            elif isinstance(model, SNGPModel):
                model.eval()
                if hasattr(model, "set_update_covariance"):
                    model.set_update_covariance(False)
                do_epoch(
                    indomain_loader,
                    do_evalbatch_sngp,
                    log_ece,
                    device,
                    model=model,
                )
            elif modelname == "resnet20_mcdrop":
                enable_mc_dropout(model)
                do_epoch(
                    indomain_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
            else:
                model.eval()
                do_epoch(
                    indomain_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        # ood
        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                OODInfo[args.ood_dataset].count["test"],
                outclass,
                f"predictions_{ood_prefix}_{save_name}.npy",
            )
        else:
            outputsaver = None

        log_ece.send((runfolder, ood_prefix, len(ood_loader), outputsaver))
        with torch.no_grad():
            if la is not None:
                do_epoch(
                    ood_loader,
                    do_evalbatch_laplace,
                    log_ece,
                    device,
                    la=la,
                    pred_type=args.pred_type,
                    link_approx=args.link_approx,
                    n_samples=args.n_samples,
                )
            elif isinstance(optimizer, IVON) or isinstance(optimizer, VOGN):
                model.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch_von,
                    log_ece,
                    device,
                    model=model,
                    optimizer=optimizer,
                    repeat=args.testrepeat,
                )
            elif isinstance(model, SWAG):
                sampledmodels = [
                    model.sampled_model(mode=args.swag_samplemode)
                    for _ in range(args.swag_modelsamples)
                ]
                for m in sampledmodels:
                    m.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch_swag,
                    log_ece,
                    device,
                    models=sampledmodels,
                )
            elif isinstance(model, DUQModel):
                model.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch_duq,
                    log_ece,
                    device,
                    model=model,
                )
            elif isinstance(model, SNGPModel):
                model.eval()
                if hasattr(model, "set_update_covariance"):
                    model.set_update_covariance(False)
                do_epoch(
                    ood_loader,
                    do_evalbatch_sngp,
                    log_ece,
                    device,
                    model=model,
                )
            elif modelname == "resnet20_mcdrop":
                enable_mc_dropout(model)
                do_epoch(
                    ood_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
            else:
                model.eval()
                do_epoch(
                    ood_loader,
                    do_evalbatch,
                    log_ece,
                    device,
                    model=model,
                    dups=args.testsamples,
                    repeat=args.testrepeat,
                )
        log_ece.throw(StopIteration)
        if args.saveoutput:
            outputsaver.close()

        if la is not None:
            del la
        else:
            del model

        indomain_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir, f"predictions_{indomain_prefix}_{save_name}.npy")
        )
        ood_conf = confidence_from_prediction_npy(
            pjoin(args.save_dir, f"predictions_{ood_prefix}_{save_name}.npy")
        )
        aucroc = auroc(indomain_conf, ood_conf)
        print(f"AUC-ROC score: {aucroc}")
        aucroc_scores.append(aucroc)

        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    compute_and_save_metrics(args.save_dir, "", valid_runs)
    summarize_csv(pjoin(args.save_dir, "metrics_test.csv"))

    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
    log_ece.close()