import argparse
from glob import glob
from os.path import join as pjoin
import sys

import torch
import torch.nn.functional as nnf
from laplace import Laplace

sys.path.append("..")

from common.utils import coro_timer, mkdirp
from common.calibration import bins2diagram
from common.dataloaders import (
    TRAINDATALOADERS,
    TESTDATALOADER,
    OUTCLASS,
    NTRAIN,
    NTEST,
)
from common.trainutils import (
    coro_log_metrics,
    do_epoch,
    check_cuda,
    deteministic_run,
    summarize_csv,
    get_outputsaver,
    loadcheckpoint,
)


def get_args():
    p = argparse.ArgumentParser(description="uCBOpt + Laplace test")
    p.add_argument("traindir", type=str, help="path that collects all trained runs")
    p.add_argument("dataset", type=str, choices=TRAINDATALOADERS, help="dataset")
    p.add_argument("-j", "--workers", default=1, type=int, help="data loader workers")
    p.add_argument("-b", "--batch", default=512, type=int, help="test batch size")
    p.add_argument("-vd", "--valdata", action="store_true", help="use validation instead of test data")
    p.add_argument("-sp", "--tvsplit", default=0.9, type=float, help="train split ratio")
    p.add_argument("-d", "--device", default="cpu", type=str, help="cpu/cuda")
    p.add_argument("-s", "--seed", default=0, type=int, help="seed for reproducibility")
    p.add_argument("-ss", "--seed_start", default=0, type=int, help="start index of seeds to test")
    p.add_argument("-se", "--seed_end", default=4, type=int, help="end index of seeds to test")

    p.add_argument("-pf", "--printfreq", default=10, type=int, help="print frequency")
    p.add_argument("-sd", "--save_dir", default="save_temp", type=str, help="directory used to save test results")
    p.add_argument("-so", "--saveoutput", action="store_true", help="save output probability")
    p.add_argument("-dd", "--data_dir", default="../data", type=str, help="directory to find/store dataset")
    p.add_argument("-nb", "--bins", default=20, type=int, help="number of bins for ece & reliability diagram")
    p.add_argument("-pd", "--plotdiagram", action="store_true", help="plot reliability diagram")

    # Optional overrides for predictive settings stored with laplace_state.pt
    p.add_argument("--pred_type", default="", type=str, help="override pred_type stored with Laplace state")
    p.add_argument("--link_approx", default="", type=str, help="override link_approx stored with Laplace state")
    p.add_argument("--n_samples", default=-1, type=int, help="override n_samples stored with Laplace state")

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


def build_laplace(model, laplace_ckpt, device):
    cfg = laplace_ckpt["config"]
    la = Laplace(
        model,
        likelihood=cfg.get("likelihood", "classification"),
        subset_of_weights=cfg["subset_of_weights"],
        hessian_structure=cfg["hessian_structure"],
        prior_precision=cfg.get("prior_precision", 1.0),
    )
    la.load_state_dict(laplace_ckpt["state_dict"])
    if device.type != "cpu":
        la.model.to(device)
    la.model.eval()
    return la


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")

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
    log_metrics = coro_log_metrics(None, args.printfreq, args.bins, args.save_dir)
    prefix = "val_laplace" if args.valdata else "test_laplace"

    for seed in range(args.seed_start, args.seed_end + 1):
        runfolder = f"seed={seed}"
        seed_dir = pjoin(args.traindir, runfolder)
        model_paths = sorted(glob(pjoin(seed_dir, "*", "checkpoint.pt")))
        laplace_paths = sorted(glob(pjoin(seed_dir, "*", "laplace_state.pt")))

        if not model_paths or not laplace_paths:
            print(f"skipping {seed_dir} (missing checkpoint.pt or laplace_state.pt)\n")
            continue

        model_path = model_paths[-1]
        laplace_path = laplace_paths[-1]

        print(f"loading model from {model_path} ...\n")
        _, model, _ = loadcheckpoint(model_path, device)[:3]

        print(f"loading Laplace state from {laplace_path} ...\n")
        laplace_ckpt = torch.load(laplace_path, map_location=device)
        la = build_laplace(model, laplace_ckpt, device)

        cfg = laplace_ckpt["config"]
        pred_type = args.pred_type or cfg.get("pred_type", "glm")
        link_approx = args.link_approx or cfg.get("link_approx", "probit")
        n_samples = args.n_samples if args.n_samples > 0 else cfg.get("n_samples", 100)

        data_loader = get_dataloader(args, device)
        dataset = args.dataset
        ndata = (
            NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset])
            if args.valdata
            else NTEST[dataset]
        )

        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        if args.saveoutput:
            outputsaver = get_outputsaver(
                args.save_dir,
                ndata,
                OUTCLASS[dataset],
                f"predictions_{prefix}_{runfolder}.npy",
            )
        else:
            outputsaver = None

        eval_laplace = LaplaceEvalWrapper(
            la,
            pred_type=pred_type,
            link_approx=link_approx,
            n_samples=n_samples,
        )

        log_metrics.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            do_epoch(data_loader, eval_laplace, log_metrics, device)

        bins, _, avgvloss = log_metrics.throw(StopIteration)[:3]
        if args.saveoutput:
            outputsaver.close()

        del la
        del model

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
