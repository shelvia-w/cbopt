import argparse
from os.path import join as pjoin
from glob import glob
import torch
import sys

sys.path.append("..")
from common.utils import coro_timer, mkdirp
from common.calibration import bins2diagram
from common.dataloaders import TRAINDATALOADERS, TESTDATALOADER, OUTCLASS, NTRAIN, NTEST
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
    p = argparse.ArgumentParser(description="SNGP test")
    p.add_argument("traindir", type=str)
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
            args.data_dir, args.tvsplit, args.workers, (device != torch.device("cpu")), args.batch, args.batch
        )
    else:
        data_loader = TESTDATALOADER[args.dataset](
            args.data_dir, args.workers, (device != torch.device("cpu")), args.batch
        )
    return data_loader

def do_evalbatch_sngp(batchinput, model):
    images, target = batchinput
    logits = model(images, mean_field=True, return_gp_cov=False)
    prob = torch.softmax(logits, dim=1)
    loss = torch.nn.functional.cross_entropy(logits, target)
    return prob, target, loss.item()

if __name__ == "__main__":
    timer = coro_timer()
    print(f">>> Test initiated at {next(timer).isoformat()} <<<\n")
    args = get_args()
    print(args, end="\n\n")
    if args.seed is not None:
        deteministic_run(seed=args.seed)
    device = torch.device(args.device)
    if device != torch.device("cpu"):
        check_cuda()
    mkdirp(args.save_dir)
    log_metrics = coro_log_metrics(None, args.printfreq, args.bins, args.save_dir)
    prefix = "val" if args.valdata else "test"

    for seed in range(args.seed_start, args.seed_end + 1):
        seed_dir = pjoin(args.traindir, f"seed={seed}")
        model_paths = sorted(glob(pjoin(seed_dir, "*", "checkpoint.pt")))
        if not model_paths:
            print(f"skipping {seed_dir}\n")
            continue
        model_path = model_paths[-1]
        print(f"loading model from {model_path} ...\n")
        _, model, optimizer = loadcheckpoint(model_path, device)[:3]
        print(optimizer.defaults)
        data_loader = get_dataloader(args, device)
        dataset = args.dataset
        ndata = NTRAIN[dataset] - int(args.tvsplit * NTRAIN[dataset]) if args.valdata else NTEST[dataset]
        print(f">>> Test starts at {next(timer)[0].isoformat()} <<<\n")

        outputsaver = None
        runfolder = f"seed={seed}"
        if args.saveoutput:
            outputsaver = get_outputsaver(args.save_dir, ndata, OUTCLASS[dataset], f"predictions_{prefix}_{runfolder}.npy")

        log_metrics.send((runfolder, prefix, len(data_loader), outputsaver))
        with torch.no_grad():
            model.eval()
            if hasattr(model, "set_update_covariance"):
                model.set_update_covariance(False)
            do_epoch(data_loader, do_evalbatch_sngp, log_metrics, device, model=model)
        bins, _, _ = log_metrics.throw(StopIteration)[:3]
        if outputsaver is not None:
            outputsaver.close()
        del model

        if args.plotdiagram:
            bins2diagram(bins, False, pjoin(args.save_dir, f"calibration_{prefix}_{runfolder}.pdf"))
        print(f">>> Time elapsed: {next(timer)[1]} <<<\n")

    summarize_csv(pjoin(args.save_dir, f"{prefix}.csv"))
    log_metrics.close()
    print(f">>> Test completed at {next(timer)[0].isoformat()} <<<\n")
