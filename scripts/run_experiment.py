"""Run one configured training and evaluation experiment."""

import argparse
import csv
import os
import sys
from os.path import join as pjoin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from core.checkpoint import savecheckpoint
from core.engine import do_epoch, do_evalbatch, do_trainbatch
from core.evaluation import predict_proba_batch, predict_proba_swag, predict_proba_von
from core.logging import coro_log_timed
from core.ood import coro_log_ood
from core.utils import check_cuda, deterministic_run, mkdirp
from datasets.dataloaders import INSIZE, OUTCLASS, TESTDATALOADER, TRAINDATALOADERS
from datasets.ood_utils import OODMetrics, get_kmnist_loader, get_svhn_loader, get_tinyimagenet_ood_loader
from methods.lcbopt_adaptcurv import lCBOptAdaptCurv
from methods.ucbopt import uCBOpt
from methods.ucbopt_adaptcurv import uCBOptAdaptCurv
from methods.baselines.ivon import IVON
from models import MCDROPMODELS, MODELS, STANDARDMODELS, SWAGMODELS
from models.uncertainty.swag import SWAG

try:
    from laplace import Laplace
except ImportError:
    Laplace = None


METHODS = {
    "sgd",
    "adamw",
    "ivon",
    "mcdrop",
    "laplace",
    "swag",
    "ucbopt",
    "ucbopt_adapt",
    "lcbopt_adapt",
}

OOD_LOADERS = {
    "kmnist": get_kmnist_loader,
    "svhn": get_svhn_loader,
    "tinyimagenet": get_tinyimagenet_ood_loader,
}


def load_config(path):
    with open(path, "r") as f:
        cfg = yaml.safe_load(f)
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("workers", 0)
    cfg.setdefault("device", "cpu")
    cfg.setdefault("train_val_split", 0.9)
    cfg.setdefault("eval_batch_size", cfg["batch_size"])
    cfg.setdefault("random_seed", 0)
    cfg.setdefault("evaluation", {"indomain": True, "ood": True})
    cfg.setdefault("method_hparams", {})
    return cfg


def model_name_for_method(model_name, method):
    if method == "mcdrop":
        return f"{model_name}_mcdrop"
    if method == "swag":
        return f"{model_name}_swag"
    return model_name


def build_model(cfg, device):
    method = cfg["method"]
    name = model_name_for_method(cfg["model"], method)
    registry = MODELS if method in {"mcdrop", "swag"} else STANDARDMODELS
    if name not in registry:
        raise ValueError(f"Unknown model/method pair: model={cfg['model']}, method={method}")
    kwargs = dict(cfg.get("model_hparams", {}))
    if method == "mcdrop":
        kwargs["p"] = cfg["method_hparams"].get("dropout_p", kwargs.get("p", 0.05))
    if method == "swag":
        kwargs["max_rank"] = cfg["method_hparams"].get("max_rank", kwargs.get("max_rank", 20))
    model = registry[name](OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]], **kwargs).to(device)
    return name, model


def build_optimizer(cfg, model):
    hp = cfg["method_hparams"]
    method = cfg["method"]
    lr = cfg["learning_rate"]
    wd = cfg["weight_decay"]
    if method in {"sgd", "mcdrop", "swag", "laplace"}:
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=hp.get("momentum", 0.9),
            weight_decay=wd,
        )
    if method == "adamw":
        return torch.optim.AdamW(
            model.parameters(),
            lr=lr,
            betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.999)),
            weight_decay=wd,
        )
    if method == "ivon":
        return IVON(
            model.parameters(),
            lr=lr,
            mc_samples=hp.get("mc_samples", 1),
            beta1=hp.get("beta1", 0.9),
            beta2=hp.get("beta2", 0.99999),
            weight_decay=wd,
            hess_approx=hp.get("hess_approx", "price"),
            hess_init=hp.get("hess_init", 0.5),
            ess=hp.get("ess", 50000.0),
            clip_radius=hp.get("clip_radius", float("inf")),
            rescale_lr=hp.get("rescale_lr", False),
        )
    if method == "ucbopt":
        return uCBOpt(
            model.parameters(),
            lr=lr,
            beta1=hp.get("beta1", 0.9),
            beta2=hp.get("beta2", 0.99999),
            hess_init=hp.get("hess_init", 0.5),
            weight_decay=wd,
            cand_curvature=hp.get("cand_curvature", 0.0),
            rescale_lr=hp.get("rescale_lr", False),
        )
    if method == "ucbopt_adapt":
        return uCBOptAdaptCurv(
            model.parameters(),
            lr=lr,
            betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.999), hp.get("beta3", 1.001)),
            weight_decay=wd,
            hess_init=hp.get("hess_init", 0.5),
            gamma=hp.get("gamma", 0.1),
            eps=hp.get("eps", 1e-8),
            clip_radius=hp.get("clip_radius", float("inf")),
            rescale_lr=hp.get("rescale_lr", False),
        )
    if method == "lcbopt_adapt":
        return lCBOptAdaptCurv(
            model.parameters(),
            lr=lr,
            betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.99999), hp.get("beta3", 0.999)),
            weight_decay=wd,
            hess_init=hp.get("hess_init", 1.0),
            gamma=hp.get("gamma", 1.05),
            eps=hp.get("eps", 1e-6),
            clip_radius=hp.get("clip_radius", float("inf")),
            rescale_lr=hp.get("rescale_lr", False),
        )
    raise ValueError(f"Unknown method: {method}")


def build_scheduler(cfg, optimizer):
    warmup = cfg.get("warmup_epochs", 5)
    epochs = cfg["epochs"]
    lr_final = cfg.get("lr_final", 0.0)
    if warmup <= 0:
        return torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=lr_final, T_max=epochs)
    warmup_sched = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup
    )
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, eta_min=lr_final, T_max=max(epochs - warmup, 1)
    )
    return torch.optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine], milestones=[warmup]
    )


def trainbatch_ivon(batchinput, model, optimizer, mc_samples=1):
    images, target = batchinput
    losses = []
    probs = []
    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            loss.backward()
        losses.append(loss.detach())
        probs.append(nnf.softmax(output.detach(), dim=1))
    optimizer.step()
    return torch.stack(probs).mean(0), target, torch.stack(losses).mean().item()


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            module.train()


def train(cfg, model_name, model, optimizer, scheduler, train_loader, val_loader, test_loader, device):
    save_dir = cfg["output_dir"]
    mkdirp(save_dir)
    log = coro_log_timed(None, cfg.get("print_freq", 100), cfg.get("bins", 20), save_dir)
    best_val_loss = float("inf")
    best_path = pjoin(save_dir, "best_checkpoint.pt")
    method = cfg["method"]
    hp = cfg["method_hparams"]

    def train_batch(batchinput, model, optimizer):
        if method == "ivon":
            return trainbatch_ivon(batchinput, model, optimizer, hp.get("mc_samples", 1))
        return do_trainbatch(batchinput, model, optimizer)

    for epoch in range(cfg["epochs"]):
        model.train()
        log.send((epoch, "train", len(train_loader), None))
        do_epoch(train_loader, train_batch, log, device, model=model, optimizer=optimizer)
        log.throw(StopIteration)

        if method == "swag" and epoch >= hp.get("swag_start", max(0, cfg["epochs"] - 40)):
            freq = hp.get("swag_freq", 1)
            if (epoch - hp.get("swag_start", max(0, cfg["epochs"] - 40))) % freq == 0:
                model.collect_model()

        if scheduler is not None and not (method == "swag" and epoch >= hp.get("swag_start", max(0, cfg["epochs"] - 40))):
            scheduler.step()

        savecheckpoint(
            pjoin(save_dir, "checkpoint.pt"),
            model_name,
            (OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]]),
            cfg.get("model_hparams", {}),
            model,
            optimizer,
            scheduler,
            epoch=epoch,
        )

        if len(val_loader) > 0:
            log.send((epoch, "val", len(val_loader), None))
            with torch.no_grad():
                model.eval()
                metrics = do_epoch(val_loader, do_evalbatch, log, device, model=model)
            log.throw(StopIteration)
            _, val_loss, *_ = metrics
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                savecheckpoint(
                    best_path,
                    model_name,
                    (OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]]),
                    cfg.get("model_hparams", {}),
                    model,
                    optimizer,
                    scheduler,
                    epoch=epoch,
                    best_val_loss=best_val_loss,
                )
        else:
            log.send((epoch, "test", len(test_loader), None))
            with torch.no_grad():
                model.eval()
                do_epoch(test_loader, do_evalbatch, log, device, model=model)
            log.throw(StopIteration)

    if method == "swag":
        savecheckpoint(
            best_path,
            model_name,
            (OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]]),
            cfg.get("model_hparams", {}),
            model,
            optimizer,
            scheduler,
            epoch=cfg["epochs"] - 1,
        )
    log.close()


def fit_laplace(cfg, model, train_loader, val_loader):
    if Laplace is None:
        raise ImportError("laplace-torch is required for method='laplace'")
    hp = cfg["method_hparams"]
    la = Laplace(
        model,
        likelihood="classification",
        subset_of_weights=hp.get("subset_of_weights", "last_layer"),
        hessian_structure=hp.get("hessian_structure", "kron"),
        prior_precision=hp.get("prior_precision", 1.0),
    )
    la.fit(train_loader)
    if hp.get("optimize_prior_precision", True):
        la.optimize_prior_precision(
            method=hp.get("prior_precision_method", "marglik"),
            pred_type=hp.get("pred_type", "glm"),
            link_approx=hp.get("link_approx", "probit"),
            val_loader=val_loader,
        )
    torch.save(
        {
            "state_dict": la.state_dict(),
            "config": {
                "subset_of_weights": hp.get("subset_of_weights", "last_layer"),
                "hessian_structure": hp.get("hessian_structure", "kron"),
                "prior_precision": float(la.prior_precision.detach().cpu().item())
                if torch.is_tensor(la.prior_precision)
                else float(la.prior_precision),
            },
        },
        pjoin(cfg["output_dir"], "laplace_state.pt"),
    )
    return la


@torch.no_grad()
def collect_predictions(loader, cfg, model, optimizer, device, la=None):
    log = coro_log_ood(None, cfg.get("print_freq", 100), "")
    log.send((0, "eval", len(loader), None))
    hp = cfg["method_hparams"]
    method = cfg["method"]
    if la is not None:
        def eval_laplace(batchinput, la):
            x = batchinput[0]
            kwargs = {"pred_type": hp.get("pred_type", "glm")}
            if kwargs["pred_type"] == "glm":
                kwargs["link_approx"] = hp.get("link_approx", "probit")
            return la(x, **kwargs)

        do_epoch(loader, eval_laplace, log, device, la=la)
    elif isinstance(optimizer, IVON):
        do_epoch(loader, predict_proba_von, log, device, model=model, optimizer=optimizer, repeat=hp.get("test_repeat", 1))
    elif isinstance(model, SWAG):
        sampled = [model.sampled_model(mode=hp.get("sample_mode", "modelwise")).to(device) for _ in range(hp.get("model_samples", 1))]
        for sampled_model in sampled:
            sampled_model.eval()
        do_epoch(loader, predict_proba_swag, log, device, models=sampled)
    elif method == "mcdrop":
        enable_mc_dropout(model)
        do_epoch(loader, predict_proba_batch, log, device, model=model, repeat=hp.get("test_repeat", 10))
    else:
        model.eval()
        do_epoch(loader, predict_proba_batch, log, device, model=model)
    log.throw(StopIteration)
    log.close()

    predictions = []
    for images, _ in loader:
        images = images.to(device, non_blocking=True)
        if la is not None:
            prob = la(images, pred_type=hp.get("pred_type", "glm"))
        elif isinstance(optimizer, IVON):
            probs = []
            for _ in range(hp.get("test_repeat", 1)):
                with optimizer.sampled_params():
                    probs.append(torch.softmax(model(images), dim=1))
            prob = torch.stack(probs).mean(0)
        elif isinstance(model, SWAG):
            sampled = [model.sampled_model(mode=hp.get("sample_mode", "modelwise")).to(device).eval() for _ in range(hp.get("model_samples", 1))]
            prob = torch.stack([torch.softmax(m(images), dim=1) for m in sampled]).mean(0)
        else:
            if method == "mcdrop":
                enable_mc_dropout(model)
            prob = torch.softmax(model(images), dim=1)
        predictions.append(prob.cpu().numpy())
    return np.concatenate(predictions, axis=0)


def evaluate(cfg, model, optimizer, test_loader, device, la=None):
    eval_cfg = cfg["evaluation"]
    rows = []
    if eval_cfg.get("indomain", True):
        probs = collect_predictions(test_loader, cfg, model, optimizer, device, la)
        np.save(pjoin(cfg["output_dir"], "predictions_indomain.npy"), probs)
        confidence = probs.max(axis=1)
        rows.append({"split": "indomain", "mean_confidence": float(confidence.mean())})
    if eval_cfg.get("ood", True):
        ood_name = eval_cfg["ood_dataset"]
        loader = OOD_LOADERS[ood_name](
            cfg["data_dir"],
            cfg["workers"],
            device.type != "cpu",
            cfg["eval_batch_size"],
            "test",
            cfg["method_hparams"].get("test_dups", 1),
        )
        in_probs = np.load(pjoin(cfg["output_dir"], "predictions_indomain.npy"))
        out_probs = collect_predictions(loader, cfg, model, optimizer, device, la)
        np.save(pjoin(cfg["output_dir"], "predictions_ood.npy"), out_probs)
        metrics = OODMetrics(in_probs.max(axis=1), out_probs.max(axis=1)).get_all()
        rows.append({"split": f"ood:{ood_name}", **metrics})
    with open(pjoin(cfg["output_dir"], "metrics.csv"), "w", newline="") as f:
        keys = sorted({k for row in rows for k in row})
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["method"] not in METHODS:
        raise ValueError(f"Unknown method: {cfg['method']}")
    if cfg["random_seed"] is not None:
        deterministic_run(cfg["random_seed"])
    device = torch.device(cfg["device"])
    if device.type != "cpu":
        check_cuda()
    mkdirp(cfg["output_dir"])

    train_loader, val_loader = TRAINDATALOADERS[cfg["dataset"]](
        cfg["data_dir"],
        cfg["train_val_split"],
        cfg["workers"],
        device.type != "cpu",
        cfg["batch_size"],
        cfg["eval_batch_size"],
    )
    test_loader = TESTDATALOADER[cfg["dataset"]](
        cfg["data_dir"], cfg["workers"], device.type != "cpu", cfg["eval_batch_size"]
    )

    model_name, model = build_model(cfg, device)
    optimizer = build_optimizer(cfg, model)
    scheduler = build_scheduler(cfg, optimizer)
    train(cfg, model_name, model, optimizer, scheduler, train_loader, val_loader, test_loader, device)
    la = fit_laplace(cfg, model, train_loader, val_loader) if cfg["method"] == "laplace" else None
    evaluate(cfg, model, optimizer, test_loader, device, la)


if __name__ == "__main__":
    main()
