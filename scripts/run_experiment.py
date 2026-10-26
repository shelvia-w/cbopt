"""Train and evaluate the image classification experiments."""

import argparse
import copy
import csv
import json
import math
import platform
import re
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as nnf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.checkpoint import loadcheckpoint, savecheckpoint
from core.engine import do_epoch, do_evalbatch, do_trainbatch
from core.logging import coro_log_timed
from core.metrics import classification_metrics
from core.utils import check_cuda, deterministic_run
from datasets.dataloaders import GENERATOR, INSIZE, OUTCLASS, TESTDATALOADER, TRAINDATALOADERS
from datasets.ood_utils import OODMetrics, get_emnist_loader, get_svhn_loader, get_tinyimagenet_ood_loader
from methods.baselines.ivon import IVON
from methods.lcbopt_adaptcurv import lCBOptAdaptCurv
from methods.ucbopt import uCBOpt
from methods.ucbopt_adaptcurv import uCBOptAdaptCurv
from models import MODELS
from models.uncertainty.swag import SWAG

METHODS = {"sgd", "adamw", "ivon", "mcdrop", "laplace", "swag", "ucbopt", "ucbopt_adapt", "lcbopt_adapt"}
OOD_LOADERS = {"emnist": get_emnist_loader, "svhn": get_svhn_loader, "tinyimagenet": get_tinyimagenet_ood_loader}


def numeric_values(value):
    """Convert numeric strings left by PyYAML, including exponent notation."""
    if isinstance(value, dict):
        return {k: numeric_values(v) for k, v in value.items()}
    if isinstance(value, list):
        return [numeric_values(v) for v in value]
    if isinstance(value, str) and re.fullmatch(r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?|[+-]?inf", value):
        return float(value)
    return value


def load_config(path):
    with open(path, encoding="utf-8") as f:
        cfg = numeric_values(yaml.safe_load(f))
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("workers", 0)
    cfg.setdefault("eval_workers", 0)
    cfg.setdefault("device", "cuda")
    cfg.setdefault("train_val_split", 0.9)
    cfg.setdefault("eval_batch_size", 256)
    cfg.setdefault("random_seed", 0)
    cfg.setdefault("warmup_epochs", 5)
    cfg.setdefault("lr_final", 0.0)
    cfg.setdefault("method_hparams", {})
    cfg.setdefault("experiment", f"{Path(path).parent.name}/{Path(path).stem}")
    defaults = {"indomain": True, "ood": True, "checkpoints": ["best", "final"], "ivon_mean": True}
    cfg["evaluation"] = {**defaults, **cfg.get("evaluation", {})}
    validate_config(cfg)
    return cfg


def validate_config(cfg):
    if cfg["method"] not in METHODS or cfg["dataset"] not in TRAINDATALOADERS:
        raise ValueError("Unknown method or dataset")
    if cfg.get("mc_dropout", False) and cfg["method"] not in {"ucbopt", "ucbopt_adapt", "lcbopt_adapt", "mcdrop"}:
        raise ValueError("mc_dropout combinations are supported for the CBOpt family")
    if not 0 < cfg["train_val_split"] < 1:
        raise ValueError("Best-validation selection requires a nonempty train/validation split")
    for key in ("epochs", "batch_size", "eval_batch_size"):
        if not isinstance(cfg[key], int) or cfg[key] < 1:
            raise ValueError(f"{key} must be a positive integer")
    if not 0 <= cfg["warmup_epochs"] < cfg["epochs"]:
        raise ValueError("warmup_epochs must be nonnegative and smaller than epochs")
    ev = cfg["evaluation"]
    if not ev["checkpoints"] or any(c not in {"best", "final"} for c in ev["checkpoints"]):
        raise ValueError("evaluation.checkpoints must contain best and/or final")
    if len(set(ev["checkpoints"])) != len(ev["checkpoints"]):
        raise ValueError("evaluation.checkpoints must not contain duplicates")
    if not ev["indomain"] and not ev["ood"]:
        raise ValueError("Enable at least one evaluation split")
    if ev["ood"] and ev.get("ood_dataset") not in OOD_LOADERS:
        raise ValueError("Set evaluation.ood_dataset to emnist, svhn, or tinyimagenet")
    hp = cfg["method_hparams"]
    for key in ("test_repeat", "model_samples", "mc_samples", "test_dups", "swag_freq"):
        if key in hp and (not isinstance(hp[key], int) or hp[key] < 1):
            raise ValueError(f"{key} must be a positive integer")
    if cfg["method"] == "swag":
        if not isinstance(hp.get("swag_start"), int) or not cfg["warmup_epochs"] < hp["swag_start"] < cfg["epochs"]:
            raise ValueError("SWAG collection must start after warmup and before training ends")
        if hp.get("swag_lr", 0.005) <= 0:
            raise ValueError("swag_lr must be positive")


def model_name_for_method(model_name, method, mc_dropout=False):
    if method == "mcdrop" or mc_dropout:
        return f"{model_name}_mcdrop"
    if method == "swag":
        return f"{model_name}_swag"
    return model_name


def model_kwargs(cfg):
    kwargs = dict(cfg.get("model_hparams", {}))
    hp = cfg["method_hparams"]
    if cfg["method"] == "mcdrop" or cfg.get("mc_dropout", False):
        kwargs["p"] = hp.get("dropout_p", 0.05)
    if cfg["method"] == "swag":
        kwargs["max_rank"] = hp.get("max_rank", 20)
    return kwargs


def build_model(cfg, device):
    name = model_name_for_method(cfg["model"], cfg["method"], cfg.get("mc_dropout", False))
    if name not in MODELS:
        raise ValueError(f"Unknown model/method pair: {name}")
    return name, MODELS[name](OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]], **model_kwargs(cfg)).to(device)


def build_optimizer(cfg, model):
    hp, method = cfg["method_hparams"], cfg["method"]
    common = {"lr": cfg["learning_rate"], "weight_decay": cfg["weight_decay"]}
    if method in {"sgd", "mcdrop", "swag", "laplace"}:
        return torch.optim.SGD(model.parameters(), momentum=hp.get("momentum", 0.9), **common)
    if method == "adamw":
        return torch.optim.AdamW(model.parameters(), betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.999)), **common)
    curvature = {"hess_init": hp.get("hess_init", 0.5), "rescale_lr": hp.get("rescale_lr", False)}
    if method == "ivon":
        return IVON(model.parameters(), beta1=hp.get("beta1", 0.9), beta2=hp.get("beta2", 0.99999),
                    ess=hp.get("ess", 54000 if cfg["dataset"] == "fmnist" else 45000),
                    mc_samples=hp.get("mc_samples", 1), hess_approx=hp.get("hess_approx", "price"),
                    clip_radius=hp.get("clip_radius", float("inf")), **curvature, **common)
    if method == "ucbopt":
        return uCBOpt(model.parameters(), beta1=hp.get("beta1", 0.9), beta2=hp.get("beta2", 0.99999),
                      cand_curvature=hp.get("cand_curvature", 0.0), eps=hp.get("eps", 1e-8), **curvature, **common)
    cls = uCBOptAdaptCurv if method == "ucbopt_adapt" else lCBOptAdaptCurv
    return cls(model.parameters(), betas=(hp.get("beta1", 0.9), hp.get("beta2", 0.99999),
                                         hp.get("beta3", 1.001 if method == "ucbopt_adapt" else 0.999)),
               gamma=hp.get("gamma", 0.9 if method == "ucbopt_adapt" else 1.05), eps=hp.get("eps", 1e-8),
               clip_radius=hp.get("clip_radius", float("inf")), **curvature, **common)


def build_scheduler(cfg, optimizer):
    warmup = cfg["warmup_epochs"]
    horizon = cfg["method_hparams"]["swag_start"] if cfg["method"] == "swag" else cfg["epochs"]
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, eta_min=cfg["lr_final"], T_max=horizon - warmup)
    if warmup == 0:
        return cosine
    linear = torch.optim.lr_scheduler.LinearLR(optimizer, start_factor=1.0 / warmup, end_factor=1.0, total_iters=warmup)
    return torch.optim.lr_scheduler.SequentialLR(optimizer, schedulers=[linear, cosine], milestones=[warmup])


def trainbatch_ivon(batchinput, model, optimizer, mc_samples=1):
    images, target = batchinput
    losses, probs = [], []
    for _ in range(mc_samples):
        with optimizer.sampled_params(train=True):
            optimizer.zero_grad(set_to_none=True)
            output = model(images)
            loss = nnf.cross_entropy(output, target)
            loss.backward()
        losses.append(loss.detach())
        probs.append(output.detach().softmax(dim=1))
    optimizer.step()
    return torch.stack(probs).mean(0), target, torch.stack(losses).mean().item()


def enable_mc_dropout(model):
    model.eval()
    for module in model.modules():
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            module.train()


def make_predictor(cfg, model, optimizer, device, la=None, prediction="predictive"):
    """Keep SWAG draws fixed across the in-domain and OOD passes."""
    hp = cfg["method_hparams"]
    model.eval()
    repeat = hp.get("test_repeat", 64)
    sampled = None
    if isinstance(model, SWAG):
        if model.n_models.item() == 0:
            raise ValueError("Cannot evaluate SWAG before collecting any models")
        sampled = [model.sampled_model(mode=hp.get("sample_mode", "modelwise")).to(device).eval()
                   for _ in range(hp.get("model_samples", 64))]
    if cfg["method"] == "mcdrop" or cfg.get("mc_dropout", False):
        enable_mc_dropout(model)

    @torch.no_grad()
    def predict(images):
        if la is not None:
            kwargs = {"pred_type": hp.get("pred_type", "glm")}
            if kwargs["pred_type"] == "glm":
                kwargs["link_approx"] = hp.get("link_approx", "probit")
            if kwargs["pred_type"] == "nn" or kwargs.get("link_approx") == "mc":
                kwargs["n_samples"] = hp.get("n_samples", 100)
            return la(images, **kwargs)
        if isinstance(optimizer, IVON) and prediction != "mean":
            result = None
            for _ in range(repeat):
                with optimizer.sampled_params():
                    prob = model(images).softmax(dim=1)
                result = prob / repeat if result is None else result + prob / repeat
            return result
        if sampled is not None:
            result = None
            for member in sampled:
                prob = member(images).softmax(dim=1)
                result = prob / len(sampled) if result is None else result + prob / len(sampled)
            return result
        passes = repeat if cfg["method"] == "mcdrop" or cfg.get("mc_dropout", False) else 1
        result = None
        for _ in range(passes):
            prob = model(images).softmax(dim=1)
            result = prob / passes if result is None else result + prob / passes
        return result

    return predict


def train(cfg, model_name, model, optimizer, scheduler, train_loader, val_loader, device):
    if not len(train_loader) or not len(val_loader):
        raise ValueError("Training and validation loaders must both be nonempty")
    save_dir = Path(cfg["output_dir"])
    log = coro_log_timed(None, cfg.get("print_freq", 100), cfg.get("bins", 20), str(save_dir))
    hp, method = cfg["method_hparams"], cfg["method"]
    best_loss = float("inf")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
        torch.cuda.reset_peak_memory_stats(device)
    start = time.perf_counter()
    epoch_times = []

    def save(name, epoch, **metadata):
        savecheckpoint(save_dir / name, model_name, (OUTCLASS[cfg["dataset"]], INSIZE[cfg["dataset"]]),
                       model_kwargs(cfg), model, optimizer, scheduler, epoch=epoch, config=cfg, **metadata)

    def train_batch(batchinput, model, optimizer):
        if method == "ivon":
            return trainbatch_ivon(batchinput, model, optimizer, hp.get("mc_samples", 1))
        return do_trainbatch(batchinput, model, optimizer)

    try:
        for epoch in range(cfg["epochs"]):
            swag_phase = method == "swag" and epoch >= hp["swag_start"]
            if swag_phase and epoch == hp["swag_start"]:
                for group in optimizer.param_groups:
                    group["lr"] = hp.get("swag_lr", 0.005)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_start = time.perf_counter()
            model.train()
            log.send((epoch, "train", len(train_loader), None))
            do_epoch(train_loader, train_batch, log, device, model=model, optimizer=optimizer)
            log.throw(StopIteration)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            epoch_times.append(time.perf_counter() - epoch_start)
            if swag_phase and (epoch - hp["swag_start"]) % hp.get("swag_freq", 1) == 0:
                model.collect_model()
            if not swag_phase:
                scheduler.step()
            save("checkpoint.pt", epoch)

            model.eval()
            can_select = method != "swag" or model.n_models.item() > 0
            # Select SWAG checkpoints only after posterior collection starts.
            predictor = make_predictor(cfg, model, optimizer, device) if method == "swag" and can_select else None

            def val_batch(batchinput, model):
                if predictor is None:
                    return do_evalbatch(batchinput, model)
                images, target = batchinput
                prob = predictor(images)
                loss = nnf.nll_loss(prob.clamp_min(torch.finfo(prob.dtype).tiny).log(), target)
                return prob, target, loss.item()

            log.send((epoch, "val", len(val_loader), None))
            with torch.no_grad():
                do_epoch(val_loader, val_batch, log, device, model=model)
            metrics = log.throw(StopIteration)
            _, val_loss, *_ = metrics
            del predictor
            if can_select and math.isfinite(val_loss) and val_loss < best_loss:
                best_loss = val_loss
                save("best_checkpoint.pt", epoch, best_val_loss=best_loss,
                     selection="posterior_validation_nll" if method == "swag" else "validation_loss")
    finally:
        log.close()
    if not math.isfinite(best_loss):
        raise RuntimeError("No finite validation loss was recorded; no best checkpoint exists")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    runtime = {
        "training_run_seconds": time.perf_counter() - start,
        "train_epoch_seconds": epoch_times,
        "mean_train_epoch_seconds": float(np.mean(epoch_times)),
        "peak_allocated_gib": torch.cuda.max_memory_allocated(device) / 1024**3 if device.type == "cuda" else None,
        "peak_reserved_gib": torch.cuda.max_memory_reserved(device) / 1024**3 if device.type == "cuda" else None,
    }
    (save_dir / "runtime.json").write_text(json.dumps(runtime, indent=2), encoding="utf-8")


def fit_laplace(cfg, model, train_loader, val_loader):
    from laplace import Laplace

    hp = cfg["method_hparams"]
    model.eval()
    la = Laplace(model, likelihood="classification", subset_of_weights=hp.get("subset_of_weights", "last_layer"),
                 hessian_structure=hp.get("hessian_structure", "kron"), prior_precision=hp.get("prior_precision", 1.0))
    la.fit(train_loader)
    if hp.get("optimize_prior_precision", True):
        la.optimize_prior_precision(method=hp.get("prior_precision_method", "marglik"),
                                    pred_type=hp.get("pred_type", "glm"), link_approx=hp.get("link_approx", "probit"),
                                    val_loader=val_loader)
    fitted_hp = dict(hp)
    precision = la.prior_precision
    fitted_hp["prior_precision"] = float(precision.detach().cpu().item()) if torch.is_tensor(precision) else float(precision)
    torch.save({"state_dict": la.state_dict(), "config": fitted_hp}, Path(cfg["output_dir"]) / "laplace_state.pt")
    return la


@torch.no_grad()
def collect_predictions(loader, predictor, device):
    predictions, targets = [], []
    for images, labels in loader:
        prob = predictor(images.to(device, non_blocking=True))
        if len(prob) != len(labels):
            if len(prob) % len(labels):
                raise ValueError("Duplicated input count does not match labels")
            prob = prob.reshape(-1, len(labels), prob.shape[-1]).mean(dim=0)
        predictions.append(prob.cpu().numpy())
        targets.append(labels.cpu().numpy())
    if not predictions:
        raise ValueError("Cannot evaluate an empty dataset")
    return np.concatenate(predictions), np.concatenate(targets)


def write_csv(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted({k for row in rows for k in row}))
        writer.writeheader()
        writer.writerows(rows)


def evaluate(cfg, model, optimizer, test_loader, device, la=None, checkpoint="best", selected_epoch=None):
    output_dir = Path(cfg["output_dir"])
    ev, hp = cfg["evaluation"], cfg["method_hparams"]
    modes = ["predictive", "mean"] if isinstance(optimizer, IVON) and ev.get("ivon_mean", True) else ["predictive"]
    ood_loader = None
    if ev["ood"]:
        ood_loader = OOD_LOADERS[ev["ood_dataset"]](cfg["data_dir"], cfg["eval_workers"], device.type == "cuda",
                                                  cfg["eval_batch_size"], "test", hp.get("test_dups", 1))
    rows = []
    for mode in modes:
        predictor = make_predictor(cfg, model, optimizer, device, la, mode)
        suffix = "_mean" if mode == "mean" else ""
        metadata = {"experiment": cfg["experiment"], "seed": cfg["random_seed"], "method": cfg["method"], "model": cfg["model"],
                    "dataset": cfg["dataset"], "checkpoint": checkpoint, "selected_epoch": selected_epoch,
                    "prediction": mode, "mc_dropout": cfg.get("mc_dropout", False)}
        if ev["indomain"] or ev["ood"]:
            probs, labels = collect_predictions(test_loader, predictor, device)
            np.save(output_dir / f"predictions_indomain{suffix}.npy", probs)
            np.save(output_dir / "labels_indomain.npy", labels)
            if ev["indomain"]:
                rows.append({**metadata, "split": "indomain", **classification_metrics(probs, labels, cfg.get("bins", 20))})
        if ood_loader is not None:
            out_probs, _ = collect_predictions(ood_loader, predictor, device)
            np.save(output_dir / f"predictions_ood{suffix}.npy", out_probs)
            metrics = OODMetrics(probs.max(axis=1), out_probs.max(axis=1)).get_all()
            rows.append({**metadata, "split": f"ood:{ev['ood_dataset']}", **metrics})
        del predictor
    write_csv(output_dir / "metrics.csv", rows)
    return rows


def run_seed(cfg, evaluate_only=False, overwrite=False):
    deterministic_run(cfg["random_seed"])
    GENERATOR.manual_seed(cfg["random_seed"])
    device = torch.device(cfg["device"])
    if device.type == "cuda":
        check_cuda()
    output_dir = Path(cfg["output_dir"])
    if not evaluate_only and not overwrite and any(output_dir.glob("*.pt")):
        raise FileExistsError(f"{output_dir} already contains checkpoints; use --evaluate-only or --overwrite")
    output_dir.mkdir(parents=True, exist_ok=True)
    if not evaluate_only:
        with open(output_dir / "config.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(cfg, f, sort_keys=False)
        environment = {"python": platform.python_version(), "torch": str(torch.__version__),
                       "device": str(device), "cuda_version": torch.version.cuda,
                       "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else platform.processor()}
        (output_dir / "environment.json").write_text(json.dumps(environment, indent=2), encoding="utf-8")
    elif (output_dir / "config.yaml").exists():
        saved = load_config(output_dir / "config.yaml")
        for key in ("method", "model", "dataset", "method_hparams", "model_hparams", "mc_dropout", "train_val_split", "random_seed"):
            if saved.get(key) != cfg.get(key):
                raise ValueError(f"Evaluation config differs from the saved run: {key}")
    else:
        raise FileNotFoundError(f"{output_dir / 'config.yaml'} is required to verify the evaluation settings")

    train_loader, val_loader = TRAINDATALOADERS[cfg["dataset"]](
        cfg["data_dir"], cfg["train_val_split"], cfg["workers"], device.type == "cuda", cfg["batch_size"], cfg["eval_batch_size"])
    test_loader = TESTDATALOADER[cfg["dataset"]](cfg["data_dir"], cfg["eval_workers"], device.type == "cuda", cfg["eval_batch_size"])
    if not evaluate_only:
        name, model = build_model(cfg, device)
        optimizer = build_optimizer(cfg, model)
        train(cfg, name, model, optimizer, build_scheduler(cfg, optimizer), train_loader, val_loader, device)
        del model, optimizer
    rows = []
    for checkpoint in cfg["evaluation"]["checkpoints"]:
        filename = "best_checkpoint.pt" if checkpoint == "best" else "checkpoint.pt"
        _, model, optimizer, _, metadata = loadcheckpoint(output_dir / filename, device, restore_scheduler=False)
        deterministic_run(cfg["random_seed"])
        GENERATOR.manual_seed(cfg["random_seed"])
        evaluation_cfg = copy.deepcopy(cfg)
        evaluation_cfg["output_dir"] = str(output_dir / f"eval_{checkpoint}")
        Path(evaluation_cfg["output_dir"]).mkdir(parents=True, exist_ok=True)
        la = fit_laplace(evaluation_cfg, model, train_loader, val_loader) if cfg["method"] == "laplace" else None
        rows.extend(evaluate(evaluation_cfg, model, optimizer, test_loader, device, la, checkpoint, metadata.get("epoch")))
        del model, optimizer, la
    write_csv(output_dir / "metrics.csv", rows)
    return rows


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True)
    seeds = parser.add_mutually_exclusive_group()
    seeds.add_argument("--seed", type=int)
    seeds.add_argument("--seeds", type=int, nargs="+")
    parser.add_argument("--device")
    parser.add_argument("--data-dir")
    parser.add_argument("--output-dir")
    parser.add_argument("--checkpoints", nargs="+", choices=("best", "final"))
    parser.add_argument("--evaluate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args(argv)
    cfg = load_config(args.config)
    for key in ("device", "data_dir", "output_dir"):
        if getattr(args, key) is not None:
            cfg[key] = getattr(args, key)
    if args.checkpoints:
        cfg["evaluation"]["checkpoints"] = args.checkpoints
    validate_config(cfg)
    selected_seeds = [args.seed] if args.seed is not None else args.seeds or cfg.get("seeds", [cfg["random_seed"]])
    if not selected_seeds or len(set(selected_seeds)) != len(selected_seeds):
        parser.error("Seeds must be nonempty and unique")
    base_dir = Path(cfg["output_dir"])
    for seed in selected_seeds:
        run_cfg = copy.deepcopy(cfg)
        run_cfg["random_seed"] = seed
        run_cfg["output_dir"] = str(base_dir / f"seed{seed}")
        run_seed(run_cfg, args.evaluate_only, args.overwrite)
    from scripts.summarize_results import summarize
    summarize(base_dir, base_dir / "summary.csv")


if __name__ == "__main__":
    main()
