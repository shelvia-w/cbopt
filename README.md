# CBOpt

Official code for **Maxitive Donsker-Varadhan Formulation for Possibilistic Variational Inference**, accepted at NeurIPS 2026.

## Installation

Use Python 3.11 with a PyTorch build suitable for your hardware:

```bash
python -m pip install -r requirements.txt
```

Run commands from the repository root. Classification configs default to CUDA,
regression selects CUDA when available. Pass `--device cpu` to override. Data is
downloaded on first use, `--data-dir` and `--output-dir` select other locations.

## Classification

| Dataset | Model | OOD dataset | Epochs |
| --- | --- | --- | --- |
| Fashion-MNIST | LeNet | EMNIST letters | 100 |
| CIFAR-10 | ResNet-20 with FRN | SVHN | 200 |
| CIFAR-100 | DenseNet-121 with FRN | TinyImageNet | 200 |

Methods include SGD, AdamW, IVON, MC Dropout, last-layer Laplace, SWAG, uCBOpt,
uCBOpt-adapt, and lCBOpt-adapt. All hyperparameters are in [configs/](configs).

IVON uses code from the [official implementation](https://github.com/team-approx-bayes/ivon).
Last-layer Laplace uses [laplace-torch](https://github.com/aleximmer/Laplace).
SWAG is adapted from Pavel Izmailov's
[understandingbdl](https://github.com/izmailovpavel/understandingbdl).

```bash
python scripts/run_experiment.py --config configs/fmnist_lenet/ucbopt.yaml
python scripts/run_suite.py --suite main
python scripts/run_suite.py --suite combined
python scripts/run_suite.py --suite ablations
```

`main` contains 27 reported dataset/method combinations, `combined` contains four
CBOpt-plus-dropout variants, `ablations` contains five sweeps with 38 settings.
Use `--suite all --list` to list commands without running them. Individual runs
accept `--seed 0`, `--seeds 0 1 2`, and device/directory overrides.

Defaults use seeds 0, 1, and 2, a contiguous 90/10 train/validation split, and
training/evaluation batches of 128/256. CIFAR uses random horizontal flips and
padded crops, Fashion-MNIST has no augmentation. Normalization constants are in
[datasets/](datasets). TinyImageNet evaluation uses its labeled validation split,
resized and center-cropped to 32 by 32 pixels. DenseNet uses blocks
`[6, 12, 24, 16]`, growth rate 12, and compression 0.5.

Ordinary methods use five warmup epochs and cosine decay to zero. SWAG instead
uses 60/160 SGD epochs for Fashion-MNIST/CIFAR, with cosine decay from
0.005/0.05 to 0.001/0.01, followed by 40 epochs at 0.005. It collects each epoch
and retains a rank-20 covariance approximation.

### Evaluation and aggregation

```bash
python scripts/run_experiment.py --config configs/fmnist_lenet/ucbopt.yaml --evaluate-only
python scripts/summarize_results.py --root outputs --require-seeds 0 1 2
```

Each seed directory contains its config, environment, runtime, training logs, and
best/final checkpoints. Evaluation reloads model and optimizer state into separate
`eval_best/` and `eval_final/` outputs, including metrics, probabilities, and labels.
The saved config must match, `--checkpoints best` or `final` selects one checkpoint.
Existing checkpoints require `--overwrite` for retraining.

Best checkpoints minimize validation loss. IVON validation uses mean parameters,
dropout validation uses one stochastic pass. SWAG selects a collected posterior
by predictive validation NLL. Test/OOD prediction averages 64 passes for MC
Dropout and IVON, or 64 SWAG models shared across ID/OOD. IVON also reports
unsampled mean-parameter predictions. Laplace is fitted separately to each MAP
checkpoint using last-layer Kronecker curvature, marginal-likelihood prior
optimization, and analytic GLM/probit prediction.

Metrics use the saved probabilities: accuracy, top-5 accuracy, NLL after predictive
averaging, Brier score, 20-bin ECE, and correctness/confidence AUROC. OOD metrics
are AUROC, AUPR-In/Out, FPR@95, and minimum balanced detection error. FPR@95 uses
the first threshold reaching 95% in-domain recall, keeping ties together without
interpolation. Summaries use sample standard deviations across seeds (`ddof=1`),
with separate experiment/checkpoint/prediction groups and contributing counts.

To compute metrics from saved arrays without inference:

```bash
python scripts/recompute_metrics.py --indomain id.npy --ood ood.npy --labels labels.npy --output metrics.csv
```

Labels are optional for OOD metrics. `runtime.json` separates training-epoch and
full-run time. Peak CUDA memory includes validation.

## Regression

```bash
python scripts/run_contamination.py
python scripts/run_underrepresentation.py
```

Both compare Adam, BBB, diagonal VOGN, Vadam, lCBOpt, and uCBOpt on all ten
datasets, using the [regression presets](configs/regression). Add `--list` to see
datasets without running anything, or use `--datasets Concrete Diabetes` to select
a subset. Dataset sources are defined in [regression/data.py](regression/data.py).

Defaults use 20 repetitions with seed
`42 + 10000 * dataset_index + 1000 * repetition`. Canonical dataset order is in
[regression/specs.py](regression/specs.py) and is preserved for subsets. Cleaning
(categorical encoding, median imputation, constant-feature removal) precedes the
60/40 fit/test split. Scaling, PCA, and target normalization use the fit pool.
The float64 network has five PCA inputs (zero-padded if needed), two tanh hidden
units, and 15 parameters. Methods share an Adam warm start and then take 60 updates. CBO uses full Hessians. Regression metrics evaluate fitted parameter means.

- **Contamination:** up to 80 training examples, the 30% with largest feature norms
  receive randomly signed shifts of six standardized target units. Test targets
  remain clean. Warmup is 100 steps, CBO learning rate is 0.2. Report test RMSE
  in original target units.
- **Under-representation:** retain 20% of cases above the fit-pool target's 80th
  percentile, then fill common cases toward 80 samples. Retained severe cases take
  priority, so training can exceed 80 samples and contain no common cases. Warmup
  is 20 steps, CBO learning rate is 0.3. Report LT-MAE in original target units,
  restricted to test targets at or above the fit-pool target's 80th percentile.

Data is cached in `data/regression/`, complete repetitions are saved atomically
under `outputs/regression/<experiment>/`. Runs resume automatically when the
config, source, and package versions match, otherwise use a new output directory.
Each `<dataset>_all_runs.csv` stores the reported metric for every method/run.
`table.csv` and `table.tex` contain dataset means ± sample standard deviations
(`ddof=1`) and a Mean Rank row, formatted to two decimals. Methods are ranked
within each dataset/run using unrounded RMSE (contamination) or LT-MAE
(under-representation), with average ranks for ties. Mean ranks average all 200
dataset/run pairs. LaTeX bolds the lowest unrounded mean in each row. Tables
require all configured datasets and repetitions to be complete.

To summarize existing per-dataset CSVs without fitting models or downloading data:

```bash
python scripts/run_contamination.py --summary-only --output-dir contamination_targeted30_results
python scripts/run_underrepresentation.py --summary-only --output-dir ten_dataset_underrepresentation_results
```

Existing CSV columns `Test RMSE` and `High-risk MAE` are also accepted for RMSE
and LT-MAE respectively. LaTeX tables use `booktabs` and `graphicx`.

Use any script's `--help` for its options.
