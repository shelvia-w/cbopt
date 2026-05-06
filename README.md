# CBO Experiments

This repository contains the code needed to reproduce the in-domain
classification and out-of-domain detection experiments.

## Experiments

In-domain:

- Fashion-MNIST with LeNet
- CIFAR-10 with ResNet-20
- CIFAR-100 with DenseNet-101

OOD:

- EMNIST for Fashion-MNIST
- SVHN for CIFAR-10
- TinyImageNet for CIFAR-100

Methods:

- SGD
- AdamW
- IVON
- MC Dropout
- Laplace
- SWAG
- uCBOpt
- uCBOpt-adapt
- lCBOpt-adapt

## Setup

```bash
pip install -r requirements.txt
```

## Run

Each experiment is defined by a YAML config under `configs/`.

```bash
python scripts/run_experiment.py --config configs/fmnist_lenet/adamw.yaml
```

Outputs are written to the config's `output_dir`. The runner trains the model, saves checkpoints, evaluates the in-domain test set, evaluates the configured OOD dataset, and writes `metrics.csv` plus saved prediction arrays.
