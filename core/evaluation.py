"""Generic prediction helpers."""

import torch
import torch.nn.functional as nnf
from .engine import avgdups


# =============================
# Method-specific prediction helpers
# =============================

def predict_proba_batch(batchinput, model, dups: int = 1, repeat: int = 1):
    """Run a prediction-only batch and return averaged class probabilities."""
    inputs = batchinput[:-1]
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        output = model(*inputs)
        prob = nnf.softmax(output, 1)
        prob = avgdups(prob, dups) if dups > 1 else prob
        cumprob = cumprob + prob / repeat
    return cumprob


def predict_proba_von(batchinput, model, optimizer, repeat: int = 1):
    """Run repeated sampled predictions with a VON optimizer."""
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / repeat
    return cumprob


def predict_proba_swag(batchinput, models):
    """Average class probabilities across a SWAG ensemble."""
    inputs = batchinput[:-1]
    cumprob = torch.zeros([])
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob


def predict_proba_duq(batchinput, model):
    """Convert DUQ scores into normalized class probabilities."""
    inputs = batchinput[:-1]
    scores = model(*inputs)
    prob = scores / scores.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return prob


def predict_proba_sngp(batchinput, model):
    """Run SNGP mean-field inference and return class probabilities."""
    inputs = batchinput[:-1]
    logits = model(*inputs, mean_field=True, return_gp_cov=False)
    prob = torch.softmax(logits, dim=1)
    return prob
