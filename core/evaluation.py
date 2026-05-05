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
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    for _ in range(repeat):
        with optimizer.sampled_params():
            output = model(*inputs)
        prob = nnf.softmax(output, 1)
        cumprob = cumprob + prob / repeat
    return cumprob


def predict_proba_swag(batchinput, models):
    """Average class probabilities across a SWAG ensemble."""
    inputs = batchinput[:-1]
    cumprob = torch.zeros([], device=inputs[0].device, dtype=inputs[0].dtype)
    nmodel = len(models)
    for model in models:
        output = model(*inputs)
        cumprob = cumprob + nnf.softmax(output, 1) / nmodel
    return cumprob
