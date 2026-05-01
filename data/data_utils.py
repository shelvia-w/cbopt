"""Data-specific helpers."""

import random
import torch


def dup_collate_fn(dups: int):
    """Create a collate function that duplicates input images."""
    def collate_fn(data):
        imgs, gts = tuple(zip(*data))
        t = torch.stack(imgs, dim=0)
        return t.repeat(dups, *(1,) * (t.ndim - 1)), torch.as_tensor(gts)
    return collate_fn

def corrupt_labels(dataset, noise_rate, seed=None, indices=None):
    """Randomly corrupt dataset labels at the given noise rate."""
    if seed is not None:
        random.seed(seed)

    targets = dataset.targets
    n_classes = max(targets) + 1
    idx_to_corrupt = indices if indices is not None else range(len(targets))

    n_corrupted = 0
    for i in idx_to_corrupt:
        if random.random() < noise_rate:
            wrong_labels = [c for c in range(n_classes) if c != targets[i]]
            targets[i] = random.choice(wrong_labels)
            n_corrupted += 1

    print(f"Corrupted {n_corrupted}/{len(idx_to_corrupt)} labels ({100*noise_rate:.0f}% noise)")
