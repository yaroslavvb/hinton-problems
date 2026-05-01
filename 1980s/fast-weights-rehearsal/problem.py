"""
Fast weights to deblur old memories (Hinton & Plaut 1987).
"""

import numpy as np


def generate_associations(n_pairs: int, dim: int):
    """Random binary input/output vector pairs."""
    raise NotImplementedError


def build_model(dim: int, slow_lr: float = 0.1, fast_lr: float = 0.5, fast_decay: float = 0.9):
    """Each weight has a slow (plastic) and a fast (decaying) component."""
    raise NotImplementedError


def learn_set(model, data, n_sweeps: int):
    raise NotImplementedError


def rehearse_subset(model, data, subset_idx, n_sweeps: int):
    """Brief replay of a few items in set A; fast weights should restore A."""
    raise NotImplementedError


def recall_accuracy(model, data) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
