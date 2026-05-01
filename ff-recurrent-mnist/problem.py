"""
Forward-Forward top-down recurrent on repeated-frame MNIST
(Hinton 2022).
"""

import numpy as np


def build_recurrent_ff(layer_sizes: tuple, damping: float = 0.7):
    """Each layer takes (normalized) inputs from layer above and below at t-1."""
    raise NotImplementedError


def synchronous_iterate(model, image, label_one_hot, n_iters: int = 8):
    raise NotImplementedError


def train(model, mnist, n_epochs: int, lr: float):
    raise NotImplementedError


def predict_by_iteration_goodness(model, image, n_classes: int = 10) -> int:
    """Accumulate goodness over iterations 3..5 across each candidate label."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
