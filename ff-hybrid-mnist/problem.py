"""
Forward-Forward unsupervised on MNIST with hybrid-image negatives
(Hinton 2022).
"""

import numpy as np


def make_hybrid_image(digit_a: np.ndarray, digit_b: np.ndarray) -> np.ndarray:
    """Mix two digits using a smoothly thresholded random mask.

    Mask construction:
        m = uniform random bit-mask
        m = blur(m, kernel=[1/4, 1/2, 1/4])  # repeat several times
        m = (m > 0.5).astype(float)
    """
    raise NotImplementedError


def goodness(activations: np.ndarray) -> np.ndarray:
    """Sum of squares of activations per example."""
    raise NotImplementedError


def ff_layer_step(layer, x_pos, x_neg, threshold: float, lr: float):
    """One Forward-Forward update: push goodness up for positives, down for negatives."""
    raise NotImplementedError


def build_ff_mlp(layer_sizes: tuple = (784, 2000, 2000, 2000, 2000)):
    raise NotImplementedError


def train_unsupervised(model, mnist, n_epochs: int, lr: float):
    raise NotImplementedError


def fit_softmax_on_top_layers(model, mnist):
    """Use last 3 layers' normalized activities as features."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
