"""
Multi-level glimpse MNIST (Ba, Hinton, Mnih, Leibo & Ionescu 2016).
"""

import numpy as np


def generate_glimpse_sequence(image: np.ndarray) -> np.ndarray:
    """Return 24 7x7 patches: 4 coarse 14x14 quadrants, each split into 4."""
    raise NotImplementedError


def build_glimpse_rnn_with_fast_weights(glimpse_dim: int, n_hidden: int, n_classes: int = 10):
    raise NotImplementedError


def train(model, data, n_steps: int, lr: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
