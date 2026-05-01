"""
Bars problem for RBM training (Hinton 2000).
"""

import numpy as np


def generate_bars(n_samples: int, h: int = 4, w: int = 4):
    raise NotImplementedError


def build_rbm(n_visible: int, n_hidden: int):
    raise NotImplementedError


def cd1_step(rbm, batch: np.ndarray, lr: float):
    raise NotImplementedError


def visualize_filters(rbm):
    """Each hidden unit should pick out a single bar."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
