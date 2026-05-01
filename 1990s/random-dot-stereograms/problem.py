"""
Random-dot stereograms with the Imax (mutual-information) objective
(Becker & Hinton 1992).
"""

import numpy as np


def generate_stereo_pair(surface_fn, n_dots: int, strip_width: int):
    """Render left/right dot patches with disparity from a synthetic surface."""
    raise NotImplementedError


def build_two_module_net(strip_dim: int, n_hidden: int):
    """Two parallel modules each viewing one strip; share output dim."""
    raise NotImplementedError


def imax_loss(out_a: np.ndarray, out_b: np.ndarray) -> float:
    """Mutual information between module outputs under Gaussian assumption."""
    raise NotImplementedError


def train(model, data, n_epochs: int, lr: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
