"""
Sunspots time-series prediction with soft weight-sharing
(Nowlan & Hinton 1992).
"""

import numpy as np


def load_wolfer():
    """Return yearly Wolfer sunspot counts as the Weigend benchmark."""
    raise NotImplementedError


def make_lagged_dataset(series: np.ndarray, n_lags: int = 12):
    raise NotImplementedError


def build_model(n_lags: int, n_hidden: int = 8):
    raise NotImplementedError


def train_with_soft_sharing(model, data, n_components: int, lr: float):
    """Mixture-of-Gaussians prior on weights."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
