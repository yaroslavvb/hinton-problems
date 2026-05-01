"""
N-bit parity backprop (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset(n_bits: int):
    """All 2**n_bits patterns labeled by parity."""
    raise NotImplementedError


def build_model(n_bits: int, n_hidden: int = None):
    """Default n_hidden = n_bits."""
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.5):
    raise NotImplementedError


def inspect_hidden_code(model, data) -> np.ndarray:
    """Should reveal a thermometer-like pattern when fully trained."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
