"""
2-bit distributed-to-local mapping with a 1-unit bottleneck
(Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """4 distributed 2-bit inputs -> 4 one-hot outputs."""
    raise NotImplementedError


def build_model(n_in: int = 2, n_hidden: int = 1, n_out: int = 4):
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


def hidden_values(model, data) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    pass
