"""
Binary addition of two 2-bit numbers (Rumelhart, Hinton & Williams 1986).

Inputs: 4 bits = (a1, a0, b1, b0).
Outputs: 3 bits = sum of (a1 a0) + (b1 b0).
"""

import numpy as np


def generate_dataset():
    """16 (a, b) pairs labeled by 3-bit binary sum."""
    raise NotImplementedError


def build_model(arch: str = "4-3-3"):
    """arch in {"4-3-3", "4-2-3"}; the latter often gets stuck in local minima."""
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


def local_minimum_rate(arch: str, n_trials: int = 50) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
