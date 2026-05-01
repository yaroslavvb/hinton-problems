"""
6-bit palindrome detection (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """All 64 6-bit patterns labeled 1 if palindrome else 0."""
    raise NotImplementedError


def build_model(n_in: int = 6, n_hidden: int = 2, n_out: int = 1):
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


def inspect_weight_symmetry(model) -> dict:
    """Verify the 1:2:4 magnitude ratio with opposite signs."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
