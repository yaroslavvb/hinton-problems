"""
What / where (discontinuous) population code (Zemel & Hinton 1995).
"""

import numpy as np


def generate_bars(n_samples: int, h: int = 8, w: int = 8):
    """Half horizontal bars at random y; half vertical bars at random x."""
    raise NotImplementedError


def build_population_coder(n_hidden: int = 100, n_implicit_dims: int = 2):
    raise NotImplementedError


def description_length_loss(model, data) -> float:
    raise NotImplementedError


def visualize_implicit_space(model, data):
    """Expect horizontal-bar codes in one corner, vertical in the opposite corner."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
