"""
XOR backprop demo (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """4 patterns: (0,0)->0, (0,1)->1, (1,0)->1, (1,1)->0."""
    raise NotImplementedError


def build_model(arch: str = "2-2-1"):
    """arch in {"2-2-1", "2-1-2-skip"}."""
    raise NotImplementedError


def train(model, data, n_sweeps: int = 1000, lr: float = 0.5):
    raise NotImplementedError


if __name__ == "__main__":
    pass
