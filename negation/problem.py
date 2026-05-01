"""
Negation problem (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """16 patterns: 1 flag bit + 3 data bits -> 3 output bits (or their complement)."""
    raise NotImplementedError


def build_model(n_in: int = 4, n_hidden: int = 3, n_out: int = 3):
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


if __name__ == "__main__":
    pass
