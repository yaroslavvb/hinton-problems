"""
3-bit even-parity ensemble (Ackley, Hinton & Sejnowski 1985).
"""

import numpy as np


def generate_dataset():
    """Return the 4 even-parity 3-bit patterns with uniform p=0.25."""
    raise NotImplementedError


def build_model(n_visible: int = 3, n_hidden: int = 0):
    """Visible-only Boltzmann machine when n_hidden=0."""
    raise NotImplementedError


def train(model, data, n_cycles: int):
    raise NotImplementedError


if __name__ == "__main__":
    pass
