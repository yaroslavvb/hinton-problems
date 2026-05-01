"""
8-3-8 minimum-capacity Boltzmann encoder (Ackley, Hinton & Sejnowski 1985).
"""

import numpy as np


def generate_dataset():
    raise NotImplementedError


def build_model(n_v1: int = 8, n_h: int = 3, n_v2: int = 8):
    raise NotImplementedError


def train(model, data, n_cycles: int = 4000, lr: float = 0.05):
    raise NotImplementedError


def codes_used(model) -> int:
    """Number of distinct hidden codes the network has actually adopted."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
