"""
4-2-4 encoder (Ackley, Hinton & Sejnowski 1985).

Boltzmann encoder with V1 (4 units) <-> H (2 units) <-> V2 (4 units).
Each pattern: one V1 unit on, matching V2 unit on.
"""

import numpy as np


def generate_dataset():
    """4 patterns: V1[i]=1 and V2[i]=1, others 0, for i in 0..3."""
    raise NotImplementedError


def build_model(n_v1: int = 4, n_h: int = 2, n_v2: int = 4):
    raise NotImplementedError


def train(model, data, n_cycles: int = 200, lr: float = 0.1):
    raise NotImplementedError


def evaluate(model, data) -> float:
    """Fraction of test patterns recalled correctly via Gibbs sampling."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
