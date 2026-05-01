"""
Adaptive mixture of experts on Peterson-Barney vowels
(Jacobs, Jordan, Nowlan & Hinton 1991).
"""

import numpy as np


def load_peterson_barney():
    """Return F1, F2 features and class labels for [i], [I], [a], [Lambda]."""
    raise NotImplementedError


def build_moe(n_experts: int = 4, n_in: int = 2, n_out: int = 4):
    """Linear experts with softmax gating."""
    raise NotImplementedError


def train(model, data, n_epochs: int, lr: float = 0.05):
    raise NotImplementedError


def visualize_partitioning(model, data):
    raise NotImplementedError


if __name__ == "__main__":
    pass
