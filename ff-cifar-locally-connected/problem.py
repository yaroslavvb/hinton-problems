"""
Forward-Forward on CIFAR-10 with locally-connected layers
(Hinton 2022).
"""

import numpy as np


def build_locally_connected_ff(map_size: int = 32, n_channels: int = 3,
                                receptive_field: int = 11, n_layers: int = 3):
    """Locally-connected (no weight sharing) with bottom-up and top-down inputs."""
    raise NotImplementedError


def train(model, cifar, n_epochs: int, lr: float):
    raise NotImplementedError


def evaluate(model, cifar_test) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
