"""
8-3-8 backprop autoencoder (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """8 one-hot 8-D patterns."""
    raise NotImplementedError


def build_model(n_in: int = 8, n_hidden: int = 3, n_out: int = 8):
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


def hidden_code_table(model, data) -> np.ndarray:
    """Return the 3-D hidden activations for each of the 8 inputs."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
