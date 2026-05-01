"""
Transforming autoencoders (Hinton, Krizhevsky & Wang 2011).
"""

import numpy as np


def make_transformed_pair(image: np.ndarray, transform: str = "translation"):
    """Apply a random transformation; return (image, transformed_image, params)."""
    raise NotImplementedError


def build_capsule_net(n_capsules: int = 30, recognition_dim: int = 3,
                      generative_units: int = 128, patch_size: int = 22):
    raise NotImplementedError


def train(model, mnist_loader, n_epochs: int, lr: float):
    raise NotImplementedError


def predict_transformation(model, pair) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    pass
