"""
Forward-Forward supervised MNIST with label in first 10 pixels
(Hinton 2022).
"""

import numpy as np


def encode_label_in_pixels(image: np.ndarray, label: int, n_classes: int = 10) -> np.ndarray:
    """Replace the first n_classes pixels with the one-hot label."""
    raise NotImplementedError


def make_positive(image: np.ndarray, true_label: int) -> np.ndarray:
    raise NotImplementedError


def make_negative(image: np.ndarray, true_label: int, n_classes: int = 10) -> np.ndarray:
    """Wrong label encoded in first 10 pixels."""
    raise NotImplementedError


def build_ff_mlp(layer_sizes: tuple = (784, 2000, 2000, 2000, 2000)):
    raise NotImplementedError


def train(model, data, n_epochs: int, lr: float):
    raise NotImplementedError


def predict_by_goodness(model, image: np.ndarray) -> int:
    """Try each candidate label; pick the one with highest accumulated goodness."""
    raise NotImplementedError


def jittered_augmentation(image: np.ndarray, max_shift: int = 2) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    pass
