"""
MultiMNIST overlapping digits (Sabour, Frosst & Hinton 2017).
"""

import numpy as np


def overlay_pair(digit_a: np.ndarray, digit_b: np.ndarray,
                 canvas: int = 36, max_shift: int = 4) -> np.ndarray:
    """Place two distinct-class digits with bounding-box overlap >= 80%."""
    raise NotImplementedError


def generate_multimnist(n_samples: int):
    raise NotImplementedError


def build_capsnet():
    """Conv1 256 9x9 -> PrimaryCaps (32 8-D) -> DigitCaps (10 16-D, 3-iter routing)."""
    raise NotImplementedError


def margin_loss(predictions, labels):
    raise NotImplementedError


def reconstruction_loss(predictions, target_image):
    raise NotImplementedError


def train(model, data, n_epochs: int, lr: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
