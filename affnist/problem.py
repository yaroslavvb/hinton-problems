"""
Train on translated MNIST, test on affNIST (Sabour, Frosst & Hinton 2017).
"""

import numpy as np


def make_translated_mnist(mnist, max_shift: int = 6):
    """Pad to 40x40, randomly translate each digit by up to max_shift pixels."""
    raise NotImplementedError


def load_affnist_test():
    """Load affNIST 40x40 test set."""
    raise NotImplementedError


def evaluate_robustness(model, train_data, test_data) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    pass
