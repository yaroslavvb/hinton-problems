"""
T-C discrimination on a shared-weight retina
(Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset():
    """8 patterns: T or C, each in 4 rotations, on a 2D binary retina."""
    raise NotImplementedError


def build_model(retina_h: int, retina_w: int, kernel: int = 3):
    """Conv-like with weight tying across spatial positions."""
    raise NotImplementedError


def train(model, data, n_sweeps: int = 5000, lr: float = 0.3):
    raise NotImplementedError


def visualize_filters(model):
    """Inspect the discovered 3x3 detectors."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
