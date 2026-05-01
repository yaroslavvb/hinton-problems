"""
Spline images for factorial VQ (Hinton & Zemel 1994).
"""

import numpy as np


def generate_spline_images(n_samples: int = 200, h: int = 8, w: int = 12, n_controls: int = 5):
    """Render Gaussian-blurred curves through random control points."""
    raise NotImplementedError


def build_factorial_vq(n_dims: int = 4, n_units_per_dim: int = 6):
    raise NotImplementedError


def build_baseline_vq(n_units: int = 24):
    raise NotImplementedError


def description_length(model, data) -> float:
    """Bits to describe data via the bits-back coding view."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
