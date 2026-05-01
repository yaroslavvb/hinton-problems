"""
Dipole 2D-position population code (Zemel & Hinton 1995).
"""

import numpy as np


def generate_dipole_images(n_samples: int, h: int = 8, w: int = 8):
    """Render small dipole image at random (x, y)."""
    raise NotImplementedError


def build_population_coder(n_hidden: int = 100, n_implicit_dims: int = 2):
    raise NotImplementedError


def description_length_loss(model, data) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
