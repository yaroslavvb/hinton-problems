"""
AIR variable-count Multi-MNIST scenes (Eslami et al. 2016).
"""

import numpy as np


def render_scene(canvas_size: int = 50, n_digits_dist=(0, 1, 2)):
    """Place 0, 1, or 2 MNIST digits at random affine on a blank canvas."""
    raise NotImplementedError


def build_air_model(canvas_size: int = 50, max_steps: int = 3,
                    what_dim: int = 50, where_dim: int = 3):
    raise NotImplementedError


def elbo_loss(model, batch):
    raise NotImplementedError


def train(model, data, n_steps: int, lr: float):
    raise NotImplementedError


def parse_scene(model, image) -> dict:
    """Return per-step (presence, where, what) latents."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
