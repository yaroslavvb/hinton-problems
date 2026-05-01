"""
Ellipse World (Culp, Sabour & Hinton 2022).
"""

import numpy as np


CLASSES = ("face", "sheep", "house", "tree", "car")  # placeholder set


def sample_ellipse_layout(class_name: str) -> np.ndarray:
    """Return 5 ellipse parameters (6-DoF each) for the canonical class layout."""
    raise NotImplementedError


def render_grid(layout, grid_h: int, grid_w: int) -> np.ndarray:
    """Apply a global affine to the layout and place ellipse symbols on a grid."""
    raise NotImplementedError


def generate_dataset(n_samples: int, ambiguity: float = 1.0):
    """Higher ambiguity = more individual-ellipse uncertainty -> rely on relations."""
    raise NotImplementedError


def build_eglom(n_levels: int = 2, locations: tuple = (8, 8), embedding_dim: int = 64):
    """MLP replicated per location + within-level attention."""
    raise NotImplementedError


def train(model, data, n_steps: int, lr: float):
    raise NotImplementedError


def visualize_islands(model, image):
    """Inspect island formation across iterations."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
