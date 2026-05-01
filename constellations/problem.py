"""
Constellations 2D point-cloud part-whole grouping
(Kosiorek, Sabour, Teh & Hinton 2019).
"""

import numpy as np


TEMPLATES = (
    np.array([[0, 0], [1, 0], [0, 1], [1, 1]]),  # square
    np.array([[0, 0], [1, 0], [0.5, 1], [0.5, 0]]),  # triangle-with-extra
    np.array([[0, 0], [1, 0], [0.5, 0.866]]),  # triangle
)


def affine_transform(points: np.ndarray) -> np.ndarray:
    """Apply a random rotation + scale + translation."""
    raise NotImplementedError


def generate_constellation():
    """Union of K=3 affine-transformed templates with K-means-style point counts."""
    raise NotImplementedError


def build_set_transformer_encoder():
    raise NotImplementedError


def build_capsule_decoder(n_object_capsules: int, template_set):
    raise NotImplementedError


def train(model, data, n_steps: int, lr: float):
    raise NotImplementedError


def part_capsule_recovery_accuracy(model, data) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
