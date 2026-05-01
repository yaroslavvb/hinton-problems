"""
smallNORB held-out viewpoint generalization
(Hinton, Sabour & Frosst 2018).
"""

import numpy as np


def split_by_azimuth(dataset, train_range: tuple, test_range: tuple):
    raise NotImplementedError


def split_by_elevation(dataset, train_range: tuple, test_range: tuple):
    raise NotImplementedError


def build_matrix_capsule_net():
    """Capsules with 4x4 pose matrices and EM routing."""
    raise NotImplementedError


def em_routing_step(votes, activations, n_iters: int = 3):
    raise NotImplementedError


def train(model, data, n_epochs: int, lr: float):
    raise NotImplementedError


def evaluate_viewpoint_extrapolation(model, train_data, test_data) -> dict:
    raise NotImplementedError


if __name__ == "__main__":
    pass
