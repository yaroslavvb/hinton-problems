"""
MNIST distillation with the digit '3' omitted from the student's training set
(Hinton, Vinyals & Dean 2015).
"""

import numpy as np


def filter_class(dataset, omitted_class: int = 3):
    """Drop all examples of `omitted_class` from the training set."""
    raise NotImplementedError


def build_teacher():
    """e.g. 2x1200 MLP with jittered inputs."""
    raise NotImplementedError


def build_student():
    """e.g. 2x800 MLP without regularization."""
    raise NotImplementedError


def distill(teacher, student, data, temperature: float = 20.0,
            n_epochs: int = 60, lr: float = 0.01):
    raise NotImplementedError


def bias_correct_for_omitted(student, omitted_class: int):
    """Increase logit bias for the omitted class to recover accuracy."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
