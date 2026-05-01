"""
MNIST-2x5 subclass distillation (Müller, Kornblith & Hinton 2020).
"""

import numpy as np


def relabel_to_superclass(labels: np.ndarray) -> np.ndarray:
    """0..4 -> 0; 5..9 -> 1."""
    raise NotImplementedError


def build_teacher(n_subclasses_per_super: int = 5):
    """Outputs 10 sub-logits but is trained on binary super-class labels."""
    raise NotImplementedError


def auxiliary_distance_loss(sub_logits, super_labels):
    """Maximize pairwise distance between subclass logits within each super-class."""
    raise NotImplementedError


def distill_to_student(teacher, student, data, temperature: float, n_epochs: int):
    raise NotImplementedError


def evaluate_subclass_recovery(student, mnist_test) -> float:
    """Test whether the student's logits cluster correctly with the original 10 classes."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
