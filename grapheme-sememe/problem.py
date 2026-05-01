"""
Grapheme-sememe word reading + lesion / relearning experiments
(Hinton & Sejnowski 1986).
"""

import numpy as np


def generate_mapping(n_words: int = 20, n_letters: int = 10, n_positions: int = 3):
    """Return (grapheme_vectors, sememe_vectors) for n_words random associations."""
    raise NotImplementedError


def build_model(n_grapheme: int = 30, n_hidden: int = 20, n_sememe: int = 30):
    raise NotImplementedError


def train(model, data, n_cycles: int):
    raise NotImplementedError


def lesion(model, fraction: float):
    """Randomly zero `fraction` of weights."""
    raise NotImplementedError


def relearn_subset(model, data, indices: list, n_cycles: int):
    """Retrain on only the patterns indexed by `indices`; check spontaneous recovery."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
