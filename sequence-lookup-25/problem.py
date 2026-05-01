"""
25-sequence look-up RNN task (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset(n_total: int = 25, n_train: int = 20, variable_timing: bool = False):
    """5-letter input sequences mapped to 3-digit output."""
    raise NotImplementedError


def build_rnn(n_in: int = 5, n_hidden: int = 30, n_out: int = 3):
    raise NotImplementedError


def train_bptt(model, data, n_sweeps: int = 1000, lr: float = 0.3):
    raise NotImplementedError


def test_generalization(model, test_data) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
