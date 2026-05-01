"""
Recurrent shift-register task (Rumelhart, Hinton & Williams 1986).
"""

import numpy as np


def generate_dataset(n_units: int, sequence_len: int):
    """Random binary sequences with target = input shifted by 1 step."""
    raise NotImplementedError


def build_rnn(n_units: int):
    raise NotImplementedError


def train_through_time(model, data, n_sweeps: int = 200, lr: float = 0.3):
    raise NotImplementedError


if __name__ == "__main__":
    pass
