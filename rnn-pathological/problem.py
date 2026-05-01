"""
Hochreiter-Schmidhuber pathological long-term-dependency RNN tasks
(Sutskever, Martens, Dahl & Hinton 2013).
"""

import numpy as np


TASKS = (
    "addition",
    "multiplication",
    "xor",
    "temporal_order",
    "3bit_memorization",
    "random_permutation_memorization",
    "noiseless_memorization",
)


def generate_dataset(task: str, sequence_len: int, n_samples: int):
    """Generate one of the pathological RNN tasks."""
    raise NotImplementedError


def build_rnn(n_in: int, n_hidden: int, n_out: int, init: str = "ortho"):
    raise NotImplementedError


def train_with_momentum(model, data, n_steps: int, lr: float, momentum: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
