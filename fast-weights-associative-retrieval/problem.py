"""
Associative retrieval (Ba, Hinton, Mnih, Leibo & Ionescu 2016).
"""

import numpy as np


def generate_sample(n_pairs: int = 4):
    """Random key/value pairs (letter+digit) followed by '??' and a query key."""
    raise NotImplementedError


def generate_dataset(n_samples: int, n_pairs: int = 4):
    raise NotImplementedError


def build_fast_weights_rnn(n_in: int, n_hidden: int, n_out: int,
                           lambda_decay: float = 0.95, eta: float = 0.5):
    raise NotImplementedError


def train(model, data, n_steps: int, lr: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
