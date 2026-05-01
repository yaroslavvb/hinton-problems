"""
Forward-Forward sequence learning on Aesop's Fables (Hinton 2022).

Two negative-data variants:
    1) teacher_forcing: real first 9 chars + model's wrong next-char.
    2) self_generated:  fully autoregressive rollout from first 10 chars.
"""

import numpy as np


ALPHABET = list("abcdefghijklmnopqrstuvwxyz ,;.")  # 30 symbols


def load_aesop_strings(n_strings: int = 248, str_len: int = 100):
    """Load 248 fixed-length character strings from Aesop's Fables."""
    raise NotImplementedError


def build_ff_seq_model(window: int = 10, n_hidden: int = 2000, n_layers: int = 3):
    raise NotImplementedError


def make_negatives_teacher_forcing(model, strings):
    raise NotImplementedError


def make_negatives_self_generated(model, strings):
    """Roll out autoregressively from the first 10 chars; treat all as negatives."""
    raise NotImplementedError


def train(model, strings, n_epochs: int, lr: float, negatives: str = "teacher_forcing"):
    raise NotImplementedError


def random_fixed_hidden_baseline(strings, window: int = 10):
    """Confirm that representation learning is essential."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
