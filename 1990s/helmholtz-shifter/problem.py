"""
Helmholtz-machine shifter (Dayan, Hinton, Neal & Zemel 1995).
"""

import numpy as np


def generate_dataset(n_samples: int, w: int = 8):
    """4x8 patterns; row 0 random, row 3 shifted, rows 1-2 duplicates."""
    raise NotImplementedError


def build_helmholtz_machine(n_visible: int, n_hidden_2: int, n_hidden_3: int = 1):
    raise NotImplementedError


def wake_sleep(model, data, n_passes: int, lr: float):
    raise NotImplementedError


def inspect_layer3_units(model):
    """After training, expect shift-direction-selective units."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
