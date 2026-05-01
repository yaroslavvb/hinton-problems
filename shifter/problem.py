"""
Shifter problem (Hinton & Sejnowski 1986).

V1: 8 random bits (p=0.3).
V2: V1 shifted left/none/right with wraparound.
V3: 3 one-hot units = shift direction.
"""

import numpy as np


def generate_dataset(n_samples: int, p_on: float = 0.3, n_bits: int = 8):
    """Return (V1, V2, V3) where V3 is one-hot shift in {-1, 0, +1}."""
    raise NotImplementedError


def build_model(n_visible: int = 19, n_hidden: int = 24):
    """Boltzmann machine with 8+8+3 = 19 visible units."""
    raise NotImplementedError


def train(model, data, n_cycles: int = 9000, lr: float = 0.05):
    raise NotImplementedError


def shift_recognition_accuracy(model, data) -> float:
    """Clamp V1, V2; sample V3 from p(V3 | V1, V2)."""
    raise NotImplementedError


if __name__ == "__main__":
    pass
