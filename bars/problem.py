"""
Bars task / wake-sleep (Hinton, Dayan, Frey & Neal 1995).
"""

import numpy as np


def generate_bars(n_samples: int, h: int = 4, w: int = 4,
                  p_vertical: float = 2/3, p_bar: float = 0.2):
    """4x4 binary bar images with hierarchical priors."""
    raise NotImplementedError


def build_helmholtz(n_visible: int = 16, n_hidden_1: int = 8, n_hidden_2: int = 1):
    raise NotImplementedError


def wake_sleep(model, data, n_steps: int, lr: float):
    raise NotImplementedError


def asymmetric_kl(model, true_dist) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
