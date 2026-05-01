"""
Three bouncing balls (Sutskever, Hinton & Taylor 2008).
"""

import numpy as np


def simulate_balls(n_steps: int = 100, n_balls: int = 3, h: int = 30, w: int = 30):
    raise NotImplementedError


def build_rtrbm(n_visible: int, n_hidden: int):
    """Recurrent connections between hidden states across timesteps."""
    raise NotImplementedError


def train(model, sequences, n_epochs: int, lr: float):
    raise NotImplementedError


def rollout(model, init_frames, n_future: int) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    pass
