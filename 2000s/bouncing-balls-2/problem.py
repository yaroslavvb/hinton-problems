"""
Two bouncing balls in a box (Sutskever & Hinton 2007).
"""

import numpy as np


def simulate_balls(n_steps: int, n_balls: int = 2, h: int = 30, w: int = 30,
                   ball_radius: float = 1.5):
    """Render T frames of n_balls with elastic wall collisions."""
    raise NotImplementedError


def build_trbm(n_visible: int, n_hidden: int):
    """RBM with directed temporal connections from prev hidden to current hidden / visible."""
    raise NotImplementedError


def train(model, sequences, n_epochs: int, lr: float):
    raise NotImplementedError


def rollout(model, init_frames, n_future: int) -> np.ndarray:
    raise NotImplementedError


if __name__ == "__main__":
    pass
