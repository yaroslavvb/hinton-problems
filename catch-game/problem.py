"""
Catch with delayed-blank observations (Ba et al. 2016).
"""

import numpy as np


class CatchEnv:
    def __init__(self, size: int = 24, blank_after: int = 8):
        raise NotImplementedError

    def reset(self) -> np.ndarray:
        raise NotImplementedError

    def step(self, action: int):
        """Returns (obs, reward, done). Obs blanked once t > blank_after."""
        raise NotImplementedError


def build_a3c_policy(obs_dim: int, n_actions: int = 3):
    raise NotImplementedError


def train_a3c(env_cls, model, n_episodes: int, lr: float):
    raise NotImplementedError


if __name__ == "__main__":
    pass
