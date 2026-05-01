"""
Synthetic-spectrogram riser/non-riser discrimination
(Plaut & Hinton 1987).
"""

import numpy as np


def generate_dataset(n_samples: int, n_freq: int = 6, n_time: int = 9, noise_std: float = 1.0):
    """Half rising frequency tracks, half non-rising, with Gaussian noise."""
    raise NotImplementedError


def build_model(n_in: int = 54, n_hidden: int = 24, n_out: int = 2):
    raise NotImplementedError


def train(model, data, n_sweeps: int, lr: float = 0.1):
    raise NotImplementedError


def bayes_optimal_accuracy(noise_std: float) -> float:
    raise NotImplementedError


if __name__ == "__main__":
    pass
