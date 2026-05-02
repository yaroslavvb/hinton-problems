"""
Bars problem for RBM training (Hinton 2000).

This file is kept for backwards-compatibility with the stub interface.
The real implementation lives in `bars_rbm.py`.
"""

import numpy as np

from bars_rbm import (
    BarsRBM,
    cd1_step,
    generate_bars,
    visualize_filters,
)


def build_rbm(n_visible: int, n_hidden: int) -> BarsRBM:
    """Construct an untrained BarsRBM. Convenience wrapper."""
    return BarsRBM(n_visible=n_visible, n_hidden=n_hidden)


__all__ = ["generate_bars", "build_rbm", "cd1_step", "visualize_filters",
           "BarsRBM"]


if __name__ == "__main__":
    # Tiny smoke test
    rbm = build_rbm(16, 8)
    data = generate_bars(64)
    print("data shape:", data.shape)
    cd1_step(rbm, data, lr=0.1)
    print("after one CD-1 step, |W| =", float(np.linalg.norm(rbm.W)))
