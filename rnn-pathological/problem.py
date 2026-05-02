"""
Hochreiter-Schmidhuber pathological long-term-dependency RNN tasks
(Sutskever, Martens, Dahl & Hinton 2013).

Thin re-export of the real implementation in ``rnn_pathological.py``.
This module exists so the standard stub interface
(`generate_dataset`, `build_rnn`, `train_with_momentum`) is importable
from `problem.py` for cataloguing tools that scan all stubs uniformly.
For the full API, CLI, and `--all` headline experiment, use
``rnn_pathological.py`` directly.
"""

import numpy as np

from rnn_pathological import (  # re-exports
    TASKS,
    TASK_SPEC,
    RNN,
    generate_dataset,
    train_with_momentum,
    chance_baseline,
)


def build_rnn(n_in: int, n_hidden: int, n_out: int,
              init: str = "ortho", seed: int = 0) -> RNN:
    """Convenience wrapper matching the stub signature."""
    return RNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
               init=init, seed=seed)


__all__ = [
    "TASKS",
    "TASK_SPEC",
    "RNN",
    "generate_dataset",
    "build_rnn",
    "train_with_momentum",
    "chance_baseline",
]


if __name__ == "__main__":
    # Sanity: import + smoke-generate one batch from each task
    rng = np.random.default_rng(0)
    for task in TASKS:
        x, y = generate_dataset(task, sequence_len=20, n_samples=2, rng=rng)
        print(f"{task:<24s}  x.shape={x.shape}  y.shape={y.shape}")
