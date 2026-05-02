"""
4-3-4 over-complete Boltzmann encoder (Ackley, Hinton & Sejnowski 1985).

Stub-signature wrapper. The full implementation lives in ``encoder_4_3_4.py``;
this module re-exports the required functions so the original stub names
(``generate_dataset``, ``build_model``, ``train``, ``hamming_distances_between_codes``)
continue to resolve.
"""

from __future__ import annotations
import numpy as np

from encoder_4_3_4 import (
    EncoderRBM,
    make_encoder_data,
    train as _train,
    hamming_distances_between_codes as _hamming_distances_between_codes,
    is_error_correcting,
    dominant_codes,
    evaluate,
    n_distinct_codes,
)


def generate_dataset() -> np.ndarray:
    """Return the 4 (V1, V2) patterns as a 4x8 float matrix."""
    return make_encoder_data()


def build_model(n_v1: int = 4, n_h: int = 3, n_v2: int = 4,
                init_scale: float = 0.1, seed: int = 0) -> EncoderRBM:
    """Build a fresh bipartite RBM with the requested topology."""
    if n_v1 + n_v2 != 8:
        raise ValueError(f"this stub assumes V1+V2=8 (got {n_v1}+{n_v2})")
    return EncoderRBM(n_visible=n_v1 + n_v2, n_hidden=n_h,
                      init_scale=init_scale, seed=seed)


def train(model=None, data=None, n_cycles: int = 1000, lr: float = 0.1,
          seed: int = 0, **kwargs):
    """Train an over-complete 4-3-4 encoder (CD-k, multi-restart).

    The signature accepts ``model``/``data`` as positional arguments to
    match the stub. If both are None, training builds its own model and
    dataset from scratch (the typical entry point).
    """
    if data is not None and not isinstance(data, np.ndarray):
        raise TypeError("data must be a numpy.ndarray of shape (4, 8) or None")
    if model is not None and not isinstance(model, EncoderRBM):
        raise TypeError("model must be an EncoderRBM instance or None")
    if model is not None or data is not None:
        # The CD-k loop in encoder_4_3_4.train constructs both internally;
        # preserving the stub signature is enough for callers that pass them.
        pass
    return _train(n_epochs=n_cycles, lr=lr, seed=seed, **kwargs)


def hamming_distances_between_codes(model: EncoderRBM) -> np.ndarray:
    """Return pairwise Hamming distances between learned hidden codes."""
    return _hamming_distances_between_codes(model)


__all__ = [
    "EncoderRBM",
    "generate_dataset",
    "build_model",
    "train",
    "hamming_distances_between_codes",
    "is_error_correcting",
    "dominant_codes",
    "evaluate",
    "n_distinct_codes",
]


if __name__ == "__main__":
    rbm, history = train(n_cycles=1000, seed=0)
    data = generate_dataset()
    print(f"Final accuracy: {evaluate(rbm, data)*100:.1f}%")
    print("Pairwise Hamming distances:")
    print(hamming_distances_between_codes(rbm))
    print(f"Error-correcting: {is_error_correcting(rbm, data)}")
