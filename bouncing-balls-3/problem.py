"""
Three bouncing balls (Sutskever, Hinton & Taylor 2008).

This module preserves the original stub signatures and re-exports the
working implementations from ``bouncing_balls_3.py`` so any code that
imports ``problem`` keeps working.
"""

from bouncing_balls_3 import (  # noqa: F401  (re-export)
    build_rtrbm,
    rollout,
    simulate_balls,
    train,
)


__all__ = ["simulate_balls", "build_rtrbm", "train", "rollout"]
