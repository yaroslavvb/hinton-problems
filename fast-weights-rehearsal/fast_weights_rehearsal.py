"""
Fast weights to deblur old memories (Hinton & Plaut 1987).

Source:
    G. E. Hinton and D. C. Plaut (1987), "Using Fast Weights to Deblur Old
    Memories", Proceedings of the Ninth Annual Conference of the Cognitive
    Science Society, pp. 177-186.

Problem:
    A linear associator with two-time-scale weights (slow plastic + fast
    elastic-decaying) is trained on a set A of random binary vector pairs,
    then on a disjoint set B. After learning B, recall on A drops because
    fast weights decay during B-learning and slow weights drift toward B.

Demonstration:
    A brief rehearsal of a SMALL SUBSET of A reactivates the memory of the
    *entire* set A. The slow weights still encode A in a "buried" form;
    rehearsal-driven fast-weight updates produce enough error correction to
    push every A pair back through threshold, including unrehearsed ones.

Algorithm:
    For each weight, store a slow component W_slow (small lr, no decay)
    and a fast component W_fast (large lr, multiplicative decay every
    presentation). The effective weight is W_eff = W_slow + W_fast.

    Online delta-rule update on each (x, y) presentation:
        out  = W_eff @ x
        err  = y - out
        dW   = eta_factor * outer(err, x)
        W_slow += slow_lr * dW
        W_fast += fast_lr * dW
        W_fast *= fast_decay

    Recall is the sign of W_eff @ x (patterns are in {-1, +1}).

Protocol (4 phases):
    1. Learn set A (n_a_sweeps full sweeps over set A).
    2. Learn set B (n_b_sweeps full sweeps over set B).
    3. Rehearse a small subset of A (n_rehearse_sweeps sweeps over the
       subset).
    4. Test recall on full A and full B without further updates.

This file: model + 4-phase protocol + multi-seed sweep + CLI.
"""

from __future__ import annotations

import argparse
import copy
import platform
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def generate_associations(n_pairs: int, dim: int, seed: int = 0
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Random {-1, +1} input/output vector pairs.

    Returns
    -------
    X : (n_pairs, dim) float64, entries in {-1, +1}
    Y : (n_pairs, dim) float64, entries in {-1, +1}
    """
    rng = np.random.default_rng(seed)
    X = rng.choice([-1.0, 1.0], size=(n_pairs, dim))
    Y = rng.choice([-1.0, 1.0], size=(n_pairs, dim))
    return X, Y


def generate_two_sets(n_pairs: int, dim: int, seed: int = 0
                      ) -> tuple[tuple[np.ndarray, np.ndarray],
                                 tuple[np.ndarray, np.ndarray]]:
    """Disjoint sets A and B (different RNG streams)."""
    A = generate_associations(n_pairs=n_pairs, dim=dim, seed=seed)
    B = generate_associations(n_pairs=n_pairs, dim=dim, seed=seed + 10_000)
    return A, B


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class FastWeightsAssociator:
    """Linear associator with slow + fast weight components.

    The effective weight matrix is W_eff = W_slow + W_fast.

    Parameters
    ----------
    dim : int
        Input/output dimensionality (square associator).
    slow_lr : float
        Learning rate for slow plastic weights.
    fast_lr : float
        Learning rate for fast elastic weights (typically ~5x slow_lr).
    fast_decay : float in (0, 1]
        Multiplicative decay applied to W_fast after every presentation.
        fast_decay=1.0 reduces to a single-time-scale associator.
    eta_factor : float
        Per-update normalization (defaults to 1/dim) so the delta rule
        is comparable across dimensions.
    """

    def __init__(self, dim: int,
                 slow_lr: float = 0.1,
                 fast_lr: float = 0.5,
                 fast_decay: float = 0.9,
                 eta_factor: float | None = None):
        self.dim = dim
        self.slow_lr = slow_lr
        self.fast_lr = fast_lr
        self.fast_decay = fast_decay
        self.eta_factor = (1.0 / dim) if eta_factor is None else eta_factor
        self.W_slow = np.zeros((dim, dim), dtype=np.float64)
        self.W_fast = np.zeros((dim, dim), dtype=np.float64)
        self._n_updates = 0

    # --- inference ---

    def W_effective(self) -> np.ndarray:
        return self.W_slow + self.W_fast

    def predict_raw(self, X: np.ndarray) -> np.ndarray:
        """Raw linear output X @ W_eff.T (no thresholding)."""
        return X @ self.W_effective().T

    def predict(self, X: np.ndarray) -> np.ndarray:
        """Sign-thresholded recall in {-1, +1}."""
        out = self.predict_raw(X)
        # treat exactly-zero as +1 so we don't bias toward never-seen
        return np.where(out >= 0.0, 1.0, -1.0)

    # --- learning ---

    def step(self, x: np.ndarray, y: np.ndarray) -> dict[str, float]:
        """One delta-rule update on a single (x, y) pair.

        Returns a small dict with the pre-update prediction error norm,
        useful for tracking learning curves.
        """
        W_eff = self.W_effective()
        out = W_eff @ x
        err = y - out
        update = self.eta_factor * np.outer(err, x)
        self.W_slow += self.slow_lr * update
        self.W_fast += self.fast_lr * update
        self.W_fast *= self.fast_decay
        self._n_updates += 1
        return {"err_l2": float(np.linalg.norm(err)),
                "pre_update_match_bits": float(np.mean(np.sign(out) == y))}

    def n_params(self) -> int:
        return self.W_slow.size + self.W_fast.size


def build_model(dim: int,
                slow_lr: float = 0.1,
                fast_lr: float = 0.5,
                fast_decay: float = 0.9,
                eta_factor: float | None = None,
                ) -> FastWeightsAssociator:
    """Each weight has a slow (plastic) and a fast (decaying) component."""
    return FastWeightsAssociator(dim=dim, slow_lr=slow_lr, fast_lr=fast_lr,
                                 fast_decay=fast_decay,
                                 eta_factor=eta_factor)


# ----------------------------------------------------------------------
# Recall metrics
# ----------------------------------------------------------------------

def recall_accuracy(model: FastWeightsAssociator,
                    data: tuple[np.ndarray, np.ndarray]) -> float:
    """Bit-wise recall accuracy: fraction of output bits matching target."""
    X, Y = data
    pred = model.predict(X)
    return float(np.mean(pred == Y))


def recall_pattern_accuracy(model: FastWeightsAssociator,
                            data: tuple[np.ndarray, np.ndarray]) -> float:
    """Pattern-wise: fraction of pairs where ALL bits match."""
    X, Y = data
    pred = model.predict(X)
    return float(np.mean(np.all(pred == Y, axis=1)))


def recall_per_pair(model: FastWeightsAssociator,
                    data: tuple[np.ndarray, np.ndarray]) -> np.ndarray:
    """Bit-wise accuracy per pair (length n_pairs)."""
    X, Y = data
    pred = model.predict(X)
    return np.mean(pred == Y, axis=1)


def fast_weight_norm(model: FastWeightsAssociator) -> float:
    return float(np.linalg.norm(model.W_fast))


def slow_weight_norm(model: FastWeightsAssociator) -> float:
    return float(np.linalg.norm(model.W_slow))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def learn_set(model: FastWeightsAssociator,
              data: tuple[np.ndarray, np.ndarray],
              n_sweeps: int,
              shuffle: bool = True,
              eval_data: dict | None = None,
              history: dict | None = None,
              phase_name: str = "",
              snapshot_every: int = 1,
              rng: np.random.Generator | None = None,
              ) -> dict:
    """Train `model` on `data` for `n_sweeps` epochs of online delta-rule.

    Each "sweep" is one pass through every pair (in shuffled order if
    `shuffle=True`). After each sweep we record bit-wise recall on all
    eval sets (if any) so plots can show A and B accuracies in lock-step.

    Parameters
    ----------
    eval_data : dict[str, (X, Y)] | None
        Extra subsets to track per sweep, e.g.
        {"A": (X_A, Y_A), "B": (X_B, Y_B)}.
    history : dict | None
        Existing history dict to append to (so the same dict can span
        multiple `learn_set` / `rehearse_subset` calls and produce a
        single timeline). If None, a new one is created and returned.
    snapshot_every : int
        Record metrics every Nth sweep (always also records the final
        sweep). 1 means every sweep.
    """
    if history is None:
        history = _new_history(eval_data)

    rng = rng if rng is not None else np.random.default_rng(0)

    X, Y = data
    n_pairs = X.shape[0]
    indices = np.arange(n_pairs)

    for sweep in range(n_sweeps):
        if shuffle:
            rng.shuffle(indices)
        for i in indices:
            model.step(X[i], Y[i])

        if (sweep % snapshot_every == 0) or (sweep == n_sweeps - 1):
            _record_history(history, model, phase_name=phase_name,
                            eval_data=eval_data)

    return history


def rehearse_subset(model: FastWeightsAssociator,
                    data: tuple[np.ndarray, np.ndarray],
                    subset_idx,
                    n_sweeps: int,
                    eval_data: dict | None = None,
                    history: dict | None = None,
                    snapshot_every: int = 1,
                    rng: np.random.Generator | None = None,
                    ) -> dict:
    """Brief replay of a few items in set A; fast weights should restore A.

    The mechanism: rehearsal pumps the fast weights again, but the slow
    weights still hold partial structure of all of A. The combination
    pulls every A pair back through threshold, not just the subset.
    """
    X, Y = data
    Xs, Ys = X[subset_idx], Y[subset_idx]
    return learn_set(model, (Xs, Ys), n_sweeps=n_sweeps,
                     shuffle=True, eval_data=eval_data,
                     history=history, phase_name="rehearse",
                     snapshot_every=snapshot_every, rng=rng)


# ----------------------------------------------------------------------
# History bookkeeping
# ----------------------------------------------------------------------

def _new_history(eval_data: dict | None) -> dict:
    h = {"phase": [], "sweep_global": [],
         "fast_norm": [], "slow_norm": []}
    if eval_data:
        for name in eval_data:
            h[f"acc_bit_{name}"] = []
            h[f"acc_pattern_{name}"] = []
    return h


def _record_history(history: dict,
                    model: FastWeightsAssociator,
                    phase_name: str,
                    eval_data: dict | None) -> None:
    history["phase"].append(phase_name)
    history["sweep_global"].append(len(history["phase"]))
    history["fast_norm"].append(fast_weight_norm(model))
    history["slow_norm"].append(slow_weight_norm(model))
    if eval_data:
        for name, d in eval_data.items():
            history[f"acc_bit_{name}"].append(recall_accuracy(model, d))
            history[f"acc_pattern_{name}"].append(recall_pattern_accuracy(model, d))


# ----------------------------------------------------------------------
# 4-phase protocol
# ----------------------------------------------------------------------

def run_protocol(seed: int = 0,
                 dim: int = 50,
                 n_pairs: int = 20,
                 n_rehearse: int | None = None,
                 slow_lr: float = 0.1,
                 fast_lr: float = 0.5,
                 fast_decay: float = 0.9,
                 n_a_sweeps: int = 30,
                 n_b_sweeps: int = 30,
                 n_rehearse_sweeps: int = 5,
                 snapshot_every: int = 1,
                 verbose: bool = False,
                 ) -> dict:
    """Full 4-phase protocol; returns a dict with model, history, summary.

    The "subset" rehearsed in phase 3 is the first `n_rehearse` pairs of
    set A (deterministic given the seed). Default: 25% of n_pairs.
    """
    if n_rehearse is None:
        n_rehearse = max(2, n_pairs // 4)
    if n_rehearse > n_pairs:
        raise ValueError(f"n_rehearse={n_rehearse} > n_pairs={n_pairs}")

    A, B = generate_two_sets(n_pairs=n_pairs, dim=dim, seed=seed)
    eval_data = {"A": A, "B": B}
    rng = np.random.default_rng(seed + 7919)

    model = build_model(dim=dim, slow_lr=slow_lr, fast_lr=fast_lr,
                        fast_decay=fast_decay)

    # Initial baseline (phase 0: untrained)
    history = _new_history(eval_data)
    _record_history(history, model, phase_name="init", eval_data=eval_data)
    init_acc_A = history["acc_bit_A"][-1]
    init_acc_B = history["acc_bit_B"][-1]
    if verbose:
        print(f"# init        | recall_A_bit={init_acc_A*100:5.1f}%  "
              f"recall_B_bit={init_acc_B*100:5.1f}%")

    # ---- phase 1: learn A ----
    learn_set(model, A, n_sweeps=n_a_sweeps, shuffle=True,
              eval_data=eval_data, history=history,
              phase_name="learn_A", snapshot_every=snapshot_every, rng=rng)
    after_A_acc_A = history["acc_bit_A"][-1]
    after_A_acc_B = history["acc_bit_B"][-1]
    after_A_pattern_A = history["acc_pattern_A"][-1]
    if verbose:
        print(f"# learned A   | recall_A_bit={after_A_acc_A*100:5.1f}%  "
              f"pattern_A={after_A_pattern_A*100:5.1f}%  "
              f"recall_B_bit={after_A_acc_B*100:5.1f}%")

    # snapshot model (for visualization needs)
    model_after_A = copy.deepcopy(model)

    # ---- phase 2: learn B ----
    learn_set(model, B, n_sweeps=n_b_sweeps, shuffle=True,
              eval_data=eval_data, history=history,
              phase_name="learn_B", snapshot_every=snapshot_every, rng=rng)
    after_B_acc_A = history["acc_bit_A"][-1]
    after_B_acc_B = history["acc_bit_B"][-1]
    after_B_pattern_A = history["acc_pattern_A"][-1]
    after_B_pattern_B = history["acc_pattern_B"][-1]
    if verbose:
        print(f"# learned B   | recall_A_bit={after_B_acc_A*100:5.1f}%  "
              f"pattern_A={after_B_pattern_A*100:5.1f}%  "
              f"recall_B_bit={after_B_acc_B*100:5.1f}%  "
              f"pattern_B={after_B_pattern_B*100:5.1f}%")

    model_after_B = copy.deepcopy(model)

    # ---- phase 3: rehearse subset of A ----
    subset_idx = list(range(n_rehearse))
    rehearse_subset(model, A, subset_idx=subset_idx,
                    n_sweeps=n_rehearse_sweeps,
                    eval_data=eval_data, history=history,
                    snapshot_every=snapshot_every, rng=rng)
    after_R_acc_A = history["acc_bit_A"][-1]
    after_R_acc_B = history["acc_bit_B"][-1]
    after_R_pattern_A = history["acc_pattern_A"][-1]
    after_R_pattern_B = history["acc_pattern_B"][-1]
    if verbose:
        print(f"# rehearsed   | recall_A_bit={after_R_acc_A*100:5.1f}%  "
              f"pattern_A={after_R_pattern_A*100:5.1f}%  "
              f"recall_B_bit={after_R_acc_B*100:5.1f}%  "
              f"pattern_B={after_R_pattern_B*100:5.1f}%")

    model_after_R = copy.deepcopy(model)

    # ---- phase 4: test (no updates; record one final marker) ----
    _record_history(history, model, phase_name="test",
                    eval_data=eval_data)

    # Per-pair breakdown so the viz can split rehearsed vs unrehearsed.
    per_pair_A_after_B = recall_per_pair(model_after_B, A)
    per_pair_A_after_R = recall_per_pair(model_after_R, A)
    rehearsed_mask = np.zeros(n_pairs, dtype=bool)
    rehearsed_mask[subset_idx] = True

    summary = {
        "seed": seed, "dim": dim, "n_pairs": n_pairs,
        "n_rehearse": n_rehearse,
        "slow_lr": slow_lr, "fast_lr": fast_lr, "fast_decay": fast_decay,
        "n_a_sweeps": n_a_sweeps, "n_b_sweeps": n_b_sweeps,
        "n_rehearse_sweeps": n_rehearse_sweeps,
        # phase-end accuracies
        "init_bit_A": init_acc_A,
        "init_bit_B": init_acc_B,
        "after_A_bit_A": after_A_acc_A,
        "after_A_bit_B": after_A_acc_B,
        "after_A_pattern_A": after_A_pattern_A,
        "after_B_bit_A": after_B_acc_A,
        "after_B_bit_B": after_B_acc_B,
        "after_B_pattern_A": after_B_pattern_A,
        "after_B_pattern_B": after_B_pattern_B,
        "after_R_bit_A": after_R_acc_A,
        "after_R_bit_B": after_R_acc_B,
        "after_R_pattern_A": after_R_pattern_A,
        "after_R_pattern_B": after_R_pattern_B,
        # the headline numbers
        "deblur_recovery_bits_A": after_R_acc_A - after_B_acc_A,
        "deblur_recovery_pattern_A": after_R_pattern_A - after_B_pattern_A,
        # per-pair breakdown
        "rehearsed_pair_recovery_bits": float(np.mean(
            per_pair_A_after_R[rehearsed_mask] - per_pair_A_after_B[rehearsed_mask])),
        "unrehearsed_pair_recovery_bits": float(np.mean(
            per_pair_A_after_R[~rehearsed_mask] - per_pair_A_after_B[~rehearsed_mask])),
        "n_params": model.n_params(),
    }
    return {"model": model,
            "model_after_A": model_after_A,
            "model_after_B": model_after_B,
            "model_after_R": model_after_R,
            "history": history,
            "summary": summary,
            "data": {"A": A, "B": B,
                     "subset_idx": subset_idx,
                     "rehearsed_mask": rehearsed_mask,
                     "per_pair_A_after_B": per_pair_A_after_B,
                     "per_pair_A_after_R": per_pair_A_after_R}}


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(n_seeds: int = 10, **kwargs) -> dict:
    """Repeat the protocol over `n_seeds` and aggregate the headline numbers."""
    rows = []
    for s in range(n_seeds):
        out = run_protocol(seed=s, snapshot_every=10**6, verbose=False,
                           **kwargs)
        rows.append(out["summary"])

    def stat(key: str):
        v = np.array([r[key] for r in rows])
        return {"mean": float(v.mean()), "std": float(v.std()),
                "min": float(v.min()), "max": float(v.max())}

    keys = ["after_A_bit_A", "after_A_pattern_A",
            "after_B_bit_A", "after_B_bit_B",
            "after_B_pattern_A", "after_B_pattern_B",
            "after_R_bit_A", "after_R_pattern_A",
            "deblur_recovery_bits_A", "deblur_recovery_pattern_A",
            "rehearsed_pair_recovery_bits",
            "unrehearsed_pair_recovery_bits"]
    return {"n_seeds": n_seeds, "rows": rows,
            "stats": {k: stat(k) for k in keys}}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--n-rehearse", type=int, default=None,
                   help="number of A pairs rehearsed in phase 3 "
                        "(default: 25%% of n_pairs)")
    p.add_argument("--slow-lr", type=float, default=0.1)
    p.add_argument("--fast-lr", type=float, default=0.5)
    p.add_argument("--fast-decay", type=float, default=0.9)
    p.add_argument("--n-a-sweeps", type=int, default=30)
    p.add_argument("--n-b-sweeps", type=int, default=30)
    p.add_argument("--n-rehearse-sweeps", type=int, default=5)
    p.add_argument("--sweep", type=int, default=0,
                   help="if > 0, run that many seeds and aggregate")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _print_environment()

    if args.sweep > 0:
        t0 = time.time()
        out = sweep(n_seeds=args.sweep,
                    dim=args.dim, n_pairs=args.n_pairs,
                    n_rehearse=args.n_rehearse,
                    slow_lr=args.slow_lr, fast_lr=args.fast_lr,
                    fast_decay=args.fast_decay,
                    n_a_sweeps=args.n_a_sweeps,
                    n_b_sweeps=args.n_b_sweeps,
                    n_rehearse_sweeps=args.n_rehearse_sweeps)
        dt = time.time() - t0
        print(f"\nSweep results ({args.sweep} seeds):")
        for k, s in out["stats"].items():
            print(f"  {k:34s}  mean={s['mean']*100:7.2f}%  "
                  f"std={s['std']*100:5.2f}pp  "
                  f"min={s['min']*100:6.2f}%  max={s['max']*100:6.2f}%")
        print(f"  total wallclock: {dt:.2f}s")
        return

    t0 = time.time()
    out = run_protocol(seed=args.seed,
                       dim=args.dim, n_pairs=args.n_pairs,
                       n_rehearse=args.n_rehearse,
                       slow_lr=args.slow_lr, fast_lr=args.fast_lr,
                       fast_decay=args.fast_decay,
                       n_a_sweeps=args.n_a_sweeps,
                       n_b_sweeps=args.n_b_sweeps,
                       n_rehearse_sweeps=args.n_rehearse_sweeps,
                       snapshot_every=1,
                       verbose=not args.quiet)
    dt = time.time() - t0

    s = out["summary"]
    print("\n=== Phase 1: learn A ===")
    print(f"  bit acc on A : {s['after_A_bit_A']*100:6.2f}%")
    print(f"  pat acc on A : {s['after_A_pattern_A']*100:6.2f}%")
    print(f"  bit acc on B : {s['after_A_bit_B']*100:6.2f}%   (untrained)")
    print("\n=== Phase 2: learn B (interferes with A) ===")
    print(f"  bit acc on A : {s['after_B_bit_A']*100:6.2f}%   (drops from "
          f"{s['after_A_bit_A']*100:.2f}%)")
    print(f"  pat acc on A : {s['after_B_pattern_A']*100:6.2f}%")
    print(f"  bit acc on B : {s['after_B_bit_B']*100:6.2f}%")
    print(f"  pat acc on B : {s['after_B_pattern_B']*100:6.2f}%")
    print("\n=== Phase 3: rehearse "
          f"{s['n_rehearse']} of {s['n_pairs']} A pairs ===")
    print(f"  bit acc on A : {s['after_R_bit_A']*100:6.2f}%   (recovers from "
          f"{s['after_B_bit_A']*100:.2f}%)")
    print(f"  pat acc on A : {s['after_R_pattern_A']*100:6.2f}%")
    print(f"  bit acc on B : {s['after_R_bit_B']*100:6.2f}%")
    print(f"  pat acc on B : {s['after_R_pattern_B']*100:6.2f}%")

    print("\n=== Phase 4: test (no further updates) ===")
    print(f"  recall_A bit : {s['after_R_bit_A']*100:6.2f}%")
    print(f"  recall_B bit : {s['after_R_bit_B']*100:6.2f}%")

    print("\n--- Headline ---")
    print(f"  deblur recovery (A bit acc) : "
          f"{s['after_B_bit_A']*100:.2f}% -> {s['after_R_bit_A']*100:.2f}%   "
          f"(+{s['deblur_recovery_bits_A']*100:.2f} pp)")
    print(f"  deblur recovery (A pattern) : "
          f"{s['after_B_pattern_A']*100:.2f}% -> {s['after_R_pattern_A']*100:.2f}%   "
          f"(+{s['deblur_recovery_pattern_A']*100:.2f} pp)")
    print(f"  rehearsed pairs   recovery  : "
          f"{s['rehearsed_pair_recovery_bits']*100:+.2f} pp (bit)")
    print(f"  unrehearsed pairs recovery  : "
          f"{s['unrehearsed_pair_recovery_bits']*100:+.2f} pp (bit)")

    print(f"\nWallclock: {dt:.3f}s   params: {s['n_params']}  "
          f"slow_lr={s['slow_lr']} fast_lr={s['fast_lr']} fast_decay={s['fast_decay']}")


if __name__ == "__main__":
    main()
