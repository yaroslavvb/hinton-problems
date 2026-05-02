"""
N-bit parity backprop, faithful to Rumelhart, Hinton & Williams (1986),
"Learning internal representations by error propagation" (PDP Vol. 1, Ch. 8).

Problem:
    Inputs: all 2**N binary patterns of length N (N in 2..8).
    Target: parity of the input — 1 if an odd number of bits are on, else 0.

    Parity is the canonical hard Boolean function: it cannot be approximated
    by any subset of inputs (every bit matters), and it requires k-th-order
    interaction detection across all bits. Single-layer perceptrons cannot
    learn it; the standard MLP construction with N hidden units does.

The interesting property — the "thermometer code"
    With exactly N hidden sigmoid units, the network reliably learns a
    "thermometer code": each hidden unit h_k fires when at least k of the
    N inputs are on (k = 1..N). The output unit then computes parity from
    the staircase by using alternating-sign weights into the output layer
    (h_1 - h_2 + h_3 - h_4 + ...), which produces a value ~0 when an even
    number of inputs are on and ~1 when an odd number are. This is the
    minimal hidden-layer construction for parity.

Architecture:
    N inputs -> N hidden sigmoids -> 1 output sigmoid.
    Number of params: N*N + N (W1+b1) + N + 1 (W2+b2) = N^2 + 2N + 1.

Hyperparameters (RHW1986 used eta=0.5, momentum=0.9 for parity tasks):
    learning rate eta = 0.5, momentum alpha = 0.9, full-batch updates over
    all 2**N patterns. Convergence: every output within 0.5 of its target.

This file is a numpy reproduction. Train with backprop + momentum, then
inspect the hidden-unit code to verify the thermometer pattern.
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_parity_data(n_bits: int, bipolar: bool = True
                     ) -> tuple[np.ndarray, np.ndarray]:
    """All 2**n_bits binary patterns labeled by parity.

    Parameters
    ----------
    n_bits : number of input bits.
    bipolar : if True, encode bits as {-1, +1} (recommended). If False,
        encode as {0, 1}. Bipolar trains far more reliably on parity
        because the all-zero pattern still drives every hidden unit.

    Returns
    -------
    X : (2**n_bits, n_bits) float64 in {-1, +1} (bipolar) or {0, 1}.
    y : (2**n_bits, 1) float64 in {0, 1} — 1 iff odd number of "on" bits.
    """
    n = 2 ** n_bits
    bits = np.zeros((n, n_bits), dtype=np.float64)
    for i in range(n):
        for b in range(n_bits):
            bits[i, b] = (i >> b) & 1
    y = (bits.sum(axis=1).astype(int) % 2).astype(np.float64).reshape(-1, 1)
    X = (2.0 * bits - 1.0) if bipolar else bits
    return X, y


def bit_count_for_inputs(X: np.ndarray) -> np.ndarray:
    """Number of 'on' bits per input row (works for either encoding)."""
    if X.min() < 0.0:
        return ((X > 0).sum(axis=1)).astype(int)
    return X.sum(axis=1).astype(int)


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def dsigmoid_from_y(y: np.ndarray) -> np.ndarray:
    """y * (1 - y); uses the post-activation."""
    return y * (1.0 - y)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class ParityMLP:
    """N -> H -> 1 fully-connected sigmoid net (default H = N)."""

    def __init__(self, n_bits: int, n_hidden: int | None = None,
                 init_scale: float = 1.0, seed: int = 0,
                 bipolar: bool = True, spread_biases: bool = True):
        """spread_biases=True initializes b1 with a deterministic linear
        spread across the input bit-count range. This breaks the symmetry
        between hidden units and biases the basin-of-attraction toward
        the thermometer-code solution. Each unit starts with a different
        'preferred' threshold; the random W1 chooses the direction.
        """
        if n_hidden is None:
            n_hidden = n_bits
        self.n_bits = n_bits
        self.n_hidden = n_hidden
        self.bipolar = bipolar
        self.rng = np.random.default_rng(seed)
        # uniform [-init_scale/2, +init_scale/2] for weights
        self.W1 = init_scale * (self.rng.random((n_hidden, n_bits)) - 0.5)
        if spread_biases:
            # Linearly spread bias offsets across the {-1, +1}^N or {0, 1}^N
            # input sum range so different hidden units have different
            # natural thresholds. Tiny random jitter to break ties.
            if bipolar:
                # input sums range over {-N, -N+2, ..., +N}
                offsets = np.linspace(-n_bits, n_bits, n_hidden)
            else:
                offsets = np.linspace(-n_bits, 0.0, n_hidden)
            jitter = 0.1 * init_scale * (self.rng.random(n_hidden) - 0.5)
            self.b1 = -offsets + jitter
        else:
            self.b1 = init_scale * (self.rng.random((n_hidden,)) - 0.5)
        self.W2 = init_scale * (self.rng.random((1, n_hidden)) - 0.5)
        self.b2 = init_scale * (self.rng.random((1,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (h, o). X is (n, n_bits)."""
        h = sigmoid(X @ self.W1.T + self.b1)
        o = sigmoid(h @ self.W2.T + self.b2)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        return (self.W1.size + self.b1.size
                + self.W2.size + self.b2.size)


# ----------------------------------------------------------------------
# Backprop step
# ----------------------------------------------------------------------

def backprop_grads(model: ParityMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Mean-squared-error gradients (loss = 0.5 * mean (o - y)^2)."""
    n = X.shape[0]
    h, o = model.forward(X)

    # output layer
    delta_o = (o - y) * dsigmoid_from_y(o)        # (n, 1)
    dW2 = delta_o.T @ h / n                        # (1, H)
    db2 = delta_o.mean(axis=0)                     # (1,)

    # hidden layer
    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h)   # (n, H)
    dW1 = delta_h.T @ X / n                        # (H, N)
    db1 = delta_h.mean(axis=0)                     # (H,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_mse(model: ParityMLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean((o - y) ** 2))


def accuracy(model: ParityMLP, X: np.ndarray, y: np.ndarray,
             threshold: float = 0.5) -> float:
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(pred == y))


def converged(model: ParityMLP, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    """RHW1986 criterion: every output within `tol` of its target."""
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_bits: int = 4,
          n_hidden: int | None = None,
          lr: float = 0.5,
          momentum: float = 0.9,
          init_scale: float = 1.0,
          max_epochs: int = 20000,
          seed: int = 0,
          bipolar: bool = True,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True,
          early_stop_after: int = 50,
          ) -> tuple[ParityMLP, dict]:
    """Train an N -> H -> 1 MLP on N-bit parity. Full-batch backprop+momentum.

    Returns (trained_model, history). history["converged_epoch"] is the
    first epoch where every output is within 0.5 of its target, or None.
    """
    model = ParityMLP(n_bits=n_bits, n_hidden=n_hidden,
                      init_scale=init_scale, seed=seed, bipolar=bipolar)
    X, y = make_parity_data(n_bits, bipolar=bipolar)

    velocities = {k: np.zeros_like(v) for k, v in
                  [("W1", model.W1), ("b1", model.b1),
                   ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "loss": [], "accuracy": [],
               "weight_norm": [], "converged_epoch": None}

    if verbose:
        print(f"# {n_bits}-bit parity backprop  "
              f"hidden={model.n_hidden}  params={model.n_params()}  "
              f"lr={lr}  momentum={momentum}  seed={seed}")
        print(f"# patterns: {2**n_bits}  "
              f"target distribution: {int(y.sum())} ones, "
              f"{len(y) - int(y.sum())} zeros")

    for epoch in range(max_epochs):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        loss = loss_mse(model, X, y)
        acc = accuracy(model, X, y)
        wn = float(np.linalg.norm(np.concatenate(
            [model.W1.ravel(), model.W2.ravel()])))

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["weight_norm"].append(wn)

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at epoch {epoch + 1}  "
                      f"loss={loss:.4f}  acc={acc*100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == max_epochs - 1):
            snapshot_callback(epoch, model, history)

        log_step = max(max_epochs // 10, 200)
        if verbose and (epoch % log_step == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch+1:6d}  loss={loss:.4f}  "
                  f"acc={acc*100:5.1f}%  |W|={wn:.3f}")

        # Stop a bit past convergence so curves don't flatline forever.
        if (history["converged_epoch"] is not None
                and epoch + 1 >= history["converged_epoch"] + early_stop_after):
            break

    return model, history


# ----------------------------------------------------------------------
# Hidden-code inspection (the thermometer-code claim)
# ----------------------------------------------------------------------

def thermometer_score(model: ParityMLP) -> dict:
    """Quantify how well the trained hidden layer matches a thermometer code.

    For each hidden unit, we identify its "threshold" k* — the smallest
    bit-count at which it is on across the training patterns. A perfect
    thermometer code has hidden units with thresholds {1, 2, ..., N}, all
    distinct. We return the sorted thresholds, the unique-count, and the
    binarized hidden activations grouped by input bit-count.
    """
    X, _ = make_parity_data(model.n_bits, bipolar=model.bipolar)
    h, _ = model.forward(X)                                # (2^N, H)
    bit_counts = bit_count_for_inputs(X)                   # 0..N
    levels = np.arange(model.n_bits + 1)

    # Per-unit mean activation at each bit-count level.
    mean_by_level = np.zeros((model.n_hidden, len(levels)))
    for li, lv in enumerate(levels):
        mask = bit_counts == lv
        if mask.any():
            mean_by_level[:, li] = h[mask].mean(axis=0)

    # Threshold for each hidden unit: smallest bit-count where mean act > 0.5.
    # We also report the "polarity" — whether the unit fires more when many
    # bits are on (+1) or when few are on (-1). Either is a thermometer
    # direction; the alternating sign in W2 absorbs the choice.
    thresholds, polarities, monotonicity = [], [], []
    for hi in range(model.n_hidden):
        row = mean_by_level[hi]
        # polarity: +1 if mean act at high bit-count > at low bit-count
        polarities.append(1 if row[-1] > row[0] else -1)
        # threshold: smallest level where unit is "on" (using its polarity)
        if polarities[-1] == 1:
            on_levels = np.where(row > 0.5)[0]
            thresholds.append(int(on_levels[0]) if on_levels.size > 0
                              else model.n_bits + 1)
        else:
            on_levels = np.where(row > 0.5)[0]
            # for negative-polarity, threshold = first level where it goes off
            off_levels = np.where(row < 0.5)[0]
            thresholds.append(int(off_levels[0]) if off_levels.size > 0
                              else model.n_bits + 1)
        # monotonicity: 1.0 if perfectly monotonic across bit-counts, lower
        # if the curve has reversals. Computed as (max correlation with
        # +ramp or -ramp) using Spearman-like tau.
        sorted_idx = np.argsort(row)
        # fraction of pairs (i, j) with row[i] < row[j] when bit_count[i] <
        # bit_count[j]
        lvl = np.arange(len(row))
        agree = 0; total = 0
        for i in range(len(row)):
            for j in range(i + 1, len(row)):
                if row[i] != row[j]:
                    total += 1
                    if (row[i] < row[j]) == (lvl[i] < lvl[j]):
                        agree += 1
        # |2 * agreement - 1| in [0, 1]: 1 = perfectly monotonic (either
        # direction), 0 = no monotonic signal
        monotonicity.append(abs(2.0 * agree / max(total, 1) - 1.0))

    return {
        "mean_by_level": mean_by_level,
        "bit_counts": bit_counts,
        "levels": levels,
        "thresholds": thresholds,
        "polarities": polarities,
        "monotonicity": monotonicity,
        "mean_monotonicity": float(np.mean(monotonicity)),
        "unique_thresholds": len(set(thresholds)),
        "is_thermometer": (sorted(thresholds) ==
                           list(range(1, model.n_bits + 1))),
    }


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(n_bits: int, n_seeds: int, lr: float, momentum: float,
          init_scale: float, max_epochs: int, bipolar: bool = True) -> dict:
    epochs = []
    failures = []
    thermo_hits = 0
    for s in range(n_seeds):
        model, hist = train(n_bits=n_bits, lr=lr, momentum=momentum,
                            init_scale=init_scale, max_epochs=max_epochs,
                            seed=s, bipolar=bipolar, verbose=False)
        if hist["converged_epoch"] is None:
            failures.append(s)
        else:
            epochs.append(hist["converged_epoch"])
            if thermometer_score(model)["is_thermometer"]:
                thermo_hits += 1
    return {"n_bits": n_bits, "n_seeds": n_seeds,
            "converged": len(epochs), "failed": len(failures),
            "failed_seeds": failures,
            "thermometer_hits": thermo_hits,
            "mean_epochs": float(np.mean(epochs)) if epochs else float("nan"),
            "median_epochs": float(np.median(epochs)) if epochs else float("nan"),
            "min_epochs": int(min(epochs)) if epochs else -1,
            "max_epochs": int(max(epochs)) if epochs else -1,
            "epochs": epochs}


def sweep_n(n_seeds: int, n_bits_range: tuple[int, int],
            lr: float, momentum: float, init_scale: float,
            max_epochs: int, bipolar: bool = True) -> list[dict]:
    """Sweep over n_bits for a given seed budget. Returns list of summaries."""
    summaries = []
    for n in range(n_bits_range[0], n_bits_range[1] + 1):
        s = sweep(n_bits=n, n_seeds=n_seeds, lr=lr, momentum=momentum,
                  init_scale=init_scale, max_epochs=max_epochs,
                  bipolar=bipolar)
        summaries.append(s)
    return summaries


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-bits", type=int, default=4,
                   help="number of input bits (recommended 2..8)")
    p.add_argument("--n-hidden", type=int, default=None,
                   help="hidden units (default = n_bits)")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=20000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", type=int, default=0,
                   help="If > 0, run a sweep across this many seeds.")
    p.add_argument("--sweep-n", type=str, default="",
                   help="If non-empty (e.g. '2-8'), sweep over n_bits in "
                        "that range, n_seeds per N from --sweep (default 5).")
    p.add_argument("--encoding", choices=["bipolar", "binary"],
                   default="bipolar",
                   help="Input encoding. 'bipolar' = {-1,+1} (default, "
                        "trains far more reliably). 'binary' = {0,1}.")
    args = p.parse_args()
    bipolar = (args.encoding == "bipolar")

    print(f"# python  : {sys.version.split()[0]}")
    print(f"# numpy   : {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    if args.sweep_n:
        lo, hi = (int(x) for x in args.sweep_n.split("-"))
        n_seeds = args.sweep if args.sweep > 0 else 5
        t0 = time.time()
        summaries = sweep_n(n_seeds=n_seeds, n_bits_range=(lo, hi),
                            lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs,
                            bipolar=bipolar)
        dt = time.time() - t0
        print(f"\nSweep over N = {lo}..{hi}, {n_seeds} seeds each:")
        print(f"{'N':>2}  {'conv':>6}  {'thermo':>6}  "
              f"{'median ep':>10}  {'min':>6}  {'max':>6}")
        for s in summaries:
            med = (f"{s['median_epochs']:>10.0f}"
                   if s["converged"] > 0 else f"{'-':>10}")
            print(f"{s['n_bits']:>2}  "
                  f"{s['converged']:>3}/{s['n_seeds']:<2}  "
                  f"{s['thermometer_hits']:>3}/{s['converged']:<2}  "
                  f"{med}  {s['min_epochs']:>6}  {s['max_epochs']:>6}")
        print(f"\nTotal time: {dt:.1f}s")
        return

    if args.sweep > 0:
        t0 = time.time()
        summary = sweep(n_bits=args.n_bits, n_seeds=args.sweep, lr=args.lr,
                        momentum=args.momentum, init_scale=args.init_scale,
                        max_epochs=args.max_epochs, bipolar=bipolar)
        dt = time.time() - t0
        print(f"\nSweep results (N={args.n_bits}, {args.sweep} seeds):")
        print(f"  converged       : {summary['converged']}/{summary['n_seeds']}")
        print(f"  thermo-coded    : {summary['thermometer_hits']}/"
              f"{summary['converged']}  (of converged)")
        print(f"  failed          : {summary['failed']}/{summary['n_seeds']}  "
              f"(seeds: {summary['failed_seeds']})")
        if summary["converged"] > 0:
            print(f"  epochs          : "
                  f"mean={summary['mean_epochs']:.0f}  "
                  f"median={summary['median_epochs']:.0f}  "
                  f"min={summary['min_epochs']}  max={summary['max_epochs']}")
        print(f"  total time      : {dt:.1f}s")
        return

    t0 = time.time()
    model, history = train(n_bits=args.n_bits, n_hidden=args.n_hidden,
                           lr=args.lr, momentum=args.momentum,
                           init_scale=args.init_scale,
                           max_epochs=args.max_epochs, seed=args.seed,
                           bipolar=bipolar)
    dt = time.time() - t0

    X, y = make_parity_data(args.n_bits, bipolar=bipolar)
    print(f"\nFinal accuracy : {accuracy(model, X, y) * 100:.0f}% "
          f"({int(accuracy(model,X,y) * len(y))}/{len(y)})")
    print(f"Final loss     : {loss_mse(model, X, y):.4f}")
    print(f"Converged epoch: {history['converged_epoch']}")
    print(f"Wallclock      : {dt:.3f}s")

    score = thermometer_score(model)
    print(f"\nThermometer-code analysis:")
    print(f"  per-hidden thresholds (sorted): {sorted(score['thresholds'])}")
    print(f"  per-hidden polarities (signed): {score['polarities']}")
    print(f"  per-hidden monotonicity score : "
          f"{[f'{m:.2f}' for m in score['monotonicity']]}")
    print(f"  mean monotonicity              : "
          f"{score['mean_monotonicity']:.2f}  (1.0 = perfectly monotonic)")
    print(f"  unique thresholds              : {score['unique_thresholds']}"
          f" / {model.n_hidden}")
    print(f"  perfect thermometer            : {score['is_thermometer']}")

    print(f"\nMean hidden activation by input bit-count:")
    print(f"  bit-count: {list(score['levels'])}")
    for hi in range(model.n_hidden):
        row = [f"{x:.2f}" for x in score["mean_by_level"][hi]]
        print(f"  h{hi+1:<2}      : {row}")


if __name__ == "__main__":
    main()
