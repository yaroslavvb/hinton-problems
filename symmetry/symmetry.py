"""
6-bit symmetry / palindrome detection (Rumelhart, Hinton & Williams 1986).

A 6-2-1 sigmoid network learns to output 1 iff the 6-bit input is a
palindrome (symmetric about its midpoint). All 64 patterns are enumerated;
8 are palindromes.

Famous result (RHW1986, Fig. 2 of the Nature short version):
    After convergence, the input-to-hidden weights show a unique pattern --
    each hidden unit's six weights are *mirror-symmetric in magnitude* and
    *opposite in sign* about the midpoint, with magnitudes in the ratio
    1 : 2 : 4 across the three position pairs. The output unit then
    differentiates a "near-zero net input" (palindrome) from any non-zero
    net input (non-palindrome) by means of a strong negative bias on each
    hidden unit.

This file: numpy-only, full-batch backprop with momentum, MSE loss.
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

def make_symmetry_data(encoding: str = "pm1") -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for all 64 6-bit patterns.

    encoding="pm1" -> bits in {-1, +1} (Hinton's lectures convention; the
                       1:2:4 weight pattern emerges most cleanly here).
    encoding="01"  -> bits in {0, 1}.

    y[i] = 1 if pattern i is a palindrome (x_1=x_6, x_2=x_5, x_3=x_4),
    else 0. There are 2^3 = 8 palindromes among the 64 patterns.
    """
    rows = []
    labels = []
    for code in range(64):
        bits = np.array([(code >> j) & 1 for j in range(6)], dtype=np.float64)
        rows.append(bits)
        is_pal = bool(bits[0] == bits[5] and bits[1] == bits[4]
                      and bits[2] == bits[3])
        labels.append(1.0 if is_pal else 0.0)
    X01 = np.stack(rows, axis=0)              # (64, 6) in {0,1}
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)
    if encoding == "01":
        X = X01
    elif encoding == "pm1":
        X = 2.0 * X01 - 1.0                    # in {-1, +1}
    else:
        raise ValueError(f"unknown encoding {encoding!r}")
    return X, y


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def dsigmoid_from_y(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


# ----------------------------------------------------------------------
# Model: 6-2-1 MLP
# ----------------------------------------------------------------------

class SymmetryMLP:
    """Two-layer MLP, 6 inputs -> 2 hidden sigmoids -> 1 output sigmoid."""

    def __init__(self, n_in: int = 6, n_hidden: int = 2, n_out: int = 1,
                 init_scale: float = 0.6, seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.rng = np.random.default_rng(seed)
        # Uniform [-init_scale/2, +init_scale/2] (RHW1986-style small init)
        self.W1 = init_scale * (self.rng.random((n_hidden, n_in)) - 0.5)
        self.b1 = init_scale * (self.rng.random((n_hidden,)) - 0.5)
        self.W2 = init_scale * (self.rng.random((n_out, n_hidden)) - 0.5)
        self.b2 = init_scale * (self.rng.random((n_out,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(X @ self.W1.T + self.b1)              # (n, n_hidden)
        o = sigmoid(h @ self.W2.T + self.b2)              # (n, n_out)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        return (self.W1.size + self.b1.size
                + self.W2.size + self.b2.size)

    def snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}


# ----------------------------------------------------------------------
# Backprop
# ----------------------------------------------------------------------

def backprop_grads(model: SymmetryMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Gradients of 0.5 * mean (o - y)^2 w.r.t. all parameters."""
    n = X.shape[0]
    h, o = model.forward(X)
    delta_o = (o - y) * dsigmoid_from_y(o)             # (n, 1)
    grads = {
        "W2": delta_o.T @ h / n,                        # (1, n_hidden)
        "b2": delta_o.mean(axis=0),                     # (1,)
    }
    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h)   # (n, n_hidden)
    grads["W1"] = delta_h.T @ X / n                     # (n_hidden, 6)
    grads["b1"] = delta_h.mean(axis=0)                  # (n_hidden,)
    return grads


def loss_mse(model: SymmetryMLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean((o - y) ** 2))


def accuracy(model: SymmetryMLP, X: np.ndarray, y: np.ndarray) -> float:
    o = model.predict(X)
    pred = (o >= 0.5).astype(np.float64)
    return float(np.mean(pred == y))


def converged(model: SymmetryMLP, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    """Every output within `tol` of its target (RHW1986 criterion)."""
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_sweeps: int = 5000,
          lr: float = 0.3,
          momentum: float = 0.95,
          init_scale: float = 1.0,
          encoding: str = "pm1",
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True) -> tuple[SymmetryMLP, dict]:
    """Full-batch backprop with momentum on all 64 patterns.

    `snapshot_callback(epoch, model, history)` is invoked every
    `snapshot_every` epochs (and on the final epoch). The 1986 paper
    reports ~1425 sweeps to converge for the 6-2-1 net; we default to 5000
    which is enough for the seeds that *do* converge.

    Note on per-seed success rate: with full-batch backprop on the 8/56
    palindrome class imbalance, ~80% of random inits escape the trivial
    "always non-palindrome" plateau and converge. The other ~20% stall
    around 87.5%-93.8% accuracy and never recover. RHW1986 likely used a
    perturbation-on-plateau wrapper (mentioned in the XOR section of the
    same paper) that we have NOT implemented here -- see Deviations in
    the README. To reproduce the paper-style result reliably, use one of
    the converging seeds (the multi-seed sweep prints them).
    """
    model = SymmetryMLP(init_scale=init_scale, seed=seed)
    X, y = make_symmetry_data(encoding=encoding)

    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                               ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "loss": [], "accuracy": [],
               "weight_norm": [], "converged_epoch": None,
               "ratio_2_to_1": [], "ratio_4_to_2": [],
               "snapshots": []}

    if verbose:
        print(f"# 6-2-1 symmetry  encoding={encoding}  params={model.n_params()}  "
              f"lr={lr}  momentum={momentum}  init_scale={init_scale}  seed={seed}")

    for epoch in range(n_sweeps):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        loss = loss_mse(model, X, y)
        acc = accuracy(model, X, y)
        wn = float(np.linalg.norm(model.W1))

        # Track magnitude ratios across the three position-pairs of W1.
        sym = inspect_weight_symmetry(model)
        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["weight_norm"].append(wn)
        history["ratio_2_to_1"].append(sym["mean_pair_magnitudes"][1]
                                       / max(sym["mean_pair_magnitudes"][0], 1e-9))
        history["ratio_4_to_2"].append(sym["mean_pair_magnitudes"][2]
                                       / max(sym["mean_pair_magnitudes"][1], 1e-9))

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at sweep {epoch + 1}  "
                      f"loss={loss:.4f}  acc={acc*100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)
            history["snapshots"].append((epoch + 1, model.snapshot()))

        if verbose and (epoch % 500 == 0 or epoch == n_sweeps - 1):
            r1 = history["ratio_2_to_1"][-1]
            r2 = history["ratio_4_to_2"][-1]
            print(f"  sweep {epoch+1:5d}  loss={loss:.4f}  acc={acc*100:5.1f}%  "
                  f"|W1|={wn:.2f}  pair2/pair1={r1:.2f}  pair3/pair2={r2:.2f}")

    return model, history


# ----------------------------------------------------------------------
# The 1:2:4 / opposite-sign symmetry check
# ----------------------------------------------------------------------

def inspect_weight_symmetry(model: SymmetryMLP) -> dict:
    """Quantify the famous RHW1986 result on the 6-2-1 hidden weights.

    Per hidden unit h, W1[h] has 6 entries: w_1, w_2, w_3, w_4, w_5, w_6.
    The paper's "more elegant than the human designers anticipated" result:

        |w_1| approx |w_6|   and   sign(w_1) = -sign(w_6)
        |w_2| approx |w_5|   and   sign(w_2) = -sign(w_5)
        |w_3| approx |w_4|   and   sign(w_3) = -sign(w_4)
        and the three pair-magnitudes are in 1 : 2 : 4 ratio.

    The figure in RHW1986 happens to show |w_1| < |w_2| < |w_3| (inner
    pair largest), but the symmetry of the problem allows any of the 3!
    permutations of {1, 2, 4} across the three position pairs -- the
    network picks one based on init.

    Returns a dict with per-hidden-unit measurements, the *ordering-aware*
    ratios (middle:outer and inner:middle as drawn) AND the *sorted* ratios
    (smallest:medium:largest) which reveal the 1:2:4 invariant regardless
    of which position the largest pair landed at.
    """
    W1 = model.W1                                     # (n_hidden, 6)
    n_hidden = W1.shape[0]

    per_hidden = []
    pair_mags = np.zeros(3, dtype=np.float64)         # outer -> inner
    pair_anti = []
    for h in range(n_hidden):
        w = W1[h]
        # pair 0 = (w_1, w_6) outermost; pair 1 = (w_2, w_5); pair 2 = (w_3, w_4)
        pairs = [(w[0], w[5]), (w[1], w[4]), (w[2], w[3])]
        mags = np.array([0.5 * (abs(a) + abs(b)) for a, b in pairs])
        # anti-symmetry score in [-1, 1]: 0 = opposite signs (perfect anti-sym),
        # +1 or -1 = same signs.
        anti = []
        for a, b in pairs:
            denom = abs(a) + abs(b) + 1e-12
            anti.append((a + b) / denom)
        per_hidden.append({"weights": w.tolist(),
                            "pair_magnitudes": mags.tolist(),
                            "antisymmetry_residuals": anti})
        pair_mags += mags
        pair_anti.extend(anti)

    pair_mags /= max(n_hidden, 1)
    ratio_middle_to_outer = pair_mags[1] / max(pair_mags[0], 1e-9)
    ratio_inner_to_middle = pair_mags[2] / max(pair_mags[1], 1e-9)
    ratio_inner_to_outer = pair_mags[2] / max(pair_mags[0], 1e-9)

    # Sorted version: re-rank the three pairs smallest -> largest, then
    # check 1:2:4. This is the invariant the network actually learns.
    sorted_mags = np.sort(pair_mags)
    sorted_ratio_med = sorted_mags[1] / max(sorted_mags[0], 1e-9)
    sorted_ratio_lrg = sorted_mags[2] / max(sorted_mags[1], 1e-9)
    sorted_ratio_lrg_to_sm = sorted_mags[2] / max(sorted_mags[0], 1e-9)

    anti_residuals = np.array(pair_anti)
    anti_residual_max = float(np.max(np.abs(anti_residuals)))

    # "Matches the paper" if:
    #   - each pair has opposite signs (residual < 0.30 i.e. same-sign mass < 30%)
    #   - sorted magnitudes are in 1:2:4 ratio (each ratio in [1.5, 2.5])
    matches = (anti_residual_max < 0.30
               and 1.5 <= sorted_ratio_med <= 2.5
               and 1.5 <= sorted_ratio_lrg <= 2.5)

    return {
        "per_hidden": per_hidden,
        "mean_pair_magnitudes": pair_mags.tolist(),     # [outer, middle, inner]
        "sorted_pair_magnitudes": sorted_mags.tolist(),  # [smallest, medium, largest]
        "ratio_middle_to_outer": float(ratio_middle_to_outer),
        "ratio_inner_to_middle": float(ratio_inner_to_middle),
        "ratio_inner_to_outer": float(ratio_inner_to_outer),
        "sorted_ratio_medium_to_smallest": float(sorted_ratio_med),
        "sorted_ratio_largest_to_medium": float(sorted_ratio_lrg),
        "sorted_ratio_largest_to_smallest": float(sorted_ratio_lrg_to_sm),
        "anti_residual_max": anti_residual_max,
        "matches_paper": bool(matches),
    }


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep_seeds(n_seeds: int, n_sweeps: int = 5000, **kw) -> dict:
    out = {"seeds": [], "converged_epoch": [], "final_acc": [],
           "ratio_middle_to_outer": [], "ratio_inner_to_middle": [],
           "ratio_inner_to_outer": [], "anti_residual_max": [],
           "matches_paper": []}
    for s in range(n_seeds):
        model, hist = train(n_sweeps=n_sweeps, seed=s, verbose=False, **kw)
        sym = inspect_weight_symmetry(model)
        out["seeds"].append(s)
        out["converged_epoch"].append(hist["converged_epoch"])
        out["final_acc"].append(hist["accuracy"][-1])
        out["ratio_middle_to_outer"].append(sym["ratio_middle_to_outer"])
        out["ratio_inner_to_middle"].append(sym["ratio_inner_to_middle"])
        out["ratio_inner_to_outer"].append(sym["ratio_inner_to_outer"])
        out["anti_residual_max"].append(sym["anti_residual_max"])
        out["matches_paper"].append(sym["matches_paper"])
        print(f"  seed {s:2d}  conv@{str(hist['converged_epoch']):>5}  "
              f"acc={hist['accuracy'][-1]*100:5.1f}%  "
              f"ratios=({sym['ratio_middle_to_outer']:.2f},"
              f"{sym['ratio_inner_to_middle']:.2f})  "
              f"anti_max={sym['anti_residual_max']:.2f}  "
              f"matches={sym['matches_paper']}")
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", type=int, default=5000,
                   help="number of full-batch updates")
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--encoding", choices=["pm1", "01"], default="pm1")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--multi-seed", type=int, default=0,
                   help="if > 0, run a sweep over this many seeds and exit")
    args = p.parse_args()

    _print_environment()

    if args.multi_seed > 0:
        out = sweep_seeds(n_seeds=args.multi_seed, n_sweeps=args.sweeps,
                          lr=args.lr, momentum=args.momentum,
                          init_scale=args.init_scale,
                          encoding=args.encoding)
        n = len(out["seeds"])
        n_match = sum(out["matches_paper"])
        n_conv = sum(1 for c in out["converged_epoch"] if c is not None)
        print(f"\n{n_conv}/{n} converged, {n_match}/{n} match the 1:2:4/opposite-sign pattern")
        return

    t0 = time.time()
    model, hist = train(n_sweeps=args.sweeps, lr=args.lr,
                        momentum=args.momentum,
                        init_scale=args.init_scale,
                        encoding=args.encoding, seed=args.seed)
    wallclock = time.time() - t0

    X, y = make_symmetry_data(encoding=args.encoding)
    sym = inspect_weight_symmetry(model)

    print("\n=== final ===")
    print(f"final accuracy : {hist['accuracy'][-1]*100:.1f}% "
          f"({int(hist['accuracy'][-1] * 64)}/64)")
    print(f"final loss     : {hist['loss'][-1]:.4f}")
    print(f"converged sweep: {hist['converged_epoch']}")
    print(f"wallclock      : {wallclock:.3f}s")

    print("\n=== weight-symmetry check (W1 = hidden weights, 6 inputs) ===")
    for h, info in enumerate(sym["per_hidden"]):
        w = info["weights"]
        mags = info["pair_magnitudes"]
        print(f"  hidden {h}:")
        print(f"    w           = {[f'{v:+.2f}' for v in w]}")
        print(f"    pair mags   = outer {mags[0]:.2f}, "
              f"middle {mags[1]:.2f}, inner {mags[2]:.2f}")
        if min(mags) > 1e-3:
            sorted_m = sorted(mags)
            print(f"    sorted-mag ratio = 1 : "
                  f"{sorted_m[1]/sorted_m[0]:.2f} : "
                  f"{sorted_m[2]/sorted_m[0]:.2f}")
    sm = sym["sorted_pair_magnitudes"]
    print(f"  sorted pair mags (mean over hidden units) = "
          f"{sm[0]:.3f}, {sm[1]:.3f}, {sm[2]:.3f}")
    if sm[0] > 1e-3:
        print(f"  sorted-mag ratio          = 1 : "
              f"{sym['sorted_ratio_medium_to_smallest']:.3f} : "
              f"{sym['sorted_ratio_largest_to_smallest']:.3f}  "
              f"(paper says 1 : 2 : 4)")
    print(f"  anti-symmetry residual (max) = {sym['anti_residual_max']:.3f}  "
          f"(0 = perfectly opposite signs)")
    print(f"  matches the famous 1:2:4 / opposite-sign result? "
          f"{sym['matches_paper']}")


if __name__ == "__main__":
    main()
