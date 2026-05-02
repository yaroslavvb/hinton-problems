"""
Binary addition of two 2-bit numbers (Rumelhart, Hinton & Williams 1986).

Source:
    Rumelhart, Hinton & Williams (1986),
    "Learning internal representations by error propagation",
    PDP Vol. 1, Ch. 8.

Problem:
    Inputs: 4 bits = (a1, a0, b1, b0) representing two 2-bit numbers
            a = 2*a1 + a0   (in 0..3)
            b = 2*b1 + b0   (in 0..3)
    Output: 3 bits = (s2, s1, s0) = binary representation of (a + b) in 0..6

    All 16 input combinations enumerated.

Architectures:
    4-3-3: 4 inputs -> 3 hidden sigmoids -> 3 output sigmoids. Succeeds.
    4-2-3: 4 inputs -> 2 hidden sigmoids -> 3 output sigmoids. Often stuck
           in a local minimum. The canonical "hidden units are not
           equipotential" example: 2 hidden units are not enough to
           cleanly separate the carry signal from the bit-sum signals,
           so a sizable fraction of random inits stall short of 100%.

This file: numpy-only, full-batch backprop with momentum, MSE loss.
CLI: --arch {4-3-3, 4-2-3} --seed --n-trials
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

def generate_dataset(encoding: str = "01") -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for all 16 (a, b) pairs.

    Inputs (X): shape (16, 4) = (a1, a0, b1, b0).
    Targets (y): shape (16, 3) = (s2, s1, s0) = binary representation of a+b.

    encoding="01"  -> input bits in {0, 1}
    encoding="pm1" -> input bits in {-1, +1}  (targets stay in {0, 1})
    """
    rows_x = []
    rows_y = []
    for a in range(4):
        for b in range(4):
            a1, a0 = (a >> 1) & 1, a & 1
            b1, b0 = (b >> 1) & 1, b & 1
            rows_x.append([a1, a0, b1, b0])
            s = a + b  # in 0..6
            s2, s1, s0 = (s >> 2) & 1, (s >> 1) & 1, s & 1
            rows_y.append([s2, s1, s0])
    X01 = np.array(rows_x, dtype=np.float64)
    y = np.array(rows_y, dtype=np.float64)
    if encoding == "01":
        X = X01
    elif encoding == "pm1":
        X = 2.0 * X01 - 1.0
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
# Model: 4-H-3 MLP, two flavors of H
# ----------------------------------------------------------------------

class BinaryAdditionMLP:
    """Two-layer MLP, 4 inputs -> H hidden sigmoids -> 3 output sigmoids.

    arch="4-3-3": H = 3 (succeeds reliably)
    arch="4-2-3": H = 2 (often stuck in local minima)
    """

    def __init__(self, arch: str = "4-3-3", init_scale: float = 1.0,
                 seed: int = 0):
        self.arch = arch
        if arch == "4-3-3":
            self.n_hidden = 3
        elif arch == "4-2-3":
            self.n_hidden = 2
        else:
            raise ValueError(f"unknown arch {arch!r}")
        self.n_in = 4
        self.n_out = 3
        self.rng = np.random.default_rng(seed)
        # Uniform [-init_scale/2, +init_scale/2]
        self.W1 = init_scale * (self.rng.random((self.n_hidden, self.n_in)) - 0.5)
        self.b1 = init_scale * (self.rng.random((self.n_hidden,)) - 0.5)
        self.W2 = init_scale * (self.rng.random((self.n_out, self.n_hidden)) - 0.5)
        self.b2 = init_scale * (self.rng.random((self.n_out,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(X @ self.W1.T + self.b1)              # (n, n_hidden)
        o = sigmoid(h @ self.W2.T + self.b2)              # (n, 3)
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

def backprop_grads(model: BinaryAdditionMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Gradients of 0.5 * mean (o - y)^2 w.r.t. all parameters."""
    n = X.shape[0]
    h, o = model.forward(X)
    delta_o = (o - y) * dsigmoid_from_y(o)             # (n, 3)
    grads = {
        "W2": delta_o.T @ h / n,                        # (3, n_hidden)
        "b2": delta_o.mean(axis=0),                     # (3,)
    }
    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h)   # (n, n_hidden)
    grads["W1"] = delta_h.T @ X / n                     # (n_hidden, 4)
    grads["b1"] = delta_h.mean(axis=0)                  # (n_hidden,)
    return grads


def loss_mse(model: BinaryAdditionMLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean((o - y) ** 2))


def accuracy_per_bit(model: BinaryAdditionMLP, X: np.ndarray,
                      y: np.ndarray) -> float:
    """Fraction of output bits within 0.5 of their target."""
    o = model.predict(X)
    pred = (o >= 0.5).astype(np.float64)
    return float(np.mean(pred == y))


def accuracy_per_pattern(model: BinaryAdditionMLP, X: np.ndarray,
                          y: np.ndarray) -> float:
    """Fraction of patterns where ALL 3 output bits are correct."""
    o = model.predict(X)
    pred = (o >= 0.5).astype(np.float64)
    return float(np.mean(np.all(pred == y, axis=1)))


def converged(model: BinaryAdditionMLP, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    """Every output within `tol` of its target (RHW1986 criterion)."""
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(arch: str = "4-3-3",
          n_sweeps: int = 5000,
          lr: float = 2.0,
          momentum: float = 0.9,
          init_scale: float = 2.0,
          encoding: str = "01",
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True) -> tuple[BinaryAdditionMLP, dict]:
    """Full-batch backprop with momentum on all 16 (a, b) pairs.

    Returns (trained_model, history).
    history["converged_epoch"] is the first sweep where every output is
    within 0.5 of its target, or None if never converged within n_sweeps.
    """
    model = BinaryAdditionMLP(arch=arch, init_scale=init_scale, seed=seed)
    X, y = generate_dataset(encoding=encoding)

    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                               ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "loss": [], "accuracy_bit": [],
               "accuracy_pattern": [], "weight_norm": [],
               "converged_epoch": None, "snapshots": []}

    if verbose:
        print(f"# binary-addition arch={arch} (4-{model.n_hidden}-3) "
              f"params={model.n_params()}  "
              f"lr={lr}  momentum={momentum}  init_scale={init_scale}  "
              f"seed={seed}")

    for epoch in range(n_sweeps):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        loss = loss_mse(model, X, y)
        acc_bit = accuracy_per_bit(model, X, y)
        acc_pat = accuracy_per_pattern(model, X, y)
        wn = float(np.linalg.norm(model.W1))

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy_bit"].append(acc_bit)
        history["accuracy_pattern"].append(acc_pat)
        history["weight_norm"].append(wn)

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at sweep {epoch + 1}  "
                      f"loss={loss:.4f}  acc(pat)={acc_pat*100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)
            history["snapshots"].append((epoch + 1, model.snapshot()))

        if verbose and (epoch % 500 == 0 or epoch == n_sweeps - 1):
            print(f"  sweep {epoch+1:5d}  loss={loss:.4f}  "
                  f"acc(bit)={acc_bit*100:5.1f}%  "
                  f"acc(pat)={acc_pat*100:5.1f}%  |W1|={wn:.2f}")

    return model, history


# ----------------------------------------------------------------------
# Local-minimum rate (the headline finding)
# ----------------------------------------------------------------------

def local_minimum_rate(arch: str, n_trials: int = 50,
                        n_sweeps: int = 5000, lr: float = 2.0,
                        momentum: float = 0.9, init_scale: float = 2.0,
                        encoding: str = "01",
                        seed_base: int = 0,
                        verbose: bool = False) -> dict:
    """Run `n_trials` independent training runs; return summary stats.

    A trial is "stuck in a local minimum" if it does NOT converge to
    100% per-pattern accuracy within `n_sweeps`. Returns a dict with
    keys: arch, n_trials, n_converged, n_stuck, rate, mean_epochs,
    median_epochs, min_epochs, max_epochs, epochs (list per converged
    trial), stuck_seeds, final_pattern_accs.
    """
    epochs: list[int] = []
    stuck_seeds: list[int] = []
    final_pattern_accs: list[float] = []
    for t in range(n_trials):
        seed = seed_base + t
        _, hist = train(arch=arch, n_sweeps=n_sweeps, lr=lr,
                        momentum=momentum, init_scale=init_scale,
                        encoding=encoding, seed=seed, verbose=False)
        final_acc = hist["accuracy_pattern"][-1]
        final_pattern_accs.append(final_acc)
        if hist["converged_epoch"] is None:
            stuck_seeds.append(seed)
        else:
            epochs.append(hist["converged_epoch"])
        if verbose:
            tag = (f"converged@{hist['converged_epoch']}"
                   if hist["converged_epoch"] is not None
                   else f"STUCK acc(pat)={final_acc*100:.1f}%")
            print(f"  arch={arch} seed={seed:3d}  {tag}")
    n_stuck = len(stuck_seeds)
    n_conv = len(epochs)
    return {
        "arch": arch,
        "n_trials": n_trials,
        "n_converged": n_conv,
        "n_stuck": n_stuck,
        "rate": n_stuck / n_trials,
        "mean_epochs": float(np.mean(epochs)) if epochs else float("nan"),
        "median_epochs": float(np.median(epochs)) if epochs else float("nan"),
        "min_epochs": int(min(epochs)) if epochs else -1,
        "max_epochs": int(max(epochs)) if epochs else -1,
        "epochs": epochs,
        "stuck_seeds": stuck_seeds,
        "final_pattern_accs": final_pattern_accs,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--arch", choices=["4-3-3", "4-2-3"], default="4-3-3")
    p.add_argument("--sweeps", type=int, default=5000,
                   help="number of full-batch updates")
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--encoding", choices=["01", "pm1"], default="01")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-trials", type=int, default=0,
                   help="if > 0, run a sweep of this many seeds and report "
                        "local-minimum rate")
    p.add_argument("--both-archs", action="store_true",
                   help="when --n-trials > 0, run both 4-3-3 and 4-2-3 "
                        "and print a side-by-side comparison")
    args = p.parse_args()

    _print_environment()

    if args.n_trials > 0:
        archs = ["4-3-3", "4-2-3"] if args.both_archs else [args.arch]
        results = {}
        t0 = time.time()
        for arch in archs:
            print(f"\n=== local-minimum sweep, arch={arch}, "
                  f"n_trials={args.n_trials} ===")
            r = local_minimum_rate(
                arch=arch, n_trials=args.n_trials, n_sweeps=args.sweeps,
                lr=args.lr, momentum=args.momentum,
                init_scale=args.init_scale, encoding=args.encoding,
                seed_base=args.seed, verbose=True)
            results[arch] = r
        dt = time.time() - t0

        print(f"\n=== summary (n_trials={args.n_trials}) ===")
        print(f"{'arch':<8} {'converged':>11} {'stuck':>7} {'rate':>7} "
              f"{'med epoch':>10} {'mean epoch':>11} {'range':>13}")
        for arch, r in results.items():
            range_str = (f"{r['min_epochs']}-{r['max_epochs']}"
                         if r["epochs"] else "-")
            print(f"{arch:<8} "
                  f"{r['n_converged']:>5}/{r['n_trials']:<5} "
                  f"{r['n_stuck']:>7} "
                  f"{r['rate']*100:>6.1f}% "
                  f"{r['median_epochs']:>10.0f} "
                  f"{r['mean_epochs']:>11.0f} "
                  f"{range_str:>13}")
        if len(results) == 2:
            gap = results["4-2-3"]["rate"] - results["4-3-3"]["rate"]
            print(f"\nLocal-minimum rate gap (4-2-3 minus 4-3-3): "
                  f"{gap*100:+.1f} percentage points")
        print(f"\nTotal sweep time: {dt:.1f}s")
        return

    t0 = time.time()
    model, hist = train(arch=args.arch, n_sweeps=args.sweeps, lr=args.lr,
                        momentum=args.momentum,
                        init_scale=args.init_scale,
                        encoding=args.encoding, seed=args.seed)
    wallclock = time.time() - t0

    X, y = generate_dataset(encoding=args.encoding)

    print("\n=== final ===")
    print(f"final loss              : {hist['loss'][-1]:.4f}")
    print(f"final per-bit accuracy  : "
          f"{hist['accuracy_bit'][-1]*100:.1f}% "
          f"({int(hist['accuracy_bit'][-1] * 16 * 3)}/48 bits)")
    print(f"final per-pattern acc.  : "
          f"{hist['accuracy_pattern'][-1]*100:.1f}% "
          f"({int(hist['accuracy_pattern'][-1] * 16)}/16 patterns)")
    print(f"converged sweep         : {hist['converged_epoch']}")
    print(f"wallclock               : {wallclock:.3f}s")

    print("\n=== predictions on all 16 patterns ===")
    o = model.predict(X)
    for i in range(16):
        a1, a0, b1, b0 = X[i] if args.encoding == "01" else (X[i] + 1) / 2
        a = int(2 * a1 + a0)
        b = int(2 * b1 + b0)
        target = int(4 * y[i, 0] + 2 * y[i, 1] + y[i, 2])
        pred_bits = (o[i] >= 0.5).astype(int)
        pred_dec = int(4 * pred_bits[0] + 2 * pred_bits[1] + pred_bits[2])
        ok = "OK" if pred_dec == target else "MISS"
        print(f"  {a} + {b} = {target}  predicted {pred_dec:>1d}  "
              f"output=[{o[i,0]:.2f},{o[i,1]:.2f},{o[i,2]:.2f}]  {ok}")


if __name__ == "__main__":
    main()
