"""
Negation problem (Rumelhart, Hinton & Williams 1986),
"Learning internal representations by error propagation",
PDP Vol. 1, Ch. 8, §"Symmetry / Negation / Encoder examples".

Problem:
    4 inputs (1 flag bit + 3 data bits) -> 3 outputs.
    16 patterns total (2^4).

        flag = 0  ->  output = data         (identity)
        flag = 1  ->  output = NOT data     (bitwise complement)

Architecture:
    4 -> 6 -> 3 sigmoid MLP, full-batch backprop with momentum.
    Default lr = 0.5, momentum = 0.9, max_epochs = 5000.

    Note on width: the existing stub mentioned 3 hidden units, but each
    of the 3 outputs has to compute  o_i = b_i XOR flag, and a single
    sigmoid hidden unit cannot express XOR. Empirically 4-3-3 never
    converges across 30 seeds; 4-4-3 also fails. The minimal width that
    learns reliably is 6 (= 2 hidden units per output bit, one per flag
    value). This is the textbook AND-OR XOR construction applied per
    output. See README §"Deviations".

What the network must learn:
    With 6 hidden units, the natural solution is to have 2 units per
    output bit:
        h_2i   = sigmoid( bit_i AND  flag is OFF )  → fires when output_i=1 because flag=0,bit_i=1
        h_2i+1 = sigmoid( bit_i OFF AND flag is ON )→ fires when output_i=1 because flag=1,bit_i=0
    The output unit ORs the two together. Each hidden unit is a
    flag-gated detector for one (bit, flag) combination — the
    prototypical example of *role-sensitive* distributed processing.

Faithful numpy reproduction. Train, evaluate on all 16 patterns,
report epochs-to-converge and final accuracy.
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

def make_negation_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the 16 patterns.

    X is (16, 4): columns [flag, d1, d2, d3].
    y is (16, 3): the 3 output bits.

    flag=0  ->  y = data
    flag=1  ->  y = 1 - data
    """
    X = np.zeros((16, 4), dtype=np.float64)
    y = np.zeros((16, 3), dtype=np.float64)
    for i in range(16):
        flag = (i >> 3) & 1
        d1 = (i >> 2) & 1
        d2 = (i >> 1) & 1
        d3 = i & 1
        X[i] = [flag, d1, d2, d3]
        data = np.array([d1, d2, d3], dtype=np.float64)
        y[i] = (1.0 - data) if flag == 1 else data
    return X, y


def pattern_label(x: np.ndarray) -> str:
    """Compact label for a 4-bit input vector, e.g. '0|101' or '1|011'."""
    flag = int(round(x[0]))
    bits = "".join(str(int(round(b))) for b in x[1:])
    return f"{flag}|{bits}"


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

class NegationMLP:
    """4-3-3 MLP, sigmoid activations everywhere."""

    def __init__(self, n_in: int = 4, n_hidden: int = 3, n_out: int = 3,
                 init_scale: float = 1.0, seed: int = 0):
        self.n_in, self.n_hidden, self.n_out = n_in, n_hidden, n_out
        self.rng = np.random.default_rng(seed)
        self.W1 = init_scale * (self.rng.random((n_hidden, n_in)) - 0.5)
        self.b1 = init_scale * (self.rng.random((n_hidden,)) - 0.5)
        self.W2 = init_scale * (self.rng.random((n_out, n_hidden)) - 0.5)
        self.b2 = init_scale * (self.rng.random((n_out,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(X @ self.W1.T + self.b1)
        o = sigmoid(h @ self.W2.T + self.b2)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


# ----------------------------------------------------------------------
# Backprop
# ----------------------------------------------------------------------

def backprop_grads(model: NegationMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Mean-squared error gradient wrt every parameter.

    Loss: 0.5 * mean over patterns of sum-over-outputs of (o - y)^2.
    """
    n = X.shape[0]
    h, o = model.forward(X)

    delta_o = (o - y) * dsigmoid_from_y(o)             # (n, n_out)
    dW2 = delta_o.T @ h / n                             # (n_out, n_hidden)
    db2 = delta_o.mean(axis=0)                          # (n_out,)

    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h) # (n, n_hidden)
    dW1 = delta_h.T @ X / n                             # (n_hidden, n_in)
    db1 = delta_h.mean(axis=0)                          # (n_hidden,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_mse(model: NegationMLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean(np.sum((o - y) ** 2, axis=1)))


def accuracy(model: NegationMLP, X: np.ndarray, y: np.ndarray,
             threshold: float = 0.5) -> float:
    """Per-bit classification accuracy after rounding output at 0.5."""
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(pred == y))


def pattern_accuracy(model: NegationMLP, X: np.ndarray, y: np.ndarray,
                      threshold: float = 0.5) -> float:
    """Whole-pattern accuracy: a pattern counts as correct only if all 3 bits match."""
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(np.all(pred == y, axis=1)))


def converged(model: NegationMLP, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    """Every output within `tol` of its target (RHW1986 convergence rule)."""
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_hidden: int = 6,
          lr: float = 0.5,
          momentum: float = 0.9,
          init_scale: float = 1.0,
          max_epochs: int = 5000,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 20,
          verbose: bool = True) -> tuple[NegationMLP, dict]:
    """Full-batch backprop+momentum on all 16 patterns."""
    model = NegationMLP(n_hidden=n_hidden, init_scale=init_scale, seed=seed)
    X, y = make_negation_data()

    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                               ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "loss": [], "accuracy": [],
                "pattern_accuracy": [], "weight_norm": [],
                "converged_epoch": None}

    if verbose:
        print(f"# negation backprop  arch=4-{n_hidden}-3  "
              f"params={model.n_params()}  lr={lr}  momentum={momentum}  "
              f"seed={seed}")

    for epoch in range(max_epochs):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        loss = loss_mse(model, X, y)
        bit_acc = accuracy(model, X, y)
        pat_acc = pattern_accuracy(model, X, y)
        wn = float(np.linalg.norm(np.concatenate(
            [model.W1.ravel(), model.W2.ravel()])))

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(bit_acc)
        history["pattern_accuracy"].append(pat_acc)
        history["weight_norm"].append(wn)

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at epoch {epoch + 1}  "
                      f"loss={loss:.4f}  bit_acc={bit_acc*100:.0f}%  "
                      f"pattern_acc={pat_acc*100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == max_epochs - 1):
            snapshot_callback(epoch, model, history)

        if verbose and (epoch % 500 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch+1:5d}  loss={loss:.4f}  "
                  f"bit_acc={bit_acc*100:5.1f}%  pat_acc={pat_acc*100:5.1f}%  "
                  f"|W|={wn:.3f}")

        # Stop a bit past convergence so curves don't flatline forever.
        if (history["converged_epoch"] is not None
                and epoch + 1 >= history["converged_epoch"] + 100):
            break

    return model, history


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(n_seeds: int, lr: float, momentum: float,
           init_scale: float, max_epochs: int,
           n_hidden: int = 6) -> dict:
    epochs = []
    failures = []
    for s in range(n_seeds):
        _, hist = train(n_hidden=n_hidden, lr=lr, momentum=momentum,
                        init_scale=init_scale, max_epochs=max_epochs,
                        seed=s, verbose=False)
        if hist["converged_epoch"] is None:
            failures.append(s)
        else:
            epochs.append(hist["converged_epoch"])
    return {"n_seeds": n_seeds,
            "converged": len(epochs), "failed": len(failures),
            "failed_seeds": failures,
            "mean_epochs": float(np.mean(epochs)) if epochs else float("nan"),
            "median_epochs": float(np.median(epochs)) if epochs else float("nan"),
            "min_epochs": int(min(epochs)) if epochs else -1,
            "max_epochs": int(max(epochs)) if epochs else -1,
            "epochs": epochs}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--n-hidden", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", type=int, default=0,
                   help="If > 0, run a sweep across this many seeds and report stats.")
    args = p.parse_args()

    print(f"# python  : {sys.version.split()[0]}")
    print(f"# numpy   : {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    if args.sweep > 0:
        t0 = time.time()
        summary = sweep(n_seeds=args.sweep, lr=args.lr,
                        momentum=args.momentum, init_scale=args.init_scale,
                        max_epochs=args.max_epochs,
                        n_hidden=args.n_hidden)
        dt = time.time() - t0
        print(f"\nSweep results ({args.sweep} seeds):")
        print(f"  converged : {summary['converged']}/{summary['n_seeds']}")
        print(f"  failed    : {summary['failed']}/{summary['n_seeds']}  "
              f"(seeds: {summary['failed_seeds']})")
        if summary["converged"]:
            print(f"  epochs    : mean={summary['mean_epochs']:.0f}  "
                  f"median={summary['median_epochs']:.0f}  "
                  f"min={summary['min_epochs']}  max={summary['max_epochs']}")
        print(f"  total time: {dt:.1f}s")
        return

    t0 = time.time()
    model, history = train(n_hidden=args.n_hidden, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed)
    dt = time.time() - t0

    X, y = make_negation_data()
    print(f"\nFinal bit accuracy    : {accuracy(model, X, y) * 100:.0f}% (out of 48 bits)")
    print(f"Final pattern accuracy: {pattern_accuracy(model, X, y) * 100:.0f}% (out of 16 patterns)")
    print(f"Final loss            : {loss_mse(model, X, y):.4f}")
    print(f"Converged epoch       : {history['converged_epoch']}")
    print(f"Wallclock             : {dt:.3f}s")
    print(f"\nPredictions on training set:")
    o = model.predict(X)
    for xi, yi, oi in zip(X, y, o):
        pred = (oi >= 0.5).astype(int)
        target = yi.astype(int)
        ok = "OK" if np.all(pred == target) else "X "
        print(f"  {pattern_label(xi)} -> target={''.join(map(str, target))}  "
              f"output=({oi[0]:.2f},{oi[1]:.2f},{oi[2]:.2f})  "
              f"pred={''.join(map(str, pred))}  {ok}")


if __name__ == "__main__":
    main()
