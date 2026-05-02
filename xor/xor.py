"""
XOR backprop demo, faithful to Rumelhart, Hinton & Williams (1986),
"Learning representations by back-propagating errors", Nature 323, 533-536
(short version of PDP Vol. 1, Ch. 8).

Problem:
    4 input/output patterns:
        (0, 0) -> 0
        (0, 1) -> 1
        (1, 0) -> 1
        (1, 1) -> 0

    XOR is famously not linearly separable, so the perceptron (Minsky &
    Papert, 1969) cannot solve it with a single layer. RHW1986 demonstrate
    that a multi-layer perceptron trained by backpropagation learns the
    function from those four examples.

Architectures (1986 paper):
    - 2-2-1: 2 inputs -> 2 hidden sigmoids -> 1 output sigmoid (canonical).
    - 2-1-2-skip: 2 inputs -> 1 hidden sigmoid, with a direct skip from
      both inputs to the output. Same number of weights as 2-2-1 (six)
      but a different inductive shape; the paper uses it to show that
      bypass connections don't hurt.

Hyperparameters (1986 paper):
    learning rate eta = 0.5, momentum alpha = 0.9, full-batch updates over
    the 4 patterns, "convergence" defined as every output within 0.5 of the
    target (i.e. argmax matches). Reported: ~558 sweeps (epochs) to converge
    on average, with ~2 of hundreds of runs landing in a local minimum.

This file is a faithful numpy reproduction. Train an MLP with SGD +
momentum, evaluate on the same 4 patterns, report epochs-to-converge and
final accuracy.
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

def make_xor_data() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, y) for the 4 XOR patterns. X is (4, 2); y is (4, 1)."""
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=np.float64)
    y = np.array([[0.0],
                  [1.0],
                  [1.0],
                  [0.0]], dtype=np.float64)
    return X, y


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

class XorMLP:
    """
    Two architectures, both 6 weights + biases (matching the 1986 paper):

      arch="2-2-1":
          h = sigmoid(W1 @ x + b1)        # 2 hidden units
          o = sigmoid(W2 @ h + b2)        # 1 output

      arch="2-1-2-skip":
          h = sigmoid(W1 @ x + b1)        # 1 hidden unit
          o = sigmoid(W2 @ h + Wskip @ x + b2)   # skip from inputs to output
    """

    def __init__(self, arch: str = "2-2-1", init_scale: float = 1.0,
                 seed: int = 0):
        self.arch = arch
        self.rng = np.random.default_rng(seed)
        if arch == "2-2-1":
            self.W1 = init_scale * (self.rng.random((2, 2)) - 0.5)
            self.b1 = init_scale * (self.rng.random((2,)) - 0.5)
            self.W2 = init_scale * (self.rng.random((1, 2)) - 0.5)
            self.b2 = init_scale * (self.rng.random((1,)) - 0.5)
            self.Wskip = None
        elif arch == "2-1-2-skip":
            self.W1 = init_scale * (self.rng.random((1, 2)) - 0.5)
            self.b1 = init_scale * (self.rng.random((1,)) - 0.5)
            self.W2 = init_scale * (self.rng.random((1, 1)) - 0.5)
            self.b2 = init_scale * (self.rng.random((1,)) - 0.5)
            self.Wskip = init_scale * (self.rng.random((1, 2)) - 0.5)
        else:
            raise ValueError(f"unknown arch {arch!r}")

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (h, o). X is (n, 2)."""
        h = sigmoid(X @ self.W1.T + self.b1)
        if self.arch == "2-2-1":
            o = sigmoid(h @ self.W2.T + self.b2)
        else:  # 2-1-2-skip
            o = sigmoid(h @ self.W2.T + X @ self.Wskip.T + self.b2)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        n = self.W1.size + self.b1.size + self.W2.size + self.b2.size
        if self.Wskip is not None:
            n += self.Wskip.size
        return n


# ----------------------------------------------------------------------
# Backprop step
# ----------------------------------------------------------------------

def backprop_grads(model: XorMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Compute gradients of mean-squared error w.r.t. all parameters.

    Loss: 0.5 * mean over patterns of (o - y)^2  (RHW1986 §"a simple example")
    """
    n = X.shape[0]
    h, o = model.forward(X)

    # output layer
    delta_o = (o - y) * dsigmoid_from_y(o)        # (n, 1)
    dW2 = delta_o.T @ h / n                        # (1, hidden)
    db2 = delta_o.mean(axis=0)                     # (1,)
    grads = {"W2": dW2, "b2": db2}

    # skip connection (if any)
    if model.Wskip is not None:
        grads["Wskip"] = delta_o.T @ X / n         # (1, 2)

    # hidden layer
    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h)   # (n, hidden)
    grads["W1"] = delta_h.T @ X / n                # (hidden, 2)
    grads["b1"] = delta_h.mean(axis=0)             # (hidden,)
    return grads


def loss_mse(model: XorMLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean((o - y) ** 2))


def accuracy(model: XorMLP, X: np.ndarray, y: np.ndarray,
             threshold: float = 0.5) -> float:
    """Classification accuracy: prediction matches target after rounding at 0.5."""
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(pred == y))


def converged(model: XorMLP, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    """RHW1986's convergence criterion: every output within `tol` of its target."""
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(arch: str = "2-2-1",
          lr: float = 0.5,
          momentum: float = 0.9,
          init_scale: float = 1.0,
          max_epochs: int = 5000,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 10,
          verbose: bool = True) -> tuple[XorMLP, dict]:
    """Train a 2-2-1 (or 2-1-2-skip) MLP on XOR with backprop + momentum.

    Full-batch update (4 patterns); the 1986 paper's setup. Returns
    (trained_model, history). `history["converged_epoch"]` is the first
    epoch where every output is within 0.5 of its target, or None.
    """
    model = XorMLP(arch=arch, init_scale=init_scale, seed=seed)
    X, y = make_xor_data()

    # momentum buffers
    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                               ("W2", model.W2), ("b2", model.b2)]}
    if model.Wskip is not None:
        velocities["Wskip"] = np.zeros_like(model.Wskip)

    history = {"epoch": [], "loss": [], "accuracy": [],
                "weight_norm": [], "converged_epoch": None,
                "outputs": []}

    if verbose:
        print(f"# XOR backprop  arch={arch}  params={model.n_params()}  "
              f"lr={lr}  momentum={momentum}  seed={seed}")

    for epoch in range(max_epochs):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]
        if model.Wskip is not None:
            model.Wskip += velocities["Wskip"]

        loss = loss_mse(model, X, y)
        acc = accuracy(model, X, y)
        wn = float(np.linalg.norm(np.concatenate(
            [model.W1.ravel(), model.W2.ravel()] +
            ([model.Wskip.ravel()] if model.Wskip is not None else []))))

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["weight_norm"].append(wn)
        history["outputs"].append(model.predict(X).ravel().tolist())

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at epoch {epoch + 1}  "
                      f"loss={loss:.4f}  acc={acc*100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == max_epochs - 1):
            snapshot_callback(epoch, model, history)

        if verbose and (epoch % 200 == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch+1:5d}  loss={loss:.4f}  "
                  f"acc={acc*100:5.1f}%  |W|={wn:.3f}")

        # Early stop a bit past convergence so the training curves don't
        # flatline forever.
        if (history["converged_epoch"] is not None
                and epoch + 1 >= history["converged_epoch"] + 50):
            break

    return model, history


# ----------------------------------------------------------------------
# Multi-seed sweep (for the "2 of hundreds hit a local minimum" claim)
# ----------------------------------------------------------------------

def sweep(arch: str, n_seeds: int, lr: float, momentum: float,
           init_scale: float, max_epochs: int) -> dict:
    """Run `n_seeds` independent training runs; return summary stats."""
    epochs = []
    failures = []
    for s in range(n_seeds):
        _, hist = train(arch=arch, lr=lr, momentum=momentum,
                        init_scale=init_scale, max_epochs=max_epochs,
                        seed=s, verbose=False)
        if hist["converged_epoch"] is None:
            failures.append(s)
        else:
            epochs.append(hist["converged_epoch"])
    return {"arch": arch, "n_seeds": n_seeds,
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
    p.add_argument("--arch", choices=["2-2-1", "2-1-2-skip"], default="2-2-1")
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
        summary = sweep(args.arch, n_seeds=args.sweep, lr=args.lr,
                        momentum=args.momentum, init_scale=args.init_scale,
                        max_epochs=args.max_epochs)
        dt = time.time() - t0
        print(f"\nSweep results ({args.sweep} seeds, arch={args.arch}):")
        print(f"  converged : {summary['converged']}/{summary['n_seeds']}")
        print(f"  failed    : {summary['failed']}/{summary['n_seeds']}  "
              f"(seeds: {summary['failed_seeds']})")
        print(f"  epochs    : mean={summary['mean_epochs']:.0f}  "
              f"median={summary['median_epochs']:.0f}  "
              f"min={summary['min_epochs']}  max={summary['max_epochs']}")
        print(f"  total time: {dt:.1f}s")
        return

    t0 = time.time()
    model, history = train(arch=args.arch, lr=args.lr, momentum=args.momentum,
                           init_scale=args.init_scale,
                           max_epochs=args.max_epochs, seed=args.seed)
    dt = time.time() - t0

    X, y = make_xor_data()
    print(f"\nFinal accuracy : {accuracy(model, X, y) * 100:.0f}% (4/4 if 100)")
    print(f"Final loss     : {loss_mse(model, X, y):.4f}")
    print(f"Converged epoch: {history['converged_epoch']}")
    print(f"Wallclock      : {dt:.3f}s")
    print(f"\nPredictions on training set:")
    o = model.predict(X).ravel()
    for (xi, yi, oi) in zip(X, y.ravel(), o):
        print(f"  ({xi[0]:.0f}, {xi[1]:.0f}) -> target={yi:.0f}  "
              f"output={oi:.4f}  pred={'1' if oi >= 0.5 else '0'}")


if __name__ == "__main__":
    main()
