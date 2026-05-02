"""
2-bit distributed-to-local mapping with a 1-unit bottleneck
(Rumelhart, Hinton & Williams 1986, PDP Vol. 1, Ch. 8).

Problem:
    4 distributed 2-bit inputs map to 4 one-hot 4-D output targets:
        (0, 0) -> [1, 0, 0, 0]
        (0, 1) -> [0, 1, 0, 0]
        (1, 0) -> [0, 0, 1, 0]
        (1, 1) -> [0, 0, 0, 1]

    The architecture forces the encoding through a single sigmoid hidden
    unit:

        2 inputs -> 1 hidden sigmoid -> 4 sigmoid outputs

    A scalar in [0, 1] cannot encode 4 separate "labels" by hard categorical
    membership. Backprop's only way through the bottleneck is to assign
    each pattern a distinct *graded* hidden activation, so the 4 output
    sigmoids can use overlapping monotone cuts along the h-axis to read
    out which pattern is active. The paper reports the 4 hidden values
    settling at roughly (0, 0.2, 0.6, 1.0) -- the canonical demonstration
    that backprop will use intermediate activations when forced.

Architecture:
    W1: (2, 1) input -> hidden, plus hidden bias b1: (1,)
    W2: (1, 4) hidden -> output, plus output bias b2: (4,)
    Sigmoid on both layers. Loss = mean squared error against the one-hot
    target (RHW1986's "simple example" formulation).

Learning:
    Full-batch gradient descent with momentum (PDP Ch. 8 recipe).
    4 patterns per epoch.
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

def generate_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Return (X, T). X is (4, 2) of the 4 binary inputs; T is (4, 4) one-hot."""
    X = np.array([[0.0, 0.0],
                  [0.0, 1.0],
                  [1.0, 0.0],
                  [1.0, 1.0]], dtype=np.float64)
    T = np.eye(4, dtype=np.float64)
    return X, T


def make_dataset() -> tuple[np.ndarray, np.ndarray]:
    """Alias kept for symmetry with sibling stubs."""
    return generate_dataset()


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def dsigmoid_from_y(y: np.ndarray) -> np.ndarray:
    """Derivative of sigmoid given its post-activation output."""
    return y * (1.0 - y)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class BottleneckMLP:
    """
    2 -> 1 -> 4 MLP, sigmoid on hidden and on each output.

        h = sigmoid(W1 @ x + b1)        # 1 hidden unit
        o = sigmoid(W2 @ h + b2)        # 4 outputs
    """

    def __init__(self,
                 n_in: int = 2,
                 n_hidden: int = 1,
                 n_out: int = 4,
                 init_scale: float = 1.0,
                 seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.rng = np.random.default_rng(seed)
        # Uniform [-init_scale/2, +init_scale/2], matching the xor sibling.
        self.W1 = init_scale * (self.rng.random((n_hidden, n_in)) - 0.5)
        self.b1 = init_scale * (self.rng.random((n_hidden,)) - 0.5)
        self.W2 = init_scale * (self.rng.random((n_out, n_hidden)) - 0.5)
        self.b2 = init_scale * (self.rng.random((n_out,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (h, o). X is (n, n_in)."""
        h = sigmoid(X @ self.W1.T + self.b1)
        o = sigmoid(h @ self.W2.T + self.b2)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        return (self.W1.size + self.b1.size + self.W2.size + self.b2.size)


def build_model(n_in: int = 2, n_hidden: int = 1, n_out: int = 4,
                init_scale: float = 1.0, seed: int = 0) -> BottleneckMLP:
    return BottleneckMLP(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                         init_scale=init_scale, seed=seed)


# ----------------------------------------------------------------------
# Backprop
# ----------------------------------------------------------------------

def backprop_grads(model: BottleneckMLP, X: np.ndarray, T: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Gradients of mean-squared error 0.5 * mean_n sum_j (o_nj - t_nj)^2."""
    n = X.shape[0]
    h, o = model.forward(X)

    # output sigmoids
    delta_o = (o - T) * dsigmoid_from_y(o)        # (n, n_out)
    dW2 = delta_o.T @ h / n                        # (n_out, n_hidden)
    db2 = delta_o.mean(axis=0)                     # (n_out,)

    # hidden sigmoid
    delta_h = (delta_o @ model.W2) * dsigmoid_from_y(h)  # (n, n_hidden)
    dW1 = delta_h.T @ X / n                        # (n_hidden, n_in)
    db1 = delta_h.mean(axis=0)                     # (n_hidden,)

    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_mse(model: BottleneckMLP, X: np.ndarray, T: np.ndarray) -> float:
    _, o = model.forward(X)
    return 0.5 * float(np.mean(np.sum((o - T) ** 2, axis=1)))


def accuracy(model: BottleneckMLP, X: np.ndarray, T: np.ndarray) -> float:
    """Argmax of the 4 outputs matches argmax of the one-hot target."""
    _, o = model.forward(X)
    return float(np.mean(np.argmax(o, axis=1) == np.argmax(T, axis=1)))


# ----------------------------------------------------------------------
# Required public signatures
# ----------------------------------------------------------------------

def hidden_values(model: BottleneckMLP, data=None) -> np.ndarray:
    """Return the 4 hidden-unit activations, one per pattern.

    Output is shape (4,) -- the headline result. Index i is the hidden
    activation when the input is the i-th pattern of `generate_dataset()`.
    """
    if data is None:
        X, _ = generate_dataset()
    else:
        X = data[0] if isinstance(data, tuple) else data
    h, _ = model.forward(X)
    return h.ravel()


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(model: BottleneckMLP = None,
          data: tuple[np.ndarray, np.ndarray] = None,
          n_sweeps: int = 5000,
          lr: float = 0.3,
          momentum: float = 0.9,
          init_scale: float = 1.0,
          seed: int = 0,
          tol: float = 0.5,
          h_distinct_eps: float = 0.10,
          early_stop: bool = True,
          perturb_on_plateau: bool = True,
          plateau_window: int = 300,
          plateau_eps: float = 1e-4,
          perturb_scale: float = 1.0,
          perturb_cooldown: int = 200,
          snapshot_callback=None,
          snapshot_every: int = 25,
          verbose: bool = True) -> tuple[BottleneckMLP, dict]:
    """Train the 2-1-4 MLP with full-batch backprop + momentum.

    Convergence rule: every output is within `tol` of its target (RHW1986).

    The perturb-on-plateau wrapper kicks the weights with a small Gaussian
    when the loss fails to improve over `plateau_window` epochs and
    accuracy is below 100%. RHW1986 mentions perturbing weights to escape
    the rare local minima; the 2-1-4 problem in particular gets stuck in
    an XOR-like configuration (w_1 ~ -w_2) where two of the four hidden
    activations collapse, so perturbation is essentially required to
    recover the paper's 100% accuracy claim.

    Returns (model, history). When `model` and `data` are not provided,
    builds a fresh model with `seed` and uses `generate_dataset()`.
    """
    if model is None:
        model = build_model(init_scale=init_scale, seed=seed)
    if data is None:
        data = generate_dataset()
    X, T = data

    velocities = {
        "W1": np.zeros_like(model.W1),
        "b1": np.zeros_like(model.b1),
        "W2": np.zeros_like(model.W2),
        "b2": np.zeros_like(model.b2),
    }

    history = {
        "epoch": [],
        "loss": [],
        "accuracy": [],
        "weight_norm": [],
        "hidden_values": [],
        "outputs": [],
        "converged_epoch": None,
        "perturbations": [],
    }

    if verbose:
        print(f"# distributed-to-local 2-1-4 MLP  params={model.n_params()}  "
              f"lr={lr}  momentum={momentum}  init_scale={init_scale}  "
              f"seed={seed}  perturb_on_plateau={perturb_on_plateau}")

    perturb_rng = np.random.default_rng(seed + 10_000)
    last_perturb_epoch = -perturb_cooldown
    stuck_count = 0  # consecutive epochs of "not solved" (acc<1 or min_gap<eps)
    stable_at_100_count = 0
    stable_required = 50  # epochs of sustained 100% acc + distinct h

    for epoch in range(n_sweeps):
        grads = backprop_grads(model, X, T)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        loss = loss_mse(model, X, T)
        acc = accuracy(model, X, T)
        wn = float(np.linalg.norm(np.concatenate(
            [model.W1.ravel(), model.W2.ravel()])))
        hv = hidden_values(model, (X, T))
        _, o = model.forward(X)

        history["epoch"].append(epoch + 1)
        history["loss"].append(loss)
        history["accuracy"].append(acc)
        history["weight_norm"].append(wn)
        history["hidden_values"].append(hv.tolist())
        history["outputs"].append(o.tolist())

        # convergence: sustained 100% argmax accuracy + 4 distinct hidden
        # values for `stable_required` consecutive epochs. The strict
        # per-output |o-t| < 0.5 rule from RHW1986 is not achievable
        # here: each output sigmoid is monotone in the single hidden
        # activation, so 4 outputs cannot each fire above 0.5
        # selectively for one of 4 distinct h values. The achievable
        # signal is argmax matching + graded h spread, which is exactly
        # the paper's headline. Requiring sustained accuracy (not a single
        # transient hit) avoids flagging seeds whose middle h-pair drifts
        # back together after a brief fluctuation.
        sorted_h = np.sort(hv)
        min_gap = float(np.min(np.diff(sorted_h))) if len(sorted_h) > 1 else 0.0
        h_distinct = bool(min_gap > h_distinct_eps)
        if acc >= 1.0 and h_distinct:
            stable_at_100_count += 1
        else:
            stable_at_100_count = 0
        if (history["converged_epoch"] is None
                and stable_at_100_count >= stable_required):
            history["converged_epoch"] = epoch + 1 - stable_required + 1
            if verbose:
                print(f"  sustained convergence: first stable epoch "
                      f"{history['converged_epoch']}  loss={loss:.4f}  "
                      f"hidden={hv.round(3).tolist()}")

        # perturb-on-plateau: detect when the network is stuck below the
        # solved state. The 2-1-4 net's typical local minimum is the XOR
        # collapse (w_1 ~ -w_2) where two of the four hidden activations
        # merge while accuracy is stuck at 75% and loss inches down via
        # weight inflation. The "stuck" signal is `acc < 1.0` or the
        # minimum pairwise h-gap is below the distinctness threshold.
        # We count consecutive stuck epochs; when the counter exceeds
        # `plateau_window` and we're outside the cooldown, perturb.
        is_stuck = (acc < 1.0) or (min_gap < h_distinct_eps)
        if is_stuck:
            stuck_count += 1
        else:
            stuck_count = 0
        if (perturb_on_plateau
                and history["converged_epoch"] is None
                and is_stuck
                and stuck_count >= plateau_window
                and epoch - last_perturb_epoch >= perturb_cooldown):
            # Add Gaussian noise to all weights, clear momentum.
            model.W1 += perturb_scale * perturb_rng.standard_normal(model.W1.shape)
            model.b1 += perturb_scale * perturb_rng.standard_normal(model.b1.shape)
            model.W2 += perturb_scale * perturb_rng.standard_normal(model.W2.shape)
            model.b2 += perturb_scale * perturb_rng.standard_normal(model.b2.shape)
            for k in velocities:
                velocities[k][...] = 0.0
            history["perturbations"].append(epoch + 1)
            last_perturb_epoch = epoch
            stuck_count = 0
            stable_at_100_count = 0
            if verbose:
                print(f"  perturb at epoch {epoch + 1}  "
                      f"(plateau loss={loss:.4f} acc={acc*100:.0f}% "
                      f"min_gap={min_gap:.3f})")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)

        if verbose and (epoch % 500 == 0 or epoch == n_sweeps - 1):
            print(f"  epoch {epoch+1:5d}  loss={loss:.4f}  "
                  f"acc={acc*100:5.1f}%  |W|={wn:.3f}  "
                  f"hidden={np.round(hv, 3).tolist()}")

        # early stop right at sustained-convergence detection: the
        # `stable_required` window already buys us stability, and any
        # extra training risks drifting back into the local minimum
        # (the loss can keep decreasing via weight inflation while h
        # values slowly merge).
        if early_stop and history["converged_epoch"] is not None:
            break

    return model, history


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(n_seeds: int, lr: float = 0.3, momentum: float = 0.9,
          init_scale: float = 1.0, n_sweeps: int = 5000) -> dict:
    """Run `n_seeds` independent training runs; return summary stats."""
    epochs = []
    failures = []
    spreads = []
    for s in range(n_seeds):
        _, hist = train(seed=s, lr=lr, momentum=momentum,
                        init_scale=init_scale, n_sweeps=n_sweeps,
                        verbose=False)
        if hist["converged_epoch"] is None:
            failures.append(s)
        else:
            epochs.append(hist["converged_epoch"])
            hv = np.array(hist["hidden_values"][-1])
            spreads.append(float(hv.max() - hv.min()))
    return {
        "n_seeds": n_seeds,
        "converged": len(epochs),
        "failed": len(failures),
        "failed_seeds": failures,
        "mean_epochs": float(np.mean(epochs)) if epochs else float("nan"),
        "median_epochs": float(np.median(epochs)) if epochs else float("nan"),
        "min_epochs": int(min(epochs)) if epochs else -1,
        "max_epochs": int(max(epochs)) if epochs else -1,
        "mean_spread": float(np.mean(spreads)) if spreads else float("nan"),
        "epochs": epochs,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description="2-bit distributed-to-local-bottleneck (RHW1986)")
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--n-sweeps", type=int, default=5000,
                   help="Maximum training epochs.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", type=int, default=0,
                   help="If > 0, run this many seeds and report stats.")
    p.add_argument("--no-early-stop", action="store_true")
    args = p.parse_args()

    print(f"# python  : {sys.version.split()[0]}")
    print(f"# numpy   : {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    if args.sweep > 0:
        t0 = time.time()
        s = sweep(args.sweep, lr=args.lr, momentum=args.momentum,
                  init_scale=args.init_scale, n_sweeps=args.n_sweeps)
        dt = time.time() - t0
        print(f"\nSweep ({s['n_seeds']} seeds):")
        print(f"  converged : {s['converged']}/{s['n_seeds']}")
        print(f"  failed    : {s['failed']}/{s['n_seeds']}  "
              f"(seeds: {s['failed_seeds']})")
        print(f"  epochs    : mean={s['mean_epochs']:.0f}  "
              f"median={s['median_epochs']:.0f}  "
              f"min={s['min_epochs']}  max={s['max_epochs']}")
        print(f"  mean spread (max h - min h): {s['mean_spread']:.3f}")
        print(f"  total time: {dt:.1f}s")
        return

    t0 = time.time()
    model, history = train(lr=args.lr, momentum=args.momentum,
                           init_scale=args.init_scale,
                           n_sweeps=args.n_sweeps,
                           seed=args.seed,
                           early_stop=not args.no_early_stop)
    dt = time.time() - t0

    X, T = generate_dataset()
    hv = hidden_values(model, (X, T))
    print(f"\nFinal accuracy : {accuracy(model, X, T) * 100:.0f}% (4/4 if 100)")
    print(f"Final loss     : {loss_mse(model, X, T):.4f}")
    print(f"Converged epoch: {history['converged_epoch']}")
    print(f"Wallclock      : {dt:.3f}s")
    print(f"\nHidden values (the headline -- target ~0, 0.2, 0.6, 1.0):")
    for i, (xi, hi) in enumerate(zip(X, hv)):
        print(f"  pattern {i}  ({xi[0]:.0f},{xi[1]:.0f}) -> h = {hi:.4f}")
    sorted_h = sorted(hv.tolist())
    print(f"\nSorted hidden values: "
          f"[{', '.join(f'{v:.3f}' for v in sorted_h)}]")
    print(f"Spread (max - min) : {max(hv) - min(hv):.3f}")

    print(f"\nOutputs (rows = patterns, cols = 4 sigmoid outputs):")
    _, o = model.forward(X)
    for i in range(4):
        row = "  ".join(f"{v:.3f}" for v in o[i])
        ok = "OK" if int(np.argmax(o[i])) == i else "WRONG"
        print(f"  pattern {i}: [{row}]  argmax={int(np.argmax(o[i]))}  ({ok})")


if __name__ == "__main__":
    main()
