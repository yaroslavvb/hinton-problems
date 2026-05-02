"""
8-3-8 backprop autoencoder -- backprop mirror of Ackley/Hinton/Sejnowski (1985)
encoder, trained the way Rumelhart, Hinton & Williams (1986) trained their
MLPs in "Learning internal representations by error propagation" (PDP Vol. 1,
Ch. 8).

Problem:
  8 one-hot inputs -> 3 sigmoid hidden units -> 8 sigmoid outputs.
  Target = input. The 3 hidden units are forced to encode 8 distinct patterns
  through a 3-bit bottleneck, so they self-organize into a binary code that
  hits 8 of the 8 corners of the 3-cube.

Architecture:
  W1: (8, 3) input -> hidden, plus hidden bias b1: (3,)
  W2: (3, 8) hidden -> output, plus output bias b2: (8,)
  Sigmoid activation on both hidden and output. Loss = cross-entropy.

Learning:
  Full-batch gradient descent with momentum (the recipe from PDP Ch. 8).
  Eight training patterns, so a full-batch epoch is one weight update per
  call to ``train_step``.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_encoder_data() -> np.ndarray:
    """Return the 8x8 identity matrix -- the 8 one-hot training patterns."""
    return np.eye(8, dtype=np.float64)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class EncoderMLP:
    """8-3-8 MLP with sigmoid activations on both layers."""

    def __init__(self,
                 n_in: int = 8,
                 n_hidden: int = 3,
                 n_out: int = 8,
                 init_scale: float = 0.5,
                 seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.rng = np.random.default_rng(seed)
        # Symmetric uniform init in (-init_scale, +init_scale), matching the
        # 1986 paper's small-random-weights recipe.
        self.W1 = self.rng.uniform(-init_scale, init_scale,
                                   size=(n_in, n_hidden))
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = self.rng.uniform(-init_scale, init_scale,
                                   size=(n_hidden, n_out))
        self.b2 = np.zeros(n_out, dtype=np.float64)

    # ---- forward ---------------------------------------------------------

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (hidden activations h, output activations y)."""
        h = sigmoid(x @ self.W1 + self.b1)
        y = sigmoid(h @ self.W2 + self.b2)
        return h, y

    # ---- backward --------------------------------------------------------

    def grads(self, x: np.ndarray, target: np.ndarray
              ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Cross-entropy gradients w.r.t. W1, b1, W2, b2.

        Loss = -sum(t*log(y) + (1-t)*log(1-y)).
        With sigmoid output, dLoss/dz_out = (y - t).
        """
        h, y = self.forward(x)
        n = x.shape[0]
        delta_out = (y - target) / n                     # (n, n_out)
        dW2 = h.T @ delta_out                            # (n_hidden, n_out)
        db2 = delta_out.sum(axis=0)                      # (n_out,)

        delta_h = (delta_out @ self.W2.T) * h * (1 - h)  # (n, n_hidden)
        dW1 = x.T @ delta_h                              # (n_in, n_hidden)
        db1 = delta_h.sum(axis=0)                        # (n_hidden,)
        return dW1, db1, dW2, db2

    # ---- evaluation ------------------------------------------------------

    def loss(self, x: np.ndarray, target: np.ndarray) -> float:
        _, y = self.forward(x)
        eps = 1e-12
        return float(-(target * np.log(y + eps)
                       + (1 - target) * np.log(1 - y + eps)).sum() / x.shape[0])

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Return the argmax over the 8 outputs (predicted class index)."""
        _, y = self.forward(x)
        return np.argmax(y, axis=1)

    def hidden(self, x: np.ndarray) -> np.ndarray:
        """Return the 3-D hidden activations for each input."""
        h, _ = self.forward(x)
        return h


# ----------------------------------------------------------------------
# Required public signatures
# ----------------------------------------------------------------------

def hidden_code_table(model: EncoderMLP, data: np.ndarray = None
                      ) -> np.ndarray:
    """Return the (8, 3) matrix of hidden activations for each input pattern.

    Row i is the 3-D hidden activation vector when input = one-hot(i). To get
    the binary code, threshold at 0.5: ``(hidden_code_table(model) > 0.5).astype(int)``.
    """
    if data is None:
        data = make_encoder_data()
    return model.hidden(data)


def n_distinct_codes(model: EncoderMLP, data: np.ndarray = None,
                     threshold: float = 0.5) -> int:
    """Count distinct binarized hidden codes across the 8 patterns."""
    codes = hidden_code_table(model, data)
    binary = (codes > threshold).astype(int)
    rows = {tuple(r) for r in binary}
    return len(rows)


def accuracy(model: EncoderMLP, data: np.ndarray = None) -> float:
    """Reconstruction accuracy: argmax over the 8 outputs == input index."""
    if data is None:
        data = make_encoder_data()
    preds = model.predict(data)
    return float((preds == np.arange(8)).mean())


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 5000,
          lr: float = 0.5,
          momentum: float = 0.9,
          init_scale: float = 0.1,
          seed: int = 0,
          target_acc: float = 1.0,
          early_stop: bool = True,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True) -> tuple[EncoderMLP, dict]:
    """Train the 8-3-8 MLP with full-batch gradient descent + momentum.

    Returns (model, history). Training stops early when accuracy hits
    ``target_acc`` AND all 8 hidden codes are distinct (the "solved" signal).
    """
    model = EncoderMLP(seed=seed, init_scale=init_scale)
    data = make_encoder_data()
    target = data.copy()

    vW1 = np.zeros_like(model.W1)
    vb1 = np.zeros_like(model.b1)
    vW2 = np.zeros_like(model.W2)
    vb2 = np.zeros_like(model.b2)

    history = {"epoch": [], "loss": [], "acc": [],
               "weight_norm": [], "n_distinct_codes": [],
               "code_separation": []}

    if verbose:
        print(f"# 8-3-8 backprop encoder: 8 patterns, "
              f"{model.n_in}-{model.n_hidden}-{model.n_out}, "
              f"lr={lr}, momentum={momentum}, init_scale={init_scale}, "
              f"seed={seed}")

    t0 = time.time()
    for epoch in range(n_epochs):
        dW1, db1, dW2, db2 = model.grads(data, target)
        vW1 = momentum * vW1 + lr * dW1
        vb1 = momentum * vb1 + lr * db1
        vW2 = momentum * vW2 + lr * dW2
        vb2 = momentum * vb2 + lr * db2
        model.W1 -= vW1
        model.b1 -= vb1
        model.W2 -= vW2
        model.b2 -= vb2

        loss = model.loss(data, target)
        acc = accuracy(model, data)
        n_codes = n_distinct_codes(model, data)
        codes = hidden_code_table(model, data)
        sep = mean_pairwise_distance(codes)

        history["epoch"].append(epoch + 1)
        history["loss"].append(float(loss))
        history["acc"].append(float(acc))
        history["weight_norm"].append(
            float(np.linalg.norm(model.W1) + np.linalg.norm(model.W2)))
        history["n_distinct_codes"].append(int(n_codes))
        history["code_separation"].append(float(sep))

        if verbose and (epoch % 500 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:5d}  loss={loss:.4f}  acc={acc*100:5.1f}%  "
                  f"distinct_codes={n_codes}/8  sep={sep:.3f}  "
                  f"({time.time()-t0:.2f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history)

        if (early_stop and acc >= target_acc and n_codes == 8
                and epoch >= 200):
            if verbose:
                print(f"epoch {epoch+1:5d}  *** solved (acc=100%, 8/8 distinct codes) ***")
            break

    return model, history


def mean_pairwise_distance(codes: np.ndarray) -> float:
    n = codes.shape[0]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(codes[i] - codes[j]))
            pairs += 1
    return total / max(pairs, 1)


# ----------------------------------------------------------------------
# Multi-seed sweep helper
# ----------------------------------------------------------------------

def sweep_seeds(seeds: list[int], **train_kwargs) -> list[dict]:
    """Run training for each seed; return summary dict per seed."""
    results = []
    for s in seeds:
        t0 = time.time()
        model, hist = train(seed=s, verbose=False, **train_kwargs)
        wallclock = time.time() - t0
        results.append({
            "seed": s,
            "final_acc": hist["acc"][-1],
            "final_loss": hist["loss"][-1],
            "final_n_distinct_codes": hist["n_distinct_codes"][-1],
            "epochs_run": len(hist["epoch"]),
            "wallclock_s": wallclock,
            "solved": (hist["acc"][-1] >= 1.0
                       and hist["n_distinct_codes"][-1] == 8),
        })
    return results


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no-early-stop", action="store_true")
    args = p.parse_args()

    model, history = train(n_epochs=args.epochs,
                            lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            seed=args.seed,
                            early_stop=not args.no_early_stop)

    data = make_encoder_data()
    print(f"\nFinal accuracy: {accuracy(model, data)*100:.1f}%")
    print(f"Distinct hidden codes: {n_distinct_codes(model, data)}/8")
    print("\nHidden code table (rows = input pattern, cols = hidden unit):")
    table = hidden_code_table(model, data)
    print("  raw activations:")
    for i in range(8):
        print(f"    pattern {i}: ({table[i,0]:.3f}, {table[i,1]:.3f}, {table[i,2]:.3f})")
    print("  binarized at 0.5:")
    for i in range(8):
        b = (table[i] > 0.5).astype(int)
        print(f"    pattern {i}: ({b[0]}, {b[1]}, {b[2]})")
