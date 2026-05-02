"""Forward-Forward, top-down recurrent on repeated-frame MNIST (Hinton 2022).

A static MNIST digit is treated as a "video" of repeated identical frames.
Each hidden layer at time t is computed from the L2-normalized activities of
the layer above and below at t-1. The top layer is a one-of-N label, clamped
to a candidate label across all 8 synchronous iterations. Damping mixes the
new activity with the previous state (default: 0.7 new + 0.3 old).

Test-time prediction: run 8 iterations under each candidate label and pick
the label whose hidden-layer goodness summed over iterations 3, 4, 5 is
largest.

Hinton's paper reports 1.31% test error using 4 hidden layers x 2000 ReLU,
trained for 60 epochs on the full 60k MNIST set. This file uses a smaller
network and a 30k subsample so it can train in numpy in a few minutes; it
reaches single-digit % test error on a laptop.

The implementation is pure numpy (no torch). Each forward iteration is a
two-input linear combination (bottom-up + top-down) followed by a ReLU,
then mixed with the old state via damping. Per-layer goodness is
mean-square; FF turns goodness into a logistic loss against the label
"is this a positive (image, true label) example?". Gradients flow only
through the current iteration's forward pass: the previous-step
activations on which it depends are treated as constants. This sidesteps
backprop-through-time entirely and matches the local-update spirit of the
forward-forward algorithm.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MNIST loader
# ---------------------------------------------------------------------------

MNIST_URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}
CACHE_DIR = Path.home() / ".cache" / "hinton-mnist"


def load_mnist(verbose: bool = True):
    """Download MNIST to ~/.cache/hinton-mnist (if absent) and return arrays.

    Returns (X_train, y_train, X_test, y_test) with images flattened to (N, 784)
    in [0, 1] float32 and labels as int64.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out = {}
    for key, url in MNIST_URLS.items():
        local = CACHE_DIR / Path(url).name
        if not local.exists():
            if verbose:
                print(f"  downloading {url} -> {local}")
            urllib.request.urlretrieve(url, local)
        with gzip.open(local, "rb") as f:
            data = f.read()
        if "images" in key:
            _magic, n, rows, cols = struct.unpack(">IIII", data[:16])
            arr = (
                np.frombuffer(data[16:], dtype=np.uint8)
                .reshape(n, rows * cols)
                .astype(np.float32)
                / 255.0
            )
        else:
            _magic, n = struct.unpack(">II", data[:8])
            arr = np.frombuffer(data[8:], dtype=np.uint8).astype(np.int64)
        out[key] = arr
    return out["train_images"], out["train_labels"], out["test_images"], out["test_labels"]


# ---------------------------------------------------------------------------
# Math helpers
# ---------------------------------------------------------------------------

def one_hot(y, n_classes):
    out = np.zeros((y.shape[0], n_classes), dtype=np.float32)
    out[np.arange(y.shape[0]), y] = 1.0
    return out


def l2_normalize(x, eps: float = 1e-8):
    norm = np.sqrt(np.sum(x * x, axis=-1, keepdims=True))
    return x / (norm + eps)


def relu(x):
    return np.maximum(x, 0.0)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def build_recurrent_ff(layer_sizes=(784, 256, 256, 10), damping: float = 0.7,
                       seed: int = 0, init_scale: float = 1.0):
    """Build a recurrent FF network with separate up/down weights per edge.

    layer_sizes: (input_dim, hidden_1, ..., hidden_K, n_classes). Layer 0 is
        clamped to the image, layer L-1 to the candidate label.
    damping: weight on the new activity in the synchronous update; the spec
        wording "0.3 old + 0.7 new" maps to damping=0.7.
    init_scale: multiplier on Glorot-style init.

    Inputs to each layer are L2-normalized activations of adjacent layers, so
    the variance scaling for the weights is per-output-element rather than
    1/sqrt(fan_in). We use sigma = init_scale / sqrt(2) for each direction so
    that bottom-up and top-down contributions sum to roughly unit variance.
    """
    rng = np.random.default_rng(seed)
    L = len(layer_sizes)
    sigma = init_scale / np.sqrt(2.0)
    W_up = []
    W_dn = []
    for i in range(L - 1):
        a, b = layer_sizes[i], layer_sizes[i + 1]
        W_up.append((sigma * rng.standard_normal((a, b))).astype(np.float32))
        W_dn.append((sigma * rng.standard_normal((b, a))).astype(np.float32))
    bias = [None] * L
    for k in range(1, L - 1):
        bias[k] = np.zeros(layer_sizes[k], dtype=np.float32)
    return {
        "sizes": tuple(layer_sizes),
        "L": L,
        "damping": float(damping),
        "W_up": W_up,
        "W_dn": W_dn,
        "b": bias,
    }


def init_states(model, image, label_oh):
    L = model["L"]
    sizes = model["sizes"]
    B = image.shape[0]
    states = [None] * L
    states[0] = image.astype(np.float32, copy=False)
    states[-1] = label_oh.astype(np.float32, copy=False)
    for k in range(1, L - 1):
        states[k] = np.zeros((B, sizes[k]), dtype=np.float32)
    return states


def synchronous_iterate(model, image, label_oh, n_iters: int = 8,
                        return_history: bool = False):
    """Run n_iters synchronous updates of every hidden layer.

    All hidden layers update from t-1 inputs (synchronous semantics). Input
    layer (image) and output layer (label) are clamped throughout.
    """
    states = init_states(model, image, label_oh)
    history = None
    if return_history:
        history = [tuple(s.copy() for s in states)]
    L = model["L"]
    damping = model["damping"]
    for _t in range(n_iters):
        new_states = list(states)
        for k in range(1, L - 1):
            up = l2_normalize(states[k - 1]) @ model["W_up"][k - 1]
            dn = l2_normalize(states[k + 1]) @ model["W_dn"][k]
            pre = up + dn + model["b"][k]
            new_act = relu(pre)
            new_states[k] = damping * new_act + (1.0 - damping) * states[k]
        states = new_states
        if return_history:
            history.append(tuple(s.copy() for s in states))
    if return_history:
        return states, history
    return states


def goodness(states_at_t, k):
    """Mean-square activity of layer k (per-sample). Compared against the
    fixed scalar threshold (default 1.0) - equivalent to sum-of-squares with
    a per-unit threshold of the same value, but in scaled coordinates so the
    sigmoid is not in deep saturation throughout training."""
    h = states_at_t[k]
    return np.mean(h * h, axis=1)


# ---------------------------------------------------------------------------
# Local FF training
# ---------------------------------------------------------------------------

def compute_grads(model, image, label_oh, target_per_sample,
                  n_iters: int = 8,
                  eval_iters_one_indexed=(3, 4, 5, 6, 7, 8),
                  threshold: float = 1.0):
    """Run forward iterations and accumulate local FF gradients.

    target_per_sample: (B,) float32. 1.0 for positive (push goodness up),
                        0.0 for negative (push goodness down).
    eval_iters_one_indexed: 1-indexed iterations to evaluate goodness at.
    threshold: per-unit-mean-square threshold for goodness.

    Gradients flow only through the current iteration's forward pass; the
    previous-step activations on which the inputs depend are detached.
    """
    L = model["L"]
    damping = model["damping"]
    B = image.shape[0]

    states = init_states(model, image, label_oh)

    dW_up = [np.zeros_like(w) for w in model["W_up"]]
    dW_dn = [np.zeros_like(w) for w in model["W_dn"]]
    db = [None] * L
    for k in range(1, L - 1):
        db[k] = np.zeros_like(model["b"][k])

    eval_set = set(eval_iters_one_indexed)

    total_loss = 0.0
    total_count = 0
    target_col = target_per_sample.astype(np.float32).reshape(-1, 1)

    for loop_t in range(n_iters):
        new_states = list(states)
        cache = {}  # k -> (norm_below, norm_above, pre, mask)
        for k in range(1, L - 1):
            norm_below = l2_normalize(states[k - 1])
            norm_above = l2_normalize(states[k + 1])
            pre = (
                norm_below @ model["W_up"][k - 1]
                + norm_above @ model["W_dn"][k]
                + model["b"][k]
            )
            new_act = relu(pre)
            new_states[k] = damping * new_act + (1.0 - damping) * states[k]
            cache[k] = (norm_below, norm_above, pre)

        iter_label = loop_t + 1
        if iter_label in eval_set:
            for k in range(1, L - 1):
                norm_below, norm_above, pre = cache[k]
                h = new_states[k]
                dim = h.shape[1]
                g = np.mean(h * h, axis=1)
                logit = np.clip(g - threshold, -50.0, 50.0)
                s = 1.0 / (1.0 + np.exp(-logit))
                loss = -(
                    target_per_sample * np.log(s + 1e-12)
                    + (1.0 - target_per_sample) * np.log(1.0 - s + 1e-12)
                )
                total_loss += float(loss.sum())
                total_count += B

                dl_dg = (s - target_per_sample).astype(np.float32)
                # dg/dh = 2*h/dim
                dl_dh = (dl_dg[:, None] * (2.0 / dim) * h).astype(np.float32)
                dl_dnew = damping * dl_dh
                dl_dpre = (dl_dnew * (pre > 0).astype(np.float32))
                dW_up[k - 1] += norm_below.T @ dl_dpre / B
                dW_dn[k] += norm_above.T @ dl_dpre / B
                db[k] += dl_dpre.mean(axis=0)

        states = new_states

    avg_loss = total_loss / max(total_count, 1)
    return dW_up, dW_dn, db, avg_loss


class AdamOpt:
    """Adam optimizer with per-tensor first/second moments. Matches Kingma 2014."""

    def __init__(self, model, lr: float = 3e-3,
                 b1: float = 0.9, b2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.b1 = b1
        self.b2 = b2
        self.eps = eps
        self.t = 0
        self.m_W_up = [np.zeros_like(w) for w in model["W_up"]]
        self.v_W_up = [np.zeros_like(w) for w in model["W_up"]]
        self.m_W_dn = [np.zeros_like(w) for w in model["W_dn"]]
        self.v_W_dn = [np.zeros_like(w) for w in model["W_dn"]]
        self.m_b = [None] * model["L"]
        self.v_b = [None] * model["L"]
        for k in range(1, model["L"] - 1):
            self.m_b[k] = np.zeros_like(model["b"][k])
            self.v_b[k] = np.zeros_like(model["b"][k])

    def step(self, model, dW_up, dW_dn, db):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for i in range(len(dW_up)):
            self.m_W_up[i] = self.b1 * self.m_W_up[i] + (1.0 - self.b1) * dW_up[i]
            self.v_W_up[i] = self.b2 * self.v_W_up[i] + (1.0 - self.b2) * (dW_up[i] ** 2)
            mh = self.m_W_up[i] / bc1
            vh = self.v_W_up[i] / bc2
            model["W_up"][i] -= self.lr * mh / (np.sqrt(vh) + self.eps)
            self.m_W_dn[i] = self.b1 * self.m_W_dn[i] + (1.0 - self.b1) * dW_dn[i]
            self.v_W_dn[i] = self.b2 * self.v_W_dn[i] + (1.0 - self.b2) * (dW_dn[i] ** 2)
            mh = self.m_W_dn[i] / bc1
            vh = self.v_W_dn[i] / bc2
            model["W_dn"][i] -= self.lr * mh / (np.sqrt(vh) + self.eps)
        for k in range(1, model["L"] - 1):
            if db[k] is None:
                continue
            self.m_b[k] = self.b1 * self.m_b[k] + (1.0 - self.b1) * db[k]
            self.v_b[k] = self.b2 * self.v_b[k] + (1.0 - self.b2) * (db[k] ** 2)
            mh = self.m_b[k] / bc1
            vh = self.v_b[k] / bc2
            model["b"][k] -= self.lr * mh / (np.sqrt(vh) + self.eps)


# ---------------------------------------------------------------------------
# Prediction
# ---------------------------------------------------------------------------

def predict_by_iteration_goodness(model, image, n_classes: int = 10,
                                  n_iters: int = 8,
                                  accumulate_iters_one_indexed=(3, 4, 5)):
    """For each candidate label, run synchronous_iterate then accumulate goodness.

    Returns (predictions, goodness_per_class). Goodness is the sum across all
    hidden layers and across the requested 1-indexed iterations.
    """
    B = image.shape[0]
    L = model["L"]
    damping = model["damping"]
    accumulate_set = set(accumulate_iters_one_indexed)

    all_g = np.zeros((B, n_classes), dtype=np.float32)
    for c in range(n_classes):
        label_oh = np.zeros((B, n_classes), dtype=np.float32)
        label_oh[:, c] = 1.0
        states = init_states(model, image, label_oh)
        for loop_t in range(n_iters):
            new_states = list(states)
            for k in range(1, L - 1):
                up = l2_normalize(states[k - 1]) @ model["W_up"][k - 1]
                dn = l2_normalize(states[k + 1]) @ model["W_dn"][k]
                pre = up + dn + model["b"][k]
                new_act = relu(pre)
                new_states[k] = damping * new_act + (1.0 - damping) * states[k]
            states = new_states
            iter_label = loop_t + 1
            if iter_label in accumulate_set:
                for k in range(1, L - 1):
                    all_g[:, c] += np.mean(states[k] ** 2, axis=1)
    return np.argmax(all_g, axis=1), all_g


def evaluate(model, X, y, batch_size: int = 512, n_iters: int = 8,
             accumulate_iters_one_indexed=(3, 4, 5)):
    n = X.shape[0]
    correct = 0
    for start in range(0, n, batch_size):
        x_b = X[start:start + batch_size]
        y_b = y[start:start + batch_size]
        pred, _ = predict_by_iteration_goodness(
            model, x_b,
            n_classes=int(model["sizes"][-1]),
            n_iters=n_iters,
            accumulate_iters_one_indexed=accumulate_iters_one_indexed,
        )
        correct += int((pred == y_b).sum())
    return correct / n


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(model, X_train, y_train, X_test=None, y_test=None,
          n_epochs: int = 8, batch_size: int = 256, lr: float = 3e-3,
          n_iters: int = 8,
          train_eval_iters_one_indexed=(3, 4, 5, 6, 7, 8),
          test_iters_one_indexed=(3, 4, 5),
          threshold: float = 1.0, seed: int = 0, eval_every: int = 1,
          eval_test_subset: int | None = None):
    """Train recurrent FF with positive/negative pairs per minibatch.

    Each minibatch builds:
        positives = (image, true label)
        negatives = (image, random wrong label)
    Both run through the same forward pass. The gradient of every hidden
    layer is the FF logistic-on-goodness rule, applied at iterations in
    train_eval_iters_one_indexed.
    """
    rng = np.random.default_rng(seed + 1000)
    n_train = X_train.shape[0]
    n_classes = int(model["sizes"][-1])
    opt = AdamOpt(model, lr=lr)

    history = {"epoch": [], "train_loss": [], "test_err": [],
               "wallclock": [], "n_train": int(n_train)}
    t0 = time.time()
    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            x_b = X_train[idx]
            y_b = y_train[idx]
            B = len(idx)

            offsets = rng.integers(1, n_classes, size=B)
            wrong = (y_b + offsets) % n_classes

            x_combined = np.concatenate([x_b, x_b], axis=0)
            label_combined = np.concatenate(
                [one_hot(y_b, n_classes), one_hot(wrong, n_classes)], axis=0
            )
            target = np.concatenate(
                [np.ones(B, dtype=np.float32), np.zeros(B, dtype=np.float32)]
            )

            dW_up, dW_dn, db, batch_loss = compute_grads(
                model, x_combined, label_combined, target,
                n_iters=n_iters,
                eval_iters_one_indexed=train_eval_iters_one_indexed,
                threshold=threshold,
            )
            opt.step(model, dW_up, dW_dn, db)
            epoch_loss += batch_loss
            n_batches += 1

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["wallclock"].append(time.time() - t0)

        msg = f"  epoch {epoch + 1:>2}/{n_epochs}  loss={avg_loss:.4f}"
        if X_test is not None and y_test is not None and (epoch + 1) % eval_every == 0:
            if eval_test_subset is not None and eval_test_subset < X_test.shape[0]:
                eX, ey = X_test[:eval_test_subset], y_test[:eval_test_subset]
            else:
                eX, ey = X_test, y_test
            test_acc = evaluate(
                model, eX, ey, batch_size=512, n_iters=n_iters,
                accumulate_iters_one_indexed=test_iters_one_indexed,
            )
            test_err = 1.0 - test_acc
            history["test_err"].append(test_err)
            msg += f"  test_err={test_err * 100:.2f}%"
        else:
            history["test_err"].append(None)
        msg += f"  t={time.time() - t0:.1f}s"
        print(msg, flush=True)
    return history


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def save_model(model, path: str):
    """Save weights and metadata to a .npz archive."""
    blob = {
        "sizes": np.array(model["sizes"], dtype=np.int64),
        "damping": np.array([model["damping"]], dtype=np.float32),
    }
    for i, w in enumerate(model["W_up"]):
        blob[f"W_up_{i}"] = w
    for i, w in enumerate(model["W_dn"]):
        blob[f"W_dn_{i}"] = w
    for k, b in enumerate(model["b"]):
        if b is not None:
            blob[f"b_{k}"] = b
    np.savez(path, **blob)


def load_model(path: str):
    """Inverse of save_model."""
    blob = np.load(path)
    sizes = tuple(int(x) for x in blob["sizes"])
    damping = float(blob["damping"][0])
    L = len(sizes)
    W_up = [blob[f"W_up_{i}"] for i in range(L - 1)]
    W_dn = [blob[f"W_dn_{i}"] for i in range(L - 1)]
    b = [None] * L
    for k in range(1, L - 1):
        b[k] = blob[f"b_{k}"]
    return {"sizes": sizes, "L": L, "damping": damping,
            "W_up": W_up, "W_dn": W_dn, "b": b}


def get_git_commit() -> str:
    try:
        out = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        )
        return out.decode().strip()
    except Exception:
        return "unknown"


def main():
    p = argparse.ArgumentParser(description="Recurrent FF on MNIST (Hinton 2022)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=8,
                   help="Synchronous iterations per forward pass.")
    p.add_argument("--damping", type=float, default=0.7,
                   help="Weight on the new activity in the synchronous update "
                        "(spec: '0.3 old + 0.7 new' -> damping=0.7).")
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--n-train", type=int, default=30000,
                   help="Subsample of the 60k training set.")
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-test-subset", type=int, default=None)
    p.add_argument("--results-json", type=str, default=None)
    p.add_argument("--save-model", type=str, default=None,
                   help="If set, save trained weights to this .npz path.")
    args = p.parse_args()

    print("[load] MNIST...")
    Xtr, ytr, Xte, yte = load_mnist()
    print(f"[load] train={Xtr.shape} test={Xte.shape}")

    if args.n_train < Xtr.shape[0]:
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(Xtr.shape[0])[:args.n_train]
        Xtr, ytr = Xtr[idx], ytr[idx]
        print(f"[load] subsampled to {args.n_train}")

    sizes = tuple([784] + [args.hidden] * args.n_hidden_layers + [10])
    print(f"[model] sizes={sizes} damping={args.damping} threshold={args.threshold} init_scale={args.init_scale}")
    model = build_recurrent_ff(
        layer_sizes=sizes, damping=args.damping, seed=args.seed,
        init_scale=args.init_scale,
    )

    print(f"[train] epochs={args.n_epochs} batch={args.batch_size} lr={args.lr} n_iters={args.n_iters}")
    t0 = time.time()
    history = train(
        model, Xtr, ytr, Xte, yte,
        n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
        n_iters=args.n_iters, threshold=args.threshold, seed=args.seed,
        eval_every=args.eval_every,
        eval_test_subset=args.eval_test_subset,
    )
    train_wall = time.time() - t0

    print("[eval] full test set...")
    final_acc = evaluate(model, Xte, yte, batch_size=512, n_iters=args.n_iters)
    final_err = 1.0 - final_acc
    print(f"[result] final test error: {final_err * 100:.2f}% (acc {final_acc * 100:.2f}%)")
    print(f"[result] training wallclock: {train_wall:.1f}s")

    if args.save_model:
        os.makedirs(os.path.dirname(os.path.abspath(args.save_model)) or ".", exist_ok=True)
        save_model(model, args.save_model)
        print(f"[result] saved model -> {args.save_model}")

    if args.results_json:
        results = {
            "args": vars(args),
            "sizes": list(sizes),
            "final_test_err": float(final_err),
            "final_test_acc": float(final_acc),
            "train_wallclock_sec": float(train_wall),
            "history": history,
            "env": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "git_commit": get_git_commit(),
            },
        }
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"[result] wrote {args.results_json}")


if __name__ == "__main__":
    main()
