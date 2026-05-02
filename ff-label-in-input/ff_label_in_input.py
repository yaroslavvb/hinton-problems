"""
Forward-Forward supervised MNIST with the label encoded in the first 10 pixels.

Reproduction of Hinton (2022), "The Forward-Forward Algorithm: Some Preliminary
Investigations", section 3.3 ("Using FF to model top-down effects in
perception" -- the supervised label-in-input variant).

Key idea
--------
MNIST has a natural black border, so we can encode a one-hot label in the first
10 pixels of the flattened image (positions [0..9] of the top row).

* Positive example: (image, true_label)
* Negative example: (image, wrong_label)

A stack of fully connected ReLU layers is trained -- *one layer at a time, no
backprop across layers* -- so each layer learns to produce activations whose
*goodness* (mean squared activation across hidden units) is high for positive
examples and low for negative examples.

Per-layer loss (Hinton 2022, eq. (1)):

    L = log(1 + exp(-(g_pos - theta))) + log(1 + exp(g_neg - theta))

where g = mean(h^2) over hidden units and `theta` is a fixed threshold.

Between layers, hidden activations are L2-normalised so the next layer cannot
cheat by reading off goodness magnitude from the previous layer.

Prediction
----------
At test time we try each of the 10 candidate labels in turn (encoded into the
first 10 pixels), forward-pass through the network, accumulate goodness across
layers (excluding layer 0 -- the standard Hinton recipe), and pick the label
with the highest accumulated goodness.

Reference
---------
https://www.cs.toronto.edu/~hinton/FFA13.pdf

Constraints
-----------
numpy + matplotlib + imageio/pillow + urllib only. No pytorch / tensorflow.
"""

from __future__ import annotations
import argparse
import gzip
import os
import struct
import sys
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

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

CACHE_DIR = os.path.expanduser("~/.cache/hinton-mnist")


def _download(url: str, dst: str) -> None:
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    print(f"  downloading {url} -> {dst}")
    tmp = dst + ".part"
    with urllib.request.urlopen(url) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    os.replace(tmp, dst)


def _read_idx_images(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic in {path}: {magic}")
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return arr


def _read_idx_labels(path: str) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic in {path}: {magic}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist(cache_dir: str = CACHE_DIR
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y).

    Images are uint8-normalised to float32 [0, 1] with shape (N, 28, 28).
    Labels are int32 with shape (N,).
    """
    paths = {k: os.path.join(cache_dir, os.path.basename(v))
             for k, v in MNIST_URLS.items()}
    for k, url in MNIST_URLS.items():
        _download(url, paths[k])

    train_x = _read_idx_images(paths["train_images"]).astype(np.float32) / 255.0
    train_y = _read_idx_labels(paths["train_labels"]).astype(np.int32)
    test_x = _read_idx_images(paths["test_images"]).astype(np.float32) / 255.0
    test_y = _read_idx_labels(paths["test_labels"]).astype(np.int32)
    return train_x, train_y, test_x, test_y


# ---------------------------------------------------------------------------
# Label-in-pixels encoding
# ---------------------------------------------------------------------------

def encode_label_in_pixels(image: np.ndarray, label: int,
                           n_classes: int = 10) -> np.ndarray:
    """Replace the first ``n_classes`` pixels of a flattened image with one-hot.

    Accepts (28, 28) or (784,) images, returns the same shape with a new
    array. The first ``n_classes`` pixels (top row, leftmost ``n_classes``)
    are zeroed; the position corresponding to ``label`` is set to 1.0.
    """
    out = image.astype(np.float32, copy=True)
    flat = out.reshape(-1)
    flat[:n_classes] = 0.0
    flat[label] = 1.0
    return out


def encode_label_in_pixels_batch(images: np.ndarray, labels: np.ndarray,
                                 n_classes: int = 10) -> np.ndarray:
    """Vectorised label encoding for a batch of flattened images.

    images: (B, 784) float32
    labels: (B,) int
    returns: (B, 784) float32
    """
    out = images.copy()
    out[:, :n_classes] = 0.0
    out[np.arange(out.shape[0]), labels] = 1.0
    return out


def make_positive(image: np.ndarray, true_label: int) -> np.ndarray:
    return encode_label_in_pixels(image, int(true_label))


def make_negative(image: np.ndarray, true_label: int,
                  n_classes: int = 10,
                  rng: Optional[np.random.Generator] = None) -> np.ndarray:
    """Encode a wrong label uniformly at random into the first 10 pixels."""
    if rng is None:
        rng = np.random.default_rng()
    wrong = int(rng.integers(0, n_classes - 1))
    if wrong >= true_label:
        wrong += 1  # uniform over {0..9} \ {true_label}
    return encode_label_in_pixels(image, wrong, n_classes=n_classes)


def make_negative_batch(images: np.ndarray, labels: np.ndarray,
                        n_classes: int = 10,
                        rng: Optional[np.random.Generator] = None
                        ) -> np.ndarray:
    """Vectorised: pick a wrong label per row, encode it."""
    if rng is None:
        rng = np.random.default_rng()
    offsets = rng.integers(1, n_classes, size=labels.shape[0]).astype(np.int32)
    wrong = (labels.astype(np.int32) + offsets) % n_classes
    return encode_label_in_pixels_batch(images, wrong, n_classes=n_classes)


def jittered_augmentation(image: np.ndarray, max_shift: int = 2,
                          rng: Optional[np.random.Generator] = None
                          ) -> np.ndarray:
    """Translate a 28x28 image by up to ``max_shift`` pixels in each axis.

    Hinton (2022) reports 0.64% test error with 25-shift augmentation
    (max_shift=2 -> 5x5=25 unique offsets). Empty pixels are filled with 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    if image.ndim != 2:
        raise ValueError(f"jittered_augmentation expects (H, W); got {image.shape}")
    dy = int(rng.integers(-max_shift, max_shift + 1))
    dx = int(rng.integers(-max_shift, max_shift + 1))
    out = np.zeros_like(image)
    H, W = image.shape
    src_y0 = max(0, -dy); src_y1 = min(H, H - dy)
    src_x0 = max(0, -dx); src_x1 = min(W, W - dx)
    dst_y0 = max(0, dy); dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x0 = max(0, dx); dst_x1 = dst_x0 + (src_x1 - src_x0)
    out[dst_y0:dst_y1, dst_x0:dst_x1] = image[src_y0:src_y1, src_x0:src_x1]
    return out


def jittered_augmentation_batch(images: np.ndarray, max_shift: int = 2,
                                rng: Optional[np.random.Generator] = None
                                ) -> np.ndarray:
    """Per-sample random translation by up to ``max_shift`` pixels in each axis.

    Loops in Python over the batch (B is bounded by mini-batch size, so this
    is well under 1 ms per batch). Empty pixels are filled with 0.
    """
    if rng is None:
        rng = np.random.default_rng()
    B, H, W = images.shape
    out = np.zeros_like(images)
    dys = rng.integers(-max_shift, max_shift + 1, size=B)
    dxs = rng.integers(-max_shift, max_shift + 1, size=B)
    for i in range(B):
        d_y = int(dys[i]); d_x = int(dxs[i])
        if d_y == 0 and d_x == 0:
            out[i] = images[i]
            continue
        src_y0 = max(0, -d_y); src_y1 = min(H, H - d_y)
        src_x0 = max(0, -d_x); src_x1 = min(W, W - d_x)
        dst_y0 = max(0, d_y); dst_y1 = dst_y0 + (src_y1 - src_y0)
        dst_x0 = max(0, d_x); dst_x1 = dst_x0 + (src_x1 - src_x0)
        out[i, dst_y0:dst_y1, dst_x0:dst_x1] = images[i, src_y0:src_y1, src_x0:src_x1]
    return out


# ---------------------------------------------------------------------------
# Forward-Forward layer
# ---------------------------------------------------------------------------

@dataclass
class FFLayer:
    """One ReLU layer trained with the FF rule.

    Parameters
    ----------
    W : (in_dim, out_dim)  weight matrix
    b : (out_dim,)         bias
    threshold : scalar     goodness threshold theta (eq. 1 of Hinton 2022)
    """
    W: np.ndarray
    b: np.ndarray
    threshold: float
    # Adam state
    m_W: np.ndarray = field(default=None)
    v_W: np.ndarray = field(default=None)
    m_b: np.ndarray = field(default=None)
    v_b: np.ndarray = field(default=None)
    step: int = 0

    @classmethod
    def init(cls, in_dim: int, out_dim: int, threshold: float,
             rng: np.random.Generator) -> "FFLayer":
        # Glorot-style init.
        scale = np.sqrt(2.0 / in_dim)
        W = (scale * rng.standard_normal((in_dim, out_dim))).astype(np.float32)
        b = np.zeros(out_dim, dtype=np.float32)
        return cls(W=W, b=b, threshold=threshold,
                   m_W=np.zeros_like(W), v_W=np.zeros_like(W),
                   m_b=np.zeros_like(b), v_b=np.zeros_like(b))

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Pre-normalisation hidden activation: ReLU(W^T x + b)."""
        z = x @ self.W + self.b
        return np.maximum(z, 0.0, out=z)

    @staticmethod
    def normalize(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Hinton's between-layer norm: rescale so mean(h^2) = 1.

        From Hinton (2022): "The length of this vector is fixed at the square
        root of the dimensionality of the layer. This means that the average
        squared element value is 1, so the next layer can no longer use the
        absolute length of the input vector to determine its own goodness."
        """
        D = h.shape[-1]
        norm = np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
        return h / norm

    def goodness(self, h: np.ndarray) -> np.ndarray:
        """Mean squared activation along the feature axis -- one scalar per sample."""
        return (h * h).mean(axis=-1)


def ff_layer_loss_grad(layer: FFLayer,
                       x_pos: np.ndarray, x_neg: np.ndarray
                       ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compute the FF loss for one layer and the gradients d(loss)/d(W,b).

    Returns: (loss, grad_W, grad_b, mean_g_pos, mean_g_neg).

    The loss is averaged over the 2*B samples (B positives, B negatives).
    """
    B, D = x_pos.shape[0], layer.W.shape[1]
    z_pos = x_pos @ layer.W + layer.b
    z_neg = x_neg @ layer.W + layer.b
    h_pos = np.maximum(z_pos, 0.0)
    h_neg = np.maximum(z_neg, 0.0)

    g_pos = (h_pos * h_pos).mean(axis=-1)  # (B,)
    g_neg = (h_neg * h_neg).mean(axis=-1)

    theta = layer.threshold
    # Margins: a = g_pos - theta (want large +ve), b = g_neg - theta (want large -ve).
    a_pos = g_pos - theta
    a_neg = g_neg - theta

    # Stable softplus / sigmoid.
    def softplus(x):
        return np.where(x > 0,
                        x + np.log1p(np.exp(-x)),
                        np.log1p(np.exp(x)))

    def sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    loss_pos = softplus(-a_pos)         # = -log(sigmoid(a_pos))
    loss_neg = softplus(a_neg)          # = -log(1 - sigmoid(a_neg))
    loss = float((loss_pos.sum() + loss_neg.sum()) / (2 * B))

    # d loss_pos / d g_pos = -sigmoid(-a_pos) = sigmoid(a_pos) - 1
    # d loss_neg / d g_neg =  sigmoid(a_neg)
    dg_pos = (sigmoid(a_pos) - 1.0) / (2 * B)
    dg_neg = sigmoid(a_neg) / (2 * B)

    # d g / d h = (2/D) * h
    coeff = 2.0 / D
    dh_pos = coeff * h_pos * dg_pos[:, None]
    dh_neg = coeff * h_neg * dg_neg[:, None]

    # Through ReLU: d/dz = d/dh * (z > 0)
    dz_pos = dh_pos * (z_pos > 0)
    dz_neg = dh_neg * (z_neg > 0)

    # d/dW = x^T @ dz, d/db = sum(dz)
    grad_W = (x_pos.T @ dz_pos + x_neg.T @ dz_neg).astype(np.float32)
    grad_b = (dz_pos.sum(axis=0) + dz_neg.sum(axis=0)).astype(np.float32)

    return loss, grad_W, grad_b, float(g_pos.mean()), float(g_neg.mean())


def adam_update(layer: FFLayer, gW: np.ndarray, gb: np.ndarray,
                lr: float, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8) -> None:
    layer.step += 1
    layer.m_W = beta1 * layer.m_W + (1 - beta1) * gW
    layer.v_W = beta2 * layer.v_W + (1 - beta2) * (gW * gW)
    layer.m_b = beta1 * layer.m_b + (1 - beta1) * gb
    layer.v_b = beta2 * layer.v_b + (1 - beta2) * (gb * gb)
    bc1 = 1 - beta1 ** layer.step
    bc2 = 1 - beta2 ** layer.step
    layer.W -= lr * (layer.m_W / bc1) / (np.sqrt(layer.v_W / bc2) + eps)
    layer.b -= lr * (layer.m_b / bc1) / (np.sqrt(layer.v_b / bc2) + eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class FFModel:
    layers: list  # list[FFLayer]
    n_classes: int = 10

    @classmethod
    def init(cls, layer_sizes: tuple, threshold: float,
             rng: np.random.Generator, n_classes: int = 10) -> "FFModel":
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(FFLayer.init(in_dim, out_dim, threshold, rng))
        return cls(layers=layers, n_classes=n_classes)


def build_ff_mlp(layer_sizes: tuple = (784, 2000, 2000, 2000, 2000),
                 threshold: float = 2.0,
                 seed: int = 0,
                 n_classes: int = 10) -> FFModel:
    rng = np.random.default_rng(seed)
    return FFModel.init(layer_sizes, threshold, rng, n_classes=n_classes)


def model_layer_activations(model: FFModel, x: np.ndarray) -> list:
    """Forward through all layers, returning post-ReLU activations per layer.

    Between layers the activation is L2-normalised so each layer sees a
    unit-norm input (this is the canonical Hinton recipe).
    """
    acts = []
    h = x
    for layer in model.layers:
        h_relu = layer.forward(h)
        acts.append(h_relu)
        h = FFLayer.normalize(h_relu)
    return acts


def goodness_per_layer(model: FFModel, x: np.ndarray,
                       skip_first: bool = False) -> np.ndarray:
    """Return (B, n_used) array of per-sample goodness from each used layer."""
    acts = model_layer_activations(model, x)
    if skip_first and len(acts) > 1:
        acts = acts[1:]
    return np.stack([(h * h).mean(axis=-1) for h in acts], axis=1)


def predict_by_goodness(model: FFModel, image: np.ndarray,
                        skip_first: bool = False) -> int:
    """Predict by trying each label and picking the one with highest summed goodness."""
    flat = image.reshape(-1)
    candidates = np.tile(flat, (model.n_classes, 1))
    candidates[:, :model.n_classes] = 0.0
    candidates[np.arange(model.n_classes), np.arange(model.n_classes)] = 1.0
    g = goodness_per_layer(model, candidates, skip_first=skip_first)
    summed = g.sum(axis=1)
    return int(np.argmax(summed))


def predict_by_goodness_batch(model: FFModel, images: np.ndarray,
                              skip_first: bool = False) -> np.ndarray:
    """Vectorised prediction over a batch of images.

    images: (B, 28, 28) or (B, 784)
    returns: (B,) predicted labels
    """
    B = images.shape[0]
    flat = images.reshape(B, -1)
    n = model.n_classes
    # Build a (B*n, 784) tensor of (image, candidate_label) pairs.
    expanded = np.repeat(flat, n, axis=0).copy()    # (B*n, 784)
    label_vec = np.tile(np.arange(n), B)            # (B*n,)
    expanded[:, :n] = 0.0
    expanded[np.arange(B * n), label_vec] = 1.0
    g = goodness_per_layer(model, expanded, skip_first=skip_first)
    summed = g.sum(axis=1).reshape(B, n)
    return summed.argmax(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_epochs: int = 20
    batch_size: int = 128
    lr: float = 0.03
    threshold: float = 2.0
    layer_sizes: tuple = (784, 500, 500)
    n_classes: int = 10
    seed: int = 0
    jitter: bool = False
    jitter_max_shift: int = 2
    train_subset: Optional[int] = None  # None -> full 60K
    eval_every: int = 1
    eval_subset: int = 5000             # speed up per-epoch eval


def _flatten(images: np.ndarray) -> np.ndarray:
    return images.reshape(images.shape[0], -1)


def train(model: FFModel,
          data: tuple,
          cfg: TrainConfig,
          snapshot_callback: Optional[Callable] = None,
          snapshot_every: int = 1,
          verbose: bool = True
          ) -> dict:
    """Train each layer in turn with the FF rule.

    data: (train_x, train_y, test_x, test_y) -- images shaped (N, 28, 28)
          float32 in [0,1], labels int.

    Returns a history dict with per-(layer, epoch) loss / goodness, and
    per-epoch test accuracy.
    """
    train_x, train_y, test_x, test_y = data
    rng = np.random.default_rng(cfg.seed)

    # Optionally subset training data.
    if cfg.train_subset is not None and cfg.train_subset < train_x.shape[0]:
        idx = rng.permutation(train_x.shape[0])[:cfg.train_subset]
        train_x = train_x[idx]
        train_y = train_y[idx]

    N = train_x.shape[0]
    B = cfg.batch_size
    n_layers = len(model.layers)

    history = {
        "epoch": [],
        "loss_per_layer": [[] for _ in range(n_layers)],
        "g_pos_per_layer": [[] for _ in range(n_layers)],
        "g_neg_per_layer": [[] for _ in range(n_layers)],
        "test_acc": [],
        "train_acc": [],
        "wallclock": [],
    }

    eval_idx = rng.permutation(test_x.shape[0])[:cfg.eval_subset]
    eval_x = test_x[eval_idx]
    eval_y = test_y[eval_idx]
    train_eval_idx = rng.permutation(train_x.shape[0])[:cfg.eval_subset]
    eval_train_x = train_x[train_eval_idx]
    eval_train_y = train_y[train_eval_idx]

    t0 = time.time()
    for epoch in range(cfg.n_epochs):
        perm = rng.permutation(N)
        epoch_loss = [0.0 for _ in range(n_layers)]
        epoch_gpos = [0.0 for _ in range(n_layers)]
        epoch_gneg = [0.0 for _ in range(n_layers)]
        n_batches = 0

        for start in range(0, N, B):
            idx = perm[start:start + B]
            xb = train_x[idx]   # (b, 28, 28)
            yb = train_y[idx]
            if cfg.jitter:
                xb = jittered_augmentation_batch(xb, cfg.jitter_max_shift, rng)
            xb_flat = _flatten(xb).copy()

            x_pos = encode_label_in_pixels_batch(xb_flat, yb, cfg.n_classes)
            x_neg = make_negative_batch(xb_flat, yb, cfg.n_classes, rng)

            # Train each layer in turn on its own input. The input to layer L+1
            # is the *normalised* hidden activation of layer L (with no
            # gradient passing back through normalisation, exactly as in
            # Hinton 2022).
            h_pos, h_neg = x_pos, x_neg
            for L, layer in enumerate(model.layers):
                loss, gW, gb, gp, gn = ff_layer_loss_grad(layer, h_pos, h_neg)
                adam_update(layer, gW, gb, cfg.lr)

                epoch_loss[L] += loss
                epoch_gpos[L] += gp
                epoch_gneg[L] += gn

                # Compute next layer's input (no grad).
                z_pos_next = h_pos @ layer.W + layer.b
                z_neg_next = h_neg @ layer.W + layer.b
                h_pos = FFLayer.normalize(np.maximum(z_pos_next, 0.0))
                h_neg = FFLayer.normalize(np.maximum(z_neg_next, 0.0))
            n_batches += 1

        for L in range(n_layers):
            history["loss_per_layer"][L].append(epoch_loss[L] / n_batches)
            history["g_pos_per_layer"][L].append(epoch_gpos[L] / n_batches)
            history["g_neg_per_layer"][L].append(epoch_gneg[L] / n_batches)
        history["epoch"].append(epoch + 1)
        history["wallclock"].append(time.time() - t0)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.n_epochs - 1:
            test_pred = predict_by_goodness_batch(model, eval_x)
            test_acc = float((test_pred == eval_y).mean())
            train_pred = predict_by_goodness_batch(model, eval_train_x)
            train_acc = float((train_pred == eval_train_y).mean())
            history["test_acc"].append(test_acc)
            history["train_acc"].append(train_acc)
        else:
            history["test_acc"].append(history["test_acc"][-1] if history["test_acc"] else float("nan"))
            history["train_acc"].append(history["train_acc"][-1] if history["train_acc"] else float("nan"))

        if verbose:
            losses = " ".join(f"L{L}={epoch_loss[L]/n_batches:.3f}"
                              for L in range(n_layers))
            print(f"epoch {epoch + 1:3d}/{cfg.n_epochs}  "
                  f"{losses}  "
                  f"train={history['train_acc'][-1]*100:.1f}%  "
                  f"test={history['test_acc'][-1]*100:.1f}%  "
                  f"({history['wallclock'][-1]:.1f}s)")

        if snapshot_callback is not None and (epoch + 1) % snapshot_every == 0:
            snapshot_callback(epoch, model, history)

    return history


def evaluate(model: FFModel, x: np.ndarray, y: np.ndarray,
             batch_size: int = 256) -> float:
    """Full-test-set accuracy in batches (memory-friendly)."""
    n = x.shape[0]
    correct = 0
    for s in range(0, n, batch_size):
        xb = x[s:s + batch_size]
        yb = y[s:s + batch_size]
        pred = predict_by_goodness_batch(model, xb)
        correct += int((pred == yb).sum())
    return correct / n


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--layer-sizes", type=str, default="784,500,500",
                   help="Comma-separated list, including 784 input.")
    p.add_argument("--jitter", action="store_true",
                   help="Enable 2-pixel jittered augmentation (Hinton 2022 §3.3).")
    p.add_argument("--jitter-max-shift", type=int, default=2)
    p.add_argument("--train-subset", type=int, default=None,
                   help="Subsample training set for quick experiments.")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--eval-subset", type=int, default=5000)
    p.add_argument("--save", type=str, default=None,
                   help="Save model + history to this .npz path.")
    p.add_argument("--full-test", action="store_true",
                   help="Evaluate on the full 10K test set at the end.")
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    cfg = TrainConfig(n_epochs=args.n_epochs,
                      batch_size=args.batch_size,
                      lr=args.lr,
                      threshold=args.threshold,
                      layer_sizes=layer_sizes,
                      seed=args.seed,
                      jitter=args.jitter,
                      jitter_max_shift=args.jitter_max_shift,
                      train_subset=args.train_subset,
                      eval_every=args.eval_every,
                      eval_subset=args.eval_subset)

    print(f"Loading MNIST from {CACHE_DIR} ...")
    data = load_mnist()
    train_x, train_y, test_x, test_y = data
    print(f"  train: {train_x.shape}, labels: {train_y.shape}")
    print(f"  test:  {test_x.shape}, labels: {test_y.shape}")

    model = build_ff_mlp(layer_sizes=layer_sizes, threshold=args.threshold,
                         seed=args.seed)
    print(f"Model: {len(model.layers)} layers, sizes={layer_sizes}, "
          f"threshold={args.threshold}")

    print("Training...")
    history = train(model, data, cfg, verbose=True)

    if args.full_test:
        print("\nFull test-set evaluation...")
        full_acc = evaluate(model, test_x, test_y)
        print(f"  full test accuracy: {full_acc * 100:.2f}%  "
              f"(error: {(1 - full_acc) * 100:.2f}%)")
        history["full_test_acc"] = full_acc

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        weights = {f"layer{i}_W": L.W for i, L in enumerate(model.layers)}
        biases = {f"layer{i}_b": L.b for i, L in enumerate(model.layers)}
        np.savez(args.save,
                 layer_sizes=np.array(layer_sizes, dtype=np.int32),
                 threshold=np.float32(args.threshold),
                 seed=np.int32(args.seed),
                 history_test_acc=np.array(history["test_acc"], dtype=np.float32),
                 history_train_acc=np.array(history["train_acc"], dtype=np.float32),
                 history_loss_per_layer=np.array(history["loss_per_layer"], dtype=np.float32),
                 history_g_pos=np.array(history["g_pos_per_layer"], dtype=np.float32),
                 history_g_neg=np.array(history["g_neg_per_layer"], dtype=np.float32),
                 history_wallclock=np.array(history["wallclock"], dtype=np.float32),
                 **weights, **biases)
        print(f"Saved model + history to {args.save}")


if __name__ == "__main__":
    main()
