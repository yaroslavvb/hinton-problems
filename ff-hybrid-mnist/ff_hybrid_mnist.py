"""
Forward-Forward unsupervised on MNIST with hybrid-image negatives
(Hinton 2022, "The forward-forward algorithm: some preliminary investigations").

Each layer is trained locally to push its **goodness** (sum of squared
post-ReLU activations) UP for real digits and DOWN for hybrid images that
mix two digits via a smoothly-thresholded random mask.

After unsupervised training the only "supervised" learning is a single
linear softmax on top of the L2-normalized activations of the last 3
hidden layers.

Files in this folder:
    ff_hybrid_mnist.py            -- this file (model + train + eval, CLI)
    visualize_ff_hybrid_mnist.py  -- static viz: hybrid examples,
                                     per-layer goodness, classifier acc
    make_ff_hybrid_mnist_gif.py   -- animated GIF of training dynamics
"""

from __future__ import annotations
import argparse
import gzip
import os
import platform
import sys
import time
import urllib.request

import numpy as np


# ----------------------------------------------------------------------
# MNIST
# ----------------------------------------------------------------------

CACHE = os.path.expanduser("~/.cache/hinton-mnist")
URLS = {
    "train_images": "https://storage.googleapis.com/cvdf-datasets/mnist/train-images-idx3-ubyte.gz",
    "train_labels": "https://storage.googleapis.com/cvdf-datasets/mnist/train-labels-idx1-ubyte.gz",
    "test_images":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-images-idx3-ubyte.gz",
    "test_labels":  "https://storage.googleapis.com/cvdf-datasets/mnist/t10k-labels-idx1-ubyte.gz",
}


def load_mnist() -> dict:
    """Download (once) and decode MNIST. Returns numpy arrays in [0, 1]."""
    os.makedirs(CACHE, exist_ok=True)
    out = {}
    for k, url in URLS.items():
        path = os.path.join(CACHE, os.path.basename(url))
        if not os.path.exists(path):
            print(f"  downloading {url} -> {path}")
            urllib.request.urlretrieve(url, path)
        with gzip.open(path, "rb") as f:
            data = f.read()
        if "images" in k:
            out[k] = (np.frombuffer(data, np.uint8, offset=16)
                      .reshape(-1, 28, 28).astype(np.float32) / 255.0)
        else:
            out[k] = np.frombuffer(data, np.uint8, offset=8).astype(np.int64)
    return out


# ----------------------------------------------------------------------
# Hybrid-image negatives
# ----------------------------------------------------------------------

def _blur_pass(m: np.ndarray) -> np.ndarray:
    """One [1/4, 1/2, 1/4] separable blur pass with edge padding.

    Works on both single-image (H, W) and batched (B, H, W) inputs.
    """
    if m.ndim == 2:
        m = m[None]
        squeeze = True
    else:
        squeeze = False

    # blur along last axis
    pad = np.pad(m, ((0, 0), (0, 0), (1, 1)), mode="edge")
    m = 0.25 * pad[:, :, :-2] + 0.5 * pad[:, :, 1:-1] + 0.25 * pad[:, :, 2:]
    # blur along middle axis
    pad = np.pad(m, ((0, 0), (1, 1), (0, 0)), mode="edge")
    m = 0.25 * pad[:, :-2, :] + 0.5 * pad[:, 1:-1, :] + 0.25 * pad[:, 2:, :]

    return m[0] if squeeze else m


def make_random_mask(shape: tuple, rng: np.random.Generator,
                     n_blur: int = 6) -> np.ndarray:
    """Hinton's hybrid-image mask:

    1. start with a uniform random binary mask
    2. blur with [1/4, 1/2, 1/4] kernel, repeated n_blur times in each axis
    3. threshold at 0.5

    The result is a binary mask with large coherent regions (size ~2^n_blur
    pixels) -- preserves short-range correlations, destroys long-range
    shape correlations.
    """
    m = (rng.random(shape, dtype=np.float32) > 0.5).astype(np.float32)
    for _ in range(n_blur):
        m = _blur_pass(m)
    return (m > 0.5).astype(np.float32)


def make_hybrid_image(digit_a: np.ndarray, digit_b: np.ndarray,
                      rng: np.random.Generator | None = None,
                      n_blur: int = 6) -> np.ndarray:
    """Mix two digits using a smoothly-thresholded random mask.

    digit_a, digit_b: (28, 28) arrays in [0, 1].
    Returns:        a (28, 28) mixture where each pixel is taken either
                    from digit_a (mask=1) or digit_b (mask=0).
    """
    if rng is None:
        rng = np.random.default_rng()
    mask = make_random_mask(digit_a.shape, rng, n_blur=n_blur)
    return mask * digit_a + (1.0 - mask) * digit_b


def make_hybrid_batch(images_a: np.ndarray, images_b: np.ndarray,
                      rng: np.random.Generator,
                      n_blur: int = 6) -> np.ndarray:
    """Vectorized make_hybrid_image: a separate mask per pair."""
    mask = make_random_mask(images_a.shape, rng, n_blur=n_blur)
    return mask * images_a + (1.0 - mask) * images_b


# ----------------------------------------------------------------------
# FF layer + goodness
# ----------------------------------------------------------------------

def goodness(activations: np.ndarray) -> np.ndarray:
    """Per-neuron mean of squared activations.

    This is the convention from Hinton's reference PyTorch implementation
    in the 2022 FF paper: g = mean(h^2). Using the *mean* (not the sum)
    decouples the goodness scale from the layer width, so a single
    threshold (~2.0) works regardless of how wide the layer is. With
    sum-of-squares the threshold would have to scale with layer width
    or the sigmoid saturates and gradients vanish.
    """
    return (activations ** 2).mean(axis=-1)


def l2_normalize(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """L2-normalize each row. Strips the goodness signal before passing
    activations to the next layer (so the next layer can't trivially
    reuse the previous layer's confidence)."""
    norm = np.sqrt((x ** 2).sum(axis=-1, keepdims=True) + eps)
    return x / norm


class FFLayer:
    """A single ReLU layer with locally-trained Forward-Forward weights."""

    def __init__(self, in_dim: int, out_dim: int,
                 rng: np.random.Generator,
                 init_scale: float | None = None):
        # FF inputs are L2-normalized (unit norm), so the standard He
        # 1/sqrt(in_dim) init gives preact variance ~ 1/in_dim and
        # mean(h^2) far below the goodness threshold (~2). Calibrate so
        # that mean(h^2) is near the threshold from the start: with
        # unit-norm input, Var(W x) = sigma_W^2, so sigma_W ~ sqrt(2)
        # gives mean(h^2) ~ 1.
        if init_scale is None:
            init_scale = np.sqrt(2.0)
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.W = (init_scale * rng.standard_normal((in_dim, out_dim))
                  ).astype(np.float32)
        self.b = np.zeros(out_dim, dtype=np.float32)
        # Adam state (initialized lazily)
        self.t = 0
        self.mW = np.zeros_like(self.W)
        self.vW = np.zeros_like(self.W)
        self.mb = np.zeros_like(self.b)
        self.vb = np.zeros_like(self.b)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        z = x @ self.W + self.b
        h = np.maximum(z, 0.0).astype(np.float32)
        return z, h


def ff_layer_step(layer: FFLayer,
                  x_pos: np.ndarray, x_neg: np.ndarray,
                  threshold: float, lr: float,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8,
                  weight_decay: float = 0.0) -> dict:
    """One Adam update on the FF objective for a single layer.

    FF objective (sigmoid-style, as in Hinton 2022):

        L_pos = softplus(threshold - g_pos)
        L_neg = softplus(g_neg - threshold)

    where g = sum(h^2). The positive examples are pushed to have goodness
    above threshold; negatives are pushed below.
    """
    z_pos = x_pos @ layer.W + layer.b
    h_pos = np.maximum(z_pos, 0.0).astype(np.float32)
    z_neg = x_neg @ layer.W + layer.b
    h_neg = np.maximum(z_neg, 0.0).astype(np.float32)
    N = float(layer.out_dim)
    g_pos = (h_pos ** 2).mean(axis=1)  # per-neuron mean goodness
    g_neg = (h_neg ** 2).mean(axis=1)

    # dL/dg
    # for pos: L = softplus(thr - g_pos); dL/dg = -sigmoid(thr - g_pos)
    # for neg: L = softplus(g_neg - thr); dL/dg = +sigmoid(g_neg - thr)
    sig_pos = 1.0 / (1.0 + np.exp(np.clip(g_pos - threshold, -50, 50)))
    sig_neg = 1.0 / (1.0 + np.exp(np.clip(threshold - g_neg, -50, 50)))
    dL_dg_pos = -sig_pos
    dL_dg_neg = sig_neg

    B = float(x_pos.shape[0] + x_neg.shape[0])
    # g = (1/N) sum(h^2), so dg/dh_i = 2 h_i / N.
    # dL/dh = dL/dg * 2 h / N
    # dL/dz = dL/dh * 1[z > 0]
    dL_dz_pos = (dL_dg_pos[:, None] * (2.0 / N) * h_pos) * (z_pos > 0)
    dL_dz_neg = (dL_dg_neg[:, None] * (2.0 / N) * h_neg) * (z_neg > 0)

    dW = (x_pos.T @ dL_dz_pos + x_neg.T @ dL_dz_neg) / B
    db = (dL_dz_pos.sum(axis=0) + dL_dz_neg.sum(axis=0)) / B
    if weight_decay:
        dW = dW + weight_decay * layer.W

    # Adam update
    layer.t += 1
    layer.mW = beta1 * layer.mW + (1 - beta1) * dW
    layer.vW = beta2 * layer.vW + (1 - beta2) * (dW ** 2)
    layer.mb = beta1 * layer.mb + (1 - beta1) * db
    layer.vb = beta2 * layer.vb + (1 - beta2) * (db ** 2)
    bc1 = 1 - beta1 ** layer.t
    bc2 = 1 - beta2 ** layer.t
    layer.W -= lr * (layer.mW / bc1) / (np.sqrt(layer.vW / bc2) + eps)
    layer.b -= lr * (layer.mb / bc1) / (np.sqrt(layer.vb / bc2) + eps)

    # Reporting metrics (no extra forward pass)
    L_pos = np.log1p(np.exp(np.clip(threshold - g_pos, -50, 50)))
    L_neg = np.log1p(np.exp(np.clip(g_neg - threshold, -50, 50)))
    return dict(
        loss=float(0.5 * (L_pos.mean() + L_neg.mean())),
        loss_pos=float(L_pos.mean()),
        loss_neg=float(L_neg.mean()),
        g_pos=float(g_pos.mean()),
        g_neg=float(g_neg.mean()),
        acc=float(0.5 * ((g_pos > threshold).mean()
                         + (g_neg < threshold).mean())),
    )


def build_ff_mlp(layer_sizes: tuple = (784, 2000, 2000, 2000, 2000),
                 rng: np.random.Generator | None = None) -> list[FFLayer]:
    if rng is None:
        rng = np.random.default_rng(0)
    return [FFLayer(in_d, out_d, rng)
            for in_d, out_d in zip(layer_sizes[:-1], layer_sizes[1:])]


def forward_all_layers(layers: list[FFLayer], x: np.ndarray
                       ) -> list[np.ndarray]:
    """Forward through every layer; return *raw* (un-normalized)
    post-ReLU activations as a list.

    Every layer L2-normalizes its input first. The input to layer 1 is
    the flattened pixel vector divided by its norm; the input to layer
    i+1 is the L2-normalized activations of layer i. This is the
    convention from Hinton's 2022 PyTorch reference: each layer sees a
    direction vector, never the raw goodness signal of the previous
    layer."""
    activations = []
    cur = l2_normalize(x)
    for layer in layers:
        _, h = layer.forward(cur)
        activations.append(h)
        cur = l2_normalize(h)
    return activations


# ----------------------------------------------------------------------
# Unsupervised training
# ----------------------------------------------------------------------

def train_unsupervised(layers: list[FFLayer],
                       X_train: np.ndarray,
                       n_epochs: int,
                       lr: float,
                       batch_size: int,
                       threshold: float,
                       rng: np.random.Generator,
                       weight_decay: float = 0.0,
                       n_blur: int = 6,
                       snapshot_callback=None,
                       verbose: bool = True) -> dict:
    """Train every layer with FF on positive (real) and negative
    (hybrid) examples. Layers are trained simultaneously, but each layer
    sees the L2-normalized output of the previous layer (no
    backpropagation between layers)."""
    n = X_train.shape[0]
    L = len(layers)
    history = {"epoch": [], "wallclock": []}
    for li in range(L):
        history[f"layer{li+1}_loss"] = []
        history[f"layer{li+1}_g_pos"] = []
        history[f"layer{li+1}_g_neg"] = []
        history[f"layer{li+1}_acc"] = []

    t0 = time.time()
    for epoch in range(n_epochs):
        order_pos = rng.permutation(n)
        order_neg_a = rng.permutation(n)
        order_neg_b = rng.permutation(n)
        # ensure neg pairs are different digits most of the time -- a
        # collision (a==b) on ~1/n examples is harmless
        epoch_stats = [dict(loss=0.0, g_pos=0.0, g_neg=0.0, acc=0.0)
                       for _ in range(L)]
        n_batches = 0
        for i in range(0, n, batch_size):
            idx_pos = order_pos[i:i + batch_size]
            idx_a = order_neg_a[i:i + batch_size]
            idx_b = order_neg_b[i:i + batch_size]
            B = len(idx_pos)
            if B != len(idx_a) or B != len(idx_b):
                continue  # final ragged batch alignment guard
            x_pos_img = X_train[idx_pos]
            x_neg_img = make_hybrid_batch(X_train[idx_a], X_train[idx_b],
                                          rng, n_blur=n_blur)
            cur_pos = l2_normalize(x_pos_img.reshape(B, -1))
            cur_neg = l2_normalize(x_neg_img.reshape(B, -1))
            for li, layer in enumerate(layers):
                stats = ff_layer_step(layer, cur_pos, cur_neg,
                                       threshold=threshold, lr=lr,
                                       weight_decay=weight_decay)
                for k in epoch_stats[li]:
                    epoch_stats[li][k] += stats[k]
                # propagate forward (no grad)
                _, h_pos = layer.forward(cur_pos)
                _, h_neg = layer.forward(cur_neg)
                cur_pos = l2_normalize(h_pos)
                cur_neg = l2_normalize(h_neg)
            n_batches += 1

        for li in range(L):
            for k in epoch_stats[li]:
                epoch_stats[li][k] /= max(n_batches, 1)
            history[f"layer{li+1}_loss"].append(epoch_stats[li]["loss"])
            history[f"layer{li+1}_g_pos"].append(epoch_stats[li]["g_pos"])
            history[f"layer{li+1}_g_neg"].append(epoch_stats[li]["g_neg"])
            history[f"layer{li+1}_acc"].append(epoch_stats[li]["acc"])
        history["epoch"].append(epoch + 1)
        history["wallclock"].append(time.time() - t0)

        if verbose:
            line = f"epoch {epoch+1:3d}  ({history['wallclock'][-1]:6.1f}s)"
            for li in range(L):
                line += (f"  L{li+1} loss={epoch_stats[li]['loss']:.3f}"
                         f" acc={epoch_stats[li]['acc']*100:5.1f}%")
            print(line)

        if snapshot_callback is not None:
            snapshot_callback(epoch, layers, history)

    return history


# ----------------------------------------------------------------------
# Linear softmax head on top-k layers
# ----------------------------------------------------------------------

def features_top_k(layers: list[FFLayer], X: np.ndarray,
                   top_k: int = 3) -> np.ndarray:
    """Concatenated L2-normalized activations of the last `top_k` layers."""
    flat = X.reshape(X.shape[0], -1)
    acts = forward_all_layers(layers, flat)
    use = acts[-top_k:]
    feats = [l2_normalize(a) for a in use]
    return np.concatenate(feats, axis=1)


def fit_softmax_on_top_layers(layers: list[FFLayer],
                              mnist: dict,
                              top_k: int = 3,
                              n_epochs: int = 20,
                              lr: float = 0.05,
                              batch_size: int = 256,
                              weight_decay: float = 1e-4,
                              rng: np.random.Generator | None = None,
                              verbose: bool = True) -> dict:
    """Fit a single linear softmax classifier on top of the FF features.

    The MLP weights are NOT updated here -- only the new (D, 10) softmax
    head. This is the only labeled-supervised step in the whole pipeline.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    F_train = features_top_k(layers, mnist["train_images"], top_k=top_k)
    F_test = features_top_k(layers, mnist["test_images"], top_k=top_k)
    y_train = mnist["train_labels"]
    y_test = mnist["test_labels"]
    n_classes = 10
    in_dim = F_train.shape[1]
    W = (0.01 * rng.standard_normal((in_dim, n_classes))).astype(np.float32)
    b = np.zeros(n_classes, dtype=np.float32)
    n = F_train.shape[0]

    history = {"epoch": [], "train_acc": [], "test_acc": [],
               "train_err": [], "test_err": [], "loss": []}

    for ep in range(n_epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        n_batches = 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = F_train[idx]
            yb = y_train[idx]
            logits = xb @ W + b
            logits = logits - logits.max(axis=1, keepdims=True)
            p = np.exp(logits)
            p /= p.sum(axis=1, keepdims=True)
            ep_loss += float(-np.log(p[np.arange(len(yb)), yb] + 1e-12).mean())
            target = np.zeros_like(p)
            target[np.arange(len(yb)), yb] = 1.0
            dlogits = (p - target) / len(yb)
            dW = xb.T @ dlogits + weight_decay * W
            db = dlogits.sum(axis=0)
            W -= lr * dW
            b -= lr * db
            n_batches += 1

        train_pred = (F_train @ W + b).argmax(axis=1)
        test_pred = (F_test @ W + b).argmax(axis=1)
        train_acc = float((train_pred == y_train).mean())
        test_acc = float((test_pred == y_test).mean())
        history["epoch"].append(ep + 1)
        history["loss"].append(ep_loss / max(n_batches, 1))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_err"].append(1.0 - train_acc)
        history["test_err"].append(1.0 - test_acc)
        if verbose:
            print(f"  softmax epoch {ep+1:2d}: "
                  f"loss={history['loss'][-1]:.3f}  "
                  f"train_acc={train_acc*100:5.2f}%  "
                  f"test_acc={test_acc*100:5.2f}%")

    return dict(W=W, b=b, history=history)


# ----------------------------------------------------------------------
# Environment / reproducibility
# ----------------------------------------------------------------------

def env_info() -> dict:
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor() or "unknown",
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30,
                   help="FF unsupervised epochs (paper uses 60).")
    p.add_argument("--layer-sizes", type=str, default="784,1000,1000,1000,1000",
                   help="Comma-separated layer sizes including input "
                        "(paper uses 784,2000,2000,2000,2000).")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threshold", type=float, default=2.0,
                   help="Per-neuron mean-goodness threshold. Default 2.0 "
                        "(Hinton 2022 PyTorch reference convention).")
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-blur", type=int, default=6)
    p.add_argument("--n-train", type=int, default=0,
                   help="If >0, subsample the training set to N examples "
                        "(faster iteration during dev).")
    p.add_argument("--softmax-epochs", type=int, default=20)
    p.add_argument("--top-k", type=int, default=3,
                   help="How many top hidden layers feed the softmax head.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    assert layer_sizes[0] == 784, "first layer must be 784 (MNIST flat)"
    threshold = float(args.threshold)

    print(f"# ff-hybrid-mnist (Hinton 2022)")
    print(f"# layer_sizes={layer_sizes}  threshold={threshold:.1f}  "
          f"lr={args.lr}  batch={args.batch_size}  "
          f"epochs={args.n_epochs}  seed={args.seed}")
    for k, v in env_info().items():
        print(f"#   {k}: {v}")

    rng = np.random.default_rng(args.seed)
    print("loading MNIST...")
    mnist = load_mnist()
    print(f"  train: {mnist['train_images'].shape}  "
          f"test: {mnist['test_images'].shape}")

    if args.n_train > 0:
        idx = rng.permutation(mnist["train_images"].shape[0])[:args.n_train]
        mnist["train_images"] = mnist["train_images"][idx]
        mnist["train_labels"] = mnist["train_labels"][idx]
        print(f"  subsampled to {mnist['train_images'].shape[0]}")

    print("building FF MLP...")
    layers = build_ff_mlp(layer_sizes, rng)

    print(f"unsupervised FF training, {args.n_epochs} epochs...")
    t_train = time.time()
    history = train_unsupervised(
        layers,
        mnist["train_images"],
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        threshold=threshold,
        weight_decay=args.weight_decay,
        n_blur=args.n_blur,
        rng=rng,
        verbose=not args.quiet,
    )
    train_wall = time.time() - t_train

    print("\nfitting linear softmax on top of frozen FF features...")
    t_sm = time.time()
    sm = fit_softmax_on_top_layers(layers, mnist,
                                    top_k=args.top_k,
                                    n_epochs=args.softmax_epochs,
                                    rng=np.random.default_rng(args.seed + 1),
                                    verbose=not args.quiet)
    sm_wall = time.time() - t_sm

    final_test_acc = sm["history"]["test_acc"][-1]
    final_test_err = 1.0 - final_test_acc
    print(f"\nfinal test accuracy: {final_test_acc * 100:.2f}%  "
          f"(test error: {final_test_err * 100:.2f}%)")
    print(f"unsupervised wallclock: {train_wall:.1f} s")
    print(f"softmax wallclock:      {sm_wall:.1f} s")
    print(f"total wallclock:        {train_wall + sm_wall:.1f} s")
    print("paper reports 1.37% test error (MLP) / 1.16% (locally-connected).")

    return dict(layers=layers, history=history, softmax=sm,
                args=args, env=env_info(),
                final_test_err=final_test_err,
                train_wall=train_wall, sm_wall=sm_wall)


if __name__ == "__main__":
    main()
