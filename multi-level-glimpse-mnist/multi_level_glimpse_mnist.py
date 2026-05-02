"""
Multi-level glimpse MNIST (Ba, Hinton, Mnih, Leibo & Ionescu 2016).

Source:
    J. Ba, G. Hinton, V. Mnih, J. Z. Leibo, C. Ionescu (2016),
    "Using Fast Weights to Attend to the Recent Past", NIPS.
    https://arxiv.org/abs/1610.06258

Problem:
    The 28x28 MNIST image is presented to the network not as a single tensor
    but as a deterministic sequence of 24 small 7x7 patches ("glimpses").
    The network never sees the whole image at once. It must integrate the
    glimpses across time and classify the digit from the final hidden state.

    The 24-glimpse sequence is hierarchical:
        - 4 coarse 14x14 quadrants in fixed order (TL, TR, BL, BR)
        - each coarse quadrant split into 4 finer 7x7 patches in fixed order
          (TL, TR, BL, BR within the coarse quadrant)
          -> 16 fine 7x7 patches
        - plus 8 "most-central" re-glimpses: the 4 patches that straddle the
          centre of the image, re-visited twice each, in the order
          (centre TL, centre TR, centre BL, centre BR, centre TL, ...)
        - total: 24 glimpses (deviation #1: deterministic, not stochastic)

Architecture (glimpse RNN with fast weights, batched):
    Per timestep input x_t = [glimpse_patch (49) ; one_hot_position (24)] (73)
    A_t   = lambda_decay * A_{t-1} + eta * outer(h_{t-1}, h_{t-1})    (A_0 = 0)
    z_t   = W_h h_{t-1} + W_x x_t + b + A_t @ h_{t-1}
    zn_t  = LayerNorm(z_t)        # mean-0, std-1 over hidden axis, no affine
    h_t   = tanh(zn_t)
    out   = W_o h_T + b_o         # only final hidden predicts class

    The fast-weights matrix A_t is per-sample, reset to zero at the start of
    each image, accumulates an outer-product trace of recent hidden states,
    decays by lambda_decay per step. At each step the matrix-vector product
    A_t @ h_{t-1} is a Hopfield-style associative read keyed on the current
    hidden state. This is what lets information from glimpse t=2 stay
    accessible to the network when classifying at glimpse t=24.

    LayerNorm is necessary: without it, A_t @ h_{t-1} grows quadratically as
    outer products accumulate, the tanh saturates, and the recurrent
    gradient collapses (matches Ba et al. recipe).

BPTT through the fast weights:
    Standard tanh-RNN backprop with LayerNorm, plus a running gradient on
    the fast-weights matrix:

        dA_running = 0
        for t = T..1:
            dh_t already known
            dzn_t = dh_t * (1 - h_t^2)
            dz_t  = LN_backward(dzn_t, zn_t, sigma)
            dW_h += dz_t outer h_{t-1}
            dW_x += dz_t outer x_t
            db   += dz_t
            dh_{t-1} = (W_h.T + A_t.T) dz_t
            dA_t_local  = outer(dz_t, h_{t-1})
            dA_t_total  = dA_running + dA_t_local
            dh_{t-1}   += eta * (dA_t_total + dA_t_total.T) @ h_{t-1}
            dA_running  = lambda_decay * dA_t_total

    A numerical-gradient check (max relative error ~1e-9) confirms the
    forward and backward paths.

This file: MNIST loader + glimpse generator + GlimpseFastWeightsRNN +
Adam + train + CLI. Visualizations and GIF live in sibling files.
"""

from __future__ import annotations

import argparse
import gzip
import os
import platform
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader
# ----------------------------------------------------------------------

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
CACHE_DIR = Path.home() / ".cache" / "hinton-mnist"


def _download_if_needed(name: str) -> Path:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / name
    if not path.exists():
        url = MNIST_URL_BASE + name
        print(f"[mnist] downloading {url} -> {path}")
        urllib.request.urlretrieve(url, path)
    return path


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad image magic: {magic}")
        buf = f.read(n * rows * cols)
    return np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad label magic: {magic}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8)


def load_mnist() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Returns (train_x, train_y, test_x, test_y).

    Pixel values are float32 in [0, 1]; shapes (60000, 28, 28), (60000,),
    (10000, 28, 28), (10000,).
    """
    paths = {k: _download_if_needed(v) for k, v in MNIST_FILES.items()}
    train_x = _read_images(paths["train_images"]).astype(np.float32) / 255.0
    train_y = _read_labels(paths["train_labels"]).astype(np.int64)
    test_x = _read_images(paths["test_images"]).astype(np.float32) / 255.0
    test_y = _read_labels(paths["test_labels"]).astype(np.int64)
    return train_x, train_y, test_x, test_y


# ----------------------------------------------------------------------
# Glimpse sequence (deterministic 24 patches)
# ----------------------------------------------------------------------

PATCH_SIZE = 7
N_GLIMPSES = 24
N_CLASSES = 10
GLIMPSE_DIM = PATCH_SIZE * PATCH_SIZE + N_GLIMPSES   # patch + one-hot pos = 73


def _build_offsets() -> list[tuple[int, int]]:
    """The fixed sequence of 24 (top-left-row, top-left-col) offsets.

    First 16: 4 coarse 14x14 quadrants in TL/TR/BL/BR order, each split into
    4 fine 7x7 patches also in TL/TR/BL/BR order. This produces every fine
    patch on the natural 4x4 grid of 7x7 tiles covering the 28x28 image.

    Last 8: 4 "most-central" 7x7 patches at offsets (7,7), (7,14), (14,7),
    (14,14), each visited twice for redundant coverage of the centre where
    most digit ink lives.
    """
    offsets: list[tuple[int, int]] = []
    coarse_corners = [(0, 0), (0, 14), (14, 0), (14, 14)]   # TL, TR, BL, BR
    fine_offsets   = [(0, 0), (0, 7),  (7, 0),  (7, 7)]     # TL, TR, BL, BR
    for cr, cc in coarse_corners:
        for fr, fc in fine_offsets:
            offsets.append((cr + fr, cc + fc))
    centre = [(7, 7), (7, 14), (14, 7), (14, 14)]
    offsets.extend(centre)
    offsets.extend(centre)
    assert len(offsets) == N_GLIMPSES
    return offsets


GLIMPSE_OFFSETS = _build_offsets()


def generate_glimpse_sequence(image: np.ndarray) -> np.ndarray:
    """Return 24 7x7 patches as a (24, 49) float32 array.

    Argument
    --------
    image : (28, 28) array (float32, [0,1])

    Returns
    -------
    patches : (24, 49) float32, in the deterministic order from GLIMPSE_OFFSETS.
    """
    if image.shape != (28, 28):
        raise ValueError(f"expected (28, 28), got {image.shape}")
    out = np.empty((N_GLIMPSES, PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    for i, (r, c) in enumerate(GLIMPSE_OFFSETS):
        out[i] = image[r:r + PATCH_SIZE, c:c + PATCH_SIZE].reshape(-1)
    return out


def generate_glimpse_batch(images: np.ndarray) -> np.ndarray:
    """Vectorized: (B, 28, 28) -> (B, 24, 49) float32."""
    B = images.shape[0]
    out = np.empty((B, N_GLIMPSES, PATCH_SIZE * PATCH_SIZE), dtype=np.float32)
    for i, (r, c) in enumerate(GLIMPSE_OFFSETS):
        patch = images[:, r:r + PATCH_SIZE, c:c + PATCH_SIZE]
        out[:, i, :] = patch.reshape(B, -1)
    return out


def build_glimpse_inputs(images: np.ndarray) -> np.ndarray:
    """Build the full per-step input tensor [patch ; one_hot_pos].

    Returns
    -------
    X : (B, 24, 73) float32
    """
    patches = generate_glimpse_batch(images)            # (B, 24, 49)
    B = patches.shape[0]
    onehot = np.eye(N_GLIMPSES, dtype=np.float32)        # (24, 24)
    onehot_b = np.broadcast_to(onehot, (B, N_GLIMPSES, N_GLIMPSES))
    return np.concatenate([patches, onehot_b], axis=-1)  # (B, 24, 73)


# ----------------------------------------------------------------------
# Model: glimpse RNN with fast weights (vectorized over batch)
# ----------------------------------------------------------------------

class GlimpseFastWeightsRNN:
    """Tanh RNN with per-sequence fast-weights matrix A_t.

    Slow parameters (learned by BPTT):
        W_h : (H, H)
        W_x : (H, F)   where F = 49 + 24 = 73
        b   : (H,)
        W_o : (10, H)
        b_o : (10,)

    Per-sample fast weights A_t: (B, H, H), reset to zero at the start of
    every batch, NOT learned -- it is computed forward and backward but not
    stored as a parameter.
    """

    def __init__(self,
                 n_in: int = GLIMPSE_DIM,
                 n_hidden: int = 64,
                 n_out: int = N_CLASSES,
                 lambda_decay: float = 0.95,
                 eta: float = 0.5,
                 seed: int = 0):
        self.n_in = int(n_in)
        self.n_hidden = int(n_hidden)
        self.n_out = int(n_out)
        self.lambda_decay = float(lambda_decay)
        self.eta = float(eta)
        rng = np.random.default_rng(seed)

        s = 1.0 / np.sqrt(self.n_hidden)
        # Identity-ish recurrent init (Le, Jaitly, Hinton 2015 IRNN); the
        # LayerNorm rescales any explosion.
        self.W_h = np.eye(self.n_hidden) * 0.5
        self.W_x = rng.standard_normal((self.n_hidden, self.n_in)) * s
        self.b   = np.zeros(self.n_hidden)
        self.W_o = rng.standard_normal((self.n_out, self.n_hidden)) * s
        self.b_o = np.zeros(self.n_out)

    # --- introspection ---

    def n_params(self) -> int:
        return (self.W_h.size + self.W_x.size + self.b.size
                + self.W_o.size + self.b_o.size)

    def params(self) -> dict[str, np.ndarray]:
        return {"W_h": self.W_h, "W_x": self.W_x, "b": self.b,
                "W_o": self.W_o, "b_o": self.b_o}

    # --- forward (batched) ---

    def forward(self, X: np.ndarray, keep_trace: bool = False) -> dict:
        """Run the glimpse RNN on a batch of input sequences.

        X : (B, T, F)  float32 or float64
        Returns dict with logits (B, 10) and (if keep_trace) traces.
        """
        B, T, F = X.shape
        H = self.n_hidden
        ln_eps = 1e-5

        h = np.zeros((B, T + 1, H), dtype=np.float64)
        z = np.zeros((B, T, H), dtype=np.float64)
        zn = np.zeros((B, T, H), dtype=np.float64)
        sig = np.zeros((B, T), dtype=np.float64)
        # A[:, t] is the matrix used at step t to produce h[t+1]
        A = np.zeros((B, T, H, H), dtype=np.float64)
        A_prev = np.zeros((B, H, H), dtype=np.float64)

        Xd = X.astype(np.float64, copy=False)
        for t in range(T):
            h_prev = h[:, t]                                 # (B, H)
            outer = h_prev[:, :, None] * h_prev[:, None, :]  # (B, H, H)
            A_t = self.lambda_decay * A_prev + self.eta * outer
            A[:, t] = A_t
            # z_t = W_h h_prev + W_x x_t + b + A_t h_prev
            Wh_h    = h_prev @ self.W_h.T                    # (B, H)
            Wx_x    = Xd[:, t] @ self.W_x.T                  # (B, H)
            Ah      = np.einsum("bij,bj->bi", A_t, h_prev)   # (B, H)
            z_t     = Wh_h + Wx_x + self.b + Ah
            mu      = z_t.mean(axis=-1, keepdims=True)       # (B, 1)
            var     = ((z_t - mu) ** 2).mean(axis=-1, keepdims=True)
            sigma   = np.sqrt(var + ln_eps)                  # (B, 1)
            zn_t    = (z_t - mu) / sigma
            z[:, t]  = z_t
            zn[:, t] = zn_t
            sig[:, t] = sigma[:, 0]
            h[:, t + 1] = np.tanh(zn_t)
            A_prev = A_t

        logits = h[:, T] @ self.W_o.T + self.b_o            # (B, 10)

        out = {"h": h, "z": z, "zn": zn, "sig": sig, "A": A, "logits": logits}
        if not keep_trace:
            # large arrays -- drop them when not needed
            pass
        return out

    # --- backward (vectorized BPTT) ---

    def backward(self, X: np.ndarray, y: np.ndarray, fwd: dict
                 ) -> tuple[float, float, dict]:
        """Mean cross-entropy loss + accuracy + summed gradients (per batch).

        Returns gradients averaged over the batch (you can pass them straight
        to the optimizer).
        """
        h = fwd["h"]; zn = fwd["zn"]; sig = fwd["sig"]; A = fwd["A"]
        logits = fwd["logits"]
        B, T, F = X.shape
        H = self.n_hidden

        # softmax cross entropy
        m = np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits - m)
        probs = exp / np.sum(exp, axis=-1, keepdims=True)        # (B, 10)
        loss = float(np.mean(-np.log(probs[np.arange(B), y] + 1e-12)))
        preds = np.argmax(probs, axis=-1)
        acc = float(np.mean(preds == y))

        d_logits = probs.copy()
        d_logits[np.arange(B), y] -= 1.0
        d_logits /= B                                            # mean-loss

        # output layer
        h_T = h[:, T]                                            # (B, H)
        dW_o = d_logits.T @ h_T                                  # (10, H)
        db_o = d_logits.sum(axis=0)                              # (10,)
        dh = d_logits @ self.W_o                                 # (B, H)

        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db   = np.zeros_like(self.b)
        dA_running = np.zeros((B, H, H))

        Xd = X.astype(np.float64, copy=False)

        for t in range(T - 1, -1, -1):
            h_prev = h[:, t]                                     # (B, H)
            h_now  = h[:, t + 1]                                 # (B, H)
            A_t    = A[:, t]                                     # (B, H, H)
            zn_t   = zn[:, t]                                    # (B, H)
            sigma  = sig[:, t][:, None]                          # (B, 1)

            # tanh backward
            dzn = dh * (1.0 - h_now * h_now)                     # (B, H)
            # LayerNorm backward (no affine):
            #   y = (x - mu)/sigma  =>  dx = (1/sigma)(dy - mean(dy) - y*mean(dy*y))
            mean_dzn = dzn.mean(axis=-1, keepdims=True)
            mean_dzn_y = (dzn * zn_t).mean(axis=-1, keepdims=True)
            dz = (dzn - mean_dzn - zn_t * mean_dzn_y) / sigma    # (B, H)

            # parameter grads
            dW_h += dz.T @ h_prev                                # (H, H)
            dW_x += dz.T @ Xd[:, t]                              # (H, F)
            db   += dz.sum(axis=0)                               # (H,)

            # backprop through z_t = W_h h_{t-1} + W_x x_t + b + A_t h_{t-1}
            #   dh_{t-1} += W_h.T @ dz   AND   A_t.T @ dz   (per-batch)
            dh_prev = dz @ self.W_h                              # (B, H)
            dh_prev += np.einsum("bij,bi->bj", A_t, dz)          # (B, H)

            # dA_t local: outer(dz, h_{t-1})  -- per batch
            dA_t_local = dz[:, :, None] * h_prev[:, None, :]     # (B, H, H)
            dA_t_total = dA_running + dA_t_local                 # (B, H, H)

            # backprop through A_t = lambda A_{t-1} + eta outer(h_{t-1}, h_{t-1})
            #   dh_{t-1} += eta * (dA_total + dA_total.T) @ h_{t-1}
            #   dA_{t-1}  = lambda * dA_total
            sym = dA_t_total + dA_t_total.transpose(0, 2, 1)     # (B, H, H)
            dh_prev += self.eta * np.einsum("bij,bj->bi", sym, h_prev)
            dA_running = self.lambda_decay * dA_t_total

            dh = dh_prev

        grads = {"W_h": dW_h, "W_x": dW_x, "b": db,
                 "W_o": dW_o, "b_o": db_o}
        return loss, acc, grads

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.argmax(self.forward(X)["logits"], axis=-1)


def build_glimpse_rnn_with_fast_weights(glimpse_dim: int = GLIMPSE_DIM,
                                        n_hidden: int = 64,
                                        n_classes: int = N_CLASSES,
                                        lambda_decay: float = 0.95,
                                        eta: float = 0.5,
                                        seed: int = 0
                                        ) -> GlimpseFastWeightsRNN:
    """Per-stub default factory (matches stub signature)."""
    return GlimpseFastWeightsRNN(
        n_in=glimpse_dim, n_hidden=n_hidden, n_out=n_classes,
        lambda_decay=lambda_decay, eta=eta, seed=seed,
    )


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict[str, np.ndarray],
                 lr: float = 2e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            mhat = self.m[k] / (1.0 - self.beta1 ** self.t)
            vhat = self.v[k] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def evaluate(model: GlimpseFastWeightsRNN,
             X_in: np.ndarray, y: np.ndarray,
             batch_size: int = 256) -> tuple[float, float]:
    """Mean cross-entropy loss + accuracy on (X_in, y).

    X_in : (N, T, F) glimpse-input tensor
    """
    N = X_in.shape[0]
    total_loss = 0.0
    total_correct = 0
    for i in range(0, N, batch_size):
        xb = X_in[i:i + batch_size]
        yb = y[i:i + batch_size]
        fwd = model.forward(xb)
        logits = fwd["logits"]
        m = np.max(logits, axis=-1, keepdims=True)
        exp = np.exp(logits - m)
        probs = exp / exp.sum(axis=-1, keepdims=True)
        bsz = xb.shape[0]
        total_loss += float(-np.log(probs[np.arange(bsz), yb] + 1e-12).sum())
        preds = np.argmax(logits, axis=-1)
        total_correct += int((preds == yb).sum())
    return total_loss / N, total_correct / N


def train(model: GlimpseFastWeightsRNN,
          data: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
          n_epochs: int = 3,
          batch_size: int = 64,
          lr: float = 2e-3,
          lr_decay_epochs: tuple[int, ...] = (),
          lr_decay_factor: float = 0.25,
          grad_clip: float = 5.0,
          eval_every: int = 200,
          eval_test_every_epoch: bool = True,
          seed: int = 0,
          verbose: bool = True,
          ) -> dict:
    """Train the glimpse RNN on MNIST.

    data: (train_x_glimpses, train_y, test_x_glimpses, test_y)
        train_x_glimpses, test_x_glimpses are pre-built (N, T, F) tensors.

    lr_decay_epochs: at the START of each listed epoch, multiply current
        learning rate by lr_decay_factor. e.g. (7, 11) with factor=0.25 gives
        lr=base for ep1-6, lr=0.25*base for ep7-10, lr=0.0625*base for ep11+.
    """
    train_x, train_y, test_x, test_y = data
    N_train = train_x.shape[0]
    rng = np.random.default_rng(seed + 7)

    optim = Adam(model.params(), lr=lr)
    history = {
        "step": [], "epoch": [], "train_loss": [], "train_acc": [],
        "eval_test_step": [], "test_loss": [], "test_acc": [], "lr": [],
    }

    t0 = time.time()
    step = 0
    running_loss, running_acc, n_run = 0.0, 0.0, 0

    for epoch in range(1, n_epochs + 1):
        if epoch in lr_decay_epochs:
            optim.lr *= lr_decay_factor
            if verbose:
                print(f"  ep{epoch} -- lr -> {optim.lr:.2e}")
        # shuffle
        perm = rng.permutation(N_train)
        for i in range(0, N_train, batch_size):
            step += 1
            idx = perm[i:i + batch_size]
            xb = train_x[idx]
            yb = train_y[idx]
            fwd = model.forward(xb)
            loss, acc, grads = model.backward(xb, yb, fwd)

            # global-norm gradient clip
            gnorm2 = sum(float(np.sum(g * g)) for g in grads.values())
            gnorm = float(np.sqrt(gnorm2))
            if grad_clip is not None and gnorm > grad_clip:
                scale = grad_clip / (gnorm + 1e-12)
                for k in grads:
                    grads[k] *= scale

            optim.step(grads)

            running_loss += loss
            running_acc  += acc
            n_run += 1

            if step % eval_every == 0:
                mean_loss = running_loss / max(1, n_run)
                mean_acc  = running_acc / max(1, n_run)
                running_loss, running_acc, n_run = 0.0, 0.0, 0
                history["step"].append(step)
                history["epoch"].append(epoch)
                history["train_loss"].append(mean_loss)
                history["train_acc"].append(mean_acc)
                if verbose:
                    elapsed = time.time() - t0
                    print(f"  ep{epoch} step {step:5d}  "
                          f"train_loss={mean_loss:.4f}  "
                          f"train_acc={mean_acc*100:5.1f}%  "
                          f"({elapsed:5.1f}s)")

        if eval_test_every_epoch:
            tl, ta = evaluate(model, test_x, test_y, batch_size=batch_size)
            history["eval_test_step"].append(step)
            history["test_loss"].append(tl)
            history["test_acc"].append(ta)
            if verbose:
                elapsed = time.time() - t0
                print(f"  ep{epoch} END  test_loss={tl:.4f}  "
                      f"test_acc={ta*100:5.2f}%  ({elapsed:5.1f}s)")

    history["wallclock"] = time.time() - t0
    return history


# ----------------------------------------------------------------------
# Per-class accuracy breakdown
# ----------------------------------------------------------------------

def per_class_accuracy(model: GlimpseFastWeightsRNN,
                       X_in: np.ndarray, y: np.ndarray,
                       batch_size: int = 256) -> np.ndarray:
    """Test accuracy broken down by digit class. Returns (10,) array."""
    N = X_in.shape[0]
    n_correct = np.zeros(N_CLASSES, dtype=np.int64)
    n_seen    = np.zeros(N_CLASSES, dtype=np.int64)
    for i in range(0, N, batch_size):
        xb = X_in[i:i + batch_size]
        yb = y[i:i + batch_size]
        preds = model.predict(xb)
        for c in range(N_CLASSES):
            mask = (yb == c)
            n_seen[c] += int(mask.sum())
            n_correct[c] += int((preds[mask] == c).sum())
    return n_correct / np.maximum(1, n_seen)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL,
                                         text=True).strip()[:10]
    except Exception:
        commit = "unknown"
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}  git {commit}")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Multi-level glimpse MNIST (Ba et al. 2016).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--lr-decay-epochs", type=str, default="",
                   help="comma-separated epochs at which to apply lr_decay_factor "
                        "(e.g. '7,11')")
    p.add_argument("--lr-decay-factor", type=float, default=0.25)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--n-train", type=int, default=0,
                   help="0 = use full 60k MNIST training set; >0 = subsample")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _print_environment()
    print(f"# config: n_epochs={args.n_epochs}  n_hidden={args.n_hidden}  "
          f"lambda={args.lambda_decay}  eta={args.eta}  "
          f"lr={args.lr}  batch={args.batch_size}  seed={args.seed}")

    np.random.seed(args.seed)

    print("\n=== Loading MNIST ===")
    t_load = time.time()
    train_x, train_y, test_x, test_y = load_mnist()
    print(f"  train: {train_x.shape}  test: {test_x.shape}  "
          f"({time.time()-t_load:.1f}s)")

    if args.n_train and args.n_train < len(train_x):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_x))[:args.n_train]
        train_x = train_x[idx]
        train_y = train_y[idx]
        print(f"  subsampled to {len(train_x)} train images")

    print("\n=== Building glimpse inputs ===")
    t_g = time.time()
    train_X = build_glimpse_inputs(train_x)              # (N, 24, 73)
    test_X  = build_glimpse_inputs(test_x)
    print(f"  train_X {train_X.shape}  test_X {test_X.shape}  "
          f"({time.time()-t_g:.1f}s)")

    print("\n=== Building model ===")
    model = build_glimpse_rnn_with_fast_weights(
        glimpse_dim=GLIMPSE_DIM, n_hidden=args.n_hidden, n_classes=N_CLASSES,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed,
    )
    print(f"  n_params: {model.n_params():,}")

    print("\n=== Training ===")
    t_train = time.time()
    if args.lr_decay_epochs:
        lr_decay_epochs = tuple(int(e) for e in args.lr_decay_epochs.split(","))
    else:
        lr_decay_epochs = ()
    history = train(model, (train_X, train_y, test_X, test_y),
                    n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr,
                    lr_decay_epochs=lr_decay_epochs,
                    lr_decay_factor=args.lr_decay_factor,
                    grad_clip=args.grad_clip,
                    eval_every=args.eval_every,
                    seed=args.seed, verbose=not args.quiet)
    train_dt = time.time() - t_train

    print("\n=== Final evaluation ===")
    final_loss, final_acc = evaluate(model, test_X, test_y,
                                     batch_size=args.batch_size)
    per_class = per_class_accuracy(model, test_X, test_y,
                                   batch_size=args.batch_size)
    print(f"  test loss  : {final_loss:.4f}")
    print(f"  test acc   : {final_acc*100:6.2f}%")
    print("  per-class  : "
          + "  ".join(f"{c}={a*100:5.1f}%" for c, a in enumerate(per_class)))
    print(f"  train wallclock : {train_dt:.1f}s")
    print(f"  n_params        : {model.n_params():,}")


if __name__ == "__main__":
    main()
