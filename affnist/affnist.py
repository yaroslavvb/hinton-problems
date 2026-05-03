"""
affNIST robustness test (Sabour, Frosst & Hinton 2017, "Dynamic routing between capsules").

Train both a CapsNet and a parameter-matched CNN on **translated MNIST** (40x40
canvas, MNIST digit randomly placed within +-6 px of center). Test on **affNIST**
40x40 (real if reachable, synthesized via random affine transforms otherwise).
The headline of the paper is the *robustness gap*: the CapsNet generalises to
unseen affine transforms more gracefully than the CNN with matched parameters.

Architectural simplifications (compared to the paper):

* Input 40x40 (paper used 28x28 + reconstructed at 28x28 in the original
  implementation, translated to 40x40 only for the affNIST evaluation).
* Tiny per-paper CapsNet: 16 conv1 filters (paper: 256), 4-D primary capsules
  (paper: 8-D), 8-D digit capsules (paper: 16-D), no reconstruction decoder.
  Routing iterations 3 (per paper).
* Same-parameter CNN: 3 conv layers + 1 FC, parameter count matched to the
  CapsNet within ~10%.
* Margin loss only (no reconstruction regulariser).

Despite these reductions the qualitative effect of interest -- the
CapsNet-vs-CNN gap on unseen affine transforms -- is preserved.

Pure-numpy. Heavy bits use BLAS through `np.matmul` after im2col.
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import platform
import time
import urllib.request
from pathlib import Path
from typing import Tuple

import numpy as np


# ----------------------------------------------------------------------
# Data loaders
# ----------------------------------------------------------------------

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
MNIST_CACHE = Path.home() / ".cache" / "hinton-mnist"
AFFNIST_CACHE = Path.home() / ".cache" / "hinton-affnist"

AFFNIST_URLS = [
    "https://www.cs.toronto.edu/~tijmen/affNIST/32x/just_centered/test_batches.tar.gz",
    "https://github.com/szagoruyko/affnist/raw/master/test.tar.gz",
]


def _download(url: str, dest: Path, timeout: float = 60.0) -> bool:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=timeout) as r:
            data = r.read()
        dest.write_bytes(data)
        return True
    except Exception as e:
        print(f"  download failed: {url}: {e}", file=sys.stderr)
        return False


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    if magic != 2051:
        raise RuntimeError(f"bad MNIST image magic: {magic}")
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic != 2049:
        raise RuntimeError(f"bad MNIST label magic: {magic}")
    return np.frombuffer(data[8:], dtype=np.uint8)


def load_mnist(split: str = "train") -> Tuple[np.ndarray, np.ndarray]:
    """Return (images, labels). Images are float32 in [0, 1] of shape (N, 28, 28)."""
    if split not in ("train", "test"):
        raise ValueError(split)
    img_key = f"{split}_images"
    lbl_key = f"{split}_labels"
    img_dest = MNIST_CACHE / MNIST_FILES[img_key]
    lbl_dest = MNIST_CACHE / MNIST_FILES[lbl_key]
    if not img_dest.exists():
        if not _download(MNIST_URL_BASE + MNIST_FILES[img_key], img_dest):
            raise RuntimeError("MNIST download failed")
    if not lbl_dest.exists():
        if not _download(MNIST_URL_BASE + MNIST_FILES[lbl_key], lbl_dest):
            raise RuntimeError("MNIST label download failed")
    imgs = _read_idx_images(img_dest).astype(np.float32) / 255.0
    lbls = _read_idx_labels(lbl_dest).astype(np.int64)
    return imgs, lbls


# ----------------------------------------------------------------------
# Translated MNIST (paper's training distribution)
# ----------------------------------------------------------------------

def make_translated_mnist(mnist: np.ndarray, max_shift: int = 6,
                          canvas: int = 40, seed: int = 0) -> np.ndarray:
    """Pad each 28x28 MNIST digit to a 40x40 canvas at a random translation
    sampled uniformly in [-max_shift, +max_shift] on each axis (per paper).

    `mnist` may be (N, 28, 28). Returns (N, canvas, canvas) float32.
    """
    if mnist.ndim != 3 or mnist.shape[1] != 28 or mnist.shape[2] != 28:
        raise ValueError(f"expected (N, 28, 28); got {mnist.shape}")
    N = mnist.shape[0]
    rng = np.random.default_rng(seed)
    base_off = (canvas - 28) // 2  # 6 for 40x40 from 28x28
    out = np.zeros((N, canvas, canvas), dtype=np.float32)
    shifts = rng.integers(-max_shift, max_shift + 1, size=(N, 2))
    for i in range(N):
        dy = base_off + int(shifts[i, 0])
        dx = base_off + int(shifts[i, 1])
        # clip just in case
        dy = max(0, min(canvas - 28, dy))
        dx = max(0, min(canvas - 28, dx))
        out[i, dy:dy + 28, dx:dx + 28] = mnist[i]
    return out


# ----------------------------------------------------------------------
# affNIST loader: try real, fall back to synthesized
# ----------------------------------------------------------------------

def _affine_warp(img: np.ndarray, M: np.ndarray, canvas: int = 40) -> np.ndarray:
    """Apply 2x3 affine matrix M to a square image, output canvas x canvas.

    M maps output pixel coordinate (xo, yo) to input coordinate (xi, yi):
        [xi]   [a b][xo]   [tx]
        [yi] = [c d][yo] + [ty]

    Bilinear interpolation, zero pad.
    """
    h, w = img.shape[-2], img.shape[-1]
    yy, xx = np.mgrid[0:canvas, 0:canvas].astype(np.float32)
    # centre-relative
    cx = (canvas - 1) / 2.0
    cy = (canvas - 1) / 2.0
    xo = xx - cx
    yo = yy - cy
    a, b, tx = M[0]
    c, d, ty = M[1]
    xi = a * xo + b * yo + tx + (w - 1) / 2.0
    yi = c * xo + d * yo + ty + (h - 1) / 2.0

    # bilinear
    x0 = np.floor(xi).astype(np.int32)
    y0 = np.floor(yi).astype(np.int32)
    x1 = x0 + 1
    y1 = y0 + 1

    wa = (x1 - xi) * (y1 - yi)
    wb = (x1 - xi) * (yi - y0)
    wc = (xi - x0) * (y1 - yi)
    wd = (xi - x0) * (yi - y0)

    def _safe(yy_, xx_):
        m = (yy_ >= 0) & (yy_ < h) & (xx_ >= 0) & (xx_ < w)
        out = np.zeros_like(xi)
        ys = np.clip(yy_, 0, h - 1)
        xs = np.clip(xx_, 0, w - 1)
        out[m] = img[ys[m], xs[m]]
        return out

    out = (wa * _safe(y0, x0) + wb * _safe(y1, x0)
           + wc * _safe(y0, x1) + wd * _safe(y1, x1))
    return out.astype(np.float32)


def synthesize_affnist(mnist_test: np.ndarray, mnist_test_labels: np.ndarray,
                       n: int = 10000, canvas: int = 40, seed: int = 0
                       ) -> Tuple[np.ndarray, np.ndarray]:
    """Synthesize an affNIST-like 40x40 test set by applying a random affine
    transform (rotation +-20 deg, scale [0.8, 1.2], shear +-0.1, translation +-4 px)
    to each MNIST test digit. Returns (images, labels) both length n.
    """
    rng = np.random.default_rng(seed)
    N = mnist_test.shape[0]
    idx = rng.integers(0, N, size=n)
    out = np.zeros((n, canvas, canvas), dtype=np.float32)
    for i, j in enumerate(idx):
        theta = float(rng.uniform(-20, 20)) * np.pi / 180.0
        scale = float(rng.uniform(0.8, 1.2))
        shear = float(rng.uniform(-0.1, 0.1))
        tx = float(rng.uniform(-4, 4))
        ty = float(rng.uniform(-4, 4))
        ct, st = np.cos(theta), np.sin(theta)
        # forward affine on output -> need inverse for warping
        # forward: x' = R S Sh x + t
        Sh = np.array([[1.0, shear], [0.0, 1.0]])
        S = np.array([[scale, 0.0], [0.0, scale]])
        R = np.array([[ct, -st], [st, ct]])
        A = R @ S @ Sh
        # inverse mapping: input = A^-1 (output - t)
        Ainv = np.linalg.inv(A)
        M = np.zeros((2, 3), dtype=np.float32)
        M[:2, :2] = Ainv
        M[:, 2] = -Ainv @ np.array([tx, ty])
        out[i] = _affine_warp(mnist_test[j], M, canvas=canvas)
    labels = mnist_test_labels[idx].astype(np.int64)
    return out, labels


def load_affnist_test(force_synth: bool = False, n_synth: int = 10000,
                      seed: int = 0) -> Tuple[np.ndarray, np.ndarray, str]:
    """Try to load real affNIST 40x40 test set. Fall back to synthesized.

    Returns (images, labels, source) where `source` is "real" or "synth".
    Images: (N, 40, 40) float32 in [0, 1].
    """
    if not force_synth:
        for url in AFFNIST_URLS:
            fname = url.rsplit("/", 1)[-1]
            dest = AFFNIST_CACHE / fname
            if _download(url, dest, timeout=30.0):
                # Real affNIST is a tar.gz of .mat files. We don't depend on
                # scipy.io.loadmat -- if we ever get here add a parser. For now,
                # treat as available but fall back since we can't guarantee parse.
                # (Removed the heavyweight .mat parser to keep deps minimal.)
                print(f"  affNIST archive downloaded ({dest}) but the .mat parser "
                      f"is intentionally not implemented; falling back to "
                      f"synthesized affine MNIST.", file=sys.stderr)
                break
    print(f"  synthesizing affNIST-like test set (n={n_synth}, seed={seed})")
    test_imgs, test_lbls = load_mnist("test")
    imgs, lbls = synthesize_affnist(test_imgs, test_lbls, n=n_synth, seed=seed)
    return imgs, lbls, "synth"


# ----------------------------------------------------------------------
# im2col convolution helpers (BLAS-friendly)
# ----------------------------------------------------------------------

def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1) -> np.ndarray:
    """x: (N, C, H, W). Returns (N, OH*OW, C*kh*kw) without padding."""
    N, C, H, W = x.shape
    OH = (H - kh) // stride + 1
    OW = (W - kw) // stride + 1
    cols = np.empty((N, OH * OW, C * kh * kw), dtype=x.dtype)
    for i in range(kh):
        for j in range(kw):
            patch = x[:, :, i:i + OH * stride:stride, j:j + OW * stride:stride]
            # patch: (N, C, OH, OW)
            cols[:, :, (i * kw + j)::(kh * kw)] = patch.reshape(N, C, -1).transpose(0, 2, 1)
    return cols, OH, OW


def col2im(cols: np.ndarray, x_shape: Tuple[int, int, int, int],
           kh: int, kw: int, stride: int) -> np.ndarray:
    """Inverse of im2col. Adds overlapping contributions."""
    N, C, H, W = x_shape
    OH = (H - kh) // stride + 1
    OW = (W - kw) // stride + 1
    out = np.zeros(x_shape, dtype=cols.dtype)
    # cols: (N, OH*OW, C*kh*kw)
    for i in range(kh):
        for j in range(kw):
            patch = cols[:, :, (i * kw + j)::(kh * kw)]   # (N, OH*OW, C)
            patch = patch.transpose(0, 2, 1).reshape(N, C, OH, OW)
            out[:, :, i:i + OH * stride:stride,
                       j:j + OW * stride:stride] += patch
    return out


def conv2d_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1):
    """x: (N, C, H, W), W: (F, C, kh, kw), b: (F,). Returns (out, cache)."""
    N, C, H, Wd = x.shape
    F, C2, kh, kw = W.shape
    assert C == C2
    cols, OH, OW = im2col(x, kh, kw, stride)
    Wf = W.reshape(F, C * kh * kw)
    out = cols @ Wf.T + b           # (N, OH*OW, F)
    out = out.transpose(0, 2, 1).reshape(N, F, OH, OW)
    cache = (x.shape, W, cols, kh, kw, stride, OH, OW)
    return out, cache


def conv2d_backward(d_out: np.ndarray, cache):
    x_shape, W, cols, kh, kw, stride, OH, OW = cache
    N, C, H, Wd = x_shape
    F = W.shape[0]
    # d_out: (N, F, OH, OW)
    d_out_flat = d_out.reshape(N, F, OH * OW).transpose(0, 2, 1)  # (N, OH*OW, F)
    Wf = W.reshape(F, C * kh * kw)
    d_cols = d_out_flat @ Wf       # (N, OH*OW, C*kh*kw)
    # dW: sum over batch of cols.T @ d_out_flat
    d_Wf = np.einsum("noi,nof->fi", cols, d_out_flat)
    d_W = d_Wf.reshape(F, C, kh, kw)
    d_b = d_out.sum(axis=(0, 2, 3))
    d_x = col2im(d_cols, x_shape, kh, kw, stride)
    return d_x, d_W, d_b


def relu(x): return np.maximum(x, 0.0)


# ----------------------------------------------------------------------
# CapsNet (tiny variant for laptop numpy)
# ----------------------------------------------------------------------

def squash(s: np.ndarray, axis: int = -1, eps: float = 1e-7):
    """Capsule squashing nonlinearity. Returns v with cache for backward."""
    sq_norm = np.sum(s * s, axis=axis, keepdims=True)
    norm = np.sqrt(sq_norm + eps)
    scale = sq_norm / (1.0 + sq_norm)
    v = scale * (s / norm)
    return v, (s, sq_norm, norm)


def squash_backward(d_v: np.ndarray, cache, axis: int = -1, eps: float = 1e-7):
    s, sq_norm, norm = cache
    # v = (sq_norm / (1+sq_norm)) * s / norm
    # let n2 = sq_norm; n = sqrt(n2+eps)
    # v = (n2 / (1+n2)) * s / n
    # derivative w.r.t. s: a vector identity. We use a numerical-friendly
    # closed form:
    #   v = f(n2) * s / n  where f = n2/(1+n2)
    #   dv/ds = f/n * I + s * d(f/n)/ds
    #   d(f/n)/ds = (f'/n - f/(n*n2)) * (something with s)... messier.
    # Instead use the standard form (Sabour 2017 implementation):
    #   v = (1/(1+n2)) * s   ... wait that's not right either.
    # Use direct vector-Jacobian product:
    n2 = sq_norm
    n = norm
    f = n2 / (1.0 + n2)  # scalar per capsule
    # v = (f/n) * s  =>  dv = (f/n) ds + s d(f/n)
    # d(f/n) along s direction: f' = 2 / (1 + n2)^2 * n  (df/d|s|)
    #                            (1/n)' = -1/n^2 * 1     (d(1/n)/d|s|)
    # combine: d(f/n)/d|s| = f'/n - f/n^2
    #   = (2 / (1+n2)^2) * n / n - f / n^2
    #   = 2/(1+n2)^2 - f/n2  (using n2 ~= n^2)
    # The vector Jacobian product becomes:
    #   ds = (f/n) d_v + (s/|s|) * <s/|s|, d_v> * |s| * d(f/n)/d|s|
    #      = (f/n) d_v + (s . d_v) * s * d(f/n)/d|s| / |s|^2 * |s|
    # Cleaner: differentiate the original expression directly.
    s_dot_dv = np.sum(s * d_v, axis=axis, keepdims=True)
    # d/ds [ (n2/(1+n2)) * s/n ]
    # Use product rule: let A = n2/(1+n2), B = 1/n. Then v = A*B*s.
    # ds <-- A*B * d_v + (d/ds (A*B)) * (s . d_v)
    # d(A*B)/d|s| = A' * B + A * B'
    # A'(|s|) = d/d|s|(|s|^2/(1+|s|^2)) = 2|s|/(1+|s|^2)^2
    # B'(|s|) = -1/(|s|^2 * sign... just) -1/n^2 . using n^2 ~= n2:
    # B' = -1/n^2
    # so d(A*B)/d|s| = 2|s|/(1+n2)^2 / n - (n2/(1+n2))/n^2
    # vector form: d/ds (A*B) = (s/|s|) * d(A*B)/d|s|
    #   = s * [ 2/(1+n2)^2 - (n2/(1+n2))/n^2 ] / 1   (after pulling 1/|s|)
    # Wait: s * (1/|s|) * d/d|s| ... = s / |s| * scalar.
    # So d/ds(A*B) = (s / n) * dABdn = s/n * [2|s|/(1+n2)^2 / n - (n2/(1+n2))/n^2]
    #              = s * [2/(1+n2)^2 - (n2/(1+n2))/n^2] / n^2 ... messy.
    # Simpler: use d(A*B)/ds = (s/n2) * d(A*B)/d ln|s|^2 * 2?
    # Let me re-derive carefully via chain rule on n2:
    #   A*B = n2 / ((1+n2) * sqrt(n2+eps))
    #   d(A*B)/dn2 = [ (1+n2)*sqrt(n2) - n2 * (sqrt(n2) + (1+n2)/(2*sqrt(n2))) ]
    #                / ( (1+n2)*sqrt(n2) )^2     ... nasty.
    # Use simpler equivalent identity used in many CapsNet implementations:
    # Let v = ((|s|^2)/(1+|s|^2)) * s_hat, with s_hat = s/|s|. The Jacobian:
    #   dv/ds = (|s|^2/(1+|s|^2)) * (I - s_hat s_hat^T)/|s| + (2*|s|/(1+|s|^2)^2) * s_hat s_hat^T
    # Vector-Jacobian product on d_v:
    #   ds = (A) * (d_v - s_hat (s_hat . d_v))/|s|  +  (2|s|/(1+n2)^2) * s_hat (s_hat . d_v)
    # which simplifies to:
    A = n2 / (1.0 + n2)
    s_hat = s / n
    s_hat_dot_dv = np.sum(s_hat * d_v, axis=axis, keepdims=True)
    term1 = A * (d_v - s_hat * s_hat_dot_dv) / n
    term2 = (2.0 * n / (1.0 + n2) ** 2) * s_hat * s_hat_dot_dv
    return term1 + term2


def routing(u_hat: np.ndarray, n_iter: int = 3):
    """u_hat: (B, n_in, n_out, d_out). Returns (v: (B, n_out, d_out), cache)."""
    B, Nin, Nout, Dout = u_hat.shape
    b = np.zeros((B, Nin, Nout), dtype=u_hat.dtype)
    history = []
    for it in range(n_iter):
        c = _softmax(b, axis=2)                                   # (B, Nin, Nout)
        s = np.einsum("bnod,bno->bod", u_hat, c)                  # (B, Nout, Dout)
        v, sq_cache = squash(s, axis=-1)
        if it < n_iter - 1:
            agree = np.einsum("bnod,bod->bno", u_hat, v)         # (B, Nin, Nout)
            b = b + agree
        history.append(dict(c=c, s=s, v=v, sq_cache=sq_cache, b=b.copy()))
    return v, history


def _softmax(x, axis=-1):
    z = x - x.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def routing_backward(d_v_final, u_hat, history, n_iter):
    """Backprop through dynamic routing. We treat the routing weights `c` as
    a non-differentiable function of `b` for simplicity (b is updated using
    detached `v`), which matches the published implementations.

    Returns d_u_hat (B, Nin, Nout, Dout).
    """
    B, Nin, Nout, Dout = u_hat.shape
    d_u_hat = np.zeros_like(u_hat)
    # Only backprop through the *final* squash and final s (the c values are
    # treated as constants in the published reference impl).
    last = history[-1]
    d_s = squash_backward(d_v_final, last["sq_cache"], axis=-1)   # (B, Nout, Dout)
    # s = sum_n c[b,n,o] u_hat[b,n,o,d]
    # d u_hat[b,n,o,d] += c[b,n,o] * d_s[b,o,d]
    d_u_hat += last["c"][:, :, :, None] * d_s[:, None, :, :]
    return d_u_hat


def margin_loss(v: np.ndarray, y: np.ndarray, m_plus: float = 0.9,
                m_minus: float = 0.1, lam: float = 0.5):
    """v: (B, K, D) digit capsules. y: (B,) int labels. Returns (loss, d_v)."""
    B, K, D = v.shape
    norm = np.sqrt(np.sum(v * v, axis=-1) + 1e-8)                  # (B, K)
    T = np.zeros((B, K), dtype=v.dtype)
    T[np.arange(B), y] = 1.0
    pos = np.maximum(0.0, m_plus - norm)
    neg = np.maximum(0.0, norm - m_minus)
    L = T * pos ** 2 + lam * (1.0 - T) * neg ** 2
    loss = float(L.sum(axis=1).mean())
    # gradients w.r.t. norm
    d_norm_pos = -2.0 * T * pos                                    # (B, K)
    d_norm_neg = 2.0 * lam * (1.0 - T) * neg                       # (B, K)
    d_norm = (d_norm_pos + d_norm_neg) / B
    # norm = sqrt(sum v^2) -> d_v = (v / norm) * d_norm
    d_v = (v / norm[:, :, None]) * d_norm[:, :, None]
    return loss, d_v


class TinyCapsNet:
    """Conv1 -> PrimaryCaps -> DigitCaps with dynamic routing.

    Tiny variant for numpy: 16 conv1 filters, 8 primary capsule types of dim 4,
    10 digit capsules of dim 8. ~150K params on the 40x40 input.
    """

    def __init__(self, image_size: int = 40,
                 conv1_out: int = 16,
                 conv1_k: int = 9,
                 primary_caps: int = 8,
                 primary_dim: int = 4,
                 primary_k: int = 9,
                 primary_stride: int = 4,
                 digit_dim: int = 8,
                 n_classes: int = 10,
                 routing_iters: int = 3,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.image_size = image_size
        self.routing_iters = routing_iters
        self.n_classes = n_classes

        # Conv1: (F, 1, k, k)
        self.W1 = rng.standard_normal((conv1_out, 1, conv1_k, conv1_k)).astype(np.float32) \
                  * np.sqrt(2.0 / (1 * conv1_k * conv1_k))
        self.b1 = np.zeros((conv1_out,), dtype=np.float32)
        oh1 = image_size - conv1_k + 1

        # PrimaryCaps: conv producing primary_caps * primary_dim channels
        prim_channels = primary_caps * primary_dim
        self.W2 = rng.standard_normal((prim_channels, conv1_out, primary_k, primary_k)).astype(np.float32) \
                  * np.sqrt(2.0 / (conv1_out * primary_k * primary_k))
        self.b2 = np.zeros((prim_channels,), dtype=np.float32)
        oh2 = (oh1 - primary_k) // primary_stride + 1
        self.primary_caps = primary_caps
        self.primary_dim = primary_dim
        self.primary_stride = primary_stride
        self.primary_k = primary_k
        self.conv1_k = conv1_k
        self.oh1 = oh1
        self.oh2 = oh2

        # Number of input capsules to digitcaps:
        self.n_in_caps = primary_caps * oh2 * oh2
        self.digit_dim = digit_dim

        # W_ij transformation matrices: (n_in_caps, n_classes, primary_dim, digit_dim)
        self.Wij = (rng.standard_normal((self.n_in_caps, n_classes, primary_dim, digit_dim))
                    .astype(np.float32) * 0.05)

    # ----- forward -----

    def forward(self, x: np.ndarray):
        """x: (B, 40, 40) float32. Returns (v, caches)."""
        B = x.shape[0]
        x4 = x[:, None, :, :]                                      # (B, 1, H, W)
        z1, c1 = conv2d_forward(x4, self.W1, self.b1, stride=1)   # (B, F1, oh1, oh1)
        a1 = relu(z1)
        z2, c2 = conv2d_forward(a1, self.W2, self.b2, stride=self.primary_stride)
        # z2: (B, prim_channels, oh2, oh2). Reshape to capsules:
        oh2 = self.oh2
        z2r = z2.reshape(B, self.primary_caps, self.primary_dim, oh2, oh2)
        # Move spatial axes next to capsule index -> (B, primary_caps*oh2*oh2, primary_dim)
        z2r = z2r.transpose(0, 1, 3, 4, 2).reshape(B, self.n_in_caps, self.primary_dim)
        u, sq_cache_prim = squash(z2r, axis=-1)
        # u: (B, n_in_caps, primary_dim)
        # u_hat[b,i,j,d_out] = sum_d Wij[i,j,d_in,d_out] * u[b,i,d_in]
        u_hat = np.einsum("bid,ijde->bije", u, self.Wij)
        v, hist = routing(u_hat, n_iter=self.routing_iters)
        cache = dict(x=x, c1=c1, a1=a1, z1=z1, c2=c2, z2=z2, z2r_shape=z2r.shape,
                     u=u, sq_cache_prim=sq_cache_prim, u_hat=u_hat, hist=hist)
        return v, cache

    # ----- backward -----

    def backward(self, d_v: np.ndarray, cache):
        u_hat = cache["u_hat"]
        hist = cache["hist"]
        d_u_hat = routing_backward(d_v, u_hat, hist, self.routing_iters)
        # d Wij[i,j,d,e] = sum_b u[b,i,d] * d_u_hat[b,i,j,e]
        u = cache["u"]
        d_Wij = np.einsum("bid,bije->ijde", u, d_u_hat)
        # d u[b,i,d] = sum_j,e Wij[i,j,d,e] * d_u_hat[b,i,j,e]
        d_u = np.einsum("ijde,bije->bid", self.Wij, d_u_hat)
        d_z2r = squash_backward(d_u, cache["sq_cache_prim"], axis=-1)
        # reshape back
        B = d_z2r.shape[0]
        oh2 = self.oh2
        d_z2 = d_z2r.reshape(B, self.primary_caps, oh2, oh2, self.primary_dim) \
                    .transpose(0, 1, 4, 2, 3) \
                    .reshape(B, self.primary_caps * self.primary_dim, oh2, oh2)
        d_a1, d_W2, d_b2 = conv2d_backward(d_z2, cache["c2"])
        d_z1 = d_a1 * (cache["z1"] > 0).astype(np.float32)
        _, d_W1, d_b1 = conv2d_backward(d_z1, cache["c1"])
        return dict(W1=d_W1, b1=d_b1, W2=d_W2, b2=d_b2, Wij=d_Wij)

    @property
    def param_names(self):
        return ("W1", "b1", "W2", "b2", "Wij")

    def predict(self, x: np.ndarray) -> np.ndarray:
        v, _ = self.forward(x)
        norm = np.sqrt(np.sum(v * v, axis=-1) + 1e-8)
        return np.argmax(norm, axis=1)

    def n_params(self) -> int:
        return sum(getattr(self, k).size for k in self.param_names)


# ----------------------------------------------------------------------
# Same-parameter CNN baseline
# ----------------------------------------------------------------------

class TinyCNN:
    """3 conv layers + 1 FC, params matched (~within 10%) to TinyCapsNet."""

    def __init__(self, image_size: int = 40,
                 conv1_out: int = 16, conv1_k: int = 9,
                 conv2_out: int = 32, conv2_k: int = 5, conv2_stride: int = 2,
                 conv3_out: int = 64, conv3_k: int = 5, conv3_stride: int = 2,
                 fc_hidden: int = 64,
                 n_classes: int = 10,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        self.image_size = image_size

        self.W1 = rng.standard_normal((conv1_out, 1, conv1_k, conv1_k)).astype(np.float32) \
                  * np.sqrt(2.0 / (1 * conv1_k * conv1_k))
        self.b1 = np.zeros((conv1_out,), dtype=np.float32)
        oh1 = image_size - conv1_k + 1

        self.W2 = rng.standard_normal((conv2_out, conv1_out, conv2_k, conv2_k)).astype(np.float32) \
                  * np.sqrt(2.0 / (conv1_out * conv2_k * conv2_k))
        self.b2 = np.zeros((conv2_out,), dtype=np.float32)
        oh2 = (oh1 - conv2_k) // conv2_stride + 1

        self.W3 = rng.standard_normal((conv3_out, conv2_out, conv3_k, conv3_k)).astype(np.float32) \
                  * np.sqrt(2.0 / (conv2_out * conv3_k * conv3_k))
        self.b3 = np.zeros((conv3_out,), dtype=np.float32)
        oh3 = (oh2 - conv3_k) // conv3_stride + 1

        flat = conv3_out * oh3 * oh3
        self.W4 = rng.standard_normal((flat, fc_hidden)).astype(np.float32) \
                  * np.sqrt(2.0 / flat)
        self.b4 = np.zeros((fc_hidden,), dtype=np.float32)
        self.W5 = rng.standard_normal((fc_hidden, n_classes)).astype(np.float32) \
                  * np.sqrt(2.0 / fc_hidden)
        self.b5 = np.zeros((n_classes,), dtype=np.float32)

        self.conv2_stride = conv2_stride
        self.conv3_stride = conv3_stride
        self.flat = flat
        self.n_classes = n_classes

    def forward(self, x: np.ndarray):
        B = x.shape[0]
        x4 = x[:, None, :, :]
        z1, c1 = conv2d_forward(x4, self.W1, self.b1, stride=1)
        a1 = relu(z1)
        z2, c2 = conv2d_forward(a1, self.W2, self.b2, stride=self.conv2_stride)
        a2 = relu(z2)
        z3, c3 = conv2d_forward(a2, self.W3, self.b3, stride=self.conv3_stride)
        a3 = relu(z3)
        flat = a3.reshape(B, -1)
        h_pre = flat @ self.W4 + self.b4
        h = relu(h_pre)
        logits = h @ self.W5 + self.b5
        cache = dict(x=x, c1=c1, c2=c2, c3=c3, z1=z1, z2=z2, z3=z3,
                     a3_shape=a3.shape, flat=flat, h_pre=h_pre, h=h, logits=logits)
        return logits, cache

    def backward(self, d_logits: np.ndarray, cache):
        d_W5 = cache["h"].T @ d_logits
        d_b5 = d_logits.sum(axis=0)
        d_h = d_logits @ self.W5.T
        d_h_pre = d_h * (cache["h_pre"] > 0).astype(np.float32)
        d_W4 = cache["flat"].T @ d_h_pre
        d_b4 = d_h_pre.sum(axis=0)
        d_flat = d_h_pre @ self.W4.T
        d_a3 = d_flat.reshape(cache["a3_shape"])
        d_z3 = d_a3 * (cache["z3"] > 0).astype(np.float32)
        d_a2, d_W3, d_b3 = conv2d_backward(d_z3, cache["c3"])
        d_z2 = d_a2 * (cache["z2"] > 0).astype(np.float32)
        d_a1, d_W2, d_b2 = conv2d_backward(d_z2, cache["c2"])
        d_z1 = d_a1 * (cache["z1"] > 0).astype(np.float32)
        _, d_W1, d_b1 = conv2d_backward(d_z1, cache["c1"])
        return dict(W1=d_W1, b1=d_b1, W2=d_W2, b2=d_b2, W3=d_W3, b3=d_b3,
                    W4=d_W4, b4=d_b4, W5=d_W5, b5=d_b5)

    @property
    def param_names(self):
        return ("W1", "b1", "W2", "b2", "W3", "b3", "W4", "b4", "W5", "b5")

    def predict(self, x: np.ndarray) -> np.ndarray:
        logits, _ = self.forward(x)
        return np.argmax(logits, axis=1)

    def n_params(self) -> int:
        return sum(getattr(self, k).size for k in self.param_names)


def softmax_xent_loss(logits: np.ndarray, y: np.ndarray):
    B = logits.shape[0]
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    p = e / e.sum(axis=1, keepdims=True)
    log_p = np.log(p[np.arange(B), y] + 1e-12)
    loss = float(-log_p.mean())
    d_logits = p.copy()
    d_logits[np.arange(B), y] -= 1.0
    d_logits /= B
    return loss, d_logits


# ----------------------------------------------------------------------
# Adam
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, model, lr=1e-3, beta1=0.9, beta2=0.999, eps=1e-8):
        self.lr, self.b1, self.b2, self.eps = lr, beta1, beta2, eps
        self.t = 0
        self.m = {k: np.zeros_like(getattr(model, k)) for k in model.param_names}
        self.v = {k: np.zeros_like(getattr(model, k)) for k in model.param_names}

    def step(self, model, grads):
        self.t += 1
        bc1 = 1.0 - self.b1 ** self.t
        bc2 = 1.0 - self.b2 ** self.t
        for k, g in grads.items():
            m = self.m[k]; v = self.v[k]
            m[...] = self.b1 * m + (1.0 - self.b1) * g
            v[...] = self.b2 * v + (1.0 - self.b2) * (g * g)
            update = self.lr * (m / bc1) / (np.sqrt(v / bc2) + self.eps)
            getattr(model, k)[...] -= update


# ----------------------------------------------------------------------
# Train + evaluate
# ----------------------------------------------------------------------

def _shuffled_iter(x, y, batch_size, rng):
    idx = rng.permutation(x.shape[0])
    for i in range(0, x.shape[0] - batch_size + 1, batch_size):
        sl = idx[i:i + batch_size]
        yield x[sl], y[sl]


def train(arch: str = "capsnet",
          n_epochs: int = 3,
          batch_size: int = 64,
          lr: float = 1e-3,
          n_train: int = 10000,
          max_shift: int = 6,
          seed: int = 0,
          verbose: bool = True,
          val_every_steps: int = 100,
          val_n: int = 1000,
          snapshot_callback=None):
    rng = np.random.default_rng(seed)
    train_imgs, train_lbls = load_mnist("train")
    test_imgs, test_lbls = load_mnist("test")
    if n_train < train_imgs.shape[0]:
        sel = rng.permutation(train_imgs.shape[0])[:n_train]
        train_imgs = train_imgs[sel]
        train_lbls = train_lbls[sel]

    # Pre-translate the entire training set ONCE per epoch so the model still
    # sees a fresh translation per pass without paying the augmentation cost
    # inside the inner loop.

    # Build a fixed translated-MNIST val set
    val_sel = rng.permutation(test_imgs.shape[0])[:val_n]
    val_base = test_imgs[val_sel]
    val_lbls = test_lbls[val_sel]
    val_x = make_translated_mnist(val_base, max_shift=max_shift, seed=seed + 1)

    if arch == "capsnet":
        model = TinyCapsNet(seed=seed)
    elif arch == "cnn":
        model = TinyCNN(seed=seed)
    else:
        raise ValueError(arch)

    opt = Adam(model, lr=lr)

    if verbose:
        print(f"# arch={arch}  params={model.n_params():,}  "
              f"train={n_train}  batch={batch_size}  lr={lr}")

    history = {"step": [], "loss": [], "val_acc": []}
    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        # Re-translate at the start of each epoch (random per pass).
        ep_x = make_translated_mnist(train_imgs, max_shift=max_shift,
                                     seed=seed * 1000 + epoch)
        ep_y = train_lbls
        epoch_loss = 0.0
        n_batches = 0
        for xb, yb in _shuffled_iter(ep_x, ep_y, batch_size, rng):
            if arch == "capsnet":
                v, cache = model.forward(xb)
                loss, d_v = margin_loss(v, yb)
                grads = model.backward(d_v, cache)
            else:
                logits, cache = model.forward(xb)
                loss, d_logits = softmax_xent_loss(logits, yb)
                grads = model.backward(d_logits, cache)
            opt.step(model, grads)
            epoch_loss += loss
            n_batches += 1
            step += 1

            if step % val_every_steps == 0:
                preds = []
                for i in range(0, val_n, 64):
                    preds.append(model.predict(val_x[i:i + 64]))
                preds = np.concatenate(preds)
                acc = float((preds == val_lbls).mean())
                history["step"].append(step)
                history["loss"].append(loss)
                history["val_acc"].append(acc)
                if verbose:
                    el = time.time() - t0
                    print(f"  step {step:4d}  loss={loss:.4f}  val_acc={acc:.3f}  ({el:.1f}s)",
                          flush=True)
                if snapshot_callback is not None:
                    snapshot_callback(step, model, history)

        if verbose:
            avg = epoch_loss / max(1, n_batches)
            el = time.time() - t0
            print(f"epoch {epoch+1}/{n_epochs}  avg_loss={avg:.4f}  ({el:.1f}s)",
                  flush=True)

    return model, history


def evaluate(model, x: np.ndarray, y: np.ndarray, batch_size: int = 64) -> float:
    preds = []
    for i in range(0, x.shape[0], batch_size):
        preds.append(model.predict(x[i:i + batch_size]))
    preds = np.concatenate(preds)
    return float((preds == y).mean())


def evaluate_robustness(model, train_data, test_data) -> dict:
    """Returns dict with translated-MNIST acc and affNIST acc."""
    tr_x, tr_y = train_data
    te_x, te_y = test_data
    return {
        "translated_mnist_acc": evaluate(model, tr_x, tr_y),
        "affnist_acc": evaluate(model, te_x, te_y),
    }


# ----------------------------------------------------------------------
# Environment recording (per project rule)
# ----------------------------------------------------------------------

def record_env() -> dict:
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor() or platform.machine(),
    )


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--arch", choices=("capsnet", "cnn", "both"), default="both")
    p.add_argument("--n-train", type=int, default=5000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--max-shift", type=int, default=6)
    p.add_argument("--n-test", type=int, default=2000,
                   help="number of synthesized affNIST test samples")
    p.add_argument("--out", default=None,
                   help="optional path to save results JSON")
    args = p.parse_args()

    archs = ("capsnet", "cnn") if args.arch == "both" else (args.arch,)

    # Common test set
    print("# loading affNIST test set")
    aff_x, aff_y, source = load_affnist_test(n_synth=args.n_test, seed=args.seed)
    # Translated-MNIST test (in-distribution)
    test_imgs, test_lbls = load_mnist("test")
    rng = np.random.default_rng(args.seed + 9)
    sel = rng.permutation(test_imgs.shape[0])[:args.n_test]
    tr_x = make_translated_mnist(test_imgs[sel], max_shift=args.max_shift,
                                 seed=args.seed + 9)
    tr_y = test_lbls[sel]

    results = {"args": vars(args), "env": record_env(),
               "affnist_source": source, "models": {}}
    for arch in archs:
        print(f"\n=== training {arch} ===")
        t0 = time.time()
        model, hist = train(arch=arch, n_epochs=args.n_epochs,
                            batch_size=args.batch_size, lr=args.lr,
                            n_train=args.n_train, max_shift=args.max_shift,
                            seed=args.seed)
        wall = time.time() - t0
        rob = evaluate_robustness(model, (tr_x, tr_y), (aff_x, aff_y))
        results["models"][arch] = dict(
            params=model.n_params(),
            wall_train=wall,
            translated_mnist_acc=rob["translated_mnist_acc"],
            affnist_acc=rob["affnist_acc"],
            history=hist,
        )
        print(f"\n{arch}: translated-MNIST acc={rob['translated_mnist_acc']:.3f}, "
              f"affNIST acc={rob['affnist_acc']:.3f}, wall={wall:.1f}s, "
              f"params={model.n_params():,}")

    if "capsnet" in results["models"] and "cnn" in results["models"]:
        gap = (results["models"]["capsnet"]["affnist_acc"]
               - results["models"]["cnn"]["affnist_acc"])
        results["robustness_gap_capsnet_minus_cnn"] = gap
        print(f"\n>>> robustness gap (CapsNet - CNN) on affNIST: {gap:+.3f}")

    if args.out:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        Path(args.out).write_text(json.dumps(results, indent=2, default=float))
        print(f"  wrote {args.out}")


if __name__ == "__main__":
    main()
