"""
Forward-Forward on CIFAR-10 with locally-connected layers.

Reproduction of Hinton (2022), "The Forward-Forward Algorithm: Some Preliminary
Investigations", section on CIFAR-10 — the experiment that argues FF closes
the gap with backprop on cluttered images using **locally-connected** (not
weight-shared) layers.

Key ideas
---------
* CIFAR-10 colour images, 32x32x3, 10 classes, with the label one-hot encoded
  into a small region of the image (top-left strip of the red channel).
* Two or three locally-connected layers — every spatial location has its own
  independent weight tensor, so the model is *not* a CNN. Hinton argues this
  matches the cortex (no weight sharing across V1) while still giving each
  layer a topographic 32x32 map.
* Each layer is trained with the FF goodness rule (Hinton 2022, eq. 1):

      L = log(1 + exp(-(g_pos - theta))) + log(1 + exp(g_neg - theta))

  where g = mean(h^2) over all units in the layer; theta = 2.0; positive =
  (image, true_label), negative = (image, wrong_label).
* Between layers, activations are renormalised so mean(h^2) = 1 (the standard
  Hinton recipe — strips magnitude so deeper layers cannot read off goodness).
* Prediction at test time: try each of the 10 candidate labels, sum goodness
  across layers (skip layer 0 because it sees the label pixels directly),
  and pick the argmax.

Backprop baseline (for comparison) is the *same* locally-connected stack with
a final flatten + linear softmax classifier, trained end-to-end with cross-
entropy.

Reference: Hinton (2022), arXiv:2212.13345.

Constraints: numpy + matplotlib + imageio/pillow + urllib + tarfile + pickle
only. No torch / tensorflow.
"""

from __future__ import annotations
import argparse
import os
import pickle
import platform
import subprocess
import sys
import tarfile
import time
import urllib.request
import zipfile
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


# ---------------------------------------------------------------------------
# CIFAR-10 loader
# ---------------------------------------------------------------------------

# Primary mirror (Krizhevsky's original). As of 2026-05 this returns 503 —
# Toronto migrated cs.toronto.edu to a Squarespace site and the ~kriz/ path
# no longer serves files. We try it first for completeness, then fall back
# to a Kaggle ImageFolder PNG mirror (~184 MB ZIP of 60K PNGs).
CIFAR_URL = "https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz"
CIFAR_PNG_URL = ("https://www.kaggle.com/api/v1/datasets/download/oxcdcd/"
                 "cifar10")

CACHE_DIR = os.path.expanduser("~/.cache/hinton-cifar")
TARBALL = "cifar-10-python.tar.gz"
EXTRACTED_DIR = "cifar-10-batches-py"
PNG_ZIP = "cifar10-png.zip"
NPZ_CACHE = "cifar10.npz"

CIFAR_CLASSES = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
_CLASS_TO_IDX = {c: i for i, c in enumerate(CIFAR_CLASSES)}


def _download(url: str, dst: str, expect_min_bytes: int = 0) -> None:
    """Download with a User-Agent header. Streams in 1 MB chunks. Validates
    a minimum byte count when given so HTML error pages don't masquerade as
    data.
    """
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    if os.path.exists(dst):
        return
    print(f"  downloading {url} -> {dst}")
    tmp = dst + ".part"
    req = urllib.request.Request(
        url, headers={"User-Agent":
                      "Mozilla/5.0 (compatible; hinton-problems/1.0)"})
    with urllib.request.urlopen(req) as resp, open(tmp, "wb") as f:
        chunk = 1 << 20  # 1 MB
        n = 0
        while True:
            buf = resp.read(chunk)
            if not buf:
                break
            f.write(buf)
            n += len(buf)
    if expect_min_bytes and n < expect_min_bytes:
        os.remove(tmp)
        raise IOError(
            f"download truncated: got {n} bytes from {url}, expected "
            f">= {expect_min_bytes}")
    os.replace(tmp, dst)


def _try_download_tarball(cache_dir: str) -> Optional[str]:
    tar_path = os.path.join(cache_dir, TARBALL)
    if os.path.exists(tar_path):
        return tar_path
    try:
        _download(CIFAR_URL, tar_path, expect_min_bytes=100_000_000)
        return tar_path
    except Exception as e:
        print(f"  toronto mirror failed ({e}); falling back to PNG mirror")
        for p in (tar_path, tar_path + ".part"):
            if os.path.exists(p):
                os.remove(p)
        return None


def _ensure_pickle_extracted(cache_dir: str) -> Optional[str]:
    extracted = os.path.join(cache_dir, EXTRACTED_DIR)
    if os.path.isdir(extracted):
        return extracted
    tar_path = _try_download_tarball(cache_dir)
    if tar_path is None:
        return None
    print(f"  extracting {tar_path} -> {cache_dir}")
    with tarfile.open(tar_path, "r:gz") as tf:
        tf.extractall(cache_dir)
    return extracted if os.path.isdir(extracted) else None


def _load_batch(path: str) -> tuple[np.ndarray, np.ndarray]:
    """Load one CIFAR-10 pickle batch. Raw data layout is uint8 (10000, 3072)
    with channel-major (R then G then B). We reshape to (N, 32, 32, 3).
    """
    with open(path, "rb") as f:
        d = pickle.load(f, encoding="bytes")
    raw = d[b"data"]
    labels = np.array(d[b"labels"], dtype=np.int32)
    images = raw.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    return images, labels


def _load_from_pickle(extracted: str
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    xs, ys = [], []
    for i in range(1, 6):
        x, y = _load_batch(os.path.join(extracted, f"data_batch_{i}"))
        xs.append(x)
        ys.append(y)
    train_x = np.concatenate(xs, axis=0)
    train_y = np.concatenate(ys, axis=0)
    test_x, test_y = _load_batch(os.path.join(extracted, "test_batch"))
    return train_x, train_y, test_x, test_y


def _ensure_png_zip(cache_dir: str) -> str:
    zip_path = os.path.join(cache_dir, PNG_ZIP)
    if os.path.exists(zip_path):
        return zip_path
    _download(CIFAR_PNG_URL, zip_path, expect_min_bytes=100_000_000)
    return zip_path


def _load_from_png_zip(zip_path: str
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load CIFAR-10 from the Kaggle PNG ImageFolder mirror.

    Layout inside the zip:
        cifar10/labels.txt
        cifar10/train/<class>/<n>_<class>.png
        cifar10/test/<class>/<n>_<class>.png
    50000 train + 10000 test images (32x32 RGB).
    """
    from PIL import Image  # already in deps; lazy import
    print(f"  decoding PNGs from {zip_path} (this takes ~30 s) ...")
    train_imgs, train_lbls = [], []
    test_imgs, test_lbls = [], []
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = [n for n in zf.namelist() if n.endswith(".png")]
        for name in names:
            parts = name.split("/")
            if len(parts) < 4:
                continue
            split = parts[1]
            cls = parts[2]
            label = _CLASS_TO_IDX.get(cls)
            if label is None:
                continue
            with zf.open(name) as f:
                arr = np.asarray(Image.open(f).convert("RGB"), dtype=np.uint8)
            if split == "train":
                train_imgs.append(arr); train_lbls.append(label)
            elif split == "test":
                test_imgs.append(arr); test_lbls.append(label)
    train_x = np.stack(train_imgs, axis=0)
    train_y = np.array(train_lbls, dtype=np.int32)
    test_x = np.stack(test_imgs, axis=0)
    test_y = np.array(test_lbls, dtype=np.int32)
    return train_x, train_y, test_x, test_y


# Per-channel mean of the *training* split, in [0, 1] units. Cached so we
# don't recompute on every reload. Subtracting this from every image gives
# zero-centred inputs, which is critical for FF on CIFAR: with raw [0, 1]
# pixels the squared activation in layer 0 is dominated by the constant
# image brightness rather than the label-dependent feature signal.
CIFAR_MEAN = np.array([0.4914, 0.4822, 0.4465], dtype=np.float32)


def load_cifar10(cache_dir: str = CACHE_DIR
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return (train_x, train_y, test_x, test_y).

    Images are float32 in [0, 1] with shape (N, 32, 32, 3).
    Labels are int32 (0..9) with shape (N,).

    Source preference:
      1. cached `cifar10.npz` in cache_dir (fast reload after first run).
      2. Toronto `cifar-10-python.tar.gz` (Krizhevsky pickle format) — as of
         2026-05 this returns 503 because cs.toronto.edu migrated; the code
         attempts it for robustness then falls through.
      3. Kaggle PNG mirror (`oxcdcd/cifar10`, 184 MB zip of 60K PNGs).

    First run caches the parsed uint8 arrays as a single ~180 MB
    `cifar10.npz` so subsequent runs reload in well under a second.
    """
    os.makedirs(cache_dir, exist_ok=True)
    npz_path = os.path.join(cache_dir, NPZ_CACHE)
    if os.path.exists(npz_path):
        z = np.load(npz_path)
        train_x = z["train_x"]; train_y = z["train_y"]
        test_x = z["test_x"];   test_y = z["test_y"]
    else:
        extracted = _ensure_pickle_extracted(cache_dir)
        if extracted is not None:
            train_x, train_y, test_x, test_y = _load_from_pickle(extracted)
        else:
            zip_path = _ensure_png_zip(cache_dir)
            train_x, train_y, test_x, test_y = _load_from_png_zip(zip_path)
        np.savez(npz_path,
                 train_x=train_x.astype(np.uint8),
                 train_y=train_y.astype(np.int32),
                 test_x=test_x.astype(np.uint8),
                 test_y=test_y.astype(np.int32))
        print(f"  cached parsed arrays at {npz_path}")
    train_x_f = train_x.astype(np.float32) / 255.0
    test_x_f = test_x.astype(np.float32) / 255.0
    return train_x_f, train_y.astype(np.int32), test_x_f, test_y.astype(np.int32)


# ---------------------------------------------------------------------------
# Label-in-image encoding
# ---------------------------------------------------------------------------

LABEL_LEN = 10                   # one-hot bins
LABEL_ROWS = 3                   # how many top rows are dedicated to the label
LABEL_OFF = -1.0                 # value for unset bins (centred-pixel space)
LABEL_ON = 1.0                   # value for the set bin


def encode_label_in_image(images: np.ndarray, labels: np.ndarray
                          ) -> np.ndarray:
    """Overwrite a `LABEL_ROWS x LABEL_LEN x 3` block with a (high-contrast,
    channel-broadcast, row-broadcast) one-hot.

    images: (B, 32, 32, 3) float32, *already centred* (per-channel mean
            subtracted), so typical pixel values are in roughly [-0.5, 0.5].
    labels: (B,) int.

    Why this encoding (a documented deviation from the MNIST recipe)?
        CIFAR has 32*32*3 = 3072 input dims and the FF goodness pools over
        all units in a layer. If we wrote the label into 10 red-channel
        pixels at intensity {0, 1} as for MNIST, the per-image difference
        between positive and negative is only 2 of 3072 floats with
        magnitude 1, and the layer-0 goodness gap stays at ~0 even after
        many epochs (verified empirically). Three changes restore signal:
          1. Broadcast across all 3 channels (10 -> 30 affected pixels).
          2. Replicate down `LABEL_ROWS` rows so more receptive-field
             positions in layer 0 have the label inside their RF
             (LABEL_ROWS x LABEL_LEN x 3 = 90 pixels with LABEL_ROWS=3).
          3. Use a stronger contrast (-1 vs +1, magnitude 2) so the
             affected pixels are clear outliers vs the centred [-0.5, 0.5]
             pixel range.

        Hinton (2022) does not specify a single recipe for CIFAR label
        encoding -- the paper uses recurrent label-via-attention rather
        than label-in-pixels. The encoding here is the minimum-fuss
        adaptation of the MNIST trick that gets the goodness gap to open.
    """
    out = images.copy()
    out[:, :LABEL_ROWS, :LABEL_LEN, :] = LABEL_OFF
    # Set the on-position across all rows / channels.
    out[np.arange(out.shape[0])[:, None], :LABEL_ROWS, labels[:, None], :] = LABEL_ON
    return out


def make_negative_labels(labels: np.ndarray, n_classes: int = 10,
                         rng: Optional[np.random.Generator] = None
                         ) -> np.ndarray:
    """Pick a wrong label uniformly from {0..n_classes-1} \\ {true} per row."""
    if rng is None:
        rng = np.random.default_rng()
    offsets = rng.integers(1, n_classes, size=labels.shape[0]).astype(np.int32)
    return ((labels.astype(np.int32) + offsets) % n_classes).astype(np.int32)


# ---------------------------------------------------------------------------
# Locally-connected layer
# ---------------------------------------------------------------------------

@dataclass
class LCLayer:
    """A locally-connected layer.

    Per spatial output location (i, j) we store an independent weight tensor
    of shape (RF * RF * C_in, C_out) — i.e. NO weight sharing across
    locations, unlike a CNN. This is the architectural choice Hinton (2022)
    motivates as a closer match to cortex.

    Forward (channels-last):
        patches (B, H_o, W_o, RF*RF*C_in) = sliding-window view of x
        z       (B, H_o, W_o, C_out) = einsum('bijk,ijke->bije', patches, W)
                                     + b[None, :, :, :]

    Then h = ReLU(z); the FF goodness is mean(h^2) over (H_o, W_o, C_out).

    Adam state (m, v) lives on the layer for FF training. The backprop path
    used for the baseline reuses the same layer's forward but a different
    train loop.
    """
    H_in: int
    W_in: int
    C_in: int
    RF: int
    H_out: int
    W_out: int
    C_out: int
    W: np.ndarray   # (H_out, W_out, RF * RF * C_in, C_out)
    b: np.ndarray   # (H_out, W_out, C_out)
    threshold: float
    # Adam state.
    m_W: np.ndarray = field(default=None)
    v_W: np.ndarray = field(default=None)
    m_b: np.ndarray = field(default=None)
    v_b: np.ndarray = field(default=None)
    step: int = 0

    @classmethod
    def init(cls, H_in: int, W_in: int, C_in: int, RF: int, C_out: int,
             threshold: float, rng: np.random.Generator) -> "LCLayer":
        if RF > H_in or RF > W_in:
            raise ValueError(
                f"RF={RF} too big for input ({H_in}x{W_in})")
        H_out = H_in - RF + 1
        W_out = W_in - RF + 1
        fan_in = RF * RF * C_in
        # He / Glorot-style: scale = sqrt(2 / fan_in).
        scale = np.sqrt(2.0 / fan_in)
        W = (scale * rng.standard_normal((H_out, W_out, fan_in,
                                          C_out))).astype(np.float32)
        b = np.zeros((H_out, W_out, C_out), dtype=np.float32)
        return cls(
            H_in=H_in, W_in=W_in, C_in=C_in, RF=RF,
            H_out=H_out, W_out=W_out, C_out=C_out,
            W=W, b=b, threshold=threshold,
            m_W=np.zeros_like(W), v_W=np.zeros_like(W),
            m_b=np.zeros_like(b), v_b=np.zeros_like(b))

    @property
    def n_units(self) -> int:
        return self.H_out * self.W_out * self.C_out

    def patches(self, x: np.ndarray) -> np.ndarray:
        """Sliding-window view of x.

        x: (B, H_in, W_in, C_in) float32
        returns: (B, H_out, W_out, RF * RF * C_in)  -- contiguous

        Note on perf: numpy's einsum on locally-connected weights is ~30x
        slower than per-location batched matmul (the inner sum is a tiny
        K -> E dot but einsum doesn't dispatch it to BLAS). The batched
        matmul path lives in `_lc_forward` / `_lc_grad_W` / `_lc_grad_x`
        below. We keep this method to materialise the patches in
        (B, H, W, K) form which is then transposed to (H, W, B, K) for
        matmul -- the transpose is a view, so no extra copy.
        """
        v = sliding_window_view(x, (self.RF, self.RF), axis=(1, 2))
        # v shape: (B, H_out, W_out, C_in, RF, RF)
        v = np.transpose(v, (0, 1, 2, 4, 5, 3))
        # ascontiguous so the matmul on (H, W, B, K) is well-strided.
        v = np.ascontiguousarray(v)
        return v.reshape(v.shape[0], self.H_out, self.W_out,
                         self.RF * self.RF * self.C_in)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (patches, z, h) where z = pre-ReLU and h = ReLU(z)."""
        p = self.patches(x)
        z = _lc_forward(p, self.W, self.b)
        h = np.maximum(z, 0.0)
        return p, z, h

    @staticmethod
    def normalize(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Rescale per sample so mean(h^2) = 1 over (H, W, C)."""
        # h: (B, H, W, C). Flatten the non-batch dims for the norm.
        flat = h.reshape(h.shape[0], -1)
        rms = np.sqrt((flat * flat).mean(axis=1, keepdims=True) + eps)
        return (flat / rms).reshape(h.shape)


# ---------------------------------------------------------------------------
# Locally-connected forward / backward via batched matmul
# ---------------------------------------------------------------------------
#
# A locally-connected layer has independent weights at every output spatial
# location (i, j). Naive einsum over the (H, W, B, K, E) tensor is dominated
# by Python-level dispatch overhead and runs at <1 GFLOPs effective. By
# transposing to (H, W, B, K) and (H, W, K, E) we hand each location's matmul
# to BLAS via numpy.matmul, which dispatches to GEMM and is ~30x faster
# (measured on Apple M-series, RF=11, C_out=8).

def _lc_forward(p: np.ndarray, W: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Locally-connected forward.

    p: (B, H, W, K)  patches (K = RF * RF * C_in)
    W: (H, W, K, E)
    b: (H, W, E)
    returns z: (B, H, W, E)
    """
    # (B, H, W, K) -> view (H, W, B, K). matmul -> (H, W, B, E).
    p_t = np.transpose(p, (1, 2, 0, 3))
    out_t = np.matmul(p_t, W)
    z = np.transpose(out_t, (2, 0, 1, 3)) + b
    return z


def _lc_grad_W(p: np.ndarray, dz: np.ndarray) -> np.ndarray:
    """gW[i, j, k, e] = sum_b p[b, i, j, k] * dz[b, i, j, e]."""
    p_t = np.transpose(p, (1, 2, 0, 3))     # (H, W, B, K)
    dz_t = np.transpose(dz, (1, 2, 0, 3))   # (H, W, B, E)
    # (H, W, K, B) @ (H, W, B, E) -> (H, W, K, E)
    return np.matmul(np.transpose(p_t, (0, 1, 3, 2)), dz_t)


def _lc_grad_patches(dz: np.ndarray, W: np.ndarray) -> np.ndarray:
    """d patches[b, i, j, k] = sum_e dz[b, i, j, e] * W[i, j, k, e].

    Only used by the BP baseline (FF needs no upstream input grad).
    """
    dz_t = np.transpose(dz, (1, 2, 0, 3))   # (H, W, B, E)
    # (H, W, B, E) @ (H, W, E, K) -> (H, W, B, K)
    out_t = np.matmul(dz_t, np.transpose(W, (0, 1, 3, 2)))
    return np.transpose(out_t, (2, 0, 1, 3))


# ---------------------------------------------------------------------------
# FF loss / gradients (single layer)
# ---------------------------------------------------------------------------

def _softplus(x: np.ndarray) -> np.ndarray:
    # log(1 + e^x), numerically stable.
    return np.where(x > 0, x + np.log1p(np.exp(-x)), np.log1p(np.exp(x)))


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return np.where(x >= 0,
                    1.0 / (1.0 + np.exp(-x)),
                    np.exp(x) / (1.0 + np.exp(x)))


def ff_layer_loss_grad(layer: LCLayer, x_pos: np.ndarray, x_neg: np.ndarray
                       ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """One-layer FF loss + grad of d(loss) / d(W, b).

    x_pos, x_neg: (B, H_in, W_in, C_in) float32. Returns
    (loss, gW, gb, mean_g_pos, mean_g_neg). Loss is averaged over 2*B samples.
    """
    p_pos, z_pos, h_pos = layer.forward(x_pos)
    p_neg, z_neg, h_neg = layer.forward(x_neg)

    B = x_pos.shape[0]
    N = layer.n_units  # number of units in this layer

    # g[b] = mean over (H, W, C) of h[b]^2.
    g_pos = (h_pos * h_pos).reshape(B, -1).mean(axis=1)
    g_neg = (h_neg * h_neg).reshape(B, -1).mean(axis=1)
    theta = layer.threshold
    a_pos = g_pos - theta
    a_neg = g_neg - theta

    loss = float((_softplus(-a_pos).sum() + _softplus(a_neg).sum()) / (2 * B))

    # d loss_pos / d g_pos = (sigmoid(a_pos) - 1) / (2B)
    # d loss_neg / d g_neg =  sigmoid(a_neg)      / (2B)
    dg_pos = (_sigmoid(a_pos) - 1.0) / (2 * B)            # (B,)
    dg_neg = _sigmoid(a_neg) / (2 * B)                    # (B,)

    # d g / d h = (2/N) * h
    coeff = 2.0 / N
    dh_pos = coeff * h_pos * dg_pos[:, None, None, None]
    dh_neg = coeff * h_neg * dg_neg[:, None, None, None]

    # ReLU backward: dz = dh * (z > 0)
    dz_pos = dh_pos * (z_pos > 0)
    dz_neg = dh_neg * (z_neg > 0)

    # d/dW[i,j,k,e] = sum_b patches[b,i,j,k] * dz[b,i,j,e]
    gW = (_lc_grad_W(p_pos, dz_pos) +
          _lc_grad_W(p_neg, dz_neg)).astype(np.float32)
    gb = (dz_pos.sum(axis=0) + dz_neg.sum(axis=0)).astype(np.float32)

    return loss, gW, gb, float(g_pos.mean()), float(g_neg.mean())


def adam_update(layer: LCLayer, gW: np.ndarray, gb: np.ndarray,
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
# FF model
# ---------------------------------------------------------------------------

@dataclass
class FFModel:
    layers: list  # list[LCLayer]
    n_classes: int = 10

    @classmethod
    def init(cls, layer_specs: list, threshold: float,
             rng: np.random.Generator, n_classes: int = 10) -> "FFModel":
        """layer_specs: list of (RF, C_out). Input is fixed (32, 32, 3).

        Each entry consumes the previous layer's output shape.
        """
        layers = []
        H, W, C = 32, 32, 3
        for (rf, cout) in layer_specs:
            L = LCLayer.init(H, W, C, rf, cout, threshold, rng)
            layers.append(L)
            H, W, C = L.H_out, L.W_out, L.C_out
        return cls(layers=layers, n_classes=n_classes)


def model_layer_activations(model: FFModel, x: np.ndarray) -> list:
    """Forward through every layer, returning post-ReLU activations.

    Between layers we renormalise so mean(h^2) = 1 (standard Hinton recipe).
    """
    acts = []
    h = x
    for L in model.layers:
        _, _, h_relu = L.forward(h)
        acts.append(h_relu)
        h = LCLayer.normalize(h_relu)
    return acts


def goodness_per_layer(model: FFModel, x: np.ndarray,
                       skip_first: bool = True) -> np.ndarray:
    """Return (B, n_used) array of per-sample goodness (mean(h^2))."""
    acts = model_layer_activations(model, x)
    if skip_first and len(acts) > 1:
        acts = acts[1:]
    return np.stack([h.reshape(h.shape[0], -1).mean(axis=1)
                     for h in acts], axis=1)


def predict_by_goodness(model: FFModel, images: np.ndarray,
                        skip_first: bool = True,
                        sub_batch: int = 256) -> np.ndarray:
    """Try every label; predict argmax of summed goodness across used layers.

    images: (B, 32, 32, 3) float32. Returns (B,) int32 predictions.

    The expansion is (B * 10) inputs at once — for B = 256 that's 2560 forward
    passes which fits in RAM. For larger B we sub-batch.
    """
    n = model.n_classes
    B = images.shape[0]
    preds = np.empty(B, dtype=np.int32)
    for s in range(0, B, sub_batch):
        e = min(s + sub_batch, B)
        chunk = images[s:e]
        cb = chunk.shape[0]
        # Build (cb * n, 32, 32, 3): each image repeated n times with
        # candidate label encoded in.
        repeated = np.repeat(chunk, n, axis=0).copy()
        cand_labels = np.tile(np.arange(n, dtype=np.int32), cb)
        # Erase the label strip and write the candidate one-hot. Must match
        # encode_label_in_image exactly.
        repeated[:, :LABEL_ROWS, :LABEL_LEN, :] = LABEL_OFF
        repeated[np.arange(cb * n)[:, None], :LABEL_ROWS, cand_labels[:, None], :] = LABEL_ON
        g = goodness_per_layer(model, repeated, skip_first=skip_first)
        summed = g.sum(axis=1).reshape(cb, n)
        preds[s:e] = summed.argmax(axis=1).astype(np.int32)
    return preds


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_epochs: int = 8
    batch_size: int = 64
    lr: float = 0.003
    threshold: float = 2.0
    layer_specs: tuple = ((11, 8), (5, 8))   # (RF, C_out) per layer
    n_classes: int = 10
    seed: int = 0
    train_subset: Optional[int] = 10000
    eval_subset: int = 1000
    eval_every: int = 1
    skip_first_at_predict: bool = True


def train_ff(model: FFModel, data: tuple, cfg: TrainConfig,
             snapshot_callback: Optional[Callable] = None,
             snapshot_every: int = 1,
             verbose: bool = True) -> dict:
    train_x, train_y, test_x, test_y = data
    rng = np.random.default_rng(cfg.seed)

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
        epoch_loss = [0.0] * n_layers
        epoch_gpos = [0.0] * n_layers
        epoch_gneg = [0.0] * n_layers
        n_batches = 0

        for start in range(0, N, B):
            idx = perm[start:start + B]
            xb = train_x[idx]
            yb = train_y[idx]
            wrong_y = make_negative_labels(yb, cfg.n_classes, rng)

            x_pos = encode_label_in_image(xb, yb)
            x_neg = encode_label_in_image(xb, wrong_y)

            # Train each layer in turn on its own input. Input to L+1 is the
            # *normalised* hidden activation of L (no gradient passes back
            # through normalisation — exactly as in Hinton 2022).
            h_pos, h_neg = x_pos, x_neg
            for L_idx, layer in enumerate(model.layers):
                loss, gW, gb, gp, gn = ff_layer_loss_grad(layer, h_pos, h_neg)
                adam_update(layer, gW, gb, cfg.lr)
                epoch_loss[L_idx] += loss
                epoch_gpos[L_idx] += gp
                epoch_gneg[L_idx] += gn

                # Compute next layer's input (no grad).
                _, _, h_pos_next = layer.forward(h_pos)
                _, _, h_neg_next = layer.forward(h_neg)
                h_pos = LCLayer.normalize(h_pos_next)
                h_neg = LCLayer.normalize(h_neg_next)
            n_batches += 1

        for L_idx in range(n_layers):
            history["loss_per_layer"][L_idx].append(epoch_loss[L_idx] / n_batches)
            history["g_pos_per_layer"][L_idx].append(epoch_gpos[L_idx] / n_batches)
            history["g_neg_per_layer"][L_idx].append(epoch_gneg[L_idx] / n_batches)
        history["epoch"].append(epoch + 1)
        history["wallclock"].append(time.time() - t0)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.n_epochs - 1:
            test_pred = predict_by_goodness(
                model, eval_x, skip_first=cfg.skip_first_at_predict)
            train_pred = predict_by_goodness(
                model, eval_train_x, skip_first=cfg.skip_first_at_predict)
            test_acc = float((test_pred == eval_y).mean())
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


def evaluate_ff(model: FFModel, x: np.ndarray, y: np.ndarray,
                batch_size: int = 256, skip_first: bool = True) -> float:
    n = x.shape[0]
    correct = 0
    for s in range(0, n, batch_size):
        xb = x[s:s + batch_size]
        yb = y[s:s + batch_size]
        pred = predict_by_goodness(model, xb, skip_first=skip_first)
        correct += int((pred == yb).sum())
    return correct / n


def per_class_accuracy_ff(model: FFModel, x: np.ndarray, y: np.ndarray,
                          n_classes: int = 10, batch_size: int = 256,
                          skip_first: bool = True) -> np.ndarray:
    """Return (n_classes,) array of per-class accuracy."""
    preds = np.empty(x.shape[0], dtype=np.int32)
    for s in range(0, x.shape[0], batch_size):
        e = min(s + batch_size, x.shape[0])
        preds[s:e] = predict_by_goodness(model, x[s:e], skip_first=skip_first)
    accs = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float((preds[mask] == c).mean())
    return accs


# ---------------------------------------------------------------------------
# Backprop baseline (same locally-connected architecture, end-to-end CE)
# ---------------------------------------------------------------------------

@dataclass
class BPLayer:
    """Locally-connected layer for the backprop baseline (no FF state)."""
    W: np.ndarray   # (H_out, W_out, RF*RF*C_in, C_out)
    b: np.ndarray   # (H_out, W_out, C_out)
    H_in: int; W_in: int; C_in: int; RF: int
    H_out: int; W_out: int; C_out: int
    m_W: np.ndarray = field(default=None)
    v_W: np.ndarray = field(default=None)
    m_b: np.ndarray = field(default=None)
    v_b: np.ndarray = field(default=None)
    step: int = 0

    @classmethod
    def init(cls, H_in: int, W_in: int, C_in: int, RF: int, C_out: int,
             rng: np.random.Generator) -> "BPLayer":
        H_out = H_in - RF + 1
        W_out = W_in - RF + 1
        fan_in = RF * RF * C_in
        scale = np.sqrt(2.0 / fan_in)
        W = (scale * rng.standard_normal((H_out, W_out, fan_in,
                                          C_out))).astype(np.float32)
        b = np.zeros((H_out, W_out, C_out), dtype=np.float32)
        return cls(W=W, b=b, H_in=H_in, W_in=W_in, C_in=C_in, RF=RF,
                   H_out=H_out, W_out=W_out, C_out=C_out,
                   m_W=np.zeros_like(W), v_W=np.zeros_like(W),
                   m_b=np.zeros_like(b), v_b=np.zeros_like(b))

    def patches(self, x: np.ndarray) -> np.ndarray:
        v = sliding_window_view(x, (self.RF, self.RF), axis=(1, 2))
        v = np.transpose(v, (0, 1, 2, 4, 5, 3))
        v = np.ascontiguousarray(v)
        return v.reshape(v.shape[0], self.H_out, self.W_out,
                         self.RF * self.RF * self.C_in)


@dataclass
class BPModel:
    layers: list
    W_out: np.ndarray  # (flat_features, n_classes)
    b_out: np.ndarray  # (n_classes,)
    flat_features: int
    n_classes: int
    # Adam state for the readout head.
    m_W: np.ndarray = field(default=None)
    v_W: np.ndarray = field(default=None)
    m_b: np.ndarray = field(default=None)
    v_b: np.ndarray = field(default=None)
    step: int = 0

    @classmethod
    def init(cls, layer_specs: list, n_classes: int,
             rng: np.random.Generator) -> "BPModel":
        layers = []
        H, W, C = 32, 32, 3
        for (rf, cout) in layer_specs:
            L = BPLayer.init(H, W, C, rf, cout, rng)
            layers.append(L)
            H, W, C = L.H_out, L.W_out, L.C_out
        flat_features = H * W * C
        scale = np.sqrt(2.0 / flat_features)
        W_out = (scale * rng.standard_normal((flat_features, n_classes))
                 ).astype(np.float32)
        b_out = np.zeros(n_classes, dtype=np.float32)
        return cls(layers=layers, W_out=W_out, b_out=b_out,
                   flat_features=flat_features, n_classes=n_classes,
                   m_W=np.zeros_like(W_out), v_W=np.zeros_like(W_out),
                   m_b=np.zeros_like(b_out), v_b=np.zeros_like(b_out))


def bp_forward(model: BPModel, x: np.ndarray) -> tuple:
    """End-to-end forward. Returns (cache, h_flat, logits) where cache is
    the list of (patches, z, h) for each layer.
    """
    cache = []
    h = x
    for L in model.layers:
        p = L.patches(h)
        z = _lc_forward(p, L.W, L.b)
        h = np.maximum(z, 0.0)
        cache.append((p, z, h))
    h_flat = h.reshape(h.shape[0], -1)
    logits = h_flat @ model.W_out + model.b_out
    return cache, h_flat, logits


def softmax_ce(logits: np.ndarray, y: np.ndarray
               ) -> tuple[float, np.ndarray]:
    """Softmax cross-entropy. Returns (mean loss, dlogits)."""
    m = logits.max(axis=1, keepdims=True)
    e = np.exp(logits - m)
    p = e / e.sum(axis=1, keepdims=True)
    B = logits.shape[0]
    log_p = np.log(p[np.arange(B), y] + 1e-12)
    loss = float(-log_p.mean())
    dlogits = p.copy()
    dlogits[np.arange(B), y] -= 1.0
    dlogits /= B
    return loss, dlogits


def bp_backward(model: BPModel, cache: list, h_flat: np.ndarray,
                dlogits: np.ndarray
                ) -> tuple[list, list, np.ndarray, np.ndarray]:
    """Backward through the readout + locally-connected layers.

    Returns (grads_W, grads_b, gW_out, gb_out) where grads_W[i] is the grad
    for layer i's W tensor.
    """
    # Readout grads.
    gW_out = h_flat.T @ dlogits           # (flat_features, n_classes)
    gb_out = dlogits.sum(axis=0)          # (n_classes,)
    dh_flat = dlogits @ model.W_out.T     # (B, flat_features)

    grads_W: list = [None] * len(model.layers)
    grads_b: list = [None] * len(model.layers)

    # Reshape dh_flat back to spatial.
    last_layer = model.layers[-1]
    dh = dh_flat.reshape(-1, last_layer.H_out, last_layer.W_out,
                         last_layer.C_out)

    for i in range(len(model.layers) - 1, -1, -1):
        L = model.layers[i]
        p, z, h = cache[i]
        # ReLU backward.
        dz = dh * (z > 0)
        gW = _lc_grad_W(p, dz)
        gb = dz.sum(axis=0)
        grads_W[i] = gW.astype(np.float32)
        grads_b[i] = gb.astype(np.float32)
        if i > 0:
            # Backprop through the LC layer to get d/dx via the patches.
            d_patches = _lc_grad_patches(dz, L.W)  # (B, H_o, W_o, RF*RF*C_in)
            d_patches = d_patches.reshape(d_patches.shape[0],
                                          L.H_out, L.W_out,
                                          L.RF, L.RF, L.C_in)
            dx = np.zeros((d_patches.shape[0], L.H_in, L.W_in, L.C_in),
                          dtype=np.float32)
            # Scatter back to the input. RF**2 small additions.
            for a in range(L.RF):
                for c in range(L.RF):
                    dx[:, a:a + L.H_out, c:c + L.W_out, :] += d_patches[:, :, :, a, c, :]
            dh = dx
    return grads_W, grads_b, gW_out, gb_out


def adam_update_arrays(W: np.ndarray, b: np.ndarray,
                       gW: np.ndarray, gb: np.ndarray,
                       m_W: np.ndarray, v_W: np.ndarray,
                       m_b: np.ndarray, v_b: np.ndarray,
                       step: int, lr: float,
                       beta1: float = 0.9, beta2: float = 0.999,
                       eps: float = 1e-8) -> tuple:
    m_W = beta1 * m_W + (1 - beta1) * gW
    v_W = beta2 * v_W + (1 - beta2) * (gW * gW)
    m_b = beta1 * m_b + (1 - beta1) * gb
    v_b = beta2 * v_b + (1 - beta2) * (gb * gb)
    bc1 = 1 - beta1 ** step
    bc2 = 1 - beta2 ** step
    W -= lr * (m_W / bc1) / (np.sqrt(v_W / bc2) + eps)
    b -= lr * (m_b / bc1) / (np.sqrt(v_b / bc2) + eps)
    return W, b, m_W, v_W, m_b, v_b


def train_bp(model: BPModel, data: tuple, cfg: TrainConfig,
             verbose: bool = True) -> dict:
    """Train the backprop baseline end-to-end with softmax cross-entropy.

    Note: the backprop baseline does NOT see the label-in-pixel encoding —
    it gets the raw image and a true label target. This is the apples-to-
    apples comparison Hinton uses (FF gets the label baked in; backprop gets
    cross-entropy on the raw image).
    """
    train_x, train_y, test_x, test_y = data
    rng = np.random.default_rng(cfg.seed + 1)  # different seed than FF

    if cfg.train_subset is not None and cfg.train_subset < train_x.shape[0]:
        idx = rng.permutation(train_x.shape[0])[:cfg.train_subset]
        train_x = train_x[idx]
        train_y = train_y[idx]

    N = train_x.shape[0]
    B = cfg.batch_size
    history = {"epoch": [], "loss": [], "test_acc": [], "train_acc": [],
               "wallclock": []}

    eval_idx = rng.permutation(test_x.shape[0])[:cfg.eval_subset]
    eval_x = test_x[eval_idx]
    eval_y = test_y[eval_idx]
    train_eval_idx = rng.permutation(train_x.shape[0])[:cfg.eval_subset]
    eval_train_x = train_x[train_eval_idx]
    eval_train_y = train_y[train_eval_idx]

    t0 = time.time()
    for epoch in range(cfg.n_epochs):
        perm = rng.permutation(N)
        ep_loss = 0.0
        n_batches = 0
        for start in range(0, N, B):
            idx = perm[start:start + B]
            xb = train_x[idx]
            yb = train_y[idx]
            cache, h_flat, logits = bp_forward(model, xb)
            loss, dlogits = softmax_ce(logits, yb)
            grads_W, grads_b, gW_out, gb_out = bp_backward(
                model, cache, h_flat, dlogits)

            # Adam updates.
            for i, L in enumerate(model.layers):
                L.step += 1
                L.W, L.b, L.m_W, L.v_W, L.m_b, L.v_b = adam_update_arrays(
                    L.W, L.b, grads_W[i], grads_b[i],
                    L.m_W, L.v_W, L.m_b, L.v_b, L.step, cfg.lr)
            model.step += 1
            model.W_out, model.b_out, model.m_W, model.v_W, model.m_b, model.v_b = adam_update_arrays(
                model.W_out, model.b_out, gW_out, gb_out,
                model.m_W, model.v_W, model.m_b, model.v_b, model.step, cfg.lr)

            ep_loss += loss
            n_batches += 1

        history["epoch"].append(epoch + 1)
        history["loss"].append(ep_loss / n_batches)
        history["wallclock"].append(time.time() - t0)

        # Eval.
        test_acc = float((bp_predict_batch(model, eval_x) == eval_y).mean())
        train_acc = float((bp_predict_batch(model, eval_train_x) == eval_train_y).mean())
        history["test_acc"].append(test_acc)
        history["train_acc"].append(train_acc)

        if verbose:
            print(f"[BP] epoch {epoch + 1:3d}/{cfg.n_epochs}  "
                  f"loss={ep_loss/n_batches:.3f}  "
                  f"train={train_acc*100:.1f}%  "
                  f"test={test_acc*100:.1f}%  "
                  f"({history['wallclock'][-1]:.1f}s)")

    return history


def bp_predict_batch(model: BPModel, x: np.ndarray,
                     batch_size: int = 256) -> np.ndarray:
    out = np.empty(x.shape[0], dtype=np.int32)
    for s in range(0, x.shape[0], batch_size):
        e = min(s + batch_size, x.shape[0])
        _, _, logits = bp_forward(model, x[s:e])
        out[s:e] = logits.argmax(axis=1).astype(np.int32)
    return out


def evaluate_bp(model: BPModel, x: np.ndarray, y: np.ndarray) -> float:
    pred = bp_predict_batch(model, x)
    return float((pred == y).mean())


def per_class_accuracy_bp(model: BPModel, x: np.ndarray, y: np.ndarray,
                          n_classes: int = 10) -> np.ndarray:
    pred = bp_predict_batch(model, x)
    accs = np.zeros(n_classes, dtype=np.float32)
    for c in range(n_classes):
        mask = (y == c)
        if mask.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = float((pred[mask] == c).mean())
    return accs


# ---------------------------------------------------------------------------
# Environment / repro
# ---------------------------------------------------------------------------

def collect_env() -> dict:
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True,
            stderr=subprocess.DEVNULL).strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": commit,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--n-layers", type=int, default=2,
                   choices=(2, 3),
                   help="2 or 3 locally-connected FF layers.")
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--train-subset", type=int, default=10000,
                   help="Subset size for the training set; default 10K to fit "
                        "the wave-8 5-min budget (full 50K is 5x slower).")
    p.add_argument("--eval-subset", type=int, default=1000)
    p.add_argument("--full-test", action="store_true",
                   help="Evaluate on the full 10K test set at the end.")
    p.add_argument("--bp-baseline", action="store_true",
                   help="Also train a backprop baseline on the same arch.")
    p.add_argument("--save", type=str, default=None,
                   help="Save model + history to this .npz path.")
    args = p.parse_args()

    if args.n_layers == 2:
        layer_specs = ((11, 8), (5, 8))
    else:
        layer_specs = ((11, 8), (5, 8), (5, 8))

    cfg = TrainConfig(
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        threshold=args.threshold,
        layer_specs=layer_specs,
        seed=args.seed,
        train_subset=args.train_subset,
        eval_subset=args.eval_subset,
    )

    env = collect_env()
    print("Environment:", env)
    print(f"Loading CIFAR-10 from {CACHE_DIR} ...")
    train_x, train_y, test_x, test_y = load_cifar10()
    print(f"  train: {train_x.shape}, labels: {train_y.shape}")
    print(f"  test:  {test_x.shape}, labels: {test_y.shape}")
    # Centre by per-channel training mean. Critical for FF — see CIFAR_MEAN.
    train_x = train_x - CIFAR_MEAN
    test_x = test_x - CIFAR_MEAN
    data = (train_x, train_y, test_x, test_y)

    rng = np.random.default_rng(args.seed)
    print(f"\nFF model: {len(layer_specs)} locally-connected layers, "
          f"specs={layer_specs}")
    ff_model = FFModel.init(list(layer_specs), threshold=args.threshold,
                            rng=rng, n_classes=10)
    for i, L in enumerate(ff_model.layers):
        print(f"  L{i}: in=({L.H_in}x{L.W_in}x{L.C_in})  "
              f"RF={L.RF}  out=({L.H_out}x{L.W_out}x{L.C_out})  "
              f"params={L.W.size + L.b.size:,}")

    print("\nTraining FF...")
    t0 = time.time()
    ff_history = train_ff(ff_model, data, cfg, verbose=True)
    ff_wall = time.time() - t0

    print(f"\nFF training time: {ff_wall:.1f}s")
    if args.full_test:
        full_acc = evaluate_ff(ff_model, test_x, test_y,
                               skip_first=cfg.skip_first_at_predict)
        print(f"FF full test acc: {full_acc*100:.2f}% "
              f"(error: {(1 - full_acc)*100:.2f}%)")
        ff_history["full_test_acc"] = full_acc

    bp_history = None
    bp_wall = None
    if args.bp_baseline:
        print(f"\nBP baseline: same arch, end-to-end softmax cross-entropy")
        bp_rng = np.random.default_rng(args.seed + 100)
        bp_model = BPModel.init(list(layer_specs), n_classes=10, rng=bp_rng)
        t0 = time.time()
        bp_history = train_bp(bp_model, data, cfg, verbose=True)
        bp_wall = time.time() - t0
        print(f"\nBP training time: {bp_wall:.1f}s")
        if args.full_test:
            full_acc_bp = evaluate_bp(bp_model, test_x, test_y)
            print(f"BP full test acc: {full_acc_bp*100:.2f}% "
                  f"(error: {(1 - full_acc_bp)*100:.2f}%)")
            bp_history["full_test_acc"] = full_acc_bp

    if args.save:
        os.makedirs(os.path.dirname(args.save) or ".", exist_ok=True)
        save = {
            "layer_specs": np.array(layer_specs, dtype=np.int32),
            "threshold": np.float32(args.threshold),
            "seed": np.int32(args.seed),
            "ff_test_acc": np.array(ff_history["test_acc"], dtype=np.float32),
            "ff_train_acc": np.array(ff_history["train_acc"], dtype=np.float32),
            "ff_loss_per_layer": np.array(ff_history["loss_per_layer"],
                                          dtype=np.float32),
            "ff_g_pos": np.array(ff_history["g_pos_per_layer"], dtype=np.float32),
            "ff_g_neg": np.array(ff_history["g_neg_per_layer"], dtype=np.float32),
            "ff_wallclock": np.array(ff_history["wallclock"], dtype=np.float32),
        }
        for i, L in enumerate(ff_model.layers):
            save[f"ff_layer{i}_W"] = L.W
            save[f"ff_layer{i}_b"] = L.b
        if "full_test_acc" in ff_history:
            save["ff_full_test_acc"] = np.float32(ff_history["full_test_acc"])
        if bp_history is not None:
            save["bp_test_acc"] = np.array(bp_history["test_acc"],
                                           dtype=np.float32)
            save["bp_train_acc"] = np.array(bp_history["train_acc"],
                                            dtype=np.float32)
            save["bp_loss"] = np.array(bp_history["loss"], dtype=np.float32)
            save["bp_wallclock"] = np.array(bp_history["wallclock"],
                                            dtype=np.float32)
            if "full_test_acc" in bp_history:
                save["bp_full_test_acc"] = np.float32(bp_history["full_test_acc"])
            # Save BP weights so visualisation can recompute per-class acc.
            for i, L in enumerate(bp_model.layers):
                save[f"bp_layer{i}_W"] = L.W
                save[f"bp_layer{i}_b"] = L.b
            save["bp_W_out"] = bp_model.W_out
            save["bp_b_out"] = bp_model.b_out
        np.savez(args.save, **save)
        print(f"\nSaved to {args.save}")


if __name__ == "__main__":
    main()
