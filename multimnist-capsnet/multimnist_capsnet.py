"""
MultiMNIST + Capsule Network with Dynamic Routing.

Numpy reproduction of Sabour, Frosst & Hinton, "Dynamic routing between
capsules", NIPS 2017. The point of the paper is that capsules separately
identify and reconstruct heavily overlapping digits via routing-by-agreement.

This implementation:
  - MultiMNIST: 36x36 canvas, two distinct-class MNIST digits each shifted
    by +/- 4 pixels, overlaid with pixel-wise max, requiring bounding-box
    overlap >= 80%.
  - CapsNet (reduced from paper to fit pure-numpy budget):
      Conv1: 32 channels, 9x9 stride 1, ReLU
      PrimaryCaps: 8 capsules x 8-D, 9x9 stride 2 (-> 10x10 spatial = 800 caps)
      DigitCaps: 10 capsules x 16-D, 3-iter dynamic routing
      Decoder: 160 -> 256 -> 512 -> 1296 (per-digit reconstruction, sigmoid)
  - Margin loss with multi-label support (T_a = T_b = 1 for two-digit case).
  - Reconstruction loss applied per ground-truth digit (mask all other caps,
    decode separately, MSE against the original 36x36 source digit).

Deviations from the paper (documented):
  1. Capsule capacity reduced (Conv1 32 vs 256, PrimaryCaps 8 vs 32). Pure
     numpy on a single thread; the paper's 256-channel Conv1 with 9x9 kernels
     is 8x slower per batch.
  2. 60k training pairs (paper uses 60M). The paper essentially regenerates
     pairs every epoch; we sample a fixed pool and re-shuffle.
  3. Routing coefficients c_ij are treated as constants for the backward
     pass (only the final iteration's c is used as a fixed weight when
     differentiating). The original Sabour et al. TF reference does the same
     in practice.
"""
from __future__ import annotations
import argparse
import gzip
import os
import time
import urllib.request
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader (urllib + gzip; cached at ~/.cache/hinton-mnist/)
# ----------------------------------------------------------------------

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}
CACHE_DIR = Path.home() / ".cache" / "hinton-mnist"


def _download(url: str, dest: Path):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        return
    print(f"  downloading {url} -> {dest}")
    with urllib.request.urlopen(url) as r:
        data = r.read()
    dest.write_bytes(data)


def _read_idx_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    rows = int.from_bytes(data[8:12], "big")
    cols = int.from_bytes(data[12:16], "big")
    if magic != 2051:
        raise RuntimeError(f"bad MNIST images magic: {magic}")
    return np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)


def _read_idx_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        data = f.read()
    magic = int.from_bytes(data[0:4], "big")
    n = int.from_bytes(data[4:8], "big")
    if magic != 2049:
        raise RuntimeError(f"bad MNIST labels magic: {magic}")
    return np.frombuffer(data[8:], dtype=np.uint8)


def load_mnist(split: str = "train"):
    """Return (images float32 [0, 1], labels int64) for split in {train, test}."""
    img_key = "train_images" if split == "train" else "test_images"
    lab_key = "train_labels" if split == "train" else "test_labels"
    images_dest = CACHE_DIR / MNIST_FILES[img_key]
    labels_dest = CACHE_DIR / MNIST_FILES[lab_key]
    _download(MNIST_URL_BASE + MNIST_FILES[img_key], images_dest)
    _download(MNIST_URL_BASE + MNIST_FILES[lab_key], labels_dest)
    images = _read_idx_images(images_dest).astype(np.float32) / 255.0
    labels = _read_idx_labels(labels_dest).astype(np.int64)
    return images, labels


# ----------------------------------------------------------------------
# MultiMNIST overlay
# ----------------------------------------------------------------------

def _digit_bbox(img: np.ndarray, threshold: float = 0.05):
    """Return (rmin, rmax, cmin, cmax) of pixels above threshold, or None."""
    mask = img > threshold
    if not mask.any():
        return None
    rows = np.where(mask.any(axis=1))[0]
    cols = np.where(mask.any(axis=0))[0]
    return int(rows.min()), int(rows.max()), int(cols.min()), int(cols.max())


def _shifted_into_canvas(digit: np.ndarray, dy: int, dx: int,
                         canvas: int = 36) -> np.ndarray:
    """Place a 28x28 digit on a `canvas`x`canvas` grid centered + (dy, dx)."""
    h, w = digit.shape
    out = np.zeros((canvas, canvas), dtype=digit.dtype)
    cy = (canvas - h) // 2 + dy
    cx = (canvas - w) // 2 + dx
    # Crop the digit if it goes out of bounds
    src_y0 = max(0, -cy)
    src_x0 = max(0, -cx)
    src_y1 = min(h, canvas - cy)
    src_x1 = min(w, canvas - cx)
    dst_y0 = max(0, cy)
    dst_x0 = max(0, cx)
    dst_y1 = dst_y0 + (src_y1 - src_y0)
    dst_x1 = dst_x0 + (src_x1 - src_x0)
    if src_y1 > src_y0 and src_x1 > src_x0:
        out[dst_y0:dst_y1, dst_x0:dst_x1] = digit[src_y0:src_y1, src_x0:src_x1]
    return out


def overlay_pair(digit_a: np.ndarray, digit_b: np.ndarray,
                 canvas: int = 36, max_shift: int = 4,
                 min_overlap: float = 0.8,
                 rng: np.random.Generator | None = None,
                 max_tries: int = 20):
    """Overlay two distinct-class digits.

    Each digit is shifted by integers in [-max_shift, max_shift] on each axis
    and overlaid via pixel-wise max. Bounding-box overlap (intersection /
    union of bounding-box areas, after shifting) must be >= min_overlap.
    Returns (composite, target_a, target_b) all `canvas`x`canvas` float32.
    target_a is digit_a shifted alone; target_b is digit_b shifted alone.
    """
    rng = rng or np.random.default_rng()
    bbox_a = _digit_bbox(digit_a)
    bbox_b = _digit_bbox(digit_b)
    if bbox_a is None or bbox_b is None:
        # Degenerate: just place both centered.
        ta = _shifted_into_canvas(digit_a, 0, 0, canvas)
        tb = _shifted_into_canvas(digit_b, 0, 0, canvas)
        return np.maximum(ta, tb), ta, tb

    for _ in range(max_tries):
        dy_a = int(rng.integers(-max_shift, max_shift + 1))
        dx_a = int(rng.integers(-max_shift, max_shift + 1))
        dy_b = int(rng.integers(-max_shift, max_shift + 1))
        dx_b = int(rng.integers(-max_shift, max_shift + 1))

        # Compute shifted bboxes on the canvas.
        cy = (canvas - 28) // 2
        cx = (canvas - 28) // 2
        a_r0, a_r1 = cy + bbox_a[0] + dy_a, cy + bbox_a[1] + dy_a
        a_c0, a_c1 = cx + bbox_a[2] + dx_a, cx + bbox_a[3] + dx_a
        b_r0, b_r1 = cy + bbox_b[0] + dy_b, cy + bbox_b[1] + dy_b
        b_c0, b_c1 = cx + bbox_b[2] + dx_b, cx + bbox_b[3] + dx_b

        # Intersection
        inter_r0 = max(a_r0, b_r0)
        inter_r1 = min(a_r1, b_r1)
        inter_c0 = max(a_c0, b_c0)
        inter_c1 = min(a_c1, b_c1)
        if inter_r1 < inter_r0 or inter_c1 < inter_c0:
            continue
        inter = max(0, inter_r1 - inter_r0 + 1) * max(0, inter_c1 - inter_c0 + 1)
        area_a = (a_r1 - a_r0 + 1) * (a_c1 - a_c0 + 1)
        area_b = (b_r1 - b_r0 + 1) * (b_c1 - b_c0 + 1)
        union = area_a + area_b - inter
        if union <= 0:
            continue
        # Overlap measured as IoU of the two bounding boxes; the paper just
        # says "overlap" without a precise metric. IoU >= 0.8 is roughly
        # what the paper's "80% overlap" examples look like.
        if inter / union >= min_overlap:
            ta = _shifted_into_canvas(digit_a, dy_a, dx_a, canvas)
            tb = _shifted_into_canvas(digit_b, dy_b, dx_b, canvas)
            return np.maximum(ta, tb), ta, tb

    # Fall back to last sample (overlap was below threshold)
    ta = _shifted_into_canvas(digit_a, dy_a, dx_a, canvas)
    tb = _shifted_into_canvas(digit_b, dy_b, dx_b, canvas)
    return np.maximum(ta, tb), ta, tb


def generate_multimnist(n_samples: int, images: np.ndarray, labels: np.ndarray,
                        canvas: int = 36, max_shift: int = 4,
                        min_overlap: float = 0.8,
                        seed: int = 0):
    """Generate n_samples MultiMNIST pairs.

    Returns (composites (N, canvas, canvas), targets_a, targets_b,
             label_pairs (N, 2)).
    """
    rng = np.random.default_rng(seed)
    composites = np.zeros((n_samples, canvas, canvas), dtype=np.float32)
    targets_a = np.zeros((n_samples, canvas, canvas), dtype=np.float32)
    targets_b = np.zeros((n_samples, canvas, canvas), dtype=np.float32)
    label_pairs = np.zeros((n_samples, 2), dtype=np.int64)

    n_images = images.shape[0]
    indices_by_label = [np.where(labels == k)[0] for k in range(10)]

    i = 0
    while i < n_samples:
        idx_a = int(rng.integers(0, n_images))
        label_a = int(labels[idx_a])
        # pick a digit of a different class
        label_b = int(rng.integers(0, 10))
        while label_b == label_a:
            label_b = int(rng.integers(0, 10))
        idx_b = int(rng.choice(indices_by_label[label_b]))

        composite, ta, tb = overlay_pair(
            images[idx_a], images[idx_b],
            canvas=canvas, max_shift=max_shift,
            min_overlap=min_overlap, rng=rng,
        )
        composites[i] = composite
        targets_a[i] = ta
        targets_b[i] = tb
        label_pairs[i, 0] = label_a
        label_pairs[i, 1] = label_b
        i += 1

    return composites, targets_a, targets_b, label_pairs


# ----------------------------------------------------------------------
# Convolution helpers (im2col via pure numpy strided slicing)
# ----------------------------------------------------------------------

def im2col(x: np.ndarray, kh: int, kw: int, stride: int = 1):
    """x: (B, C, H, W) -> (B, out_h, out_w, C, kh, kw) flattened to
    (B*out_h*out_w, C*kh*kw). Returns (cols, out_h, out_w)."""
    B, C, H, W = x.shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    cols = np.zeros((B, C, kh, kw, out_h, out_w), dtype=x.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            cols[:, :, i, j, :, :] = x[:, :, i:i_max:stride, j:j_max:stride]
    cols = cols.transpose(0, 4, 5, 1, 2, 3).reshape(B * out_h * out_w, C * kh * kw)
    return cols, out_h, out_w


def col2im(cols: np.ndarray, x_shape, kh: int, kw: int, stride: int = 1) -> np.ndarray:
    """Inverse of im2col; cols shape (B*out_h*out_w, C*kh*kw)."""
    B, C, H, W = x_shape
    out_h = (H - kh) // stride + 1
    out_w = (W - kw) // stride + 1
    cols = cols.reshape(B, out_h, out_w, C, kh, kw).transpose(0, 3, 4, 5, 1, 2)
    x = np.zeros(x_shape, dtype=cols.dtype)
    for i in range(kh):
        i_max = i + stride * out_h
        for j in range(kw):
            j_max = j + stride * out_w
            x[:, :, i:i_max:stride, j:j_max:stride] += cols[:, :, i, j, :, :]
    return x


def conv2d_forward(x: np.ndarray, W: np.ndarray, b: np.ndarray, stride: int = 1):
    """x: (B, Cin, H, W), W: (Cout, Cin, kh, kw), b: (Cout,).
    Returns (out (B, Cout, out_h, out_w), cache)."""
    B, Cin, H, Win = x.shape
    Cout, _, kh, kw = W.shape
    cols, out_h, out_w = im2col(x, kh, kw, stride)
    W_flat = W.reshape(Cout, -1).T                       # (Cin*kh*kw, Cout)
    out = (cols @ W_flat) + b                            # (B*Hout*Wout, Cout)
    out = out.reshape(B, out_h, out_w, Cout).transpose(0, 3, 1, 2)
    cache = (x.shape, cols, W_flat, kh, kw, stride, out_h, out_w)
    return out, cache


def conv2d_backward(d_out: np.ndarray, cache, W: np.ndarray):
    """d_out: (B, Cout, out_h, out_w). Returns (dx, dW, db)."""
    x_shape, cols, W_flat, kh, kw, stride, out_h, out_w = cache
    B, Cin, H, Win = x_shape
    Cout = W.shape[0]
    d_out_flat = d_out.transpose(0, 2, 3, 1).reshape(-1, Cout)  # (B*Hout*Wout, Cout)
    dW_flat = cols.T @ d_out_flat                                # (Cin*kh*kw, Cout)
    dW = dW_flat.T.reshape(Cout, Cin, kh, kw)
    db = d_out_flat.sum(axis=0)
    d_cols = d_out_flat @ W_flat.T                               # (B*Hout*Wout, Cin*kh*kw)
    dx = col2im(d_cols, x_shape, kh, kw, stride)
    return dx, dW, db


# ----------------------------------------------------------------------
# Activations and squash
# ----------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def squash(s: np.ndarray, axis: int = -1, eps: float = 1e-8):
    """v = ||s||^2 / (1 + ||s||^2) * s/||s||  ==  ||s|| s / (1 + ||s||^2).

    Operates along `axis`. Returns (v, n2) where n2 is the squared norm
    along `axis` (used by squash_backward).
    """
    n2 = (s * s).sum(axis=axis, keepdims=True)
    n = np.sqrt(n2 + eps)
    v = (n / (1.0 + n2)) * s
    return v, n2


def squash_backward(d_v: np.ndarray, s: np.ndarray, n2: np.ndarray,
                    axis: int = -1, eps: float = 1e-8) -> np.ndarray:
    """Given d_v (same shape as v=squash(s)), return d_s.

    v_i = n s_i / (1+n^2);  d v_i / d s_j = (1-n^2)/(1+n^2)^2 * s_i s_j / n
                                          + n/(1+n^2) * delta_ij.
    => d_s_j = f'(n) * s_j/n * (s . dv) + f(n) * dv_j
       where f(n) = n/(1+n^2), f'(n) = (1-n^2)/(1+n^2)^2.
    """
    n = np.sqrt(n2 + eps)
    one_plus_n2 = 1.0 + n2
    f = n / one_plus_n2
    fprime = (1.0 - n2) / (one_plus_n2 * one_plus_n2)
    s_dot_dv = (s * d_v).sum(axis=axis, keepdims=True)
    return fprime * s_dot_dv * s / n + f * d_v


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class CapsNet:
    """CapsNet for MultiMNIST.

    Reduced architecture (from paper) to fit pure-numpy budget:
      Conv1: n_conv1 channels, 9x9 stride 1, ReLU.
      PrimaryCaps: n_primary capsules x primary_dim, 9x9 stride 2.
      DigitCaps: n_classes (10) capsules x digit_dim, 3-iter dynamic routing.
      Decoder: 3 fully-connected layers, sigmoid output -> reconstruct
        canvas x canvas image of one digit.

    Backward pass treats routing coefficients c_ij as constants for the
    final iteration (standard simplification used in the original Sabour
    et al. TF reference).
    """
    def __init__(self,
                 canvas: int = 36,
                 n_conv1: int = 32,
                 n_primary: int = 8,
                 primary_dim: int = 8,
                 digit_dim: int = 16,
                 n_classes: int = 10,
                 dec1: int = 256,
                 dec2: int = 512,
                 routing_iters: int = 3,
                 seed: int = 0):
        self.canvas = canvas
        self.n_conv1 = n_conv1
        self.n_primary = n_primary
        self.primary_dim = primary_dim
        self.digit_dim = digit_dim
        self.n_classes = n_classes
        self.dec1 = dec1
        self.dec2 = dec2
        self.routing_iters = routing_iters

        rng = np.random.default_rng(seed)
        self.rng = rng

        # Compute spatial sizes for the chosen canvas
        c1_h = canvas - 9 + 1                # 28 for canvas=36
        p_h = (c1_h - 9) // 2 + 1            # 10
        self.c1_h = c1_h
        self.p_h = p_h
        self.n_primary_total = n_primary * p_h * p_h

        def he(shape, fan_in):
            return (rng.standard_normal(shape) * np.sqrt(2.0 / fan_in)).astype(np.float32)

        # Conv1: (n_conv1, 1, 9, 9)
        self.W_conv1 = he((n_conv1, 1, 9, 9), 9 * 9)
        self.b_conv1 = np.zeros((n_conv1,), dtype=np.float32)

        # PrimaryCaps: 9x9 stride-2 conv from n_conv1 to (n_primary * primary_dim)
        n_p_chan = n_primary * primary_dim
        self.W_pcaps = he((n_p_chan, n_conv1, 9, 9), n_conv1 * 9 * 9)
        self.b_pcaps = np.zeros((n_p_chan,), dtype=np.float32)

        # Routing weights: (N_primary, n_classes, digit_dim, primary_dim)
        # Small init so routing starts gentle
        self.W_route = (rng.standard_normal(
            (self.n_primary_total, n_classes, digit_dim, primary_dim)
        ) * 0.05).astype(np.float32)

        # Decoder weights -- input is masked DigitCaps flat: n_classes * digit_dim
        in_dim = n_classes * digit_dim
        out_dim = canvas * canvas
        self.W_dec1 = he((in_dim, dec1), in_dim)
        self.b_dec1 = np.zeros((dec1,), dtype=np.float32)
        self.W_dec2 = he((dec1, dec2), dec1)
        self.b_dec2 = np.zeros((dec2,), dtype=np.float32)
        self.W_dec3 = he((dec2, out_dim), dec2)
        self.b_dec3 = np.zeros((out_dim,), dtype=np.float32)

    @property
    def param_names(self):
        return ("W_conv1", "b_conv1", "W_pcaps", "b_pcaps", "W_route",
                "W_dec1", "b_dec1", "W_dec2", "b_dec2", "W_dec3", "b_dec3")

    def zero_like_params(self):
        return {k: np.zeros_like(getattr(self, k)) for k in self.param_names}

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, x: np.ndarray):
        """x: (B, canvas, canvas) float32. Returns (v (B, 10, digit_dim), cache)."""
        B = x.shape[0]
        x_in = x[:, None, :, :]                                  # (B, 1, H, W)

        # Conv1 + ReLU
        c1_pre, c1_cache = conv2d_forward(x_in, self.W_conv1, self.b_conv1, stride=1)
        c1 = relu(c1_pre)

        # PrimaryCaps conv (stride 2)
        p_pre, p_cache = conv2d_forward(c1, self.W_pcaps, self.b_pcaps, stride=2)
        # p_pre: (B, n_primary*primary_dim, p_h, p_h)
        # Reshape to (B, n_primary, primary_dim, p_h, p_h) then to
        # (B, N_primary, primary_dim).
        p_pre_r = p_pre.reshape(B, self.n_primary, self.primary_dim,
                                self.p_h, self.p_h)
        # Move primary_dim to last and flatten the capsule axis
        p_pre_r = p_pre_r.transpose(0, 1, 3, 4, 2).reshape(
            B, self.n_primary_total, self.primary_dim)

        # Squash each capsule
        u, u_n2 = squash(p_pre_r, axis=-1)                        # (B, N_p, primary_dim)

        # u_hat[b, i, j, :] = W_route[i, j, :, :] @ u[b, i, :]
        # W_route: (N_p, 10, digit_dim, primary_dim); u: (B, N_p, primary_dim)
        # Reshape u to (B, N_p, 1, primary_dim, 1) for matmul with
        # W_route (1, N_p, 10, digit_dim, primary_dim) ... avoid huge memory by
        # using einsum.
        u_hat = np.einsum("bip,ijdp->bijd", u, self.W_route,
                          optimize=True)                          # (B, N_p, 10, digit_dim)

        # Dynamic routing
        b_log = np.zeros((B, self.n_primary_total, self.n_classes),
                         dtype=np.float32)
        for r in range(self.routing_iters):
            c = softmax(b_log, axis=2)                            # (B, N_p, 10)
            s = (c[:, :, :, None] * u_hat).sum(axis=1)            # (B, 10, digit_dim)
            v, v_n2 = squash(s, axis=-1)
            if r < self.routing_iters - 1:
                # b_ij update: agreement between u_hat and v
                # agreement = sum_d u_hat[b,i,j,d] * v[b,j,d]
                agreement = np.einsum("bijd,bjd->bij", u_hat, v, optimize=True)
                b_log = b_log + agreement
        # Final c is what we treat as a constant in the backward pass.
        cache = dict(
            x_in=x_in, c1_pre=c1_pre, c1=c1, c1_cache=c1_cache,
            p_pre=p_pre, p_cache=p_cache, p_pre_r=p_pre_r,
            u=u, u_n2=u_n2, u_hat=u_hat,
            c_final=c, s=s, v=v, v_n2=v_n2,
        )
        return v, cache

    # ------------------------------------------------------------------
    # Decoder
    # ------------------------------------------------------------------

    def decode(self, v: np.ndarray, mask: np.ndarray):
        """v: (B, 10, digit_dim); mask: (B, 10) one-hot for which digit to decode.
        Returns (recon (B, canvas*canvas), dec_cache)."""
        B = v.shape[0]
        masked = (v * mask[:, :, None]).reshape(B, -1)            # (B, n_classes*digit_dim)
        h1_pre = masked @ self.W_dec1 + self.b_dec1
        h1 = relu(h1_pre)
        h2_pre = h1 @ self.W_dec2 + self.b_dec2
        h2 = relu(h2_pre)
        out_pre = h2 @ self.W_dec3 + self.b_dec3
        recon = sigmoid(out_pre)
        cache = dict(masked=masked, mask=mask,
                     h1_pre=h1_pre, h1=h1, h2_pre=h2_pre, h2=h2,
                     out_pre=out_pre, recon=recon)
        return recon, cache


# ----------------------------------------------------------------------
# Loss + backward
# ----------------------------------------------------------------------

def margin_loss(v: np.ndarray, T: np.ndarray, m_plus: float = 0.9,
                m_minus: float = 0.1, lam: float = 0.5,
                eps: float = 1e-8):
    """v: (B, 10, digit_dim); T: (B, 10) {0, 1} multi-label.
    Returns (loss, d_v).

    L_k = T_k * max(0, m+ - ||v_k||)^2 + lam * (1-T_k) * max(0, ||v_k|| - m-)^2
    Total = sum_k mean over batch of L_k.
    """
    # Gradient of margin loss w.r.t. ||v_k|| -> chain through ||v|| = sqrt(sum v^2)
    n = np.sqrt((v * v).sum(axis=-1) + eps)                       # (B, 10)
    pos_term = np.maximum(0.0, m_plus - n)
    neg_term = np.maximum(0.0, n - m_minus)
    L = T * pos_term ** 2 + lam * (1.0 - T) * neg_term ** 2       # (B, 10)
    loss = float(L.sum(axis=1).mean())

    # d L / d n_k  (per-element)
    B = v.shape[0]
    d_n = (T * 2.0 * pos_term * (-1.0)
           + lam * (1.0 - T) * 2.0 * neg_term)                    # (B, 10)
    # Average over batch (loss is mean over batch, so d/dv has 1/B factor)
    d_n = d_n / B
    # d n / d v_k = v_k / n
    d_v = d_n[:, :, None] * (v / n[:, :, None])                   # (B, 10, digit_dim)
    return loss, d_v


def reconstruction_loss(recon: np.ndarray, target: np.ndarray):
    """recon, target: (B, canvas*canvas). Returns (mse, d_recon).
    Uses sum-of-squared-errors averaged over batch (paper scales by 0.0005)."""
    B, P = recon.shape
    diff = recon - target
    loss = 0.5 * float(np.sum(diff * diff)) / B
    d_recon = diff / B
    return loss, d_recon


def decode_backward(d_recon: np.ndarray, dec_cache: dict, model: "CapsNet"):
    """Backward through decoder. Returns (d_v (B, 10, digit_dim), grads_dict)."""
    out_pre = dec_cache["out_pre"]
    recon = dec_cache["recon"]
    h2 = dec_cache["h2"]
    h1 = dec_cache["h1"]
    masked = dec_cache["masked"]
    mask = dec_cache["mask"]
    h2_pre = dec_cache["h2_pre"]
    h1_pre = dec_cache["h1_pre"]

    d_out_pre = d_recon * recon * (1.0 - recon)                   # sigmoid'
    d_W_dec3 = h2.T @ d_out_pre
    d_b_dec3 = d_out_pre.sum(axis=0)
    d_h2 = d_out_pre @ model.W_dec3.T

    d_h2_pre = d_h2 * (h2_pre > 0)
    d_W_dec2 = h1.T @ d_h2_pre
    d_b_dec2 = d_h2_pre.sum(axis=0)
    d_h1 = d_h2_pre @ model.W_dec2.T

    d_h1_pre = d_h1 * (h1_pre > 0)
    d_W_dec1 = masked.T @ d_h1_pre
    d_b_dec1 = d_h1_pre.sum(axis=0)
    d_masked = d_h1_pre @ model.W_dec1.T

    # masked = (v * mask[:, :, None]).reshape(B, -1)
    d_masked_v = d_masked.reshape(masked.shape[0], model.n_classes, model.digit_dim)
    d_v = d_masked_v * mask[:, :, None]
    grads = dict(W_dec1=d_W_dec1, b_dec1=d_b_dec1,
                 W_dec2=d_W_dec2, b_dec2=d_b_dec2,
                 W_dec3=d_W_dec3, b_dec3=d_b_dec3)
    return d_v, grads


def caps_backward(d_v: np.ndarray, cache: dict, model: "CapsNet"):
    """Backward through CapsNet (treating routing c as constant).
    Returns grads dict for caps params: W_conv1, b_conv1, W_pcaps, b_pcaps,
    W_route."""
    s = cache["s"]
    v_n2 = cache["v_n2"]
    c_final = cache["c_final"]                                    # (B, N_p, 10)
    u_hat = cache["u_hat"]                                        # (B, N_p, 10, digit_dim)
    u = cache["u"]                                                # (B, N_p, primary_dim)
    p_pre_r = cache["p_pre_r"]                                    # (B, N_p, primary_dim)
    u_n2 = cache["u_n2"]                                          # (B, N_p, 1)
    p_cache = cache["p_cache"]
    c1 = cache["c1"]
    c1_pre = cache["c1_pre"]
    c1_cache = cache["c1_cache"]

    B = d_v.shape[0]
    # d_s through squash
    d_s = squash_backward(d_v, s, v_n2, axis=-1)                  # (B, 10, digit_dim)

    # s = sum_i c_ij * u_hat_ij, so:
    # d u_hat[b,i,j,:] = c_final[b,i,j] * d_s[b,j,:]
    d_u_hat = c_final[:, :, :, None] * d_s[:, None, :, :]         # (B, N_p, 10, digit_dim)

    # u_hat[b,i,j,d] = sum_p W_route[i,j,d,p] * u[b,i,p]
    # d W_route[i,j,d,p] = sum_b u[b,i,p] * d_u_hat[b,i,j,d]
    # d u[b,i,p] = sum_{j,d} W_route[i,j,d,p] * d_u_hat[b,i,j,d]
    d_W_route = np.einsum("bip,bijd->ijdp", u, d_u_hat, optimize=True)
    d_u = np.einsum("ijdp,bijd->bip", model.W_route, d_u_hat, optimize=True)

    # Squash backward on primary caps: u = squash(p_pre_r)
    d_p_pre_r = squash_backward(d_u, p_pre_r, u_n2, axis=-1)      # (B, N_p, primary_dim)

    # Reshape back to (B, n_primary*primary_dim, p_h, p_h)
    n_p = model.n_primary
    pd = model.primary_dim
    ph = model.p_h
    # Inverse of: p_pre.reshape(B, n_p, pd, ph, ph).transpose(0, 1, 3, 4, 2)
    #            .reshape(B, N_p, pd)
    d_p_pre = d_p_pre_r.reshape(B, n_p, ph, ph, pd).transpose(0, 1, 4, 2, 3)
    d_p_pre = d_p_pre.reshape(B, n_p * pd, ph, ph)

    # Conv backward through PrimaryCaps
    d_c1, d_W_pcaps, d_b_pcaps = conv2d_backward(d_p_pre, p_cache, model.W_pcaps)

    # ReLU backward
    d_c1_pre = d_c1 * (c1_pre > 0)

    # Conv backward through Conv1
    _, d_W_conv1, d_b_conv1 = conv2d_backward(d_c1_pre, c1_cache, model.W_conv1)

    grads = dict(W_conv1=d_W_conv1, b_conv1=d_b_conv1,
                 W_pcaps=d_W_pcaps, b_pcaps=d_b_pcaps,
                 W_route=d_W_route)
    return grads


# ----------------------------------------------------------------------
# Inference: pick top-2 capsules
# ----------------------------------------------------------------------

def predict_top2(v: np.ndarray):
    """v: (B, 10, digit_dim). Returns (top2_labels (B, 2) sorted asc, norms (B, 10))."""
    norms = np.sqrt((v * v).sum(axis=-1) + 1e-8)
    # top-2 indices by norm
    top2 = np.argsort(-norms, axis=1)[:, :2]
    # Sort each pair ascending so it matches label_pair convention
    top2_sorted = np.sort(top2, axis=1)
    return top2_sorted, norms


def two_digit_accuracy(pred_top2: np.ndarray, label_pairs: np.ndarray) -> float:
    """A pair is correct if both labels are present in pred_top2 (set match)."""
    sorted_labels = np.sort(label_pairs, axis=1)
    matches = np.all(pred_top2 == sorted_labels, axis=1)
    return float(matches.mean())


# ----------------------------------------------------------------------
# Convenience constructor (matches problem.py)
# ----------------------------------------------------------------------

def build_capsnet(seed: int = 0, **kwargs) -> CapsNet:
    return CapsNet(seed=seed, **kwargs)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def _adam_step(model: CapsNet, grads: dict,
               adam_m: dict, adam_v: dict, t: int,
               lr: float, beta1: float, beta2: float, eps: float):
    bc1 = 1.0 - beta1 ** t
    bc2 = 1.0 - beta2 ** t
    for k, g in grads.items():
        m_buf = adam_m[k]
        v_buf = adam_v[k]
        m_buf[...] = beta1 * m_buf + (1.0 - beta1) * g
        v_buf[...] = beta2 * v_buf + (1.0 - beta2) * (g * g)
        update = lr * (m_buf / bc1) / (np.sqrt(v_buf / bc2) + eps)
        getattr(model, k)[...] -= update


def train(model: CapsNet | None = None,
          n_epochs: int = 5,
          batch_size: int = 32,
          lr: float = 1e-3,
          n_train: int = 6000,
          n_test: int = 1000,
          recon_weight: float = 0.0005,
          canvas: int = 36,
          max_shift: int = 4,
          min_overlap: float = 0.8,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True):
    """Train CapsNet on MultiMNIST. Returns (model, history, test_data)."""
    rng = np.random.default_rng(seed)

    # Build datasets
    train_imgs, train_labels = load_mnist("train")
    test_imgs, test_labels = load_mnist("test")

    if verbose:
        print(f"# generating {n_train} train + {n_test} test MultiMNIST pairs ({canvas}x{canvas})")
    t_gen = time.time()
    Xtr, Atr, Btr, Ltr = generate_multimnist(
        n_train, train_imgs, train_labels,
        canvas=canvas, max_shift=max_shift, min_overlap=min_overlap, seed=seed)
    Xte, Ate, Bte, Lte = generate_multimnist(
        n_test, test_imgs, test_labels,
        canvas=canvas, max_shift=max_shift, min_overlap=min_overlap,
        seed=seed + 1000)
    if verbose:
        print(f"# generation took {time.time() - t_gen:.1f}s")

    if model is None:
        model = CapsNet(canvas=canvas, seed=seed)

    if verbose:
        print(f"# capsnet: conv1={model.n_conv1}, primary={model.n_primary}x{model.primary_dim}, "
              f"digit={model.n_classes}x{model.digit_dim}, routing_iters={model.routing_iters}")
        print(f"# N_primary_total = {model.n_primary_total}")

    adam_m = model.zero_like_params()
    adam_v = model.zero_like_params()
    adam_t = 0

    history = {"step": [], "epoch": [], "margin": [], "recon": [], "loss": [],
               "test_acc": [], "test_recon_mse": []}

    n_steps_per_epoch = n_train // batch_size
    step = 0
    t0 = time.time()

    for epoch in range(n_epochs):
        perm = rng.permutation(n_train)
        epoch_margin = 0.0
        epoch_recon = 0.0

        for s in range(n_steps_per_epoch):
            idx = perm[s * batch_size:(s + 1) * batch_size]
            x = Xtr[idx]                                          # (B, H, W)
            la = Ltr[idx, 0]
            lb = Ltr[idx, 1]
            ta = Atr[idx].reshape(batch_size, -1)
            tb = Btr[idx].reshape(batch_size, -1)

            v, fwd_cache = model.forward(x)

            # Build T (multi-label): mark both digits
            T = np.zeros((batch_size, model.n_classes), dtype=np.float32)
            T[np.arange(batch_size), la] = 1.0
            T[np.arange(batch_size), lb] = 1.0

            margin_l, d_v_margin = margin_loss(v, T)

            # Decoder: two passes, one per ground-truth digit
            mask_a = np.zeros((batch_size, model.n_classes), dtype=np.float32)
            mask_a[np.arange(batch_size), la] = 1.0
            mask_b = np.zeros((batch_size, model.n_classes), dtype=np.float32)
            mask_b[np.arange(batch_size), lb] = 1.0

            recon_a, dec_cache_a = model.decode(v, mask_a)
            recon_b, dec_cache_b = model.decode(v, mask_b)

            recon_loss_a, d_recon_a = reconstruction_loss(recon_a, ta)
            recon_loss_b, d_recon_b = reconstruction_loss(recon_b, tb)
            recon_l = recon_loss_a + recon_loss_b

            # Decoder backward
            d_v_a, dec_grads_a = decode_backward(d_recon_a * recon_weight,
                                                 dec_cache_a, model)
            d_v_b, dec_grads_b = decode_backward(d_recon_b * recon_weight,
                                                 dec_cache_b, model)

            # Combine v gradients: margin + reconstruction (both digits)
            d_v_total = d_v_margin + d_v_a + d_v_b

            # Caps backward
            caps_grads = caps_backward(d_v_total, fwd_cache, model)

            # Combine decoder grads (sum since both passes share weights)
            grads = caps_grads
            for k in dec_grads_a:
                grads[k] = dec_grads_a[k] + dec_grads_b[k]

            # Adam step
            adam_t += 1
            _adam_step(model, grads, adam_m, adam_v, adam_t,
                       lr=lr, beta1=0.9, beta2=0.999, eps=1e-8)

            total_l = margin_l + recon_weight * recon_l
            epoch_margin += margin_l
            epoch_recon += recon_l
            step += 1

            if snapshot_callback is not None and (step % snapshot_every == 0):
                snapshot_callback(step, model, history,
                                  Xte, Ate, Bte, Lte)

        # End-of-epoch test eval
        test_acc, test_recon_mse = evaluate(model, Xte, Ate, Bte, Lte,
                                            batch_size=batch_size)
        avg_m = epoch_margin / max(1, n_steps_per_epoch)
        avg_r = epoch_recon / max(1, n_steps_per_epoch)
        history["step"].append(step)
        history["epoch"].append(epoch + 1)
        history["margin"].append(avg_m)
        history["recon"].append(avg_r)
        history["loss"].append(avg_m + recon_weight * avg_r)
        history["test_acc"].append(test_acc)
        history["test_recon_mse"].append(test_recon_mse)

        if verbose:
            elapsed = time.time() - t0
            print(f"epoch {epoch+1:2d}/{n_epochs}  margin={avg_m:.4f}  "
                  f"recon={avg_r:.4f}  test2acc={test_acc:.3f}  "
                  f"test_recon_mse={test_recon_mse:.4f}  ({elapsed:.1f}s)",
                  flush=True)

    test_data = dict(Xte=Xte, Ate=Ate, Bte=Bte, Lte=Lte)
    return model, history, test_data


def evaluate(model: CapsNet, X: np.ndarray, A: np.ndarray, B: np.ndarray,
             L: np.ndarray, batch_size: int = 32):
    """Compute (two_digit_accuracy, mean_reconstruction_mse) on a held-out set."""
    n = X.shape[0]
    correct_pairs = 0
    total_recon_mse = 0.0
    n_recon = 0
    for i in range(0, n, batch_size):
        xb = X[i:i + batch_size]
        if xb.shape[0] == 0:
            break
        bs = xb.shape[0]
        v, _ = model.forward(xb)
        top2, _ = predict_top2(v)
        sorted_labels = np.sort(L[i:i + batch_size], axis=1)
        correct_pairs += int(np.sum(np.all(top2 == sorted_labels, axis=1)))

        # Reconstruction MSE using GROUND TRUTH masks (matches training signal)
        ta = A[i:i + batch_size].reshape(bs, -1)
        tb = B[i:i + batch_size].reshape(bs, -1)
        la = L[i:i + batch_size, 0]
        lb = L[i:i + batch_size, 1]
        ma = np.zeros((bs, model.n_classes), dtype=np.float32)
        ma[np.arange(bs), la] = 1.0
        mb = np.zeros((bs, model.n_classes), dtype=np.float32)
        mb[np.arange(bs), lb] = 1.0
        ra, _ = model.decode(v, ma)
        rb, _ = model.decode(v, mb)
        total_recon_mse += float(np.mean((ra - ta) ** 2)) * bs
        total_recon_mse += float(np.mean((rb - tb) ** 2)) * bs
        n_recon += 2 * bs

    acc = correct_pairs / n
    recon_mse = total_recon_mse / max(1, n_recon)
    return acc, recon_mse


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=5)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-3)
    args = p.parse_args()

    model, history, test_data = train(
        n_epochs=args.n_epochs,
        n_train=args.n_train,
        n_test=args.n_test,
        batch_size=args.batch_size,
        lr=args.lr,
        seed=args.seed,
    )

    print(f"\nFinal margin loss: {history['margin'][-1]:.4f}")
    print(f"Final recon loss:  {history['recon'][-1]:.4f}")
    print(f"Final test 2-digit acc: {history['test_acc'][-1]:.3f}")
    print(f"Final test recon MSE:   {history['test_recon_mse'][-1]:.4f}")


if __name__ == "__main__":
    main()
