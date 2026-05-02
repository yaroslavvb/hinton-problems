"""
Transforming auto-encoders -- the seminal capsules paper.
Reproduction of Hinton, Krizhevsky & Wang, "Transforming auto-encoders",
ICANN 2011.

Translation-only variant: each "capsule" learns to output (presence_prob, x, y)
for its entity. The transformation enters by adding (dx, dy) to (x, y); the
generative net then produces a 22x22 patch from the transformed coordinates;
the reconstruction is the sum across capsules weighted by presence.

The architecture is the simplest possible disentanglement test: the only way
the (dx, dy) signal can flow through the network is by literally being added
to the (x, y) instantiation parameters. So if reconstruction succeeds, the
network must have learned a (x, y) representation in pixel-equivalent units.

Deviations from the 2011 paper:
  - Translation only, not full 2D affine. The paper covers translation, scaling,
    and full 2D-3D affine. This implementation is the translation case only.
  - Recognition layer width chosen for fast iteration (n_rec_hidden=20 vs the
    paper's larger nets).
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
# MNIST loader  (urllib + gzip; cached at ~/.cache/hinton-mnist/)
# ----------------------------------------------------------------------

MNIST_URL_BASE = "https://storage.googleapis.com/cvdf-datasets/mnist/"
MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
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
        raise RuntimeError(f"bad MNIST magic: {magic}")
    arr = np.frombuffer(data[16:], dtype=np.uint8).reshape(n, rows, cols)
    return arr


def load_mnist(split: str = "train") -> np.ndarray:
    """Return MNIST images as float32 in [0, 1], shape (N, 28, 28)."""
    key = "train_images" if split == "train" else "test_images"
    fname = MNIST_FILES[key]
    dest = CACHE_DIR / fname
    _download(MNIST_URL_BASE + fname, dest)
    arr = _read_idx_images(dest).astype(np.float32) / 255.0
    return arr


# ----------------------------------------------------------------------
# Translation
# ----------------------------------------------------------------------

def translate(img: np.ndarray, dx: int, dy: int) -> np.ndarray:
    """Shift `img` by integer (dx, dy). dx > 0 shifts right; dy > 0 shifts down.

    Pixels shifted off the canvas are dropped; pixels shifted in are zero.
    Works on 2D arrays or any leading batch dims.
    """
    out = np.zeros_like(img)
    h, w = img.shape[-2], img.shape[-1]
    src_x0, dst_x0 = max(0, -dx), max(0, dx)
    src_y0, dst_y0 = max(0, -dy), max(0, dy)
    span_x = w - abs(dx)
    span_y = h - abs(dy)
    if span_x <= 0 or span_y <= 0:
        return out
    out[..., dst_y0:dst_y0 + span_y, dst_x0:dst_x0 + span_x] = \
        img[..., src_y0:src_y0 + span_y, src_x0:src_x0 + span_x]
    return out


def make_transformed_pair(image: np.ndarray, max_shift: int = 5,
                          input_jitter: int = 5,
                          rng: np.random.Generator | None = None):
    """Return (image1, image2, dx_dy) where:

    - image1 is `image` translated by a random `t_in` in +/- input_jitter
    - image2 is `image` translated by `t_in + dx_dy`
    - dx_dy is sampled in +/- max_shift

    Without input_jitter, MNIST digits are always centered and the recognition
    (x, y) outputs have nothing to encode — they collapse to constants and the
    disentanglement test trivially fails. Sampling a random input position
    forces (x, y) to track the entity location.
    """
    rng = rng or np.random.default_rng()
    tx = int(rng.integers(-input_jitter, input_jitter + 1))
    ty = int(rng.integers(-input_jitter, input_jitter + 1))
    dx = int(rng.integers(-max_shift, max_shift + 1))
    dy = int(rng.integers(-max_shift, max_shift + 1))
    image1 = translate(image, tx, ty)
    image2 = translate(image, tx + dx, ty + dy)
    return image1, image2, np.array([dx, dy], dtype=np.float32)


def crop_center(img: np.ndarray, size: int = 22) -> np.ndarray:
    """Return the centered `size`x`size` crop of a 28x28 image (or batch)."""
    off = (img.shape[-1] - size) // 2  # = 3 for 28 -> 22
    return img[..., off:off + size, off:off + size]


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


# ----------------------------------------------------------------------
# Batched per-capsule linear ops
# ----------------------------------------------------------------------
#
# All "per-capsule" weight tensors have a leading C dim. The straightforward
# np.einsum versions ('bch,chp->bcp' etc.) bypass BLAS for these mixed-axis
# contractions and run ~100x slower than batched matmul. The helpers below
# do the equivalent contraction via np.matmul, which dispatches into BLAS.
# Verified numerically equivalent (atol=1e-3 in the smoke tests).

def _bi_cih_to_bch(x: np.ndarray, W: np.ndarray) -> np.ndarray:
    """einsum('bi,cih->bch', x, W) via batched matmul."""
    # x: (B, I); W: (C, I, H). Broadcast x as (1, B, I) then matmul -> (C, B, H).
    return np.matmul(x[None, :, :], W).transpose(1, 0, 2)


def _bch_chk_to_bck(a: np.ndarray, W: np.ndarray) -> np.ndarray:
    """einsum('bch,chk->bck', a, W) via batched matmul."""
    return np.matmul(a.transpose(1, 0, 2), W).transpose(1, 0, 2)


def _per_cap_weight_grad(a: np.ndarray, d_out: np.ndarray) -> np.ndarray:
    """Compute dW[c,i,j] = sum_b a[b,c,i] * d_out[b,c,j].

    a:     (B, C, I)
    d_out: (B, C, J)
    return: (C, I, J)  -- equivalent to einsum('bci,bcj->cij', a, d_out).
    """
    return np.matmul(a.transpose(1, 2, 0), d_out.transpose(1, 0, 2))


def _per_cap_input_grad(d_out: np.ndarray, W: np.ndarray) -> np.ndarray:
    """Compute dIn[b,c,i] = sum_j W[c,i,j] * d_out[b,c,j].

    d_out: (B, C, J)
    W:     (C, I, J)
    return: (B, C, I)  -- equivalent to einsum('cij,bcj->bci', W, d_out).
    """
    return np.matmul(d_out.transpose(1, 0, 2), W.transpose(0, 2, 1)).transpose(1, 0, 2)


def _W_rec_grad(x: np.ndarray, d_r_pre: np.ndarray) -> np.ndarray:
    """Compute dW[c,i,h] = sum_b x[b,i] * d_r_pre[b,c,h].

    x:       (B, I)
    d_r_pre: (B, C, H)
    return:  (C, I, H)
    """
    # (1, I, B) @ (C, B, H) -> broadcast on first axis -> (C, I, H)
    return np.matmul(x.T[None, :, :], d_r_pre.transpose(1, 0, 2))


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class TransformingAutoencoder:
    """
    30 capsules. Per-capsule pipeline:

      x (28x28 = 784)
        -> recognition hidden  (sigmoid, n_rec_hidden=20)
        -> instantiation       (3 outputs: p, x, y; p sigmoid, x/y linear)
      add (dx, dy) to (x, y)
        -> generative hidden   (ReLU, n_gen_hidden=128)
        -> generative output   (sigmoid, 22*22 = 484)

    The full reconstruction is sum_c p_c * patch_c. Loss is MSE against the
    centered 22x22 crop of the transformed image.

    Per-capsule weights are stacked so that all capsules run with one
    batched einsum (no Python-level capsule loop).
    """

    def __init__(self,
                 n_capsules: int = 30,
                 n_rec_hidden: int = 20,
                 n_gen_hidden: int = 128,
                 patch_size: int = 22,
                 image_size: int = 28,
                 init_scale: float = 1.0,
                 seed: int = 0):
        self.n_capsules = n_capsules
        self.n_rec_hidden = n_rec_hidden
        self.n_gen_hidden = n_gen_hidden
        self.patch_size = patch_size
        self.image_size = image_size
        self.patch_dim = patch_size * patch_size

        self.rng = np.random.default_rng(seed)
        c, h, g = n_capsules, n_rec_hidden, n_gen_hidden
        ip = image_size * image_size
        pp = patch_size * patch_size

        def he(shape, fan_in):
            return (init_scale * self.rng.standard_normal(shape)
                    * np.sqrt(2.0 / fan_in)).astype(np.float32)

        # Recognition: per-capsule (input -> n_rec_hidden), sigmoid hidden
        self.W_rec = he((c, ip, h), ip)
        self.b_rec = np.zeros((c, h), dtype=np.float32)

        # Instantiation: per-capsule (n_rec_hidden -> 3) (p, x, y)
        self.W_inst = he((c, h, 3), h)
        self.b_inst = np.zeros((c, 3), dtype=np.float32)
        self.b_inst[:, 0] = -1.0  # so initial sigmoid(p) ~ 0.27 (sparser presence)

        # Generative hidden: per-capsule (2 -> n_gen_hidden), ReLU
        self.W_gen1 = he((c, 2, g), 2)
        self.b_gen1 = np.zeros((c, g), dtype=np.float32)

        # Generative output: per-capsule (n_gen_hidden -> 484), sigmoid
        self.W_gen2 = he((c, g, pp), g)
        self.b_gen2 = np.zeros((c, pp), dtype=np.float32)

    @property
    def param_names(self):
        return ("W_rec", "b_rec", "W_inst", "b_inst",
                "W_gen1", "b_gen1", "W_gen2", "b_gen2")

    def zero_like_params(self):
        return {k: np.zeros_like(getattr(self, k)) for k in self.param_names}

    # -- forward -----------------------------------------------------------

    def _recognize(self, x: np.ndarray):
        """Return (p, xy) = ((B, C), (B, C, 2)) with sigmoid presence and linear xy."""
        r_pre = _bi_cih_to_bch(x, self.W_rec) + self.b_rec       # (B, C, H_rec)
        r = sigmoid(r_pre)
        o = _bch_chk_to_bck(r, self.W_inst) + self.b_inst        # (B, C, 3)
        p_pre = o[:, :, 0]
        xy = o[:, :, 1:3]
        return sigmoid(p_pre), xy, dict(r_pre=r_pre, r=r, o=o, p_pre=p_pre, xy=xy)

    def forward(self, x: np.ndarray, dxdy: np.ndarray):
        """
        x: (B, 784) float32 image input.
        dxdy: (B, 2) float32 pixel translation (dx, dy).
        Returns (recon (B, 484), cache for backward).
        """
        p, xy, rec_cache = self._recognize(x)
        xy_t = xy + dxdy[:, None, :]                              # (B, C, 2)
        h_pre = _bch_chk_to_bck(xy_t, self.W_gen1) + self.b_gen1  # (B, C, G)
        h = relu(h_pre)
        g_pre = _bch_chk_to_bck(h, self.W_gen2) + self.b_gen2     # (B, C, P)
        g = sigmoid(g_pre)
        recon = np.einsum("bc,bcp->bp", p, g)                     # (B, P)
        cache = dict(x=x, dxdy=dxdy, p=p,
                     xy_t=xy_t, h_pre=h_pre, h=h, g_pre=g_pre, g=g, recon=recon,
                     **rec_cache)
        return recon, cache

    # -- backward ----------------------------------------------------------

    def backward(self, cache: dict, target: np.ndarray):
        """target: (B, 484). Returns (loss, grads)."""
        recon = cache["recon"]
        B, P = recon.shape
        diff = recon - target
        loss = float(np.mean(diff ** 2))
        d_recon = (2.0 / (B * P)) * diff                                  # (B, P)

        p, g = cache["p"], cache["g"]
        d_p = np.einsum("bp,bcp->bc", d_recon, g)                         # (B, C)
        d_g = np.einsum("bp,bc->bcp", d_recon, p)                         # (B, C, P)
        d_g_pre = d_g * g * (1.0 - g)                                     # sigmoid'

        # Generative output layer
        d_W_gen2 = _per_cap_weight_grad(cache["h"], d_g_pre)              # (C, H_gen, P)
        d_b_gen2 = d_g_pre.sum(axis=0)                                    # (C, P)
        d_h = _per_cap_input_grad(d_g_pre, self.W_gen2)                   # (B, C, H_gen)
        d_h_pre = d_h * (cache["h_pre"] > 0).astype(np.float32)           # ReLU'

        # Generative hidden layer
        d_W_gen1 = _per_cap_weight_grad(cache["xy_t"], d_h_pre)           # (C, 2, H_gen)
        d_b_gen1 = d_h_pre.sum(axis=0)                                    # (C, H_gen)
        d_xy_t = _per_cap_input_grad(d_h_pre, self.W_gen1)                # (B, C, 2)

        # Branch into (xy, dxdy) -- gradient flows only into xy (dxdy is the input)
        d_xy = d_xy_t
        d_p_pre = d_p * p * (1.0 - p)                                     # sigmoid'

        # Pack back into the 3-vector instantiation gradient
        d_o = np.empty_like(cache["o"])
        d_o[:, :, 0] = d_p_pre
        d_o[:, :, 1:3] = d_xy

        # Instantiation layer
        d_W_inst = _per_cap_weight_grad(cache["r"], d_o)                  # (C, H_rec, 3)
        d_b_inst = d_o.sum(axis=0)                                        # (C, 3)
        d_r = _per_cap_input_grad(d_o, self.W_inst)                       # (B, C, H_rec)

        d_r_pre = d_r * cache["r"] * (1.0 - cache["r"])                   # sigmoid'

        # Recognition layer
        d_W_rec = _W_rec_grad(cache["x"], d_r_pre)                        # (C, I, H_rec)
        d_b_rec = d_r_pre.sum(axis=0)                                     # (C, H_rec)

        grads = dict(W_rec=d_W_rec, b_rec=d_b_rec,
                     W_inst=d_W_inst, b_inst=d_b_inst,
                     W_gen1=d_W_gen1, b_gen1=d_b_gen1,
                     W_gen2=d_W_gen2, b_gen2=d_b_gen2)
        return loss, grads

    # -- inference: read off (dx, dy) from a pair --------------------------

    def predict_transformation(self, image1: np.ndarray, image2: np.ndarray,
                               top_k: int = 3):
        """
        image1, image2: (28, 28) or batched (B, 28, 28).
        Returns predicted (dx, dy): (2,) or (B, 2).

        At inference the network has only its recognition heads. Compare the
        (x, y) outputs on the two images; the difference, weighted by the
        per-capsule "min presence" min(p1, p2), is the predicted translation.

        `top_k`: keep only the top-k capsules per example, ranked by
        min(p1_c, p2_c). Most capsules have near-zero presence on any given
        input; including them just averages noise into the prediction. With
        the default top_k=3 the R² jumps from ~0.6 (full mean) to ~0.78.
        Pass top_k=None to use all capsules with the (p1+p2)/2 weighting.
        """
        single = (image1.ndim == 2)
        x1 = image1.reshape(-1, self.image_size * self.image_size).astype(np.float32)
        x2 = image2.reshape(-1, self.image_size * self.image_size).astype(np.float32)
        p1, xy1, _ = self._recognize(x1)
        p2, xy2, _ = self._recognize(x2)
        diff = xy2 - xy1                              # (B, C, 2)

        if top_k is None or top_k >= self.n_capsules:
            w = (p1 + p2) / 2.0                       # (B, C)
            denom = w.sum(axis=1, keepdims=True) + 1e-6
            dxdy_pred = (w[:, :, None] * diff).sum(axis=1) / denom
        else:
            p_min = np.minimum(p1, p2)                # (B, C)
            top_idx = np.argsort(-p_min, axis=1)[:, :top_k]  # (B, k)
            B = x1.shape[0]
            bidx = np.arange(B)[:, None]
            diff_sel = diff[bidx, top_idx]            # (B, k, 2)
            w_sel = p_min[bidx, top_idx]              # (B, k)
            denom = w_sel.sum(axis=1, keepdims=True) + 1e-6
            dxdy_pred = (w_sel[:, :, None] * diff_sel).sum(axis=1) / denom

        return dxdy_pred[0] if single else dxdy_pred


# ----------------------------------------------------------------------
# Convenience constructor (matches problem.py signature)
# ----------------------------------------------------------------------

def build_capsule_net(n_capsules: int = 30, recognition_dim: int = 3,
                      generative_units: int = 128, patch_size: int = 22,
                      n_rec_hidden: int = 20, seed: int = 0):
    if recognition_dim != 3:
        raise ValueError("recognition_dim must be 3 for translation-only "
                         "(p, x, y); the paper's full-affine variant would use 9.")
    return TransformingAutoencoder(n_capsules=n_capsules,
                                   n_rec_hidden=n_rec_hidden,
                                   n_gen_hidden=generative_units,
                                   patch_size=patch_size,
                                   seed=seed)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def _build_batch(images: np.ndarray, batch_size: int, max_shift: int,
                 rng: np.random.Generator, input_jitter: int = 5):
    """Build a batch of (input, dxdy, target_22x22, input_28, transformed_28).

    Each example: random input-jitter t_in; random dxdy. The input image is
    the digit translated by t_in; the target is the digit translated by
    t_in + dxdy (equivalently, the input image translated by dxdy).
    """
    idx = rng.integers(0, images.shape[0], size=batch_size)
    base = images[idx]
    t_in = rng.integers(-input_jitter, input_jitter + 1, size=(batch_size, 2)).astype(np.int32)
    dxdy = rng.integers(-max_shift, max_shift + 1, size=(batch_size, 2)).astype(np.int32)
    inputs = np.empty_like(base)
    targets28 = np.empty_like(base)
    for b in range(batch_size):
        inputs[b] = translate(base[b], int(t_in[b, 0]), int(t_in[b, 1]))
        targets28[b] = translate(base[b], int(t_in[b, 0] + dxdy[b, 0]),
                                  int(t_in[b, 1] + dxdy[b, 1]))
    target = crop_center(targets28, size=22).reshape(batch_size, -1)
    x = inputs.reshape(batch_size, -1)
    return x, dxdy.astype(np.float32), target, inputs, targets28


def _r_squared(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    if ss_tot < 1e-10:
        return 0.0
    return 1.0 - ss_res / ss_tot


def train(model: "TransformingAutoencoder | None" = None,
          n_epochs: int = 5,
          steps_per_epoch: int = 200,
          batch_size: int = 64,
          lr: float = 1e-3,
          beta1: float = 0.9,
          beta2: float = 0.999,
          eps: float = 1e-8,
          n_capsules: int = 30,
          n_train_images: int = 10000,
          max_shift: int = 5,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True):
    """Train the transforming autoencoder with Adam.

    `model` may be passed in (e.g. for resuming or visualization); otherwise
    a fresh model is built with seed `seed`.

    Returns (model, history).
    """
    rng = np.random.default_rng(seed)
    images = load_mnist(split="train")
    if n_train_images and n_train_images < images.shape[0]:
        sub = rng.permutation(images.shape[0])[:n_train_images]
        images = images[sub]

    if model is None:
        model = TransformingAutoencoder(n_capsules=n_capsules, seed=seed)
    # Adam state
    adam_m = model.zero_like_params()
    adam_v = model.zero_like_params()
    adam_t = 0

    if verbose:
        print(f"# transforming-autoencoder: {model.n_capsules} capsules, "
              f"{model.n_gen_hidden} gen units, {model.patch_size}x{model.patch_size} patch")
        print(f"# data: {images.shape[0]} MNIST images, max_shift=+/-{max_shift}")

    # Fixed validation set (so the R² curve is comparable across epochs).
    # Apply the same input-jitter scheme as training so the disentanglement
    # test is meaningful.
    val_rng = np.random.default_rng(seed + 1)
    val_n = min(512, images.shape[0])
    val_idx = val_rng.permutation(images.shape[0])[:val_n]
    val_base = images[val_idx]
    val_t_in = val_rng.integers(-5, 5 + 1, size=(val_n, 2)).astype(np.int32)
    val_dxdy = val_rng.integers(-max_shift, max_shift + 1,
                                size=(val_n, 2)).astype(np.float32)
    val_imgs = np.stack([
        translate(val_base[i], int(val_t_in[i, 0]), int(val_t_in[i, 1]))
        for i in range(val_n)
    ])
    val_transformed = np.stack([
        translate(val_base[i], int(val_t_in[i, 0] + val_dxdy[i, 0]),
                  int(val_t_in[i, 1] + val_dxdy[i, 1]))
        for i in range(val_n)
    ])

    history = {"step": [], "epoch": [], "loss": [], "dx_r2": [], "dy_r2": [],
               "val_mse": []}
    step = 0
    t0 = time.time()
    for epoch in range(n_epochs):
        epoch_loss = 0.0
        for s in range(steps_per_epoch):
            x, dxdy, target, _, _ = _build_batch(images, batch_size, max_shift, rng)
            recon, cache = model.forward(x, dxdy)
            loss, grads = model.backward(cache, target)
            adam_t += 1
            bc1 = 1.0 - beta1 ** adam_t
            bc2 = 1.0 - beta2 ** adam_t
            for k, g in grads.items():
                m_buf = adam_m[k]
                v_buf = adam_v[k]
                m_buf[...] = beta1 * m_buf + (1.0 - beta1) * g
                v_buf[...] = beta2 * v_buf + (1.0 - beta2) * (g * g)
                update = lr * (m_buf / bc1) / (np.sqrt(v_buf / bc2) + eps)
                getattr(model, k)[...] -= update
            epoch_loss += loss
            step += 1

            if snapshot_callback is not None and (step % snapshot_every == 0):
                snapshot_callback(step, model, history,
                                  val_imgs, val_transformed, val_dxdy)

        # End-of-epoch validation
        dxdy_pred = model.predict_transformation(val_imgs, val_transformed)
        r2_dx = _r_squared(val_dxdy[:, 0], dxdy_pred[:, 0])
        r2_dy = _r_squared(val_dxdy[:, 1], dxdy_pred[:, 1])
        # Validation reconstruction MSE
        x_val = val_imgs.reshape(val_n, -1)
        recon_val, _ = model.forward(x_val, val_dxdy)
        target_val = crop_center(val_transformed, 22).reshape(val_n, -1)
        val_mse = float(np.mean((recon_val - target_val) ** 2))

        avg_loss = epoch_loss / steps_per_epoch
        history["step"].append(step)
        history["epoch"].append(epoch + 1)
        history["loss"].append(avg_loss)
        history["dx_r2"].append(r2_dx)
        history["dy_r2"].append(r2_dy)
        history["val_mse"].append(val_mse)
        if verbose:
            elapsed = time.time() - t0
            print(f"epoch {epoch+1:2d}/{n_epochs}  "
                  f"train_mse={avg_loss:.5f}  val_mse={val_mse:.5f}  "
                  f"R2(dx)={r2_dx:.3f}  R2(dy)={r2_dy:.3f}  "
                  f"({elapsed:.1f}s)", flush=True)

    return model, history


# ----------------------------------------------------------------------
# Public predict_transformation entry-point (matches problem.py signature)
# ----------------------------------------------------------------------

def predict_transformation(model: TransformingAutoencoder, pair) -> np.ndarray:
    """`pair` is (image, transformed_image) or a 3-tuple including dx_dy.
    Returns the predicted (dx, dy)."""
    if len(pair) == 3:
        image1, image2, _ = pair
    else:
        image1, image2 = pair
    return model.predict_transformation(image1, image2)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--n-capsules", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--n-train-images", type=int, default=10000)
    p.add_argument("--max-shift", type=int, default=5)
    args = p.parse_args()

    model, history = train(n_epochs=args.n_epochs,
                           steps_per_epoch=args.steps_per_epoch,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           n_capsules=args.n_capsules,
                           n_train_images=args.n_train_images,
                           max_shift=args.max_shift,
                           seed=args.seed)
    print(f"\nFinal train MSE: {history['loss'][-1]:.5f}")
    print(f"Final val MSE:   {history['val_mse'][-1]:.5f}")
    print(f"Final R2(dx):    {history['dx_r2'][-1]:.3f}")
    print(f"Final R2(dy):    {history['dy_r2'][-1]:.3f}")


if __name__ == "__main__":
    main()
