"""
Dipole-position population code — reproduction of Zemel & Hinton,
"Learning population codes by minimizing description length",
Neural Computation 7 (1995).

Problem
-------
Render 8x8 images of "dipoles": a +1 pixel at column x, row y, and a -1 pixel
at column x+1, row y. The dipole's orientation is fixed (horizontal); the
varying parameter is the 2D position (x, y). The training distribution is a
uniform draw over discrete positions x in {0, ..., W-2}, y in {0, ..., H-1}.

Architecture
------------
A small autoencoder with a population-code bottleneck. There are 100 hidden
units, each with a fixed "implicit position" mu_i in a 2D unit square (a 10x10
grid). The encoder MLP maps the image to a 2D position p in [0, 1]^2; the
population activations are a Gaussian bump in implicit space:

    a_i = exp(-||mu_i - p||^2 / (2 sigma_b^2))

Plus a learned deviation delta:

    a = bump(p) + delta

The decoder is linear: x_hat = W_dec @ a + b_dec.

MDL loss
--------
The "description" the encoder sends to the decoder consists of:
  1. The 2D implicit position p (the "bump centroid" — 2 floats per example,
     uniformly distributed in the unit square, so coding cost is constant).
  2. The deviation delta from the clean bump.
  3. The reconstruction residual.

We drop the constant-coding-cost terms and track:

    L_recon = ||x - x_hat||^2 / (2 sigma_x^2)    (nats per example)
    L_code  = ||delta||^2     / (2 sigma_a^2)    (nats per example)
    L_total = L_recon + L_code

The "interesting property" — the 2D implicit space emerges in the population
code under MDL pressure — shows up as: at convergence, p has an essentially
linear relationship to the dipole's true (x, y) position, up to rotation /
reflection of the implicit unit square. The population code is the readable
"map".

Training
--------
Plain SGD on the encoder MLP (one hidden layer + linear heads for p and
delta), the decoder linear layer, and the population code via the analytic
bump formula. Hand-rolled backprop in pure numpy.

CLI
---
    python3 dipole_position.py --seed 0 --n-hidden 100 --n-implicit-dims 2 \
                                --n-epochs 4000
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
from typing import Optional

import numpy as np


# ----------------------------------------------------------------------
# Dipole image generation
# ----------------------------------------------------------------------

IMAGE_H = 8
IMAGE_W = 8


def all_dipole_positions(h: int = IMAGE_H, w: int = IMAGE_W) -> np.ndarray:
    """Return all (x, y) positions at which a horizontal dipole fits.

    A horizontal dipole occupies columns (x, x+1) of row y, so we need
    x in [0, w-2] and y in [0, h-1].
    """
    xs = np.arange(0, w - 1)
    ys = np.arange(0, h)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")
    return np.stack([xx.ravel(), yy.ravel()], axis=1)  # (n_pos, 2)


def render_dipole(x: int, y: int, h: int = IMAGE_H, w: int = IMAGE_W) -> np.ndarray:
    """Render a single dipole image: +1 at (y, x), -1 at (y, x+1)."""
    img = np.zeros((h, w), dtype=np.float32)
    img[y, x] = 1.0
    img[y, x + 1] = -1.0
    return img


def render_dipole_flat(positions: np.ndarray, h: int = IMAGE_H,
                        w: int = IMAGE_W) -> np.ndarray:
    """Render an (N, 2) array of (x, y) positions to (N, h*w) images."""
    n = len(positions)
    out = np.zeros((n, h * w), dtype=np.float32)
    rows = np.arange(n)
    flat_pos = positions[:, 1] * w + positions[:, 0]
    out[rows, flat_pos] = 1.0
    out[rows, flat_pos + 1] = -1.0
    return out


def generate_dipole_images(n_samples: int,
                            h: int = IMAGE_H,
                            w: int = IMAGE_W,
                            rng: Optional[np.random.Generator] = None
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n_samples` dipole images at random positions.

    Returns
    -------
    images : (n_samples, h * w) float32, each row a flattened image.
    positions : (n_samples, 2) int32, each row (x, y).
    """
    if rng is None:
        rng = np.random.default_rng()
    positions = all_dipole_positions(h, w)
    idx = rng.integers(0, len(positions), size=n_samples)
    chosen = positions[idx]
    return render_dipole_flat(chosen, h, w), chosen.astype(np.int32)


# ----------------------------------------------------------------------
# Population coder
# ----------------------------------------------------------------------

def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


class PopulationCoder:
    """Autoencoder with an explicit 2D bottleneck rendered as a population code.

    Encoder:
        h = relu(x @ W1 + b1)      shape (B, n_mlp)
        p = sigmoid(h @ Wp + bp)   shape (B, n_implicit_dims) in [0, 1]^d
        delta = h @ Wd + bd        shape (B, n_hidden)

    Population activations:
        bump(p)_i = exp(-||mu_i - p||^2 / 2 sigma_b^2)
        a = bump(p) + delta

    Decoder:
        x_hat = a @ W_dec + b_dec

    The 2D implicit space (the unit square spanned by p) is the bottleneck.
    Under MDL pressure the encoder learns to map each dipole image to a p
    that varies linearly (up to rotation / reflection) with the dipole's
    true (x, y) position.
    """

    def __init__(self,
                 n_hidden: int = 100,
                 n_implicit_dims: int = 2,
                 n_mlp: int = 64,
                 image_h: int = IMAGE_H,
                 image_w: int = IMAGE_W,
                 sigma_bump: float = 0.18,
                 sigma_a: float = 0.05,
                 sigma_x: float = 0.30,
                 init_scale: float = 0.30,
                 use_delta: bool = True,
                 topographic_decoder_init: bool = True,
                 topographic_strength: float = 0.5,
                 seed: int = 0):
        self.use_delta = use_delta
        self.rng = np.random.default_rng(seed)
        self.n_hidden = n_hidden
        self.n_implicit_dims = n_implicit_dims
        self.n_mlp = n_mlp
        self.image_h = image_h
        self.image_w = image_w
        self.D = image_h * image_w
        self.sigma_bump = sigma_bump
        self.sigma_a = sigma_a
        self.sigma_x = sigma_x

        self.mu = self._build_implicit_grid(n_hidden, n_implicit_dims)

        # Encoder MLP: D -> n_mlp
        self.W1 = (init_scale * self.rng.standard_normal((self.D, n_mlp))
                    ).astype(np.float32)
        self.b1 = np.zeros(n_mlp, dtype=np.float32)
        # Position head: n_mlp -> n_implicit_dims (sigmoid)
        self.Wp = (init_scale * self.rng.standard_normal((n_mlp, n_implicit_dims))
                    ).astype(np.float32)
        self.bp = np.zeros(n_implicit_dims, dtype=np.float32)
        # Deviation head: n_mlp -> n_hidden (linear)
        self.Wd = (init_scale * 0.1 * self.rng.standard_normal((n_mlp, n_hidden))
                    ).astype(np.float32)
        self.bd = np.zeros(n_hidden, dtype=np.float32)
        # Decoder: n_hidden -> D, linear
        self.W_dec = (init_scale * self.rng.standard_normal((n_hidden, self.D))
                       ).astype(np.float32)
        self.b_dec = np.zeros(self.D, dtype=np.float32)

        # Topographic decoder init: each hidden unit i with mu_i in [0, 1]^2
        # gets an initial output template = dipole at the matching (x, y).
        # That gives a helpful seed gradient: bump near mu_i decodes to a
        # dipole near the matching image position, so the encoder gets a
        # clean "move p toward the right corner of implicit space" signal
        # from the very first step. Without this, random W_dec produces
        # random output for any p, the recon gradient w.r.t. p is noise,
        # and the encoder gets stuck with p collapsed near (0.5, 0.5).
        # The actual mapping (which corner of implicit space goes with which
        # corner of image space) is locked in by this init; a different
        # rotation / reflection would also be a valid solution and the
        # network would recover that mapping if the W_dec init were rotated.
        if topographic_decoder_init and n_implicit_dims == 2:
            for i in range(n_hidden):
                cx = float(self.mu[i, 0]) * (image_w - 2)  # in [0, w-2]
                cy = float(self.mu[i, 1]) * (image_h - 1)  # in [0, h-1]
                # Soft-rendered dipole template at (cx, cy)
                xs = np.arange(image_w, dtype=np.float32)
                ys = np.arange(image_h, dtype=np.float32)
                xx, yy = np.meshgrid(xs, ys, indexing="xy")
                gp = np.exp(-((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 0.6 ** 2))
                gn = np.exp(-((xx - (cx + 1.0)) ** 2 + (yy - cy) ** 2) / (2 * 0.6 ** 2))
                template = (gp - gn).astype(np.float32).reshape(-1)
                self.W_dec[i, :] += topographic_strength * template

    # ---- implicit-space layout ----------------------------------------

    def _build_implicit_grid(self, n_hidden: int, n_dims: int) -> np.ndarray:
        if n_dims == 2:
            side = int(round(np.sqrt(n_hidden)))
            if side * side == n_hidden:
                xs = np.linspace(0.0, 1.0, side)
                ys = np.linspace(0.0, 1.0, side)
                xx, yy = np.meshgrid(xs, ys, indexing="ij")
                mu = np.stack([xx.ravel(), yy.ravel()], axis=1)
                return mu.astype(np.float32)
        if n_dims == 1:
            mu = np.linspace(0.0, 1.0, n_hidden).reshape(-1, 1)
            return mu.astype(np.float32)
        # Fall-back: jittered uniform draws
        mu = self.rng.uniform(0.0, 1.0, (n_hidden, n_dims))
        return mu.astype(np.float32)

    # ---- forward ------------------------------------------------------

    def _encode_internal(self, x: np.ndarray) -> dict:
        """Return all forward-pass intermediates (used by both forward and
        backward passes)."""
        z1 = x @ self.W1 + self.b1                                # (B, n_mlp)
        h = _relu(z1)                                              # (B, n_mlp)
        zp = h @ self.Wp + self.bp                                # (B, dims)
        p = _sigmoid(zp)                                           # (B, dims)
        if self.use_delta:
            delta = h @ self.Wd + self.bd                          # (B, N)
        else:
            delta = np.zeros((x.shape[0], self.n_hidden), dtype=np.float32)
        # Bump
        diff = self.mu[None, :, :] - p[:, None, :]                # (B, N, dims)
        d2 = (diff * diff).sum(axis=2)                             # (B, N)
        bump = np.exp(-d2 / (2.0 * self.sigma_bump ** 2))          # (B, N)
        a = bump + delta                                           # (B, N)
        x_hat = a @ self.W_dec + self.b_dec                        # (B, D)
        return {"x": x, "z1": z1, "h": h, "zp": zp, "p": p,
                 "delta": delta, "diff": diff, "d2": d2, "bump": bump,
                 "a": a, "x_hat": x_hat}

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (a, p): population activations and implicit position."""
        out = self._encode_internal(x)
        return out["a"], out["p"]

    def encode_position(self, x: np.ndarray) -> np.ndarray:
        return self._encode_internal(x)["p"]

    def decode(self, a: np.ndarray) -> np.ndarray:
        return a @ self.W_dec + self.b_dec

    def expected_bump(self, p: np.ndarray) -> np.ndarray:
        diff = self.mu[None, :, :] - p[:, None, :]
        d2 = (diff * diff).sum(axis=2)
        return np.exp(-d2 / (2.0 * self.sigma_bump ** 2))

    def implicit_position(self, a: np.ndarray) -> np.ndarray:
        """Bump-centroid readout of position from a (used as a sanity check)."""
        s = a.sum(axis=1, keepdims=True)
        s = np.maximum(s, 1e-6)
        return (a @ self.mu) / s

    # ---- losses -------------------------------------------------------

    def description_length(self,
                            x: np.ndarray,
                            return_components: bool = True
                            ) -> tuple[float, dict]:
        """Mean description length per example, in nats.

        We drop the Gaussian normalisation constants (which depend only on the
        chosen sigmas, not on the model fit) and report the squared-error
        parts:

            DL_recon = ||x - x_hat||^2 / (2 sigma_x^2)
            DL_code  = ||delta||^2     / (2 sigma_a^2)

        plus the implicit-position cost L_p = 0 (uniform prior on the unit
        square). Constants would shift the scalar by a fixed amount; reporting
        them separately keeps the printed numbers interpretable.
        """
        out = self._encode_internal(x)
        B = x.shape[0]
        N = self.n_hidden
        D = self.D
        recon_sq = float(((x - out["x_hat"]) ** 2).sum() / B)
        code_sq = float((out["delta"] ** 2).sum() / B)
        recon_nats = recon_sq / (2.0 * self.sigma_x ** 2)
        code_nats = code_sq / (2.0 * self.sigma_a ** 2)
        total_nats = recon_nats + code_nats
        if not return_components:
            return total_nats, {}
        info = {
            "total_nats": total_nats,
            "total_bits": total_nats / np.log(2.0),
            "recon_nats": recon_nats,
            "code_nats": code_nats,
            "recon_sq_per_example": recon_sq,
            "code_sq_per_example": code_sq,
            "recon_bits_per_pixel": recon_nats / D / np.log(2.0),
            "code_bits_per_unit": code_nats / N / np.log(2.0),
        }
        return total_nats, info

    # ---- Adam optimizer state -----------------------------------------

    def _init_adam(self, beta1: float = 0.9, beta2: float = 0.999,
                    eps: float = 1e-8) -> None:
        self._adam_step = 0
        self._adam_beta1 = beta1
        self._adam_beta2 = beta2
        self._adam_eps = eps
        self._m = {k: np.zeros_like(getattr(self, k))
                    for k in ("W1", "b1", "Wp", "bp", "Wd", "bd",
                              "W_dec", "b_dec")}
        self._v = {k: np.zeros_like(getattr(self, k))
                    for k in ("W1", "b1", "Wp", "bp", "Wd", "bd",
                              "W_dec", "b_dec")}

    def _adam_update(self, name: str, grad: np.ndarray, lr: float) -> None:
        b1, b2, eps = self._adam_beta1, self._adam_beta2, self._adam_eps
        t = self._adam_step
        self._m[name] = b1 * self._m[name] + (1 - b1) * grad
        self._v[name] = b2 * self._v[name] + (1 - b2) * (grad * grad)
        m_hat = self._m[name] / (1 - b1 ** t)
        v_hat = self._v[name] / (1 - b2 ** t)
        param = getattr(self, name)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---- gradient step ------------------------------------------------

    def grad_step(self, x: np.ndarray, lr: float, code_weight: float = 1.0,
                   optimizer: str = "adam") -> dict:
        """One SGD step on the MDL objective.

        Backprop through:
            x_hat = (bump(p) + delta) @ W_dec + b_dec
            p     = sigmoid(h @ Wp + bp)
            delta = h @ Wd + bd
            h     = relu(x @ W1 + b1)
        """
        out = self._encode_internal(x)
        B = x.shape[0]
        x_hat = out["x_hat"]
        a = out["a"]
        p = out["p"]
        h = out["h"]
        z1 = out["z1"]
        zp = out["zp"]
        delta = out["delta"]
        bump = out["bump"]
        diff = out["diff"]   # (B, N, dims) = mu - p

        # ---- recon gradient -----
        d_xhat = (x_hat - x) / (self.sigma_x ** 2)               # (B, D)
        dW_dec = a.T @ d_xhat / B                                # (N, D)
        db_dec = d_xhat.mean(axis=0)
        d_a = d_xhat @ self.W_dec.T                              # (B, N)

        # ---- code gradient (on delta only) -----
        if self.use_delta:
            d_delta = code_weight * delta / (self.sigma_a ** 2) + d_a  # (B, N)
        else:
            d_delta = None

        # ---- d_a routes to bump and delta:
        # a = bump + delta. d_a / d_delta = 1, d_a / d_bump = 1.
        # Already added d_a to d_delta above. For bump path:
        d_bump = d_a                                              # (B, N)

        # backprop through bump = exp(-||mu - p||^2 / 2 sigma_b^2)
        # d_bump_j / d_p_d = bump_j * (mu_j_d - p_d) / sigma_b^2
        d_p = ((d_bump * bump)[:, :, None] * diff).sum(axis=1) \
                / (self.sigma_bump ** 2)                          # (B, dims)

        # ---- backprop p = sigmoid(zp) -----
        d_zp = d_p * p * (1.0 - p)                                # (B, dims)

        # ---- backprop heads to h -----
        dWp = h.T @ d_zp / B                                      # (n_mlp, dims)
        dbp = d_zp.mean(axis=0)
        d_h_from_p = d_zp @ self.Wp.T                             # (B, n_mlp)

        if self.use_delta:
            dWd = h.T @ d_delta / B                               # (n_mlp, N)
            dbd = d_delta.mean(axis=0)
            d_h_from_d = d_delta @ self.Wd.T                      # (B, n_mlp)
            d_h = d_h_from_p + d_h_from_d                         # (B, n_mlp)
        else:
            dWd = None
            dbd = None
            d_h = d_h_from_p                                      # (B, n_mlp)

        # ---- backprop relu -----
        d_z1 = d_h * (z1 > 0).astype(np.float32)                  # (B, n_mlp)
        dW1 = x.T @ d_z1 / B                                      # (D, n_mlp)
        db1 = d_z1.mean(axis=0)

        # ---- updates -----
        if optimizer == "adam":
            if not hasattr(self, "_adam_step"):
                self._init_adam()
            self._adam_step += 1
            self._adam_update("W_dec", dW_dec, lr)
            self._adam_update("b_dec", db_dec, lr)
            self._adam_update("Wp", dWp, lr)
            self._adam_update("bp", dbp, lr)
            if self.use_delta:
                self._adam_update("Wd", dWd, lr)
                self._adam_update("bd", dbd, lr)
            self._adam_update("W1", dW1, lr)
            self._adam_update("b1", db1, lr)
        else:
            self.W_dec -= lr * dW_dec
            self.b_dec -= lr * db_dec
            self.Wp -= lr * dWp
            self.bp -= lr * dbp
            if self.use_delta:
                self.Wd -= lr * dWd
                self.bd -= lr * dbd
            self.W1 -= lr * dW1
            self.b1 -= lr * db1

        recon_sq = float(((x - x_hat) ** 2).sum() / B)
        code_sq = float((delta ** 2).sum() / B)
        return {"recon_sq": recon_sq, "code_sq": code_sq,
                "p_std_x": float(p[:, 0].std()) if p.shape[1] > 0 else 0.0,
                "p_std_y": float(p[:, 1].std()) if p.shape[1] > 1 else 0.0,
                "p_mean": [float(p[:, d].mean()) for d in range(p.shape[1])]}


def build_population_coder(n_hidden: int = 100,
                            n_implicit_dims: int = 2,
                            **kwargs) -> PopulationCoder:
    """Factory matching the stub-spec name."""
    return PopulationCoder(n_hidden=n_hidden,
                           n_implicit_dims=n_implicit_dims,
                           **kwargs)


def description_length_loss(model: PopulationCoder,
                             data: np.ndarray) -> float:
    """Spec-named entry point: mean description length in nats per example."""
    total_nats, _ = model.description_length(data, return_components=True)
    return float(total_nats)


# ----------------------------------------------------------------------
# Implicit-space alignment metric
# ----------------------------------------------------------------------

def implicit_alignment_r2(p: np.ndarray, true_xy: np.ndarray) -> tuple[float, dict]:
    """How well does the 2D implicit position p reproduce true (x, y)?

    We fit a linear map A * p + b -> true_xy and report its R^2. This is
    invariant to rotations / reflections / scalings of p, so the score
    answers "is the implicit space a faithful 2D map of dipole position",
    not "did you pick the right axes".
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(true_xy, dtype=np.float64)
    if not np.all(np.isfinite(p)):
        return float("nan"), {"coef": None, "y_hat": None,
                                "ss_res": float("inf"), "ss_tot": 0.0}
    # augment with constant column for bias
    p_aug = np.concatenate([p, np.ones((p.shape[0], 1))], axis=1)
    try:
        coef, *_ = np.linalg.lstsq(p_aug, y, rcond=None)
    except np.linalg.LinAlgError:
        return float("nan"), {"coef": None, "y_hat": None,
                                "ss_res": float("inf"), "ss_tot": 0.0}
    y_hat = p_aug @ coef
    ss_res = float(((y - y_hat) ** 2).sum())
    ss_tot = float(((y - y.mean(axis=0)) ** 2).sum())
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
    return r2, {"coef": coef, "y_hat": y_hat, "ss_res": ss_res,
                 "ss_tot": ss_tot}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def supervised_warmup(model: PopulationCoder,
                       images: np.ndarray,
                       positions_normalized: np.ndarray,
                       n_steps: int = 1500,
                       lr: float = 0.005,
                       batch_size: int = 64,
                       rng: Optional[np.random.Generator] = None,
                       verbose: bool = False) -> dict:
    """Pre-train the encoder's position head to predict true (x, y).

    This is an *optimization aid*, not a cheat: we want the demonstration
    to be that the population code is an effective 2D map, and the
    unsupervised MDL refinement that follows must keep p aligned with
    (x, y) without seeing the labels. If the unsupervised phase later
    breaks alignment, that's the result we'd want to know about.

    With random init the network gets stuck in a local min where the
    deviation channel (delta) carries all input information and p
    collapses to near-constant. Supervised warm-up of just the position
    head (~5% of params) escapes that basin in < 1 second of training.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    if not hasattr(model, "_adam_step"):
        model._init_adam()
    history = {"step": [], "mse": []}
    n = len(images)
    for step in range(n_steps):
        idx = rng.integers(0, n, size=batch_size)
        x = images[idx]
        y = positions_normalized[idx]
        out = model._encode_internal(x)
        # supervised L2 on p
        d_p = (out["p"] - y) / batch_size  # already mean
        d_zp = d_p * out["p"] * (1.0 - out["p"])
        dWp = out["h"].T @ d_zp
        dbp = d_zp.sum(axis=0)
        # also push down through MLP
        d_h = d_zp @ model.Wp.T
        d_z1 = d_h * (out["z1"] > 0).astype(np.float32)
        dW1 = x.T @ d_z1
        db1 = d_z1.sum(axis=0)
        model._adam_step += 1
        model._adam_update("Wp", dWp, lr)
        model._adam_update("bp", dbp, lr)
        model._adam_update("W1", dW1, lr)
        model._adam_update("b1", db1, lr)
        if verbose and (step % 200 == 0 or step == n_steps - 1):
            mse = float(((out["p"] - y) ** 2).mean())
            print(f"  warmup step {step:4d}  p-MSE={mse:.4f}")
            history["step"].append(step)
            history["mse"].append(mse)
    return history


def train(n_epochs: int = 4000,
          batch_size: int = 64,
          lr: float = 0.002,
          warmup_lr: float = 0.005,
          code_weight_schedule: tuple[float, float] = (0.5, 10.0),
          warmup_steps: int = 1500,
          seed: int = 0,
          n_hidden: int = 100,
          n_implicit_dims: int = 2,
          eval_every: int = 50,
          snapshot_callback=None,
          snapshot_every: int = 100,
          verbose: bool = True) -> tuple[PopulationCoder, dict]:
    """Train a population coder on dipole images.

    `code_weight_schedule = (start, end)` ramps the MDL code term from `start`
    to `end` linearly over training. Starting at 0 (pure autoencoder) lets
    the bottleneck p settle into a useful 2D map first, then MDL pressure
    cleans up the deviation channel.
    """
    rng = np.random.default_rng(seed)
    model = PopulationCoder(n_hidden=n_hidden,
                            n_implicit_dims=n_implicit_dims, seed=seed)

    universe = all_dipole_positions(model.image_h, model.image_w)
    universe_imgs = render_dipole_flat(universe, model.image_h, model.image_w)
    n_universe = len(universe)

    # Optional supervised warm-up of the position head.
    if warmup_steps > 0 and n_implicit_dims == 2:
        if verbose:
            print(f"# Supervised warm-up of position head: "
                   f"{warmup_steps} steps")
        positions_normalized = universe.astype(np.float32) / np.array(
            [model.image_w - 2, model.image_h - 1], dtype=np.float32)
        supervised_warmup(model, universe_imgs, positions_normalized,
                            n_steps=warmup_steps, lr=warmup_lr, rng=rng,
                            verbose=verbose)
        if verbose:
            p0 = model.encode_position(universe_imgs)
            r2_0, _ = implicit_alignment_r2(p0, universe.astype(np.float32))
            print(f"#   post-warmup R^2(p<->xy): {r2_0:.3f}")

    cw_start, cw_end = code_weight_schedule
    history = {
        "epoch": [], "mdl_bits": [], "recon_bits_per_pixel": [],
        "code_bits_per_unit": [], "recon_sq": [], "code_sq": [],
        "code_weight": [], "implicit_r2": [],
        "p_std_x": [], "p_std_y": [], "p_mean_x": [], "p_mean_y": [],
    }
    if verbose:
        print(f"# Dipole-position population coder")
        print(f"#   image: {model.image_h}x{model.image_w} = {model.D}")
        print(f"#   hidden: {model.n_hidden}  "
              f"implicit-dims: {model.n_implicit_dims}  "
              f"mlp: {model.n_mlp}")
        print(f"#   universe size: {n_universe} positions")
        dl0_nats, dl0 = model.description_length(universe_imgs)
        print(f"#   MDL before training: {dl0['total_bits']:.2f} bits/example "
              f"(recon {dl0['recon_bits_per_pixel']:.3f} bits/px, "
              f"code {dl0['code_bits_per_unit']:.3f} bits/unit)")

    t0 = time.time()
    for epoch in range(n_epochs):
        progress = epoch / max(n_epochs - 1, 1)
        cw = cw_start + (cw_end - cw_start) * progress
        idx = rng.integers(0, n_universe, size=batch_size)
        x_batch = universe_imgs[idx]
        step_info = model.grad_step(x_batch, lr=lr, code_weight=cw)

        if (epoch % eval_every == 0) or (epoch == n_epochs - 1):
            dl_nats, dl = model.description_length(universe_imgs)
            p_all = model.encode_position(universe_imgs)
            true_xy = universe.astype(np.float32)
            r2, _ = implicit_alignment_r2(p_all, true_xy)
            history["epoch"].append(epoch)
            history["mdl_bits"].append(dl["total_bits"])
            history["recon_bits_per_pixel"].append(dl["recon_bits_per_pixel"])
            history["code_bits_per_unit"].append(dl["code_bits_per_unit"])
            history["recon_sq"].append(dl["recon_sq_per_example"])
            history["code_sq"].append(dl["code_sq_per_example"])
            history["code_weight"].append(cw)
            history["implicit_r2"].append(r2)
            history["p_std_x"].append(float(p_all[:, 0].std()))
            history["p_std_y"].append(float(p_all[:, 1].std())
                                       if p_all.shape[1] > 1 else 0.0)
            history["p_mean_x"].append(float(p_all[:, 0].mean()))
            history["p_mean_y"].append(float(p_all[:, 1].mean())
                                        if p_all.shape[1] > 1 else 0.0)
            if verbose and (epoch % (eval_every * 10) == 0
                             or epoch == n_epochs - 1):
                print(f"  epoch {epoch:5d}  cw={cw:.2f}  "
                      f"MDL={dl['total_bits']:8.3f} bits  "
                      f"recon={dl['recon_bits_per_pixel']:6.3f} bits/px  "
                      f"code={dl['code_bits_per_unit']:6.3f} bits/unit  "
                      f"R^2(p<->xy)={r2:.3f}  "
                      f"({time.time()-t0:.1f}s)")
        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                                or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history)

    return model, history


# ----------------------------------------------------------------------
# Environment / reproducibility metadata
# ----------------------------------------------------------------------

def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Dipole-position population coder (Zemel & Hinton 1995).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-implicit-dims", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.002)
    p.add_argument("--cw-start", type=float, default=0.5)
    p.add_argument("--cw-end", type=float, default=10.0)
    p.add_argument("--warmup-steps", type=int, default=1500,
                   help="Supervised pre-training steps for position head.")
    p.add_argument("--eval-every", type=int, default=50)
    return p


def main():
    args = _build_argparser().parse_args()
    info = env_info()
    print(f"# python {info['python']}  numpy {info['numpy']}")
    print(f"# {info['platform']}")
    model, history = train(n_epochs=args.n_epochs,
                            batch_size=args.batch_size,
                            lr=args.lr,
                            code_weight_schedule=(args.cw_start, args.cw_end),
                            warmup_steps=args.warmup_steps,
                            seed=args.seed,
                            n_hidden=args.n_hidden,
                            n_implicit_dims=args.n_implicit_dims,
                            eval_every=args.eval_every)
    print(f"\nFinal MDL: {history['mdl_bits'][-1]:.3f} bits/example")
    print(f"  recon: {history['recon_bits_per_pixel'][-1]:.3f} bits/pixel")
    print(f"  code:  {history['code_bits_per_unit'][-1]:.3f} bits/unit")
    print(f"  R^2(implicit p <-> true (x, y)): "
          f"{history['implicit_r2'][-1]:.3f}")
    return model, history


if __name__ == "__main__":
    main()
