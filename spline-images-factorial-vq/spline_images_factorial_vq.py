"""
Spline images & factorial vector quantization — reproduction of Hinton & Zemel,
"Autoencoders, MDL and Helmholtz free energy", NIPS 6 (1994).

Problem
-------
200 images (8 x 12) formed by Gaussian-blurring a curve through 5 control
points. The y-position of each control point is sampled uniformly; the x
positions are evenly spaced across the image width. A natural cubic spline
through the 5 (x, y) knots is rendered as a Gaussian-blurred 8 x 12 image.
The data manifold is therefore 5-dimensional (the 5 free y-values).

Models compared
---------------
1. **Standard stochastic VQ**: one encoder MLP -> 24-way softmax q(k|x). The
   "description" is one of 24 codewords. Bits-back KL[q || p] is at most
   log2(24) = 4.585 bits.
2. **Four-separate stochastic VQs**: four 6-code VQs, each with its own
   encoder/codebook, summed additively but trained INDEPENDENTLY (no joint
   KL coordination). They tend to learn redundant codebooks because nothing
   forces specialization.
3. **Factorial stochastic VQ** (the headline): four 6-code VQs whose codes
   factor a single distribution q(k1, k2, k3, k4 | x) = prod_d q_d(k_d | x).
   Trained jointly under the free-energy bound; specialisation emerges
   because the additive reconstruction couples the four codebooks.
4. **PCA** (5 components, continuous Gaussian code): a free-energy lower
   bound for a Gaussian-prior continuous code, used as a "no quantization"
   reference.

The "headline" comparison is total description length (bits/example) under
each model. Hinton & Zemel report ~25 bits per example for factorial VQ
(18 reconstruction + 7 code), a clear improvement over the 24-code standard
VQ on the same data.

Bits-back coding
----------------
For a stochastic encoder q(k|x) and a prior p(k), the *bits-back* description
length per example is

    DL(x) = E_{k ~ q(k|x)}[ -log p(x|k) ]  +  KL[q(k|x) || p(k)]
          = recon_cost                   +  code_cost

This is the negative ELBO / Helmholtz free energy. Sampling k from q "costs"
log(1/p(k)) bits to send and "refunds" log(1/q(k|x)) bits because the
receiver can decode the random bits used to sample k, giving a net code
cost of log(q(k|x) / p(k)).

For factorial q the KL decomposes:

    KL[q(k|x) || p(k)] = sum_d KL[q_d(k_d|x) || p_d(k_d)]

so the code cost grows linearly in the number of factor dims, while the
effective codebook size grows as 6^4 = 1296. That asymmetry is the whole
point.

CLI
---
    python3 spline_images_factorial_vq.py --seed 0 --n-dims 4 \
                                             --n-units-per-dim 6
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
from typing import Optional

import numpy as np


IMAGE_H = 8
IMAGE_W = 12


# ----------------------------------------------------------------------
# Spline image generation
# ----------------------------------------------------------------------

def _natural_cubic_spline(t_knots: np.ndarray, y_knots: np.ndarray,
                           t_eval: np.ndarray) -> np.ndarray:
    """Evaluate a natural cubic spline (zero second derivative at endpoints)
    on `t_eval` given knots (`t_knots`, `y_knots`). Pure numpy — no scipy.
    """
    n = len(t_knots) - 1
    h = np.diff(t_knots)
    A = np.zeros((n + 1, n + 1), dtype=np.float64)
    b = np.zeros(n + 1, dtype=np.float64)
    A[0, 0] = 1.0
    A[n, n] = 1.0
    for i in range(1, n):
        A[i, i - 1] = h[i - 1]
        A[i, i] = 2.0 * (h[i - 1] + h[i])
        A[i, i + 1] = h[i]
        b[i] = 6.0 * ((y_knots[i + 1] - y_knots[i]) / h[i]
                      - (y_knots[i] - y_knots[i - 1]) / h[i - 1])
    M = np.linalg.solve(A, b)

    out = np.empty_like(t_eval, dtype=np.float64)
    bin_idx = np.clip(np.searchsorted(t_knots, t_eval, side="right") - 1,
                       0, n - 1)
    for i in range(n):
        mask = bin_idx == i
        if not mask.any():
            continue
        ti = t_eval[mask]
        a_i = (t_knots[i + 1] - ti) / h[i]
        b_i = (ti - t_knots[i]) / h[i]
        out[mask] = (a_i * y_knots[i] + b_i * y_knots[i + 1]
                      + ((a_i ** 3 - a_i) * M[i]
                          + (b_i ** 3 - b_i) * M[i + 1]) * h[i] ** 2 / 6.0)
    return out


def render_spline_image(y_controls: np.ndarray,
                         h: int = IMAGE_H, w: int = IMAGE_W,
                         sigma: float = 0.6, t_dense: int = 200) -> np.ndarray:
    """Render a 2D image of the curve `y(x)` interpolated through y_controls.

    `y_controls` are the control-point y-values; their x-positions are evenly
    spaced across `[0, w-1]`. The curve is rasterised as the sum of Gaussian
    bumps at `t_dense` equally-spaced points along it, then peak-normalised.
    """
    n_controls = len(y_controls)
    t_knots = np.linspace(0.0, w - 1.0, n_controls)
    t_eval = np.linspace(0.0, w - 1.0, t_dense)
    y_eval = _natural_cubic_spline(t_knots, np.asarray(y_controls, dtype=np.float64),
                                    t_eval)
    xs = np.arange(w, dtype=np.float64)
    ys = np.arange(h, dtype=np.float64)
    xx, yy = np.meshgrid(xs, ys, indexing="xy")  # (h, w)
    dx = xx[:, :, None] - t_eval[None, None, :]
    dy = yy[:, :, None] - y_eval[None, None, :]
    bumps = np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))
    img = bumps.sum(axis=2)
    img /= max(float(img.max()), 1e-8)
    return img.astype(np.float32)


def generate_spline_images(n_samples: int = 200,
                            h: int = IMAGE_H, w: int = IMAGE_W,
                            n_controls: int = 5,
                            sigma: float = 0.6,
                            seed: int = 0
                            ) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n_samples` flattened spline images of size h x w.

    Returns
    -------
    images : (n_samples, h*w) float32 in [0, 1].
    controls : (n_samples, n_controls) float32, the y-values used.
    """
    rng = np.random.default_rng(seed)
    # Y-controls in [0.5, h-1.5] so the curve stays inside the canvas (with
    # a half-pixel margin to keep the Gaussian bump on-canvas).
    margin = 0.5
    controls = rng.uniform(margin, h - 1 - margin,
                            size=(n_samples, n_controls)).astype(np.float32)
    imgs = np.zeros((n_samples, h * w), dtype=np.float32)
    for i in range(n_samples):
        imgs[i] = render_spline_image(controls[i], h=h, w=w, sigma=sigma).reshape(-1)
    return imgs, controls


# ----------------------------------------------------------------------
# Stochastic VQ — single codebook
# ----------------------------------------------------------------------

def _relu(z: np.ndarray) -> np.ndarray:
    return np.maximum(z, 0.0)


def _softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class StochasticVQ:
    """One encoder MLP + softmax + linear codebook, trained under bits-back.

    encode(x):
        h    = relu(x @ W1 + b1)
        l    = h @ W2 + b2          shape (B, K)
        q    = softmax(l)            shape (B, K)
    decode(q):
        x_hat = q @ C                shape (B, D)

    Free-energy loss per example:
        L = ||x - x_hat||^2 / (2 sigma_x^2) + KL[q || p]
    where p is the prior over codes (uniform here).
    """

    def __init__(self, n_codes: int, D: int, n_mlp: int = 64,
                  sigma_x: float = 0.15, init_scale: float = 0.30,
                  seed: int = 0):
        self.K = n_codes
        self.D = D
        self.n_mlp = n_mlp
        self.sigma_x = sigma_x
        rng = np.random.default_rng(seed)
        self.W1 = (init_scale * rng.standard_normal((D, n_mlp))
                    ).astype(np.float32)
        self.b1 = np.zeros(n_mlp, dtype=np.float32)
        self.W2 = (init_scale * rng.standard_normal((n_mlp, n_codes))
                    ).astype(np.float32)
        self.b2 = np.zeros(n_codes, dtype=np.float32)
        # Codebook initialised with small data-scale magnitudes
        self.C = (init_scale * 0.3 * rng.standard_normal((n_codes, D))
                   ).astype(np.float32)
        self._init_adam()

    def _init_adam(self) -> None:
        self._adam_t = 0
        self._b1_a, self._b2_a, self._eps = 0.9, 0.999, 1e-8
        self._m = {k: np.zeros_like(getattr(self, k))
                    for k in ("W1", "b1", "W2", "b2", "C")}
        self._v = {k: np.zeros_like(getattr(self, k))
                    for k in ("W1", "b1", "W2", "b2", "C")}

    def _adam_update(self, name: str, grad: np.ndarray, lr: float) -> None:
        b1, b2, eps = self._b1_a, self._b2_a, self._eps
        t = self._adam_t
        self._m[name] = b1 * self._m[name] + (1 - b1) * grad
        self._v[name] = b2 * self._v[name] + (1 - b2) * (grad * grad)
        m_hat = self._m[name] / (1 - b1 ** t)
        v_hat = self._v[name] / (1 - b2 ** t)
        param = getattr(self, name)
        param -= lr * m_hat / (np.sqrt(v_hat) + eps)

    # ---- forward pass with cached intermediates ----------------------

    def _forward(self, x: np.ndarray) -> dict:
        z1 = x @ self.W1 + self.b1
        hh = _relu(z1)
        logits = hh @ self.W2 + self.b2
        q = _softmax(logits)
        x_hat = q @ self.C
        return {"x": x, "z1": z1, "h": hh, "logits": logits, "q": q,
                 "x_hat": x_hat}

    def encode(self, x: np.ndarray) -> np.ndarray:
        """Return q(k|x) of shape (B, K)."""
        return self._forward(x)["q"]

    def decode(self, q: np.ndarray) -> np.ndarray:
        return q @ self.C

    # ---- description length ------------------------------------------

    def description_length(self, x: np.ndarray) -> dict:
        """Per-example DL components, in nats."""
        out = self._forward(x)
        q = out["q"]
        x_hat = out["x_hat"]
        recon_nats = ((x - x_hat) ** 2).sum(axis=1) / (2.0 * self.sigma_x ** 2)
        log_q = np.log(q + 1e-12)
        log_p = -np.log(self.K)  # uniform prior
        kl = (q * (log_q - log_p)).sum(axis=1)
        return {"recon_nats": recon_nats, "code_nats": kl,
                 "total_nats": recon_nats + kl}

    # ---- gradient step ----------------------------------------------

    def grad_step(self, x: np.ndarray, lr: float, kl_weight: float = 1.0) -> dict:
        out = self._forward(x)
        B = x.shape[0]
        q = out["q"]
        x_hat = out["x_hat"]
        hh = out["h"]
        z1 = out["z1"]

        # Recon gradient: d L_recon / d x_hat
        d_xhat = (x_hat - x) / (self.sigma_x ** 2)         # (B, D)
        # x_hat = q @ C
        dC = q.T @ d_xhat / B                              # (K, D)
        d_q_recon = d_xhat @ self.C.T                      # (B, K)

        # KL gradient w.r.t. q: d/dq_k [ q_k * (log q_k - log p_k) ]
        #                     = log q_k + 1 - log p_k
        # but we need the gradient w.r.t. logits via softmax; it's easier to
        # use the closed form: d KL / d logits = q - p (for uniform p) when
        # the KL is taken with a softmax q. Derivation:
        #   KL = sum_k q_k log q_k - sum_k q_k log p_k
        #   d/d l_j q_k = q_k(delta_{jk} - q_j)
        # Combining, for any p (constant in logits):
        #   d KL / d l_j = q_j (log q_j - log p_j)
        #                  - q_j sum_k q_k (log q_k - log p_k)
        # Equivalently for logp_k = -log K (constant in k):
        #   d KL / d l_j = q_j (log q_j - <log q>_q)
        log_q = np.log(q + 1e-12)
        kl_per_class = log_q + np.log(self.K)               # log q - log p
        mean_kl = (q * kl_per_class).sum(axis=1, keepdims=True)
        d_logits_kl = q * (kl_per_class - mean_kl)          # (B, K)
        d_logits_kl *= kl_weight

        # Recon gradient through softmax:
        # d_q_recon -> d_logits_recon via Jacobian of softmax
        # d_logits = q * (d_q - sum_k q_k d_q_k)
        d_logits_recon = q * (d_q_recon
                                - (q * d_q_recon).sum(axis=1, keepdims=True))

        d_logits = d_logits_recon + d_logits_kl              # (B, K)

        dW2 = hh.T @ d_logits / B
        db2 = d_logits.mean(axis=0)
        d_h = d_logits @ self.W2.T

        d_z1 = d_h * (z1 > 0).astype(np.float32)
        dW1 = x.T @ d_z1 / B
        db1 = d_z1.mean(axis=0)

        # Adam update
        self._adam_t += 1
        self._adam_update("W1", dW1, lr)
        self._adam_update("b1", db1, lr)
        self._adam_update("W2", dW2, lr)
        self._adam_update("b2", db2, lr)
        self._adam_update("C", dC, lr)

        recon_sq = float(((x - x_hat) ** 2).sum(axis=1).mean())
        kl_mean = float((q * (log_q + np.log(self.K))).sum(axis=1).mean())
        ent_mean = float((-q * log_q).sum(axis=1).mean())
        return {"recon_sq": recon_sq, "kl_mean": kl_mean,
                 "ent_mean": ent_mean}


def build_baseline_vq(n_units: int = 24, D: int = IMAGE_H * IMAGE_W,
                       **kwargs) -> StochasticVQ:
    """One big stochastic VQ with `n_units` codes."""
    return StochasticVQ(n_codes=n_units, D=D, **kwargs)


# ----------------------------------------------------------------------
# Factorial VQ — additive reconstruction across n_dims independent VQs
# ----------------------------------------------------------------------

class FactorialVQ:
    """Several stochastic VQs whose codes factorise a single posterior.

    q(k_1, ..., k_M | x) = prod_d q_d(k_d | x)
    p(k_1, ..., k_M) = prod_d uniform(k_d) over n_codes
    x_hat = sum_d (q_d @ C_d)            (mean-field reconstruction)

    Free-energy loss:
        L = ||x - x_hat||^2 / (2 sigma_x^2)  +  sum_d KL[q_d || p_d]

    The four codebooks specialise because each contributes additively to a
    shared reconstruction; redundant codebook directions are penalised by
    the KL terms.

    `independent_training=True` decouples the four reconstructions: each
    factor is trained against the residual (x - sum_{d'<d} x_hat_{d'})
    without coordinating its KL with the others. This is the "4 separate
    stochastic VQs" baseline and tends to leave the four codebooks doing
    redundant work.
    """

    def __init__(self, n_dims: int = 4, n_codes_per_dim: int = 6,
                  D: int = IMAGE_H * IMAGE_W, n_mlp: int = 64,
                  sigma_x: float = 0.15, init_scale: float = 0.30,
                  independent_training: bool = False,
                  seed: int = 0):
        self.n_dims = n_dims
        self.n_codes_per_dim = n_codes_per_dim
        self.D = D
        self.sigma_x = sigma_x
        self.independent_training = independent_training
        # Each factor gets a fresh subseed so we don't get identical inits
        self.factors = [
            StochasticVQ(n_codes=n_codes_per_dim, D=D, n_mlp=n_mlp,
                          sigma_x=sigma_x, init_scale=init_scale,
                          seed=seed * 1000 + d + 1)
            for d in range(n_dims)
        ]

    # ---- forward ----------------------------------------------------

    def _forward(self, x: np.ndarray) -> dict:
        forwards = [f._forward(x) for f in self.factors]
        contributions = [fw["q"] @ f.C
                          for fw, f in zip(forwards, self.factors)]
        x_hat = np.zeros_like(x)
        for c in contributions:
            x_hat += c
        return {"forwards": forwards, "contributions": contributions,
                 "x_hat": x_hat}

    def encode(self, x: np.ndarray) -> list[np.ndarray]:
        return [fw["q"] for fw in self._forward(x)["forwards"]]

    def decode(self, qs: list[np.ndarray]) -> np.ndarray:
        x_hat = np.zeros((qs[0].shape[0], self.D), dtype=np.float32)
        for q, f in zip(qs, self.factors):
            x_hat += q @ f.C
        return x_hat

    # ---- description length -----------------------------------------

    def description_length(self, x: np.ndarray) -> dict:
        out = self._forward(x)
        x_hat = out["x_hat"]
        recon_nats = ((x - x_hat) ** 2).sum(axis=1) / (2.0 * self.sigma_x ** 2)
        kl_total = np.zeros(x.shape[0], dtype=np.float64)
        for fw, f in zip(out["forwards"], self.factors):
            q = fw["q"]
            log_q = np.log(q + 1e-12)
            log_p = -np.log(f.K)
            kl_total += (q * (log_q - log_p)).sum(axis=1)
        return {"recon_nats": recon_nats, "code_nats": kl_total,
                 "total_nats": recon_nats + kl_total}

    # ---- gradient step ----------------------------------------------

    def grad_step(self, x: np.ndarray, lr: float, kl_weight: float = 1.0) -> dict:
        if self.independent_training:
            return self._grad_step_independent(x, lr, kl_weight)
        return self._grad_step_factorial(x, lr, kl_weight)

    def _grad_step_factorial(self, x: np.ndarray, lr: float,
                                kl_weight: float) -> dict:
        out = self._forward(x)
        B = x.shape[0]
        x_hat = out["x_hat"]
        d_xhat = (x_hat - x) / (self.sigma_x ** 2)            # (B, D)

        recon_sq = float(((x - x_hat) ** 2).sum(axis=1).mean())
        kl_total = 0.0
        for fw, f in zip(out["forwards"], self.factors):
            q = fw["q"]
            hh = fw["h"]
            z1 = fw["z1"]
            log_q = np.log(q + 1e-12)
            kl_per_class = log_q + np.log(f.K)
            mean_kl = (q * kl_per_class).sum(axis=1, keepdims=True)
            d_logits_kl = q * (kl_per_class - mean_kl) * kl_weight

            # Recon: d_xhat -> contribution_d = q_d @ C_d
            dC = q.T @ d_xhat / B
            d_q_recon = d_xhat @ f.C.T
            d_logits_recon = q * (d_q_recon
                                    - (q * d_q_recon).sum(axis=1, keepdims=True))
            d_logits = d_logits_recon + d_logits_kl

            dW2 = hh.T @ d_logits / B
            db2 = d_logits.mean(axis=0)
            d_h = d_logits @ f.W2.T
            d_z1 = d_h * (z1 > 0).astype(np.float32)
            dW1 = x.T @ d_z1 / B
            db1 = d_z1.mean(axis=0)

            f._adam_t += 1
            f._adam_update("W1", dW1, lr)
            f._adam_update("b1", db1, lr)
            f._adam_update("W2", dW2, lr)
            f._adam_update("b2", db2, lr)
            f._adam_update("C", dC, lr)

            kl_total += float((q * (log_q + np.log(f.K))).sum(axis=1).mean())

        return {"recon_sq": recon_sq, "kl_total": kl_total}

    def _grad_step_independent(self, x: np.ndarray, lr: float,
                                  kl_weight: float) -> dict:
        """Greedy / matching-pursuit-style training: each factor sees the
        residual (x - sum_{d'<d} x_hat_{d'}) and optimises its own loss
        without coordinating with the rest. No joint free-energy bound.
        Used as the "4 separate stochastic VQs" baseline.
        """
        residual = x.copy()
        recon_sq = 0.0
        kl_total = 0.0
        for f in self.factors:
            info = f.grad_step(residual, lr, kl_weight=kl_weight)
            # subtract this factor's reconstruction from the residual using
            # the post-update parameters. This is what each VQ sees on the
            # next pass.
            fw = f._forward(residual)
            residual = residual - fw["x_hat"]
            recon_sq = info["recon_sq"]
            kl_total += info["kl_mean"]
        return {"recon_sq": recon_sq, "kl_total": kl_total}


def build_factorial_vq(n_dims: int = 4, n_units_per_dim: int = 6,
                        D: int = IMAGE_H * IMAGE_W, **kwargs) -> FactorialVQ:
    return FactorialVQ(n_dims=n_dims, n_codes_per_dim=n_units_per_dim,
                        D=D, **kwargs)


def build_separate_vq(n_dims: int = 4, n_units_per_dim: int = 6,
                       D: int = IMAGE_H * IMAGE_W, **kwargs) -> FactorialVQ:
    """Four-separate-VQs baseline: same architecture, independent training."""
    return FactorialVQ(n_dims=n_dims, n_codes_per_dim=n_units_per_dim,
                        D=D, independent_training=True, **kwargs)


# ----------------------------------------------------------------------
# PCA reference (continuous Gaussian code)
# ----------------------------------------------------------------------

class PCAModel:
    """Continuous Gaussian-code reference.

    Encoder: x_centred -> top-`n_components` PCA scores.
    Posterior: q(z|x) = delta(z - score(x)). Replaced with an isotropic
    Gaussian of variance sigma_q^2 to make KL finite (this *is* bits-back
    for continuous codes; see Hinton & van Camp 1993).

    Code cost (KL[N(mu, sigma_q^2 I) || N(0, sigma_p^2 I)]) per dim:
        (sigma_q^2 + mu^2) / (2 sigma_p^2) - 1/2 + log(sigma_p / sigma_q)

    Reconstruction is x_hat = mu @ V^T + mean. Recon cost as for the VQ
    models, ||x - x_hat||^2 / (2 sigma_x^2).
    """

    def __init__(self, n_components: int = 5, sigma_x: float = 0.15,
                  sigma_q: float = 0.10):
        self.n_components = n_components
        self.sigma_x = sigma_x
        self.sigma_q = sigma_q
        self.mean = None       # (D,)
        self.V = None          # (D, n_components)
        self.scales = None     # (n_components,) std-dev of each PC over data

    def fit(self, x: np.ndarray) -> "PCAModel":
        self.mean = x.mean(axis=0)
        xc = x - self.mean
        # SVD: xc = U S V^T, columns of V are principal directions
        u, s, vt = np.linalg.svd(xc, full_matrices=False)
        self.V = vt[:self.n_components].T.astype(np.float32)  # (D, K)
        scores = xc @ self.V                                  # (N, K)
        self.scales = scores.std(axis=0).astype(np.float32)
        return self

    def encode(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean) @ self.V

    def decode(self, z: np.ndarray) -> np.ndarray:
        return z @ self.V.T + self.mean

    def description_length(self, x: np.ndarray) -> dict:
        z = self.encode(x)
        x_hat = self.decode(z)
        recon_nats = ((x - x_hat) ** 2).sum(axis=1) / (2.0 * self.sigma_x ** 2)
        # Per-dim Gaussian KL with sigma_p chosen as data std for that dim
        # (so the prior matches the marginal). This makes the KL numbers
        # comparable to "free choice of prior".
        sigma_p = np.maximum(self.scales, 1e-3)
        var_q = self.sigma_q ** 2
        var_p = sigma_p ** 2
        # KL(N(mu, var_q I) || N(0, var_p I)) per dim
        kl_per_dim = (var_q + z ** 2) / (2.0 * var_p) - 0.5 \
                      + np.log(sigma_p / self.sigma_q)
        kl = kl_per_dim.sum(axis=1)
        return {"recon_nats": recon_nats, "code_nats": kl,
                 "total_nats": recon_nats + kl}


# ----------------------------------------------------------------------
# Generic spec-compatible "description_length" entry point
# ----------------------------------------------------------------------

def description_length(model, data: np.ndarray) -> float:
    """Mean description length per example, in bits (bits-back coding)."""
    info = model.description_length(data)
    return float(info["total_nats"].mean() / np.log(2.0))


# ----------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------

def train_vq(model, images: np.ndarray, n_epochs: int = 800,
              batch_size: int = 32, lr: float = 0.005,
              kl_weight_schedule: tuple[float, float] = (0.1, 1.0),
              eval_every: int = 25, seed: int = 0,
              snapshot_callback=None, snapshot_every: int = 50,
              verbose: bool = False) -> dict:
    """Train a VQ-style model under bits-back free energy."""
    rng = np.random.default_rng(seed + 1000)
    n = len(images)
    history = {"epoch": [], "recon_bits": [], "code_bits": [], "total_bits": []}
    cw_start, cw_end = kl_weight_schedule

    if verbose:
        info = model.description_length(images)
        print(f"  [pre]  total {info['total_nats'].mean()/np.log(2):.2f} bits  "
              f"recon {info['recon_nats'].mean()/np.log(2):.2f}  "
              f"code {info['code_nats'].mean()/np.log(2):.2f}")

    t0 = time.time()
    for epoch in range(n_epochs):
        progress = epoch / max(n_epochs - 1, 1)
        cw = cw_start + (cw_end - cw_start) * progress
        idx = rng.integers(0, n, size=batch_size)
        x_batch = images[idx]
        model.grad_step(x_batch, lr=lr, kl_weight=cw)

        if (epoch % eval_every == 0) or (epoch == n_epochs - 1):
            info = model.description_length(images)
            recon_b = float(info["recon_nats"].mean() / np.log(2.0))
            code_b = float(info["code_nats"].mean() / np.log(2.0))
            total_b = recon_b + code_b
            history["epoch"].append(epoch)
            history["recon_bits"].append(recon_b)
            history["code_bits"].append(code_b)
            history["total_bits"].append(total_b)
            if verbose and (epoch % (eval_every * 8) == 0
                             or epoch == n_epochs - 1):
                print(f"  epoch {epoch:4d}  cw={cw:.2f}  "
                      f"total={total_b:7.2f} bits  "
                      f"recon={recon_b:7.2f}  code={code_b:5.2f}  "
                      f"({time.time()-t0:.1f}s)")
        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                                 or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history)
    history["wall_time"] = time.time() - t0
    return history


# ----------------------------------------------------------------------
# Top-level four-way comparison
# ----------------------------------------------------------------------

def run_comparison(n_samples: int = 200, n_epochs: int = 800,
                    n_dims: int = 4, n_units_per_dim: int = 6,
                    seed: int = 0, verbose: bool = True) -> dict:
    """Train all four models on the same dataset and return DL summaries."""
    images, controls = generate_spline_images(n_samples=n_samples, seed=seed)
    D = images.shape[1]

    if verbose:
        print(f"# Generated {n_samples} spline images (8x12 -> {D}-dim).")
        print(f"#   intrinsic dim = {controls.shape[1]} y-control points")
        print(f"#   training each model for {n_epochs} epochs.\n")

    results = {}

    # -- Standard 24-VQ -----------------------------------------------
    n_total = n_dims * n_units_per_dim
    if verbose:
        print(f"[1/4] Standard stochastic VQ ({n_total} codes)...")
    big_vq = build_baseline_vq(n_units=n_total, D=D, seed=seed)
    h_big = train_vq(big_vq, images, n_epochs=n_epochs, seed=seed,
                      verbose=verbose)
    results["baseline_vq"] = {"model": big_vq, "history": h_big,
                                "dl": big_vq.description_length(images)}

    # -- Four separate VQs --------------------------------------------
    if verbose:
        print(f"\n[2/4] Four separate stochastic VQs "
              f"({n_dims} x {n_units_per_dim})...")
    sep_vq = build_separate_vq(n_dims=n_dims, n_units_per_dim=n_units_per_dim,
                                D=D, seed=seed)
    h_sep = train_vq(sep_vq, images, n_epochs=n_epochs, seed=seed,
                      verbose=verbose)
    results["separate_vq"] = {"model": sep_vq, "history": h_sep,
                                "dl": sep_vq.description_length(images)}

    # -- Factorial VQ -------------------------------------------------
    if verbose:
        print(f"\n[3/4] Factorial stochastic VQ "
              f"({n_dims} x {n_units_per_dim})...")
    fac_vq = build_factorial_vq(n_dims=n_dims, n_units_per_dim=n_units_per_dim,
                                  D=D, seed=seed)
    h_fac = train_vq(fac_vq, images, n_epochs=n_epochs, seed=seed,
                      verbose=verbose)
    results["factorial_vq"] = {"model": fac_vq, "history": h_fac,
                                 "dl": fac_vq.description_length(images)}

    # -- PCA reference ------------------------------------------------
    if verbose:
        print(f"\n[4/4] PCA (5 continuous Gaussian components)...")
    pca = PCAModel(n_components=5).fit(images)
    results["pca"] = {"model": pca, "history": None,
                        "dl": pca.description_length(images)}

    if verbose:
        print("\n" + "=" * 60)
        print("Description length (bits per example)")
        print("=" * 60)
        for name, label in [("baseline_vq", f"Standard {n_total}-VQ"),
                              ("separate_vq", f"Four separate {n_dims}x{n_units_per_dim} VQs"),
                              ("factorial_vq", f"Factorial {n_dims}x{n_units_per_dim} VQ"),
                              ("pca", "PCA (5 components)")]:
            dl = results[name]["dl"]
            recon_b = float(dl["recon_nats"].mean() / np.log(2.0))
            code_b = float(dl["code_nats"].mean() / np.log(2.0))
            tot_b = recon_b + code_b
            print(f"  {label:38s}  total={tot_b:7.2f}  "
                  f"recon={recon_b:7.2f}  code={code_b:5.2f}")
        print("=" * 60)

    return {"images": images, "controls": controls, **results}


# ----------------------------------------------------------------------
# Reproducibility metadata
# ----------------------------------------------------------------------

def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or "unknown",
    }


def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Spline images & factorial VQ (Hinton & Zemel 1994).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--n-units-per-dim", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=800)
    return p


def main():
    args = _build_argparser().parse_args()
    info = env_info()
    print(f"# python {info['python']}  numpy {info['numpy']}")
    print(f"# {info['platform']}\n")
    run_comparison(n_samples=args.n_samples, n_epochs=args.n_epochs,
                    n_dims=args.n_dims, n_units_per_dim=args.n_units_per_dim,
                    seed=args.seed, verbose=True)


if __name__ == "__main__":
    main()
