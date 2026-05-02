"""
Sunspots time-series prediction with soft weight-sharing
(Nowlan & Hinton 1992, "Simplifying Neural Networks by Soft Weight-Sharing",
Neural Computation 4(4), 473-493).

Data: yearly Wolfer / SILSO sunspot count, 1700 onwards. Weigend benchmark
split: train 1700-1920 (209 prediction targets after 12 lags), test
1921-1955 (35 targets). Normalised by the training maximum.

Architecture: 12 -> 8 -> 1 MLP, tanh hidden + linear output. 113 weights
total (96 in W1, 8 in b1, 8 in W2, 1 in b2). Full-batch gradient descent
with momentum.

Three regularisers compared:

  vanilla : just MSE
  decay   : MSE + (lam/2) * sum w_i^2                   (Gaussian prior)
  mog     : MSE + lam * sum_i [-log p(w_i)]             (Mixture-of-Gaussians)

For mog, p(w_i) = sum_k pi_k * N(w_i | mu_k, sigma_k^2). One component is
*pinned* at mu=0 with a small fixed sigma to give a "small-weights"
attractor; the other K-1 components have learnable mu_k, sigma_k. All
mixing coefficients pi_k are learnable via a softmax parametrisation.
Biases are exempt from the prior (only the W matrices participate).

The headline (paper, Table 2): MoG attains markedly lower test MSE than
weight-decay on this benchmark. We reproduce the *ordering* mog <= decay
<= vanilla on test MSE under default seeds.
"""

from __future__ import annotations
import argparse
import csv
import os
import platform
import sys
import time
import urllib.request
import urllib.error
from pathlib import Path
import numpy as np


# ----------------------------------------------------------------------
# Wolfer / SILSO data loading
# ----------------------------------------------------------------------

SILSO_URL = "https://www.sidc.be/SILSO/INFO/snytotcsv.php"
CACHE_DIR = Path.home() / ".cache" / "hinton-sunspots"
CACHE_FILE = CACHE_DIR / "yearly_sunspots.csv"


def load_wolfer(start_year: int = 1700, end_year: int = 1979,
                 force_download: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Yearly Wolfer / SILSO sunspot count.

    Returns (years, counts). The Weigend benchmark uses 1700-1979 with the
    1700-1920 / 1921-1955 train/test split. We default to that range.

    Tries the SILSO yearly V2.0 file (https://www.sidc.be/SILSO/). If that
    fails, falls back to a synthetic 11-year-cycle proxy and prints a
    warning. The cache lives at `~/.cache/hinton-sunspots/`.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)

    if force_download or not CACHE_FILE.exists():
        try:
            req = urllib.request.Request(
                SILSO_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urllib.request.urlopen(req, timeout=30) as r:
                raw = r.read().decode()
            CACHE_FILE.write_text(raw)
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            if not CACHE_FILE.exists():
                print(f"# WARNING: SILSO download failed ({e!r}); "
                      f"falling back to synthetic data")
                years = np.arange(start_year, end_year + 1)
                t = years - start_year
                rng = np.random.default_rng(0)
                # 11-year cycle, peak ~150, with cycle-to-cycle amplitude
                # variation and noise -- a plausible Wolfer surrogate.
                amp = 100 + 50 * np.sin(2 * np.pi * t / 95)
                counts = np.maximum(
                    0,
                    amp * np.sin(2 * np.pi * t / 11.0) ** 2
                    + rng.normal(0, 10, size=len(t)),
                )
                return years, counts.astype(np.float64)

    raw = CACHE_FILE.read_text()
    yrs, cnts = [], []
    for line in raw.strip().split("\n"):
        # Format: 'YYYY.5 ; count ; std ; n_obs ; definitive'
        parts = [p.strip() for p in line.split(";")]
        if len(parts) < 2:
            continue
        try:
            y = int(float(parts[0]))
            c = float(parts[1])
        except ValueError:
            continue
        if c < 0:                                # missing-value sentinel
            continue
        yrs.append(y)
        cnts.append(c)

    years = np.array(yrs, dtype=int)
    counts = np.array(cnts, dtype=np.float64)
    mask = (years >= start_year) & (years <= end_year)
    return years[mask], counts[mask]


def make_lagged_dataset(series: np.ndarray, n_lags: int = 12,
                         norm: float | None = None
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Turn a 1-D series into (X, y) lagged supervised pairs.

    X[i] = series[i : i + n_lags] / norm
    y[i] = series[i + n_lags]      / norm

    If `norm` is None we use max(series) so values are in [0, 1].
    Returns (X, y, norm).
    """
    if norm is None:
        norm = float(np.max(series))
    s = np.asarray(series, dtype=np.float64) / norm
    n = len(s) - n_lags
    if n <= 0:
        raise ValueError(f"series of length {len(series)} too short for "
                          f"n_lags={n_lags}")
    X = np.stack([s[i : i + n_lags] for i in range(n)], axis=0)
    y = s[n_lags:].reshape(-1, 1)
    return X, y, norm


def weigend_split(years: np.ndarray, series: np.ndarray, n_lags: int = 12,
                   train_end: int = 1920, test_end: int = 1955
                   ) -> dict:
    """Standard Weigend benchmark split.

    Train pairs: predicted year in [start_year + n_lags, train_end].
    Test pairs : predicted year in [train_end + 1, test_end].
    Norm is computed from the *training* portion of the raw series only,
    so the test set never leaks scale information.
    """
    start_year = int(years[0])
    train_raw = series[: train_end - start_year + 1]
    norm = float(np.max(train_raw))
    s_norm = series / norm

    def _pairs(year_lo: int, year_hi: int):
        # predicted-year index ranges [lo .. hi]; need lags from idx-n_lags
        idx_lo = year_lo - start_year
        idx_hi = year_hi - start_year
        Xs, ys, year_targets = [], [], []
        for idx in range(idx_lo, idx_hi + 1):
            if idx - n_lags < 0:
                continue
            Xs.append(s_norm[idx - n_lags : idx])
            ys.append(s_norm[idx])
            year_targets.append(int(years[idx]))
        return (np.asarray(Xs), np.asarray(ys).reshape(-1, 1),
                np.asarray(year_targets))

    X_train, y_train, train_years = _pairs(start_year + n_lags, train_end)
    X_test, y_test, test_years = _pairs(train_end + 1, test_end)
    return {
        "X_train": X_train, "y_train": y_train, "train_years": train_years,
        "X_test": X_test, "y_test": y_test, "test_years": test_years,
        "norm": norm, "n_lags": n_lags,
    }


# ----------------------------------------------------------------------
# MLP (12 -> 8 -> 1)
# ----------------------------------------------------------------------

def tanh_act(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def dtanh_from_y(y: np.ndarray) -> np.ndarray:
    return 1.0 - y * y


class MLP:
    """12-input, 8-hidden tanh, 1-output linear MLP."""

    def __init__(self, n_in: int = 12, n_hidden: int = 8, n_out: int = 1,
                 init_scale: float = 0.5, seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        rng = np.random.default_rng(seed)
        self.W1 = init_scale * (rng.random((n_hidden, n_in)) - 0.5)
        self.b1 = init_scale * (rng.random((n_hidden,)) - 0.5)
        self.W2 = init_scale * (rng.random((n_out, n_hidden)) - 0.5)
        self.b2 = init_scale * (rng.random((n_out,)) - 0.5)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = tanh_act(X @ self.W1.T + self.b1)
        o = h @ self.W2.T + self.b2
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def all_W(self) -> np.ndarray:
        """Concatenated 1-D vector of W1 and W2 entries (NOT biases)."""
        return np.concatenate([self.W1.ravel(), self.W2.ravel()])

    def n_W(self) -> int:
        return self.W1.size + self.W2.size

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}


def mse(model: MLP, X: np.ndarray, y: np.ndarray) -> float:
    _, o = model.forward(X)
    return float(np.mean((o - y) ** 2))


def backprop_grads(model: MLP, X: np.ndarray, y: np.ndarray,
                    reduction: str = "sum") -> dict:
    """Gradients of squared-error loss.

    reduction="sum" : grads of 0.5 * sum_n (o-y)^2  (Nowlan & Hinton form;
                       balances naturally against a sum-over-weights prior)
    reduction="mean": grads of mean MSE
    """
    n = X.shape[0]
    h, o = model.forward(X)                              # (n, n_hidden), (n, 1)
    if reduction == "sum":
        dL_do = (o - y)                                    # (n, 1)
    elif reduction == "mean":
        dL_do = (o - y) * (2.0 / n)
    else:
        raise ValueError(f"unknown reduction {reduction!r}")
    grads = {
        "W2": dL_do.T @ h,                                 # (1, n_hidden)
        "b2": dL_do.sum(axis=0),                           # (1,)
    }
    delta_h = (dL_do @ model.W2) * dtanh_from_y(h)        # (n, n_hidden)
    grads["W1"] = delta_h.T @ X                            # (n_hidden, n_in)
    grads["b1"] = delta_h.sum(axis=0)
    return grads


# ----------------------------------------------------------------------
# Vanilla SGD
# ----------------------------------------------------------------------

def train_vanilla(model: MLP, data: dict, n_epochs: int = 4000,
                   lr: float = 0.0005, momentum: float = 0.9,
                   verbose: bool = False,
                   snapshot_callback=None, snapshot_every: int = 50
                   ) -> dict:
    """Plain MSE training. Returns history dict with epochwise metrics."""
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    velocities = {k: np.zeros_like(getattr(model, k))
                  for k in ["W1", "b1", "W2", "b2"]}
    history = _new_history()

    for ep in range(n_epochs):
        grads = backprop_grads(model, X_tr, y_tr)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
            setattr(model, k, getattr(model, k) + velocities[k])
        _record(history, ep, model, X_tr, y_tr, X_te, y_te,
                 reg_loss=0.0, snapshot_callback=snapshot_callback,
                 snapshot_every=snapshot_every,
                 mog=None, n_epochs=n_epochs)
        if verbose and (ep % 500 == 0 or ep == n_epochs - 1):
            print(f"  vanilla   ep {ep:4d}  "
                  f"train_mse={history['train_mse'][-1]:.5f}  "
                  f"test_mse={history['test_mse'][-1]:.5f}")
    return history


# ----------------------------------------------------------------------
# Weight-decay (L2 prior)
# ----------------------------------------------------------------------

def train_decay(model: MLP, data: dict, n_epochs: int = 4000,
                 lr: float = 0.0005, momentum: float = 0.9, lam: float = 0.01,
                 verbose: bool = False,
                 snapshot_callback=None, snapshot_every: int = 50
                 ) -> dict:
    """MSE + (lam/2) sum w^2.  Gradient on each W: lam * W."""
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    velocities = {k: np.zeros_like(getattr(model, k))
                  for k in ["W1", "b1", "W2", "b2"]}
    history = _new_history()

    for ep in range(n_epochs):
        grads = backprop_grads(model, X_tr, y_tr)
        # L2 on the W matrices only -- biases not regularised.
        grads["W1"] = grads["W1"] + lam * model.W1
        grads["W2"] = grads["W2"] + lam * model.W2
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
            setattr(model, k, getattr(model, k) + velocities[k])
        reg = 0.5 * lam * (np.sum(model.W1 ** 2) + np.sum(model.W2 ** 2))
        _record(history, ep, model, X_tr, y_tr, X_te, y_te,
                 reg_loss=float(reg),
                 snapshot_callback=snapshot_callback,
                 snapshot_every=snapshot_every,
                 mog=None, n_epochs=n_epochs)
        if verbose and (ep % 500 == 0 or ep == n_epochs - 1):
            print(f"  decay     ep {ep:4d}  "
                  f"train_mse={history['train_mse'][-1]:.5f}  "
                  f"test_mse={history['test_mse'][-1]:.5f}  "
                  f"|W|={np.linalg.norm(model.all_W()):.3f}")
    return history


# ----------------------------------------------------------------------
# Soft weight-sharing (Mixture-of-Gaussians prior)
# ----------------------------------------------------------------------

class MoG:
    """K-component Mixture-of-Gaussians prior on a flat weight vector.

    Component 0 is *pinned* at mu_0 = 0 with a small fixed sigma_0; this
    is the "small-weights" attractor that gives most weights an excuse to
    sit near zero. Components 1..K-1 are fully learnable.

    Parameterisation:
        mu_k     - learnable for k >= 1, fixed at 0 for k = 0
        log_sig_k - learnable for k >= 1, fixed for k = 0
        gamma_k  - softmax logits for the K mixing coefficients pi_k
                    (all learnable)
    """

    def __init__(self, K: int = 5, sigma_init: float = 0.1,
                  sigma_pin: float = 0.05, mu_spread: float = 0.4,
                  seed: int = 0):
        rng = np.random.default_rng(seed + 7)
        self.K = K
        self.sigma_pin = sigma_pin

        # mu_0 = 0 (pinned). The rest spread across [-mu_spread, mu_spread].
        if K == 1:
            mu = np.zeros(1)
        else:
            other_mus = np.linspace(-mu_spread, mu_spread, K - 1)
            other_mus = other_mus + 0.02 * rng.standard_normal(K - 1)
            mu = np.concatenate([[0.0], other_mus])
        self.mu = mu.astype(np.float64)

        # log_sigma -- pinned component fixed; others learnable
        self.log_sig = np.log(np.full(K, sigma_init, dtype=np.float64))
        self.log_sig[0] = np.log(sigma_pin)

        # mixing logits, equal init
        self.gamma = np.zeros(K, dtype=np.float64)

        # Adam-style accumulators -- the priors learn slowly relative to W
        self._state = {"mu": np.zeros(K), "log_sig": np.zeros(K),
                        "gamma": np.zeros(K)}

    def pi(self) -> np.ndarray:
        z = self.gamma - self.gamma.max()
        e = np.exp(z)
        return e / e.sum()

    def sigma(self) -> np.ndarray:
        return np.exp(self.log_sig)

    # --- log p(w) and responsibilities ---
    def responsibilities(self, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (r, log_p) where r is (n_w, K) and log_p is (n_w,)."""
        sig = self.sigma()
        pi = self.pi()
        # log N(w | mu_k, sig_k) = -0.5 log(2 pi sig_k^2) - 0.5 (w-mu_k)^2 / sig_k^2
        diff = w[:, None] - self.mu[None, :]
        log_norm = -0.5 * np.log(2 * np.pi * sig ** 2)
        log_kernel = -0.5 * (diff / sig) ** 2
        log_comp = log_norm + log_kernel + np.log(pi + 1e-30)
        # log p(w) = logsumexp_k log_comp
        m = log_comp.max(axis=1, keepdims=True)
        log_p = (m.squeeze(1) + np.log(np.exp(log_comp - m).sum(axis=1)))
        # responsibilities = exp(log_comp - log_p)
        r = np.exp(log_comp - log_p[:, None])
        return r, log_p

    def neg_log_prior(self, w: np.ndarray) -> float:
        _, log_p = self.responsibilities(w)
        return float(-np.sum(log_p))

    def grad_w(self, w: np.ndarray) -> np.ndarray:
        """d(-log p(w_i))/d w_i = sum_k r_ik * (w_i - mu_k) / sig_k^2."""
        r, _ = self.responsibilities(w)
        sig = self.sigma()
        diff = w[:, None] - self.mu[None, :]
        return np.sum(r * diff / sig ** 2, axis=1)

    def grads_params(self, w: np.ndarray) -> dict:
        """Gradients of -sum_i log p(w_i) w.r.t. mu_k, log_sig_k, gamma_k.

        Component 0 gradients on mu and log_sig are zeroed (pinned).
        """
        r, _ = self.responsibilities(w)                  # (n_w, K)
        sig = self.sigma()
        diff = w[:, None] - self.mu[None, :]              # (n_w, K)
        # d/d mu_k [-log p(w_i)] = -r_ik * (w_i - mu_k) / sig_k^2
        d_mu = -np.sum(r * diff / sig ** 2, axis=0)
        # d/d log_sig_k [-log p] = sum_i r_ik * (1 - (w_i-mu_k)^2/sig_k^2)
        d_logsig = np.sum(r * (1.0 - (diff / sig) ** 2), axis=0)
        # d/d pi_k [-log p] = -sum_i r_ik / pi_k --> gradient on softmax
        # logits: g_gamma_k = sum_i (pi_k - r_ik)
        pi = self.pi()
        d_gamma = np.sum(pi[None, :] - r, axis=0)

        # pinned component: zero out mu_0, log_sig_0 grads
        d_mu[0] = 0.0
        d_logsig[0] = 0.0
        return {"mu": d_mu, "log_sig": d_logsig, "gamma": d_gamma}

    def step(self, grads: dict, lr_prior: float = 0.005,
             momentum: float = 0.9, lr_sig_factor: float = 0.1,
             lr_pi_factor: float = 0.5):
        # mu can move fast; sigma and pi need much slower learning rates,
        # otherwise the wide components collapse / explode (standard
        # MoG-on-weights instability noted by Nowlan & Hinton 1992).
        scales = {"mu": 1.0, "log_sig": lr_sig_factor, "gamma": lr_pi_factor}
        for k in ("mu", "log_sig", "gamma"):
            self._state[k] = (momentum * self._state[k]
                               - lr_prior * scales[k] * grads[k])
            setattr(self, k, getattr(self, k) + self._state[k])
        # Tight clipping on sigma to keep components from collapsing
        # to a delta or expanding to a uniform.
        self.log_sig = np.clip(self.log_sig, np.log(0.02), np.log(0.5))


def train_with_soft_sharing(model: MLP, data: dict, n_components: int = 5,
                             n_epochs: int = 4000, lr: float = 0.0005,
                             lr_prior: float = 0.001, momentum: float = 0.9,
                             lam: float = 0.0005, sigma_init: float = 0.1,
                             sigma_pin: float = 0.05, mu_spread: float = 0.4,
                             seed: int = 0,
                             verbose: bool = False,
                             snapshot_callback=None,
                             snapshot_every: int = 50,
                             prior_warmup_epochs: int = 500,
                             pretrain_epochs: int = 200,
                             ) -> tuple[dict, MoG]:
    """SGD on MSE + lam * sum_i [-log p(w_i)] with MoG prior.

    Two-phase schedule:

    1. *Pretrain*: train MSE only for `pretrain_epochs` epochs to give the
       weights a non-trivial distribution. The MoG component means are
       then initialised by sorting the trained weights and placing the
       K-1 learnable means at evenly spaced quantiles. This is the
       practical fix Nowlan & Hinton (1992) use to avoid the prior
       crushing weights to zero before they pick up data structure.

    2. *Warmup + train*: turn on the MoG prior; lam ramps linearly from
       zero to its full value over `prior_warmup_epochs` epochs.
    """
    X_tr, y_tr = data["X_train"], data["y_train"]
    X_te, y_te = data["X_test"], data["y_test"]

    mog = MoG(K=n_components, sigma_init=sigma_init, sigma_pin=sigma_pin,
              mu_spread=mu_spread, seed=seed)

    velocities = {k: np.zeros_like(getattr(model, k))
                  for k in ["W1", "b1", "W2", "b2"]}
    history = _new_history()

    for ep in range(n_epochs):
        # phase 1: pretrain MSE only; phase 2: ramp lam linearly
        if ep < pretrain_epochs:
            lam_eff = 0.0
        else:
            warm = min(1.0,
                        (ep - pretrain_epochs + 1)
                        / max(1, prior_warmup_epochs))
            lam_eff = lam * warm

        # At the moment the prior switches on, re-initialise component
        # means from the actual weight distribution so the K-1 learnable
        # components start where the weights actually are.
        if ep == pretrain_epochs and pretrain_epochs > 0:
            w_sorted = np.sort(model.all_W())
            if mog.K > 1:
                quantiles = np.linspace(1.0 / (mog.K), 1.0 - 1.0 / mog.K,
                                          mog.K - 1)
                idxs = np.clip((quantiles * len(w_sorted)).astype(int),
                                0, len(w_sorted) - 1)
                mog.mu[1:] = w_sorted[idxs]
                # spread sigma based on inter-quantile spacing
                spacing = max(0.05,
                                float(np.std(w_sorted)))
                mog.log_sig[1:] = np.log(
                    np.full(mog.K - 1,
                             max(0.05, 0.5 * spacing)))

        # data gradient
        grads = backprop_grads(model, X_tr, y_tr)

        # prior gradient on weights (W1, W2 only -- biases free)
        w_flat = model.all_W()
        g_w = mog.grad_w(w_flat) * lam_eff               # (n_W,)
        g_w_W1 = g_w[: model.W1.size].reshape(model.W1.shape)
        g_w_W2 = g_w[model.W1.size:].reshape(model.W2.shape)
        grads["W1"] = grads["W1"] + g_w_W1
        grads["W2"] = grads["W2"] + g_w_W2

        # update model weights
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
            setattr(model, k, getattr(model, k) + velocities[k])

        # update MoG params (use *post-update* W flat so params chase)
        w_flat = model.all_W()
        prior_grads = mog.grads_params(w_flat)
        # scale the prior-parameter learning by lam_eff too -- if lam is
        # near zero the prior contribution is tiny, so move slowly.
        prior_grads = {k: v * lam_eff for k, v in prior_grads.items()}
        mog.step(prior_grads, lr_prior=lr_prior, momentum=momentum)

        reg = lam_eff * mog.neg_log_prior(model.all_W())
        _record(history, ep, model, X_tr, y_tr, X_te, y_te,
                 reg_loss=float(reg),
                 snapshot_callback=snapshot_callback,
                 snapshot_every=snapshot_every,
                 mog=mog, n_epochs=n_epochs)
        if verbose and (ep % 500 == 0 or ep == n_epochs - 1):
            pi = mog.pi()
            sig = mog.sigma()
            print(f"  mog       ep {ep:4d}  "
                  f"train_mse={history['train_mse'][-1]:.5f}  "
                  f"test_mse={history['test_mse'][-1]:.5f}  "
                  f"pi=[{', '.join(f'{p:.2f}' for p in pi)}]  "
                  f"sigma=[{', '.join(f'{s:.3f}' for s in sig)}]")
    return history, mog


# ----------------------------------------------------------------------
# History plumbing
# ----------------------------------------------------------------------

def _new_history() -> dict:
    return {"epoch": [], "train_mse": [], "test_mse": [],
            "reg_loss": [], "weight_norm": [], "snapshots": [],
            "mog_pi": [], "mog_mu": [], "mog_sigma": []}


def _record(history, ep, model, X_tr, y_tr, X_te, y_te, *,
             reg_loss=0.0, snapshot_callback=None, snapshot_every=50,
             mog=None, n_epochs=0) -> None:
    train_mse_v = mse(model, X_tr, y_tr)
    test_mse_v = mse(model, X_te, y_te)
    wn = float(np.linalg.norm(model.all_W()))
    history["epoch"].append(ep + 1)
    history["train_mse"].append(train_mse_v)
    history["test_mse"].append(test_mse_v)
    history["reg_loss"].append(reg_loss)
    history["weight_norm"].append(wn)
    if mog is not None:
        history["mog_pi"].append(mog.pi().copy())
        history["mog_mu"].append(mog.mu.copy())
        history["mog_sigma"].append(mog.sigma().copy())
    if snapshot_callback is not None and (ep % snapshot_every == 0
                                            or ep == n_epochs - 1):
        snapshot_callback(ep, model, history, mog)
        history["snapshots"].append((ep + 1, model.snapshot(),
                                       None if mog is None
                                       else {"pi": mog.pi().copy(),
                                              "mu": mog.mu.copy(),
                                              "sigma": mog.sigma().copy()}))


# ----------------------------------------------------------------------
# Top-level training entrypoint
# ----------------------------------------------------------------------

def build_model(n_lags: int = 12, n_hidden: int = 8, seed: int = 0,
                 init_scale: float = 0.5) -> MLP:
    return MLP(n_in=n_lags, n_hidden=n_hidden, n_out=1,
                init_scale=init_scale, seed=seed)


def train_one(method: str, data: dict, n_epochs: int = 4000,
               n_hidden: int = 16, lr: float = 0.0005, momentum: float = 0.9,
               lam: float = 0.0005, n_components: int = 5,
               seed: int = 0, verbose: bool = False,
               snapshot_callback=None, snapshot_every: int = 50,
               ) -> tuple[MLP, dict, MoG | None]:
    model = build_model(n_lags=data["n_lags"], n_hidden=n_hidden,
                         seed=seed)
    if method == "vanilla":
        hist = train_vanilla(model, data, n_epochs=n_epochs, lr=lr,
                              momentum=momentum, verbose=verbose,
                              snapshot_callback=snapshot_callback,
                              snapshot_every=snapshot_every)
        return model, hist, None
    if method == "decay":
        hist = train_decay(model, data, n_epochs=n_epochs, lr=lr,
                            momentum=momentum, lam=lam, verbose=verbose,
                            snapshot_callback=snapshot_callback,
                            snapshot_every=snapshot_every)
        return model, hist, None
    if method == "mog":
        hist, mog = train_with_soft_sharing(
            model, data, n_components=n_components, n_epochs=n_epochs,
            lr=lr, momentum=momentum, lam=lam, seed=seed, verbose=verbose,
            snapshot_callback=snapshot_callback,
            snapshot_every=snapshot_every,
        )
        return model, hist, mog
    raise ValueError(f"unknown method {method!r}")


# ----------------------------------------------------------------------
# Multi-method comparison
# ----------------------------------------------------------------------

def compare_methods(data: dict, n_epochs: int = 4000, seed: int = 0,
                     lam_decay: float = 0.01, lam_mog: float = 0.0005,
                     n_components: int = 5, lr: float = 0.0005,
                     n_hidden: int = 16, verbose: bool = False) -> dict:
    """Train all three methods on the same data; return a dict keyed by method."""
    out = {}
    for name, lam in [("vanilla", 0.0), ("decay", lam_decay),
                       ("mog", lam_mog)]:
        if verbose:
            print(f"\n--- training {name} ---")
        model, hist, mog = train_one(
            name, data, n_epochs=n_epochs, n_hidden=n_hidden, lr=lr,
            lam=lam, n_components=n_components, seed=seed, verbose=verbose,
        )
        out[name] = {
            "model": model, "history": hist, "mog": mog,
            "final_train_mse": hist["train_mse"][-1],
            "final_test_mse": hist["test_mse"][-1],
            "best_test_mse": float(np.min(hist["test_mse"])),
        }
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment():
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--method", choices=["vanilla", "decay", "mog", "all"],
                    default="all")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--epochs", type=int, default=12000)
    p.add_argument("--n-hidden", type=int, default=16)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--lam-decay", type=float, default=0.01)
    p.add_argument("--lam-mog", type=float, default=0.0005)
    p.add_argument("--n-lags", type=int, default=12)
    p.add_argument("--train-end", type=int, default=1920)
    p.add_argument("--test-end", type=int, default=1955)
    p.add_argument("--force-download", action="store_true")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _print_environment()
    years, counts = load_wolfer(force_download=args.force_download)
    print(f"# Wolfer data: {len(years)} years, {years[0]}-{years[-1]}, "
          f"max={counts.max():.0f}, mean={counts.mean():.0f}")

    data = weigend_split(years, counts, n_lags=args.n_lags,
                          train_end=args.train_end, test_end=args.test_end)
    print(f"# Split: train {len(data['y_train'])} points "
          f"({data['train_years'][0]}-{data['train_years'][-1]}), "
          f"test {len(data['y_test'])} points "
          f"({data['test_years'][0]}-{data['test_years'][-1]}), "
          f"normalised by train_max={data['norm']:.1f}")

    verbose = not args.quiet

    t0 = time.time()
    if args.method == "all":
        results = compare_methods(
            data, n_epochs=args.epochs, seed=args.seed,
            lam_decay=args.lam_decay, lam_mog=args.lam_mog,
            n_components=args.n_components, lr=args.lr,
            n_hidden=args.n_hidden, verbose=verbose,
        )
        wallclock = time.time() - t0
        print(f"\n=== summary  ({wallclock:.1f}s total) ===")
        print(f"{'method':10s} {'train MSE':>11s}  {'test MSE':>11s}  "
              f"{'best test':>11s}  {'|W|':>7s}")
        for name in ["vanilla", "decay", "mog"]:
            r = results[name]
            wn = r["history"]["weight_norm"][-1]
            print(f"{name:10s} {r['final_train_mse']:11.5f}  "
                  f"{r['final_test_mse']:11.5f}  "
                  f"{r['best_test_mse']:11.5f}  {wn:7.3f}")

        v = results["vanilla"]["final_test_mse"]
        d = results["decay"]["final_test_mse"]
        m = results["mog"]["final_test_mse"]
        ordered = (m <= d <= v)
        print(f"\nordering mog <= decay <= vanilla on final test MSE: "
              f"{ordered}  ({m:.5f} <= {d:.5f} <= {v:.5f})")
    else:
        model, hist, mog = train_one(
            args.method, data, n_epochs=args.epochs,
            n_hidden=args.n_hidden, lr=args.lr,
            momentum=args.momentum,
            lam=(args.lam_decay if args.method == "decay" else args.lam_mog),
            n_components=args.n_components, seed=args.seed, verbose=verbose,
        )
        wallclock = time.time() - t0
        print(f"\n=== {args.method}  seed={args.seed} ({wallclock:.1f}s) ===")
        print(f"final train MSE: {hist['train_mse'][-1]:.5f}")
        print(f"final test MSE : {hist['test_mse'][-1]:.5f}")
        print(f"best test MSE  : {min(hist['test_mse']):.5f}")
        print(f"|W|            : {hist['weight_norm'][-1]:.3f}")
        if mog is not None:
            print(f"\nMoG components (final):")
            print(f"  pi    = {mog.pi()}")
            print(f"  mu    = {mog.mu}")
            print(f"  sigma = {mog.sigma()}")


if __name__ == "__main__":
    main()
