"""
Dipole 3D-constraint population code (Zemel & Hinton 1995).

Generates 8x8 dipole images parameterized by (x, y, theta) -- three varying
parameters -- trains an unsupervised population coder with 225 hidden units,
and recovers a 3D implicit space whose coordinates correlate with the true
generative parameters.

Architecture:
  image (64) --W_enc--> hidden(225) --softmax--> a
  a, mu --> implicit position m_hat = a @ mu  in R^3
  predicted bump b_i = exp(-||mu_i - m_hat||^2 / 2 sigma^2)
  b --W_dec--> reconstruction (64)

The bottleneck is m_hat (3 numbers). Reconstruction must pass through it,
so the network is forced to map images onto a 3D manifold parameterised by
the unit positions mu, which themselves are learnable. The headline
expectation: a 3D implicit space emerges whose coordinates are an affine
mapping of the true (x, y, theta).

Reference:
    Zemel & Hinton (1995). "Learning population codes by minimising
    description length". Neural Computation 7(3), 549-564.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------

def _gaussian_blob(h: int, w: int, cy: float, cx: float, sigma: float) -> np.ndarray:
    ys = np.arange(h)[:, None].astype(np.float64)
    xs = np.arange(w)[None, :].astype(np.float64)
    return np.exp(-((ys - cy) ** 2 + (xs - cx) ** 2) / (2.0 * sigma ** 2))


def _dipole_image(x: float, y: float, theta: float,
                  h: int = 8, w: int = 8,
                  separation: float = 2.0, blob_sigma: float = 0.55) -> np.ndarray:
    """One dipole: a positive blob and a negative blob along a vector of angle theta."""
    dy = (separation / 2.0) * np.sin(theta)
    dx = (separation / 2.0) * np.cos(theta)
    pos = _gaussian_blob(h, w, y + dy, x + dx, blob_sigma)
    neg = _gaussian_blob(h, w, y - dy, x - dx, blob_sigma)
    return pos - neg


def generate_dipole_images(n_samples: int, h: int = 8, w: int = 8,
                           seed: int = 0, separation: float = 2.0,
                           blob_sigma: float = 0.55, pad: float = 1.5):
    """Render small dipole images at random (x, y, orientation).

    theta in [0, pi) because a dipole is symmetric under theta -> theta + pi
    (positive and negative blobs swap roles, which is the same image up to sign).
    Here we keep theta in [0, pi) so the relationship is a true bijection.

    Returns
    -------
    X      : (n_samples, h*w) flattened images
    params : (n_samples, 3) ground-truth (x, y, theta)
    """
    rng = np.random.default_rng(seed)
    cx = rng.uniform(pad, w - 1 - pad, size=n_samples)
    cy = rng.uniform(pad, h - 1 - pad, size=n_samples)
    theta = rng.uniform(0.0, np.pi, size=n_samples)
    X = np.empty((n_samples, h * w), dtype=np.float64)
    for i in range(n_samples):
        X[i] = _dipole_image(cx[i], cy[i], theta[i],
                             h=h, w=w,
                             separation=separation,
                             blob_sigma=blob_sigma).ravel()
    params = np.stack([cx, cy, theta], axis=1)
    return X, params


# --------------------------------------------------------------------------
# Population coder
# --------------------------------------------------------------------------

@dataclass
class PopulationCoder:
    """Population-code autoencoder with an explicit 3D implicit space.

    The encoder is a small MLP (image -> hidden_e -> m_hat in [0,1]^K). The
    decoder is a radial-basis-function bank: 225 units with learnable
    positions mu in implicit space, each contributing a bump activation
    ``b_i = exp(-||mu_i - m_hat||^2 / 2 sigma^2)``. The decoder linearly
    maps the bump pattern back to image space.

    All reconstruction information must pass through the 3D ``m_hat``, so
    the population structure on the hidden layer is forced to track the
    generative parameters.
    """

    W1: np.ndarray   # (D, He)  encoder layer 1
    b1: np.ndarray   # (He,)
    W2: np.ndarray   # (He, K)  encoder layer 2 -> m_hat (pre-sigmoid)
    b2: np.ndarray   # (K,)
    mu: np.ndarray   # (H, K)  implicit-space positions of RBF basis
    W_dec: np.ndarray   # (H, D)
    b_dec: np.ndarray   # (D,)
    sigma: float

    @property
    def n_hidden(self) -> int:
        return self.mu.shape[0]

    @property
    def n_implicit(self) -> int:
        return self.mu.shape[1]

    def copy(self) -> "PopulationCoder":
        return PopulationCoder(
            W1=self.W1.copy(), b1=self.b1.copy(),
            W2=self.W2.copy(), b2=self.b2.copy(),
            mu=self.mu.copy(),
            W_dec=self.W_dec.copy(), b_dec=self.b_dec.copy(),
            sigma=self.sigma,
        )


def build_population_coder(n_hidden: int = 225, n_implicit_dims: int = 3,
                           n_in: int = 64, n_enc_hidden: int = 32,
                           sigma: float = 0.18,
                           seed: int = 0) -> PopulationCoder:
    """Initialise encoder, decoder, and RBF positions.

    The 225 RBF centres ``mu`` are placed on a near-uniform 3D grid in the
    unit cube (with a small jitter), then refined by gradient descent.
    """
    rng = np.random.default_rng(seed)
    # Place mu on a roughly uniform 3D grid plus small jitter.
    # 225 = 9 * 5 * 5 (closest factorisation we can hit cleanly).
    K = n_implicit_dims
    if n_hidden == 225 and K == 3:
        gx = np.linspace(0.05, 0.95, 9)
        gy = np.linspace(0.05, 0.95, 5)
        gz = np.linspace(0.05, 0.95, 5)
        mu = np.array(np.meshgrid(gx, gy, gz, indexing="ij")).reshape(K, -1).T
        assert mu.shape == (n_hidden, K), mu.shape
    else:
        mu = rng.uniform(0.0, 1.0, size=(n_hidden, K))
    mu = mu + rng.normal(0.0, 0.01, size=mu.shape)

    # Encoder: image -> tanh(He) -> sigmoid(K)
    # Glorot-ish init.
    s1 = np.sqrt(2.0 / (n_in + n_enc_hidden))
    s2 = np.sqrt(2.0 / (n_enc_hidden + K))
    sd = np.sqrt(2.0 / (n_hidden + n_in))
    return PopulationCoder(
        W1=rng.normal(0.0, s1, size=(n_in, n_enc_hidden)),
        b1=np.zeros(n_enc_hidden),
        W2=rng.normal(0.0, s2, size=(n_enc_hidden, K)),
        b2=np.zeros(K),
        mu=mu,
        W_dec=rng.normal(0.0, sd, size=(n_hidden, n_in)),
        b_dec=np.zeros(n_in),
        sigma=sigma,
    )


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def forward(model: PopulationCoder, X: np.ndarray) -> dict:
    """Forward pass on a batch X of shape (N, D)."""
    h_pre = X @ model.W1 + model.b1                    # (N, He)
    h = np.tanh(h_pre)                                 # (N, He)
    m_pre = h @ model.W2 + model.b2                    # (N, K)
    m_hat = _sigmoid(m_pre)                            # (N, K) in (0,1)
    d = model.mu[None, :, :] - m_hat[:, None, :]       # (N, H, K)
    r2 = (d ** 2).sum(axis=2)                          # (N, H)
    bump = np.exp(-r2 / (2.0 * model.sigma ** 2))      # (N, H)
    x_hat = bump @ model.W_dec + model.b_dec           # (N, D)
    return {
        "h_pre": h_pre, "h": h,
        "m_pre": m_pre, "m_hat": m_hat,
        "d": d, "r2": r2, "bump": bump,
        "x_hat": x_hat,
    }


def loss_and_grads(model: PopulationCoder, X: np.ndarray,
                   spread_penalty: float = 0.0) -> tuple:
    """Forward, then backprop. Returns (loss, grads, fwd_cache).

    ``spread_penalty`` is unused here (kept for API compat); the grid init
    keeps mu spread out automatically.
    """
    fwd = forward(model, X)
    N = X.shape[0]
    err = fwd["x_hat"] - X
    loss = 0.5 * (err ** 2).sum() / N

    # ---- backward ----
    dx_hat = err / N                                   # (N, D)
    dW_dec = fwd["bump"].T @ dx_hat                    # (H, D)
    db_dec = dx_hat.sum(axis=0)                        # (D,)
    d_bump = dx_hat @ model.W_dec.T                    # (N, H)

    dr2 = -d_bump * fwd["bump"] / (2.0 * model.sigma ** 2)   # (N, H)
    dd = 2.0 * fwd["d"] * dr2[:, :, None]              # (N, H, K)

    # d = mu - m_hat
    dmu = dd.sum(axis=0)                               # (H, K)
    dm_hat = -dd.sum(axis=1)                           # (N, K)

    # sigmoid backward: dm_pre = dm_hat * m_hat * (1 - m_hat)
    dm_pre = dm_hat * fwd["m_hat"] * (1.0 - fwd["m_hat"])  # (N, K)

    dW2 = fwd["h"].T @ dm_pre                          # (He, K)
    db2 = dm_pre.sum(axis=0)
    dh = dm_pre @ model.W2.T                           # (N, He)

    # tanh backward: dh_pre = dh * (1 - tanh^2)
    dh_pre = dh * (1.0 - fwd["h"] ** 2)
    dW1 = X.T @ dh_pre                                 # (D, He)
    db1 = dh_pre.sum(axis=0)

    grads = {
        "W1": dW1, "b1": db1,
        "W2": dW2, "b2": db2,
        "W_dec": dW_dec, "b_dec": db_dec,
        "mu": dmu,
    }
    return loss, grads, fwd


def description_length_loss(model: PopulationCoder, X: np.ndarray,
                            sigma_recon: float = 0.15) -> float:
    """A simple MDL-flavoured description length per image, in bits.

    For each image: encode the implicit position m_hat at sigma resolution
    inside a unit cube, then encode the residual under a Gaussian noise
    model with stdev sigma_recon.

    L_total = L_code + L_data
    L_code  = K * log2(1 / sigma)                       (bits to specify m_hat)
    L_data  = 0.5 D log2(2 pi sigma_recon^2)
              + ||x - x_hat||^2 / (2 sigma_recon^2 ln 2)

    Returns the mean over the batch. The number is interpretation-dependent
    (depends on sigma_recon and on the assumption that m_hat lives in a unit
    cube), so use it as a relative metric, not an absolute one. The Zemel &
    Hinton paper reports ~1.16 bits using a different (and more careful)
    bookkeeping.
    """
    fwd = forward(model, X)
    N, D = X.shape
    err = fwd["x_hat"] - X
    sse_per = (err ** 2).sum(axis=1)
    data_bits = (
        sse_per / (2.0 * sigma_recon ** 2 * np.log(2.0))
        + 0.5 * D * np.log2(2.0 * np.pi * sigma_recon ** 2)
    )
    code_bits = model.n_implicit * np.log2(1.0 / model.sigma)
    return float((data_bits + code_bits).mean())


# --------------------------------------------------------------------------
# Verification helpers
# --------------------------------------------------------------------------

def _expand_theta(theta: np.ndarray) -> np.ndarray:
    """Map theta in [0, pi) to (cos 2theta, sin 2theta) in R^2 (unit-circle)."""
    return np.stack([np.cos(2.0 * theta), np.sin(2.0 * theta)], axis=1)


def _poly_features(M: np.ndarray, degree: int = 1) -> np.ndarray:
    """Polynomial features up to ``degree`` over each column of M.

    Returns shape (N, F). F = 1 + K, 1 + K + K(K+1)/2, etc.
    """
    cols = [np.ones(len(M))]
    K = M.shape[1]
    if degree >= 1:
        for i in range(K):
            cols.append(M[:, i])
    if degree >= 2:
        for i in range(K):
            for j in range(i, K):
                cols.append(M[:, i] * M[:, j])
    if degree >= 3:
        for i in range(K):
            for j in range(i, K):
                for k in range(j, K):
                    cols.append(M[:, i] * M[:, j] * M[:, k])
    return np.column_stack(cols)


def implicit_space_recovery(model: PopulationCoder, X: np.ndarray,
                            params: np.ndarray, degree: int = 1) -> dict:
    """Fit a polynomial regression from m_hat to (x, y, cos2theta, sin2theta).

    With ``degree=1`` this is the linear linear-fit R^2 (used as a fast
    diagnostic during training). The cubic fit (``degree=3``) is the more
    honest measure: m_hat lives on a 3D manifold and the dipole's true
    parameters can be a smooth nonlinear function of m_hat.

    Returns R^2 per dimension and overall mean, plus the predicted m_hat.
    """
    fwd = forward(model, X)
    M = fwd["m_hat"]                                   # (N, K)
    x_t, y_t, theta = params[:, 0], params[:, 1], params[:, 2]
    target = np.column_stack([x_t, y_t, _expand_theta(theta)])  # (N, 4)

    P = _poly_features(M, degree=degree)
    coef, *_ = np.linalg.lstsq(P, target, rcond=None)
    pred = P @ coef
    ss_res = ((target - pred) ** 2).sum(axis=0)
    ss_tot = ((target - target.mean(axis=0)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    return {
        "m_hat": M,
        "r2_x": float(r2[0]),
        "r2_y": float(r2[1]),
        "r2_cos2theta": float(r2[2]),
        "r2_sin2theta": float(r2[3]),
        "r2_mean": float(r2.mean()),
        "degree": degree,
    }


# --------------------------------------------------------------------------
# Training
# --------------------------------------------------------------------------

def train(model: PopulationCoder,
          X: np.ndarray,
          n_epochs: int = 80,
          lr: float = 0.05,
          batch_size: int = 64,
          spread_penalty: float = 1e-3,
          eval_every: int = 1,
          params: Optional[np.ndarray] = None,
          snapshot_epochs: Optional[list] = None,
          verbose: bool = True,
          seed: int = 0) -> dict:
    """SGD with a constant learning rate on the population-code loss."""
    rng = np.random.default_rng(seed)
    N = X.shape[0]
    history = {
        "epoch": [],
        "loss": [],
        "recon_mse": [],
        "dl_bits": [],
        "r2_mean": [],
    }
    snapshots = {}

    for epoch in range(n_epochs):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        for start in range(0, N, batch_size):
            batch = X[idx[start:start + batch_size]]
            loss, grads, _ = loss_and_grads(model, batch,
                                            spread_penalty=spread_penalty)
            epoch_loss += float(loss) * batch.shape[0] / N
            sgd_update(model, grads, lr)

        if epoch % eval_every == 0 or epoch == n_epochs - 1:
            fwd = forward(model, X)
            mse = float(((fwd["x_hat"] - X) ** 2).mean())
            dl = description_length_loss(model, X)
            history["epoch"].append(epoch)
            history["loss"].append(epoch_loss)
            history["recon_mse"].append(mse)
            history["dl_bits"].append(dl)
            if params is not None:
                rec = implicit_space_recovery(model, X, params)
                history["r2_mean"].append(rec["r2_mean"])
            else:
                history["r2_mean"].append(float("nan"))

        if snapshot_epochs is not None and epoch in snapshot_epochs:
            snapshots[epoch] = model.copy()

        if verbose and (epoch % max(1, n_epochs // 10) == 0 or epoch == n_epochs - 1):
            tag = (f" r2={history['r2_mean'][-1]:.3f}"
                   if params is not None and history["r2_mean"] else "")
            print(f"epoch {epoch:4d}: loss={epoch_loss:.4f} "
                  f"mse={history['recon_mse'][-1]:.4f} "
                  f"dl={history['dl_bits'][-1]:.2f} bits{tag}")

    return {"history": history, "snapshots": snapshots}


def sgd_update(model: PopulationCoder, grads: dict, lr: float) -> None:
    model.W1    -= lr * grads["W1"]
    model.b1    -= lr * grads["b1"]
    model.W2    -= lr * grads["W2"]
    model.b2    -= lr * grads["b2"]
    model.W_dec -= lr * grads["W_dec"]
    model.b_dec -= lr * grads["b_dec"]
    model.mu    -= lr * grads["mu"]


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def _git_commit() -> str:
    try:
        out = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                      stderr=subprocess.DEVNULL).decode().strip()
        return out
    except Exception:
        return "unknown"


def main():
    parser = argparse.ArgumentParser(
        description="Dipole 3D-constraint population code (Zemel & Hinton 1995)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-hidden", type=int, default=225)
    parser.add_argument("--n-implicit", type=int, default=3)
    parser.add_argument("--n-enc-hidden", type=int, default=64)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.18,
                        help="bump width in implicit space")
    parser.add_argument("--out-json", type=str, default=None,
                        help="optional path to write a results JSON")
    args = parser.parse_args()

    np.random.seed(args.seed)

    print("Generating data...")
    X, params = generate_dipole_images(args.n_train, seed=args.seed)
    Xt, params_t = generate_dipole_images(500, seed=args.seed + 1000)

    print("Building model...")
    model = build_population_coder(n_hidden=args.n_hidden,
                                   n_implicit_dims=args.n_implicit,
                                   n_enc_hidden=args.n_enc_hidden,
                                   sigma=args.sigma,
                                   seed=args.seed)

    print("Training...")
    t0 = time.time()
    out = train(model, X,
                n_epochs=args.n_epochs,
                lr=args.lr,
                batch_size=args.batch_size,
                params=params,
                seed=args.seed)
    train_time = time.time() - t0

    rec1 = implicit_space_recovery(model, Xt, params_t, degree=1)
    rec3 = implicit_space_recovery(model, Xt, params_t, degree=3)
    dl = description_length_loss(model, Xt)
    fwd = forward(model, Xt)
    test_mse = float(((fwd["x_hat"] - Xt) ** 2).mean())

    # SVD of m_hat to confirm 3D usage
    M = fwd["m_hat"] - fwd["m_hat"].mean(axis=0)
    svs = np.linalg.svd(M, compute_uv=False)

    print()
    print(f"Train time:               {train_time:.2f} s")
    print(f"Test MSE:                 {test_mse:.4f}  "
          f"(naive predict-the-mean: {float(Xt.var(axis=0).sum() / Xt.shape[1]):.4f})")
    print(f"Description length:       {dl:.2f} bits/image (relative MDL proxy)")
    print(f"m_hat singular values:    {svs}")
    print(f"R^2 linear (degree=1):    x={rec1['r2_x']:.3f}  y={rec1['r2_y']:.3f}  "
          f"cos2theta={rec1['r2_cos2theta']:.3f}  sin2theta={rec1['r2_sin2theta']:.3f}  "
          f"mean={rec1['r2_mean']:.3f}")
    print(f"R^2 cubic (degree=3):     x={rec3['r2_x']:.3f}  y={rec3['r2_y']:.3f}  "
          f"cos2theta={rec3['r2_cos2theta']:.3f}  sin2theta={rec3['r2_sin2theta']:.3f}  "
          f"mean={rec3['r2_mean']:.3f}")

    if args.out_json:
        results = {
            "args": vars(args),
            "train_time_sec": train_time,
            "test_mse": test_mse,
            "test_dl_bits": dl,
            "m_hat_singular_values": svs.tolist(),
            "r2_linear": {
                "x": rec1["r2_x"], "y": rec1["r2_y"],
                "cos2theta": rec1["r2_cos2theta"],
                "sin2theta": rec1["r2_sin2theta"],
                "mean": rec1["r2_mean"],
            },
            "r2_cubic": {
                "x": rec3["r2_x"], "y": rec3["r2_y"],
                "cos2theta": rec3["r2_cos2theta"],
                "sin2theta": rec3["r2_sin2theta"],
                "mean": rec3["r2_mean"],
            },
            "env": {
                "python": sys.version,
                "numpy": np.__version__,
                "platform": platform.platform(),
                "processor": platform.processor(),
                "git_commit": _git_commit(),
            },
        }
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Wrote {args.out_json}")


if __name__ == "__main__":
    main()
