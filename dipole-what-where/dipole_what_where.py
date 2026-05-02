"""
Dipole what / where (Zemel & Hinton 1995).

The first explicit "what / where" toy in Hinton's corpus. We render 8x8 binary
images of *either* a horizontal bar (at random row y) *or* a vertical bar (at
random column x), 50/50 mix. A 64-100-2-100-64 autoencoder is trained with
description-length pressure on its 2-D bottleneck:

  L = mean_pixel_BCE(x, decoder(z + noise))    # reconstruction under noise
    + lambda_mdl * 0.5 * ||z||^2                # cost to send the position

The noise on z (Gaussian, fixed sigma_z) is the key device: a code is "useful"
only if a small ball around it decodes correctly. Discontinuous image families
(an h-bar at y=3 vs. a v-bar at x=3) cannot live in the same neighbourhood of
implicit space, because the decoder must produce two qualitatively different
images for nearby codes. The MDL prior keeps the cloud compact. The result:
the network places h-bars in one region and v-bars in another, perpendicular
region of the 2-D space, with the in-class position varying smoothly along
each region's local axis.

This is the *discontinuous* sibling of `dipole-position`: there is no smooth
parameter that turns a horizontal bar into a vertical one, so the optimal
2-D layout is two perpendicular line segments, not a connected manifold.

Source:
  R. S. Zemel & G. E. Hinton, "Learning population codes by minimizing
  description length", Neural Computation 7(3):549-564, 1995.

Architecture:
  encoder: 64 input -> Linear -> 100 hidden (sigmoid) -> Linear -> 2 implicit
  decoder: 2 implicit -> Linear -> 100 hidden (sigmoid) -> Linear -> 64 logits
  output:  sigmoid(logits)

Spec-required helpers exported from this module:
  generate_bars, build_population_coder, description_length_loss,
  visualize_implicit_space.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def generate_bars(n_samples: int, h: int = 8, w: int = 8,
                  bar_sigma: float = 0.7,
                  rng: np.random.Generator | None = None
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate balanced horizontal / vertical bar images with continuous position.

    For each sample, with prob 1/2 draw a horizontal bar with continuous row
    centre y in [0, h-1], else a vertical bar with continuous column centre
    x in [0, w-1]. Pixel intensity falls off as a 1-D Gaussian of width
    `bar_sigma` from the bar's centre row (resp. column), full intensity
    along the orthogonal direction. Image values are floats in [0, 1].

    The continuous bar position is what gives the implicit space *within*-class
    smoothness: an h-bar at y=3.0 and one at y=3.2 share most of their pixel
    mass, so the autoencoder is rewarded for placing nearby positions at
    nearby codes. Across class (h vs v) the pixel overlap is much smaller,
    which is what makes the eventual cluster split a *discontinuity*.

    Returns
    -------
    images   : (n, h*w) float32, flattened, intensity in [0, 1].
    orient   : (n,) int32 -- 0 for horizontal, 1 for vertical.
    position : (n,) float32 -- continuous y for horizontal, continuous x
               for vertical.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    images = np.zeros((n_samples, h, w), dtype=np.float32)
    orient = rng.integers(0, 2, size=n_samples).astype(np.int32)
    position = np.empty(n_samples, dtype=np.float32)
    rows = np.arange(h, dtype=np.float32)
    cols = np.arange(w, dtype=np.float32)
    for i in range(n_samples):
        if orient[i] == 0:                    # horizontal bar at row y
            y = float(rng.uniform(0.0, h - 1.0))
            profile = np.exp(-0.5 * ((rows - y) / bar_sigma) ** 2)
            images[i] = profile[:, None] * np.ones(w, dtype=np.float32)
            position[i] = y
        else:                                  # vertical bar at column x
            x = float(rng.uniform(0.0, w - 1.0))
            profile = np.exp(-0.5 * ((cols - x) / bar_sigma) ** 2)
            images[i] = np.ones(h, dtype=np.float32)[:, None] * profile[None, :]
            position[i] = x
    return images.reshape(n_samples, h * w), orient, position


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def bce_with_logits(logits: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Numerically stable elementwise BCE with logits."""
    return np.maximum(logits, 0) - logits * target + np.log1p(np.exp(-np.abs(logits)))


class WhatWhereCoder:
    """64-100-2-100-64 autoencoder with noisy bottleneck.

    Forward pass during training adds Gaussian noise of std `sigma_z` to z
    before decoding (`encode_decode_noisy`). At evaluation time `encode` is
    deterministic.
    """

    def __init__(self, n_input: int = 64,
                 n_hidden: int = 100,
                 n_implicit: int = 2,
                 sigma_z: float = 0.3,
                 init_scale: float = 0.1,
                 seed: int = 0):
        rng = np.random.default_rng(seed)
        s = init_scale
        self.rng = rng
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_implicit = n_implicit
        self.sigma_z = sigma_z

        # Encoder
        self.W1 = (s * rng.standard_normal((n_input, n_hidden))).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = (s * rng.standard_normal((n_hidden, n_implicit))).astype(np.float32)
        self.b2 = np.zeros(n_implicit, dtype=np.float32)
        # Decoder
        self.W3 = (s * rng.standard_normal((n_implicit, n_hidden))).astype(np.float32)
        self.b3 = np.zeros(n_hidden, dtype=np.float32)
        self.W4 = (s * rng.standard_normal((n_hidden, n_input))).astype(np.float32)
        self.b4 = np.zeros(n_input, dtype=np.float32)

    # ---- forward pieces --------------------------------------------------

    def encode(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h1 = sigmoid(x @ self.W1 + self.b1)
        z = h1 @ self.W2 + self.b2
        return z, h1

    def decode(self, z: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        h2 = sigmoid(z @ self.W3 + self.b3)
        logits = h2 @ self.W4 + self.b4
        return sigmoid(logits), h2, logits

    def forward(self, x: np.ndarray, noise: np.ndarray | None = None) -> dict:
        z, h1 = self.encode(x)
        if noise is not None:
            z_used = z + noise
        else:
            z_used = z
        x_hat, h2, logits = self.decode(z_used)
        return {"x": x, "h1": h1, "z": z, "z_used": z_used,
                "h2": h2, "logits": logits, "x_hat": x_hat}

    # ---- backward / gradient --------------------------------------------

    def loss_and_grads(self, x: np.ndarray, lambda_mdl: float,
                       inject_noise: bool = True
                       ) -> tuple[dict, dict]:
        """One forward + backward pass on a mini-batch.

        Returns (metrics, grads).
        """
        n = x.shape[0]
        if inject_noise and self.sigma_z > 0:
            noise = self.sigma_z * self.rng.standard_normal(
                (n, self.n_implicit)).astype(np.float32)
        else:
            noise = None
        cache = self.forward(x, noise=noise)
        h1, z, z_used, h2, logits, x_hat = (cache["h1"], cache["z"],
                                            cache["z_used"], cache["h2"],
                                            cache["logits"], cache["x_hat"])

        # ---- losses ----
        bce = bce_with_logits(logits, x)
        recon = float(bce.mean())
        mdl = float(0.5 * (z ** 2).sum(axis=1).mean() / max(self.n_implicit, 1))
        total = recon + lambda_mdl * mdl

        # ---- gradients ----
        d_logits = (x_hat - x) / float(n * self.n_input)

        # Decoder
        dW4 = h2.T @ d_logits
        db4 = d_logits.sum(axis=0)
        d_h2 = d_logits @ self.W4.T
        d_pre_h2 = d_h2 * h2 * (1.0 - h2)
        dW3 = z_used.T @ d_pre_h2
        db3 = d_pre_h2.sum(axis=0)
        d_z_used_recon = d_pre_h2 @ self.W3.T

        # Noise is independent of z, so d z_used / d z = I and the recon
        # gradient on z is the same as on z_used.
        d_z_recon = d_z_used_recon

        # MDL gradient: d/dz of 0.5 ||z||^2 / n_implicit, batch-averaged.
        d_z_mdl = lambda_mdl * z / float(n * max(self.n_implicit, 1))
        d_z = d_z_recon + d_z_mdl

        # Encoder
        dW2 = h1.T @ d_z
        db2 = d_z.sum(axis=0)
        d_h1 = d_z @ self.W2.T
        d_pre_h1 = d_h1 * h1 * (1.0 - h1)
        dW1 = x.T @ d_pre_h1
        db1 = d_pre_h1.sum(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2,
                 "W3": dW3, "b3": db3, "W4": dW4, "b4": db4}
        metrics = {"loss": total, "recon": recon, "mdl": mdl}
        return metrics, grads

    # ---- parameter access ------------------------------------------------

    def params(self) -> dict:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2,
                "W3": self.W3, "b3": self.b3, "W4": self.W4, "b4": self.b4}


def build_population_coder(n_hidden: int = 100, n_implicit_dims: int = 2,
                           seed: int = 0,
                           init_scale: float = 0.1,
                           sigma_z: float = 0.3) -> WhatWhereCoder:
    """Spec-required factory.

    Builds the noisy-bottleneck autoencoder. The 2-D implicit space is the
    learned bottleneck z; sigma_z controls how aggressively neighbouring
    codes must reconstruct alike (the "MDL precision" of the population
    code).
    """
    return WhatWhereCoder(n_input=64, n_hidden=n_hidden,
                          n_implicit=n_implicit_dims, seed=seed,
                          init_scale=init_scale, sigma_z=sigma_z)


def description_length_loss(model: WhatWhereCoder, data: np.ndarray,
                            lambda_mdl: float = 0.05) -> dict:
    """Spec-required helper.

    Returns {'loss', 'recon', 'mdl'} on `data` (n, 64) using model's current
    parameters. Evaluation does NOT inject noise (so the reported recon is
    the deterministic-decode reconstruction).
    """
    metrics, _ = model.loss_and_grads(data, lambda_mdl=lambda_mdl,
                                      inject_noise=False)
    return metrics


# ----------------------------------------------------------------------
# Optimizer
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict, lr: float = 1e-2,
                 beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8):
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.t = 0
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}

    def step(self, params: dict, grads: dict) -> None:
        self.t += 1
        b1, b2 = self.beta1, self.beta2
        bc1 = 1.0 - b1 ** self.t
        bc2 = 1.0 - b2 ** self.t
        for k in params:
            g = grads[k]
            self.m[k] = b1 * self.m[k] + (1 - b1) * g
            self.v[k] = b2 * self.v[k] + (1 - b2) * (g * g)
            m_hat = self.m[k] / bc1
            v_hat = self.v[k] / bc2
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


# ----------------------------------------------------------------------
# Visualization
# ----------------------------------------------------------------------

def _ridge_classifier_accuracy(z: np.ndarray, label: np.ndarray,
                               ridge: float = 1e-2) -> float:
    """Closed-form ridge-regression linear classifier accuracy.

    Used to score how cleanly the orientation can be read off from z by a
    *linear* probe. For two perpendicular line clusters meeting at the
    origin this returns ~0.5 (chance); for two well-separated clusters
    living in different half-planes it returns ~1.0.
    """
    n = z.shape[0]
    X = np.hstack([z, np.ones((n, 1), dtype=z.dtype)])
    y = (2 * label.astype(np.float32) - 1.0)               # +1 / -1
    A = X.T @ X + ridge * np.eye(X.shape[1], dtype=X.dtype)
    b = X.T @ y
    w = np.linalg.solve(A, b)
    pred = (X @ w > 0).astype(np.int32)
    return float((pred == label).mean())


def _orientation_axis_angle(z: np.ndarray, label: np.ndarray) -> float:
    """Angle (degrees) between the principal axes of the H and V code clouds.

    For the canonical "what / where" cross structure this is close to 90.
    Computed via SVD on each centred class.
    """
    angles = []
    for c in (0, 1):
        zc = z[label == c]
        if len(zc) < 2:
            return float("nan")
        zc = zc - zc.mean(axis=0, keepdims=True)
        _, _, vh = np.linalg.svd(zc, full_matrices=False)
        angles.append(vh[0])                                # 1st right singvec
    cos = float(abs(angles[0] @ angles[1]) /
                (np.linalg.norm(angles[0]) * np.linalg.norm(angles[1]) + 1e-9))
    cos = min(1.0, max(0.0, cos))
    return float(np.degrees(np.arccos(cos)))


def visualize_implicit_space(model: WhatWhereCoder, images: np.ndarray,
                             orient: np.ndarray) -> dict:
    """Encode `images` (deterministic, no noise), return scatter-ready data
    plus several "is this discontinuously clustered?" diagnostics.

    Returned diagnostics
    --------------------
    cluster_separation  : centre-difference / within-class spread along the
                          centre-difference axis. Catches "in opposite
                          corners" geometries.
    linear_separability : ridge-regression classifier accuracy. Catches the
                          same "opposite corners" geometry.
    axis_angle_deg      : angle between the principal axes of the H and V
                          point clouds. Catches the *cross* geometry (two
                          perpendicular 1-D manifolds sharing the origin).
                          90 degrees = maximally orthogonal what/where split.
    """
    z, _ = model.encode(images)
    out = {"z": z, "orient": orient}
    if model.n_implicit < 2:
        out["cluster_separation"] = float("nan")
        out["linear_separability"] = float("nan")
        out["axis_angle_deg"] = float("nan")
        return out
    mu_h = z[orient == 0].mean(axis=0)
    mu_v = z[orient == 1].mean(axis=0)
    diff = mu_v - mu_h
    norm = float(np.linalg.norm(diff))
    if norm < 1e-9:
        sep = 0.0
    else:
        axis = diff / norm
        proj = z @ axis
        s_h = float(np.std(proj[orient == 0]))
        s_v = float(np.std(proj[orient == 1]))
        sep = norm / (s_h + s_v + 1e-6)
    out["mu_h"] = mu_h
    out["mu_v"] = mu_v
    out["cluster_separation"] = float(sep)
    out["linear_separability"] = _ridge_classifier_accuracy(z, orient)
    out["axis_angle_deg"] = _orientation_axis_angle(z, orient)
    return out


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 150,
          n_train: int = 2000,
          batch_size: int = 64,
          lr: float = 5e-3,
          lambda_mdl: float = 0.05,
          n_hidden: int = 100,
          n_implicit: int = 2,
          sigma_z: float = 0.5,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 1,
          verbose: bool = True) -> tuple[WhatWhereCoder, dict, dict]:
    """Train the noisy AE on a fresh balanced bars dataset.

    Returns (model, history, fixed_data).
    """
    rng = np.random.default_rng(seed)
    train_images, train_orient, train_pos = generate_bars(n_train, rng=rng)
    eval_images, eval_orient, eval_pos = generate_bars(400, rng=rng)

    model = build_population_coder(n_hidden=n_hidden,
                                   n_implicit_dims=n_implicit,
                                   seed=seed,
                                   sigma_z=sigma_z)
    opt = Adam(model.params(), lr=lr)

    history = {"epoch": [], "loss": [], "recon": [], "mdl": [],
               "cluster_separation": []}

    if verbose:
        print(f"# what/where AE: {n_train} train, {n_hidden} hidden, "
              f"{n_implicit}-D implicit, lambda_mdl={lambda_mdl}, "
              f"sigma_z={sigma_z}")

    n_batches = max(1, n_train // batch_size)
    for epoch in range(n_epochs):
        t0 = time.time()
        order = rng.permutation(n_train)
        epoch_loss = 0.0
        epoch_recon = 0.0
        epoch_mdl = 0.0
        for b in range(n_batches):
            idx = order[b * batch_size: (b + 1) * batch_size]
            x = train_images[idx]
            metrics, grads = model.loss_and_grads(x, lambda_mdl=lambda_mdl,
                                                  inject_noise=True)
            opt.step(model.params(), grads)
            epoch_loss += metrics["loss"]
            epoch_recon += metrics["recon"]
            epoch_mdl += metrics["mdl"]
        epoch_loss /= n_batches
        epoch_recon /= n_batches
        epoch_mdl /= n_batches

        info = visualize_implicit_space(model, eval_images, eval_orient)
        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss)
        history["recon"].append(epoch_recon)
        history["mdl"].append(epoch_mdl)
        history["cluster_separation"].append(info["cluster_separation"])
        history.setdefault("linear_separability", [])
        history.setdefault("axis_angle_deg", [])
        history["linear_separability"].append(info["linear_separability"])
        history["axis_angle_deg"].append(info["axis_angle_deg"])

        if verbose and (epoch % 10 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:3d}  loss={epoch_loss:.4f}  "
                  f"recon={epoch_recon:.4f}  mdl={epoch_mdl:.4f}  "
                  f"lin_sep={info['linear_separability']:.2f}  "
                  f"angle={info['axis_angle_deg']:.0f}deg  "
                  f"({time.time()-t0:.2f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history,
                              eval_images, eval_orient, eval_pos)

    fixed_data = {"images": eval_images, "orient": eval_orient,
                  "position": eval_pos}
    return model, history, fixed_data


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=150)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--lambda-mdl", type=float, default=0.05)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-implicit", type=int, default=2)
    p.add_argument("--sigma-z", type=float, default=0.5)
    args = p.parse_args()

    model, history, eval_data = train(n_epochs=args.n_epochs,
                                      n_train=args.n_train,
                                      batch_size=args.batch_size,
                                      lr=args.lr,
                                      lambda_mdl=args.lambda_mdl,
                                      n_hidden=args.n_hidden,
                                      n_implicit=args.n_implicit,
                                      sigma_z=args.sigma_z,
                                      seed=args.seed)

    info = visualize_implicit_space(model, eval_data["images"],
                                    eval_data["orient"])
    print(f"\nFinal diagnostics:")
    print(f"  cluster_separation (centre/spread): {info['cluster_separation']:.2f}")
    print(f"  linear_separability (ridge):        {info['linear_separability']:.2f}")
    print(f"  axis angle (degrees):               {info['axis_angle_deg']:.0f}")
    print(f"  horizontal cluster mean: {info['mu_h']}")
    print(f"  vertical   cluster mean: {info['mu_v']}")


if __name__ == "__main__":
    main()
