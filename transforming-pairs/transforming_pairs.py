"""
Gated three-way conditional RBM trained on (x, y) pairs of binary
random-dot images, where y is x transformed by a known transformation
drawn from {shifts of +/- k pixels, rotations of multiples of 90 deg}.

Reference: Memisevic & Hinton, "Unsupervised learning of image
transformations", CVPR 2007.

Architecture
============

x : binary input image, length n_in (clamped)
y : binary output image, length n_out
h : binary hidden, length n_hidden
The energy of (y, h) given x is

    E(y, h | x) = - sum_f a_x_f * a_y_f * a_h_f
                  - b_y . y - b_h . h

where the per-factor projections are

    a_x = Wx^T x   (shape F)
    a_y = Wy^T y   (shape F)
    a_h = Wh^T h   (shape F)

This is the factored form of the third-order weight tensor
W_{i,o,j} = sum_f Wx_{i,f} * Wy_{o,f} * Wh_{j,f}, which is what makes the
parameter count linear in F instead of cubic.

Conditional distributions follow directly:

    p(h_j = 1 | x, y) = sigmoid( ( (Wx^T x) * (Wy^T y) ) @ Wh[j, :] + b_h_j )
    p(y_o = 1 | x, h) = sigmoid( ( (Wx^T x) * (Wh^T h) ) @ Wy[o, :] + b_y_o )

We treat x as clamped (the model is conditional on x), so CD-1 alternates
between (h | x, y) and (y | x, h) and the gradient is the standard
contrastive-divergence form.

Training
========

CD-1 with mean-field hidden units. Per minibatch:

    a_x      = X  @ Wx                              # (B, F)
    a_y_pos  = Y  @ Wy                              # (B, F)
    h_pos    = sigmoid((a_x * a_y_pos) @ Wh.T + b_h)  # (B, H)
    a_h_pos  = h_pos @ Wh                           # (B, F)
    Y_neg_p  = sigmoid((a_x * a_h_pos) @ Wy.T + b_y)  # (B, n_out)
    Y_neg    = bernoulli(Y_neg_p)
    a_y_neg  = Y_neg @ Wy                           # (B, F)
    h_neg    = sigmoid((a_x * a_y_neg) @ Wh.T + b_h)  # (B, H)
    a_h_neg  = h_neg @ Wh                           # (B, F)

    dWx = X.T @ (a_y_pos * a_h_pos  -  a_y_neg * a_h_neg) / B
    dWy = Y.T @ (a_x     * a_h_pos) / B
        - Y_neg.T @ (a_x * a_h_neg) / B
    dWh = h_pos.T @ (a_x * a_y_pos) / B
        - h_neg.T @ (a_x * a_y_neg) / B
    db_y = (Y - Y_neg).mean(0)
    db_h = (h_pos - h_neg).mean(0)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def build_transform_pool(transforms: tuple[str, ...] = ("shift", "rotate"),
                          shift_max: int = 2) -> list[tuple]:
    """List of transformation descriptors used for sampling pairs.

    Each entry is (name, *params). For "shift", params are (dx, dy)
    integer pixel offsets in [-shift_max, shift_max] excluding (0, 0).
    For "rotate", params are (k,) where k in {1, 2, 3} (rot90 multiples).
    """
    pool: list[tuple] = []
    if "shift" in transforms:
        for dy in range(-shift_max, shift_max + 1):
            for dx in range(-shift_max, shift_max + 1):
                if dx == 0 and dy == 0:
                    continue
                pool.append(("shift", dx, dy))
    if "rotate" in transforms:
        for k in (1, 2, 3):
            pool.append(("rotate", k))
    if not pool:
        raise ValueError(f"empty transform pool for transforms={transforms}")
    return pool


def apply_transform(img: np.ndarray, descriptor: tuple) -> np.ndarray:
    name = descriptor[0]
    if name == "shift":
        _, dx, dy = descriptor
        return np.roll(img, shift=(dy, dx), axis=(0, 1))
    if name == "rotate":
        _, k = descriptor
        return np.rot90(img, k=k).copy()
    raise ValueError(f"unknown transform: {descriptor}")


def transform_label(descriptor: tuple) -> str:
    if descriptor[0] == "shift":
        return f"shift({descriptor[1]:+d},{descriptor[2]:+d})"
    return f"rot{90 * descriptor[1]}"


def generate_transformed_pairs(n_samples: int,
                                h: int = 13,
                                w: int = 13,
                                transforms: tuple[str, ...] = ("shift", "rotate"),
                                dot_density: float = 0.10,
                                shift_max: int = 2,
                                rng: np.random.Generator | None = None,
                                ) -> tuple[np.ndarray, np.ndarray,
                                            np.ndarray, list[tuple]]:
    """Generate `n_samples` pairs (x, y) where y = transform(x).

    Returns
    -------
    X : (n_samples, h*w) float32 in {0., 1.}
    Y : (n_samples, h*w) float32 in {0., 1.}
    transform_ids : (n_samples,) int64, index into the returned pool
    pool : list of transformation descriptors
    """
    if rng is None:
        rng = np.random.default_rng(0)
    pool = build_transform_pool(transforms, shift_max=shift_max)
    X = np.zeros((n_samples, h * w), dtype=np.float32)
    Y = np.zeros((n_samples, h * w), dtype=np.float32)
    ids = np.zeros(n_samples, dtype=np.int64)
    for n in range(n_samples):
        img = (rng.random((h, w)) < dot_density).astype(np.float32)
        # Reject empty patterns -- they carry no transformation signal.
        while img.sum() == 0:
            img = (rng.random((h, w)) < dot_density).astype(np.float32)
        idx = int(rng.integers(len(pool)))
        y_img = apply_transform(img, pool[idx])
        X[n] = img.ravel()
        Y[n] = y_img.ravel()
        ids[n] = idx
    return X, Y, ids, pool


# ----------------------------------------------------------------------
# Gated three-way RBM
# ----------------------------------------------------------------------

def sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50.0, 50.0)))


class GatedRBM:
    """Factored (Wx, Wy, Wh) three-way conditional RBM."""

    def __init__(self,
                 n_in: int,
                 n_out: int,
                 n_hidden: int,
                 n_factors: int,
                 init_scale: float = 0.02,
                 seed: int = 0):
        self.n_in = n_in
        self.n_out = n_out
        self.n_hidden = n_hidden
        self.n_factors = n_factors
        self.rng = np.random.default_rng(seed)
        s = init_scale
        self.Wx = (s * self.rng.standard_normal((n_in, n_factors))).astype(np.float32)
        self.Wy = (s * self.rng.standard_normal((n_out, n_factors))).astype(np.float32)
        self.Wh = (s * self.rng.standard_normal((n_hidden, n_factors))).astype(np.float32)
        self.b_y = np.zeros(n_out, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)

    # --- conditional distributions -----------------------------------

    def hidden_prob(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        a_x = X @ self.Wx
        a_y = Y @ self.Wy
        return sigmoid((a_x * a_y) @ self.Wh.T + self.b_h)

    def output_prob(self, X: np.ndarray, h: np.ndarray) -> np.ndarray:
        a_x = X @ self.Wx
        a_h = h @ self.Wh
        return sigmoid((a_x * a_h) @ self.Wy.T + self.b_y)

    def sample(self, p: np.ndarray) -> np.ndarray:
        return (self.rng.random(p.shape) < p).astype(np.float32)

    # --- one CD-1 step on conditional p(y, h | x) --------------------

    def cd_step(self, X: np.ndarray, Y: np.ndarray
                ) -> tuple[dict[str, np.ndarray], dict[str, float]]:
        """Run one CD-1 update; return parameter gradients and metrics."""
        B = X.shape[0]

        a_x = X @ self.Wx                            # (B, F)
        a_y_pos = Y @ self.Wy                        # (B, F)
        h_pos = sigmoid((a_x * a_y_pos) @ self.Wh.T + self.b_h)  # (B, H)
        a_h_pos = h_pos @ self.Wh                    # (B, F)

        Y_neg_p = sigmoid((a_x * a_h_pos) @ self.Wy.T + self.b_y)  # (B, n_out)
        Y_neg = self.sample(Y_neg_p)
        a_y_neg = Y_neg @ self.Wy
        h_neg = sigmoid((a_x * a_y_neg) @ self.Wh.T + self.b_h)
        a_h_neg = h_neg @ self.Wh

        dWx = (X.T @ (a_y_pos * a_h_pos)
               - X.T @ (a_y_neg * a_h_neg)) / B
        dWy = (Y.T @ (a_x * a_h_pos)
               - Y_neg.T @ (a_x * a_h_neg)) / B
        dWh = (h_pos.T @ (a_x * a_y_pos)
               - h_neg.T @ (a_x * a_y_neg)) / B
        db_y = (Y - Y_neg_p).mean(axis=0)
        db_h = (h_pos - h_neg).mean(axis=0)

        recon_mse = float(((Y - Y_neg_p) ** 2).mean())
        h_pos_mean = float(h_pos.mean())

        grads = dict(Wx=dWx.astype(np.float32), Wy=dWy.astype(np.float32),
                     Wh=dWh.astype(np.float32),
                     b_y=db_y.astype(np.float32),
                     b_h=db_h.astype(np.float32))
        metrics = dict(recon_mse=recon_mse, h_pos_mean=h_pos_mean)
        return grads, metrics

    # --- inference helper for transfer experiments -------------------

    def transfer(self, X_query: np.ndarray, X_ref: np.ndarray,
                 Y_ref: np.ndarray, n_gibbs: int = 1) -> np.ndarray:
        """Apply the transformation inferred from (X_ref, Y_ref) to X_query.

        Steps:
          h = E[h | X_ref, Y_ref]
          y_q = E[y | X_query, h]   (one mean-field step is fine)
        Optionally clamp X_query and rerun (h | X_query, y_q) -> y_q for
        n_gibbs - 1 additional refinements.
        """
        h = self.hidden_prob(X_ref, Y_ref)
        Y_q = self.output_prob(X_query, h)
        for _ in range(n_gibbs - 1):
            h = self.hidden_prob(X_query, Y_q)
            Y_q = self.output_prob(X_query, h)
        return Y_q


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def build_gated_rbm(n_in: int, n_out: int, n_hidden: int,
                     n_factors: int, init_scale: float = 0.02,
                     seed: int = 0) -> GatedRBM:
    return GatedRBM(n_in, n_out, n_hidden, n_factors,
                    init_scale=init_scale, seed=seed)


def train(model: GatedRBM,
          X: np.ndarray,
          Y: np.ndarray,
          n_epochs: int = 30,
          lr: float = 0.05,
          momentum: float = 0.5,
          weight_decay: float = 1e-4,
          batch_size: int = 100,
          eval_fn=None,
          eval_every: int = 1,
          snapshot_callback=None,
          snapshot_every: int = 1,
          verbose: bool = True,
          ) -> dict:
    """SGD with momentum on CD-1 gradients.

    `eval_fn(model, epoch)` is called every `eval_every` epochs and may
    return a dict of scalar metrics that are merged into the history.
    """
    n = X.shape[0]
    rng = np.random.default_rng(int(model.rng.integers(2**31 - 1)))

    vWx = np.zeros_like(model.Wx)
    vWy = np.zeros_like(model.Wy)
    vWh = np.zeros_like(model.Wh)
    vby = np.zeros_like(model.b_y)
    vbh = np.zeros_like(model.b_h)

    history: dict[str, list] = {
        "epoch": [], "recon_mse": [], "weight_norm": [],
        "h_mean": [], "lr": [],
    }

    for epoch in range(n_epochs):
        t0 = time.time()
        order = rng.permutation(n)
        epoch_recon = []
        epoch_h_mean = []
        for s in range(0, n, batch_size):
            idx = order[s:s + batch_size]
            grads, metrics = model.cd_step(X[idx], Y[idx])
            vWx = momentum * vWx + lr * (grads["Wx"] - weight_decay * model.Wx)
            vWy = momentum * vWy + lr * (grads["Wy"] - weight_decay * model.Wy)
            vWh = momentum * vWh + lr * (grads["Wh"] - weight_decay * model.Wh)
            vby = momentum * vby + lr * grads["b_y"]
            vbh = momentum * vbh + lr * grads["b_h"]
            model.Wx += vWx
            model.Wy += vWy
            model.Wh += vWh
            model.b_y += vby
            model.b_h += vbh
            epoch_recon.append(metrics["recon_mse"])
            epoch_h_mean.append(metrics["h_pos_mean"])

        wnorm = float(np.linalg.norm(model.Wx) + np.linalg.norm(model.Wy)
                      + np.linalg.norm(model.Wh))
        history["epoch"].append(epoch + 1)
        history["recon_mse"].append(float(np.mean(epoch_recon)))
        history["weight_norm"].append(wnorm)
        history["h_mean"].append(float(np.mean(epoch_h_mean)))
        history["lr"].append(lr)

        if eval_fn is not None and ((epoch + 1) % eval_every == 0
                                    or epoch == n_epochs - 1):
            extra = eval_fn(model, epoch + 1)
            target_len = len(history["epoch"])
            for k, v in extra.items():
                history.setdefault(k, [])
                # Pad with None up to one short of the target, then append v
                # so v lands at index `target_len - 1` (same as the current
                # epoch entry).
                while len(history[k]) < target_len - 1:
                    history[k].append(None)
                history[k].append(v)
            # Pad any other auxiliary keys that weren't set this epoch.
            for k, v in history.items():
                if k in ("epoch", "recon_mse", "weight_norm",
                         "h_mean", "lr"):
                    continue
                while len(v) < target_len:
                    v.append(None)
        else:
            # If eval didn't run this epoch, still extend any existing
            # auxiliary keys with None so they stay aligned with epoch.
            target_len = len(history["epoch"])
            for k, v in history.items():
                if k in ("epoch", "recon_mse", "weight_norm",
                         "h_mean", "lr"):
                    continue
                while len(v) < target_len:
                    v.append(None)

        if verbose and (epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0
                        or epoch == n_epochs - 1):
            extra_msg = ""
            if "transform_acc" in history and history["transform_acc"][-1] is not None:
                extra_msg = f"  xform_acc={history['transform_acc'][-1] * 100:5.1f}%"
            print(f"epoch {epoch + 1:4d}/{n_epochs}  "
                  f"recon={history['recon_mse'][-1]:.4f}  "
                  f"|W|={wnorm:.2f}  "
                  f"<h>={history['h_mean'][-1]:.3f}{extra_msg}  "
                  f"({time.time() - t0:.2f}s)")

        if snapshot_callback is not None and ((epoch + 1) % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history)

    return history


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def reconstruction_metrics(model: GatedRBM,
                            X: np.ndarray,
                            Y: np.ndarray) -> dict[str, float]:
    """Mean reconstruction probability of y given x (using one mean-field
    pass through h)."""
    h = model.hidden_prob(X, Y)
    Y_p = model.output_prob(X, h)
    mse = float(((Y - Y_p) ** 2).mean())
    bit_acc = float(((Y_p > 0.5).astype(np.float32) == Y).mean())
    return dict(recon_mse=mse, bit_acc=bit_acc)


def transform_classification_accuracy(model: GatedRBM,
                                       X: np.ndarray,
                                       Y: np.ndarray,
                                       ids: np.ndarray,
                                       n_classes: int,
                                       n_train: int = 800,
                                       reg: float = 1.0,
                                       rng: np.random.Generator | None = None
                                       ) -> tuple[float, np.ndarray]:
    """Train a multinomial logistic regression on hidden activations and
    report classification accuracy on the held-out portion.

    Returns
    -------
    accuracy : float in [0, 1]
    confusion : (n_classes, n_classes) int array, rows = true, cols = pred
    """
    if rng is None:
        rng = np.random.default_rng(0)
    H = model.hidden_prob(X, Y)
    n = H.shape[0]
    perm = rng.permutation(n)
    n_train = min(n_train, n // 2)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    # Closed-form is unavailable for multinomial logreg; use a few epochs of
    # SGD with cross-entropy. Tiny problem -- runs in < 0.1 s.
    H_tr, y_tr = H[train_idx], ids[train_idx]
    H_te, y_te = H[test_idx], ids[test_idx]

    n_h = H.shape[1]
    W_clf = np.zeros((n_h, n_classes), dtype=np.float64)
    b_clf = np.zeros(n_classes, dtype=np.float64)
    lr_clf = 0.5
    for ep in range(200):
        z = H_tr @ W_clf + b_clf
        z -= z.max(axis=1, keepdims=True)
        p = np.exp(z)
        p /= p.sum(axis=1, keepdims=True)
        target = np.zeros_like(p)
        target[np.arange(len(y_tr)), y_tr] = 1.0
        grad_z = (p - target) / len(y_tr)
        gW = H_tr.T @ grad_z + reg * W_clf / max(len(y_tr), 1)
        gb = grad_z.sum(axis=0)
        W_clf -= lr_clf * gW
        b_clf -= lr_clf * gb

    z = H_te @ W_clf + b_clf
    pred = np.argmax(z, axis=1)
    acc = float((pred == y_te).mean())
    confusion = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_te, pred):
        confusion[int(t), int(p)] += 1
    return acc, confusion


def per_unit_transform_profile(model: GatedRBM,
                                X: np.ndarray,
                                Y: np.ndarray,
                                ids: np.ndarray,
                                n_classes: int) -> np.ndarray:
    """Mean hidden activation per (transformation, hidden unit).

    Returns array of shape (n_classes, n_hidden).
    """
    H = model.hidden_prob(X, Y)
    profile = np.zeros((n_classes, H.shape[1]), dtype=np.float32)
    counts = np.zeros(n_classes, dtype=np.int64)
    for c in range(n_classes):
        mask = ids == c
        counts[c] = int(mask.sum())
        if counts[c] > 0:
            profile[c] = H[mask].mean(axis=0)
    return profile


def transform_specificity_score(profile: np.ndarray) -> float:
    """How preferentially each hidden unit fires for one transform.

    For each hidden unit, take the ratio of (max - mean) to mean across
    transformations. Higher means the unit responds peakily to one
    transformation. Returns the median ratio across units.
    """
    eps = 1e-6
    means = profile.mean(axis=0) + eps          # (H,)
    maxes = profile.max(axis=0)                 # (H,)
    ratios = (maxes - means) / means
    return float(np.median(ratios))


# ----------------------------------------------------------------------
# Environment / reproducibility
# ----------------------------------------------------------------------

def _git_commit() -> str:
    try:
        sha = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=os.path.dirname(os.path.abspath(__file__)),
            stderr=subprocess.DEVNULL,
        ).decode().strip()
        return sha
    except Exception:
        return "unknown"


def env_info() -> dict[str, str]:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
        "git_commit": _git_commit(),
    }


# ----------------------------------------------------------------------
# Visualization helper kept here for completeness (the heavy plotting
# code is in visualize_transforming_pairs.py)
# ----------------------------------------------------------------------

def visualize_transformation_filters(model: GatedRBM,
                                      h_w: int = 13,
                                      n_top: int = 16) -> tuple:
    """Pick the top-`n_top` factors by joint norm and reshape their input/
    output filters into (h_w, h_w) image pairs.

    Returns (factor_indices, Wx_imgs, Wy_imgs).
    """
    Nx = np.linalg.norm(model.Wx, axis=0)
    Ny = np.linalg.norm(model.Wy, axis=0)
    score = Nx * Ny
    top = np.argsort(-score)[:n_top]
    Wx_imgs = model.Wx[:, top].T.reshape(n_top, h_w, h_w)
    Wy_imgs = model.Wy[:, top].T.reshape(n_top, h_w, h_w)
    return top, Wx_imgs, Wy_imgs


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_transforms(s: str) -> tuple[str, ...]:
    parts = tuple(p.strip() for p in s.split(",") if p.strip())
    if not parts:
        raise ValueError("empty --transforms list")
    valid = {"shift", "rotate"}
    bad = set(parts) - valid
    if bad:
        raise ValueError(f"unknown transforms: {bad}; valid: {sorted(valid)}")
    return parts


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transforms", type=str, default="shift",
                   help='comma-separated subset of {shift, rotate}')
    p.add_argument("--shift-max", type=int, default=1)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--dot-density", type=float, default=0.10)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--n-factors", type=int, default=64)
    p.add_argument("--init-scale", type=float, default=0.10)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.10)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--results-json", type=str, default=None,
                   help="Where to dump the final metrics JSON.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    transforms = _parse_transforms(args.transforms)
    rng = np.random.default_rng(args.seed)

    print(f"# Generating data: {args.n_train} train + {args.n_test} test "
          f"pairs at 13x13, dot_density={args.dot_density}")
    X_tr, Y_tr, ids_tr, pool = generate_transformed_pairs(
        args.n_train, transforms=transforms, shift_max=args.shift_max,
        dot_density=args.dot_density, rng=rng)
    X_te, Y_te, ids_te, _ = generate_transformed_pairs(
        args.n_test, transforms=transforms, shift_max=args.shift_max,
        dot_density=args.dot_density, rng=rng)
    n_classes = len(pool)
    print(f"# Transformation pool: {n_classes} types ({transforms})")
    for i, t in enumerate(pool):
        print(f"   [{i:2d}] {transform_label(t)}")

    print(f"# Building gated RBM: n_in=169, n_out=169, "
          f"n_hidden={args.n_hidden}, n_factors={args.n_factors}")
    model = build_gated_rbm(169, 169, args.n_hidden, args.n_factors,
                             init_scale=args.init_scale, seed=args.seed)

    def eval_fn(m, epoch):
        acc, _ = transform_classification_accuracy(
            m, X_te, Y_te, ids_te, n_classes, rng=np.random.default_rng(epoch))
        recon = reconstruction_metrics(m, X_te, Y_te)
        return dict(transform_acc=acc, test_recon_mse=recon["recon_mse"],
                    test_bit_acc=recon["bit_acc"])

    t0 = time.time()
    history = train(model, X_tr, Y_tr,
                    n_epochs=args.epochs, lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    batch_size=args.batch_size,
                    eval_fn=eval_fn,
                    eval_every=max(1, args.epochs // 10),
                    verbose=not args.quiet)
    train_time = time.time() - t0

    final_recon = reconstruction_metrics(model, X_te, Y_te)
    final_acc, confusion = transform_classification_accuracy(
        model, X_te, Y_te, ids_te, n_classes,
        rng=np.random.default_rng(args.seed + 1))
    profile = per_unit_transform_profile(model, X_te, Y_te, ids_te, n_classes)
    specificity = transform_specificity_score(profile)

    print("\n# Final results")
    print(f"  transform-classification accuracy : {final_acc * 100:5.2f}% "
          f"(chance = {100.0 / n_classes:5.2f}%)")
    print(f"  reconstruction MSE                 : {final_recon['recon_mse']:.4f}")
    print(f"  reconstruction bit accuracy        : {final_recon['bit_acc'] * 100:5.2f}%")
    print(f"  hidden-unit transform specificity  : {specificity:.3f}")
    print(f"  total training time                : {train_time:.1f} s")

    results = {
        "config": vars(args),
        "transforms": list(transforms),
        "transform_pool": [list(t) for t in pool],
        "n_classes": n_classes,
        "final": {
            "transform_acc": final_acc,
            "recon_mse": final_recon["recon_mse"],
            "bit_acc": final_recon["bit_acc"],
            "specificity": specificity,
        },
        "train_time_seconds": train_time,
        "env": env_info(),
        "history": history,
    }

    if args.results_json:
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        print(f"\n# Results JSON: {args.results_json}")

    return results


if __name__ == "__main__":
    main()
