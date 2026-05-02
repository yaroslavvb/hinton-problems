"""
Ellipse World — eGLOM-lite reproduction of the ambiguous-parts test from
Culp, Sabour & Hinton (2022), "Testing GLOM's ability to infer wholes from
ambiguous parts".

Setup
-----
A 2D grid (default 8x8). Some cells contain a single ellipse. Each "object" is
a class-defining configuration of 5 ellipses, perturbed by a global affine and
optional per-ellipse "ambiguity" noise on shape parameters (semi-axes,
orientation). Five placeholder classes: face, sheep, house, tree, car.

Higher ambiguity makes individual ellipses indistinguishable from each other,
so the only remaining signal is the *spatial relationship* between the five
ellipses making up an object — which is exactly the regime GLOM is built for.

Architecture (eGLOM-lite)
-------------------------
  - Per-location MLP encoder, weights shared across the 64 cells.
    Input per cell (9-d): grid x, grid y, occupancy mask, semi-axis a,
    semi-axis b, sin(2θ), cos(2θ), sub-cell dx, sub-cell dy.
  - Within-level attention: softmax(e e^T / √D) @ e, parameter-free, masked
    so empty cells never appear as keys.
  - Iterative refinement: T iterations of  e ← (1-α)·e + α·attention(e)
    (a leaky-residual update). Trained end-to-end with T=2; can be run with
    T=0 (no refinement) or T=3 (overshoot) at inference.
  - Pool: mean over occupied cells of the final embedding, then a linear head
    to 5 logits.
  - Trained with mini-batch SGD via hand-written backprop, no autograd.

This is *not* faithful eGLOM — there is one level (so "n_levels" is just a CLI
hint), no per-level transformer, and no patch-level autoencoder. The point is
to demonstrate the two GLOM ingredients that matter for the ambiguous-parts
test: (1) shared per-location MLP, (2) within-level attention with iterative
refinement that lets locations form "islands of agreement".
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Classes and canonical ellipse layouts
# ----------------------------------------------------------------------

CLASSES = ("face", "sheep", "house", "tree", "car")
N_CLASSES = len(CLASSES)
F_PER_LOC = 9   # grid_x, grid_y, mask, a, b, sin(2θ), cos(2θ), dx, dy

# Canonical layouts: each entry is a list of (cx, cy, a, b, theta) tuples in
# the canonical [-1, 1]^2 frame. Designed so a small global affine never
# pushes ellipses off the grid and so inter-ellipse spacing usually exceeds
# one cell width (=2/grid_size = 0.25 at grid=8) — collisions on the grid
# are then rare.
LAYOUTS = {
    "face": [
        (-0.30,  0.40, 0.10, 0.06, 0.0),   # left eye
        ( 0.30,  0.40, 0.10, 0.06, 0.0),   # right eye
        ( 0.00,  0.00, 0.07, 0.20, 0.0),   # nose (vertical)
        (-0.30, -0.40, 0.12, 0.05, 0.0),   # mouth left
        ( 0.30, -0.40, 0.12, 0.05, 0.0),   # mouth right
    ],
    "sheep": [
        (-0.10,  0.10, 0.40, 0.18, 0.0),   # body
        ( 0.50,  0.20, 0.15, 0.15, 0.0),   # head
        (-0.45, -0.40, 0.05, 0.22, 0.0),   # left leg
        (-0.05, -0.40, 0.05, 0.22, 0.0),   # mid leg
        ( 0.30, -0.40, 0.05, 0.22, 0.0),   # right leg
    ],
    "house": [
        ( 0.00,  0.55, 0.40, 0.18, 0.0),   # roof
        (-0.35,  0.00, 0.10, 0.25, 0.0),   # left wall
        ( 0.35,  0.00, 0.10, 0.25, 0.0),   # right wall
        ( 0.00,  0.00, 0.10, 0.20, 0.0),   # door
        ( 0.00, -0.55, 0.45, 0.05, 0.0),   # ground
    ],
    "tree": [
        ( 0.00, -0.50, 0.05, 0.30, 0.0),   # trunk (vertical)
        ( 0.00,  0.40, 0.18, 0.18, 0.0),   # canopy
        (-0.35,  0.15, 0.13, 0.13, 0.0),   # left leaves
        ( 0.35,  0.15, 0.13, 0.13, 0.0),   # right leaves
        ( 0.00,  0.65, 0.18, 0.13, 0.0),   # top leaves
    ],
    "car": [
        ( 0.00, -0.05, 0.45, 0.13, 0.0),   # body
        (-0.30, -0.40, 0.10, 0.10, 0.0),   # left wheel
        ( 0.30, -0.40, 0.10, 0.10, 0.0),   # right wheel
        (-0.18,  0.25, 0.13, 0.10, 0.0),   # left window
        ( 0.18,  0.25, 0.13, 0.10, 0.0),   # right window
    ],
}


def sample_ellipse_layout(class_name: str) -> np.ndarray:
    """Return the canonical 5-ellipse layout for a class.

    Shape (5, 5): (cx, cy, a, b, theta) per row. No randomness — the
    randomness is added by `apply_affine` and `add_ambiguity`.
    """
    return np.asarray(LAYOUTS[class_name], dtype=np.float64).copy()


def random_affine(rng: np.random.Generator,
                  rot_max: float = np.pi / 6,
                  scale_range: tuple = (0.85, 1.15),
                  trans_max: float = 0.15) -> tuple:
    """Sample (theta, scale, tx, ty) for a global affine."""
    return (rng.uniform(-rot_max, rot_max),
            rng.uniform(*scale_range),
            rng.uniform(-trans_max, trans_max),
            rng.uniform(-trans_max, trans_max))


def apply_affine(layout: np.ndarray, affine: tuple) -> np.ndarray:
    """Apply rotation, scale, translation to centers and orientations."""
    theta_g, scale, tx, ty = affine
    cos_t, sin_t = np.cos(theta_g), np.sin(theta_g)
    R = np.array([[cos_t, -sin_t], [sin_t, cos_t]])
    out = layout.copy()
    out[:, :2] = layout[:, :2] @ R.T * scale + np.array([tx, ty])
    out[:, 2] = layout[:, 2] * scale
    out[:, 3] = layout[:, 3] * scale
    out[:, 4] = layout[:, 4] + theta_g
    return out


def add_ambiguity(layout: np.ndarray, ambiguity: float,
                  rng: np.random.Generator) -> np.ndarray:
    """Noise individual ellipse shape parameters (a, b, theta).

    `ambiguity = 0.0` leaves shapes pristine — the body of a sheep is still a
    long horizontal ellipse, instantly distinguishing sheep from face.
    `ambiguity = 1.0` perturbs a, b in log-space by exp(N(0,1)) and rotates
    theta by ~N(0, π) — most ellipses end up looking like fuzzy round blobs.

    Crucially, ambiguity does NOT touch ellipse positions. The whole point of
    this test is that GLOM can recover wholes from spatial relationships even
    when each ellipse is locally uninformative.
    """
    out = layout.copy()
    n = len(layout)
    out[:, 2] = layout[:, 2] * np.exp(ambiguity * rng.standard_normal(n))
    out[:, 3] = layout[:, 3] * np.exp(ambiguity * rng.standard_normal(n))
    out[:, 4] = layout[:, 4] + ambiguity * np.pi * rng.standard_normal(n)
    return out


# ----------------------------------------------------------------------
# Grid rendering: snap ellipses to discrete cells
# ----------------------------------------------------------------------

def render_grid(layout: np.ndarray, grid_size: int = 8) -> tuple:
    """Snap each ellipse to its nearest grid cell and produce per-cell features.

    Returns:
        features: (grid_size**2, F_PER_LOC) — flattened in row-major order.
            Each cell always carries its own (grid_x, grid_y) regardless of
            occupancy, so a per-location MLP can use position even for empty
            cells (this matters: occupied/empty pattern alone is informative).
        mask: (grid_size**2,) — 1.0 if an ellipse landed in this cell.

    On collision (two ellipses snapping to the same cell), keep the first.
    With well-spaced canonical layouts and small affine, this is rare.
    """
    L = grid_size * grid_size
    features = np.zeros((L, F_PER_LOC), dtype=np.float32)
    mask = np.zeros(L, dtype=np.float32)
    cell = 2.0 / grid_size
    cell_coords = -1 + (np.arange(grid_size) + 0.5) * cell

    # Always populate positional features (cells 0 and 1) for every cell.
    # An empty-but-positioned cell still contributes information through the
    # encoder: "I am at (x, y) and contain nothing".
    for i in range(grid_size):
        for j in range(grid_size):
            idx = i * grid_size + j
            features[idx, 0] = cell_coords[j]
            features[idx, 1] = cell_coords[i]

    for cx, cy, a, b, theta in layout:
        # snap to nearest cell
        j = int(np.clip(np.round((cx - cell_coords[0]) / cell), 0, grid_size - 1))
        i = int(np.clip(np.round((cy - cell_coords[0]) / cell), 0, grid_size - 1))
        idx = i * grid_size + j
        if mask[idx] > 0:
            continue
        mask[idx] = 1.0
        features[idx, 2] = 1.0
        features[idx, 3] = a
        features[idx, 4] = b
        features[idx, 5] = np.sin(2 * theta)
        features[idx, 6] = np.cos(2 * theta)
        features[idx, 7] = cx - cell_coords[j]
        features[idx, 8] = cy - cell_coords[i]
    return features, mask


def generate_dataset(n_samples: int,
                     grid_size: int = 8,
                     ambiguity: float = 0.5,
                     rng: np.random.Generator | None = None) -> tuple:
    """Generate (X, M, Y).

    X: (n_samples, L, F_PER_LOC) feature grids.
    M: (n_samples, L) occupancy masks.
    Y: (n_samples,) class indices in [0, N_CLASSES).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    L = grid_size * grid_size
    X = np.zeros((n_samples, L, F_PER_LOC), dtype=np.float32)
    M = np.zeros((n_samples, L), dtype=np.float32)
    Y = np.zeros(n_samples, dtype=np.int64)
    for s in range(n_samples):
        cls = int(rng.integers(N_CLASSES))
        Y[s] = cls
        layout = sample_ellipse_layout(CLASSES[cls])
        layout = apply_affine(layout, random_affine(rng))
        layout = add_ambiguity(layout, ambiguity, rng)
        X[s], M[s] = render_grid(layout, grid_size)
    return X, M, Y


# ----------------------------------------------------------------------
# eGLOM-lite model
# ----------------------------------------------------------------------

def build_eglom(F: int = F_PER_LOC,
                hidden: int = 32,
                embed_dim: int = 16,
                n_classes: int = N_CLASSES,
                init_scale: float = 0.2,
                seed: int = 0) -> dict:
    """Build the parameter dict.

    The encoder is a single hidden-layer MLP shared across all grid cells:
        x_loc -> ReLU(x_loc W1 + b1) -> z_loc W2 + b2 = e_loc.
    Within-level attention is parameter-free. The classifier head W3, b3 maps
    the mean-pooled embedding to logits.
    """
    rng = np.random.default_rng(seed)
    return {
        "W1": (init_scale * rng.standard_normal((F, hidden))).astype(np.float32),
        "b1": np.zeros(hidden, dtype=np.float32),
        "W2": (init_scale * rng.standard_normal((hidden, embed_dim))).astype(np.float32),
        "b2": np.zeros(embed_dim, dtype=np.float32),
        "W3": (init_scale * rng.standard_normal((embed_dim, n_classes))).astype(np.float32),
        "b3": np.zeros(n_classes, dtype=np.float32),
    }


def softmax_axis(x: np.ndarray, axis: int) -> np.ndarray:
    x = x - x.max(axis=axis, keepdims=True)
    ex = np.exp(x)
    return ex / ex.sum(axis=axis, keepdims=True)


def forward(X: np.ndarray, M: np.ndarray, params: dict,
            n_iters: int = 1, alpha: float = 0.5,
            return_cache: bool = False):
    """Forward pass.

    n_iters=0 disables attention entirely (pool the raw encoder output).
    """
    h_pre = X @ params["W1"] + params["b1"]               # (B, L, H)
    h = np.maximum(h_pre, 0)
    e0 = h @ params["W2"] + params["b2"]                  # (B, L, D)

    D = e0.shape[-1]
    sqrt_D = np.sqrt(D)
    e_traj = [e0]
    A_traj = []
    e = e0
    # Suppress the harmless underflow / spurious "invalid" warnings that
    # arise from the -1e9 mask sentinel inside the softmax. Numerically the
    # masked positions are exactly zero in the final attention; the warnings
    # are noise.
    with np.errstate(under="ignore", invalid="ignore"):
        for _ in range(n_iters):
            sim = (e @ e.transpose(0, 2, 1)) / sqrt_D     # (B, L, L)
            sim = sim + (M[:, None, :] - 1.0) * 1e9
            A = softmax_axis(sim, axis=-1)
            Y = A @ e                                     # (B, L, D)
            e = (1.0 - alpha) * e + alpha * Y
            e_traj.append(e)
            A_traj.append(A)

    pool_count = np.maximum(M.sum(axis=1, keepdims=True), 1.0)
    z = (e * M[:, :, None]).sum(axis=1) / pool_count       # (B, D)
    logits = z @ params["W3"] + params["b3"]               # (B, C)

    if return_cache:
        return logits, {
            "h_pre": h_pre, "h": h, "e0": e0,
            "e_traj": e_traj, "A_traj": A_traj,
            "z": z, "pool_count": pool_count,
        }
    return logits


def cross_entropy_loss(logits: np.ndarray, y: np.ndarray) -> tuple:
    """Mean cross-entropy. Returns (loss, probs)."""
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    Z = exp_l.sum(axis=-1, keepdims=True)
    p = exp_l / Z
    log_p = shifted - np.log(Z)
    nll = -log_p[np.arange(len(y)), y]
    return float(nll.mean()), p


def backward(X: np.ndarray, M: np.ndarray, y: np.ndarray, params: dict,
             cache: dict, n_iters: int = 1, alpha: float = 0.5) -> dict:
    """Backprop end-to-end through the eGLOM-lite forward pass.

    Returns a dict of gradients in the same shape as `params`.
    """
    h_pre = cache["h_pre"]
    h = cache["h"]
    e0 = cache["e0"]
    e_traj = cache["e_traj"]
    A_traj = cache["A_traj"]
    z = cache["z"]
    pool_count = cache["pool_count"]

    B = X.shape[0]
    D = e0.shape[-1]
    sqrt_D = np.sqrt(D)
    C = params["W3"].shape[1]

    logits = z @ params["W3"] + params["b3"]
    shifted = logits - logits.max(axis=-1, keepdims=True)
    exp_l = np.exp(shifted)
    p = exp_l / exp_l.sum(axis=-1, keepdims=True)

    y_onehot = np.zeros((B, C), dtype=np.float32)
    y_onehot[np.arange(B), y] = 1.0

    grads = {}
    dlogits = (p - y_onehot) / B                          # (B, C)
    grads["W3"] = z.T @ dlogits
    grads["b3"] = dlogits.sum(0)
    dz = dlogits @ params["W3"].T                         # (B, D)

    # mean pool over occupied cells
    de = (M[:, :, None] / pool_count[:, :, None]) * dz[:, None, :]   # (B, L, D)

    # attention iterations, in reverse
    for t in reversed(range(n_iters)):
        e_prev = e_traj[t]
        A = A_traj[t]
        # e_t = (1-alpha) e_prev + alpha (A @ e_prev)
        de_residual = (1.0 - alpha) * de
        dY = alpha * de
        dA = dY @ e_prev.transpose(0, 2, 1)               # (B, L, L)
        de_value = A.transpose(0, 2, 1) @ dY              # (B, L, D)
        # softmax: dS = A * (dA - sum(A * dA, axis=-1, keepdims=True))
        dS = A * (dA - (dA * A).sum(axis=-1, keepdims=True))
        # zero gradient for masked-out keys (already implicit, but keeps it
        # clean against numerical fuzz).
        dS = dS * M[:, None, :]
        # S = e_prev @ e_prev.T / sqrt(D); grad symmetric in i,j
        de_sim = (dS + dS.transpose(0, 2, 1)) @ e_prev / sqrt_D
        de = de_residual + de_value + de_sim

    # encoder backward
    h_flat = h.reshape(-1, h.shape[-1])
    de_flat = de.reshape(-1, de.shape[-1])
    grads["W2"] = h_flat.T @ de_flat
    grads["b2"] = de_flat.sum(0)
    dh = de @ params["W2"].T

    dh_pre = dh * (h_pre > 0)
    X_flat = X.reshape(-1, X.shape[-1])
    dh_pre_flat = dh_pre.reshape(-1, dh_pre.shape[-1])
    grads["W1"] = X_flat.T @ dh_pre_flat
    grads["b1"] = dh_pre_flat.sum(0)
    return grads


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def accuracy(X: np.ndarray, M: np.ndarray, y: np.ndarray, params: dict,
             n_iters: int = 1, alpha: float = 0.5) -> float:
    logits = forward(X, M, params, n_iters=n_iters, alpha=alpha)
    return float((logits.argmax(-1) == y).mean())


def train(X_tr: np.ndarray, M_tr: np.ndarray, y_tr: np.ndarray,
          X_va: np.ndarray, M_va: np.ndarray, y_va: np.ndarray,
          n_epochs: int = 20, batch_size: int = 64, lr: float = 0.01,
          n_iters: int = 2, alpha: float = 0.5,
          hidden: int = 32, embed_dim: int = 16, init_scale: float = 0.2,
          seed: int = 0, verbose: bool = True,
          beta1: float = 0.9, beta2: float = 0.999, eps: float = 1e-8) -> tuple:
    """Train with Adam.

    Plain SGD is too slow on this problem because per-batch only ~5 cells
    out of 64 carry a non-bias signal, so the gradient on the encoder is
    sparse. Adam's per-parameter step normalisation handles that well.
    """
    params = build_eglom(hidden=hidden, embed_dim=embed_dim,
                         init_scale=init_scale, seed=seed)
    m_state = {k: np.zeros_like(v) for k, v in params.items()}
    v_state = {k: np.zeros_like(v) for k, v in params.items()}
    history = {
        "epoch": [], "train_loss": [], "train_acc": [],
        "val_loss": [], "val_acc": [], "val_acc_t0": [], "val_acc_t3": [],
    }
    rng = np.random.default_rng(seed + 1)
    n = len(X_tr)
    step = 0
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        losses = []
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            xb, mb, yb = X_tr[idx], M_tr[idx], y_tr[idx]
            logits, cache = forward(xb, mb, params, n_iters=n_iters,
                                    alpha=alpha, return_cache=True)
            loss, _ = cross_entropy_loss(logits, yb)
            grads = backward(xb, mb, yb, params, cache,
                             n_iters=n_iters, alpha=alpha)
            step += 1
            bc1 = 1.0 - beta1 ** step
            bc2 = 1.0 - beta2 ** step
            for k in params:
                g = grads[k]
                m_state[k] = beta1 * m_state[k] + (1 - beta1) * g
                v_state[k] = beta2 * v_state[k] + (1 - beta2) * (g * g)
                m_hat = m_state[k] / bc1
                v_hat = v_state[k] / bc2
                params[k] -= lr * m_hat / (np.sqrt(v_hat) + eps)
            losses.append(loss)

        train_loss = float(np.mean(losses))
        tr_acc = accuracy(X_tr, M_tr, y_tr, params, n_iters=n_iters, alpha=alpha)
        va_acc = accuracy(X_va, M_va, y_va, params, n_iters=n_iters, alpha=alpha)
        va_acc_t0 = accuracy(X_va, M_va, y_va, params, n_iters=0, alpha=alpha)
        va_acc_t3 = accuracy(X_va, M_va, y_va, params, n_iters=3, alpha=alpha)
        va_logits = forward(X_va, M_va, params, n_iters=n_iters, alpha=alpha)
        va_loss, _ = cross_entropy_loss(va_logits, y_va)

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(train_loss)
        history["train_acc"].append(tr_acc)
        history["val_loss"].append(va_loss)
        history["val_acc"].append(va_acc)
        history["val_acc_t0"].append(va_acc_t0)
        history["val_acc_t3"].append(va_acc_t3)

        if verbose:
            print(f"epoch {epoch+1:3d}: loss={train_loss:.3f} "
                  f"train_acc={tr_acc*100:5.1f}% "
                  f"val_acc={va_acc*100:5.1f}% "
                  f"(T=0: {va_acc_t0*100:.1f}%  T=3: {va_acc_t3*100:.1f}%)")
    return params, history


# ----------------------------------------------------------------------
# Island visualization helper
# ----------------------------------------------------------------------

def visualize_islands(features: np.ndarray, mask: np.ndarray, params: dict,
                      n_iters: int = 3, alpha: float = 0.5) -> tuple:
    """Run forward and return per-iteration cosine-similarity matrices.

    Returns:
        sims:    list of length (n_iters + 1) of (L, L) pairwise cosine sim.
                 sims[0] is the raw encoder output (before any attention).
        e_traj:  list of (L, D) embeddings.
    """
    X = features[None].astype(np.float32)
    M = mask[None].astype(np.float32)
    _, cache = forward(X, M, params, n_iters=n_iters, alpha=alpha,
                       return_cache=True)
    sims = []
    e_traj = []
    for e in cache["e_traj"]:
        e_b = e[0]
        norms = np.linalg.norm(e_b, axis=-1, keepdims=True) + 1e-8
        e_norm = e_b / norms
        sims.append(e_norm @ e_norm.T)
        e_traj.append(e_b)
    return sims, e_traj


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ambiguity", type=float, default=0.4)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--n-levels", type=int, default=1,
                   help="placeholder; this implementation has 1 GLOM level")
    p.add_argument("--n-iters", type=int, default=2,
                   help="GLOM refinement iterations during training")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--hidden", type=int, default=32)
    p.add_argument("--embed", type=int, default=16)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--init-scale", type=float, default=0.2)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    if args.n_levels != 1:
        print(f"# note: --n-levels={args.n_levels} ignored (single-level model)")
    print(f"# ellipse-world: ambiguity={args.ambiguity} "
          f"grid={args.grid_size}x{args.grid_size} n_iters={args.n_iters}")
    t0 = time.time()
    X_tr, M_tr, y_tr = generate_dataset(args.n_train, args.grid_size,
                                        args.ambiguity, rng)
    X_va, M_va, y_va = generate_dataset(args.n_val, args.grid_size,
                                        args.ambiguity, rng)
    print(f"# generated {len(X_tr)} train / {len(X_va)} val "
          f"in {time.time()-t0:.1f}s")
    print(f"# class balance (train): "
          f"{[int((y_tr == c).sum()) for c in range(N_CLASSES)]}")

    params, history = train(
        X_tr, M_tr, y_tr, X_va, M_va, y_va,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_iters=args.n_iters, alpha=args.alpha,
        hidden=args.hidden, embed_dim=args.embed,
        init_scale=args.init_scale, seed=args.seed,
    )

    print(f"\nFinal val accuracy (T={args.n_iters}): "
          f"{history['val_acc'][-1]*100:.1f}%   chance = 20%")
    print(f"  T=0 (no refinement)    : {history['val_acc_t0'][-1]*100:.1f}%")
    print(f"  T=3 (over-refined)     : {history['val_acc_t3'][-1]*100:.1f}%")
    print(f"  total time             : {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
