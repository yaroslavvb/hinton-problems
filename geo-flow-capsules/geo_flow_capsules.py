"""
Unsupervised flow capsules on the Geo dataset (synthetic 2D shapes moving
with known per-shape affines).

Reference: Sabour, Tagliasacchi, Yazdani, Hinton & Fleet,
"Unsupervised part representation by flow capsules", ICML 2021.

Architecture (numpy implementation, no torch)
=============================================

We follow the spec issue #1 v2 clarification: each shape has 6 affine
params (a, b, c, d, tx, ty); given (x, y) in frame 1, frame 2 position is
[a b; c d] @ [x, y] + [tx, ty]. Ground-truth flow at pixel (x, y) inside
shape s in frame 1 is therefore (M_s - I) @ [x, y] + t_s. Background
pixels have zero flow.

The decoder side of the Sabour et al. pipeline is a parametric mixture of
K affine motion models, one per part / capsule, each with a soft spatial
prior. We fit it via EM:

    E-step:  per-pixel responsibility r_k(p) ∝ N(flow(p);
                M_k @ p - p,  sigma_flow^2 I) * N(p_xy; mu_k, Sigma_k)

    M-step:  per-capsule
              - affine M_k = argmin sum_p r_k(p) * ||M_k @ p - (p + flow_p)||^2
                (closed form: weighted least squares on (P, P + flow))
              - spatial prior mu_k, Sigma_k = weighted moments

Ground-truth flow stands in for the encoder output. The Sabour paper
trains a CNN encoder to predict flow from raw frame pairs; we are doing
the *decomposition* half of the pipeline. See README "Deviations from the
original procedure" for why and what changes.

Evaluation: per-shape segmentation IoU between argmax-K capsule masks and
ground-truth visible shape masks at frame 1, with greedy best-match
assignment. Chance for K=3 random argmax assignment is ~1/3 ≈ 0.33 of a
shape's pixels but the IoU includes the union, so chance IoU per shape is
roughly N_shape / (3 * N_shape + 2 * N_shape) ≈ 0.20 in expectation.
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
# Data: Geo frame-pair generator
# ----------------------------------------------------------------------

def _render_ellipse_mask(h: int, w: int,
                         center: np.ndarray,
                         R: np.ndarray) -> np.ndarray:
    """Filled ellipse mask. Pixel p is inside iff R^{-1} @ (p - center) has
    norm <= 1.

    `R` is a 2x2 matrix: each column is a (signed) semi-axis vector. So a
    diagonal R = diag(a, b) with R^{-1} = diag(1/a, 1/b) gives an
    axis-aligned ellipse with semi-axes (a, b).
    """
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    dx = xx - center[0]
    dy = yy - center[1]
    Rinv = np.linalg.inv(R)
    ux = Rinv[0, 0] * dx + Rinv[0, 1] * dy
    uy = Rinv[1, 0] * dx + Rinv[1, 1] * dy
    return ux * ux + uy * uy <= 1.0


def _sample_shape_params(rng: np.random.Generator,
                         h: int, w: int,
                         n_shapes: int,
                         min_axis: float = 5.0,
                         max_axis: float = 11.0) -> list[dict]:
    """Sample non-overlapping initial poses for `n_shapes` ellipses.

    We try a few times to keep centers >= 16 px apart so EM has a chance.
    """
    margin = 8
    shapes: list[dict] = []
    intensities = np.linspace(0.45, 0.95, n_shapes)
    rng.shuffle(intensities)
    for s in range(n_shapes):
        for _ in range(50):
            cx = rng.uniform(margin, w - margin)
            cy = rng.uniform(margin, h - margin)
            ok = all(
                np.hypot(cx - sp["center"][0], cy - sp["center"][1]) >= 18
                for sp in shapes
            )
            if ok:
                break
        major = rng.uniform(min_axis + 1, max_axis)
        minor = rng.uniform(min_axis, major)
        theta = rng.uniform(0, np.pi)
        c, sn = np.cos(theta), np.sin(theta)
        # Columns of R are the semi-axis vectors.
        R = np.array([[c * major, -sn * minor],
                      [sn * major,  c * minor]], dtype=np.float32)
        shapes.append(dict(
            center=np.array([cx, cy], dtype=np.float32),
            R=R,
            intensity=float(intensities[s]),
        ))
    return shapes


def _sample_affine(rng: np.random.Generator,
                   max_translation: float = 5.0,
                   max_rotation: float = 0.20,
                   scale_jitter: float = 0.10) -> tuple[np.ndarray, np.ndarray]:
    """Sample a per-shape affine M = (L, t) for the frame1 -> frame2 motion."""
    theta = rng.uniform(-max_rotation, max_rotation)
    sx = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    sy = 1.0 + rng.uniform(-scale_jitter, scale_jitter)
    c, s = np.cos(theta), np.sin(theta)
    # Rotation followed by per-axis scale.
    L = np.array([[sx * c, -sx * s],
                  [sy * s,  sy * c]], dtype=np.float32)
    t = rng.uniform(-max_translation, max_translation, size=2).astype(np.float32)
    return L, t


def generate_geo_pair(h: int = 64,
                      w: int = 64,
                      n_shapes: int = 3,
                      rng: np.random.Generator | None = None,
                      max_translation: float = 5.0,
                      max_rotation: float = 0.20,
                      scale_jitter: float = 0.10,
                      ) -> dict:
    """Render frame1, frame2, and the ground-truth flow + visible shape masks.

    Returns a dict with:
        frame1, frame2 : (h, w) float32 in [0, 1]
        flow           : (h, w, 2) float32, dx and dy at each pixel of frame1
        masks1         : list of n_shapes (h, w) bool, *visible* shape masks
                         in frame 1 (after z-order occlusion)
        full_masks1    : list of n_shapes (h, w) bool, full shape masks before
                         occlusion (used for sanity checks)
        masks2         : list of n_shapes (h, w) bool, full shape masks at f2
        affines        : list of (L, t) tuples, the per-shape frame1->frame2
                         affines
        shapes         : list of shape param dicts (for visualization)
    """
    if rng is None:
        rng = np.random.default_rng(0)

    shapes = _sample_shape_params(rng, h, w, n_shapes)
    affines = [_sample_affine(rng,
                              max_translation=max_translation,
                              max_rotation=max_rotation,
                              scale_jitter=scale_jitter)
               for _ in range(n_shapes)]

    frame1 = np.zeros((h, w), dtype=np.float32)
    frame2 = np.zeros((h, w), dtype=np.float32)
    full_masks1: list[np.ndarray] = []
    masks2: list[np.ndarray] = []

    for s, sp in enumerate(shapes):
        L, t = affines[s]
        m1 = _render_ellipse_mask(h, w, sp["center"], sp["R"])
        # Frame2 pose = M-warped frame1 pose: c' = L c + t, R' = L R.
        c2 = L @ sp["center"] + t
        R2 = L @ sp["R"]
        m2 = _render_ellipse_mask(h, w, c2, R2)
        # Z-order: later shapes occlude earlier ones.
        frame1[m1] = sp["intensity"]
        frame2[m2] = sp["intensity"]
        full_masks1.append(m1)
        masks2.append(m2)

    # Visible (post-occlusion) masks at frame 1.
    vis_masks1: list[np.ndarray] = []
    for s in range(n_shapes):
        v = full_masks1[s].copy()
        for sp_idx in range(s + 1, n_shapes):
            v &= ~full_masks1[sp_idx]
        vis_masks1.append(v)

    # Ground-truth flow at every pixel of frame 1 (background -> 0).
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    flow = np.zeros((h, w, 2), dtype=np.float32)
    for s in range(n_shapes):
        L, t = affines[s]
        fx = (L[0, 0] - 1.0) * xx + L[0, 1] * yy + t[0]
        fy = L[1, 0] * xx + (L[1, 1] - 1.0) * yy + t[1]
        flow[..., 0] = np.where(vis_masks1[s], fx, flow[..., 0])
        flow[..., 1] = np.where(vis_masks1[s], fy, flow[..., 1])

    return dict(
        frame1=frame1, frame2=frame2,
        flow=flow,
        masks1=vis_masks1,
        full_masks1=full_masks1,
        masks2=masks2,
        affines=affines,
        shapes=shapes,
    )


# ----------------------------------------------------------------------
# Flow capsule fitter (EM)
# ----------------------------------------------------------------------

def _kmeanspp_init(features: np.ndarray, k: int,
                   rng: np.random.Generator) -> np.ndarray:
    """Standard K-means++ seeding. Returns (k, d) centers."""
    n = features.shape[0]
    idx0 = int(rng.integers(n))
    centers = [features[idx0]]
    for _ in range(k - 1):
        d2 = np.full(n, np.inf, dtype=np.float64)
        for c in centers:
            d2 = np.minimum(d2, ((features - c) ** 2).sum(axis=1))
        if d2.sum() <= 0:
            centers.append(features[int(rng.integers(n))])
            continue
        p = d2 / d2.sum()
        centers.append(features[int(rng.choice(n, p=p))])
    return np.stack(centers, axis=0)


def fit_flow_capsules(flow: np.ndarray,
                      K: int,
                      n_iters: int = 30,
                      sigma_flow: float = 0.8,
                      sigma_xy_init: float = 14.0,
                      rng: np.random.Generator | None = None,
                      foreground_threshold: float = 0.25,
                      n_restarts: int = 3,
                      ) -> dict:
    """Fit K (affine + Gaussian-spatial) flow capsules to `flow` via EM.

    Returns the best of `n_restarts` runs (lowest final reconstruction MSE).
    """
    if rng is None:
        rng = np.random.default_rng(0)
    best = None
    for r in range(n_restarts):
        sub_rng = np.random.default_rng(int(rng.integers(2**31 - 1)))
        result = _fit_flow_capsules_once(
            flow, K,
            n_iters=n_iters,
            sigma_flow=sigma_flow,
            sigma_xy_init=sigma_xy_init,
            rng=sub_rng,
            foreground_threshold=foreground_threshold,
        )
        if best is None or result["final_mse"] < best["final_mse"]:
            best = result
    return best


def _fit_flow_capsules_once(flow: np.ndarray,
                            K: int,
                            n_iters: int,
                            sigma_flow: float,
                            sigma_xy_init: float,
                            rng: np.random.Generator,
                            foreground_threshold: float,
                            ) -> dict:
    h, w, _ = flow.shape
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    XY = np.stack([xx.ravel(), yy.ravel()], axis=-1).astype(np.float32)  # (N, 2)
    P_hom = np.concatenate([XY, np.ones((XY.shape[0], 1), dtype=np.float32)],
                           axis=-1)  # (N, 3)
    F = flow.reshape(-1, 2).astype(np.float32)
    N = XY.shape[0]

    mag = np.linalg.norm(F, axis=1)
    fg = mag > foreground_threshold
    n_fg = int(fg.sum())

    # K-means++ on (x, y, flow_x, flow_y) of foreground pixels for a smart init
    if n_fg >= K * 4:
        feats = np.concatenate([XY[fg], F[fg]], axis=1)  # (n_fg, 4)
        # Scale flow features so they share order of magnitude with xy.
        feats_scaled = feats.copy()
        feats_scaled[:, 2:] *= 4.0
        init_centers = _kmeanspp_init(feats_scaled, K, rng)
        init_xy = init_centers[:, :2]
        init_flow = init_centers[:, 2:] / 4.0
    else:
        # Degenerate: little or no flow. Sprinkle priors uniformly.
        init_xy = np.column_stack([
            rng.uniform(0, w, K),
            rng.uniform(0, h, K),
        ]).astype(np.float32)
        init_flow = np.zeros((K, 2), dtype=np.float32)

    # Initialize affines with the per-cluster mean translation (identity L).
    affines: list[tuple[np.ndarray, np.ndarray]] = []
    spatial: list[tuple[np.ndarray, np.ndarray]] = []
    for k in range(K):
        affines.append((np.eye(2, dtype=np.float32),
                        init_flow[k].astype(np.float32)))
        spatial.append((init_xy[k].astype(np.float32),
                        (sigma_xy_init ** 2) * np.eye(2, dtype=np.float32)))

    # Background prior: weights pixels with low flow toward "background"
    # so they don't pull capsules off the shapes. We model background as a
    # capsule with identity affine + zero translation + uniform spatial prior.
    log_uniform_xy = -np.log(float(h * w))

    history: list[dict] = []
    resp = None
    snapshots: list[np.ndarray] = []

    for it in range(n_iters):
        # ---------------- E-step ----------------
        log_resp = np.zeros((N, K + 1), dtype=np.float32)
        for k in range(K):
            L, t = affines[k]
            mu, Sigma = spatial[k]
            # predicted flow at pixel = (L - I) @ p + t
            pred = XY @ (L - np.eye(2, dtype=np.float32)).T + t  # (N, 2)
            r2 = ((F - pred) ** 2).sum(axis=1)
            log_flow_k = -0.5 * r2 / (sigma_flow ** 2)
            diff = XY - mu
            Sinv = np.linalg.inv(Sigma)
            mahal = (diff @ Sinv * diff).sum(axis=1)
            sign, logdet = np.linalg.slogdet(Sigma)
            log_xy_k = -0.5 * mahal - 0.5 * logdet
            log_resp[:, k] = log_flow_k + log_xy_k

        # Background capsule (zero-flow, uniform xy)
        bg_r2 = (F ** 2).sum(axis=1)
        log_resp[:, K] = -0.5 * bg_r2 / (sigma_flow ** 2) + log_uniform_xy

        log_resp -= log_resp.max(axis=1, keepdims=True)
        unn = np.exp(log_resp)
        resp = unn / (unn.sum(axis=1, keepdims=True) + 1e-12)
        # We treat the background column as a soft mask on what the K capsules
        # explain. The K capsules' mass is resp[:, :K]; the remainder goes to
        # background.

        # ---------------- M-step ----------------
        new_affines = []
        new_spatial = []
        for k in range(K):
            r_k = resp[:, k]
            tot = r_k.sum() + 1e-6

            # Affine: weighted least squares for M with rows m_x, m_y where
            # M @ [x, y, 1] approximates [x + flow_x, y + flow_y].
            target = F + XY  # (N, 2): warped pixel positions
            sqrt_w = np.sqrt(r_k)[:, None]
            A_mat = sqrt_w * P_hom              # (N, 3)
            b_mat = sqrt_w * target             # (N, 2)
            sol, *_ = np.linalg.lstsq(A_mat, b_mat, rcond=None)  # (3, 2)
            a, b, tx = sol[:, 0]
            c, d, ty = sol[:, 1]
            L_new = np.array([[a, b], [c, d]], dtype=np.float32)
            t_new = np.array([tx, ty], dtype=np.float32)

            # Spatial prior: weighted moments
            mu_new = (r_k[:, None] * XY).sum(axis=0) / tot
            diff = XY - mu_new
            Sigma_new = ((r_k[:, None, None]
                          * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
                         / tot)
            Sigma_new += 1.0 * np.eye(2, dtype=np.float32)  # regularize

            new_affines.append((L_new, t_new.astype(np.float32)))
            new_spatial.append((mu_new.astype(np.float32),
                                Sigma_new.astype(np.float32)))

        affines = new_affines
        spatial = new_spatial

        # Reconstruction MSE (only over capsule mass; background pixels' mass
        # goes to the zero-flow background, which exactly explains them).
        recon = np.zeros_like(F)
        for k in range(K):
            L, t = affines[k]
            pred = XY @ (L - np.eye(2, dtype=np.float32)).T + t
            recon += resp[:, k:k + 1] * pred
        # background contributes zero-flow; its responsibility times zero is 0
        mse = float(((recon - F) ** 2).mean())
        history.append({"iter": it, "mse": mse,
                        "fg_mass_per_capsule":
                            [float(resp[:, k].sum()) for k in range(K)]})
        snapshots.append(resp.copy().reshape(h, w, K + 1))

    return dict(
        responsibilities=resp.reshape(h, w, K + 1),
        affines=affines,
        spatial_priors=spatial,
        final_mse=history[-1]["mse"],
        history=history,
        snapshots=snapshots,
        K=K,
    )


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def _greedy_match(iou_matrix: np.ndarray) -> list[tuple[int, int]]:
    """Pick disjoint (pred, gt) pairs that maximize total IoU greedily."""
    K_pred, K_gt = iou_matrix.shape
    rem_pred = set(range(K_pred))
    rem_gt = set(range(K_gt))
    pairs: list[tuple[int, int]] = []
    while rem_pred and rem_gt:
        best = (-1.0, None)
        for p in rem_pred:
            for g in rem_gt:
                if iou_matrix[p, g] > best[0]:
                    best = (float(iou_matrix[p, g]), (p, g))
        if best[1] is None:
            break
        pairs.append(best[1])
        rem_pred.discard(best[1][0])
        rem_gt.discard(best[1][1])
    return pairs


def part_segmentation_iou(responsibilities: np.ndarray,
                          gt_masks: list[np.ndarray]) -> dict:
    """Compute per-shape IoU after greedy capsule->shape matching.

    `responsibilities`: (h, w, K+1) with the last channel being background.
    `gt_masks`: list of K (h, w) bool arrays of visible shape masks.

    Returns dict with mean_iou, per_shape_iou (list, ordered by gt index),
    pairs (list of (pred_idx, gt_idx)), assignment_map (h, w) int.
    """
    h, w, Kp1 = responsibilities.shape
    K_pred = Kp1 - 1
    K_gt = len(gt_masks)
    fg_union = np.zeros((h, w), dtype=bool)
    for m in gt_masks:
        fg_union |= m

    # Restrict to true foreground; argmax over capsule channels only.
    cap_resp = responsibilities[..., :K_pred]
    bg_resp = responsibilities[..., K_pred]
    # A pixel is "claimed" by capsule k if k = argmax cap_resp AND cap_resp_k
    # exceeds bg_resp (otherwise the background owns it).
    cap_argmax = np.argmax(cap_resp, axis=-1)
    cap_max = cap_resp.max(axis=-1)
    is_capsule = cap_max > bg_resp

    pred_label = np.where(is_capsule, cap_argmax, -1)  # -1 = background

    iou_matrix = np.zeros((K_pred, K_gt), dtype=np.float64)
    for p in range(K_pred):
        pi = pred_label == p
        for g in range(K_gt):
            gj = gt_masks[g]
            inter = int((pi & gj).sum())
            union = int((pi | gj).sum())
            iou_matrix[p, g] = inter / max(union, 1)
    pairs = _greedy_match(iou_matrix)
    pred_to_gt = {p: g for p, g in pairs}

    per_shape_iou = [0.0] * K_gt
    for p, g in pairs:
        per_shape_iou[g] = float(iou_matrix[p, g])

    return dict(
        mean_iou=float(np.mean(per_shape_iou)),
        per_shape_iou=per_shape_iou,
        pairs=pairs,
        iou_matrix=iou_matrix.tolist(),
        pred_label=pred_label,
        pred_to_gt=pred_to_gt,
    )


# ----------------------------------------------------------------------
# Spec-required entry points
# ----------------------------------------------------------------------

def build_flow_capsule_net(K: int,
                           n_iters: int = 30,
                           sigma_flow: float = 0.8,
                           sigma_xy_init: float = 14.0,
                           n_restarts: int = 3) -> dict:
    """The "model" is just the EM hyperparameters. This is the parameter-free
    variant of the flow-capsule decoder; see the module docstring for why.
    """
    return dict(K=K, n_iters=n_iters, sigma_flow=sigma_flow,
                sigma_xy_init=sigma_xy_init, n_restarts=n_restarts)


def train_unsupervised(model: dict, data: list[dict],
                       n_steps: int | None = None,
                       lr: float | None = None,
                       rng: np.random.Generator | None = None,
                       verbose: bool = False) -> dict:
    """No global parameters to train; we just run the unsupervised fit on
    the training data and report mean reconstruction MSE / mean IoU.

    Kept signature-compatible with the spec stub so external callers can
    invoke it. `n_steps` and `lr` are unused.
    """
    if rng is None:
        rng = np.random.default_rng(0)
    mses: list[float] = []
    ious: list[float] = []
    for i, d in enumerate(data):
        sub_rng = np.random.default_rng(int(rng.integers(2**31 - 1)))
        fit = fit_flow_capsules(
            d["flow"],
            K=model["K"],
            n_iters=model["n_iters"],
            sigma_flow=model["sigma_flow"],
            sigma_xy_init=model["sigma_xy_init"],
            n_restarts=model["n_restarts"],
            rng=sub_rng,
        )
        ev = part_segmentation_iou(fit["responsibilities"], d["masks1"])
        mses.append(fit["final_mse"])
        ious.append(ev["mean_iou"])
        if verbose and (i + 1) % max(1, len(data) // 5) == 0:
            print(f"  train pair {i+1}/{len(data)}  "
                  f"mse={fit['final_mse']:.3f}  iou={ev['mean_iou']:.3f}")
    return dict(mean_mse=float(np.mean(mses)) if mses else float("nan"),
                mean_iou=float(np.mean(ious)) if ious else float("nan"),
                n_pairs=len(data))


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
# CLI
# ----------------------------------------------------------------------

def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-shapes", type=int, default=3)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--n-train", type=int, default=64)
    p.add_argument("--n-test", type=int, default=200)
    p.add_argument("--n-epochs", type=int, default=1,
                   help="Re-run EM over training set this many times "
                        "(no-op for the parameter-free decoder; kept for "
                        "spec compatibility).")
    p.add_argument("--K", type=int, default=None,
                   help="Number of capsules (defaults to n_shapes).")
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--sigma-flow", type=float, default=0.8)
    p.add_argument("--sigma-xy-init", type=float, default=14.0)
    p.add_argument("--max-translation", type=float, default=5.0)
    p.add_argument("--max-rotation", type=float, default=0.20)
    p.add_argument("--scale-jitter", type=float, default=0.10)
    p.add_argument("--results-json", type=str, default=None)
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    K = args.K if args.K is not None else args.n_shapes
    rng = np.random.default_rng(args.seed)

    if not args.quiet:
        print(f"# Geo flow capsules: {args.resolution}x{args.resolution} "
              f"frames, n_shapes={args.n_shapes}, K={K}")
        print(f"# Generating {args.n_train} train + {args.n_test} test "
              f"frame pairs (seed={args.seed})")

    def gen_pair():
        return generate_geo_pair(
            h=args.resolution, w=args.resolution,
            n_shapes=args.n_shapes,
            rng=rng,
            max_translation=args.max_translation,
            max_rotation=args.max_rotation,
            scale_jitter=args.scale_jitter,
        )

    train_data = [gen_pair() for _ in range(args.n_train)]
    test_data = [gen_pair() for _ in range(args.n_test)]

    model = build_flow_capsule_net(
        K=K,
        n_iters=args.n_iters,
        sigma_flow=args.sigma_flow,
        sigma_xy_init=args.sigma_xy_init,
        n_restarts=args.n_restarts,
    )

    if not args.quiet:
        print(f"# Fitting flow capsules on training set ...")
    t0 = time.time()
    train_stats = train_unsupervised(model, train_data,
                                      rng=rng, verbose=not args.quiet)
    train_time = time.time() - t0

    if not args.quiet:
        print(f"# Train: mean_mse={train_stats['mean_mse']:.3f}  "
              f"mean_iou={train_stats['mean_iou']:.3f}  "
              f"({train_time:.1f}s)")
        print(f"# Fitting flow capsules on test set ...")

    t0 = time.time()
    test_mses: list[float] = []
    test_ious: list[float] = []
    test_per_shape_iou: list[list[float]] = []
    for i, d in enumerate(test_data):
        sub_rng = np.random.default_rng(int(rng.integers(2**31 - 1)))
        fit = fit_flow_capsules(
            d["flow"], K=K,
            n_iters=args.n_iters,
            sigma_flow=args.sigma_flow,
            sigma_xy_init=args.sigma_xy_init,
            n_restarts=args.n_restarts,
            rng=sub_rng,
        )
        ev = part_segmentation_iou(fit["responsibilities"], d["masks1"])
        test_mses.append(fit["final_mse"])
        test_ious.append(ev["mean_iou"])
        test_per_shape_iou.append(ev["per_shape_iou"])
        if not args.quiet and (i + 1) % max(1, args.n_test // 10) == 0:
            print(f"  test pair {i+1}/{args.n_test}  "
                  f"mean_iou_so_far={np.mean(test_ious):.3f}")
    test_time = time.time() - t0

    per_shape_arr = np.array(test_per_shape_iou)
    mean_per_shape = per_shape_arr.mean(axis=0).tolist()

    if not args.quiet:
        print("\n# Final test results")
        print(f"  mean per-shape IoU         : {np.mean(test_ious):.3f}  "
              f"(over {args.n_test} pairs)")
        print(f"  per-shape IoU              : "
              f"{[round(v, 3) for v in mean_per_shape]}")
        print(f"  mean reconstruction MSE    : {np.mean(test_mses):.3f}")
        print(f"  test wallclock             : {test_time:.1f}s")
        print(f"  train wallclock            : {train_time:.1f}s")

    results = {
        "config": vars(args),
        "K": K,
        "train": train_stats,
        "test": {
            "mean_iou": float(np.mean(test_ious)),
            "median_iou": float(np.median(test_ious)),
            "min_iou": float(np.min(test_ious)),
            "max_iou": float(np.max(test_ious)),
            "per_shape_iou_mean": mean_per_shape,
            "mean_recon_mse": float(np.mean(test_mses)),
            "n_pairs": args.n_test,
        },
        "train_time_seconds": train_time,
        "test_time_seconds": test_time,
        "env": env_info(),
    }
    if args.results_json:
        os.makedirs(os.path.dirname(os.path.abspath(args.results_json)),
                    exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(results, f, indent=2, default=float)
        if not args.quiet:
            print(f"\n# Results JSON: {args.results_json}")
    return results


if __name__ == "__main__":
    main()
