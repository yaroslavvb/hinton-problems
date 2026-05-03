"""
Deep Lambertian Spheres -- joint recovery of albedo + surface normals from
multiple lit views of synthetic spheres.

Source:
    Tang, Salakhutdinov & Hinton (2012),
    "Deep Lambertian Networks", ICML.

Architecture (v1, simplified from the paper's GRBM):
    Per-pixel MLP encoder maps (N RGB observations, N known light directions)
    -> (albedo RGB, surface normal). A *fixed* Lambertian decoder re-renders
    each view from the inferred albedo + normal + the light direction.
    Loss = reconstruction MSE summed over views and color channels.

    The original paper uses a GRBM prior over albedo / normals; this v1 drops
    the prior and supervises via reconstruction alone. Per-pixel structure
    keeps the inverse problem identifiable -- with >=3 lights covering the
    upper hemisphere it is well-conditioned (Woodham 1980 photometric stereo
    is the closed-form analogue).

Image formation (Lambertian, no shadows):
    pixel(p, k) = albedo(p) * max(0, normal(p) . light_dir(k)) * intensity

Synthetic data:
    Centred unit sphere on a 32x32 RGB grid filling ~80% of the image.
    Per-sphere RGB albedo ~ U(0, 1)^3. Per-view light direction ~ random unit
    vector on the upper hemisphere (n_z > 0).

CLI:
    python3 deep_lambertian_spheres.py --seed 0 --n-spheres 200 \\
            --n-lights-per-sphere 4 --n-epochs 30 --resolution 32
"""

from __future__ import annotations

import argparse
import json
import os
import time
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Geometry / dataset
# ----------------------------------------------------------------------

def sphere_geometry(resolution: int = 32, fill: float = 0.80):
    """Return (mask, normals) for a centred sphere on an HxW grid.

    `fill` is the fraction of the smaller image dimension occupied by the
    sphere's diameter (the paper uses spheres that "fill most of the image").

    `normals` has shape (H, W, 3) and contains the outward unit normal at
    every pixel inside the sphere mask, with `n_z = sqrt(1 - n_x^2 - n_y^2)`
    (we only see the front hemisphere). Pixels outside the mask have
    `normals = 0`.
    """
    H = W = resolution
    # Pixel centres in normalised coords [-1, 1]
    ys, xs = np.meshgrid(
        np.linspace(-1.0, 1.0, H, dtype=np.float32),
        np.linspace(-1.0, 1.0, W, dtype=np.float32),
        indexing="ij",
    )
    radius = float(fill)  # in the normalised [-1, 1] coords
    # Sphere of radius `radius` centred at origin; project pixel coords.
    nx = xs / radius
    ny = ys / radius
    rho2 = nx * nx + ny * ny
    mask = rho2 <= 1.0
    nz = np.zeros_like(nx)
    nz[mask] = np.sqrt(np.clip(1.0 - rho2[mask], 0.0, 1.0))
    normals = np.stack([nx, ny, nz], axis=-1).astype(np.float32)
    normals[~mask] = 0.0
    return mask, normals


def random_upper_hemisphere(rng: np.random.Generator, n: int) -> np.ndarray:
    """Random unit vectors with n_z > 0. Shape (n, 3)."""
    out = np.empty((n, 3), dtype=np.float32)
    filled = 0
    while filled < n:
        v = rng.standard_normal((n, 3)).astype(np.float32)
        norm = np.linalg.norm(v, axis=1, keepdims=True)
        norm[norm == 0] = 1.0
        v = v / norm
        v[:, 2] = np.abs(v[:, 2])  # flip to upper hemisphere
        # Filter out near-grazing lights (n_z very small) -- they cause
        # numerical headaches and are pathological for photometric stereo.
        keep = v[:, 2] > 0.15
        take = min(n - filled, int(keep.sum()))
        if take > 0:
            out[filled:filled + take] = v[keep][:take]
            filled += take
    return out


def render_sphere(
    albedo: np.ndarray,
    normals: np.ndarray,
    mask: np.ndarray,
    light_dir: np.ndarray,
    light_intensity: float = 1.0,
) -> np.ndarray:
    """Render a Lambertian sphere image.

    Args:
        albedo: per-sphere RGB albedo, shape (3,) in [0, 1].
        normals: per-pixel surface normals, shape (H, W, 3).
        mask: sphere mask, shape (H, W).
        light_dir: light direction (unit vector), shape (3,).
        light_intensity: scalar light intensity.

    Returns:
        image: shape (H, W, 3) in [0, 1] (assuming albedo in [0,1]).
    """
    # cos_theta in [-1, 1]; clip to [0, 1] (no shadows but no negative light)
    cos_theta = np.einsum("hwc,c->hw", normals, light_dir)
    cos_theta = np.clip(cos_theta, 0.0, None)
    cos_theta = cos_theta * mask.astype(cos_theta.dtype)
    image = albedo[None, None, :] * cos_theta[:, :, None] * light_intensity
    return image.astype(np.float32)


@dataclass
class SphereDataset:
    """Holds a pre-rendered batch of synthetic spheres.

    All tensors are flattened over the sphere-mask pixel set so the encoder
    can process each pixel independently (per-pixel MLP).
    """
    # Per-sphere
    albedos: np.ndarray         # (S, 3)              ground-truth albedo
    light_dirs: np.ndarray      # (S, K, 3)           K lights per sphere
    # Per-pixel (flattened over mask, repeated per sphere)
    pixel_normals: np.ndarray   # (S, P, 3)           ground-truth normals
    pixel_obs: np.ndarray       # (S, P, K, 3)        observed RGB per light
    # Geometry
    mask: np.ndarray            # (H, W) bool
    normals_full: np.ndarray    # (H, W, 3)
    pixel_indices: np.ndarray   # (P, 2) -> (i, j)
    # Metadata
    resolution: int
    n_lights: int

    @property
    def n_spheres(self) -> int:
        return self.albedos.shape[0]

    @property
    def n_pixels(self) -> int:
        return self.pixel_normals.shape[1]


def generate_dataset(
    n_spheres: int,
    n_lights_per_sphere: int = 4,
    resolution: int = 32,
    fill: float = 0.80,
    light_intensity: float = 1.0,
    seed: int = 0,
) -> SphereDataset:
    """Render `n_spheres` synthetic Lambertian spheres, each under
    `n_lights_per_sphere` random upper-hemisphere lights.

    Returns a `SphereDataset` with ground-truth albedo + normals for eval.
    """
    rng = np.random.default_rng(seed)
    mask, normals_full = sphere_geometry(resolution=resolution, fill=fill)
    pix_ij = np.argwhere(mask)  # (P, 2)
    P = pix_ij.shape[0]

    pixel_normals = np.broadcast_to(
        normals_full[pix_ij[:, 0], pix_ij[:, 1]],  # (P, 3)
        (n_spheres, P, 3),
    ).astype(np.float32).copy()

    albedos = rng.uniform(0.0, 1.0, size=(n_spheres, 3)).astype(np.float32)
    light_dirs = np.empty((n_spheres, n_lights_per_sphere, 3), dtype=np.float32)
    for s in range(n_spheres):
        light_dirs[s] = random_upper_hemisphere(rng, n_lights_per_sphere)

    # cos_theta(s, p, k) = dot(normal(p), light_dir(s, k))
    cos = np.einsum("spc,skc->spk", pixel_normals, light_dirs)
    cos = np.clip(cos, 0.0, None)  # Lambertian, no negative light
    # observed = albedo(s) * cos(s, p, k) * intensity, broadcast to RGB
    obs = (
        albedos[:, None, None, :]
        * cos[:, :, :, None]
        * light_intensity
    ).astype(np.float32)

    return SphereDataset(
        albedos=albedos,
        light_dirs=light_dirs,
        pixel_normals=pixel_normals,
        pixel_obs=obs,
        mask=mask,
        normals_full=normals_full,
        pixel_indices=pix_ij,
        resolution=resolution,
        n_lights=n_lights_per_sphere,
    )


# ----------------------------------------------------------------------
# Network
# ----------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0.0, x)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


# tanh range is shrunk slightly so n_z = sqrt(1 - n_x^2 - n_y^2) stays
# bounded away from zero (no division-by-zero in backprop).
_NXY_SCALE = 0.985


@dataclass
class DeepLambertianNet:
    """Per-pixel MLP encoder + fixed Lambertian decoder.

    Encoder input  per pixel: concat(N RGB observations, N light directions)
                              dimension = 6 * N
    Encoder output per pixel: 5 = 3 (albedo logits) + 2 (normal_xy logits)
    """
    n_lights: int
    hidden: int
    W1: np.ndarray
    b1: np.ndarray
    W2: np.ndarray
    b2: np.ndarray

    @classmethod
    def init(
        cls,
        n_lights: int = 4,
        hidden: int = 128,
        seed: int = 0,
    ) -> "DeepLambertianNet":
        rng = np.random.default_rng(seed)
        n_in = 6 * n_lights  # N RGB + N light dirs (3 each)
        # He init for ReLU
        W1 = rng.standard_normal((n_in, hidden)).astype(np.float32) * np.sqrt(2.0 / n_in)
        b1 = np.zeros(hidden, dtype=np.float32)
        # Small final-layer scale so albedo starts near 0.5 and normals near (0, 0, 1)
        W2 = rng.standard_normal((hidden, 5)).astype(np.float32) * 0.01
        b2 = np.zeros(5, dtype=np.float32)
        return cls(n_lights=n_lights, hidden=hidden,
                   W1=W1, b1=b1, W2=W2, b2=b2)


def build_deep_lambertian_net(
    n_lights: int = 4, hidden: int = 128, seed: int = 0,
) -> DeepLambertianNet:
    """Public alias matching the spec's required name."""
    return DeepLambertianNet.init(n_lights=n_lights, hidden=hidden, seed=seed)


def encode(net: DeepLambertianNet, x: np.ndarray):
    """Run encoder on flat per-pixel features `x` of shape (B, 6N).

    Returns (albedo, normal, cache) where cache holds the intermediates
    needed for backprop.
    """
    z1 = x @ net.W1 + net.b1                # (B, H)
    a1 = relu(z1)                            # (B, H)
    z2 = a1 @ net.W2 + net.b2                # (B, 5)
    albedo = sigmoid(z2[:, :3])              # (B, 3) in [0, 1]
    nxy_pre = np.tanh(z2[:, 3:5])            # (B, 2) in [-1, 1]
    nxy = _NXY_SCALE * nxy_pre               # (B, 2) in [-0.985, 0.985]
    nz_sq = np.clip(1.0 - nxy[:, 0] ** 2 - nxy[:, 1] ** 2, 1e-6, 1.0)
    nz = np.sqrt(nz_sq)                      # (B,)
    normal = np.stack([nxy[:, 0], nxy[:, 1], nz], axis=1)  # (B, 3)
    cache = {
        "x": x, "z1": z1, "a1": a1, "z2": z2,
        "albedo": albedo, "nxy_pre": nxy_pre, "nxy": nxy,
        "nz": nz, "normal": normal,
    }
    return albedo, normal, cache


def decode(albedo: np.ndarray, normal: np.ndarray, light_dirs: np.ndarray):
    """Lambertian render.

    Args:
        albedo:    (B, 3)
        normal:    (B, 3)
        light_dirs:(B, K, 3)  one light set per pixel-row (broadcast OK)
    Returns:
        yhat:    (B, K, 3)
        d_clip:  (B, K)         max(0, n . l)
        active:  (B, K) bool    indicator for ReLU mask
    """
    dot = np.einsum("bc,bkc->bk", normal, light_dirs)  # (B, K)
    active = dot > 0.0
    d_clip = np.where(active, dot, 0.0)                # (B, K)
    yhat = albedo[:, None, :] * d_clip[:, :, None]     # (B, K, 3)
    return yhat, d_clip, active


def forward_loss(
    net: DeepLambertianNet,
    x: np.ndarray,
    light_dirs: np.ndarray,
    y: np.ndarray,
):
    """One forward pass, returning (loss, cache_for_backward).

    x:          (B, 6N)
    light_dirs: (B, K, 3)
    y:          (B, K, 3)         observed pixel RGB per light
    """
    albedo, normal, enc_cache = encode(net, x)
    yhat, d_clip, active = decode(albedo, normal, light_dirs)
    diff = yhat - y                              # (B, K, 3)
    loss = float(np.mean(diff ** 2))
    cache = {
        **enc_cache,
        "light_dirs": light_dirs,
        "yhat": yhat, "y": y,
        "d_clip": d_clip, "active": active,
        "diff": diff,
    }
    return loss, cache


def backward(net: DeepLambertianNet, cache: dict):
    """Backprop. Returns gradients dict keyed like net's parameters."""
    x = cache["x"]
    z1 = cache["z1"]; a1 = cache["a1"]; z2 = cache["z2"]
    albedo = cache["albedo"]
    nxy_pre = cache["nxy_pre"]; nxy = cache["nxy"]
    nz = cache["nz"]
    light_dirs = cache["light_dirs"]
    diff = cache["diff"]            # (B, K, 3)
    d_clip = cache["d_clip"]        # (B, K)
    active = cache["active"]        # (B, K)

    B, K, _ = diff.shape
    n_terms = B * K * 3  # number of squared error terms (matches np.mean)

    # dL / d yhat   = (2/n_terms) * diff
    e = (2.0 / n_terms) * diff                                      # (B, K, 3)

    # yhat = albedo[:, None, :] * d_clip[:, :, None]
    # d L/d albedo[b, c] = sum_k e[b, k, c] * d_clip[b, k]
    d_albedo = np.einsum("bkc,bk->bc", e, d_clip)                   # (B, 3)
    # d L/d d_clip[b, k] = sum_c e[b, k, c] * albedo[b, c]
    d_d_clip = np.einsum("bkc,bc->bk", e, albedo)                    # (B, K)

    # d_clip = max(0, dot)  ->  d L/d dot = d L/d d_clip * I[dot>0]
    d_dot = d_d_clip * active.astype(d_d_clip.dtype)                 # (B, K)

    # dot = einsum('bc,bkc->bk', normal, light_dirs)
    # d L/d normal[b, :] = sum_k d_dot[b, k] * light_dirs[b, k, :]
    d_normal = np.einsum("bk,bkc->bc", d_dot, light_dirs)            # (B, 3)

    # normal = (nxy[:, 0], nxy[:, 1], nz),  nz = sqrt(1 - nxy0^2 - nxy1^2)
    # d L/d nxy[b, 0] = d_normal[b, 0] + d_normal[b, 2] * (-nxy[b,0]/nz[b])
    # d L/d nxy[b, 1] = d_normal[b, 1] + d_normal[b, 2] * (-nxy[b,1]/nz[b])
    d_nxy = np.empty_like(nxy)
    d_nxy[:, 0] = d_normal[:, 0] - d_normal[:, 2] * (nxy[:, 0] / nz)
    d_nxy[:, 1] = d_normal[:, 1] - d_normal[:, 2] * (nxy[:, 1] / nz)
    # nxy = _NXY_SCALE * tanh(z2[:, 3:5])
    d_z2_nxy = d_nxy * _NXY_SCALE * (1.0 - nxy_pre ** 2)             # (B, 2)

    # albedo = sigmoid(z2[:, :3])
    d_z2_alb = d_albedo * albedo * (1.0 - albedo)                    # (B, 3)

    d_z2 = np.concatenate([d_z2_alb, d_z2_nxy], axis=1)              # (B, 5)

    d_W2 = a1.T @ d_z2                                                # (H, 5)
    d_b2 = d_z2.sum(axis=0)                                           # (5,)
    d_a1 = d_z2 @ net.W2.T                                            # (B, H)
    d_z1 = d_a1 * (z1 > 0).astype(d_a1.dtype)                         # ReLU
    d_W1 = x.T @ d_z1                                                 # (6N, H)
    d_b1 = d_z1.sum(axis=0)                                           # (H,)

    return {"W1": d_W1, "b1": d_b1, "W2": d_W2, "b2": d_b2}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def assemble_features(dataset: SphereDataset):
    """Flatten the dataset to per-pixel training tensors.

    Returns:
        x:           (S * P, 6 * K)   features (RGB obs concat with light dirs)
        y:           (S * P, K, 3)    observed pixel RGB per light
        light_dirs:  (S * P, K, 3)    light dirs (broadcast from per-sphere)
        gt_albedo:   (S * P, 3)       broadcast of per-sphere albedo
        gt_normal:   (S * P, 3)       per-pixel ground-truth normal
        sphere_idx:  (S * P,)         which sphere each row came from
    """
    S = dataset.n_spheres
    P = dataset.n_pixels
    K = dataset.n_lights
    obs = dataset.pixel_obs        # (S, P, K, 3)
    light_dirs = np.broadcast_to(
        dataset.light_dirs[:, None, :, :], (S, P, K, 3)
    ).copy()                       # (S, P, K, 3)
    rgb_flat = obs.reshape(S, P, K * 3)
    ldir_flat = light_dirs.reshape(S, P, K * 3)
    x = np.concatenate([rgb_flat, ldir_flat], axis=-1)  # (S, P, 6K)
    x = x.reshape(S * P, 6 * K).astype(np.float32)
    y = obs.reshape(S * P, K, 3).astype(np.float32)
    light_dirs_flat = light_dirs.reshape(S * P, K, 3).astype(np.float32)
    gt_albedo = np.broadcast_to(
        dataset.albedos[:, None, :], (S, P, 3)
    ).reshape(S * P, 3).astype(np.float32)
    gt_normal = dataset.pixel_normals.reshape(S * P, 3).astype(np.float32)
    sphere_idx = np.repeat(np.arange(S, dtype=np.int32), P)
    return x, y, light_dirs_flat, gt_albedo, gt_normal, sphere_idx


def angular_error_deg(pred: np.ndarray, gt: np.ndarray) -> np.ndarray:
    """Per-row angular error in degrees between two unit vectors."""
    pred = pred / np.maximum(np.linalg.norm(pred, axis=-1, keepdims=True), 1e-8)
    gt = gt / np.maximum(np.linalg.norm(gt, axis=-1, keepdims=True), 1e-8)
    cos = np.clip(np.sum(pred * gt, axis=-1), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def evaluate(
    net: DeepLambertianNet,
    dataset: SphereDataset,
    batch: int = 8192,
):
    """Run the encoder on `dataset` and return per-pixel metrics."""
    x, y, light_dirs, gt_albedo, gt_normal, sphere_idx = assemble_features(dataset)
    n = x.shape[0]
    pred_albedo = np.empty_like(gt_albedo)
    pred_normal = np.empty_like(gt_normal)
    for i in range(0, n, batch):
        sl = slice(i, i + batch)
        a, nrm, _ = encode(net, x[sl])
        pred_albedo[sl] = a
        pred_normal[sl] = nrm

    # Reconstruction error
    yhat, _, _ = decode(pred_albedo, pred_normal, light_dirs)
    recon_mse = float(np.mean((yhat - y) ** 2))

    # Albedo: per-sphere predictions should agree across pixels. Take per-sphere
    # pixel-mean of predicted albedo, compare to the per-sphere ground truth.
    S = dataset.n_spheres
    P = dataset.n_pixels
    pred_albedo_per_sphere = pred_albedo.reshape(S, P, 3).mean(axis=1)  # (S, 3)
    albedo_mse = float(np.mean((pred_albedo_per_sphere - dataset.albedos) ** 2))

    # Normal angular error (per pixel)
    ang = angular_error_deg(pred_normal, gt_normal)
    return {
        "recon_mse": recon_mse,
        "albedo_mse": albedo_mse,
        "albedo_pred_per_sphere": pred_albedo_per_sphere,
        "normal_angular_error_deg_mean": float(ang.mean()),
        "normal_angular_error_deg_median": float(np.median(ang)),
        "pred_albedo_pixels": pred_albedo,
        "pred_normal_pixels": pred_normal,
    }


def train(
    net: DeepLambertianNet,
    dataset: SphereDataset,
    n_epochs: int = 30,
    batch_size: int = 1024,
    lr: float = 5e-3,
    momentum: float = 0.9,
    weight_decay: float = 0.0,
    lr_decay_start: float = 0.5,
    lr_decay_end: float = 0.05,
    eval_dataset: SphereDataset | None = None,
    snapshot_callback=None,
    snapshot_every: int = 1,
    seed: int = 0,
    verbose: bool = True,
):
    """SGD with momentum on per-pixel reconstruction loss.

    `lr_decay_start` / `lr_decay_end` give the fraction of `lr` used at the
    start and end of training; LR is linearly interpolated between these
    after the first half of training.
    """
    rng = np.random.default_rng(seed)
    x, y, light_dirs, gt_albedo, gt_normal, sphere_idx = assemble_features(dataset)
    n = x.shape[0]
    velocity = {k: np.zeros_like(v) for k, v in {
        "W1": net.W1, "b1": net.b1, "W2": net.W2, "b2": net.b2,
    }.items()}

    history = {
        "epoch": [], "loss": [],
        "recon_mse": [], "albedo_mse": [],
        "normal_angular_error_deg_mean": [],
        "normal_angular_error_deg_median": [],
        "wallclock_s": [],
    }
    t_start = time.time()

    for epoch in range(n_epochs):
        t_epoch = time.time()
        # Cosine LR schedule between start_scale (epoch 0) and end_scale (final)
        if n_epochs > 1:
            phase = epoch / (n_epochs - 1)
        else:
            phase = 0.0
        cos = 0.5 * (1.0 + np.cos(np.pi * phase))
        lr_scale = lr_decay_end + (lr_decay_start - lr_decay_end) * cos
        cur_lr = lr * lr_scale
        perm = rng.permutation(n)
        losses = []
        for i in range(0, n, batch_size):
            idx = perm[i:i + batch_size]
            xb = x[idx]; yb = y[idx]; lb = light_dirs[idx]
            loss, cache = forward_loss(net, xb, lb, yb)
            grads = backward(net, cache)
            for k in ("W1", "b1", "W2", "b2"):
                g = grads[k]
                if weight_decay:
                    g = g + weight_decay * getattr(net, k)
                velocity[k] = momentum * velocity[k] - cur_lr * g
                setattr(net, k, getattr(net, k) + velocity[k])
            losses.append(loss)
        epoch_loss = float(np.mean(losses))

        eval_set = eval_dataset if eval_dataset is not None else dataset
        metrics = evaluate(net, eval_set)
        history["epoch"].append(epoch + 1)
        history["loss"].append(epoch_loss)
        history["recon_mse"].append(metrics["recon_mse"])
        history["albedo_mse"].append(metrics["albedo_mse"])
        history["normal_angular_error_deg_mean"].append(
            metrics["normal_angular_error_deg_mean"])
        history["normal_angular_error_deg_median"].append(
            metrics["normal_angular_error_deg_median"])
        history["wallclock_s"].append(time.time() - t_start)

        if verbose:
            print(
                f"epoch {epoch + 1:3d}  lr={cur_lr:.4f}  "
                f"loss={epoch_loss:.5f}  "
                f"recon={metrics['recon_mse']:.5f}  "
                f"albedo_mse={metrics['albedo_mse']:.4f}  "
                f"normal_ang(deg) mean={metrics['normal_angular_error_deg_mean']:.2f} "
                f"median={metrics['normal_angular_error_deg_median']:.2f}  "
                f"({time.time() - t_epoch:.2f}s)"
            )

        if snapshot_callback is not None and (
            epoch % snapshot_every == 0 or epoch == n_epochs - 1
        ):
            snapshot_callback(epoch, net, history, eval_set, metrics)

    return net, history


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-spheres", type=int, default=400,
                   help="number of training spheres")
    p.add_argument("--n-test-spheres", type=int, default=64,
                   help="held-out spheres for eval")
    p.add_argument("--n-lights-per-sphere", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=120)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--lr-decay-start", type=float, default=1.0,
                   help="LR multiplier at epoch 0 (cosine schedule)")
    p.add_argument("--lr-decay-end", type=float, default=0.05,
                   help="LR multiplier at the final epoch")
    p.add_argument("--results-json", type=str, default=None,
                   help="optional path to dump final metrics as JSON")
    return p.parse_args()


def main():
    args = _parse_args()
    print(
        f"# Deep Lambertian Spheres -- seed={args.seed} "
        f"n_spheres={args.n_spheres} n_lights={args.n_lights_per_sphere} "
        f"resolution={args.resolution} hidden={args.hidden} "
        f"epochs={args.n_epochs}"
    )
    train_set = generate_dataset(
        n_spheres=args.n_spheres,
        n_lights_per_sphere=args.n_lights_per_sphere,
        resolution=args.resolution,
        seed=args.seed,
    )
    test_set = generate_dataset(
        n_spheres=args.n_test_spheres,
        n_lights_per_sphere=args.n_lights_per_sphere,
        resolution=args.resolution,
        seed=args.seed + 10_000,
    )
    net = build_deep_lambertian_net(
        n_lights=args.n_lights_per_sphere,
        hidden=args.hidden,
        seed=args.seed,
    )
    t0 = time.time()
    net, history = train(
        net, train_set,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        weight_decay=args.weight_decay,
        lr_decay_start=args.lr_decay_start,
        lr_decay_end=args.lr_decay_end,
        eval_dataset=test_set,
        seed=args.seed,
    )
    train_wallclock = time.time() - t0
    final = evaluate(net, test_set)
    print(
        f"\nFinal (held-out): "
        f"recon_mse={final['recon_mse']:.5f}  "
        f"albedo_mse={final['albedo_mse']:.4f}  "
        f"normal_angular_deg(mean)={final['normal_angular_error_deg_mean']:.2f}  "
        f"normal_angular_deg(median)={final['normal_angular_error_deg_median']:.2f}  "
        f"train_wallclock={train_wallclock:.1f}s"
    )
    if args.results_json:
        out = {
            "args": vars(args),
            "train_wallclock_s": train_wallclock,
            "final_test": {
                "recon_mse": final["recon_mse"],
                "albedo_mse": final["albedo_mse"],
                "normal_angular_error_deg_mean": final["normal_angular_error_deg_mean"],
                "normal_angular_error_deg_median": final["normal_angular_error_deg_median"],
            },
            "history": history,
        }
        os.makedirs(os.path.dirname(args.results_json) or ".", exist_ok=True)
        with open(args.results_json, "w") as f:
            json.dump(out, f, indent=2)
        print(f"Wrote {args.results_json}")


if __name__ == "__main__":
    main()
