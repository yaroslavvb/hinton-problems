"""
Static visualisations for the deep-lambertian-spheres experiment.

Generates four PNG panels in `viz/`:

  - `dataset_examples.png`     a grid of random spheres under their N lights
                               (rows = different spheres, columns = lights).
  - `albedo_recovery.png`      ground-truth albedo vs recovered albedo
                               (held-out test spheres).
  - `normal_recovery.png`      ground-truth normal map vs recovered normal map
                               vs angular-error heatmap.
  - `per_light_reconstruction.png`
                               for one held-out sphere, the input view, the
                               net's re-rendered view, and the residual,
                               for each of the N lights.
  - `training_curves.png`      loss / recon-MSE / albedo-MSE / normal-angle
                               curves.

Run after training:
    python3 visualize_deep_lambertian_spheres.py --seed 0 --outdir viz
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from deep_lambertian_spheres import (
    build_deep_lambertian_net,
    decode,
    encode,
    evaluate,
    generate_dataset,
    render_sphere,
    train,
)


def _normal_to_rgb(normals: np.ndarray, mask: np.ndarray | None = None) -> np.ndarray:
    """Map a 3-channel normal map (in [-1, 1]) to a viewable RGB image.

    The classic visualisation: r = (n_x + 1) / 2, g = (n_y + 1) / 2, b = n_z.
    Pixels outside `mask` are zeroed.
    """
    rgb = np.zeros((*normals.shape[:2], 3), dtype=np.float32)
    rgb[..., 0] = (normals[..., 0] + 1.0) * 0.5
    rgb[..., 1] = (normals[..., 1] + 1.0) * 0.5
    rgb[..., 2] = np.clip(normals[..., 2], 0.0, 1.0)
    if mask is not None:
        rgb[~mask] = 0.0
    return np.clip(rgb, 0.0, 1.0)


def _pixels_to_image(values: np.ndarray, pixel_indices: np.ndarray,
                     resolution: int, channels: int) -> np.ndarray:
    """Scatter per-pixel predictions back onto a (H, W, C) image."""
    img = np.zeros((resolution, resolution, channels), dtype=np.float32)
    img[pixel_indices[:, 0], pixel_indices[:, 1]] = values
    return img


def make_dataset_examples_figure(
    test_set, outpath: str, n_rows: int = 4, n_cols: int | None = None
):
    """Show a grid of synthetic spheres lit from each of their N directions."""
    K = test_set.n_lights
    if n_cols is None:
        n_cols = K
    n_rows = min(n_rows, test_set.n_spheres)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(1.6 * n_cols, 1.7 * n_rows))
    axes = np.atleast_2d(axes)
    for r in range(n_rows):
        for c in range(min(K, n_cols)):
            img = render_sphere(
                test_set.albedos[r],
                test_set.normals_full,
                test_set.mask,
                test_set.light_dirs[r, c],
            )
            axes[r, c].imshow(np.clip(img, 0, 1))
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            if r == 0:
                axes[r, c].set_title(f"light {c}", fontsize=8)
            if c == 0:
                axes[r, c].set_ylabel(f"sphere {r}", fontsize=8)
    fig.suptitle(
        f"Synthetic Lambertian spheres "
        f"({test_set.resolution}x{test_set.resolution}, "
        f"{K} lights/sphere)",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_albedo_recovery_figure(test_set, metrics, outpath: str, n: int = 8):
    """Side-by-side ground-truth vs recovered per-sphere albedo (RGB swatches)."""
    n = min(n, test_set.n_spheres)
    gt = test_set.albedos[:n]
    pred = metrics["albedo_pred_per_sphere"][:n]
    swatch_size = 32
    gt_strip = np.repeat(np.repeat(gt[:, None, None, :], swatch_size, axis=1),
                         swatch_size, axis=2)  # (n, S, S, 3)
    pred_strip = np.repeat(np.repeat(pred[:, None, None, :], swatch_size, axis=1),
                           swatch_size, axis=2)
    fig, axes = plt.subplots(2, n, figsize=(0.9 * n + 1.2, 2.2))
    if n == 1:
        axes = np.array([[axes[0]], [axes[1]]])
    for i in range(n):
        axes[0, i].imshow(np.clip(gt_strip[i], 0, 1))
        axes[1, i].imshow(np.clip(pred_strip[i], 0, 1))
        axes[0, i].set_xticks([]); axes[0, i].set_yticks([])
        axes[1, i].set_xticks([]); axes[1, i].set_yticks([])
        if i == 0:
            axes[0, i].set_ylabel("GT", fontsize=9)
            axes[1, i].set_ylabel("recovered", fontsize=9)
        err = np.linalg.norm(pred[i] - gt[i])
        axes[1, i].set_xlabel(f"|err|={err:.2f}", fontsize=7)
    fig.suptitle(
        f"Albedo recovery (mean MSE = {metrics['albedo_mse']:.4f})",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_normal_recovery_figure(test_set, metrics, outpath: str, n: int = 4):
    """For n test spheres: GT normals (RGB-encoded), predicted normals,
    angular error heatmap."""
    n = min(n, test_set.n_spheres)
    P = test_set.n_pixels
    pred_normal_pix = metrics["pred_normal_pixels"].reshape(test_set.n_spheres, P, 3)
    fig, axes = plt.subplots(n, 3, figsize=(6.5, 2.0 * n))
    if n == 1:
        axes = axes[None, :]
    for r in range(n):
        gt_img = _normal_to_rgb(test_set.normals_full, test_set.mask)
        pred_img = _pixels_to_image(
            pred_normal_pix[r], test_set.pixel_indices, test_set.resolution, 3
        )
        pred_rgb = _normal_to_rgb(pred_img, test_set.mask)
        # angular error heatmap
        gt_pix = test_set.pixel_normals[r]                           # (P, 3)
        cos = np.clip(np.sum(pred_normal_pix[r] * gt_pix, axis=-1), -1, 1)
        ang = np.degrees(np.arccos(cos))                             # (P,)
        ang_img = np.zeros((test_set.resolution, test_set.resolution),
                           dtype=np.float32)
        ang_img[test_set.pixel_indices[:, 0], test_set.pixel_indices[:, 1]] = ang

        axes[r, 0].imshow(gt_img)
        axes[r, 1].imshow(pred_rgb)
        im = axes[r, 2].imshow(ang_img, cmap="magma", vmin=0, vmax=60)
        for c in range(3):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
        if r == 0:
            axes[r, 0].set_title("GT normals", fontsize=9)
            axes[r, 1].set_title("recovered", fontsize=9)
            axes[r, 2].set_title("angular err (deg)", fontsize=9)
        axes[r, 0].set_ylabel(f"sphere {r}", fontsize=8)
        axes[r, 2].set_xlabel(f"mean={ang.mean():.1f}", fontsize=7)
    cbar = fig.colorbar(im, ax=axes[:, 2], shrink=0.7, label="degrees")
    fig.suptitle(
        f"Surface-normal recovery "
        f"(test mean ang err = "
        f"{metrics['normal_angular_error_deg_mean']:.2f}deg, "
        f"median = "
        f"{metrics['normal_angular_error_deg_median']:.2f}deg)",
        fontsize=10,
    )
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_per_light_reconstruction_figure(net, test_set, outpath: str,
                                         sphere_idx: int = 0):
    """For one held-out sphere, show input view / re-rendered view / residual
    for each of the N lights."""
    K = test_set.n_lights
    # Run encoder on this sphere's pixels
    x_obs = test_set.pixel_obs[sphere_idx]                # (P, K, 3)
    K_, _ = test_set.light_dirs[sphere_idx].shape
    P = test_set.n_pixels
    rgb_flat = x_obs.reshape(P, K * 3)
    ldir_flat = np.broadcast_to(
        test_set.light_dirs[sphere_idx][None, :, :], (P, K, 3)
    ).reshape(P, K * 3)
    x = np.concatenate([rgb_flat, ldir_flat], axis=-1).astype(np.float32)
    albedo, normal, _ = encode(net, x)
    light_dirs = np.broadcast_to(
        test_set.light_dirs[sphere_idx][None, :, :], (P, K, 3)
    ).copy().astype(np.float32)
    yhat, _, _ = decode(albedo, normal, light_dirs)        # (P, K, 3)

    # Scatter back to full images
    res = test_set.resolution
    pix = test_set.pixel_indices
    fig, axes = plt.subplots(3, K, figsize=(1.6 * K, 4.5))
    if K == 1:
        axes = axes[:, None]
    for k in range(K):
        gt_view = render_sphere(
            test_set.albedos[sphere_idx], test_set.normals_full,
            test_set.mask, test_set.light_dirs[sphere_idx, k],
        )
        rec_img = np.zeros((res, res, 3), dtype=np.float32)
        rec_img[pix[:, 0], pix[:, 1]] = yhat[:, k, :]
        residual = np.abs(gt_view - rec_img).sum(axis=-1)
        axes[0, k].imshow(np.clip(gt_view, 0, 1))
        axes[1, k].imshow(np.clip(rec_img, 0, 1))
        im = axes[2, k].imshow(residual, cmap="magma", vmin=0, vmax=0.5)
        for r in range(3):
            axes[r, k].set_xticks([]); axes[r, k].set_yticks([])
        axes[0, k].set_title(f"light {k}", fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=9)
    axes[1, 0].set_ylabel("re-render", fontsize=9)
    axes[2, 0].set_ylabel("|residual|", fontsize=9)
    fig.colorbar(im, ax=axes[2, :], shrink=0.7, location="bottom",
                 label="abs residual")
    fig.suptitle(
        f"Per-light reconstruction (sphere {sphere_idx}, "
        f"GT albedo = "
        f"({test_set.albedos[sphere_idx, 0]:.2f},"
        f" {test_set.albedos[sphere_idx, 1]:.2f},"
        f" {test_set.albedos[sphere_idx, 2]:.2f}))",
        fontsize=10,
    )
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


def make_training_curves_figure(history: dict, outpath: str):
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 6.0))
    e = history["epoch"]
    axes[0, 0].plot(e, history["loss"], label="train loss")
    axes[0, 0].plot(e, history["recon_mse"], label="test recon MSE")
    axes[0, 0].set_xlabel("epoch"); axes[0, 0].set_ylabel("MSE")
    axes[0, 0].set_yscale("log"); axes[0, 0].legend(fontsize=8)
    axes[0, 0].set_title("Reconstruction loss", fontsize=10)

    axes[0, 1].plot(e, history["albedo_mse"], color="tab:purple")
    axes[0, 1].set_xlabel("epoch"); axes[0, 1].set_ylabel("MSE")
    axes[0, 1].set_title("Albedo MSE (test, per-sphere mean)", fontsize=10)

    axes[1, 0].plot(e, history["normal_angular_error_deg_mean"], label="mean")
    axes[1, 0].plot(e, history["normal_angular_error_deg_median"],
                    label="median", linestyle="--")
    axes[1, 0].axhline(30.0, color="red", linestyle=":", alpha=0.6,
                       label="30 deg target")
    axes[1, 0].set_xlabel("epoch"); axes[1, 0].set_ylabel("degrees")
    axes[1, 0].set_title("Normal angular error (test)", fontsize=10)
    axes[1, 0].legend(fontsize=8)

    axes[1, 1].plot(e, history["wallclock_s"], color="tab:gray")
    axes[1, 1].set_xlabel("epoch"); axes[1, 1].set_ylabel("seconds")
    axes[1, 1].set_title("Cumulative wallclock", fontsize=10)
    fig.tight_layout()
    fig.savefig(outpath, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {outpath}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-spheres", type=int, default=400)
    p.add_argument("--n-test-spheres", type=int, default=64)
    p.add_argument("--n-lights-per-sphere", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=120)
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--outdir", type=str, default="viz")
    return p.parse_args()


def main():
    args = _parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    train_set = generate_dataset(
        n_spheres=args.n_spheres,
        n_lights_per_sphere=args.n_lights_per_sphere,
        resolution=args.resolution, seed=args.seed,
    )
    test_set = generate_dataset(
        n_spheres=args.n_test_spheres,
        n_lights_per_sphere=args.n_lights_per_sphere,
        resolution=args.resolution, seed=args.seed + 10_000,
    )
    net = build_deep_lambertian_net(
        n_lights=args.n_lights_per_sphere,
        hidden=args.hidden, seed=args.seed,
    )
    net, history = train(
        net, train_set,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_dataset=test_set,
        seed=args.seed,
        verbose=True,
    )
    metrics = evaluate(net, test_set)

    make_dataset_examples_figure(
        test_set, os.path.join(args.outdir, "dataset_examples.png"))
    make_albedo_recovery_figure(
        test_set, metrics, os.path.join(args.outdir, "albedo_recovery.png"))
    make_normal_recovery_figure(
        test_set, metrics, os.path.join(args.outdir, "normal_recovery.png"))
    make_per_light_reconstruction_figure(
        net, test_set, os.path.join(args.outdir, "per_light_reconstruction.png"))
    make_training_curves_figure(
        history, os.path.join(args.outdir, "training_curves.png"))

    print(
        f"\nFinal: ang_mean={metrics['normal_angular_error_deg_mean']:.2f}deg "
        f"ang_median={metrics['normal_angular_error_deg_median']:.2f}deg "
        f"albedo_mse={metrics['albedo_mse']:.4f}"
    )


if __name__ == "__main__":
    main()
