"""
Build `deep_lambertian_spheres.gif` showing the network's recovery progress.

For each frame we hold the same held-out sphere fixed and snapshot:
  - input view under one light direction (constant across frames)
  - GT albedo swatch
  - recovered albedo swatch (current epoch)
  - GT normal map (RGB-encoded)
  - recovered normal map (RGB-encoded, current epoch)
  - angular-error heatmap (current epoch)

Run:
    python3 make_deep_lambertian_spheres_gif.py --seed 0 --n-epochs 80 \\
            --snapshot-every 4 --fps 10
"""

from __future__ import annotations

import argparse
import io
import os

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np

from deep_lambertian_spheres import (
    build_deep_lambertian_net,
    encode,
    evaluate,
    generate_dataset,
    render_sphere,
    train,
)
from visualize_deep_lambertian_spheres import _normal_to_rgb, _pixels_to_image


def render_frame(epoch: int, history: dict, net, test_set, sphere_idx: int = 0,
                 view_idx: int = 0, dpi: int = 95) -> np.ndarray:
    """Render one GIF frame as an RGB numpy image."""
    P = test_set.n_pixels
    K = test_set.n_lights
    res = test_set.resolution
    pix = test_set.pixel_indices

    # Encode this sphere's pixels with current weights
    obs = test_set.pixel_obs[sphere_idx]                  # (P, K, 3)
    rgb_flat = obs.reshape(P, K * 3)
    ldir = np.broadcast_to(
        test_set.light_dirs[sphere_idx][None, :, :], (P, K, 3)
    ).reshape(P, K * 3)
    x = np.concatenate([rgb_flat, ldir], axis=-1).astype(np.float32)
    pred_albedo, pred_normal, _ = encode(net, x)

    # Per-sphere predicted albedo (mean over pixels)
    albedo_avg = pred_albedo.mean(axis=0)

    # Build full-image normal predictions
    pred_normal_img = _pixels_to_image(pred_normal, pix, res, 3)
    pred_normal_rgb = _normal_to_rgb(pred_normal_img, test_set.mask)
    gt_normal_rgb = _normal_to_rgb(test_set.normals_full, test_set.mask)

    # Angular error heatmap
    gt_pix = test_set.pixel_normals[sphere_idx]
    cos = np.clip(np.sum(pred_normal * gt_pix, axis=-1), -1.0, 1.0)
    ang = np.degrees(np.arccos(cos))
    ang_img = np.zeros((res, res), dtype=np.float32)
    ang_img[pix[:, 0], pix[:, 1]] = ang

    # Input view
    view = render_sphere(
        test_set.albedos[sphere_idx], test_set.normals_full,
        test_set.mask, test_set.light_dirs[sphere_idx, view_idx],
    )

    # Albedo swatches
    swatch = 32
    gt_alb = np.clip(test_set.albedos[sphere_idx], 0, 1)
    pred_alb = np.clip(albedo_avg, 0, 1)
    gt_swatch = np.tile(gt_alb, (swatch, swatch, 1))
    pred_swatch = np.tile(pred_alb, (swatch, swatch, 1))

    fig, axes = plt.subplots(2, 3, figsize=(7.5, 5.0))
    axes[0, 0].imshow(np.clip(view, 0, 1))
    axes[0, 0].set_title(f"input view (light {view_idx})", fontsize=9)
    axes[0, 1].imshow(gt_swatch)
    axes[0, 1].set_title("GT albedo", fontsize=9)
    axes[0, 2].imshow(pred_swatch)
    axes[0, 2].set_title(
        f"recovered albedo "
        f"(MSE={history['albedo_mse'][-1]:.3f})", fontsize=9)
    axes[1, 0].imshow(gt_normal_rgb)
    axes[1, 0].set_title("GT normal map", fontsize=9)
    axes[1, 1].imshow(pred_normal_rgb)
    axes[1, 1].set_title("recovered normals", fontsize=9)
    im = axes[1, 2].imshow(ang_img, cmap="magma", vmin=0, vmax=60)
    axes[1, 2].set_title(
        f"angular err  mean={ang.mean():.1f}deg "
        f"median={np.median(ang):.1f}deg", fontsize=9)
    for a in axes.flat:
        a.set_xticks([]); a.set_yticks([])
    fig.suptitle(
        f"Deep Lambertian Spheres   epoch {epoch + 1}   "
        f"loss={history['loss'][-1]:.4f}   "
        f"recon={history['recon_mse'][-1]:.4f}",
        fontsize=11,
    )
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    # Render to RGB array via PNG buffer (avoids the deprecated
    # `tostring_rgb` / `buffer_rgba` divergence between matplotlib versions).
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    plt.close(fig)
    buf.seek(0)
    img = imageio.imread(buf)
    if img.shape[-1] == 4:
        img = img[..., :3]
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-spheres", type=int, default=400)
    p.add_argument("--n-test-spheres", type=int, default=32)
    p.add_argument("--n-lights-per-sphere", type=int, default=6)
    p.add_argument("--n-epochs", type=int, default=80,
                   help="GIF uses fewer epochs by default to keep size small")
    p.add_argument("--resolution", type=int, default=32)
    p.add_argument("--hidden", type=int, default=192)
    p.add_argument("--batch-size", type=int, default=1024)
    p.add_argument("--lr", type=float, default=2e-2)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--sphere-idx", type=int, default=0)
    p.add_argument("--view-idx", type=int, default=0)
    p.add_argument("--out", type=str, default="deep_lambertian_spheres.gif")
    p.add_argument("--dpi", type=int, default=85,
                   help="Lower DPI -> smaller GIF.")
    args = p.parse_args()

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

    frames: list[np.ndarray] = []

    def snapshot(epoch, current_net, current_history, eval_set, _metrics):
        if (epoch % args.snapshot_every == 0) or (epoch == args.n_epochs - 1):
            frames.append(render_frame(
                epoch, current_history, current_net, eval_set,
                sphere_idx=args.sphere_idx,
                view_idx=args.view_idx, dpi=args.dpi,
            ))
            print(
                f"  frame {len(frames):3d} captured at epoch {epoch + 1}, "
                f"ang_mean="
                f"{current_history['normal_angular_error_deg_mean'][-1]:.2f}"
            )

    # Capture initial frame (untrained net)
    pre_metrics = evaluate(net, test_set)
    pre_history = {
        "loss": [pre_metrics["recon_mse"]],
        "recon_mse": [pre_metrics["recon_mse"]],
        "albedo_mse": [pre_metrics["albedo_mse"]],
        "normal_angular_error_deg_mean":
            [pre_metrics["normal_angular_error_deg_mean"]],
        "normal_angular_error_deg_median":
            [pre_metrics["normal_angular_error_deg_median"]],
    }
    frames.append(render_frame(-1, pre_history, net, test_set,
                               sphere_idx=args.sphere_idx,
                               view_idx=args.view_idx, dpi=args.dpi))
    print(f"  frame   1 captured at epoch 0 (untrained)")

    train(
        net, train_set,
        n_epochs=args.n_epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        eval_dataset=test_set,
        snapshot_callback=snapshot,
        snapshot_every=args.snapshot_every,
        seed=args.seed,
        verbose=False,
    )

    # Hold last frame for ~1.5s
    hold_frames = max(1, int(args.fps * 1.5))
    frames = frames + [frames[-1]] * hold_frames

    print(f"writing {len(frames)} frames to {args.out} at {args.fps} fps")
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    size_mb = os.path.getsize(args.out) / (1024 * 1024)
    print(f"wrote {args.out} ({size_mb:.2f} MB)")


if __name__ == "__main__":
    main()
