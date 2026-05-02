"""
Render an animated GIF of the transforming auto-encoder learning to disentangle
"what" (presence) from "where" (instantiation parameters).

Each frame shows:
  - Top row: 4 fixed validation pairs as (input | target) at the current epoch
  - Middle row: corresponding 22x22 reconstructions
  - Bottom: training curves (val MSE + R²(dx) + R²(dy)) up to the current step

Usage:
    python3 make_transforming_autoencoders_gif.py
    python3 make_transforming_autoencoders_gif.py --n-epochs 30 --snapshot-every 200 --fps 8
"""
from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transforming_autoencoders import (
    TransformingAutoencoder, train, load_mnist, translate, crop_center,
    _r_squared,
)


# Fixed pairs to track through training so the eye can see reconstruction improve.
def _fixed_pairs(n: int = 4, seed: int = 7):
    images = load_mnist("train")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(images.shape[0])[:n]
    base = images[idx]
    t_in = rng.integers(-5, 6, size=(n, 2))
    dxdy = rng.integers(-5, 6, size=(n, 2)).astype(np.float32)
    img1 = np.stack([translate(base[i], int(t_in[i, 0]), int(t_in[i, 1]))
                     for i in range(n)])
    img2 = np.stack([translate(base[i],
                                int(t_in[i, 0] + dxdy[i, 0]),
                                int(t_in[i, 1] + dxdy[i, 1]))
                     for i in range(n)])
    return img1, img2, dxdy


def render_frame(model: TransformingAutoencoder,
                 history: dict,
                 step: int,
                 fixed_img1: np.ndarray,
                 fixed_img2: np.ndarray,
                 fixed_dxdy: np.ndarray) -> Image.Image:
    n = fixed_img1.shape[0]
    x1 = fixed_img1.reshape(n, -1)
    recon, _ = model.forward(x1, fixed_dxdy)
    recon22 = np.clip(recon, 0, 1).reshape(n, 22, 22)

    fig = plt.figure(figsize=(10, 5.5), dpi=100)
    gs = fig.add_gridspec(3, n + 1, height_ratios=[1.0, 1.0, 1.2],
                          hspace=0.4, wspace=0.20,
                          width_ratios=[1.0] * n + [0.05])

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        # Show input (left half) and target (right half) side by side, with a thin gap
        composite = np.concatenate(
            [fixed_img1[i],
             np.zeros((28, 1), dtype=fixed_img1.dtype),
             fixed_img2[i]], axis=1)
        ax.imshow(composite, cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"input | target\n(dx,dy)=({int(fixed_dxdy[i,0])},{int(fixed_dxdy[i,1])})",
                     fontsize=8)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(recon22[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        if i == 0:
            ax.set_ylabel("recon", fontsize=9)

    # Bottom row: training curves
    ax_curve = fig.add_subplot(gs[2, :])
    if history["epoch"]:
        ax_curve.plot(history["epoch"], history["val_mse"],
                      color="#ff7f0e", linewidth=1.5, label="val MSE")
        ax_curve.set_ylabel("val MSE", color="#ff7f0e", fontsize=9)
        ax_curve.tick_params(axis="y", labelcolor="#ff7f0e")
        ax_curve.set_xlabel("epoch", fontsize=9)
        ax_curve.set_ylim(0.0, max(0.15, max(history["val_mse"]) * 1.05))
        ax_curve.grid(alpha=0.3)

        ax2 = ax_curve.twinx()
        ax2.plot(history["epoch"], history["dx_r2"], color="#2ca02c",
                 linewidth=1.5, label="R²(dx)")
        ax2.plot(history["epoch"], history["dy_r2"], color="#d62728",
                 linewidth=1.5, label="R²(dy)")
        ax2.set_ylabel("R²", fontsize=9)
        ax2.set_ylim(-0.05, 1.0)

        lines1, labels1 = ax_curve.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_curve.legend(lines1 + lines2, labels1 + labels2, loc="lower right",
                        fontsize=8, framealpha=0.85)

    ep = history["epoch"][-1] if history["epoch"] else 0
    fig.suptitle(f"Transforming auto-encoder — step {step}, epoch {ep}",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # Downscale a touch to keep total GIF size in budget.
    w, h = img.size
    img = img.resize((int(w * 0.85), int(h * 0.85)), Image.LANCZOS)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--snapshot-every", type=int, default=200,
                   help="Snapshot every N training steps")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="transforming_autoencoders.gif")
    p.add_argument("--hold-final", type=int, default=10,
                   help="Repeat the last frame this many times")
    args = p.parse_args()

    fixed_img1, fixed_img2, fixed_dxdy = _fixed_pairs(n=4, seed=7)

    frames: list[Image.Image] = []

    def cb(step, model, history, val_imgs, val_transformed, val_dxdy):
        # The training history hasn't logged this epoch yet at intra-epoch
        # snapshots. Build a temporary history view that includes a fresh
        # mid-epoch validation point so the curve animates smoothly.
        tmp_hist = dict(history)
        # If we're between epoch boundaries, append a synthetic epoch entry
        # so the curve stays roughly continuous.
        x_val = val_imgs.reshape(val_imgs.shape[0], -1)
        recon_val, _ = model.forward(x_val, val_dxdy)
        target_val = crop_center(val_transformed, 22).reshape(val_imgs.shape[0], -1)
        val_mse = float(np.mean((recon_val - target_val) ** 2))
        dxdy_pred = model.predict_transformation(val_imgs, val_transformed)
        r2_dx = _r_squared(val_dxdy[:, 0], dxdy_pred[:, 0])
        r2_dy = _r_squared(val_dxdy[:, 1], dxdy_pred[:, 1])
        # If history hasn't recorded this epoch yet, append a transient point
        if not tmp_hist["epoch"] or tmp_hist["step"][-1] < step:
            tmp_hist = {
                "epoch": list(tmp_hist["epoch"]) + [step / args.steps_per_epoch],
                "step":  list(tmp_hist["step"])  + [step],
                "loss":  list(tmp_hist["loss"])  + [val_mse],
                "val_mse": list(tmp_hist["val_mse"]) + [val_mse],
                "dx_r2": list(tmp_hist["dx_r2"]) + [r2_dx],
                "dy_r2": list(tmp_hist["dy_r2"]) + [r2_dy],
            }
        frame = render_frame(model, tmp_hist, step,
                             fixed_img1, fixed_img2, fixed_dxdy)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  step {step}  val_mse={val_mse:.4f}  "
              f"R²(dx)={r2_dx:.3f}  R²(dy)={r2_dy:.3f}")

    print(f"Training {args.n_epochs} epochs, snapshot every {args.snapshot_every} steps...")
    model, history = train(n_epochs=args.n_epochs,
                           steps_per_epoch=args.steps_per_epoch,
                           seed=args.seed,
                           snapshot_callback=cb,
                           snapshot_every=args.snapshot_every,
                           verbose=False)

    print(f"Final R²(dx)={history['dx_r2'][-1]:.3f}  R²(dy)={history['dy_r2'][-1]:.3f}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
