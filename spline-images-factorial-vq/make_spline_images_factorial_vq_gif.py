"""
Render an animated GIF of factorial-VQ training on spline images.

Per-frame layout:
  Top-left:    one example spline image (held fixed across frames)
  Top-right:   the four factor posteriors q_d(k|x) shown as bar plots
  Mid-left:    the factorial reconstruction x_hat
  Mid-right:   per-factor contributions (small thumbnails)
  Bottom:      DL bits trajectory for all three trainable models so far

Usage:
    python3 make_spline_images_factorial_vq_gif.py
    python3 make_spline_images_factorial_vq_gif.py --n-epochs 800 \
                                                     --snapshot-every 25 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from spline_images_factorial_vq import (
    IMAGE_H, IMAGE_W,
    generate_spline_images,
    build_factorial_vq, build_baseline_vq, build_separate_vq,
    train_vq,
)


def _bg_train_companion(model, images, n_epochs, seed, history):
    """Lightweight companion trainer that writes its DL trajectory into
    `history` (mutated dict). Used so the GIF can plot the baseline / separate
    curves alongside the factorial trajectory.
    """
    rng = np.random.default_rng(seed + 12345)
    n = len(images)
    cw_start, cw_end = 0.1, 1.0
    eval_every = max(1, n_epochs // 40)
    for epoch in range(n_epochs):
        progress = epoch / max(n_epochs - 1, 1)
        cw = cw_start + (cw_end - cw_start) * progress
        idx = rng.integers(0, n, size=32)
        x_batch = images[idx]
        model.grad_step(x_batch, lr=0.005, kl_weight=cw)
        if (epoch % eval_every == 0) or (epoch == n_epochs - 1):
            info = model.description_length(images)
            history["epoch"].append(epoch)
            history["total_bits"].append(
                float(info["total_nats"].mean() / np.log(2.0)))


def render_frame(model, baseline_history, separate_history,
                  factorial_history, epoch: int, example_x: np.ndarray,
                  example_idx: int) -> Image.Image:
    fig = plt.figure(figsize=(11, 7.5), dpi=100)
    gs = fig.add_gridspec(4, 4, height_ratios=[1.0, 0.55, 1.0, 1.4],
                           hspace=0.65, wspace=0.45)

    out = model._forward(example_x[None, :])
    x_hat = out["x_hat"][0]
    n_dims = model.n_dims
    K = model.n_codes_per_dim

    # ---- row 0 col 0: example spline image ---------------------------
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(example_x.reshape(IMAGE_H, IMAGE_W),
                   cmap="gray", vmin=0, vmax=1)
    ax_img.set_title(f"Input #{example_idx}", fontsize=10)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # ---- row 0 col 1: factorial reconstruction -----------------------
    ax_rec = fig.add_subplot(gs[0, 1])
    ax_rec.imshow(x_hat.reshape(IMAGE_H, IMAGE_W),
                   cmap="gray", vmin=0, vmax=1)
    sq = float(((example_x - x_hat) ** 2).sum())
    ax_rec.set_title(f"Recon (||err||^2={sq:.2f})", fontsize=10)
    ax_rec.set_xticks([]); ax_rec.set_yticks([])

    # ---- row 0 col 2-3: per-factor codeword posteriors q_d -----------
    # Plot all four bar plots in a single subplot, side by side
    ax_q = fig.add_subplot(gs[0, 2:4])
    bar_xs = np.arange(K * n_dims)
    qs = np.concatenate([out["forwards"][d]["q"][0] for d in range(n_dims)])
    colors = ["#2ca02c", "#1f77b4", "#ff7f0e", "#d62728"][:n_dims]
    bar_colors = np.repeat(colors, K)
    ax_q.bar(bar_xs, qs, color=bar_colors, alpha=0.85)
    ax_q.set_ylim(0, 1)
    ax_q.set_xticks([d * K + (K - 1) / 2 for d in range(n_dims)])
    ax_q.set_xticklabels([f"q_{d}" for d in range(n_dims)], fontsize=9)
    ax_q.set_yticks([0.0, 0.5, 1.0])
    ax_q.set_title("Per-factor posteriors q_d(k|x)", fontsize=10)
    ax_q.grid(axis="y", alpha=0.3)
    for d in range(1, n_dims):
        ax_q.axvline(d * K - 0.5, color="black", linewidth=0.5, alpha=0.4)

    # ---- row 1 (skip — spacer) ----------------------------------------

    # ---- row 2: per-factor contribution thumbnails -------------------
    for d in range(n_dims):
        ax = fig.add_subplot(gs[2, d])
        contrib = out["contributions"][d][0]
        vmax = max(abs(contrib).max(), 1e-6)
        ax.imshow(contrib.reshape(IMAGE_H, IMAGE_W), cmap="RdBu_r",
                   vmin=-vmax, vmax=vmax)
        ax.set_title(f"factor {d} contrib.", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])

    # ---- row 3: DL trajectory ----------------------------------------
    ax_dl = fig.add_subplot(gs[3, :])
    if factorial_history["epoch"]:
        ax_dl.plot(factorial_history["epoch"], factorial_history["total_bits"],
                    color="#2ca02c", linewidth=2.0, label="Factorial 4x6")
    if baseline_history["epoch"]:
        ax_dl.plot(baseline_history["epoch"], baseline_history["total_bits"],
                    color="#1f77b4", linewidth=1.5, label="Standard 24-VQ")
    if separate_history["epoch"]:
        ax_dl.plot(separate_history["epoch"], separate_history["total_bits"],
                    color="#ff7f0e", linewidth=1.5, label="Four separate VQs")
    ax_dl.axvline(epoch, color="black", linewidth=0.7, alpha=0.4)
    ax_dl.set_xlabel("epoch")
    ax_dl.set_ylabel("DL (bits / example)")
    ax_dl.set_yscale("log")
    ax_dl.grid(alpha=0.3, which="both")
    ax_dl.legend(loc="upper right", fontsize=9)
    ax_dl.set_title("Bits-back description length over training", fontsize=10)

    fig.suptitle(f"Spline images, factorial VQ — epoch {epoch}", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--n-units-per-dim", type=int, default=6)
    p.add_argument("--out", type=str, default="spline_images_factorial_vq.gif")
    p.add_argument("--hold-final", type=int, default=18)
    p.add_argument("--max-frames", type=int, default=60)
    args = p.parse_args()

    images, controls = generate_spline_images(n_samples=200, seed=args.seed)
    D = images.shape[1]

    # Pick one example to highlight
    rng = np.random.default_rng(args.seed)
    example_idx = int(rng.integers(0, len(images)))
    example_x = images[example_idx]

    # Pre-train the two baselines so their full curves are available in
    # every frame. (Cheap: ~1 second total.)
    print("Pre-training baselines for the trajectory plot...")
    baseline_history = {"epoch": [], "total_bits": []}
    big_vq = build_baseline_vq(n_units=args.n_dims * args.n_units_per_dim,
                                 D=D, seed=args.seed)
    _bg_train_companion(big_vq, images, args.n_epochs, args.seed,
                         baseline_history)

    separate_history = {"epoch": [], "total_bits": []}
    sep_vq = build_separate_vq(n_dims=args.n_dims,
                                 n_units_per_dim=args.n_units_per_dim,
                                 D=D, seed=args.seed)
    _bg_train_companion(sep_vq, images, args.n_epochs, args.seed,
                         separate_history)

    # Train the factorial VQ with snapshots
    fac_vq = build_factorial_vq(n_dims=args.n_dims,
                                  n_units_per_dim=args.n_units_per_dim,
                                  D=D, seed=args.seed)

    factorial_history = {"epoch": [], "total_bits": []}
    frames = []

    snapshot_every = args.snapshot_every
    # Cap total frames
    if args.n_epochs // snapshot_every > args.max_frames:
        snapshot_every = max(1, args.n_epochs // args.max_frames)
        print(f"  snapshot-every adjusted to {snapshot_every} to keep "
              f"frame count <= {args.max_frames}")

    print(f"Training factorial VQ for {args.n_epochs} epochs, "
           f"snapshots every {snapshot_every}...")
    rng_train = np.random.default_rng(args.seed + 1000)
    cw_start, cw_end = 0.1, 1.0
    n = len(images)
    for epoch in range(args.n_epochs):
        progress = epoch / max(args.n_epochs - 1, 1)
        cw = cw_start + (cw_end - cw_start) * progress
        idx = rng_train.integers(0, n, size=32)
        x_batch = images[idx]
        fac_vq.grad_step(x_batch, lr=0.005, kl_weight=cw)
        if (epoch % snapshot_every == 0) or (epoch == args.n_epochs - 1):
            info = fac_vq.description_length(images)
            tot_b = float(info["total_nats"].mean() / np.log(2.0))
            factorial_history["epoch"].append(epoch)
            factorial_history["total_bits"].append(tot_b)
            frame = render_frame(fac_vq, baseline_history, separate_history,
                                   factorial_history, epoch, example_x,
                                   example_idx)
            frames.append(frame)
            print(f"  frame {len(frames):3d}  epoch {epoch:5d}  "
                  f"factorial DL={tot_b:.2f} bits")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
