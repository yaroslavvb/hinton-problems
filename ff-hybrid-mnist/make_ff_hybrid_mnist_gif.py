"""
Render an animated GIF showing FF-hybrid-MNIST training dynamics.

Layout per frame:
  Top-left:    real digit + corresponding hybrid negative (the FF problem)
  Top-right:   per-layer histograms of goodness for real vs hybrid
               (shows the layers learning to separate them)
  Bottom:      per-layer FF training loss over epochs, current epoch marked

Usage:
    python3 make_ff_hybrid_mnist_gif.py --n-epochs 30 --snapshot-every 1
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from ff_hybrid_mnist import (
    build_ff_mlp,
    forward_all_layers,
    goodness,
    load_mnist,
    make_hybrid_image,
    train_unsupervised,
)


def render_frame(layers, history, epoch, mnist, eval_idx, eval_b_idx,
                 rng_eval, n_blur=6) -> Image.Image:
    """One frame: examples + per-layer goodness histograms + loss curves."""
    L = len(layers)
    fig = plt.figure(figsize=(11, 6.0), dpi=90)
    gs = fig.add_gridspec(2, max(L, 3) + 1,
                          height_ratios=[1.0, 1.1],
                          hspace=0.45, wspace=0.35)

    # ---- top-left: example digit + hybrid ----
    a_idx = int(eval_idx[0])
    b_idx = int(eval_b_idx[0])
    a = mnist["test_images"][a_idx]
    b = mnist["test_images"][b_idx]
    hyb = make_hybrid_image(a, b, rng_eval, n_blur=n_blur)

    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.imshow(a, cmap="gray_r", vmin=0, vmax=1)
    ax_a.set_xticks([]); ax_a.set_yticks([])
    ax_a.set_title("real digit (+)", fontsize=9)

    ax_h = fig.add_subplot(gs[0, 1])
    ax_h.imshow(hyb, cmap="gray_r", vmin=0, vmax=1)
    ax_h.set_xticks([]); ax_h.set_yticks([])
    ax_h.set_title("hybrid neg ($-$)", fontsize=9)

    # ---- top-right: per-layer goodness histograms (small samples) ----
    sample_pos_imgs = mnist["test_images"][eval_idx]
    sample_b_imgs = mnist["test_images"][eval_b_idx]
    pos_flat = sample_pos_imgs.reshape(len(eval_idx), -1)
    neg_flat = np.empty_like(pos_flat)
    for k in range(len(eval_idx)):
        neg_flat[k] = make_hybrid_image(sample_pos_imgs[k], sample_b_imgs[k],
                                        rng_eval, n_blur=n_blur).reshape(-1)
    pos_acts = forward_all_layers(layers, pos_flat)
    neg_acts = forward_all_layers(layers, neg_flat)

    for li in range(L):
        ax = fig.add_subplot(gs[0, 2 + li] if 2 + li < gs.ncols
                             else gs[0, gs.ncols - 1])
        g_pos = goodness(pos_acts[li])
        g_neg = goodness(neg_acts[li])
        lo = float(min(g_pos.min(), g_neg.min(), 0.0))
        hi = float(max(g_pos.max(), g_neg.max(), 4.0))
        bins = np.linspace(lo, hi, 24)
        ax.hist(g_pos, bins=bins, alpha=0.6, color="#1f77b4",
                label="pos", density=True)
        ax.hist(g_neg, bins=bins, alpha=0.6, color="#d62728",
                label="neg", density=True)
        ax.axvline(2.0, color="black", linestyle="--", linewidth=0.7)
        ax.set_title(f"layer {li+1}", fontsize=9)
        ax.set_xlabel("goodness", fontsize=7)
        ax.tick_params(labelsize=7)
        if li == 0:
            ax.legend(fontsize=7, loc="upper right")

    # ---- bottom: training loss curves ----
    ax_l = fig.add_subplot(gs[1, :])
    palette = ["#1f77b4", "#2ca02c", "#ff7f0e", "#9467bd", "#8c564b"]
    for li in range(L):
        ax_l.plot(history["epoch"],
                  history[f"layer{li+1}_loss"],
                  label=f"L{li+1}", color=palette[li % len(palette)],
                  linewidth=1.5)
    ax_l.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_l.set_xlabel("FF epoch", fontsize=9)
    ax_l.set_ylabel("loss", fontsize=9)
    ax_l.legend(fontsize=8, ncol=L, loc="upper right")
    ax_l.grid(alpha=0.3)
    ax_l.set_xlim(0, max(history["epoch"][-1] + 1, 2))
    ax_l.set_ylim(0, max(0.75,
                         max(max(history[f"layer{li+1}_loss"])
                             for li in range(L)) * 1.05))

    fig.suptitle(f"FF hybrid-MNIST — epoch {epoch + 1}  "
                 f"(wallclock {history['wallclock'][-1]:.0f}s)",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--layer-sizes", type=str,
                   default="784,1000,1000,1000,1000")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--n-blur", type=int, default=6)
    p.add_argument("--n-train", type=int, default=0,
                   help="Subsample MNIST training set (0 = all 60k).")
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", type=str, default="ff_hybrid_mnist.gif")
    p.add_argument("--n-eval", type=int, default=300,
                   help="How many examples for the histogram in each frame.")
    p.add_argument("--hold-final", type=int, default=8)
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    rng = np.random.default_rng(args.seed)
    rng_eval = np.random.default_rng(args.seed + 999)

    print("loading MNIST...")
    mnist = load_mnist()

    if args.n_train > 0:
        idx = rng.permutation(mnist["train_images"].shape[0])[:args.n_train]
        mnist["train_images"] = mnist["train_images"][idx]
        mnist["train_labels"] = mnist["train_labels"][idx]

    n_test = mnist["test_images"].shape[0]
    eval_idx = rng_eval.choice(n_test, size=args.n_eval, replace=False)
    eval_b_idx = rng_eval.choice(n_test, size=args.n_eval, replace=False)

    layers = build_ff_mlp(layer_sizes, rng)
    frames: list[Image.Image] = []

    def cb(epoch, layers, history):
        if (epoch + 1) % max(1, args.snapshot_every) != 0 \
           and epoch != args.n_epochs - 1:
            return
        # use a fixed eval rng so the frame-to-frame mask is consistent
        # for the example pair, but the histogram pool stays the same set
        local_rng = np.random.default_rng(args.seed + 999)
        frame = render_frame(layers, history, epoch, mnist,
                             eval_idx, eval_b_idx, local_rng,
                             n_blur=args.n_blur)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}")

    print(f"training {args.n_epochs} epochs (snapshot every "
          f"{args.snapshot_every})...")
    train_unsupervised(layers, mnist["train_images"],
                       n_epochs=args.n_epochs, lr=args.lr,
                       batch_size=args.batch_size,
                       threshold=args.threshold, rng=rng,
                       n_blur=args.n_blur,
                       snapshot_callback=cb, verbose=True)

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
