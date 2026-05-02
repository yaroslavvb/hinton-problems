"""
Animated GIF for T-C discrimination weight-tied conv net.

Per frame layout:
  Top row    — the K discovered 3x3 kernels at the current epoch as
               heatmaps, annotated with the live taxonomy classification.
               THIS IS THE HEADLINE: the bar / compactness / on-centre /
               off-surround detectors crystallising out of random noise.
  Middle row — 8 input patterns laid out as a 2 x 4 grid (T x 4, C x 4).
  Bottom     — training curves (loss + accuracy) up to current epoch with a
               vertical marker for the current frame.

Usage:
    python3 make_t_c_discrimination_gif.py
    python3 make_t_c_discrimination_gif.py --seed 0 --max-epochs 1500 --snapshot-every 25
"""

from __future__ import annotations
import argparse
import os
import warnings
from collections import Counter
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

warnings.filterwarnings("ignore",
                         message=".*not compatible with tight_layout.*")

from t_c_discrimination import (
    WeightTiedConvNet, train, make_dataset,
    visualize_filters, filter_taxonomy,
)


_TAX_COLOR = {
    "bar":          "#d62728",
    "compactness":  "#2ca02c",
    "on-centre":    "#1f77b4",
    "off-centre":   "#ff7f0e",
    "mixed":        "#888888",
    "dead":         "#cccccc",
    "unknown":      "#cccccc",
}


def render_frame(model: WeightTiedConvNet, history: dict,
                  X: np.ndarray, y: np.ndarray, names: list[str],
                  epoch: int, max_epoch: int) -> Image.Image:
    K = model.n_kernels
    fig = plt.figure(figsize=(10.5, 8.0), dpi=100)
    gs = fig.add_gridspec(3, 8, height_ratios=[1.6, 1.4, 1.0],
                           hspace=0.55, wspace=0.30)

    # ---- top: K filter heatmaps, equally split across 8 columns ----
    filters = visualize_filters(model)        # (K, 3, 3)
    types = filter_taxonomy(model)
    max_abs = max(np.abs(filters).max(), 1e-3)
    cols_per_filter = 8 // K
    for k in range(K):
        ax = fig.add_subplot(gs[0, k * cols_per_filter:(k + 1) * cols_per_filter])
        W = filters[k]
        ax.imshow(W, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
        for i in range(3):
            for j in range(3):
                v = W[i, j]
                col = "white" if abs(v) > 0.55 * max_abs else "black"
                ax.text(j, i, f"{v:+.1f}", ha="center", va="center",
                         fontsize=8, color=col)
        col = _TAX_COLOR.get(types[k], "#888")
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"k{k + 1}  [{types[k]}]", fontsize=9, color=col)

    # ---- middle: 8 input patterns as 1 x 8 row ----
    for idx in range(8):
        ax = fig.add_subplot(gs[1, idx])
        ax.imshow(X[idx], cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(names[idx], fontsize=8)

    # ---- bottom: loss + accuracy curves ----
    ax_c = fig.add_subplot(gs[2, :])
    if history["epoch"]:
        ax_c.plot(history["epoch"], history["loss"],
                  color="#9467bd", linewidth=1.5, label="MSE loss")
        ax2 = ax_c.twinx()
        ax2.plot(history["epoch"], np.array(history["accuracy"]) * 100,
                 color="#1f77b4", linewidth=1.5, label="accuracy")
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("accuracy (%)", fontsize=9, color="#1f77b4")
        ax2.tick_params(axis="y", colors="#1f77b4")
        if history["converged_epoch"] is not None:
            ax_c.axvline(history["converged_epoch"], color="green",
                         linestyle="--", linewidth=0.9, alpha=0.7)
        ax_c.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_c.set_xlim(0, max_epoch)
    ax_c.set_ylim(0, 0.16)
    ax_c.set_xlabel("epoch", fontsize=9)
    ax_c.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_c.tick_params(axis="y", colors="#9467bd")
    ax_c.grid(alpha=0.3)

    acc = history["accuracy"][-1] * 100 if history["accuracy"] else 0.0
    counts = Counter(types)
    summary = ", ".join(f"{c} {t}" for t, c in counts.most_common())
    fig.suptitle(f"T-C discrimination via weight-tied 3x3 conv  —  "
                  f"epoch {epoch + 1}  acc {acc:.0f}%  —  {summary}",
                  fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retina-size", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--n-kernels", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=0.5)
    p.add_argument("--max-epochs", type=int, default=1500)
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="t_c_discrimination.gif")
    p.add_argument("--hold-final", type=int, default=20)
    p.add_argument("--target-size-mb", type=float, default=3.0,
                   help="If GIF exceeds this, retry at smaller dpi/frame "
                        "count (best-effort).")
    args = p.parse_args()

    X, y, names = make_dataset(args.retina_size, augment_positions=False)

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, X, y, names, epoch,
                              args.max_epochs)
        frames.append(frame)

    print(f"Training seed={args.seed}, snapshot every "
          f"{args.snapshot_every} epochs (max {args.max_epochs})...")
    model, history = train(retina_size=args.retina_size,
                            kernel_size=args.kernel_size,
                            n_kernels=args.n_kernels, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']}, "
          f"final acc {history['accuracy'][-1] * 100:.0f}%, "
          f"frames captured: {len(frames)}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")
    if size_kb / 1024.0 > args.target_size_mb:
        print(f"  WARNING: exceeds target {args.target_size_mb} MB. "
              f"Consider --snapshot-every {args.snapshot_every * 2}.")


if __name__ == "__main__":
    main()
