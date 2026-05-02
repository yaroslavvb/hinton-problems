"""
Animated GIF for N-bit parity backprop.

Per frame layout (top-to-bottom, two columns wide):

  Top row:
    Left   — thermometer-code panel: hidden activation by input bit-count
             (this is the "interesting property" the spec calls out)
    Right  — Hinton diagram of W1 (input → hidden) at the current epoch

  Bottom row:
    Spans both columns — training curves (loss + accuracy) up to the
    current epoch, with a vertical marker for the current frame.

Usage:
    python3 make_n_bit_parity_gif.py
    python3 make_n_bit_parity_gif.py --n-bits 4 --seed 0 --snapshot-every 30
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from n_bit_parity import (
    ParityMLP, train, make_parity_data, thermometer_score,
    bit_count_for_inputs,
)


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.7):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                           facecolor=color, edgecolor="black", linewidth=0.3))


def render_frame(model: ParityMLP, history: dict, epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(11, 6.5), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0],
                           hspace=0.50, wspace=0.30)

    # ---- top-left: thermometer-code panel ----
    ax_t = fig.add_subplot(gs[0, 0])
    score = thermometer_score(model)
    levels = score["levels"]
    mean_by_level = score["mean_by_level"]
    polarities = score["polarities"]
    cmap = plt.get_cmap("viridis", model.n_hidden)
    for hi in range(model.n_hidden):
        marker = "o" if polarities[hi] == 1 else "s"
        ax_t.plot(levels, mean_by_level[hi],
                  marker=marker, color=cmap(hi),
                  linewidth=1.8, markersize=6, label=f"h{hi+1}")
    ax_t.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.6)
    ax_t.set_xticks(levels)
    ax_t.set_xlabel("input bit-count", fontsize=9)
    ax_t.set_ylabel("mean hidden activation", fontsize=9)
    ax_t.set_ylim(-0.05, 1.10)
    mono = score["mean_monotonicity"]
    ax_t.set_title(f"Hidden code by bit-count  "
                    f"(monotonicity = {mono:.2f})", fontsize=10)
    ax_t.legend(fontsize=7, loc="center right", ncol=1)
    ax_t.grid(alpha=0.3)

    # ---- top-right: W1 Hinton diagram ----
    ax_w = fig.add_subplot(gs[0, 1])
    W = np.column_stack([model.W1, model.b1[:, None]])    # (H, N+1)
    max_abs = max(abs(W).max(), 1e-3)
    n_rows, n_cols = W.shape
    for i in range(n_rows):
        for j in range(n_cols):
            _hinton_rect(ax_w, j, i, W[i, j], max_abs)
    ax_w.set_xlim(-0.6, n_cols - 0.4)
    ax_w.set_ylim(-0.6, n_rows - 0.4)
    ax_w.invert_yaxis()
    ax_w.set_xticks(range(n_cols))
    ax_w.set_xticklabels([f"x{i+1}" for i in range(model.n_bits)] + ["b"],
                          fontsize=8)
    ax_w.set_yticks(range(n_rows))
    ax_w.set_yticklabels([f"h{i+1}" for i in range(n_rows)], fontsize=8)
    ax_w.set_aspect("equal")
    wn = history["weight_norm"][-1] if history["weight_norm"] else 0.0
    ax_w.set_title(f"W1 weights  (red +, blue −,  $\\|W\\|_F$={wn:.1f})",
                    fontsize=10)

    # ---- bottom: loss + accuracy curves ----
    ax_c = fig.add_subplot(gs[1, :])
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
    fig.suptitle(f"N={model.n_bits} parity  —  epoch {epoch + 1}  "
                  f"acc {acc:.0f}%", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-bits", type=int, default=4)
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=3000)
    p.add_argument("--snapshot-every", type=int, default=40)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="n_bit_parity.gif")
    p.add_argument("--hold-final", type=int, default=20)
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.max_epochs)
        frames.append(frame)

    print(f"Training N={args.n_bits}, seed={args.seed}, "
          f"snapshot every {args.snapshot_every}...")
    model, history = train(n_bits=args.n_bits, n_hidden=args.n_hidden,
                            lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']}  "
          f"final acc {history['accuracy'][-1]*100:.0f}%  "
          f"frames captured: {len(frames)}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
