"""
Render an animated GIF showing the XOR problem and the learning dynamics.

Layout per frame:
  Top-left:    decision-surface heatmap with the 4 training patterns overlaid
  Top-right:   Hinton diagram of the current weights
  Bottom:      training curves (loss + accuracy) up to the current epoch

Usage:
    python3 make_xor_gif.py             # xor.gif at default settings
    python3 make_xor_gif.py --snapshot-every 4 --fps 12 --seed 0
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

# tight_layout warns when twin axes are present; the result is still fine.
warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from xor import XorMLP, train, make_xor_data


PATTERN_COLORS = ["#1f77b4", "#d62728", "#d62728", "#1f77b4"]
PATTERN_MARKERS = ["o", "s", "s", "o"]


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.7):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                           facecolor=color, edgecolor="black", linewidth=0.3))


def render_frame(model: XorMLP,
                 history: dict,
                 epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0],
                           hspace=0.45, wspace=0.30)

    # ---- top-left: decision surface ----
    ax_d = fig.add_subplot(gs[0, 0])
    grid = 80
    xs = np.linspace(-0.1, 1.1, grid)
    XX, YY = np.meshgrid(xs, xs)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    o = model.predict(pts).reshape(grid, grid)
    ax_d.imshow(o, extent=(-0.1, 1.1, -0.1, 1.1), origin="lower",
                cmap="RdBu_r", vmin=0, vmax=1, aspect="equal", alpha=0.85)
    ax_d.contour(XX, YY, o, levels=[0.5], colors="black", linewidths=1.0)

    X, y = make_xor_data()
    for i, (xi, yi) in enumerate(zip(X, y.ravel())):
        ax_d.scatter(xi[0], xi[1], marker=PATTERN_MARKERS[i],
                     c=PATTERN_COLORS[i], s=180, edgecolor="black",
                     linewidths=1.2, zorder=3)
        ax_d.annotate(f"{int(yi)}", (xi[0], xi[1]),
                      textcoords="offset points", xytext=(0, 0),
                      ha="center", va="center", fontsize=9,
                      color="white", weight="bold", zorder=4)
    ax_d.set_xlim(-0.1, 1.1); ax_d.set_ylim(-0.1, 1.1)
    ax_d.set_xticks([0, 1]); ax_d.set_yticks([0, 1])
    ax_d.set_xlabel("$x_1$", fontsize=10)
    ax_d.set_ylabel("$x_2$", fontsize=10)
    acc = history["accuracy"][-1] if history["accuracy"] else 0.0
    ax_d.set_title(f"Decision surface  (acc = {acc*100:.0f}%)", fontsize=10)

    # ---- top-right: weight Hinton diagram ----
    ax_w = fig.add_subplot(gs[0, 1])
    if model.arch == "2-2-1":
        # row 0: x1->h1, x2->h1, b->h1 ; row 1: x1->h2, x2->h2, b->h2
        # row 2: h1->o, h2->o, b->o
        rows = []
        for i in range(2):
            rows.append([model.W1[i, 0], model.W1[i, 1], model.b1[i]])
        rows.append([model.W2[0, 0], model.W2[0, 1], model.b2[0]])
        labels_y = ["h₁", "h₂", "o"]
        labels_x = ["←x₁", "←x₂", "←bias"]
    else:
        # row 0: x1->h, x2->h, b->h
        # row 1: x1->o (skip), x2->o (skip), h->o, b->o  — pad to 3 cols + Wskip
        rows = []
        rows.append([model.W1[0, 0], model.W1[0, 1], model.b1[0], 0.0])
        rows.append([model.Wskip[0, 0], model.Wskip[0, 1],
                     model.W2[0, 0], model.b2[0]])
        labels_y = ["h", "o"]
        labels_x = ["←x₁", "←x₂", "←h/skip", "←bias"]
    rows = np.array(rows)
    max_abs = max(abs(rows).max(), 1e-3)
    n_rows, n_cols = rows.shape
    for i in range(n_rows):
        for j in range(n_cols):
            if not (model.arch == "2-1-2-skip" and i == 0 and j == 3):
                _hinton_rect(ax_w, j, i, rows[i, j], max_abs)
    ax_w.set_xlim(-0.6, n_cols - 0.4)
    ax_w.set_ylim(-0.6, n_rows - 0.4)
    ax_w.invert_yaxis()
    ax_w.set_xticks(range(n_cols))
    ax_w.set_xticklabels(labels_x, fontsize=9)
    ax_w.set_yticks(range(n_rows))
    ax_w.set_yticklabels(labels_y, fontsize=10)
    ax_w.set_aspect("equal")
    wn = history["weight_norm"][-1] if history["weight_norm"] else 0.0
    ax_w.set_title(f"Weights  (red +, blue −;  $\\|W\\|_F$ = {wn:.2f})",
                    fontsize=10)

    # ---- bottom: loss + accuracy curves ----
    ax_t = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["loss"],
                  color="#9467bd", linewidth=1.5, label="MSE loss")
        ax2 = ax_t.twinx()
        ax2.plot(history["epoch"], np.array(history["accuracy"]) * 100,
                 color="#1f77b4", linewidth=1.5, label="accuracy (%)")
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("accuracy (%)", fontsize=9, color="#1f77b4")
        ax2.tick_params(axis="y", colors="#1f77b4")
        if history["converged_epoch"] is not None:
            ax_t.axvline(history["converged_epoch"], color="green",
                         linestyle="--", linewidth=0.9, alpha=0.7)
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max_epoch)
    ax_t.set_ylim(0, 0.15)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_t.tick_params(axis="y", colors="#9467bd")
    ax_t.grid(alpha=0.3)

    fig.suptitle(f"XOR  {model.arch}  —  epoch {epoch + 1}",
                  fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["2-2-1", "2-1-2-skip"], default="2-2-1")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="xor.gif")
    p.add_argument("--hold-final", type=int, default=24)
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.max_epochs)
        frames.append(frame)

    print(f"Training {args.arch}, seed={args.seed}, "
          f"snapshot every {args.snapshot_every}...")
    model, history = train(arch=args.arch, lr=args.lr, momentum=args.momentum,
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
