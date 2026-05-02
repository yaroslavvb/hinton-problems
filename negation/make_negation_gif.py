"""
Render an animated GIF showing the negation problem and the learning dynamics.

Layout per frame:
  Top-left:    flag=0 hidden-activation heatmap (8 patterns × n_hidden)
  Top-right:   flag=1 hidden-activation heatmap (8 patterns × n_hidden)
  Bottom-left: Hinton diagram of W1 (input → hidden) and W2 (hidden → output)
  Bottom-right: training curves (loss + per-pattern accuracy) up to current epoch

The gating story is "look how the same hidden units route differently when the
flag flips" — making the flag-gated structure visible as it forms.

Usage:
    python3 make_negation_gif.py             # negation.gif at default settings
    python3 make_negation_gif.py --snapshot-every 10 --fps 14 --seed 0
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

from negation import (NegationMLP, train, make_negation_data, pattern_label)


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.7):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                           facecolor=color, edgecolor="black", linewidth=0.3))


def render_frame(model: NegationMLP,
                 history: dict,
                 epoch: int,
                 max_epoch: int) -> Image.Image:
    n_h = model.n_hidden
    fig = plt.figure(figsize=(11, 7), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.1, 1.0],
                           hspace=0.5, wspace=0.32)

    X, _ = make_negation_data()
    H = model.forward(X)[0]                         # (16, n_hidden)
    order_f0 = [i for i in range(16) if X[i, 0] == 0]
    order_f1 = [i for i in range(16) if X[i, 0] == 1]
    H_f0 = H[order_f0]
    H_f1 = H[order_f1]
    labels_f0 = [pattern_label(X[i]) for i in order_f0]
    labels_f1 = [pattern_label(X[i]) for i in order_f1]

    # ---- top-left: flag=0 heatmap ----
    ax_f0 = fig.add_subplot(gs[0, 0])
    ax_f0.imshow(H_f0.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_f0.set_xticks(range(8))
    ax_f0.set_xticklabels(labels_f0, rotation=45, ha="right", fontsize=8)
    ax_f0.set_yticks(range(n_h))
    ax_f0.set_yticklabels([f"h{i+1}" for i in range(n_h)], fontsize=9)
    ax_f0.set_title("Hidden activations  |  flag = 0  →  output = data",
                    fontsize=10)

    # ---- top-right: flag=1 heatmap ----
    ax_f1 = fig.add_subplot(gs[0, 1])
    im = ax_f1.imshow(H_f1.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    ax_f1.set_xticks(range(8))
    ax_f1.set_xticklabels(labels_f1, rotation=45, ha="right", fontsize=8)
    ax_f1.set_yticks(range(n_h))
    ax_f1.set_yticklabels([f"h{i+1}" for i in range(n_h)], fontsize=9)
    ax_f1.set_title("Hidden activations  |  flag = 1  →  output = NOT data",
                    fontsize=10)

    # ---- bottom-left: weight Hinton diagram (W1 + W2 side by side) ----
    ax_w = fig.add_subplot(gs[1, 0])
    # Compose: rows 0..n_h-1 → W1[i, :] + b1[i]    (5 columns: flag, b1, b2, b3, bias)
    #          rows n_h..n_h+2 → W2[i, :] + b2[i]  (n_h+1 columns: h1..h_n_h, bias)
    # Place into a single (n_h + 3) × max(5, n_h + 1) grid for visual unity.
    n_cols = max(5, n_h + 1)
    rows = []
    for i in range(n_h):
        row = list(model.W1[i, :]) + [model.b1[i]]
        row += [None] * (n_cols - len(row))
        rows.append(row)
    rows.append([None] * n_cols)  # spacer
    for i in range(3):
        row = list(model.W2[i, :]) + [model.b2[i]]
        row += [None] * (n_cols - len(row))
        rows.append(row)
    flat = [v for row in rows for v in row if v is not None]
    max_abs = max(abs(np.array(flat)).max(), 1e-3)
    for r, row in enumerate(rows):
        for c, v in enumerate(row):
            if v is None:
                continue
            _hinton_rect(ax_w, c, r, v, max_abs)
    ax_w.set_xlim(-0.6, n_cols - 0.4)
    ax_w.set_ylim(-0.6, len(rows) - 0.4)
    ax_w.invert_yaxis()
    ax_w.set_aspect("equal")
    # Y labels
    ylabels = [f"h{i+1}" for i in range(n_h)] + [""] + [f"o{i+1}" for i in range(3)]
    ax_w.set_yticks(range(len(rows)))
    ax_w.set_yticklabels(ylabels, fontsize=8)
    ax_w.set_xticks([])
    ax_w.text(2, -0.85, "W₁ rows  (cols: flag, b₁, b₂, b₃, bias)",
              fontsize=8, ha="center")
    wn = history["weight_norm"][-1] if history["weight_norm"] else 0.0
    ax_w.set_title(f"Weights  (red +, blue −,  $\\|W\\|_F$ = {wn:.2f})",
                    fontsize=10)

    # ---- bottom-right: training curves ----
    ax_t = fig.add_subplot(gs[1, 1])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["loss"],
                  color="#9467bd", linewidth=1.5, label="MSE loss")
        ax2 = ax_t.twinx()
        ax2.plot(history["epoch"],
                  np.array(history["pattern_accuracy"]) * 100,
                  color="#d62728", linewidth=1.5, label="pattern accuracy")
        ax2.plot(history["epoch"],
                  np.array(history["accuracy"]) * 100,
                  color="#1f77b4", linewidth=1.0, alpha=0.7,
                  label="bit accuracy")
        ax2.set_ylim(0, 110)
        ax2.set_ylabel("accuracy (%)", fontsize=9)
        if history["converged_epoch"] is not None:
            ax_t.axvline(history["converged_epoch"], color="green",
                         linestyle="--", linewidth=0.9, alpha=0.7)
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max_epoch)
    ax_t.set_ylim(0, 0.5)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_t.tick_params(axis="y", colors="#9467bd")
    ax_t.grid(alpha=0.3)
    pat_acc = (history["pattern_accuracy"][-1] * 100
               if history["pattern_accuracy"] else 0)
    ax_t.set_title(f"Training curves  (pattern acc = {pat_acc:.0f}%)",
                    fontsize=10)

    fig.suptitle(f"Negation 4-{n_h}-3  —  epoch {epoch + 1}",
                  fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=1500)
    p.add_argument("--snapshot-every", type=int, default=15)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="negation.gif")
    p.add_argument("--hold-final", type=int, default=24)
    p.add_argument("--max-frames", type=int, default=120,
                   help="If exceeded, drop intermediate frames evenly.")
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.max_epochs)
        frames.append(frame)

    print(f"Training 4-{args.n_hidden}-3, seed={args.seed}, "
          f"snapshot every {args.snapshot_every}...")
    model, history = train(n_hidden=args.n_hidden, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']}  "
          f"final pattern acc {history['pattern_accuracy'][-1]*100:.0f}%  "
          f"frames captured: {len(frames)}")

    # Subsample if too many frames (keeps GIF small).
    if len(frames) > args.max_frames:
        idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in idx]
        print(f"  subsampled to {len(frames)} frames")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                    duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
