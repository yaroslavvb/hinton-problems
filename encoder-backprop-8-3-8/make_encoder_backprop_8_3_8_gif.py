"""
Render an animated GIF showing the 8-3-8 backprop encoder learning a 3-bit code.

Layout per frame:
  Top-left:    W2 (hidden -> output) heatmap
  Top-right:   3-D scatter of hidden activations on the 3-cube
  Bottom:      training curves (loss + accuracy + distinct codes) up to current epoch.

Usage:
    python3 make_encoder_backprop_8_3_8_gif.py
    python3 make_encoder_backprop_8_3_8_gif.py --epochs 4000 --snapshot-every 30 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from PIL import Image

from encoder_backprop_8_3_8 import (
    EncoderMLP, train, make_encoder_data, hidden_code_table, n_distinct_codes,
)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#17becf"]
CUBE_EDGES = [
    [(0, 0, 0), (1, 0, 0)], [(1, 0, 0), (1, 1, 0)],
    [(1, 1, 0), (0, 1, 0)], [(0, 1, 0), (0, 0, 0)],
    [(0, 0, 1), (1, 0, 1)], [(1, 0, 1), (1, 1, 1)],
    [(1, 1, 1), (0, 1, 1)], [(0, 1, 1), (0, 0, 1)],
    [(0, 0, 0), (0, 0, 1)], [(1, 0, 0), (1, 0, 1)],
    [(1, 1, 0), (1, 1, 1)], [(0, 1, 0), (0, 1, 1)],
]


def render_frame(model: EncoderMLP, history: dict, epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          hspace=0.50, wspace=0.30)

    data = make_encoder_data()
    codes = hidden_code_table(model, data)
    n_codes = n_distinct_codes(model, data)
    acc = history["acc"][-1] if history["acc"] else 0.0

    # ---- top-left: W2 (hidden -> output) heatmap ----
    ax_w = fig.add_subplot(gs[0, 0])
    W2 = model.W2
    vmax = max(abs(W2).max(), 1e-3)
    im = ax_w.imshow(W2, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax_w.set_xticks(range(8))
    ax_w.set_yticks(range(3))
    ax_w.set_xlabel("output unit", fontsize=9)
    ax_w.set_ylabel("hidden unit", fontsize=9)
    ax_w.set_title(r"$W_2$ (hidden $\to$ output)  "
                   f"$\\|W\\|$={np.linalg.norm(model.W1)+np.linalg.norm(model.W2):.1f}",
                   fontsize=10)
    fig.colorbar(im, ax=ax_w, shrink=0.7)

    # ---- top-right: hidden codes on 3-cube ----
    ax_c = fig.add_subplot(gs[0, 1], projection="3d")
    for a, b in CUBE_EDGES:
        ax_c.plot(*zip(a, b), color="lightgray", linewidth=0.7, zorder=1)
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                ax_c.scatter([x], [y], [z], color="lightgray", s=20,
                             depthshade=False, zorder=2)
    for i in range(8):
        ax_c.scatter(codes[i, 0], codes[i, 1], codes[i, 2],
                     color=PATTERN_COLORS[i], s=80, depthshade=False, zorder=5,
                     edgecolor="black", linewidth=0.5)
    ax_c.set_xlim(-0.05, 1.05)
    ax_c.set_ylim(-0.05, 1.05)
    ax_c.set_zlim(-0.05, 1.05)
    ax_c.set_xlabel(r"$h_0$", fontsize=9)
    ax_c.set_ylabel(r"$h_1$", fontsize=9)
    ax_c.set_zlabel(r"$h_2$", fontsize=9)
    ax_c.set_title(f"Hidden codes  acc={acc*100:.0f}%  "
                   f"distinct={n_codes}/8", fontsize=10)
    ax_c.view_init(elev=20, azim=35)

    # ---- bottom: training curves ----
    ax_t = fig.add_subplot(gs[1, :])
    epochs_so_far = history["epoch"]
    if epochs_so_far:
        loss = np.array(history["loss"])
        # rescale loss for overlay (so the loss curve and accuracy are visible)
        loss_max = max(loss.max(), 1e-3)
        loss_norm = loss / loss_max * 100
        ax_t.plot(epochs_so_far, loss_norm,
                  color="#9467bd", linewidth=1.4,
                  label=f"loss (% of max {loss_max:.2f})")
        ax_t.plot(epochs_so_far, np.array(history["acc"]) * 100,
                  color="#1f77b4", linewidth=1.4, label="accuracy (%)")
        ax_t.plot(epochs_so_far,
                  np.array(history["n_distinct_codes"]) / 8.0 * 100,
                  color="#2ca02c", linewidth=1.4,
                  label="distinct codes / 8 (%)")
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max(max_epoch, 1))
    ax_t.set_ylim(0, 110)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.legend(loc="center right", fontsize=8, framealpha=0.9)
    ax_t.grid(alpha=0.3)

    fig.suptitle(f"8-3-8 backprop encoder — epoch {epoch + 1}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=4000)
    p.add_argument("--snapshot-every", type=int, default=40)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="encoder_backprop_8_3_8.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.epochs)
        frames.append(frame)
        if len(frames) % 10 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
                  f"acc={history['acc'][-1]*100:.0f}%  "
                  f"distinct={history['n_distinct_codes'][-1]}/8")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    model, history = train(n_epochs=args.epochs,
                            seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)

    final_acc = history["acc"][-1] * 100
    final_codes = history["n_distinct_codes"][-1]
    print(f"Final: acc={final_acc:.0f}%  distinct={final_codes}/8  "
          f"epochs_run={len(history['epoch'])}")

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
