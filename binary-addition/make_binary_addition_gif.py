"""
Render an animated GIF showing the binary-addition problem and the
learning dynamics.

Layout per frame:
  Top-left:    The 16 input patterns + their learned 3-bit outputs as a
                 (16 x 7) heatmap (4 input bits | 3 output bits), with the
                 column showing the network's current prediction next to
                 the target.
  Top-right:   Hinton diagram of W1 + W2 (current weight state).
  Bottom:      training curves -- loss + per-pattern accuracy -- with a
                 vertical cursor at the current sweep.

Usage:
    python3 make_binary_addition_gif.py            # binary_addition.gif at default
    python3 make_binary_addition_gif.py --seed 10 --snapshot-every 10 --fps 14
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

# tight_layout warns about twin axes; the result is still fine.
warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from binary_addition import (BinaryAdditionMLP, train, generate_dataset)


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.85):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                            facecolor=color, edgecolor="black",
                            linewidth=0.4))


def render_frame(model: BinaryAdditionMLP,
                 history: dict,
                 epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10.5, 6.5), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                           hspace=0.50, wspace=0.30)

    # ---- top-left: input/output table heatmap ----
    ax_t = fig.add_subplot(gs[0, 0])
    X, y = generate_dataset()
    _, o = model.forward(X)
    # Build a (16, 8) matrix: 4 input bits | gap | 3 output preds | 3 targets
    grid = np.zeros((16, 8))
    grid[:, :4] = X
    grid[:, 4] = np.nan  # spacer
    grid[:, 5:8] = o
    # Plot
    im = ax_t.imshow(grid, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    # Overlay target characters in red over wrong predictions
    pred_bits = (o >= 0.5).astype(int)
    for i in range(16):
        a = int(2 * X[i, 0] + X[i, 1])
        b = int(2 * X[i, 2] + X[i, 3])
        target_dec = a + b
        # Display target on right-most column tick area
        for j in range(3):
            if pred_bits[i, j] != int(y[i, j]):
                ax_t.text(5 + j, i, "x", ha="center", va="center",
                           color="black", fontsize=7, weight="bold")
            else:
                ax_t.text(5 + j, i, str(int(y[i, j])),
                           ha="center", va="center",
                           color="white" if abs(o[i, j] - 0.5) > 0.3 else "black",
                           fontsize=7)
    ax_t.set_xticks([0, 1, 2, 3, 5, 6, 7])
    ax_t.set_xticklabels(["a₁", "a₀", "b₁", "b₀", "s₂", "s₁", "s₀"],
                          fontsize=8)
    ax_t.set_yticks(range(16))
    ax_t.set_yticklabels(
        [f"{int(2*X[i,0]+X[i,1])}+{int(2*X[i,2]+X[i,3])}={int(2*X[i,0]+X[i,1])+int(2*X[i,2]+X[i,3])}"
         for i in range(16)], fontsize=7)
    ax_t.axvline(3.5, color="white", linewidth=1.5)
    n_correct = int(np.sum(np.all(pred_bits == y, axis=1)))
    ax_t.set_title(f"inputs | predictions    "
                    f"({n_correct}/16 patterns correct)",
                    fontsize=10)

    # ---- top-right: Hinton diagram of W1 + W2 stacked ----
    ax_w = fig.add_subplot(gs[0, 1])
    n_h = model.n_hidden
    # Stack: top n_h rows = W1 (4 inputs), bottom 3 rows = W2 (n_h hidden)
    # Column count = max(4, n_h) padded; we draw two stacked Hinton blocks
    W1 = model.W1
    W2 = model.W2
    # Place W1 at rows [0..n_h-1], cols [0..3]
    # Place W2 at rows [n_h+1..n_h+3], cols [0..n_h-1]
    max_abs = max(abs(W1).max(), abs(W2).max(), 1e-3)
    for i in range(n_h):
        for j in range(4):
            _hinton_rect(ax_w, j, i, W1[i, j], max_abs)
        # bias of W1
        _hinton_rect(ax_w, 4, i, model.b1[i], max_abs)
    # separator row
    for i in range(3):
        for j in range(n_h):
            _hinton_rect(ax_w, j, n_h + 1 + i, W2[i, j], max_abs)
        _hinton_rect(ax_w, n_h, n_h + 1 + i, model.b2[i], max_abs)
    ax_w.set_xlim(-0.7, 4.7)
    ax_w.set_ylim(-0.7, n_h + 4 - 0.3)
    ax_w.invert_yaxis()
    # X labels: only 4 input + bias for the W1 rows
    ax_w.set_xticks([0, 1, 2, 3, 4])
    ax_w.set_xticklabels(["in 1", "in 2", "in 3", "in 4", "bias"],
                          fontsize=8)
    yticks = list(range(n_h)) + list(range(n_h + 1, n_h + 4))
    ylabels = [f"h{i+1}" for i in range(n_h)] + ["s₂", "s₁", "s₀"]
    ax_w.set_yticks(yticks)
    ax_w.set_yticklabels(ylabels, fontsize=8)
    ax_w.axhline(n_h - 0.4, color="gray", linewidth=0.4, linestyle=":")
    ax_w.axhline(n_h + 0.4, color="gray", linewidth=0.4, linestyle=":")
    wn = float(np.linalg.norm(W1))
    ax_w.set_title(f"$W_1$ (top) / $W_2$ (bottom)  "
                    f"$\\|W_1\\|_F$ = {wn:.2f}", fontsize=10)
    ax_w.set_aspect("equal")

    # ---- bottom: loss + per-pattern accuracy ----
    ax_l = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_l.plot(history["epoch"], history["loss"],
                   color="#9467bd", linewidth=1.4, label="MSE loss")
        ax_l.set_yscale("log")
        ax_l.set_ylim(1e-4, 0.3)
        if history["converged_epoch"] is not None:
            ax_l.axvline(history["converged_epoch"], color="green",
                          linestyle="--", linewidth=0.9, alpha=0.7)
        ax_l.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_l.set_xlim(0, max_epoch)
    ax_l.set_xlabel("sweep", fontsize=9)
    ax_l.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_l.tick_params(axis="y", colors="#9467bd")
    ax_l.grid(alpha=0.3)

    if history["epoch"]:
        ax_r = ax_l.twinx()
        ax_r.plot(history["epoch"],
                   np.array(history["accuracy_pattern"]) * 100,
                   color="#d62728", linewidth=1.4,
                   label="per-pattern accuracy (%)")
        ax_r.plot(history["epoch"],
                   np.array(history["accuracy_bit"]) * 100,
                   color="#1f77b4", linewidth=1.0, alpha=0.7,
                   label="per-bit accuracy (%)")
        ax_r.set_ylim(0, 110)
        ax_r.set_ylabel("accuracy (%)", fontsize=9, color="#d62728")
        ax_r.tick_params(axis="y", colors="#d62728")
        ax_r.legend(fontsize=8, loc="lower right")

    acc_now = (history["accuracy_pattern"][-1]
               if history["accuracy_pattern"] else 0)
    fig.suptitle(f"binary-addition  arch={model.arch}  "
                  f"sweep {epoch + 1}  acc(pat) {acc_now*100:.0f}%",
                  fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["4-3-3", "4-2-3"], default="4-3-3")
    p.add_argument("--sweeps", type=int, default=1500,
                    help="enough to show convergence at default seed")
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--snapshot-every", type=int, default=15)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--seed", type=int, default=10)
    p.add_argument("--out", type=str, default="binary_addition.gif")
    p.add_argument("--hold-final", type=int, default=20)
    p.add_argument("--max-frame-side", type=int, default=900,
                    help="downsize frames to keep GIF small")
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.sweeps)
        if max(frame.size) > args.max_frame_side:
            scale = args.max_frame_side / max(frame.size)
            new_size = (int(frame.size[0] * scale),
                         int(frame.size[1] * scale))
            frame = frame.resize(new_size, Image.LANCZOS)
        frames.append(frame)

    print(f"Training {args.arch}, seed={args.seed}, sweeps={args.sweeps}, "
           f"snapshot every {args.snapshot_every}...")
    model, history = train(arch=args.arch, n_sweeps=args.sweeps, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            seed=args.seed, snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ sweep {history['converged_epoch']},  "
           f"final acc(pat) {history['accuracy_pattern'][-1]*100:.0f}%,  "
           f"frames captured: {len(frames)}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    palette_frame = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    frames_q = [f.quantize(colors=128, method=Image.MEDIANCUT,
                            palette=palette_frame) for f in frames]
    frames_q[0].save(args.out, save_all=True, append_images=frames_q[1:],
                      duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
