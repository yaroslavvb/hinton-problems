"""
Render an animated GIF showing the 1:2:4 anti-symmetric weight pattern emerging.

Layout per frame:
  Top-left:    Hinton diagram of W1 (2 hidden rows x 6 input cols), with the
                 midpoint dashed so the mirror-symmetry is visible.
  Top-right:   bar chart of |w_i| per input position for both hidden units,
                 plus dashed lines at 1x / 2x / 4x times the smallest pair
                 magnitude so the 1:2:4 ratio is visible at a glance.
  Bottom:      training curves (loss + accuracy) + the running pair-ratio
                 trajectories with the paper's "2.0" target dashed in.

Usage:
    python3 make_symmetry_gif.py                   # symmetry.gif at defaults
    python3 make_symmetry_gif.py --seed 1 --snapshot-every 25 --fps 14
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from symmetry import (SymmetryMLP, train, make_symmetry_data,
                       inspect_weight_symmetry)


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.85):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                            facecolor=color, edgecolor="black",
                            linewidth=0.4))


def render_frame(model: SymmetryMLP,
                 history: dict,
                 epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10.5, 6.5), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                           hspace=0.45, wspace=0.30)

    # ---- top-left: Hinton diagram of W1 ----
    ax_w = fig.add_subplot(gs[0, 0])
    W1 = model.W1
    max_abs = max(abs(W1).max(), 1e-3)
    for i in range(W1.shape[0]):
        for j in range(W1.shape[1]):
            _hinton_rect(ax_w, j, i, W1[i, j], max_abs)
    ax_w.set_xlim(-0.7, 5.7)
    ax_w.set_ylim(-0.7, W1.shape[0] - 0.3)
    ax_w.invert_yaxis()
    ax_w.set_xticks(range(6))
    ax_w.set_xticklabels([f"x{i+1}" for i in range(6)])
    ax_w.set_yticks(range(W1.shape[0]))
    ax_w.set_yticklabels([f"h{i+1}" for i in range(W1.shape[0])])
    ax_w.axvline(2.5, color="gray", linewidth=0.7, linestyle=":")
    ax_w.set_aspect("equal")
    ax_w.set_title(f"$W_1$  (red +, blue $-$;  $\\|W_1\\|_F$ = "
                    f"{np.linalg.norm(W1):.2f})", fontsize=10)

    # ---- top-right: |w_i| bar chart with 1:2:4 reference lines ----
    ax_b = fig.add_subplot(gs[0, 1])
    width = 0.36
    xpos = np.arange(6)
    h_colors = ["#2ca02c", "#9467bd"]
    for h in range(W1.shape[0]):
        ax_b.bar(xpos + (h - (W1.shape[0] - 1) / 2) * width,
                  np.abs(W1[h]), width=width,
                  color=h_colors[h % 2], label=f"h{h+1}")

    sym = inspect_weight_symmetry(model)
    sm = np.array(sym["sorted_pair_magnitudes"])
    if sm[0] > 1e-3:
        for r, label in zip([1, 2, 4], ["1x", "2x", "4x"]):
            ax_b.axhline(r * sm[0], color="black", linewidth=0.6,
                          linestyle=":", alpha=0.5)
            ax_b.text(5.7, r * sm[0], f"  {label}", fontsize=8,
                       color="black", va="center")
    ax_b.axvline(2.5, color="gray", linewidth=0.7, linestyle=":")
    ax_b.set_xticks(range(6))
    ax_b.set_xticklabels([f"x{i+1}" for i in range(6)])
    ax_b.set_xlabel("input position", fontsize=9)
    ax_b.set_ylabel(r"$|w_i|$", fontsize=9)
    ax_b.set_xlim(-0.7, 6.5)
    ratio_str = (f"1:{sym['sorted_ratio_medium_to_smallest']:.2f}:"
                  f"{sym['sorted_ratio_largest_to_smallest']:.2f}")
    ax_b.set_title(f"Hidden weight magnitudes  "
                    f"(sorted ratio = {ratio_str})", fontsize=10)
    ax_b.legend(fontsize=8, loc="upper left")
    ax_b.grid(alpha=0.3, axis="y")

    # ---- bottom: training curves + pair ratios on twin axis ----
    ax_t = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["loss"],
                   color="#9467bd", linewidth=1.4, label="MSE loss")
        ax_t.set_yscale("log")
        if history["converged_epoch"] is not None:
            ax_t.axvline(history["converged_epoch"], color="green",
                          linestyle="--", linewidth=0.9, alpha=0.7)
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max_epoch)
    ax_t.set_xlabel("sweep", fontsize=9)
    ax_t.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_t.tick_params(axis="y", colors="#9467bd")
    ax_t.grid(alpha=0.3)

    if history["epoch"]:
        ax_r = ax_t.twinx()
        ax_r.plot(history["epoch"], history["ratio_2_to_1"],
                   color="#2ca02c", linewidth=1.2, alpha=0.85,
                   label="middle:outer")
        ax_r.plot(history["epoch"], history["ratio_4_to_2"],
                   color="#d62728", linewidth=1.2, alpha=0.85,
                   label="inner:middle")
        ax_r.axhline(2.0, color="gray", linestyle=":", linewidth=0.9,
                      alpha=0.7)
        ax_r.set_ylim(0, 5)
        ax_r.set_ylabel("pair ratio", fontsize=9)
        ax_r.legend(fontsize=8, loc="upper right")

    acc_now = history["accuracy"][-1] if history["accuracy"] else 0
    fig.suptitle(f"6-2-1 symmetry  -  sweep {epoch + 1}  -  "
                  f"acc {acc_now*100:.0f}%", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", type=int, default=2200,
                    help="enough to show convergence at default seed")
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--encoding", choices=["pm1", "01"], default="pm1")
    p.add_argument("--snapshot-every", type=int, default=25)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--out", type=str, default="symmetry.gif")
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

    print(f"Training {args.sweeps} sweeps, seed={args.seed}, "
           f"snapshot every {args.snapshot_every}...")
    model, history = train(n_sweeps=args.sweeps, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            encoding=args.encoding, seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ sweep {history['converged_epoch']},  "
           f"final acc {history['accuracy'][-1]*100:.0f}%,  "
           f"frames captured: {len(frames)}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    # Quantize to a global palette to keep the GIF small.
    palette_frame = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    frames_q = [f.quantize(colors=128, method=Image.MEDIANCUT,
                            palette=palette_frame) for f in frames]
    frames_q[0].save(args.out, save_all=True, append_images=frames_q[1:],
                      duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
