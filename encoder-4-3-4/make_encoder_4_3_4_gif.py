"""
Render an animated GIF showing the 4-3-4 over-complete encoder learning
an error-correcting 3-bit code.

Layout per frame:
  Top-left:    Hinton-diagram weight matrix (8 visible rows x 3 hidden cols)
  Top-right:   3-cube viz with the 4 chosen corners highlighted (red edges
               between two chosen corners signal a Hamming-1 collision --
               the bad case the network is trying to escape).
  Bottom:      training curves (accuracy + min-Hamming) up to current epoch.

Usage:
    python3 make_encoder_4_3_4_gif.py            # encoder_4_3_4.gif at defaults
    python3 make_encoder_4_3_4_gif.py --epochs 800 --snapshot-every 8 --fps 14
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)
from PIL import Image

from encoder_4_3_4 import (EncoderRBM, train, make_encoder_data,
                           dominant_codes, is_error_correcting)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PATTERN_LABELS = ["0", "1", "2", "3"]


def render_frame(rbm: EncoderRBM,
                 history: dict,
                 epoch: int,
                 perturbation_epochs: list[int]) -> Image.Image:
    fig = plt.figure(figsize=(11, 6.2), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0],
                          hspace=0.45, wspace=0.30)

    data = make_encoder_data()

    # ---- top-left: weight matrix as Hinton diagram ----
    ax_w = fig.add_subplot(gs[0, 0])
    W = rbm.W
    n_h = W.shape[1]
    ax_w.set_xlim(-0.6, n_h - 0.4)
    ax_w.set_ylim(-0.6, 7.6)
    ax_w.invert_yaxis()
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(8):
        for j in range(n_h):
            w = W[i, j]
            sz = 0.45 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax_w.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                     facecolor=color, edgecolor="black",
                                     linewidth=0.3))
    ax_w.set_xticks(range(n_h))
    ax_w.set_xticklabels([f"H[{j}]" for j in range(n_h)], fontsize=10)
    ax_w.set_yticks(range(8))
    ax_w.set_yticklabels([f"V1[{i}]" for i in range(4)] +
                         [f"V2[{i}]" for i in range(4)], fontsize=8)
    ax_w.axhline(3.5, color="gray", linewidth=0.5, linestyle=":")
    ax_w.set_title("Weights $W_{V \\leftrightarrow H}$  "
                   f"(red=+, blue=-; |W|$_F$ = {np.linalg.norm(W):.2f})",
                   fontsize=10)
    ax_w.set_aspect("equal")

    # ---- top-right: 3-cube ----
    ax_c = fig.add_subplot(gs[0, 1], projection="3d")
    codes = dominant_codes(rbm, data)
    chosen = {tuple(int(x) for x in c) for c in codes}
    code_to_pattern = {tuple(int(x) for x in c): i for i, c in enumerate(codes)}
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]

    for a in corners:
        for b in corners:
            if (sum(int(ax_ != bx_) for ax_, bx_ in zip(a, b)) == 1
                    and a < b):
                bad = (a in chosen) and (b in chosen)
                ax_c.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                          color="red" if bad else "lightgray",
                          linewidth=2.5 if bad else 0.8,
                          alpha=1.0 if bad else 0.6)

    for c in corners:
        ax_c.scatter(*c, s=40, facecolor="white", edgecolor="gray",
                     linewidth=0.8)
    for c, pi in code_to_pattern.items():
        ax_c.scatter(*c, s=180, color=PATTERN_COLORS[pi],
                     edgecolor="black", linewidth=0.8)
        ax_c.text(c[0], c[1], c[2] + 0.12, PATTERN_LABELS[pi],
                  fontsize=10, color="black", ha="center", weight="bold")

    ax_c.set_xlabel("H[0]", fontsize=8)
    ax_c.set_ylabel("H[1]", fontsize=8)
    ax_c.set_zlabel("H[2]", fontsize=8)
    ax_c.set_xticks([0, 1])
    ax_c.set_yticks([0, 1])
    ax_c.set_zticks([0, 1])
    ec = is_error_correcting(rbm, data)
    ax_c.set_title(f"Hidden codes on 3-cube  (acc={history['acc'][-1]*100:.0f}%, "
                   f"min H={history['min_hamming'][-1]}, "
                   f"{'EC' if ec else 'not EC'})", fontsize=10)

    # ---- bottom: training curves ----
    ax_acc = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_acc.plot(history["epoch"], np.array(history["acc"]) * 100,
                    color="#1f77b4", linewidth=1.5, label="accuracy (%)")
        ax_acc.plot(history["epoch"],
                    np.array(history["min_hamming"]) * 33,
                    color="#9467bd", linewidth=1.5,
                    drawstyle="steps-post",
                    label="min Hamming x 33")
        for pe in perturbation_epochs:
            ax_acc.axvline(pe, color="red", linewidth=0.7,
                           linestyle="--", alpha=0.6)
        ax_acc.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
        ax_acc.axhline(66, color="black", linewidth=0.4,
                       linestyle=":", alpha=0.5)
    ax_acc.set_xlim(0, max(history["epoch"][-1] if history["epoch"] else 1, 1))
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xlabel("epoch", fontsize=9)
    ax_acc.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"4-3-4 over-complete encoder -- epoch {epoch + 1}",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=10)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-after", type=int, default=40)
    p.add_argument("--out", type=str, default="encoder_4_3_4.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames = []

    def cb(epoch, rbm, history):
        frame = render_frame(rbm, history, epoch, history["perturbations"])
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
              f"acc={history['acc'][-1]*100:.0f}%  "
              f"min_h={history['min_hamming'][-1]}")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    rbm, history = train(n_epochs=args.epochs,
                         seed=args.seed,
                         perturb_after=args.perturb_after,
                         snapshot_callback=cb,
                         snapshot_every=args.snapshot_every,
                         verbose=False)

    final_acc = history["acc"][-1] * 100
    final_min_h = history["min_hamming"][-1]
    print(f"Final accuracy: {final_acc:.0f}%   "
          f"min Hamming: {final_min_h}   "
          f"restarts: {history['perturbations']}")

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
