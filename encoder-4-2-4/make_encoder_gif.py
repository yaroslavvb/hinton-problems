"""
Render an animated GIF showing the 4-2-4 encoder learning a 2-bit code.

Layout per frame:
  Top-left:    Hinton-diagram weight matrix (8 visible rows x 2 hidden cols)
  Top-right:   2D scatter of hidden codes for the 4 training patterns,
               with the 4 corners of {0,1}^2 marked as targets.
  Bottom:      training curves (accuracy + code-separation) up to current epoch.

Usage:
    python3 make_encoder_gif.py            # encoder.gif at default settings
    python3 make_encoder_gif.py --epochs 300 --snapshot-every 4 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from encoder_4_2_4 import (EncoderRBM, train, make_encoder_data,
                            mean_pairwise_distance)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PATTERN_LABELS = ["0", "1", "2", "3"]


def render_frame(rbm: EncoderRBM,
                 history: dict,
                 epoch: int,
                 codes: np.ndarray,
                 acc: float,
                 perturbation_epochs: list[int]) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0], hspace=0.45, wspace=0.30)

    # ---- top-left: weight matrix as Hinton diagram ----
    ax_w = fig.add_subplot(gs[0, 0])
    W = rbm.W
    ax_w.set_xlim(-0.6, 1.6)
    ax_w.set_ylim(-0.6, 7.6)
    ax_w.invert_yaxis()
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(8):
        for j in range(2):
            w = W[i, j]
            sz = 0.45 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax_w.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                     facecolor=color, edgecolor="black",
                                     linewidth=0.3))
    ax_w.set_xticks([0, 1])
    ax_w.set_xticklabels(["H[0]", "H[1]"], fontsize=10)
    ax_w.set_yticks(range(8))
    ax_w.set_yticklabels([f"V1[{i}]" for i in range(4)] +
                          [f"V2[{i}]" for i in range(4)], fontsize=8)
    ax_w.axhline(3.5, color="gray", linewidth=0.5, linestyle=":")
    ax_w.set_title("Weights $W_{V \\leftrightarrow H}$  "
                   f"(red = +, blue = -; |W|$_F$ = {np.linalg.norm(W):.2f})",
                   fontsize=10)
    ax_w.set_aspect("equal")

    # ---- top-right: hidden-code scatter ----
    ax_c = fig.add_subplot(gs[0, 1])
    for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax_c.plot(*corner, marker="s", markersize=18, markerfacecolor="none",
                  markeredgecolor="lightgray", linestyle="None", zorder=1)
    for i in range(4):
        ax_c.plot(codes[i, 0], codes[i, 1], "o", color=PATTERN_COLORS[i],
                  markersize=14, zorder=3)
        ax_c.annotate(PATTERN_LABELS[i],
                      (codes[i, 0], codes[i, 1]),
                      textcoords="offset points", xytext=(0, 0),
                      ha="center", va="center",
                      fontsize=9, color="white", weight="bold", zorder=4)
    ax_c.set_xlim(-0.15, 1.15)
    ax_c.set_ylim(-0.15, 1.15)
    ax_c.set_xlabel("$\\langle H_0 \\rangle$", fontsize=10)
    ax_c.set_ylabel("$\\langle H_1 \\rangle$", fontsize=10)
    ax_c.set_aspect("equal")
    ax_c.grid(alpha=0.3)
    ax_c.set_title(f"Hidden codes  (acc = {acc*100:.0f}%)", fontsize=10)

    # ---- bottom: training curves ----
    ax_acc = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_acc.plot(history["epoch"], np.array(history["acc"]) * 100,
                    color="#1f77b4", linewidth=1.5, label="accuracy (%)")
        ax_acc.plot(history["epoch"],
                    np.array(history["code_separation"]) * 50,
                    color="#2ca02c", linewidth=1.5,
                    label="code separation x 50")
        for pe in perturbation_epochs:
            ax_acc.axvline(pe, color="red", linewidth=0.7,
                           linestyle="--", alpha=0.6)
        ax_acc.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_acc.set_xlim(0, max(history["epoch"][-1] if history["epoch"] else 1, 1))
    ax_acc.set_ylim(0, 110)
    ax_acc.set_xlabel("epoch", fontsize=9)
    ax_acc.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"4-2-4 encoder — epoch {epoch + 1}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--snapshot-every", type=int, default=5)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="encoder.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames = []

    def cb(epoch, rbm, history):
        codes = np.array([rbm.hidden_code(make_encoder_data()[i, :4],
                                          n_steps=20)
                          for i in range(4)])
        frame = render_frame(rbm, history, epoch, codes,
                             history["acc"][-1],
                             history["perturbations"])
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
              f"acc={history['acc'][-1]*100:.0f}%")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    rbm, history = train(n_epochs=args.epochs,
                         seed=args.seed,
                         perturb_after=80,
                         snapshot_callback=cb,
                         snapshot_every=args.snapshot_every,
                         verbose=False)

    final_acc = history["acc"][-1] * 100
    print(f"Final accuracy: {final_acc:.0f}%   restarts: {history['perturbations']}")

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
