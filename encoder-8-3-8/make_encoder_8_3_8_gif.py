"""
Render an animated GIF showing the 8-3-8 encoder learning a 3-bit code.

Layout per frame:
  Top-left:    Hinton-diagram weight matrix (16 visible rows x 3 hidden cols)
  Top-right:   3-cube viz with hidden codes for the 8 training patterns,
               with the 8 corners of {0,1}^3 marked as targets.
  Bottom:      training curves (n_distinct_codes + accuracy + sep) up to current epoch.

Usage:
    python3 make_encoder_8_3_8_gif.py            # encoder_8_3_8.gif at default settings
    python3 make_encoder_8_3_8_gif.py --n-cycles 4000 --snapshot-every 40 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from PIL import Image

from encoder_8_3_8 import (EncoderRBM, train, make_encoder_data,
                           N_PATTERNS, N_GROUP, N_HIDDEN)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#17becf", "#bcbd22"]
PATTERN_LABELS = [str(i) for i in range(N_PATTERNS)]


def _unit_cube_corners() -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def render_frame(rbm: EncoderRBM,
                 history: dict,
                 epoch: int,
                 codes_continuous: np.ndarray,
                 codes_dominant: list[tuple[int, ...]],
                 acc: float,
                 perturbation_epochs: list[int]) -> Image.Image:
    fig = plt.figure(figsize=(11, 7), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0],
                          width_ratios=[1.0, 1.4],
                          hspace=0.45, wspace=0.30)

    # ---- top-left: weight matrix as Hinton diagram ----
    ax_w = fig.add_subplot(gs[0, 0])
    W = rbm.W
    n_v = W.shape[0]
    ax_w.set_xlim(-0.6, N_HIDDEN - 0.4)
    ax_w.set_ylim(-0.6, n_v - 0.4)
    ax_w.invert_yaxis()
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(n_v):
        for j in range(N_HIDDEN):
            w = W[i, j]
            sz = 0.55 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax_w.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                     facecolor=color, edgecolor="black",
                                     linewidth=0.3))
    ax_w.set_xticks(range(N_HIDDEN))
    ax_w.set_xticklabels([f"H[{j}]" for j in range(N_HIDDEN)], fontsize=9)
    ax_w.set_yticks(range(n_v))
    ax_w.set_yticklabels([f"V1[{i}]" for i in range(N_GROUP)] +
                         [f"V2[{i}]" for i in range(N_GROUP)], fontsize=6)
    ax_w.axhline(N_GROUP - 0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_w.set_title(f"Weights $W$  (red=+, blue=-; $\\|W\\|_F$={np.linalg.norm(W):.1f})",
                   fontsize=9)
    ax_w.set_aspect("equal")

    # ---- top-right: 3-cube hidden codes ----
    ax_c = fig.add_subplot(gs[0, 1], projection="3d")
    for i, c1 in enumerate(_unit_cube_corners()):
        for c2 in _unit_cube_corners()[i + 1:]:
            if sum(int(a != b) for a, b in zip(c1, c2)) == 1:
                ax_c.plot(*zip(c1, c2), color="lightgray",
                          linewidth=0.8, zorder=1)
    for c in _unit_cube_corners():
        ax_c.scatter(*c, s=80, facecolors="none", edgecolors="gray",
                     linewidth=0.8, zorder=2)

    # Use the *continuous* p(H_j=1|V1) marginals so we can see the chain
    # converging to corners over time, instead of jumping discretely.
    for pat in range(N_PATTERNS):
        x, y, z = codes_continuous[pat]
        ax_c.scatter(x, y, z, s=140, color=PATTERN_COLORS[pat],
                     edgecolors="black", linewidth=0.6, zorder=3)
        ax_c.text(x, y, z, PATTERN_LABELS[pat], fontsize=7, color="white",
                  weight="bold", ha="center", va="center", zorder=4)

    n_used = len(set(codes_dominant))
    ax_c.set_xticks([0, 1]); ax_c.set_yticks([0, 1]); ax_c.set_zticks([0, 1])
    ax_c.set_xlabel("H[0]", fontsize=8)
    ax_c.set_ylabel("H[1]", fontsize=8)
    ax_c.set_zlabel("H[2]", fontsize=8)
    ax_c.set_title(f"Hidden codes  ({n_used}/{N_PATTERNS} corners, "
                   f"acc={acc*100:.0f}%)", fontsize=10)

    # ---- bottom: training curves ----
    ax_acc = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_acc.plot(history["epoch"], history["n_distinct_codes"],
                    color="#d62728", linewidth=1.5, label="codes used (max=8)")
        ax_acc.plot(history["epoch"], np.array(history["acc"]) * 8,
                    color="#1f77b4", linewidth=1.0, alpha=0.8,
                    label="accuracy x 8")
        ax_acc.plot(history["epoch"],
                    np.array(history["code_separation"]) * 4,
                    color="#2ca02c", linewidth=1.0, alpha=0.8,
                    label="code sep x 4")
        for pe in perturbation_epochs:
            ax_acc.axvline(pe, color="red", linewidth=0.7,
                           linestyle="--", alpha=0.5)
        ax_acc.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_acc.axhline(N_PATTERNS, color="black", linewidth=0.5, linestyle=":")
    ax_acc.set_xlim(0, max(history["epoch"][-1] if history["epoch"] else 1, 1))
    ax_acc.set_ylim(0, N_PATTERNS + 0.5)
    ax_acc.set_xlabel("epoch", fontsize=9)
    ax_acc.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"8-3-8 encoder — epoch {epoch + 1}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=4000)
    p.add_argument("--snapshot-every", type=int, default=40)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="encoder_8_3_8.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    p.add_argument("--max-frames", type=int, default=80,
                   help="Cap on total frames to keep GIF under 3 MB.")
    args = p.parse_args()

    frames = []
    data = make_encoder_data()

    def cb(epoch, rbm, history):
        if len(frames) >= args.max_frames:
            return
        codes_cont = np.array([rbm.hidden_code_exact(data[i, :N_GROUP])
                               for i in range(N_PATTERNS)])
        codes_dom = [rbm.dominant_code(data[i, :N_GROUP])
                     for i in range(N_PATTERNS)]
        frame = render_frame(rbm, history, epoch, codes_cont, codes_dom,
                             history["acc"][-1],
                             history["perturbations"])
        frames.append(frame)
        if len(frames) % 10 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
                  f"acc={history['acc'][-1]*100:.0f}%  "
                  f"codes={history['n_distinct_codes'][-1]}/{N_PATTERNS}")

    print(f"Training {args.n_cycles} epochs, snapshot every {args.snapshot_every}...")
    rbm, history = train(n_epochs=args.n_cycles,
                         seed=args.seed,
                         snapshot_callback=cb,
                         snapshot_every=args.snapshot_every,
                         verbose=False)

    final_acc = history["acc"][-1] * 100
    final_codes = history["n_distinct_codes"][-1]
    print(f"Final accuracy: {final_acc:.0f}%   codes: {final_codes}/{N_PATTERNS}   "
          f"restarts: {history['perturbations']}")
    print(f"Total frames captured: {len(frames)}")

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
