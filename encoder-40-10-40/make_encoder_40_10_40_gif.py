"""
Render an animated GIF showing the 40-10-40 encoder learning.

Layout per frame:
  Top-left:    Hinton-diagram weight matrix (80 visible rows x 10 hidden cols).
  Top-right:   Speed/accuracy curve at the *current* training epoch.
               Shows how the headline metric evolves during training:
               early on the curve is flat near chance; late it climbs to
               the asymptotic 100% line.
  Bottom:      Training curves (codes used + accuracy + code separation)
               up to the current epoch.

Usage:
    python3 make_encoder_40_10_40_gif.py            # encoder_40_10_40.gif
    python3 make_encoder_40_10_40_gif.py --n-cycles 2000 --snapshot-every 80 --fps 8
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from PIL import Image

from encoder_40_10_40 import (EncoderRBM, train, make_encoder_data,
                              speed_accuracy_curve, evaluate_exact,
                              N_PATTERNS, N_GROUP, N_HIDDEN)


SWEEP_BUDGETS = (1, 2, 4, 8, 16, 32, 64, 128)
N_TRIALS_GIF = 20  # smaller than viz to keep frame rendering fast


def render_frame(rbm: EncoderRBM,
                 history: dict,
                 epoch: int,
                 perturbation_epochs: list[int],
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(11, 6.8), dpi=95)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.5, 1.0],
                          width_ratios=[1.0, 1.4],
                          hspace=0.45, wspace=0.30)

    # ---- top-left: weights ----
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
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax_w.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                     facecolor=color, edgecolor="black",
                                     linewidth=0.2))
    ax_w.set_xticks(range(N_HIDDEN))
    ax_w.set_xticklabels([f"H[{j}]" for j in range(N_HIDDEN)], fontsize=7)
    ax_w.set_yticks([0, N_GROUP - 1, N_GROUP, n_v - 1])
    ax_w.set_yticklabels(["V1[0]", f"V1[{N_GROUP-1}]",
                          "V2[0]", f"V2[{N_GROUP-1}]"], fontsize=7)
    ax_w.axhline(N_GROUP - 0.5, color="gray", linewidth=0.5, linestyle=":")
    ax_w.set_title(f"Weights $W$ (red=+, blue=-; $\\|W\\|_F$={np.linalg.norm(W):.1f})",
                   fontsize=9)
    ax_w.set_aspect("equal")

    # ---- top-right: speed/accuracy curve at the current epoch ----
    ax_s = fig.add_subplot(gs[0, 1])
    pt = speed_accuracy_curve(rbm, sweep_budgets=SWEEP_BUDGETS,
                              n_trials=N_TRIALS_GIF, T=1.0,
                              mode="per_trial", seed=42)
    asy_now = evaluate_exact(rbm)
    xs = [b for b, _ in pt]
    ys = [a * 100 for _, a in pt]
    ax_s.plot(xs, ys, marker="o", color="#1f77b4", linewidth=2.0,
              label="per-trial")
    ax_s.axhline(asy_now * 100, color="black", linestyle="--", linewidth=1.0,
                 alpha=0.7,
                 label=f"asymptotic (exact) = {asy_now*100:.0f}%")
    ax_s.axhline(100.0 / N_PATTERNS, color="gray", linestyle=":",
                 linewidth=0.7, alpha=0.7, label=f"chance")
    ax_s.set_xscale("log", base=2)
    ax_s.set_xlabel("Gibbs sweeps", fontsize=8)
    ax_s.set_ylabel("accuracy (%)", fontsize=8)
    ax_s.set_ylim(-2, 105)
    ax_s.set_xticks(xs)
    ax_s.set_xticklabels([str(x) for x in xs], fontsize=7)
    ax_s.set_title(f"Speed/accuracy curve at retrieval (epoch {epoch + 1})",
                   fontsize=10)
    ax_s.grid(alpha=0.3)
    ax_s.legend(loc="lower right", fontsize=7)

    # ---- bottom: training curves ----
    ax_t = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["n_distinct_codes"],
                  color="#d62728", linewidth=1.4,
                  label=f"codes used (target={N_PATTERNS})")
        ax_t.plot(history["epoch"], np.array(history["acc"]) * N_PATTERNS,
                  color="#1f77b4", linewidth=1.0, alpha=0.85,
                  label=f"accuracy x {N_PATTERNS}")
        ax_t.plot(history["epoch"],
                  np.array(history["code_separation"]) * 10,
                  color="#2ca02c", linewidth=1.0, alpha=0.85,
                  label="code sep x 10")
        for pe in perturbation_epochs:
            ax_t.axvline(pe, color="red", linewidth=0.6,
                         linestyle="--", alpha=0.5)
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.axhline(N_PATTERNS, color="black", linewidth=0.5, linestyle=":")
    ax_t.set_xlim(0, max_epoch)
    ax_t.set_ylim(0, N_PATTERNS + 2)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_t.grid(alpha=0.3)

    fig.suptitle(f"40-10-40 encoder — epoch {epoch + 1}", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=2000)
    p.add_argument("--snapshot-every", type=int, default=80)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="encoder_40_10_40.gif")
    p.add_argument("--hold-final", type=int, default=15,
                   help="Repeat the last frame this many times.")
    p.add_argument("--max-frames", type=int, default=30,
                   help="Cap on total frames to keep the GIF under 3 MB.")
    args = p.parse_args()

    frames = []

    def cb(epoch, rbm, history):
        if len(frames) >= args.max_frames:
            return
        frame = render_frame(rbm, history, epoch,
                             history["perturbations"],
                             max_epoch=args.n_cycles)
        frames.append(frame)
        if len(frames) % 5 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
                  f"acc={history['acc'][-1]*100:.0f}%  "
                  f"codes={history['n_distinct_codes'][-1]}/{N_PATTERNS}")

    print(f"Training {args.n_cycles} epochs, snapshot every {args.snapshot_every}...")
    rbm, history = train(n_epochs=args.n_cycles,
                         seed=args.seed,
                         snapshot_callback=cb,
                         snapshot_every=args.snapshot_every,
                         verbose=False)

    print(f"Final accuracy: {history['acc'][-1]*100:.0f}%   "
          f"codes: {history['n_distinct_codes'][-1]}/{N_PATTERNS}   "
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
