"""
Render an animated GIF showing two parallel runs of the 3-bit even-parity
problem:

  Top row:    visible-only Boltzmann (n_hidden = 0) — the negative result.
              Distribution collapses to uniform almost instantly.
  Bottom row: RBM with hidden units — learns the parity ensemble.

Layout per frame:
  Top-left:    8-pattern bar chart (target vs visible-only learned).
  Top-right:   KL trajectory for visible-only vs the log(2) baseline.
  Bottom-left: 8-pattern bar chart (target vs RBM learned).
  Bottom-right: KL trajectory for the RBM, on the same axes.

Usage:
    python3 make_encoder_3_parity_gif.py
    python3 make_encoder_3_parity_gif.py --hidden-epochs 600 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from encoder_3_parity import (
    ALL_PATTERNS, EVEN_MASK, ODD_MASK, PARITY,
    target_distribution, train_visible, train_hidden,
    pattern_string,
)


PATTERN_LABELS = [pattern_string(ALL_PATTERNS[i]) for i in range(8)]
EVEN_COLOR = "#2ca02c"
ODD_COLOR = "#d62728"
TARGET_COLOR = "#888888"


def _bar(ax, p_model: np.ndarray, title: str):
    target = target_distribution()
    x = np.arange(8)
    width = 0.38
    bar_colors = [EVEN_COLOR if PARITY[i] == 0 else ODD_COLOR
                  for i in range(8)]
    ax.bar(x - width / 2, target, width, color=TARGET_COLOR, alpha=0.65,
           edgecolor="black", linewidth=0.4, label="target")
    ax.bar(x + width / 2, p_model, width, color=bar_colors,
           edgecolor="black", linewidth=0.4, label="learned")
    ax.axhline(0.125, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xticks(x)
    ax.set_xticklabels(PATTERN_LABELS, fontfamily="monospace", fontsize=8)
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(EVEN_COLOR if PARITY[i] == 0 else ODD_COLOR)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("probability", fontsize=8)
    ax.set_title(title, fontsize=10)
    ax.tick_params(axis="y", labelsize=7)
    ax.grid(axis="y", alpha=0.3)


def _kl_curve(ax, history: dict, color: str, title: str, xmax: int):
    if history["step"]:
        ax.plot(history["step"], history["kl"], color=color, linewidth=1.5)
        ax.scatter([history["step"][-1]], [history["kl"][-1]],
                   color=color, s=22, zorder=3)
    ax.axhline(np.log(2), color="gray", linewidth=0.7, linestyle="--",
               label="log 2 (uniform)")
    ax.set_xlim(0, xmax)
    ax.set_ylim(-0.02, 0.85)
    ax.set_xlabel("step", fontsize=8)
    ax.set_ylabel("KL", fontsize=8)
    ax.tick_params(labelsize=7)
    ax.set_title(title, fontsize=10)
    ax.legend(loc="upper right", fontsize=7, framealpha=0.9)
    ax.grid(alpha=0.3)


def render_frame(p_visible: np.ndarray, p_hidden: np.ndarray,
                 hist_v: dict, hist_h: dict,
                 step_v: int, step_h: int,
                 max_v: int, max_h: int) -> Image.Image:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=100,
                             gridspec_kw={"width_ratios": [1.6, 1.0]})

    _bar(axes[0, 0], p_visible,
         f"Visible-only Boltzmann (n_hidden=0)  —  step {step_v}")
    _kl_curve(axes[0, 1], hist_v, "#1f77b4", "Visible-only KL", max_v)
    axes[0, 1].text(0.02, 0.05,
                    f"p(even) = {hist_v['p_even_total'][-1]:.2f}\n"
                    f"KL = {hist_v['kl'][-1]:.3f}",
                    transform=axes[0, 1].transAxes,
                    fontsize=8, color="#1f77b4",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#1f77b4", alpha=0.9))

    _bar(axes[1, 0], p_hidden,
         f"RBM with hidden units  —  epoch {step_h}")
    _kl_curve(axes[1, 1], hist_h, "#ff7f0e", "RBM KL", max_h)
    axes[1, 1].text(0.02, 0.05,
                    f"p(even) = {hist_h['p_even_total'][-1]:.2f}\n"
                    f"KL = {hist_h['kl'][-1]:.3f}",
                    transform=axes[1, 1].transAxes,
                    fontsize=8, color="#ff7f0e",
                    bbox=dict(boxstyle="round", facecolor="white",
                              edgecolor="#ff7f0e", alpha=0.9))

    fig.suptitle("3-bit even-parity ensemble: visible-only fails, "
                 "hidden units fix it", fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visible-steps", type=int, default=200)
    p.add_argument("--hidden-epochs", type=int, default=800)
    p.add_argument("--n-frames", type=int, default=60)
    p.add_argument("--n-hidden", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="encoder_3_parity.gif")
    p.add_argument("--hold-final", type=int, default=15)
    args = p.parse_args()

    # Pre-compute snapshot indices for both runs so frames pair up cleanly.
    n_frames = args.n_frames
    visible_idx = sorted(set(np.linspace(0, args.visible_steps - 1,
                                          n_frames, dtype=int).tolist()))
    hidden_idx = sorted(set(np.linspace(0, args.hidden_epochs - 1,
                                         n_frames, dtype=int).tolist()))
    # match length (some indices may collapse if requests overlap)
    n_frames = min(len(visible_idx), len(hidden_idx))
    visible_idx = visible_idx[:n_frames]
    hidden_idx = hidden_idx[:n_frames]
    visible_set = set(visible_idx)
    hidden_set = set(hidden_idx)

    visible_snaps: list[tuple[int, np.ndarray, dict]] = []
    hidden_snaps: list[tuple[int, np.ndarray, dict]] = []

    def cb_v(step, model, history):
        if step in visible_set:
            visible_snaps.append((step,
                                  model.model_distribution().copy(),
                                  {k: list(v) for k, v in history.items()}))

    def cb_h(epoch, model, history):
        if epoch in hidden_set:
            hidden_snaps.append((epoch,
                                 model.model_distribution().copy(),
                                 {k: list(v) for k, v in history.items()}))

    print(f"Training visible-only ({args.visible_steps} steps)...")
    train_visible(n_steps=args.visible_steps, seed=args.seed,
                  snapshot_callback=cb_v, snapshot_every=1, verbose=False)
    print(f"  collected {len(visible_snaps)} snapshots")

    print(f"Training RBM (n_hidden={args.n_hidden}, "
          f"{args.hidden_epochs} epochs)...")
    train_hidden(n_hidden=args.n_hidden, n_epochs=args.hidden_epochs,
                 seed=args.seed,
                 snapshot_callback=cb_h, snapshot_every=1, verbose=False)
    print(f"  collected {len(hidden_snaps)} snapshots")

    # Pair snapshots by index. They're in the same order because both training
    # loops are sequential.
    n_pairs = min(len(visible_snaps), len(hidden_snaps))
    print(f"Rendering {n_pairs} frames...")
    frames = []
    for i in range(n_pairs):
        sv, pv, hv = visible_snaps[i]
        sh, ph, hh = hidden_snaps[i]
        frame = render_frame(pv, ph, hv, hh,
                             step_v=sv + 1, step_h=sh + 1,
                             max_v=args.visible_steps,
                             max_h=args.hidden_epochs)
        frames.append(frame)
        if (i + 1) % 10 == 0:
            print(f"  frame {i+1}/{n_pairs}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
