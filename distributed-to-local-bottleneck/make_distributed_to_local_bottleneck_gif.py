"""
Render an animated GIF showing the 4 hidden values emerging during training.

Layout per frame:
  Top-left:    output sigmoids (4 curves over h) + the 4 current h values
               marked as colored vertical lines, with argmax-region shading
  Top-right:   bar chart of the 4 hidden values per pattern, with paper
               targets as dashed reference lines
  Bottom:      training curves (loss + accuracy) + each per-pattern hidden
               trajectory up to the current epoch

Usage:
    python3 make_distributed_to_local_bottleneck_gif.py
    python3 make_distributed_to_local_bottleneck_gif.py --snapshot-every 10 --fps 12 --seed 0
"""

from __future__ import annotations
import argparse
import os
import warnings
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

# tight_layout warns when twin axes / mixed grids are present; the result is fine.
warnings.filterwarnings("ignore",
                        message=".*not compatible with tight_layout.*")

from distributed_to_local_bottleneck import (
    BottleneckMLP, train, generate_dataset, hidden_values, sigmoid,
)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PATTERN_LABELS = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
PAPER_HIDDEN_TARGETS = np.array([0.0, 0.2, 0.6, 1.0])


def render_frame(model: BottleneckMLP, history: dict, epoch: int,
                 max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=90)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.3, 1.0],
                          hspace=0.50, wspace=0.30)

    X, _ = generate_dataset()
    hv = hidden_values(model, (X, None))
    acc = history["accuracy"][-1] if history["accuracy"] else 0.0

    # ---- top-left: 4 output sigmoids over h, with current h-positions ----
    ax_o = fig.add_subplot(gs[0, 0])
    h_grid = np.linspace(0, 1, 200)
    W2 = model.W2.ravel()
    b2 = model.b2
    out = sigmoid(np.outer(h_grid, W2) + b2)
    argmax = np.argmax(out, axis=1)
    # shade argmax regions
    for j in range(4):
        idx = np.where(argmax == j)[0]
        if len(idx) == 0:
            continue
        starts, ends = [idx[0]], []
        for k in range(1, len(idx)):
            if idx[k] != idx[k - 1] + 1:
                ends.append(idx[k - 1])
                starts.append(idx[k])
        ends.append(idx[-1])
        for a, b in zip(starts, ends):
            ax_o.axvspan(h_grid[a], h_grid[b], color=PATTERN_COLORS[j],
                         alpha=0.10, zorder=0)
    for j in range(4):
        ax_o.plot(h_grid, out[:, j], color=PATTERN_COLORS[j], linewidth=1.4)
    for i in range(4):
        ax_o.axvline(float(hv[i]), color=PATTERN_COLORS[i], linewidth=1.3,
                     linestyle="--", alpha=0.85)
    ax_o.set_xlim(0, 1)
    ax_o.set_ylim(0, 1.05)
    ax_o.set_xlabel("hidden $h$", fontsize=9)
    ax_o.set_ylabel("output $o_j$", fontsize=9)
    ax_o.set_title(f"Output sigmoids over $h$  (acc={acc*100:.0f}%)",
                   fontsize=10)
    ax_o.grid(alpha=0.3)

    # ---- top-right: bar chart of 4 hidden values per pattern ----
    ax_b = fig.add_subplot(gs[0, 1])
    bar_x = np.arange(4)
    ax_b.bar(bar_x, hv, color=PATTERN_COLORS, edgecolor="black",
             linewidth=0.6)
    for tgt in PAPER_HIDDEN_TARGETS:
        ax_b.axhline(tgt, color="#888888", linestyle=":", linewidth=0.8,
                     alpha=0.7)
    ax_b.set_xticks(bar_x)
    ax_b.set_xticklabels(PATTERN_LABELS, fontsize=9)
    ax_b.set_ylim(-0.02, 1.10)
    ax_b.set_ylabel("hidden $h$", fontsize=9)
    sorted_h = np.sort(hv)
    ax_b.set_title(f"Hidden values  sorted=[{', '.join(f'{v:.2f}' for v in sorted_h)}]",
                   fontsize=10)
    ax_b.grid(alpha=0.3, axis="y")

    # ---- bottom: training curves (loss + accuracy + hidden trajectories) ----
    ax_t = fig.add_subplot(gs[1, :])
    epochs_so_far = history["epoch"]
    if epochs_so_far:
        loss = np.array(history["loss"])
        acc_arr = np.array(history["accuracy"])
        hv_traj = np.array(history["hidden_values"])  # (N, 4)
        loss_max = max(loss.max(), 1e-3)
        ax_t.plot(epochs_so_far, loss / loss_max * 100, color="#9467bd",
                  linewidth=1.2, label=f"loss (% of {loss_max:.2f})")
        ax_t.plot(epochs_so_far, acc_arr * 100, color="black", linewidth=1.4,
                  label="accuracy (%)")
        for i in range(4):
            ax_t.plot(epochs_so_far, hv_traj[:, i] * 100,
                      color=PATTERN_COLORS[i], linewidth=1.0, alpha=0.85,
                      label=f"h {PATTERN_LABELS[i]}")
        for p in history.get("perturbations", []):
            ax_t.axvline(p, color="red", linewidth=0.4, alpha=0.5)
        ax_t.axvline(epoch + 1, color="gray", linewidth=1.0, alpha=0.5)
    ax_t.set_xlim(0, max(max_epoch, 1))
    ax_t.set_ylim(-5, 110)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.set_ylabel("value (%)", fontsize=9)
    ax_t.legend(loc="center right", fontsize=7, framealpha=0.85, ncol=1)
    ax_t.grid(alpha=0.3)

    fig.suptitle(f"distributed-to-local-bottleneck — epoch {epoch + 1}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-sweeps", type=int, default=1500)
    p.add_argument("--snapshot-every", type=int, default=15)
    p.add_argument("--fps", type=int, default=15)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--out", type=str, default="distributed_to_local_bottleneck.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.n_sweeps)
        frames.append(frame)
        if len(frames) % 10 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
                  f"acc={history['accuracy'][-1]*100:.0f}%")

    print(f"Training {args.n_sweeps} epochs  snapshot_every={args.snapshot_every}...")
    model, history = train(seed=args.seed, n_sweeps=args.n_sweeps,
                           lr=args.lr, momentum=args.momentum,
                           init_scale=args.init_scale,
                           snapshot_callback=cb,
                           snapshot_every=args.snapshot_every,
                           early_stop=True,
                           verbose=False)

    final_acc = history["accuracy"][-1] * 100
    final_hv = hidden_values(model)
    print(f"Final acc={final_acc:.0f}%  hidden={[f'{v:.3f}' for v in final_hv]}  "
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
