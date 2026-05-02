"""
Render an animated GIF showing the bars-RBM's receptive fields evolve
during CD-1 training.

Layout per frame:
  Top:    8 (or 16) receptive fields, one per hidden unit, as 4x4 images
  Bottom: training curves (recon MSE + bars-covered count) up to current epoch

Usage:
    python3 make_bars_rbm_gif.py
    python3 make_bars_rbm_gif.py --n-hidden 16 --snapshot-every 4
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from bars_rbm import BarsRBM, train, per_unit_bar_purity


def render_frame(rbm: BarsRBM,
                 history: dict,
                 epoch: int,
                 max_epoch: int,
                 max_recon: float,
                 n_bars: int) -> Image.Image:
    h, w = rbm.image_shape
    n = rbm.n_hidden
    cols = min(8, n)
    rows = (n + cols - 1) // cols

    fig = plt.figure(figsize=(cols * 0.85, rows * 1.05 + 2.4), dpi=100)
    gs = fig.add_gridspec(rows + 1, cols,
                          height_ratios=[1.0] * rows + [1.6],
                          hspace=0.45, wspace=0.20)

    score = per_unit_bar_purity(rbm)
    bar_names = ([f"H{i}" for i in range(h)] + [f"V{j}" for j in range(w)])
    max_abs = max(float(np.abs(rbm.W).max()), 1e-3)

    for j in range(rows * cols):
        r, c = j // cols, j % cols
        ax = fig.add_subplot(gs[r, c])
        if j < n:
            rf = rbm.W[:, j].reshape(h, w)
            ax.imshow(rf, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
                      interpolation="nearest")
            ax.set_title(f"u{j}: {bar_names[score['best_bar'][j]]} "
                         f"({score['purity'][j]:.2f})",
                         fontsize=7)
        ax.set_xticks([])
        ax.set_yticks([])

    # Bottom: training curves spanning all columns
    ax_curve = fig.add_subplot(gs[rows, :])
    if history["epoch"]:
        ax_curve.plot(history["epoch"], history["recon_error"],
                      color="#9467bd", linewidth=1.4, label="recon MSE")
        ax_curve.set_ylim(0, max(max_recon, 1e-3) * 1.05)
        ax_curve.set_xlim(0, max_epoch)
        ax_curve.set_ylabel("MSE", color="#9467bd", fontsize=9)
        ax_curve.set_xlabel("epoch", fontsize=9)
        ax_curve.tick_params(axis="y", labelcolor="#9467bd", labelsize=8)
        ax_curve.tick_params(axis="x", labelsize=8)
        ax_curve.grid(alpha=0.3)

        ax_bars = ax_curve.twinx()
        ax_bars.plot(history["epoch"], history["bars_covered"],
                     color="#2ca02c", linewidth=1.4, label="bars covered")
        ax_bars.set_ylim(-0.5, n_bars + 0.5)
        ax_bars.set_ylabel(f"bars (max {n_bars})", color="#2ca02c", fontsize=9)
        ax_bars.tick_params(axis="y", labelcolor="#2ca02c", labelsize=8)
        ax_bars.axhline(n_bars, color="gray", linestyle="--", linewidth=0.6)

    fig.suptitle(f"bars-RBM, epoch {epoch + 1}  "
                 f"(bars covered {score['bars_covered']}/{n_bars}, "
                 f"mean purity {score['mean_purity']:.2f})",
                 fontsize=10, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-hidden", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--snapshot-every", type=int, default=8)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="bars_rbm.gif")
    p.add_argument("--hold-final", type=int, default=15,
                   help="Repeat the last frame this many times.")
    p.add_argument("--max-width", type=int, default=560,
                   help="Resize frames to this max width (kB control).")
    args = p.parse_args()

    frames = []
    state = {"max_recon": 0.0}

    def cb(epoch, rbm, history):
        if history["recon_error"]:
            state["max_recon"] = max(state["max_recon"],
                                     history["recon_error"][0])
        frame = render_frame(rbm, history, epoch,
                             max_epoch=args.n_epochs,
                             max_recon=state["max_recon"],
                             n_bars=2 * rbm.image_shape[0])
        if args.max_width and frame.width > args.max_width:
            new_h = int(frame.height * args.max_width / frame.width)
            frame = frame.resize((args.max_width, new_h), Image.LANCZOS)
        frames.append(frame)

    print(f"Training {args.n_epochs} epochs (seed={args.seed}, "
          f"n_hidden={args.n_hidden}, snapshot_every={args.snapshot_every})...")
    rbm, history = train(n_epochs=args.n_epochs,
                         n_hidden=args.n_hidden,
                         seed=args.seed,
                         snapshot_callback=cb,
                         snapshot_every=args.snapshot_every,
                         verbose=False)

    score = per_unit_bar_purity(rbm)
    print(f"Final: bars covered {score['bars_covered']}/{score['n_bars']}, "
          f"mean purity {score['mean_purity']:.3f}, "
          f"recon MSE {history['recon_error'][-1]:.4f}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    # Convert to a shared palette for smaller GIFs.
    palette_frame = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    pal_frames = [f.quantize(colors=128, method=Image.MEDIANCUT,
                             palette=palette_frame) for f in frames]
    pal_frames[0].save(args.out, save_all=True, append_images=pal_frames[1:],
                       duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(pal_frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
