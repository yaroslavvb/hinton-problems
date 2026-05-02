"""
Render an animated GIF of the bars Helmholtz machine learning to dream.

Each frame shows:

  Top-left:   8x8 grid of fantasy samples drawn from the *current* generative
              net  (top -> hidden -> visible)
  Top-right:  per-hidden-unit generative receptive field
              (4x4 image of  p(v | h_j = 1, others off)), one row per snapshot
  Bottom:     KL[p_data || p_model] curve up to the current step

The script reads `viz/canonical_snapshots.npz` if present (snapshots saved
during a previous training run), otherwise re-trains the network.
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import bars
from _train_canonical import train_and_save


# ------------------------------------------------------------------
# Frame rendering
# ------------------------------------------------------------------

def _draw_image_grid(ax, imgs: np.ndarray, n_rows: int, n_cols: int,
                     pad: int = 1) -> None:
    H, W = bars.H, bars.W
    canvas = np.full((n_rows * (H + pad) + pad, n_cols * (W + pad) + pad),
                     0.5, dtype=np.float32)
    for k in range(min(len(imgs), n_rows * n_cols)):
        r, c = divmod(k, n_cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = imgs[k].reshape(H, W)
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


def _draw_hidden_field_strip(ax, W_hv: np.ndarray, b_v: np.ndarray) -> None:
    """Plot 8 receptive fields (one per hidden unit) side by side."""
    n_h = W_hv.shape[0]
    H, W = bars.H, bars.W
    pad = 1
    canvas = np.full((H + 2 * pad, n_h * (W + pad) + pad), 0.5,
                     dtype=np.float32)
    for j in range(n_h):
        h_one = np.zeros((1, n_h), dtype=np.float32)
        h_one[0, j] = 1.0
        z_v = h_one @ W_hv + b_v
        p_v = bars.sigmoid(z_v).reshape(H, W)
        x0 = pad + j * (W + pad)
        canvas[pad:pad + H, x0:x0 + W] = p_v
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0,
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])


def render_frame(step: int, kl: float,
                 fantasy: np.ndarray,
                 W_hv: np.ndarray, b_v: np.ndarray,
                 history_steps: np.ndarray,
                 history_kl: np.ndarray,
                 max_step: int,
                 max_kl: float) -> Image.Image:
    fig = plt.figure(figsize=(9, 6), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[3.5, 1.0, 1.6],
                          hspace=0.40, wspace=0.20)

    # top-left: fantasy samples
    ax_f = fig.add_subplot(gs[0, 0])
    _draw_image_grid(ax_f, fantasy, n_rows=8, n_cols=8)
    ax_f.set_title("Fantasy samples  $v \\sim p_\\mathrm{model}$", fontsize=10)

    # top-right: hidden specialization
    ax_h = fig.add_subplot(gs[0, 1])
    _draw_image_grid(ax_h,
                     np.stack([_hidden_field(W_hv, b_v, j)
                               for j in range(W_hv.shape[0])]),
                     n_rows=2, n_cols=4)
    ax_h.set_title("Per-hidden generative field  $p(v | h_j=1)$",
                   fontsize=10)

    # mid: long strip of hidden specialisation, current step only
    ax_strip = fig.add_subplot(gs[1, :])
    _draw_hidden_field_strip(ax_strip, W_hv, b_v)
    ax_strip.set_title("h[0] ... h[7]  generative fields", fontsize=9)

    # bottom: KL trajectory
    ax_k = fig.add_subplot(gs[2, :])
    if len(history_steps):
        ax_k.plot(history_steps, history_kl, color="#1f77b4", linewidth=1.5)
    ax_k.set_xlim(0, max_step)
    ax_k.set_ylim(0.05, max(max_kl, 1.0))
    ax_k.set_yscale("log")
    ax_k.axvline(step, color="black", linewidth=1.0, alpha=0.4)
    ax_k.axhline(0.10, color="red", linewidth=0.8, linestyle="--", alpha=0.7,
                 label="paper target 0.10 bits")
    ax_k.set_xlabel("wake-sleep step")
    ax_k.set_ylabel("KL  (bits)")
    ax_k.legend(loc="upper right", fontsize=8, framealpha=0.9)
    ax_k.grid(alpha=0.3)

    fig.suptitle(f"bars Helmholtz machine — step {step:>10,d}    "
                 f"KL = {kl:.3f} bits",
                 fontsize=11, y=0.995)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def _hidden_field(W_hv: np.ndarray, b_v: np.ndarray, j: int) -> np.ndarray:
    n_h = W_hv.shape[0]
    h_one = np.zeros((1, n_h), dtype=np.float32)
    h_one[0, j] = 1.0
    z_v = h_one @ W_hv + b_v
    return bars.sigmoid(z_v).reshape(bars.N_VIS)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--snapshot-every", type=int, default=50_000)
    p.add_argument("--out", type=str, default="bars.gif")
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--hold-final", type=int, default=12)
    p.add_argument("--snapshot-file", type=str,
                   default="viz/canonical_snapshots.npz")
    p.add_argument("--reuse", action="store_true",
                   help="if set, read snapshots from --snapshot-file rather "
                        "than re-training")
    args = p.parse_args()

    if args.reuse and os.path.exists(args.snapshot_file):
        print(f"Loading snapshots from {args.snapshot_file}")
        a = np.load(args.snapshot_file)
        steps = a["steps"]
        W_hv_seq = a["W_hv"]
        b_v_seq = a["b_v"]
        fantasy_seq = a["fantasy_samples"]
        kl_seq = a["kl_bits"]
    else:
        print(f"Training (seed={args.seed}, n_steps={args.n_steps}, "
              f"snapshot_every={args.snapshot_every})...")
        model, history, snap = train_and_save(
            args.seed, args.n_steps, args.lr, args.batch_size,
            eval_every=args.snapshot_every,
            snapshot_every=args.snapshot_every,
            outdir="viz")
        steps = np.array([s[0] for s in snap])
        W_hv_seq = np.array([s[1] for s in snap])
        b_v_seq = np.array([s[3] for s in snap])
        fantasy_seq = np.array([s[6] for s in snap])
        kl_seq = np.array([s[7] for s in snap])

    max_step = int(steps[-1])
    max_kl = float(max(kl_seq.max(), 1.0))
    frames = []
    for k in range(len(steps)):
        frame = render_frame(int(steps[k]), float(kl_seq[k]),
                             fantasy_seq[k],
                             W_hv_seq[k], b_v_seq[k],
                             steps[:k + 1], kl_seq[:k + 1],
                             max_step, max_kl)
        frames.append(frame)
        if (k + 1) % 5 == 0 or k == len(steps) - 1:
            print(f"  frame {k + 1}/{len(steps)}  step={int(steps[k])}  "
                  f"KL={float(kl_seq[k]):.3f}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 50)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
