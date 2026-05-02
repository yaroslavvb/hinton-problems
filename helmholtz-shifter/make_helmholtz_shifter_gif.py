"""
Render an animated GIF of the Helmholtz-shifter machine learning to dream.

Each frame shows:

  Top-left:    8x8 grid of fantasy samples drawn from the *current* generative
               net (top -> hidden -> visible)
  Top-right:   per-top-unit shift selectivity bar chart
  Mid:         layer-2 generative receptive fields, one per hidden unit
  Bottom:      IS-NLL trajectory and direction-recovery accuracy up to the
               current step
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

import helmholtz_shifter as hs
from _train_canonical import train_and_save


# ------------------------------------------------------------------
# Frame rendering helpers
# ------------------------------------------------------------------

def _draw_image_grid(ax, imgs: np.ndarray, n_rows: int, n_cols: int,
                     pad: int = 1) -> None:
    H, W = hs.H, hs.W
    canvas = np.full((n_rows * (H + pad) + pad, n_cols * (W + pad) + pad),
                     0.5, dtype=np.float32)
    for k in range(min(len(imgs), n_rows * n_cols)):
        r, c = divmod(k, n_cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = imgs[k].reshape(H, W)
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])


def _draw_layer2_strip(ax, W_hv: np.ndarray, b_v: np.ndarray) -> None:
    """One per-hidden-unit receptive field, side by side."""
    H, W = hs.H, hs.W
    n_h = W_hv.shape[0]
    pad = 1
    canvas = np.full((H + 2 * pad, n_h * (W + pad) + pad), 0.5,
                     dtype=np.float32)
    p_baseline = hs.sigmoid(b_v).reshape(H, W)
    max_abs = 1e-3
    fields = []
    for j in range(n_h):
        h_one = np.zeros((1, n_h), dtype=np.float32)
        h_one[0, j] = 1.0
        z_v = h_one @ W_hv + b_v
        f = hs.sigmoid(z_v).reshape(H, W) - p_baseline
        fields.append(f)
        max_abs = max(max_abs, abs(f).max())
    # paint deltas centred at 0.5 grey, mapped to [0, 1] via diverging scale
    for j, f in enumerate(fields):
        x0 = pad + j * (W + pad)
        normed = 0.5 + 0.5 * (f / max_abs)
        canvas[pad:pad + H, x0:x0 + W] = normed
    ax.imshow(canvas, cmap="seismic", vmin=0.0, vmax=1.0,
              interpolation="nearest", aspect="auto")
    ax.set_xticks([]); ax.set_yticks([])


def _selectivity_for(W_hv: np.ndarray, W_th: np.ndarray, b_v: np.ndarray,
                     b_h: np.ndarray, n_fantasy: int = 512,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """For each top unit k, sample fantasies under one-hot top=e_k and return
    P(right shift), P(left shift) signatures."""
    if rng is None:
        rng = np.random.default_rng(0)
    n_top = W_th.shape[0]
    p_right = np.zeros(n_top, dtype=np.float32)
    p_left = np.zeros(n_top, dtype=np.float32)
    for k in range(n_top):
        t = np.zeros((n_fantasy, n_top), dtype=np.float32)
        t[:, k] = 1.0
        p_h = hs.sigmoid(t @ W_th + b_h)
        h = (rng.random(p_h.shape) < p_h).astype(np.float32)
        p_v = hs.sigmoid(h @ W_hv + b_v)
        v = (rng.random(p_v.shape) < p_v).astype(np.float32)
        sig = hs._shift_signature(v)
        p_right[k] = float((sig == +1).mean())
        p_left[k] = float((sig == -1).mean())
    return p_right, p_left


# ------------------------------------------------------------------
# Render a single frame
# ------------------------------------------------------------------

def render_frame(step: int, nll: float, dir_acc: float,
                 fantasy: np.ndarray,
                 W_hv: np.ndarray, W_th: np.ndarray,
                 b_v: np.ndarray, b_h: np.ndarray,
                 history_steps: np.ndarray,
                 history_nll: np.ndarray,
                 history_dir: np.ndarray,
                 max_step: int,
                 max_nll: float,
                 rng: np.random.Generator) -> Image.Image:
    fig = plt.figure(figsize=(9.5, 7), dpi=100)
    gs = fig.add_gridspec(4, 2,
                          height_ratios=[3.5, 1.0, 1.5, 1.5],
                          hspace=0.45, wspace=0.20)

    # top-left: fantasies
    ax_f = fig.add_subplot(gs[0, 0])
    _draw_image_grid(ax_f, fantasy, n_rows=8, n_cols=8)
    ax_f.set_title("Fantasy samples  $v \\sim p_\\mathrm{model}$",
                   fontsize=10)

    # top-right: layer-3 selectivity bars
    ax_b = fig.add_subplot(gs[0, 1])
    p_right, p_left = _selectivity_for(W_hv, W_th, b_v, b_h, n_fantasy=512,
                                       rng=rng)
    n_top = len(p_right)
    x = np.arange(n_top)
    width = 0.35
    ax_b.bar(x - width / 2, p_right, width, label="right",
             color="#cc6633")
    ax_b.bar(x + width / 2, p_left, width, label="left",
             color="#3366cc")
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([f"$t_{k}$" for k in range(n_top)])
    ax_b.set_ylim(0, 1.05)
    ax_b.set_ylabel("P(shift signature | $t_k=1$)")
    ax_b.legend(fontsize=9, loc="upper right", ncol=2, frameon=True)
    ax_b.grid(alpha=0.3, axis="y")
    ax_b.set_title("Layer-3 unit shift-direction selectivity",
                   fontsize=10)

    # mid: layer-2 receptive fields strip
    ax_strip = fig.add_subplot(gs[1, :])
    _draw_layer2_strip(ax_strip, W_hv, b_v)
    ax_strip.set_title("Layer-2 generative receptive fields  "
                       "$p(v \\,|\\, h_j=1) - p(v \\,|\\, \\mathbf{0})$",
                       fontsize=9)

    # bottom-left: NLL curve
    ax_nll = fig.add_subplot(gs[2, :])
    if len(history_steps):
        ax_nll.plot(history_steps, history_nll, color="#9467bd",
                    linewidth=1.5)
    ax_nll.axvline(step, color="black", linewidth=1.0, alpha=0.4)
    ax_nll.set_xlim(0, max_step)
    ax_nll.set_ylim(min(8.0, history_nll.min() - 0.5)
                    if len(history_nll) else 0,
                    max_nll)
    ax_nll.set_xlabel("wake-sleep step")
    ax_nll.set_ylabel("IS-NLL  (bits)")
    ax_nll.grid(alpha=0.3)

    # bottom: direction recovery curve
    ax_dir = fig.add_subplot(gs[3, :])
    if len(history_steps):
        ax_dir.plot(history_steps, history_dir, color="#1f77b4",
                    linewidth=1.5)
    ax_dir.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
                   label="chance")
    ax_dir.axvline(step, color="black", linewidth=1.0, alpha=0.4)
    ax_dir.set_xlim(0, max_step)
    ax_dir.set_ylim(0.45, 1.02)
    ax_dir.set_xlabel("wake-sleep step")
    ax_dir.set_ylabel("dir recovery acc")
    ax_dir.legend(fontsize=8, loc="lower right")
    ax_dir.grid(alpha=0.3)

    fig.suptitle(f"Helmholtz-shifter machine — step {step:>10,d}    "
                 f"IS-NLL = {nll:.2f} bits    dir-acc = {dir_acc:.3f}",
                 fontsize=11, y=0.995)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-passes", type=int, default=1_500_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--snapshot-every", type=int, default=50_000)
    p.add_argument("--p-on", type=float, default=hs.P_ON)
    p.add_argument("--out", type=str, default="helmholtz_shifter.gif")
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
        W_th_seq = a["W_th"]
        b_v_seq = a["b_v"]
        b_h_seq = a["b_h"]
        fantasy_seq = a["fantasy_samples"]
        nll_seq = a["is_nll_bits"]
        dir_seq = a["dir_acc"]
    else:
        print(f"Training (seed={args.seed}, n_passes={args.n_passes}, "
              f"snapshot_every={args.snapshot_every})...")
        model, history, snap = train_and_save(
            args.seed, args.n_passes, args.lr, args.batch_size,
            eval_every=args.snapshot_every,
            snapshot_every=args.snapshot_every,
            outdir="viz", p_on=args.p_on)
        steps = np.array([s[0] for s in snap])
        W_hv_seq = np.array([s[1] for s in snap])
        W_th_seq = np.array([s[2] for s in snap])
        b_v_seq = np.array([s[3] for s in snap])
        b_h_seq = np.array([s[4] for s in snap])
        fantasy_seq = np.array([s[6] for s in snap])
        nll_seq = np.array([s[7] for s in snap])
        dir_seq = np.array([s[8] for s in snap])

    max_step = int(steps[-1])
    max_nll = float(max(nll_seq.max(), 12.0))
    rng = np.random.default_rng(0)
    frames = []
    for k in range(len(steps)):
        frame = render_frame(int(steps[k]),
                             float(nll_seq[k]), float(dir_seq[k]),
                             fantasy_seq[k],
                             W_hv_seq[k], W_th_seq[k],
                             b_v_seq[k], b_h_seq[k],
                             steps[:k + 1],
                             nll_seq[:k + 1],
                             dir_seq[:k + 1],
                             max_step, max_nll, rng)
        frames.append(frame)
        if (k + 1) % 5 == 0 or k == len(steps) - 1:
            print(f"  frame {k + 1}/{len(steps)}  step={int(steps[k])}  "
                  f"NLL={float(nll_seq[k]):.3f}  dir={float(dir_seq[k]):.3f}")

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
