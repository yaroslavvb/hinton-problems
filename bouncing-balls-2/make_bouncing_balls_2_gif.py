"""
Render `bouncing_balls_2.gif` — input video vs TRBM rollout, side by side.

Layout per frame:
  [ ground truth | TRBM rollout ]   T_seed seed frames followed by N future
  with the seed frames highlighted (orange border) and a label switching from
  "SEED (clamped)" to "ROLLOUT (predicted)" once the model takes over.
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from bouncing_balls_2 import (build_trbm, train, make_dataset)


def render_frame(seed_frames: np.ndarray,
                 truth_frames: np.ndarray,
                 pred_frames: np.ndarray,
                 t: int,
                 n_seed: int,
                 h: int, w: int) -> Image.Image:
    """Render one frame of the side-by-side comparison.

    `t` runs 0..(n_seed + n_future - 1). For t < n_seed both panels show the
    seed frame; for t >= n_seed left = truth, right = predicted rollout.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(5.0, 2.6), dpi=120)

    if t < n_seed:
        left = seed_frames[t]
        right = seed_frames[t]
        label = f"SEED frame  t={t}"
        border_color = "#ff9800"
    else:
        idx = t - n_seed
        left = truth_frames[idx]
        right = pred_frames[idx]
        label = f"ROLLOUT step  t={t}  (seed length {n_seed})"
        border_color = "#1f77b4"

    ax1.imshow(left.reshape(h, w), cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax1.set_title("ground truth", fontsize=9)
    ax1.set_xticks([]); ax1.set_yticks([])

    ax2.imshow(right.reshape(h, w), cmap="gray", vmin=0, vmax=1,
               interpolation="nearest")
    ax2.set_title("TRBM rollout", fontsize=9)
    ax2.set_xticks([]); ax2.set_yticks([])

    for ax in (ax1, ax2):
        for spine in ax.spines.values():
            spine.set_color(border_color)
            spine.set_linewidth(2)

    fig.suptitle(label, fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--w", type=int, default=16)
    p.add_argument("--n-balls", type=int, default=2)
    p.add_argument("--n-sequences", type=int, default=60)
    p.add_argument("--seq-len", type=int, default=50)
    p.add_argument("--n-hidden", type=int, default=200)
    p.add_argument("--n-lag", type=int, default=2)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--n-seed-frames", type=int, default=10)
    p.add_argument("--n-future", type=int, default=20)
    p.add_argument("--feedback", type=str, default="sample")
    p.add_argument("--out", type=str, default="bouncing_balls_2.gif")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--hold-final", type=int, default=8)
    args = p.parse_args()

    np.random.seed(args.seed)
    print(f"Building dataset ({args.n_sequences} seq, T={args.seq_len}, "
          f"{args.h}x{args.w})...")
    seqs = make_dataset(args.n_sequences, args.seq_len,
                        n_balls=args.n_balls, h=args.h, w=args.w,
                        seed=args.seed)

    print(f"Training TRBM ({args.n_epochs} epochs)...")
    model = build_trbm(args.h * args.w, args.n_hidden, n_lag=args.n_lag,
                       seed=args.seed)
    train(model, seqs, n_epochs=args.n_epochs, lr=args.lr,
          batch_size=10, verbose=False)

    print("Generating held-out test sequence + rollout...")
    test_seq = make_dataset(1, args.n_seed_frames + args.n_future,
                            n_balls=args.n_balls, h=args.h, w=args.w,
                            seed=args.seed + 9999)[0]
    seed_frames = test_seq[:args.n_seed_frames]
    truth = test_seq[args.n_seed_frames:]
    pred = model.rollout(seed_frames, args.n_future, k_gibbs=5,
                         feedback=args.feedback)

    print(f"Rendering frames...")
    n_total = args.n_seed_frames + args.n_future
    frames = []
    for t in range(n_total):
        frames.append(render_frame(seed_frames, truth, pred, t,
                                    args.n_seed_frames, args.h, args.w))
        if t % 5 == 0:
            print(f"  rendered frame {t+1}/{n_total}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
