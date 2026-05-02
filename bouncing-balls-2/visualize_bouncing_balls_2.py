"""
Static visualizations for the trained bouncing-balls TRBM.

Outputs (in `viz/`):
  example_frames.png    - 8 frames of the input video
  rbm_filters.png       - selected hidden-unit receptive fields (W columns reshaped)
  training_curves.png   - reconstruction MSE + weight norm over epochs
  rollout_compare.png   - ground truth vs rollout, side-by-side, 20 steps
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from bouncing_balls_2 import (
    build_trbm, train, make_dataset, evaluate_rollout, simulate_balls)


def plot_example_frames(seq: np.ndarray, h: int, w: int, out_path: str,
                         n_show: int = 8):
    """seq: (T, V). Renders n_show evenly-spaced frames in one row."""
    T = seq.shape[0]
    idx = np.linspace(0, T - 1, n_show).astype(int)
    fig, axes = plt.subplots(1, n_show, figsize=(n_show * 1.4, 1.6), dpi=120)
    for ax, i in zip(axes, idx):
        ax.imshow(seq[i].reshape(h, w), cmap="gray", vmin=0, vmax=1,
                  interpolation="nearest")
        ax.set_title(f"t={i}", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Bouncing balls — example input frames", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_filters(model, h: int, w: int, out_path: str, n_show: int = 25):
    """Visualise W's columns (hidden-unit visible receptive fields).

    Pick the n_show hidden units with the largest L2 norm so we are looking
    at the most informative filters.
    """
    W = model.W                                          # (V, H)
    norms = np.linalg.norm(W, axis=0)
    order = np.argsort(-norms)[:n_show]

    n_cols = int(np.ceil(np.sqrt(n_show)))
    n_rows = int(np.ceil(n_show / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols * 1.1, n_rows * 1.1),
                             dpi=140)
    if n_rows == 1 and n_cols == 1:
        axes_flat = [axes]
    else:
        axes_flat = axes.ravel()

    vmax = float(np.abs(W[:, order]).max())
    for ax, j in zip(axes_flat, order):
        ax.imshow(W[:, j].reshape(h, w), cmap="seismic", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"h={j}", fontsize=7)
    for k in range(len(order), len(axes_flat)):
        axes_flat[k].axis("off")
    fig.suptitle(f"Top {n_show} W filters by L2 norm "
                 f"(visible-side weights of each hidden unit)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2), dpi=120)
    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["recon_mse"], color="#1f77b4")
    ax.set_ylabel("CD-1 reconstruction MSE")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Per-frame reconstruction error")

    ax = axes[1]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Weight norm")

    fig.suptitle("TRBM training curves", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rollout_compare(seed_frames: np.ndarray,
                          truth_frames: np.ndarray,
                          predicted_frames: np.ndarray,
                          h: int, w: int, out_path: str,
                          n_show: int = 12):
    """Side-by-side: top row = seed + truth, bottom row = seed + rollout."""
    n_seed = seed_frames.shape[0]
    n_show_truth = min(n_show, truth_frames.shape[0])

    n_total = n_seed + n_show_truth
    fig, axes = plt.subplots(2, n_total, figsize=(n_total * 1.0, 2.4), dpi=140)

    for j in range(n_seed):
        for row in range(2):
            ax = axes[row, j]
            ax.imshow(seed_frames[j].reshape(h, w), cmap="gray",
                      vmin=0, vmax=1, interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_color("orange"); spine.set_linewidth(1.2)
            if row == 0:
                ax.set_title(f"s{j}", fontsize=7)
    for j in range(n_show_truth):
        col = n_seed + j
        ax = axes[0, col]
        ax.imshow(truth_frames[j].reshape(h, w), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"t{j}", fontsize=7)
        ax = axes[1, col]
        ax.imshow(predicted_frames[j].reshape(h, w), cmap="gray",
                  vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([]); ax.set_yticks([])

    axes[0, 0].set_ylabel("ground truth", fontsize=8)
    axes[1, 0].set_ylabel("TRBM rollout", fontsize=8)
    fig.suptitle(f"Bouncing balls — seed (orange) + ground truth vs rollout",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


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
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    print(f"Generating training data ({args.n_sequences} sequences, "
          f"T={args.seq_len}, {args.h}x{args.w})...")
    seqs = make_dataset(args.n_sequences, args.seq_len,
                        n_balls=args.n_balls, h=args.h, w=args.w,
                        seed=args.seed)
    print(f"Building TRBM (V={args.h * args.w}, H={args.n_hidden}, "
          f"n_lag={args.n_lag})...")
    model = build_trbm(args.h * args.w, args.n_hidden, n_lag=args.n_lag,
                       seed=args.seed)
    print(f"Training {args.n_epochs} epochs...")
    history = train(model, seqs, n_epochs=args.n_epochs, lr=args.lr,
                    batch_size=10, verbose=False)
    print(f"  final recon MSE: {history['recon_mse'][-1]:.5f}")

    print("Generating held-out test sequence...")
    test_seq = make_dataset(1, args.n_seed_frames + args.n_future,
                            n_balls=args.n_balls, h=args.h, w=args.w,
                            seed=args.seed + 9999)[0]
    seed_frames = test_seq[:args.n_seed_frames]
    truth = test_seq[args.n_seed_frames:]
    pred = model.rollout(seed_frames, args.n_future, k_gibbs=5,
                         feedback=args.feedback)

    plot_example_frames(seqs[0], args.h, args.w,
                        os.path.join(args.outdir, "example_frames.png"))
    plot_filters(model, args.h, args.w,
                 os.path.join(args.outdir, "rbm_filters.png"))
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_rollout_compare(seed_frames, truth, pred, args.h, args.w,
                         os.path.join(args.outdir, "rollout_compare.png"))

    rollout_mse = float(np.mean((pred - truth) ** 2))
    mean_frame = seqs.reshape(-1, args.h * args.w).mean(axis=0)
    mean_baseline = float(np.mean((mean_frame[None, :] - truth) ** 2))
    print(f"\nFinal rollout MSE: {rollout_mse:.5f}   "
          f"(mean-frame baseline: {mean_baseline:.5f})")


if __name__ == "__main__":
    main()
