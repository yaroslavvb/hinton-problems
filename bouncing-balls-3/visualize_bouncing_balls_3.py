"""Visualize a trained RTRBM on bouncing balls.

Outputs PNGs to ``viz/``:

  data_samples.png      grid of frames from the simulator
  receptive_fields.png  W columns reshaped to (h, w) -- each hidden unit's
                          incoming weight pattern (the "feature dictionary")
  recurrent_matrix.png  the recurrent W_h matrix, hidden-vs-hidden
  training_curves.png   recon-MSE per epoch + log scale
  reconstructions.png   data | teacher-forced reconstruction | hidden state
  rollout_grid.png      a single rollout: warmup ground truth vs predicted
                          continuation, side-by-side
"""
from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from bouncing_balls_3 import (
    build_rtrbm,
    forward_mean_field,
    free_rollout_mse,
    make_dataset,
    rollout,
    sigmoid,
    teacher_forced_recon,
    train,
)


def _ensure(d: str) -> None:
    os.makedirs(d, exist_ok=True)


def _grid(frames: np.ndarray, h: int, w: int, n_cols: int) -> np.ndarray:
    """Lay out (T, h*w) frames as a (rows*h, cols*w) image with 1px gutter."""
    T = frames.shape[0]
    n_rows = (T + n_cols - 1) // n_cols
    sep = 1
    gh = n_rows * h + (n_rows - 1) * sep
    gw = n_cols * w + (n_cols - 1) * sep
    grid = np.full((gh, gw), 0.5, dtype=np.float32)
    for k in range(T):
        r, c = divmod(k, n_cols)
        ys = r * (h + sep)
        xs = c * (w + sep)
        grid[ys:ys + h, xs:xs + w] = frames[k].reshape(h, w)
    return grid


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_data_samples(sequences: np.ndarray, h: int, w: int,
                      out_path: str, n_show: int = 16) -> None:
    """Show n_show consecutive frames from sequence 0."""
    frames = sequences[0, :n_show]
    grid = _grid(frames, h, w, n_cols=8)
    fig, ax = plt.subplots(figsize=(8, max(2, grid.shape[0] / 30)))
    ax.imshow(grid, cmap="gray", vmin=0, vmax=1)
    ax.set_title(f"Bouncing balls: first {n_show} frames of seq 0")
    ax.set_axis_off()
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_receptive_fields(model, h: int, w: int, out_path: str,
                          n_show: int = 64) -> None:
    """Show the first n_show incoming weight slices W[:, j] reshaped to (h, w).

    Hidden units in an RTRBM should learn motion-related receptive fields:
    blob detectors at specific positions.
    """
    n_show = min(n_show, model.n_hidden)
    n_cols = 8
    n_rows = (n_show + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows))
    if n_rows == 1:
        axes = np.array([axes])
    axes = axes.reshape(n_rows, n_cols)
    vmax = float(np.abs(model.W).max())
    for k in range(n_rows * n_cols):
        ax = axes[k // n_cols, k % n_cols]
        ax.set_axis_off()
        if k < n_show:
            rf = model.W[:, k].reshape(h, w)
            ax.imshow(rf, cmap="seismic", vmin=-vmax, vmax=vmax)
            ax.set_title(f"h{k}", fontsize=6)
    fig.suptitle(
        f"Receptive fields W[:, :{n_show}] (red=+, blue=-)",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_recurrent_matrix(model, out_path: str) -> None:
    """The recurrent W_h matrix (hidden_t x hidden_{t-1}). The defining
    structural piece of the RTRBM; should be informative once trained.
    """
    fig, ax = plt.subplots(figsize=(5, 4))
    vmax = float(np.abs(model.W_h).max())
    im = ax.imshow(model.W_h, cmap="seismic", vmin=-vmax, vmax=vmax,
                   aspect="auto")
    ax.set_title("Recurrent matrix W_h (rows=hidden_t, cols=hidden_{t-1})")
    ax.set_xlabel("hidden_{t-1}")
    ax.set_ylabel("hidden_t")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(8, 3))
    epochs = history["epoch"]
    axes[0].plot(epochs, history["recon_mse"], color="tab:blue")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("CD-1 reconstruction MSE")
    axes[0].set_title("Training reconstruction MSE")
    axes[0].grid(alpha=0.3)

    axes[1].semilogy(epochs, history["recon_mse"], color="tab:blue")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("CD-1 reconstruction MSE (log)")
    axes[1].set_title("Same, log scale")
    axes[1].grid(alpha=0.3, which="both")
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(model, sequence: np.ndarray, h: int, w: int,
                         out_path: str, n_show: int = 12) -> None:
    """3-row: data | teacher-forced recon | hidden activations. Validates
    that the RBM piece works (reconstruction) and the hidden code is sparse.
    """
    n_show = min(n_show, sequence.shape[0])
    r = forward_mean_field(model, sequence)
    v_recon = sigmoid(r @ model.W.T + model.b_v)

    fig, axes = plt.subplots(3, n_show, figsize=(n_show, 3.4))
    for k in range(n_show):
        axes[0, k].imshow(sequence[k].reshape(h, w),
                          cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_axis_off()
        axes[1, k].imshow(v_recon[k].reshape(h, w),
                          cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_axis_off()
        axes[2, k].imshow(r[k].reshape(1, -1),
                          cmap="viridis", vmin=0, vmax=1, aspect="auto")
        axes[2, k].set_axis_off()
    axes[0, 0].set_ylabel("data")
    axes[1, 0].set_ylabel("recon")
    axes[2, 0].set_ylabel("hidden")
    fig.suptitle("data | teacher-forced reconstruction | mean hidden activation",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def plot_rollout_grid(model, sequence: np.ndarray, h: int, w: int,
                      warmup: int, n_future: int,
                      out_path: str, n_gibbs: int = 25) -> None:
    """Single rollout. Top row: ground truth. Bottom row: warmup +
    free-running prediction. Color-codes warmup vs predicted segments.
    """
    pred = rollout(model, sequence[:warmup], n_future=n_future,
                   n_gibbs=n_gibbs, sample_visible=False, sample_hidden=True)
    truth = sequence[:warmup + n_future]

    n_show = min(warmup + n_future, sequence.shape[0])
    fig, axes = plt.subplots(2, n_show, figsize=(n_show * 0.8, 2.0))
    for k in range(n_show):
        axes[0, k].imshow(truth[k].reshape(h, w),
                          cmap="gray", vmin=0, vmax=1)
        axes[0, k].set_axis_off()
        axes[1, k].imshow(pred[k].reshape(h, w),
                          cmap="gray", vmin=0, vmax=1)
        axes[1, k].set_axis_off()
        if k < warmup:
            axes[1, k].set_title("warmup", fontsize=6, color="blue")
        else:
            axes[1, k].set_title(f"+{k - warmup + 1}",
                                 fontsize=6, color="darkred")
    axes[0, 0].set_title("truth", fontsize=8)
    fig.suptitle(
        f"Top: ground truth | Bottom: rollout (warmup={warmup} frames, "
        f"then free for {n_future} steps, {n_gibbs} Gibbs steps each)",
        fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Visualize bouncing-balls-3 RTRBM")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h", type=int, default=30)
    p.add_argument("--w", type=int, default=30)
    p.add_argument("--n-balls", type=int, default=3)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--n-sequences", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--rollout-warmup", type=int, default=10)
    p.add_argument("--rollout-future", type=int, default=30)
    p.add_argument("--n-gibbs", type=int, default=25)
    p.add_argument("--outdir", type=str, default="viz")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    _ensure(args.outdir)

    n_visible = args.h * args.w
    print(f"# generating {args.n_sequences} sequences of length "
          f"{args.seq_len} ({args.h}x{args.w})")
    sequences = make_dataset(n_sequences=args.n_sequences,
                             seq_len=args.seq_len,
                             n_balls=args.n_balls,
                             h=args.h, w=args.w,
                             radius=args.radius, speed=args.speed,
                             seed=args.seed)
    plot_data_samples(sequences, args.h, args.w,
                      os.path.join(args.outdir, "data_samples.png"))

    model = build_rtrbm(n_visible=n_visible, n_hidden=args.n_hidden,
                        seed=args.seed)
    data_mean = np.clip(sequences.reshape(-1, n_visible).mean(axis=0),
                        1e-3, 1 - 1e-3)
    model.b_v[:] = np.log(data_mean / (1.0 - data_mean)).astype(np.float32)

    print(f"# training: n_hidden={args.n_hidden} n_epochs={args.n_epochs}")
    t0 = time.time()
    history = train(model, sequences,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    verbose=True)
    train_time = time.time() - t0

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_receptive_fields(model, args.h, args.w,
                          os.path.join(args.outdir, "receptive_fields.png"))
    plot_recurrent_matrix(model, os.path.join(args.outdir, "recurrent_matrix.png"))

    val = make_dataset(n_sequences=1, seq_len=args.seq_len,
                       n_balls=args.n_balls,
                       h=args.h, w=args.w, radius=args.radius,
                       speed=args.speed,
                       seed=args.seed + 9999)[0]
    tf = teacher_forced_recon(model, val)
    free = free_rollout_mse(model, val,
                            warmup=args.rollout_warmup,
                            n_future=min(args.rollout_future,
                                         args.seq_len - args.rollout_warmup),
                            n_gibbs=args.n_gibbs)
    plot_reconstructions(model, val, args.h, args.w,
                         os.path.join(args.outdir, "reconstructions.png"))
    plot_rollout_grid(model, val, args.h, args.w,
                      warmup=args.rollout_warmup,
                      n_future=min(args.rollout_future,
                                   args.seq_len - args.rollout_warmup),
                      out_path=os.path.join(args.outdir, "rollout_grid.png"),
                      n_gibbs=args.n_gibbs)

    print()
    print(f"# train time: {train_time:.2f}s")
    print(f"# final CD-1 recon MSE: {history['recon_mse'][-1]:.4f}")
    print(f"# val teacher-forced MSE: {tf:.4f}")
    print(f"# val free-rollout MSE:   {free:.4f}")
    print(f"# wrote PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
