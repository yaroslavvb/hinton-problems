"""
Static visualizations for the trained bars-RBM.

Outputs (in `viz/`):
  receptive_fields.png  - each hidden unit's W column reshaped to a 4x4 image
  training_curves.png   - reconstruction MSE, mean purity, bars covered
  data_samples.png      - 16 random training samples (so the reader can see
                          the bars-on-bars OR mixture)
  reconstructions.png   - data | one-step reconstruction | hidden code
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from bars_rbm import (BarsRBM, train, generate_bars,
                      per_unit_bar_purity, make_bar_templates)


def plot_receptive_fields(rbm: BarsRBM, out_path: str):
    """One subplot per hidden unit, showing its incoming weights as an image.

    The bar-purity score and the best-matching bar label are written above
    each subplot.
    """
    score = per_unit_bar_purity(rbm)
    h, w = rbm.image_shape
    n = rbm.n_hidden

    cols = min(8, n)
    rows = (n + cols - 1) // cols

    bar_names = ([f"H{i}" for i in range(h)] + [f"V{j}" for j in range(w)])

    fig, axes = plt.subplots(rows, cols,
                             figsize=(cols * 1.5, rows * 1.7),
                             dpi=140)
    axes = np.atleast_2d(axes)

    max_abs = max(float(np.abs(rbm.W).max()), 1e-3)

    for j in range(rows * cols):
        ax = axes[j // cols, j % cols]
        if j < n:
            rf = rbm.W[:, j].reshape(h, w)
            ax.imshow(rf, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs,
                      interpolation="nearest")
            best = bar_names[score["best_bar"][j]]
            ax.set_title(f"u{j}: {best}\np={score['purity'][j]:.2f}",
                         fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])

    fig.suptitle(
        f"Bars-RBM receptive fields  "
        f"(n_hidden={n}, {score['bars_covered']}/{score['n_bars']} bars covered, "
        f"mean purity {score['mean_purity']:.2f})",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history: dict, n_bars: int, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.2), dpi=140)
    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["recon_error"], color="#9467bd", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE")
    ax.set_title("Reconstruction MSE")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, history["mean_purity"], color="#1f77b4", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean cosine sim with best bar")
    ax.set_ylim(0, 1.05)
    ax.set_title("Mean per-unit bar purity")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["bars_covered"], color="#2ca02c", linewidth=1.5)
    ax.axhline(n_bars, color="gray", linestyle="--", linewidth=0.7,
               label=f"all {n_bars} bars")
    ax.set_xlabel("epoch")
    ax.set_ylabel("# distinct bars detected")
    ax.set_ylim(-0.5, n_bars + 0.5)
    ax.set_title("Bars covered (purity ≥ 0.5)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("bars-RBM training curves", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_data_samples(images: np.ndarray, h: int, w: int, out_path: str,
                      n_show: int = 16):
    fig, axes = plt.subplots(2, 8, figsize=(8, 2.5), dpi=140)
    for i, ax in enumerate(axes.flat):
        if i < n_show:
            ax.imshow(images[i].reshape(h, w), cmap="gray_r",
                      vmin=0, vmax=1, interpolation="nearest")
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Training samples (OR of bars, p_bar = 0.125)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(rbm: BarsRBM, data: np.ndarray, out_path: str,
                         n_show: int = 8):
    h, w = rbm.image_shape
    pick = data[:n_show]
    h_prob = rbm.hidden_prob(pick)
    v_recon = rbm.visible_prob(h_prob)

    fig, axes = plt.subplots(3, n_show, figsize=(n_show * 1.2, 3.6), dpi=140)
    for i in range(n_show):
        axes[0, i].imshow(pick[i].reshape(h, w), cmap="gray_r",
                          vmin=0, vmax=1, interpolation="nearest")
        axes[1, i].imshow(v_recon[i].reshape(h, w), cmap="gray_r",
                          vmin=0, vmax=1, interpolation="nearest")
        # Hidden code as a row of n_hidden squares
        axes[2, i].imshow(h_prob[i:i+1], cmap="gray_r",
                          vmin=0, vmax=1, interpolation="nearest",
                          aspect="auto")
        for ax in (axes[0, i], axes[1, i], axes[2, i]):
            ax.set_xticks([])
            ax.set_yticks([])

    axes[0, 0].set_ylabel("data", fontsize=9)
    axes[1, 0].set_ylabel("recon", fontsize=9)
    axes[2, 0].set_ylabel("hidden", fontsize=9)
    fig.suptitle("Data, one-step CD reconstruction, and hidden activations",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_bar_templates(h: int, w: int, out_path: str):
    """Reference plot: the 8 single-bar templates the RBM is trying to find."""
    templates = make_bar_templates(h, w)
    n_bars = h + w
    fig, axes = plt.subplots(2, max(h, w),
                             figsize=(max(h, w) * 1.0, 2.4), dpi=140)
    for j in range(2 * max(h, w)):
        ax = axes[j // max(h, w), j % max(h, w)]
        if j < n_bars:
            ax.imshow(templates[j], cmap="gray_r", vmin=0, vmax=1,
                      interpolation="nearest")
            label = f"H{j}" if j < h else f"V{j - h}"
            ax.set_title(label, fontsize=9)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Single-bar templates (reference)", fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-hidden", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=300)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--p-bar", type=float, default=0.125)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_epochs} epochs (seed={args.seed}, "
          f"n_hidden={args.n_hidden})...")
    rbm, history = train(n_epochs=args.n_epochs,
                         n_hidden=args.n_hidden,
                         n_train=args.n_train,
                         batch_size=args.batch_size,
                         lr=args.lr,
                         p_bar=args.p_bar,
                         seed=args.seed,
                         verbose=False)
    score = per_unit_bar_purity(rbm)
    print(f"  final recon MSE = {history['recon_error'][-1]:.4f}")
    print(f"  bars covered = {score['bars_covered']}/{score['n_bars']}")
    print(f"  mean purity = {score['mean_purity']:.3f}")

    plot_bar_templates(4, 4,
                       os.path.join(args.outdir, "bar_templates.png"))
    plot_data_samples(generate_bars(16, 4, 4, args.p_bar,
                                    rng=np.random.default_rng(args.seed + 999)),
                      4, 4,
                      os.path.join(args.outdir, "data_samples.png"))
    plot_receptive_fields(rbm,
                          os.path.join(args.outdir, "receptive_fields.png"))
    plot_training_curves(history, n_bars=score["n_bars"],
                         out_path=os.path.join(args.outdir,
                                               "training_curves.png"))
    plot_reconstructions(rbm,
                         generate_bars(8, 4, 4, args.p_bar,
                                       rng=np.random.default_rng(args.seed + 7)),
                         os.path.join(args.outdir, "reconstructions.png"))


if __name__ == "__main__":
    main()
