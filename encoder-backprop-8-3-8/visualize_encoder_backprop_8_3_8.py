"""
Static visualizations for the trained 8-3-8 backprop encoder.

Outputs (in `viz/`):
  training_curves.png  - loss, accuracy, distinct codes, weight norm
  weights.png          - W1 (input->hidden) and W2 (hidden->output) heatmaps
  hidden_codes_3cube.png - 3-D scatter of hidden activations on the 3-cube
  code_table.png       - 8x3 heatmap of raw + binarized hidden codes
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from encoder_backprop_8_3_8 import (
    EncoderMLP, train, make_encoder_data, hidden_code_table, n_distinct_codes,
)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#e377c2", "#17becf"]


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=120)
    epochs = history["epoch"]

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="#9467bd")
    ax.set_yscale("log")
    ax.set_ylabel("cross-entropy loss")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Loss (log scale)")

    ax = axes[0, 1]
    ax.plot(epochs, np.array(history["acc"]) * 100, color="#1f77b4")
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction accuracy")

    ax = axes[1, 0]
    ax.plot(epochs, history["n_distinct_codes"], color="#2ca02c")
    ax.set_ylabel("# distinct binarized codes")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 8.5)
    ax.set_yticks(range(0, 9))
    ax.grid(alpha=0.3)
    ax.set_title("Distinct hidden codes (binarized at 0.5)")

    ax = axes[1, 1]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    ax.set_ylabel(r"$\|W_1\|_F + \|W_2\|_F$")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Weight norm")

    fig.suptitle(f"8-3-8 backprop encoder (epochs run = {len(epochs)})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(model: EncoderMLP, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4.5), dpi=120)

    W1 = model.W1
    vmax = max(abs(W1).max(), 1e-3)
    im0 = axes[0].imshow(W1, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    axes[0].set_title(r"$W_1$ (input $\to$ hidden)")
    axes[0].set_xlabel("hidden unit")
    axes[0].set_ylabel("input pattern (one-hot bit)")
    axes[0].set_xticks(range(model.n_hidden))
    axes[0].set_yticks(range(model.n_in))
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    W2 = model.W2
    vmax = max(abs(W2).max(), 1e-3)
    im1 = axes[1].imshow(W2, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                         aspect="auto")
    axes[1].set_title(r"$W_2$ (hidden $\to$ output)")
    axes[1].set_xlabel("output unit")
    axes[1].set_ylabel("hidden unit")
    axes[1].set_xticks(range(model.n_out))
    axes[1].set_yticks(range(model.n_hidden))
    fig.colorbar(im1, ax=axes[1], shrink=0.8)

    fig.suptitle(f"Final weights  ($\\|W_1\\|+\\|W_2\\|$ = "
                 f"{np.linalg.norm(W1)+np.linalg.norm(W2):.2f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_codes_3cube(model: EncoderMLP, out_path: str):
    """3-D scatter of hidden activations relative to the corners of the 3-cube."""
    data = make_encoder_data()
    codes = hidden_code_table(model, data)

    fig = plt.figure(figsize=(6, 5.5), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    # Draw the unit cube edges.
    cube_edges = [
        # bottom
        [(0, 0, 0), (1, 0, 0)],
        [(1, 0, 0), (1, 1, 0)],
        [(1, 1, 0), (0, 1, 0)],
        [(0, 1, 0), (0, 0, 0)],
        # top
        [(0, 0, 1), (1, 0, 1)],
        [(1, 0, 1), (1, 1, 1)],
        [(1, 1, 1), (0, 1, 1)],
        [(0, 1, 1), (0, 0, 1)],
        # verticals
        [(0, 0, 0), (0, 0, 1)],
        [(1, 0, 0), (1, 0, 1)],
        [(1, 1, 0), (1, 1, 1)],
        [(0, 1, 0), (0, 1, 1)],
    ]
    for a, b in cube_edges:
        ax.plot(*zip(a, b), color="lightgray", linewidth=0.8, zorder=1)

    # Mark corners.
    for x in (0, 1):
        for y in (0, 1):
            for z in (0, 1):
                ax.scatter([x], [y], [z], color="lightgray", s=40,
                           depthshade=False, zorder=2)

    # Plot codes.
    for i in range(8):
        ax.scatter(codes[i, 0], codes[i, 1], codes[i, 2],
                   color=PATTERN_COLORS[i], s=140, depthshade=False, zorder=5,
                   edgecolor="black", linewidth=0.6)
        ax.text(codes[i, 0], codes[i, 1], codes[i, 2] + 0.05, str(i),
                color="black", fontsize=10, ha="center", weight="bold",
                zorder=6)

    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.05, 1.05)
    ax.set_zlim(-0.05, 1.05)
    ax.set_xlabel(r"$h_0$")
    ax.set_ylabel(r"$h_1$")
    ax.set_zlabel(r"$h_2$")
    n_codes = n_distinct_codes(model, data)
    ax.set_title(f"Hidden codes on the 3-cube  ({n_codes}/8 distinct binarized)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_code_table(model: EncoderMLP, out_path: str):
    """Side-by-side heatmaps of raw + binarized hidden codes."""
    data = make_encoder_data()
    codes = hidden_code_table(model, data)
    binary = (codes > 0.5).astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(7, 4.5), dpi=140)

    im0 = axes[0].imshow(codes, cmap="viridis", vmin=0, vmax=1, aspect="auto")
    axes[0].set_title("Raw hidden activations")
    axes[0].set_xlabel("hidden unit")
    axes[0].set_ylabel("input pattern")
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_yticks(range(8))
    for i in range(8):
        for j in range(3):
            axes[0].text(j, i, f"{codes[i, j]:.2f}", ha="center",
                         va="center", fontsize=9,
                         color="white" if codes[i, j] < 0.5 else "black")
    fig.colorbar(im0, ax=axes[0], shrink=0.8)

    axes[1].imshow(binary, cmap="gray_r", vmin=0, vmax=1, aspect="auto")
    axes[1].set_title("Binarized at 0.5")
    axes[1].set_xlabel("hidden unit")
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_yticks(range(8))
    for i in range(8):
        for j in range(3):
            axes[1].text(j, i, str(int(binary[i, j])), ha="center",
                         va="center", fontsize=11,
                         color="white" if binary[i, j] == 1 else "black",
                         weight="bold")

    n_codes = n_distinct_codes(model, data)
    fig.suptitle(f"Hidden code table  ({n_codes}/8 distinct binarized)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.epochs} epochs (seed={args.seed})...")
    model, history = train(n_epochs=args.epochs, seed=args.seed, verbose=False)
    print(f"  final accuracy: {history['acc'][-1]*100:.0f}%   "
          f"distinct codes: {history['n_distinct_codes'][-1]}/8   "
          f"epochs: {len(history['epoch'])}")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_hidden_codes_3cube(model, os.path.join(args.outdir, "hidden_codes_3cube.png"))
    plot_code_table(model, os.path.join(args.outdir, "code_table.png"))


if __name__ == "__main__":
    main()
