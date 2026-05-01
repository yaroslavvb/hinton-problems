"""
Static visualizations for the trained 4-2-4 encoder.

Outputs (in `viz/`):
  training_curves.png  - accuracy, code separation, weight norm, recon error
  weights.png          - final weight matrix as a Hinton diagram
  hidden_codes.png     - final hidden codes for the 4 patterns
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from encoder_4_2_4 import (EncoderRBM, train, make_encoder_data)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=120)

    epochs = history["epoch"]
    pert = history.get("perturbations", [])

    ax = axes[0, 0]
    ax.plot(epochs, np.array(history["acc"]) * 100, color="#1f77b4")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction accuracy")

    ax = axes[0, 1]
    ax.plot(epochs, history["code_separation"], color="#2ca02c")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("mean pairwise L2 distance")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Hidden-code separation")

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Weight norm")

    ax = axes[1, 1]
    ax.plot(epochs, history["reconstruction_error"], color="#9467bd")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_ylabel("MSE on V2")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction MSE")

    if pert:
        fig.suptitle(f"4-2-4 encoder, CD-5 — restarts at epochs {pert}",
                     fontsize=11)
    else:
        fig.suptitle("4-2-4 encoder, CD-5", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(rbm: EncoderRBM, out_path: str):
    fig, ax = plt.subplots(figsize=(3.5, 6), dpi=140)
    W = rbm.W
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(8):
        for j in range(2):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.4))
    ax.set_xlim(-0.7, 1.7)
    ax.set_ylim(-0.7, 7.7)
    ax.invert_yaxis()
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["H[0]", "H[1]"])
    ax.set_yticks(range(8))
    ax.set_yticklabels([f"V1[{i}]" for i in range(4)] +
                       [f"V2[{i}]" for i in range(4)])
    ax.axhline(3.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(f"Final weights  ($\\|W\\|_F$ = {np.linalg.norm(W):.2f})")
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_codes(rbm: EncoderRBM, out_path: str):
    data = make_encoder_data()
    codes = np.array([rbm.hidden_code(data[i, :4], n_steps=60) for i in range(4)])
    fig, ax = plt.subplots(figsize=(5, 5), dpi=140)
    for corner in [(0, 0), (0, 1), (1, 0), (1, 1)]:
        ax.plot(*corner, marker="s", markersize=24, markerfacecolor="none",
                markeredgecolor="lightgray", linestyle="None")
    for i in range(4):
        ax.plot(codes[i, 0], codes[i, 1], "o", color=PATTERN_COLORS[i],
                markersize=18, zorder=3)
        ax.annotate(str(i), (codes[i, 0], codes[i, 1]),
                    textcoords="offset points", xytext=(0, 0),
                    ha="center", va="center",
                    fontsize=11, color="white", weight="bold")
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_xlabel(r"$\langle H_0 \rangle$")
    ax.set_ylabel(r"$\langle H_1 \rangle$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Hidden codes for the 4 patterns")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.epochs} epochs (seed={args.seed})...")
    rbm, history = train(n_epochs=args.epochs, seed=args.seed,
                         perturb_after=80, verbose=False)
    print(f"  final accuracy: {history['acc'][-1]*100:.0f}%   "
          f"restarts: {history['perturbations']}")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(rbm, os.path.join(args.outdir, "weights.png"))
    plot_hidden_codes(rbm, os.path.join(args.outdir, "hidden_codes.png"))


if __name__ == "__main__":
    main()
