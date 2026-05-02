"""
Static visualizations for the trained 4-3-4 over-complete encoder.

Outputs (in `viz/`):
  training_curves.png     - accuracy, code separation, weight norm, recon error,
                            min Hamming distance over training
  weights.png             - final weight matrix as a Hinton diagram (8 vis x 3 hid)
  three_cube.png          - 3-cube viz: 8 corners with the 4 chosen codes highlighted
  hamming_matrix.png      - 4x4 pairwise Hamming-distance matrix as a heatmap
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (registers 3D projection)

from encoder_4_3_4 import (EncoderRBM, train, make_encoder_data,
                           hamming_distances_between_codes,
                           dominant_codes, is_error_correcting)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PATTERN_LABELS = ["0", "1", "2", "3"]


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=120)

    epochs = history["epoch"]
    pert = history.get("perturbations", [])

    def _draw_perts(ax):
        for pe in pert:
            ax.axvline(pe, color="red", linestyle="--",
                       linewidth=0.8, alpha=0.6)

    ax = axes[0, 0]
    ax.plot(epochs, np.array(history["acc"]) * 100, color="#1f77b4")
    _draw_perts(ax)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction accuracy")

    ax = axes[0, 1]
    ax.plot(epochs, history["code_separation"], color="#2ca02c")
    _draw_perts(ax)
    ax.set_ylabel("mean pairwise L2 distance")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Hidden-code separation (continuous)")

    ax = axes[0, 2]
    ax.plot(epochs, history["min_hamming"], color="#9467bd",
            drawstyle="steps-post")
    _draw_perts(ax)
    ax.axhline(2, color="black", linestyle=":", linewidth=0.8,
               label="EC threshold (>=2)")
    ax.set_ylabel("min off-diag Hamming")
    ax.set_xlabel("epoch")
    ax.set_ylim(-0.2, 3.2)
    ax.set_yticks([0, 1, 2, 3])
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)
    ax.set_title("Headline metric: min Hamming")

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    _draw_perts(ax)
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Weight norm")

    ax = axes[1, 1]
    ax.plot(epochs, history["reconstruction_error"], color="#d62728")
    _draw_perts(ax)
    ax.set_ylabel("MSE on V2")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction MSE")

    ax = axes[1, 2]
    ax.plot(epochs, history["n_distinct_codes"], color="#17becf",
            drawstyle="steps-post")
    _draw_perts(ax)
    ax.set_ylabel("distinct dominant codes")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 4.3)
    ax.set_yticks([1, 2, 3, 4])
    ax.grid(alpha=0.3)
    ax.set_title("Distinct codes (need 4)")

    if pert:
        fig.suptitle(f"4-3-4 over-complete encoder, CD-3 -- "
                     f"restarts at epochs {pert}", fontsize=11)
    else:
        fig.suptitle("4-3-4 over-complete encoder, CD-3", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(rbm: EncoderRBM, out_path: str):
    fig, ax = plt.subplots(figsize=(4.0, 6), dpi=140)
    W = rbm.W
    n_h = W.shape[1]
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(8):
        for j in range(n_h):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.4))
    ax.set_xlim(-0.7, n_h - 0.3)
    ax.set_ylim(-0.7, 7.7)
    ax.invert_yaxis()
    ax.set_xticks(range(n_h))
    ax.set_xticklabels([f"H[{j}]" for j in range(n_h)])
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


def plot_three_cube(rbm: EncoderRBM, out_path: str):
    """Render the 3-cube with the 4 chosen corners highlighted."""
    data = make_encoder_data()
    codes = dominant_codes(rbm, data)
    chosen = {tuple(int(x) for x in c) for c in codes}
    ec = is_error_correcting(rbm, data)

    # all 8 corners
    corners = [(x, y, z) for x in (0, 1) for y in (0, 1) for z in (0, 1)]

    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    # Edges of the cube (Hamming-1 pairs)
    for a in corners:
        for b in corners:
            if sum(int(ax_ != bx_) for ax_, bx_ in zip(a, b)) == 1 and a < b:
                # color edge red if it lies between two CHOSEN codes
                a_in = a in chosen
                b_in = b in chosen
                bad = a_in and b_in
                ax.plot([a[0], b[0]], [a[1], b[1]], [a[2], b[2]],
                        color="red" if bad else "lightgray",
                        linewidth=2.5 if bad else 0.8,
                        alpha=1.0 if bad else 0.6,
                        zorder=2 if bad else 1)

    # All 8 corners as outline markers
    for c in corners:
        ax.scatter(*c, s=80, facecolor="white", edgecolor="gray",
                   linewidth=1.0, zorder=3)

    # Chosen 4 corners colored by pattern
    code_to_pattern = {tuple(int(x) for x in c): i for i, c in enumerate(codes)}
    for c, pi in code_to_pattern.items():
        ax.scatter(*c, s=320, color=PATTERN_COLORS[pi], edgecolor="black",
                   linewidth=1.0, zorder=5)
        ax.text(c[0], c[1], c[2] + 0.12, str(pi), fontsize=12,
                color="black", ha="center", weight="bold", zorder=6)

    ax.set_xlabel("H[0]")
    ax.set_ylabel("H[1]")
    ax.set_zlabel("H[2]")
    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_zticks([0, 1])
    ax.set_xlim(-0.15, 1.15)
    ax.set_ylim(-0.15, 1.15)
    ax.set_zlim(-0.15, 1.15)

    title = "3-cube: 4 chosen codes (colored = patterns 0..3)"
    if ec:
        title += "\nerror-correcting (no Hamming-1 pair)"
    else:
        title += "\nNOT error-correcting (red edges = Hamming-1 collisions)"
    ax.set_title(title, fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hamming_matrix(rbm: EncoderRBM, out_path: str):
    data = make_encoder_data()
    D = hamming_distances_between_codes(rbm, data)
    codes = dominant_codes(rbm, data)
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=140)
    im = ax.imshow(D, cmap="viridis", vmin=0, vmax=3)
    for i in range(4):
        for j in range(4):
            txt_color = "white" if D[i, j] < 2 else "black"
            ax.text(j, i, str(D[i, j]), ha="center", va="center",
                    color=txt_color, fontsize=14, weight="bold")
    label_strings = [f"P{i}\n({codes[i, 0]}{codes[i, 1]}{codes[i, 2]})"
                     for i in range(4)]
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(label_strings, fontsize=9)
    ax.set_yticklabels(label_strings, fontsize=9)
    ec = is_error_correcting(rbm, data)
    title = "Pairwise Hamming distance"
    title += "  (error-correcting)" if ec else "  (NOT error-correcting)"
    ax.set_title(title, fontsize=11)
    cb = plt.colorbar(im, ax=ax, ticks=[0, 1, 2, 3])
    cb.set_label("Hamming distance")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=1000)
    p.add_argument("--perturb-after", type=int, default=40)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.epochs} epochs (seed={args.seed})...")
    rbm, history = train(n_epochs=args.epochs, seed=args.seed,
                         perturb_after=args.perturb_after, verbose=False)
    data = make_encoder_data()
    print(f"  final accuracy: {history['acc'][-1]*100:.0f}%   "
          f"restarts: {history['perturbations']}   "
          f"min_h: {history['min_hamming'][-1]}   "
          f"EC: {is_error_correcting(rbm, data)}")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(rbm, os.path.join(args.outdir, "weights.png"))
    plot_three_cube(rbm, os.path.join(args.outdir, "three_cube.png"))
    plot_hamming_matrix(rbm, os.path.join(args.outdir, "hamming_matrix.png"))


if __name__ == "__main__":
    main()
