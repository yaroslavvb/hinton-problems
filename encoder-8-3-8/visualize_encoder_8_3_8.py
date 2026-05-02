"""
Static visualizations for the trained 8-3-8 encoder.

Outputs (in `viz/`):
  training_curves.png  - accuracy, n_distinct_codes, code separation, weight norm, recon error
  weights.png          - final weight matrix as a Hinton diagram (16 rows x 3 cols)
  hidden_codes_3cube.png - 3-cube with 8 corners and which patterns landed where
  code_occupancy.png   - bar chart of how many patterns map to each of the 8 corners
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (3D backend registration)

from encoder_8_3_8 import (EncoderRBM, train, make_encoder_data,
                           N_PATTERNS, N_GROUP, N_HIDDEN)


PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e",
                  "#9467bd", "#8c564b", "#17becf", "#bcbd22"]


def _dominant_codes(rbm: EncoderRBM) -> list[tuple[int, ...]]:
    data = make_encoder_data()
    return [rbm.dominant_code(data[i, :N_GROUP]) for i in range(N_PATTERNS)]


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 3, figsize=(13, 6.5), dpi=120)

    epochs = history["epoch"]
    pert = history.get("perturbations", [])

    ax = axes[0, 0]
    ax.plot(epochs, np.array(history["acc"]) * 100, color="#1f77b4")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("accuracy (%)")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction accuracy")

    ax = axes[0, 1]
    ax.plot(epochs, history["n_distinct_codes"], color="#d62728", linewidth=1.5)
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(N_PATTERNS, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("distinct hidden codes")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, N_PATTERNS + 0.5)
    ax.grid(alpha=0.3)
    ax.set_title(f"Codes used (target = {N_PATTERNS})")

    ax = axes[0, 2]
    ax.plot(epochs, history["code_separation"], color="#2ca02c")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("mean pairwise L2 distance")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Hidden-code separation")

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Weight norm")

    ax = axes[1, 1]
    ax.plot(epochs, history["reconstruction_error"], color="#9467bd")
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.set_ylabel("MSE on V2")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.set_title("Reconstruction MSE")

    # Empty trailing panel: usage string for the seed.
    ax = axes[1, 2]
    ax.axis("off")
    if pert:
        msg = (f"restart epochs: {pert}\n"
               f"final n_codes: {history['n_distinct_codes'][-1]}/{N_PATTERNS}")
    else:
        msg = (f"no restarts\n"
               f"final n_codes: {history['n_distinct_codes'][-1]}/{N_PATTERNS}")
    ax.text(0.05, 0.5, msg, fontsize=10, family="monospace",
            verticalalignment="center")

    fig.suptitle("8-3-8 encoder — training curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(rbm: EncoderRBM, out_path: str):
    """Hinton diagram of the 16 x 3 weight matrix."""
    fig, ax = plt.subplots(figsize=(4.0, 8.0), dpi=140)
    W = rbm.W
    max_abs = max(abs(W).max(), 1e-3)
    n_v = W.shape[0]
    for i in range(n_v):
        for j in range(N_HIDDEN):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.4))
    ax.set_xlim(-0.7, N_HIDDEN - 0.3)
    ax.set_ylim(-0.7, n_v - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(N_HIDDEN))
    ax.set_xticklabels([f"H[{j}]" for j in range(N_HIDDEN)])
    ax.set_yticks(range(n_v))
    ax.set_yticklabels([f"V1[{i}]" for i in range(N_GROUP)] +
                       [f"V2[{i}]" for i in range(N_GROUP)], fontsize=7)
    ax.axhline(N_GROUP - 0.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(f"Final weights\n($\\|W\\|_F$ = {np.linalg.norm(W):.2f})",
                 fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_3cube(rbm: EncoderRBM, out_path: str):
    """3D scatter of the 8 cube corners with the patterns landing on them."""
    codes = _dominant_codes(rbm)
    fig = plt.figure(figsize=(7, 6), dpi=140)
    ax = fig.add_subplot(111, projection="3d")

    # Draw the cube edges.
    for i, c1 in enumerate(_unit_cube_corners()):
        for c2 in _unit_cube_corners()[i + 1:]:
            if sum(int(a != b) for a, b in zip(c1, c2)) == 1:
                ax.plot(*zip(c1, c2), color="lightgray", linewidth=1.0,
                        zorder=1)

    # Mark every corner as a small grey dot.
    for c in _unit_cube_corners():
        ax.scatter(*c, s=120, facecolors="none", edgecolors="gray",
                   linewidth=1.0, zorder=2)

    # Drop each pattern's marker on its corner with a small jitter for
    # visibility (multiple patterns sharing a corner = visible cluster).
    by_corner: dict = {}
    for pat, code in enumerate(codes):
        by_corner.setdefault(code, []).append(pat)
    rng = np.random.default_rng(0)
    for corner, patterns in by_corner.items():
        for k, pat in enumerate(patterns):
            jitter = rng.normal(scale=0.05, size=3) if len(patterns) > 1 else np.zeros(3)
            xyz = np.array(corner, dtype=float) + jitter
            ax.scatter(*xyz, s=200, color=PATTERN_COLORS[pat],
                       edgecolors="black", linewidth=0.8, zorder=3)
            ax.text(*xyz, str(pat), fontsize=9, color="white",
                    weight="bold", ha="center", va="center", zorder=4)

    n_used = len(set(codes))
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1]); ax.set_zticks([0, 1])
    ax.set_xlabel("H[0]"); ax.set_ylabel("H[1]"); ax.set_zlabel("H[2]")
    ax.set_title(f"Hidden codes on $\\{{0,1\\}}^3$ "
                 f"({n_used}/{N_PATTERNS} corners used)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_code_occupancy(rbm: EncoderRBM, out_path: str):
    """Bar chart: how many patterns map to each of the 8 corners?"""
    codes = _dominant_codes(rbm)
    counts = {c: 0 for c in _unit_cube_corners()}
    for c in codes:
        counts[c] += 1
    labels = [f"({c[0]}{c[1]}{c[2]})" for c in counts]
    values = list(counts.values())
    fig, ax = plt.subplots(figsize=(6, 3.2), dpi=140)
    bars = ax.bar(labels, values, color=["#2ca02c" if v == 1 else
                                          ("#d62728" if v > 1 else "#bbbbbb")
                                          for v in values],
                  edgecolor="black", linewidth=0.6)
    ax.set_ylabel("# patterns")
    ax.set_xlabel("hidden corner")
    ax.set_ylim(0, max(2, max(values) + 0.5))
    ax.set_title("Code occupancy per cube corner "
                 f"(green = used by 1 pattern, red = collision, grey = empty)")
    ax.grid(axis="y", alpha=0.3)
    for b, v in zip(bars, values):
        if v > 0:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.05, str(v),
                    ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _unit_cube_corners() -> list[tuple[int, int, int]]:
    return [(a, b, c) for a in (0, 1) for b in (0, 1) for c in (0, 1)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=4000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_cycles} epochs (seed={args.seed})...")
    rbm, history = train(n_epochs=args.n_cycles, seed=args.seed, verbose=False)
    from encoder_8_3_8 import codes_used, evaluate
    print(f"  final accuracy: {evaluate(rbm, make_encoder_data())*100:.0f}%   "
          f"codes used: {codes_used(rbm)}/{N_PATTERNS}   "
          f"restarts: {history['perturbations']}")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(rbm, os.path.join(args.outdir, "weights.png"))
    plot_3cube(rbm, os.path.join(args.outdir, "hidden_codes_3cube.png"))
    plot_code_occupancy(rbm, os.path.join(args.outdir, "code_occupancy.png"))


if __name__ == "__main__":
    main()
