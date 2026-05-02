"""
Static visualizations for the trained 40-10-40 encoder.

Outputs (in `viz/`):
  training_curves.png  - accuracy, n_distinct_codes, code separation, weight norm, recon error
  weights.png          - final weight matrix as a Hinton diagram (80 rows x 10 cols)
  speed_accuracy.png   - the headline plot: per-trial accuracy vs Gibbs-sweep budget
  code_occupancy.png   - per-pattern dominant code (which of the 1024 corners),
                         displayed as a heatmap of the 40 dominant 10-bit codes
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from encoder_40_10_40 import (EncoderRBM, train, make_encoder_data,
                              speed_accuracy_curve, evaluate_exact,
                              n_distinct_codes,
                              N_PATTERNS, N_GROUP, N_HIDDEN)


def _dominant_codes(rbm: EncoderRBM) -> np.ndarray:
    """Shape (N_PATTERNS, N_HIDDEN). Each row = the dominant 10-bit hidden code."""
    data = make_encoder_data()
    p_h = rbm.hidden_posterior_exact_batch(data[:, :N_GROUP])
    H = rbm._h_state_table()
    return H[np.argmax(p_h, axis=1)]


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
    ax.set_title("Reconstruction accuracy (exact)")

    ax = axes[0, 1]
    ax.plot(epochs, history["n_distinct_codes"], color="#d62728", linewidth=1.5)
    for pe in pert:
        ax.axvline(pe, color="red", linestyle="--", linewidth=0.7, alpha=0.5)
    ax.axhline(N_PATTERNS, color="black", linewidth=0.6, linestyle=":")
    ax.set_ylabel("distinct dominant codes")
    ax.set_xlabel("epoch")
    ax.set_ylim(0, N_PATTERNS + 1)
    ax.grid(alpha=0.3)
    ax.set_title(f"Codes used (target = {N_PATTERNS} of 1024)")

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

    ax = axes[1, 2]
    ax.axis("off")
    if pert:
        msg = (f"restart epochs: {pert}\n"
               f"final n_codes:  {history['n_distinct_codes'][-1]}/{N_PATTERNS}\n"
               f"final acc:      {history['acc'][-1]*100:.1f}%")
    else:
        msg = (f"no restarts\n"
               f"final n_codes:  {history['n_distinct_codes'][-1]}/{N_PATTERNS}\n"
               f"final acc:      {history['acc'][-1]*100:.1f}%")
    ax.text(0.05, 0.5, msg, fontsize=10, family="monospace",
            verticalalignment="center")

    fig.suptitle("40-10-40 encoder — training curves", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(rbm: EncoderRBM, out_path: str):
    """Hinton diagram of the 80 x 10 weight matrix."""
    fig, ax = plt.subplots(figsize=(5.5, 12.0), dpi=120)
    W = rbm.W
    max_abs = max(abs(W).max(), 1e-3)
    n_v = W.shape[0]
    for i in range(n_v):
        for j in range(N_HIDDEN):
            w = W[i, j]
            sz = 0.85 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.3))
    ax.set_xlim(-0.7, N_HIDDEN - 0.3)
    ax.set_ylim(-0.7, n_v - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(N_HIDDEN))
    ax.set_xticklabels([f"H[{j}]" for j in range(N_HIDDEN)], fontsize=8)
    ax.set_yticks(range(0, n_v, 4))
    ax.set_yticklabels([f"V1[{i}]" if i < N_GROUP else f"V2[{i - N_GROUP}]"
                        for i in range(0, n_v, 4)], fontsize=6)
    ax.axhline(N_GROUP - 0.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_title(f"Final weights\n($\\|W\\|_F$ = {np.linalg.norm(W):.1f})",
                 fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_speed_accuracy_curve(rbm: EncoderRBM, out_path: str,
                              budgets=(1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                              n_trials: int = 100):
    """Headline plot: per-trial accuracy vs Gibbs-sweep budget at retrieval.

    Two curves:
      - blue: per-trial accuracy (single chain) — what one Gibbs run delivers
      - orange: ensemble accuracy (argmax of trial-averaged V2 prob) — what
        many parallel chains converge to

    Horizontal dashed line at the asymptotic exact accuracy (1024-state
    enumeration) marks the limit reachable as sweeps -> infinity.
    """
    print("  computing speed/accuracy curve (per-trial)...")
    t0 = time.time()
    pt = speed_accuracy_curve(rbm, sweep_budgets=budgets,
                              n_trials=n_trials, T=1.0,
                              mode="per_trial", seed=42)
    print(f"  computing speed/accuracy curve (averaged) ...")
    avg = speed_accuracy_curve(rbm, sweep_budgets=budgets,
                               n_trials=n_trials, T=1.0,
                               mode="averaged", seed=42)
    asy = evaluate_exact(rbm)
    print(f"  computed in {time.time() - t0:.1f}s; "
          f"asymptotic = {asy*100:.1f}%")

    xs = [b for b, _ in pt]
    ys_pt = [a for _, a in pt]
    ys_avg = [a for _, a in avg]
    fig, ax = plt.subplots(figsize=(7.5, 4.5), dpi=140)
    ax.plot(xs, [y * 100 for y in ys_pt],
            marker="o", color="#1f77b4", linewidth=2.0,
            label=f"per-trial (single chain, {n_trials} trials averaged)")
    ax.plot(xs, [y * 100 for y in ys_avg],
            marker="s", color="#ff7f0e", linewidth=2.0, alpha=0.85,
            label=f"ensemble argmax ({n_trials}-chain mean)")
    ax.axhline(asy * 100, color="black", linestyle="--", linewidth=1.0,
               alpha=0.7, label=f"asymptotic (exact) = {asy*100:.1f}%")
    ax.axhline(100.0 / N_PATTERNS, color="gray", linestyle=":", linewidth=0.8,
               alpha=0.7, label=f"chance = {100.0/N_PATTERNS:.1f}%")
    ax.set_xscale("log", base=2)
    ax.set_xlabel("Gibbs sweeps at retrieval")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-2, 105)
    ax.set_xticks(xs)
    ax.set_xticklabels([str(x) for x in xs])
    ax.set_title("Speed/accuracy curve at retrieval (T = 1.0)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_code_assignments(rbm: EncoderRBM, out_path: str):
    """Show each of the 40 dominant 10-bit codes as a row in a binary heatmap.

    Each row = one pattern's hidden-state argmax (10 bits). All 40 rows
    distinct = solved.
    """
    codes = _dominant_codes(rbm)                                # (40, 10)
    fig, ax = plt.subplots(figsize=(6.0, 7.5), dpi=130)
    ax.imshow(codes, cmap="Greys", aspect="auto", vmin=0, vmax=1,
              interpolation="nearest")
    ax.set_xlabel("hidden bit (H[0]..H[9])")
    ax.set_ylabel("pattern index")
    ax.set_xticks(range(N_HIDDEN))
    ax.set_xticklabels([f"H[{j}]" for j in range(N_HIDDEN)], fontsize=8)
    ax.set_yticks(range(0, N_PATTERNS, 2))
    n_used = len({tuple(int(x) for x in row) for row in codes})
    ax.set_title(f"Dominant hidden codes per pattern\n"
                 f"({n_used}/{N_PATTERNS} distinct of {2**N_HIDDEN} corners)",
                 fontsize=11)
    # Emphasize the structure: gridlines between rows.
    for i in range(N_PATTERNS + 1):
        ax.axhline(i - 0.5, color="white", linewidth=0.4)
    for j in range(N_HIDDEN + 1):
        ax.axvline(j - 0.5, color="white", linewidth=0.4)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--n-trials", type=int, default=100,
                   help="Trials per (sweep budget) for the speed/accuracy curve.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_cycles} epochs (seed={args.seed})...")
    t0 = time.time()
    rbm, history = train(n_epochs=args.n_cycles, seed=args.seed,
                         verbose=False)
    train_secs = time.time() - t0
    print(f"  train time: {train_secs:.1f}s   "
          f"asymptotic acc: {evaluate_exact(rbm)*100:.1f}%   "
          f"codes used: {n_distinct_codes(rbm)}/{N_PATTERNS}   "
          f"restarts: {history['perturbations']}")

    plot_training_curves(history, os.path.join(args.outdir,
                                               "training_curves.png"))
    plot_weights(rbm, os.path.join(args.outdir, "weights.png"))
    plot_speed_accuracy_curve(rbm,
                              os.path.join(args.outdir, "speed_accuracy.png"),
                              n_trials=args.n_trials)
    plot_code_assignments(rbm, os.path.join(args.outdir,
                                            "code_occupancy.png"))


if __name__ == "__main__":
    main()
