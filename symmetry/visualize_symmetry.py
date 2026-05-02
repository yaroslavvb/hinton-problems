"""
Static visualizations for the trained 6-2-1 symmetry network.

Outputs (in `viz/`):
  training_curves.png   - loss + accuracy + |W1| + ratio dynamics
  weights.png           - Hinton-diagram of W1 + bar chart of |w_i| per
                           input position, showing the 1:2:4 anti-symmetric
                           pattern
  weight_pattern.png    - the per-pair magnitude bars side-by-side with
                           the paper's 1:2:4 prediction (sorted view)
  hidden_activations.png - what the two hidden units fire for on the 64
                           patterns, palindromes highlighted
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from symmetry import (SymmetryMLP, train, make_symmetry_data,
                       inspect_weight_symmetry)


PAL_COLOR = "#d62728"
NONPAL_COLOR = "#1f77b4"


def _hinton_rect(ax, x, y, w, max_abs, max_size=0.85):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                            facecolor=color, edgecolor="black",
                            linewidth=0.4))


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="#9467bd", linewidth=1.4)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8,
                    label=f"converged @ sweep {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("sweep")
    ax.set_ylabel("MSE  (0.5 mean (o-y)$^2$)")
    ax.set_yscale("log")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(epochs, np.array(history["accuracy"]) * 100,
            color="#1f77b4", linewidth=1.4)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("classification accuracy (%)")
    ax.set_ylim(80, 102)
    ax.set_title("Accuracy on all 64 patterns")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e", linewidth=1.4)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel(r"$\|W_1\|_F$")
    ax.set_title("Input-to-hidden weight norm")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["ratio_2_to_1"],
            color="#2ca02c", linewidth=1.4, label="middle : outer")
    ax.plot(epochs, history["ratio_4_to_2"],
            color="#d62728", linewidth=1.4, label="inner : middle")
    ax.axhline(2.0, color="gray", linestyle=":", linewidth=1.0,
                label="paper target (2.0)")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("magnitude ratio")
    ax.set_ylim(0, max(5, 1.1 * max(history["ratio_2_to_1"]
                                       + history["ratio_4_to_2"])))
    ax.set_title("Pair-magnitude ratios over training")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(model: SymmetryMLP, out_path: str):
    """Side-by-side: Hinton diagram (raw weights) + magnitude bar chart."""
    W1 = model.W1
    sym = inspect_weight_symmetry(model)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), dpi=130,
                              gridspec_kw={"width_ratios": [1, 1.2]})

    # ---- left: Hinton diagram of W1 (rows: hidden units, cols: input pos) ----
    ax = axes[0]
    max_abs = max(abs(W1).max(), 1e-3)
    for i in range(W1.shape[0]):       # 2 hidden
        for j in range(W1.shape[1]):   # 6 inputs
            _hinton_rect(ax, j, i, W1[i, j], max_abs)
    ax.set_xlim(-0.7, 5.7)
    ax.set_ylim(-0.7, W1.shape[0] - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"x{i+1}" for i in range(6)])
    ax.set_yticks(range(W1.shape[0]))
    ax.set_yticklabels([f"h{i+1}" for i in range(W1.shape[0])])
    ax.axvline(2.5, color="gray", linewidth=0.7, linestyle=":")
    ax.set_title("$W_1$ (red +, blue $-$, area $\\propto \\sqrt{|w|}$)")
    ax.set_aspect("equal")

    # ---- right: |w_i| bar chart per input position, both hidden units ----
    ax = axes[1]
    width = 0.36
    xpos = np.arange(6)
    h_colors = ["#2ca02c", "#9467bd"]
    for h in range(W1.shape[0]):
        ax.bar(xpos + (h - (W1.shape[0] - 1) / 2) * width,
               np.abs(W1[h]), width=width,
               color=h_colors[h % 2], label=f"h{h+1}")
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"x{i+1}" for i in range(6)])
    ax.set_xlabel("input position")
    ax.set_ylabel(r"$|w_i|$")
    ax.set_title("Hidden weight magnitudes by input position")
    ax.axvline(2.5, color="gray", linewidth=0.7, linestyle=":")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")

    matches = sym["matches_paper"]
    ratio_text = (f"sorted ratios = 1 : "
                   f"{sym['sorted_ratio_medium_to_smallest']:.2f} : "
                   f"{sym['sorted_ratio_largest_to_smallest']:.2f}    "
                   f"anti-sym residual = {sym['anti_residual_max']:.2f}    "
                   f"matches paper = {matches}")
    fig.suptitle(ratio_text, fontsize=10, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_pattern(model: SymmetryMLP, out_path: str):
    """The 1:2:4 ratio shown directly: bars per pair, sorted, with the
    paper's predicted 1:2:4 ratio as gray reference bars."""
    sym = inspect_weight_symmetry(model)

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)

    # ---- per-pair magnitudes in their original (outer/middle/inner) order ----
    ax = axes[0]
    mags = sym["mean_pair_magnitudes"]
    pair_labels = ["outer\n(x1, x6)", "middle\n(x2, x5)", "inner\n(x3, x4)"]
    bars = ax.bar(np.arange(3), mags, color="#1f77b4",
                   edgecolor="black", linewidth=0.5)
    for b, m in zip(bars, mags):
        ax.text(b.get_x() + b.get_width() / 2, m * 1.02,
                 f"{m:.2f}", ha="center", va="bottom", fontsize=10)
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(pair_labels)
    ax.set_ylabel(r"mean $|w|$ over both hidden units")
    ax.set_title("Pair magnitudes (positional order)")
    ax.grid(alpha=0.3, axis="y")

    # ---- sorted vs paper's 1:2:4 prediction ----
    ax = axes[1]
    sorted_mags = sym["sorted_pair_magnitudes"]
    paper_pred = np.array([1.0, 2.0, 4.0]) * sorted_mags[0]
    width = 0.36
    xs = np.arange(3)
    ax.bar(xs - width / 2, sorted_mags, width=width,
            color="#d62728", edgecolor="black", linewidth=0.5,
            label="observed (sorted)")
    ax.bar(xs + width / 2, paper_pred, width=width,
            color="#cccccc", edgecolor="black", linewidth=0.5,
            label="paper 1:2:4")
    ax.set_xticks(xs)
    ax.set_xticklabels(["smallest", "medium", "largest"])
    ax.set_ylabel(r"$|w|$ pair magnitude")
    ax.set_title(f"Observed vs paper's 1:2:4 prediction\n"
                  f"sorted ratios = 1 : "
                  f"{sym['sorted_ratio_medium_to_smallest']:.2f} : "
                  f"{sym['sorted_ratio_largest_to_smallest']:.2f}")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper left")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_activations(model: SymmetryMLP, out_path: str,
                              encoding: str = "pm1"):
    """Hidden-unit activations on all 64 patterns, palindromes highlighted."""
    X, y = make_symmetry_data(encoding=encoding)
    h, _ = model.forward(X)
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130, sharey=True)
    for hi in range(h.shape[1]):
        ax = axes[hi]
        order = np.argsort(h[:, hi])
        colors = [PAL_COLOR if y[k] == 1 else NONPAL_COLOR for k in order]
        ax.bar(np.arange(64), h[order, hi], color=colors,
                edgecolor="black", linewidth=0.2)
        ax.set_title(f"hidden unit h{hi+1}")
        ax.set_xlabel("pattern (sorted by activation)")
        ax.set_ylabel("activation")
        ax.set_ylim(0, 1.05)
        ax.grid(alpha=0.3, axis="y")

    handles = [
        plt.Rectangle((0, 0), 1, 1, color=PAL_COLOR, label="palindrome (8)"),
        plt.Rectangle((0, 0), 1, 1, color=NONPAL_COLOR, label="non-palindrome (56)"),
    ]
    axes[-1].legend(handles=handles, fontsize=9, loc="upper left")
    fig.suptitle("Hidden-unit activations across all 64 patterns",
                  fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--sweeps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.95)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--encoding", choices=["pm1", "01"], default="pm1")
    p.add_argument("--seed", type=int, default=1,
                    help="seed 1 reliably gives the paper's exact pattern")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training 6-2-1 symmetry, seed={args.seed}, sweeps={args.sweeps}...")
    model, history = train(n_sweeps=args.sweeps, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            encoding=args.encoding, seed=args.seed,
                            verbose=False)
    print(f"  converged @ sweep {history['converged_epoch']},  "
          f"final acc {history['accuracy'][-1]*100:.0f}%")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_weight_pattern(model, os.path.join(args.outdir, "weight_pattern.png"))
    plot_hidden_activations(model, os.path.join(args.outdir,
                                                  "hidden_activations.png"),
                              encoding=args.encoding)


if __name__ == "__main__":
    main()
