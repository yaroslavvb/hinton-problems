"""
Static visualizations for the XOR backprop run.

Outputs (in `viz/`):
  training_curves.png   — loss + classification accuracy + |W|
  weights.png           — Hinton diagram of W1, W2, biases (and Wskip if 2-1-2)
  decision_boundary.png — final 2-D decision surface with the 4 training points
  hidden_activations.png — what the hidden units fire for at the 4 inputs
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from xor import XorMLP, train, make_xor_data


PATTERN_COLORS = ["#1f77b4", "#d62728", "#d62728", "#1f77b4"]
PATTERN_MARKERS = ["o", "s", "s", "o"]
PATTERN_LABELS = ["(0,0)→0", "(0,1)→1", "(1,0)→1", "(1,1)→0"]


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    ax = axes[0]
    ax.plot(epochs, history["loss"], color="#9467bd")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8, label=f"converged @ {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE  (0.5 · mean (o-y)²)")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, np.array(history["accuracy"]) * 100, color="#1f77b4")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Classification accuracy")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_title("Weight norm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _hinton_rect(ax, x: float, y: float, w: float, max_abs: float,
                 max_size: float = 0.85):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                           facecolor=color, edgecolor="black", linewidth=0.4))


def plot_weights(model: XorMLP, out_path: str):
    """Hinton-style diagram of every weight in the network."""
    if model.arch == "2-2-1":
        fig, axes = plt.subplots(1, 2, figsize=(7, 3.5), dpi=130,
                                  gridspec_kw={"width_ratios": [1, 1]})
        # ---- W1: 2 hidden x 2 inputs (+ bias column) ----
        ax = axes[0]
        W = np.column_stack([model.W1, model.b1[:, None]])  # (2, 3)
        max_abs = max(abs(W).max(), 1e-3)
        for i in range(2):
            for j in range(3):
                _hinton_rect(ax, j, i, W[i, j], max_abs)
        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(-0.7, 1.7)
        ax.invert_yaxis()
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["x₁", "x₂", "bias"], fontsize=10)
        ax.set_yticks([0, 1])
        ax.set_yticklabels(["h₁", "h₂"], fontsize=10)
        ax.set_aspect("equal")
        ax.set_title("Hidden layer weights")

        # ---- W2: 1 output x 2 hidden (+ bias) ----
        ax = axes[1]
        W = np.column_stack([model.W2, model.b2[:, None]])  # (1, 3)
        max_abs = max(abs(W).max(), 1e-3)
        for j in range(3):
            _hinton_rect(ax, j, 0, W[0, j], max_abs)
        ax.set_xlim(-0.7, 2.7)
        ax.set_ylim(-0.7, 0.7)
        ax.invert_yaxis()
        ax.set_xticks([0, 1, 2])
        ax.set_xticklabels(["h₁", "h₂", "bias"], fontsize=10)
        ax.set_yticks([0])
        ax.set_yticklabels(["o"], fontsize=10)
        ax.set_aspect("equal")
        ax.set_title("Output layer weights")
    else:
        fig, axes = plt.subplots(1, 3, figsize=(9, 3), dpi=130,
                                  gridspec_kw={"width_ratios": [1, 1, 1]})
        # W1 (1, 2) + bias
        ax = axes[0]
        W = np.column_stack([model.W1, model.b1[:, None]])  # (1, 3)
        max_abs = max(abs(W).max(), 1e-3)
        for j in range(3):
            _hinton_rect(ax, j, 0, W[0, j], max_abs)
        ax.set_xlim(-0.7, 2.7); ax.set_ylim(-0.7, 0.7)
        ax.invert_yaxis()
        ax.set_xticks([0, 1, 2]); ax.set_xticklabels(["x₁", "x₂", "bias"])
        ax.set_yticks([0]); ax.set_yticklabels(["h"])
        ax.set_aspect("equal"); ax.set_title("Input → hidden")
        # Wskip (1, 2) + W2 (1, 1) + bias
        ax = axes[1]
        W = np.column_stack([model.Wskip, model.W2, model.b2[:, None]])  # (1, 4)
        max_abs = max(abs(W).max(), 1e-3)
        for j in range(4):
            _hinton_rect(ax, j, 0, W[0, j], max_abs)
        ax.set_xlim(-0.7, 3.7); ax.set_ylim(-0.7, 0.7)
        ax.invert_yaxis()
        ax.set_xticks([0, 1, 2, 3])
        ax.set_xticklabels(["x₁ (skip)", "x₂ (skip)", "h", "bias"])
        ax.set_yticks([0]); ax.set_yticklabels(["o"])
        ax.set_aspect("equal"); ax.set_title("→ output  (skip + hidden)")
        # legend panel
        ax = axes[2]
        ax.axis("off")
        ax.text(0.0, 0.7, "red  = positive", color="#cc0000", fontsize=11)
        ax.text(0.0, 0.5, "blue = negative", color="#003366", fontsize=11)
        ax.text(0.0, 0.3, "size ∝ √|w|", fontsize=11, color="black")

    fig.suptitle(f"Final weights ({model.arch})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_decision_boundary(model: XorMLP, out_path: str):
    """Shade output sigmoid over the [0,1]x[0,1] input plane; overlay 4 patterns."""
    grid = 200
    xs = np.linspace(-0.1, 1.1, grid)
    ys = np.linspace(-0.1, 1.1, grid)
    XX, YY = np.meshgrid(xs, ys)
    pts = np.column_stack([XX.ravel(), YY.ravel()])
    o = model.predict(pts).reshape(grid, grid)

    X, y = make_xor_data()
    fig, ax = plt.subplots(figsize=(5, 4.5), dpi=130)
    im = ax.imshow(o, extent=(-0.1, 1.1, -0.1, 1.1), origin="lower",
                   cmap="RdBu_r", vmin=0, vmax=1, aspect="equal", alpha=0.85)
    cs = ax.contour(XX, YY, o, levels=[0.5], colors="black", linewidths=1.2)
    for i, (xi, yi) in enumerate(zip(X, y.ravel())):
        ax.scatter(xi[0], xi[1], marker=PATTERN_MARKERS[i],
                   c=PATTERN_COLORS[i], s=240, edgecolor="black",
                   linewidths=1.5, zorder=3)
        ax.annotate(f"{int(yi)}", (xi[0], xi[1]), textcoords="offset points",
                    xytext=(0, 0), ha="center", va="center", fontsize=10,
                    color="white", weight="bold", zorder=4)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("network output  o(x)", fontsize=10)
    ax.set_xlabel(r"$x_1$"); ax.set_ylabel(r"$x_2$")
    ax.set_title("Decision surface  (black: o = 0.5)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_activations(model: XorMLP, out_path: str):
    X, _ = make_xor_data()
    h, o = model.forward(X)
    n_hidden = h.shape[1]
    fig, ax = plt.subplots(figsize=(5.5, 3.5), dpi=130)
    width = 0.35 if n_hidden == 2 else 0.6
    xpos = np.arange(4)
    for hi in range(n_hidden):
        ax.bar(xpos + (hi - (n_hidden - 1) / 2) * width, h[:, hi],
               width=width, label=f"h{hi+1}",
               color=["#2ca02c", "#9467bd"][hi % 2])
    ax.set_xticks(xpos); ax.set_xticklabels(PATTERN_LABELS, fontsize=9)
    ax.set_ylabel("activation")
    ax.set_ylim(0, 1.05)
    ax.set_title("Hidden-unit activations on the 4 patterns")
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="upper right", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["2-2-1", "2-1-2-skip"], default="2-2-1")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training {args.arch}, seed={args.seed}, max_epochs={args.max_epochs}...")
    model, history = train(arch=args.arch, lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']},  "
          f"final acc {history['accuracy'][-1]*100:.0f}%")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_decision_boundary(model, os.path.join(args.outdir, "decision_boundary.png"))
    plot_hidden_activations(model, os.path.join(args.outdir, "hidden_activations.png"))


if __name__ == "__main__":
    main()
