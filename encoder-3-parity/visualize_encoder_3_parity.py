"""
Static visualizations for the 3-bit even-parity encoder.

Outputs (in `viz/`):
  distribution_visible.png  - learned vs target p(v) for visible-only
                              Boltzmann (the negative result)
  distribution_hidden.png   - same plot for the RBM with hidden units
                              (the fix)
  distribution_compare.png  - side-by-side, target vs visible-only vs RBM
  training_curves.png       - KL and p(even) over training, both variants
  weights_rbm.png           - Hinton diagram of final RBM weights
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from encoder_3_parity import (
    ALL_PATTERNS, EVEN_MASK, ODD_MASK, PARITY,
    target_distribution, train_visible, train_hidden,
    pattern_string,
)


PATTERN_LABELS = [pattern_string(ALL_PATTERNS[i]) for i in range(8)]
EVEN_COLOR = "#2ca02c"   # green: should have mass
ODD_COLOR = "#d62728"    # red: should be zero
TARGET_COLOR = "#888888"


def plot_distribution(p_model: np.ndarray, title: str, out_path: str,
                      subtitle: str | None = None):
    target = target_distribution()
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=130)
    x = np.arange(8)
    width = 0.38

    # bar colors by parity
    bar_colors = [EVEN_COLOR if PARITY[i] == 0 else ODD_COLOR
                  for i in range(8)]
    ax.bar(x - width / 2, target, width, label="target",
           color=TARGET_COLOR, alpha=0.65, edgecolor="black", linewidth=0.5)
    ax.bar(x + width / 2, p_model, width, label="learned",
           color=bar_colors, edgecolor="black", linewidth=0.5)

    ax.axhline(0.125, color="gray", linewidth=0.5, linestyle=":",
               label="uniform 1/8")
    ax.set_xticks(x)
    ax.set_xticklabels(PATTERN_LABELS, fontfamily="monospace", fontsize=11)
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(EVEN_COLOR if PARITY[i] == 0 else ODD_COLOR)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("probability")
    ax.set_xlabel("3-bit pattern  (green = even parity, red = odd parity)")
    ax.set_title(title)
    if subtitle:
        ax.text(0.5, 0.97, subtitle, transform=ax.transAxes,
                ha="center", va="top", fontsize=9, color="#444")
    ax.legend(loc="upper right", framealpha=0.9, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_distribution_compare(p_visible: np.ndarray, p_hidden: np.ndarray,
                              out_path: str):
    target = target_distribution()
    fig, ax = plt.subplots(figsize=(10, 5), dpi=130)
    x = np.arange(8)
    width = 0.27

    ax.bar(x - width, target, width, label="target",
           color=TARGET_COLOR, alpha=0.7, edgecolor="black", linewidth=0.5)
    ax.bar(x, p_visible, width,
           label="visible-only Boltzmann (n_hidden=0)",
           color="#1f77b4", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.bar(x + width, p_hidden, width,
           label="RBM with hidden units",
           color="#ff7f0e", alpha=0.85, edgecolor="black", linewidth=0.5)
    ax.axhline(0.125, color="gray", linewidth=0.5, linestyle=":",
               label="uniform 1/8")
    ax.set_xticks(x)
    ax.set_xticklabels(PATTERN_LABELS, fontfamily="monospace", fontsize=11)
    for i, lbl in enumerate(ax.get_xticklabels()):
        lbl.set_color(EVEN_COLOR if PARITY[i] == 0 else ODD_COLOR)
    ax.set_ylim(0, 0.45)
    ax.set_ylabel("probability")
    ax.set_xlabel("3-bit pattern")
    ax.set_title("3-bit even-parity ensemble: visible-only fails, hidden units fix it")
    ax.legend(loc="upper right", framealpha=0.95, fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history_visible: dict, history_hidden: dict,
                         out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

    ax = axes[0]
    ax.plot(history_visible["step"], history_visible["kl"],
            color="#1f77b4", linewidth=1.5, label="visible-only")
    ax.plot(history_hidden["step"], history_hidden["kl"],
            color="#ff7f0e", linewidth=1.5, label="hidden RBM")
    ax.axhline(np.log(2), color="gray", linewidth=0.8, linestyle="--",
               label="KL to uniform = log 2")
    ax.set_xlabel("training step")
    ax.set_ylabel("KL(target || model)")
    ax.set_title("KL divergence")
    ax.set_ylim(-0.02, 0.85)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=9, framealpha=0.95)

    ax = axes[1]
    ax.plot(history_visible["step"],
            np.array(history_visible["p_even_total"]) * 100,
            color="#1f77b4", linewidth=1.5, label="visible-only")
    ax.plot(history_hidden["step"],
            np.array(history_hidden["p_even_total"]) * 100,
            color="#ff7f0e", linewidth=1.5, label="hidden RBM")
    ax.axhline(50, color="gray", linewidth=0.8, linestyle="--",
               label="uniform = 50%")
    ax.axhline(100, color=EVEN_COLOR, linewidth=0.8, linestyle=":",
               label="target = 100%")
    ax.set_xlabel("training step")
    ax.set_ylabel("p(even-parity patterns), %")
    ax.set_title("Mass on the 4 correct (even-parity) patterns")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9, framealpha=0.95)

    fig.suptitle("Training curves: visible-only stalls at the uniform baseline; "
                 "hidden RBM escapes",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_rbm_weights(rbm, out_path: str):
    W = rbm.W
    n_v, n_h = W.shape
    fig, ax = plt.subplots(figsize=(0.7 + 0.55 * n_h, 2.5), dpi=140)
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(n_v):
        for j in range(n_h):
            w = W[i, j]
            sz = 0.7 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.4))
    ax.set_xlim(-0.7, n_h - 0.3)
    ax.set_ylim(-0.7, n_v - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(n_h))
    ax.set_xticklabels([f"H[{j}]" for j in range(n_h)], fontsize=8)
    ax.set_yticks(range(n_v))
    ax.set_yticklabels([f"V[{i}]" for i in range(n_v)], fontsize=9)
    ax.set_title(f"RBM weights  ($\\|W\\|_F$ = {np.linalg.norm(W):.2f})",
                 fontsize=10)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--visible-steps", type=int, default=400)
    p.add_argument("--hidden-epochs", type=int, default=800)
    p.add_argument("--n-hidden", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training visible-only Boltzmann ({args.visible_steps} steps, "
          f"seed={args.seed})...")
    bm, hist_v = train_visible(n_steps=args.visible_steps, seed=args.seed,
                               verbose=False)
    p_visible = bm.model_distribution()
    kl_v = hist_v["kl"][-1]
    print(f"  final KL = {kl_v:.4f}, p(even) = {hist_v['p_even_total'][-1]:.3f}")

    print(f"Training RBM (n_hidden={args.n_hidden}, "
          f"{args.hidden_epochs} epochs, seed={args.seed})...")
    rbm, hist_h = train_hidden(n_hidden=args.n_hidden,
                               n_epochs=args.hidden_epochs, seed=args.seed,
                               verbose=False)
    p_hidden = rbm.model_distribution()
    kl_h = hist_h["kl"][-1]
    print(f"  final KL = {kl_h:.4f}, p(even) = {hist_h['p_even_total'][-1]:.3f}")

    plot_distribution(
        p_visible,
        title="Visible-only Boltzmann: collapses to uniform (the negative result)",
        subtitle=f"KL(target || model) = {kl_v:.3f}, "
                 f"p(even) = {hist_v['p_even_total'][-1]:.3f}",
        out_path=os.path.join(args.outdir, "distribution_visible.png"),
    )
    plot_distribution(
        p_hidden,
        title=f"RBM with {args.n_hidden} hidden units: learns the parity ensemble",
        subtitle=f"KL(target || model) = {kl_h:.3f}, "
                 f"p(even) = {hist_h['p_even_total'][-1]:.3f}",
        out_path=os.path.join(args.outdir, "distribution_hidden.png"),
    )
    plot_distribution_compare(
        p_visible, p_hidden,
        out_path=os.path.join(args.outdir, "distribution_compare.png"),
    )
    plot_training_curves(hist_v, hist_h,
                         out_path=os.path.join(args.outdir,
                                               "training_curves.png"))
    plot_rbm_weights(rbm, out_path=os.path.join(args.outdir, "weights_rbm.png"))


if __name__ == "__main__":
    main()
