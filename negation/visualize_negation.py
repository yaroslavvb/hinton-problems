"""
Static visualizations for the negation backprop run.

Outputs (in `viz/`):
  training_curves.png     — loss + per-bit accuracy + per-pattern accuracy + |W|
  weights.png             — Hinton diagram of W1 (input → hidden) and W2 (hidden → output)
  hidden_routing.png      — flag-gated routing: hidden activations across all 16 patterns,
                            split into the flag=0 and flag=1 halves so the gating is visible
  hidden_role_map.png     — per-hidden-unit role: which (flag, bit) combination it detects,
                            inferred from the (16, n_hidden) activation matrix
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from negation import (NegationMLP, train, make_negation_data, pattern_label,
                      sigmoid)


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    ax = axes[0]
    ax.plot(epochs, history["loss"], color="#9467bd")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8, label=f"converged @ {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"MSE  (½ · mean $\sum_i(o_i-y_i)^2$)")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, np.array(history["accuracy"]) * 100,
            color="#1f77b4", label="bit accuracy")
    ax.plot(epochs, np.array(history["pattern_accuracy"]) * 100,
            color="#d62728", label="pattern accuracy", alpha=0.85)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Accuracy: per-bit and per-pattern")
    ax.legend(fontsize=9, loc="lower right")
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


def plot_weights(model: NegationMLP, out_path: str):
    """Hinton-style diagram of W1 (n_hidden x 4) and W2 (3 x n_hidden)."""
    n_h = model.n_hidden
    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.55 * n_h)),
                              dpi=130,
                              gridspec_kw={"width_ratios": [1.1, 1]})

    # ---- W1: n_hidden x 4 (+ bias column) ----
    ax = axes[0]
    W = np.column_stack([model.W1, model.b1[:, None]])  # (n_h, 5)
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(n_h):
        for j in range(5):
            _hinton_rect(ax, j, i, W[i, j], max_abs)
    ax.set_xlim(-0.7, 4.7)
    ax.set_ylim(-0.7, n_h - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(5))
    ax.set_xticklabels(["flag", "b₁", "b₂", "b₃", "bias"], fontsize=10)
    ax.set_yticks(range(n_h))
    ax.set_yticklabels([f"h{i+1}" for i in range(n_h)], fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Input → hidden  (W₁)")

    # ---- W2: 3 x n_hidden (+ bias column) ----
    ax = axes[1]
    W = np.column_stack([model.W2, model.b2[:, None]])  # (3, n_h+1)
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(3):
        for j in range(n_h + 1):
            _hinton_rect(ax, j, i, W[i, j], max_abs)
    ax.set_xlim(-0.7, n_h + 0.7)
    ax.set_ylim(-0.7, 2.7)
    ax.invert_yaxis()
    ax.set_xticks(range(n_h + 1))
    ax.set_xticklabels([f"h{i+1}" for i in range(n_h)] + ["bias"], fontsize=9)
    ax.set_yticks([0, 1, 2])
    ax.set_yticklabels(["o₁", "o₂", "o₃"], fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("Hidden → output  (W₂)")

    fig.suptitle(f"Final weights  (4-{n_h}-3,  red = +, blue = −,  area ∝ √|w|)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_routing(model: NegationMLP, out_path: str):
    """Hidden-unit activations across all 16 patterns, split by flag value.

    Reveals flag-gated routing: which hidden units fire when flag=0 vs flag=1.
    """
    X, _ = make_negation_data()
    H = model.forward(X)[0]                # (16, n_hidden)
    n_h = H.shape[1]

    # Sort patterns by (flag, b1, b2, b3) — already in this order from make_negation_data
    order_f0 = [i for i in range(16) if X[i, 0] == 0]
    order_f1 = [i for i in range(16) if X[i, 0] == 1]
    H_f0 = H[order_f0]
    H_f1 = H[order_f1]
    labels_f0 = [pattern_label(X[i]) for i in order_f0]
    labels_f1 = [pattern_label(X[i]) for i in order_f1]

    fig, axes = plt.subplots(1, 2, figsize=(11, max(3.5, 0.4 * n_h + 2)),
                              dpi=130, sharey=True)
    for ax, H_block, labels, title in [
        (axes[0], H_f0, labels_f0, "flag = 0  (output = data)"),
        (axes[1], H_f1, labels_f1, "flag = 1  (output = NOT data)"),
    ]:
        im = ax.imshow(H_block.T, aspect="auto", cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(8))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=9)
        ax.set_yticks(range(n_h))
        ax.set_yticklabels([f"h{i+1}" for i in range(n_h)], fontsize=10)
        ax.set_title(title, fontsize=11)
        ax.set_xlabel("input pattern (flag|b₁b₂b₃)")
        # annotate cells
        for i in range(n_h):
            for j in range(8):
                v = H_block[j, i]
                ax.text(j, i, f"{v:.1f}", ha="center", va="center",
                        color="white" if v < 0.55 else "black", fontsize=7)
    axes[0].set_ylabel("hidden unit")
    fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02,
                  label="hidden activation")
    fig.suptitle("Flag-gated hidden routing  —  16 patterns × {} hidden units"
                  .format(n_h), fontsize=12)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def _classify_role(activations_16: np.ndarray, X: np.ndarray) -> str:
    """Best-effort: name the (flag, bit, polarity) that this hidden unit detects.

    Approach: a hidden unit's "role" is the (flag-value, bit-index, bit-value)
    triple where the unit is most discriminative. Score each candidate by the
    AUC-style separation between matching and non-matching patterns.
    """
    best = None
    best_gap = -1.0
    for flag_val in (0, 1):
        for bit_idx in (1, 2, 3):
            for bit_val in (0, 1):
                mask = (X[:, 0] == flag_val) & (X[:, bit_idx] == bit_val)
                if mask.sum() == 0 or mask.sum() == 16:
                    continue
                gap = activations_16[mask].mean() - activations_16[~mask].mean()
                if abs(gap) > best_gap:
                    best_gap = abs(gap)
                    polarity = "+" if gap > 0 else "−"
                    name = f"flag={flag_val} ∧ b{bit_idx}={bit_val}  ({polarity})"
                    best = (name, gap)
    return best[0] if best else "?"


def plot_hidden_role_map(model: NegationMLP, out_path: str):
    """For each hidden unit, name the (flag, bit) combo it detects.

    Plot a small grid: rows = hidden units, columns = (flag×bit) combinations,
    colored by the unit's mean activation conditional on that combination.
    """
    X, _ = make_negation_data()
    H = model.forward(X)[0]                # (16, n_hidden)
    n_h = H.shape[1]

    # 6 condition columns: (flag, bit_idx, bit_val) where bit_val=1 (i.e.,
    # "flag=0 ∧ b1=1", "flag=0 ∧ b2=1", "flag=0 ∧ b3=1", same for flag=1).
    cond_labels = []
    cond_means = np.zeros((n_h, 6))
    for c, (flag_val, bit_idx) in enumerate(
            [(0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)]):
        mask = (X[:, 0] == flag_val) & (X[:, bit_idx] == 1)
        cond_means[:, c] = H[mask].mean(axis=0)
        cond_labels.append(f"flag={flag_val}\nb{bit_idx}=1")

    # Each row also gets a textual "role" guess.
    roles = [_classify_role(H[:, i], X) for i in range(n_h)]

    fig, ax = plt.subplots(figsize=(9.5, max(3.0, 0.55 * n_h + 1.0)), dpi=130)
    im = ax.imshow(cond_means, cmap="RdBu_r", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(6))
    ax.set_xticklabels(cond_labels, fontsize=9)
    ax.set_yticks(range(n_h))
    ax.set_yticklabels([f"h{i+1}: {roles[i]}" for i in range(n_h)],
                        fontsize=9)
    for i in range(n_h):
        for j in range(6):
            ax.text(j, i, f"{cond_means[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cond_means[i, j] < 0.35
                    or cond_means[i, j] > 0.75 else "black",
                    fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02,
                  label="mean hidden activation")
    ax.set_title("Hidden-unit roles  (mean activation conditional on flag ∧ bit)",
                  fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=6)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training 4-{args.n_hidden}-3, seed={args.seed}, "
          f"max_epochs={args.max_epochs}...")
    model, history = train(n_hidden=args.n_hidden, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']},  "
          f"final pattern acc {history['pattern_accuracy'][-1]*100:.0f}%")

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_hidden_routing(model,
                        os.path.join(args.outdir, "hidden_routing.png"))
    plot_hidden_role_map(model,
                         os.path.join(args.outdir, "hidden_role_map.png"))


if __name__ == "__main__":
    main()
