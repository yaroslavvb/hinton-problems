"""
Static visualizations for the trained binary-addition network.

Outputs (in `viz/`):
  training_curves.png    - loss + per-bit accuracy + per-pattern acc + |W|
  weights.png            - Hinton diagram of W1, W2, biases
  hidden_activations.png - what each hidden unit fires for on the 16 patterns
  local_minima_gap.png   - side-by-side bar chart: 4-3-3 vs 4-2-3
                            convergence rates + final-accuracy distributions
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from binary_addition import (BinaryAdditionMLP, train, generate_dataset,
                              local_minimum_rate)


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
    ax.set_ylabel(r"MSE  $0.5 \cdot \mathrm{mean}(o-y)^2$")
    ax.set_yscale("log")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(epochs, np.array(history["accuracy_bit"]) * 100,
            color="#1f77b4", linewidth=1.4, label="per-bit (48 bits)")
    ax.plot(epochs, np.array(history["accuracy_pattern"]) * 100,
            color="#d62728", linewidth=1.4, label="per-pattern (16 patterns)")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy")
    ax.legend(fontsize=9, loc="lower right")
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
    # Final output bits per pattern (predictions vs targets) -- summary table
    ax.axis("off")
    summary = [
        f"final loss             : {history['loss'][-1]:.4f}",
        f"final per-bit acc      : {history['accuracy_bit'][-1]*100:.1f}%",
        f"final per-pattern acc  : {history['accuracy_pattern'][-1]*100:.1f}%",
        f"converged at sweep     : {converged_at}",
        f"final $\\|W_1\\|_F$            : {history['weight_norm'][-1]:.2f}",
    ]
    ax.text(0.02, 0.98, "\n".join(summary), va="top", ha="left",
            fontsize=11, family="monospace",
            transform=ax.transAxes)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(model: BinaryAdditionMLP, out_path: str):
    """Hinton diagram of W1 (input->hidden) and W2 (hidden->output)."""
    n_h = model.n_hidden
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.6), dpi=130,
                              gridspec_kw={"width_ratios": [1, 1]})

    # ---- left: W1 Hinton (rows=hidden, cols=4 inputs + 1 bias) ----
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
    ax.set_xticklabels(["a₁", "a₀", "b₁", "b₀", "bias"])
    ax.set_yticks(range(n_h))
    ax.set_yticklabels([f"h{i+1}" for i in range(n_h)])
    ax.axvline(3.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_aspect("equal")
    ax.set_title(f"$W_1$ (input → hidden)  arch={model.arch}",
                  fontsize=10)

    # ---- right: W2 Hinton (rows=3 outputs, cols=hidden + 1 bias) ----
    ax = axes[1]
    W = np.column_stack([model.W2, model.b2[:, None]])  # (3, n_h + 1)
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(3):
        for j in range(n_h + 1):
            _hinton_rect(ax, j, i, W[i, j], max_abs)
    ax.set_xlim(-0.7, n_h + 0.3)
    ax.set_ylim(-0.7, 2.7)
    ax.invert_yaxis()
    ax.set_xticks(range(n_h + 1))
    ax.set_xticklabels([f"h{i+1}" for i in range(n_h)] + ["bias"])
    ax.set_yticks(range(3))
    ax.set_yticklabels(["s₂ (4)", "s₁ (2)", "s₀ (1)"])
    ax.axvline(n_h - 0.5, color="gray", linewidth=0.6, linestyle=":")
    ax.set_aspect("equal")
    ax.set_title(f"$W_2$ (hidden → output)  red +, blue −", fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_activations(model: BinaryAdditionMLP, out_path: str):
    """What each hidden unit fires for on the 16 patterns."""
    X, y = generate_dataset()
    h, o = model.forward(X)
    n_h = h.shape[1]

    fig, axes = plt.subplots(1, n_h, figsize=(4 * n_h, 4), dpi=130,
                              sharey=True)
    if n_h == 1:
        axes = [axes]

    pattern_labels = []
    for a in range(4):
        for b in range(4):
            pattern_labels.append(f"{a}+{b}")

    # Color by output sum for visual clarity
    sums = np.array([a + b for a in range(4) for b in range(4)])
    cmap = plt.get_cmap("viridis")
    colors = [cmap(s / 6.0) for s in sums]

    for hi in range(n_h):
        ax = axes[hi]
        ax.bar(np.arange(16), h[:, hi], color=colors,
                edgecolor="black", linewidth=0.3)
        ax.set_xticks(np.arange(16))
        ax.set_xticklabels(pattern_labels, rotation=60, fontsize=8)
        ax.set_ylim(0, 1.05)
        ax.set_title(f"hidden unit h{hi+1}", fontsize=10)
        ax.set_xlabel("pattern")
        if hi == 0:
            ax.set_ylabel("activation")
        ax.grid(alpha=0.3, axis="y")

    # legend for sum colors
    handles = [Rectangle((0, 0), 1, 1, color=cmap(s / 6.0),
                          label=f"sum={s}") for s in range(7)]
    axes[-1].legend(handles=handles, fontsize=7, loc="upper right",
                    title="a+b", ncol=2)

    fig.suptitle(f"Hidden-unit activations across all 16 patterns "
                  f"({model.arch})", fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_local_minima_gap(out_path: str, n_trials: int = 30,
                           n_sweeps: int = 5000, lr: float = 2.0,
                           momentum: float = 0.9,
                           init_scale: float = 2.0):
    """Headline figure: side-by-side comparison of 4-3-3 vs 4-2-3."""
    print(f"  running {n_trials}-trial sweep for 4-3-3 ...")
    r33 = local_minimum_rate(arch="4-3-3", n_trials=n_trials,
                              n_sweeps=n_sweeps, lr=lr, momentum=momentum,
                              init_scale=init_scale)
    print(f"  running {n_trials}-trial sweep for 4-2-3 ...")
    r23 = local_minimum_rate(arch="4-2-3", n_trials=n_trials,
                              n_sweeps=n_sweeps, lr=lr, momentum=momentum,
                              init_scale=init_scale)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), dpi=130,
                              gridspec_kw={"width_ratios": [1, 1.4]})

    # ---- left: bar chart of stuck rate ----
    ax = axes[0]
    bars = ax.bar([0, 1], [r33["rate"] * 100, r23["rate"] * 100],
                   color=["#2ca02c", "#d62728"],
                   edgecolor="black", linewidth=0.6)
    for b, r in zip(bars, [r33, r23]):
        ax.text(b.get_x() + b.get_width() / 2, r["rate"] * 100 + 1.0,
                f"{r['n_stuck']}/{r['n_trials']}\n({r['rate']*100:.0f}%)",
                ha="center", va="bottom", fontsize=10)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["4-3-3\n(3 hidden)", "4-2-3\n(2 hidden)"], fontsize=11)
    ax.set_ylabel("local-minimum rate (%)")
    ax.set_ylim(0, 110)
    ax.set_title(f"Local-minimum rate over {n_trials} random seeds",
                  fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # ---- right: histogram of final per-pattern accuracy ----
    ax = axes[1]
    bins = np.linspace(0, 1, 17)
    ax.hist(r33["final_pattern_accs"], bins=bins, alpha=0.65,
             color="#2ca02c", edgecolor="black", linewidth=0.3,
             label=f"4-3-3 (succ {r33['n_converged']}/{r33['n_trials']})")
    ax.hist(r23["final_pattern_accs"], bins=bins, alpha=0.65,
             color="#d62728", edgecolor="black", linewidth=0.3,
             label=f"4-2-3 (succ {r23['n_converged']}/{r23['n_trials']})")
    ax.axvline(1.0, color="black", linestyle="--", linewidth=0.8, alpha=0.6)
    ax.set_xlabel("final per-pattern accuracy")
    ax.set_ylabel("# seeds")
    ax.set_title("Distribution of final per-pattern accuracies", fontsize=10)
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3, axis="y")

    gap_pp = (r23["rate"] - r33["rate"]) * 100
    fig.suptitle(f"Local-minima gap: 4-2-3 stuck rate {r23['rate']*100:.0f}% "
                  f"vs 4-3-3 stuck rate {r33['rate']*100:.0f}% "
                  f"(gap {gap_pp:+.0f} pp)",
                  fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return r33, r23


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--arch", choices=["4-3-3", "4-2-3"], default="4-3-3")
    p.add_argument("--sweeps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=2.0)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=2.0)
    p.add_argument("--seed", type=int, default=10,
                    help="seed=10 reliably converges with the default config")
    p.add_argument("--n-trials", type=int, default=30,
                    help="trials for the local-minima gap figure")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training {args.arch}, seed={args.seed}, sweeps={args.sweeps}...")
    model, history = train(arch=args.arch, n_sweeps=args.sweeps,
                            lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale,
                            seed=args.seed, verbose=False)
    print(f"  converged @ sweep {history['converged_epoch']},  "
          f"final per-pattern acc {history['accuracy_pattern'][-1]*100:.0f}%")

    plot_training_curves(history,
                          os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_hidden_activations(model,
                              os.path.join(args.outdir, "hidden_activations.png"))
    plot_local_minima_gap(os.path.join(args.outdir, "local_minima_gap.png"),
                            n_trials=args.n_trials, n_sweeps=args.sweeps,
                            lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale)


if __name__ == "__main__":
    main()
