"""
Static visualizations for the distributed-to-local-bottleneck run.

Outputs (in `viz/`):
  training_curves.png   - loss + accuracy + |W| + min pairwise h-gap
  graded_values.png     - the headline 1-D viz: 4 hidden values on a number line
                          + bar chart, with paper's target values overlaid
  output_curves.png     - the 4 output sigmoids as functions of h, with the
                          4 graded hidden values marked (1-D decision boundary)
  weights.png           - W1 (input -> hidden) and W2 (hidden -> output) heatmaps
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from distributed_to_local_bottleneck import (
    BottleneckMLP, train, generate_dataset, hidden_values, sigmoid,
)


# 4 distinct colors for the 4 patterns.
PATTERN_COLORS = ["#d62728", "#1f77b4", "#2ca02c", "#ff7f0e"]
PATTERN_LABELS = ["(0,0)", "(0,1)", "(1,0)", "(1,1)"]
PAPER_HIDDEN_TARGETS = np.array([0.0, 0.2, 0.6, 1.0])


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    epochs = history["epoch"]
    converged = history["converged_epoch"]
    perturbs = history.get("perturbations", [])

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="#9467bd", linewidth=1.2)
    if converged:
        ax.axvline(converged, color="green", linestyle="--", linewidth=0.9,
                   alpha=0.8, label=f"converged @ {converged}")
        ax.legend(fontsize=9)
    for p in perturbs:
        ax.axvline(p, color="red", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"MSE  $\frac{1}{2}\,\overline{\sum_j (o_j-t_j)^2}$")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, np.array(history["accuracy"]) * 100, color="#1f77b4",
            linewidth=1.2)
    if converged:
        ax.axvline(converged, color="green", linestyle="--", linewidth=0.9,
                   alpha=0.8)
    for p in perturbs:
        ax.axvline(p, color="red", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("argmax accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Classification accuracy")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e", linewidth=1.2)
    if converged:
        ax.axvline(converged, color="green", linestyle="--", linewidth=0.9,
                   alpha=0.8)
    for p in perturbs:
        ax.axvline(p, color="red", linewidth=0.4, alpha=0.5,
                   label="perturb" if p == perturbs[0] else None)
    if perturbs:
        ax.legend(fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_title("Weight norm")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    hv_traj = np.array(history["hidden_values"])  # (epochs, 4)
    for i in range(4):
        ax.plot(epochs, hv_traj[:, i], color=PATTERN_COLORS[i],
                linewidth=1.2, label=PATTERN_LABELS[i])
    if converged:
        ax.axvline(converged, color="green", linestyle="--", linewidth=0.9,
                   alpha=0.8)
    for p in perturbs:
        ax.axvline(p, color="red", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("hidden activation $h$")
    ax.set_ylim(-0.05, 1.05)
    ax.set_title("Hidden values per pattern (headline)")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    fig.suptitle(f"distributed-to-local-bottleneck  (epochs = {len(epochs)}, "
                 f"perturbations = {len(perturbs)})", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Graded-values 1-D viz (the unique deliverable)
# ----------------------------------------------------------------------

def plot_graded_values(model: BottleneckMLP, out_path: str):
    """Two side-by-side panels:
        Left: number-line view -- 4 colored markers at their h positions,
              paper targets shown as gray ticks underneath.
        Right: bar chart of the 4 hidden values, paper targets as horizontal
               reference lines.
    The point of this plot is to show that one sigmoid hidden unit takes
    4 distinct graded values to discriminate the 4 patterns.
    """
    X, _ = generate_dataset()
    hv = hidden_values(model, (X, None))
    sorted_idx = np.argsort(hv)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=140,
                             gridspec_kw={"width_ratios": [1.4, 1.0]})

    # ---- LEFT: number line ----
    ax = axes[0]
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.65, 0.85)

    # Main axis line
    ax.axhline(0.0, color="black", linewidth=1.4, zorder=2)
    for tick in [0.0, 0.25, 0.5, 0.75, 1.0]:
        ax.plot([tick, tick], [-0.04, 0.04], color="black", linewidth=1.0,
                zorder=2)
        ax.text(tick, -0.12, f"{tick:.2f}", ha="center", va="top", fontsize=9)
    ax.text(0.5, -0.30, "hidden activation  $h$", ha="center", fontsize=11)

    # Paper targets (small dark gray ticks below the axis).
    for tgt in PAPER_HIDDEN_TARGETS:
        ax.plot([tgt, tgt], [-0.45, -0.36], color="#888888", linewidth=1.2,
                zorder=2)
    ax.text(0.5, -0.55, "paper targets:  $h \\approx 0,\\ 0.2,\\ 0.6,\\ 1.0$",
            ha="center", color="#555555", fontsize=10, style="italic")

    # Observed hidden values: colored markers, label above with pattern.
    for i in range(4):
        h_i = float(hv[i])
        ax.scatter(h_i, 0.0, color=PATTERN_COLORS[i], s=240, zorder=4,
                   edgecolor="black", linewidth=0.9)
        ax.annotate(PATTERN_LABELS[i],
                    xy=(h_i, 0.05),
                    xytext=(h_i, 0.18 + 0.12 * (i % 2)),
                    ha="center", fontsize=9, color=PATTERN_COLORS[i],
                    weight="bold",
                    arrowprops=dict(arrowstyle="-",
                                    color=PATTERN_COLORS[i],
                                    linewidth=0.8, alpha=0.7))
        ax.text(h_i, 0.50 + 0.12 * (i % 2), f"h={h_i:.3f}",
                ha="center", fontsize=8, color="#222222")

    ax.set_yticks([])
    ax.set_xticks([])
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.set_title("4 distinct graded hidden values (1-D number line)",
                 fontsize=11)

    # ---- RIGHT: bar chart ----
    ax = axes[1]
    bar_x = np.arange(4)
    bars = ax.bar(bar_x, hv, color=PATTERN_COLORS, edgecolor="black",
                  linewidth=0.7, zorder=3)
    # Paper targets as dashed horizontals at 0, 0.2, 0.6, 1.0.
    for tgt in PAPER_HIDDEN_TARGETS:
        ax.axhline(tgt, color="#888888", linestyle=":", linewidth=0.9,
                   alpha=0.7, zorder=2)
    for i, h_i in enumerate(hv):
        ax.text(i, float(h_i) + 0.025, f"{float(h_i):.3f}", ha="center",
                fontsize=9, color="#222222")
    # Annotate paper targets on the right side
    for tgt in PAPER_HIDDEN_TARGETS:
        ax.text(3.55, tgt, f"{tgt:.1f}", ha="left", va="center", fontsize=8,
                color="#666666")
    ax.set_ylim(-0.05, 1.10)
    ax.set_xlim(-0.5, 4.0)
    ax.set_xticks(bar_x)
    ax.set_xticklabels(PATTERN_LABELS, fontsize=10)
    ax.set_xlabel("input pattern")
    ax.set_ylabel("hidden activation  $h$")
    ax.set_title("Per-pattern hidden values  (dotted = paper targets)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")

    fig.suptitle("Graded-values readout — the 1-unit bottleneck takes 4 distinct values",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 4 output sigmoids as functions of h
# ----------------------------------------------------------------------

def plot_output_curves(model: BottleneckMLP, out_path: str):
    """The 4 output sigmoids y_j = sigma(W2[j] * h + b2[j]) plotted over
    h in [0, 1], with the 4 hidden values marked by vertical dotted lines
    colored by pattern. This is the 1-D analog of a 2-D decision boundary.
    """
    X, _ = generate_dataset()
    hv = hidden_values(model, (X, None))

    h_grid = np.linspace(0.0, 1.0, 400)
    # outputs given hidden h: o_j = sigmoid(W2[j,0] * h + b2[j])
    W2 = model.W2.ravel()  # (4,) since n_hidden=1
    b2 = model.b2  # (4,)
    out = sigmoid(np.outer(h_grid, W2) + b2)  # (400, 4)

    fig, ax = plt.subplots(figsize=(8.5, 4.5), dpi=140)
    for j in range(4):
        ax.plot(h_grid, out[:, j], color=PATTERN_COLORS[j], linewidth=1.5,
                label=f"output {j} (target for {PATTERN_LABELS[j]})")

    # Argmax band: shade the regions of h where each output is the max.
    argmax = np.argmax(out, axis=1)
    for j in range(4):
        mask = argmax == j
        if not mask.any():
            continue
        # find contiguous runs of mask
        idx = np.where(mask)[0]
        # group contiguous indices
        starts, ends = [], []
        s = idx[0]
        for k in range(1, len(idx)):
            if idx[k] != idx[k - 1] + 1:
                starts.append(s)
                ends.append(idx[k - 1])
                s = idx[k]
        starts.append(s)
        ends.append(idx[-1])
        for a, b in zip(starts, ends):
            ax.axvspan(h_grid[a], h_grid[b], color=PATTERN_COLORS[j],
                       alpha=0.10, zorder=0)

    # Vertical lines at the 4 observed hidden values.
    for i in range(4):
        h_i = float(hv[i])
        ax.axvline(h_i, color=PATTERN_COLORS[i], linewidth=1.2, alpha=0.85,
                   linestyle="--")
        ax.text(h_i, 1.04, PATTERN_LABELS[i], ha="center",
                color=PATTERN_COLORS[i], fontsize=9, weight="bold")

    ax.set_xlim(0.0, 1.0)
    ax.set_ylim(-0.03, 1.10)
    ax.set_xlabel("hidden activation  $h$")
    ax.set_ylabel("output activation  $o_j$")
    ax.set_title("Output sigmoids as functions of $h$  "
                 "(1-D decision regions, shaded by argmax)")
    ax.legend(fontsize=8, loc="center left")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Weights heatmap
# ----------------------------------------------------------------------

def plot_weights(model: BottleneckMLP, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(8, 4), dpi=130)

    # W1 row (1 hidden x 2 inputs) + bias as a third column
    W1 = np.column_stack([model.W1, model.b1[:, None]])  # (1, 3)
    vmax = max(abs(W1).max(), 1e-3)
    im = axes[0].imshow(W1, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="auto")
    axes[0].set_yticks([0])
    axes[0].set_yticklabels(["h"], fontsize=10)
    axes[0].set_xticks([0, 1, 2])
    axes[0].set_xticklabels(["$x_1$", "$x_2$", "bias"], fontsize=10)
    axes[0].set_title(r"$W_1$  (input $\to$ hidden)")
    for j, name in enumerate(["w1_x1", "w1_x2", "b1"]):
        axes[0].text(j, 0, f"{W1[0, j]:+.2f}", ha="center", va="center",
                     fontsize=10, color="white" if abs(W1[0, j]) > 0.6 * vmax
                     else "black")
    fig.colorbar(im, ax=axes[0], shrink=0.8)

    # W2 column (4 outputs x 1 hidden) + bias column
    W2 = np.column_stack([model.W2, model.b2[:, None]])  # (4, 2)
    vmax = max(abs(W2).max(), 1e-3)
    im = axes[1].imshow(W2, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                        aspect="auto")
    axes[1].set_yticks(range(4))
    axes[1].set_yticklabels([f"o{i} ({PATTERN_LABELS[i]})" for i in range(4)],
                            fontsize=9)
    axes[1].set_xticks([0, 1])
    axes[1].set_xticklabels(["$h$", "bias"], fontsize=10)
    axes[1].set_title(r"$W_2$  (hidden $\to$ output)")
    for i in range(4):
        for j in range(2):
            axes[1].text(j, i, f"{W2[i, j]:+.2f}", ha="center", va="center",
                         fontsize=9,
                         color="white" if abs(W2[i, j]) > 0.6 * vmax
                         else "black")
    fig.colorbar(im, ax=axes[1], shrink=0.8)

    fig.suptitle(f"Final weights  ($\\|W\\|_F$ = "
                 f"{np.linalg.norm(np.concatenate([model.W1.ravel(), model.W2.ravel()])):.2f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sweeps", type=int, default=5000)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training (seed={args.seed}, n_sweeps={args.n_sweeps})...")
    model, history = train(seed=args.seed, n_sweeps=args.n_sweeps,
                           lr=args.lr, momentum=args.momentum,
                           init_scale=args.init_scale, verbose=False)
    X, T = generate_dataset()
    hv = hidden_values(model, (X, T))
    print(f"  final accuracy: {history['accuracy'][-1] * 100:.0f}%   "
          f"hidden values: {[f'{v:.3f}' for v in hv]}   "
          f"epochs run: {len(history['epoch'])}")

    plot_training_curves(history,
                          os.path.join(args.outdir, "training_curves.png"))
    plot_graded_values(model,
                        os.path.join(args.outdir, "graded_values.png"))
    plot_output_curves(model,
                        os.path.join(args.outdir, "output_curves.png"))
    plot_weights(model, os.path.join(args.outdir, "weights.png"))


if __name__ == "__main__":
    main()
