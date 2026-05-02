"""
Static visualizations for the trained family-trees network.

Outputs (in `viz/`):
  training_curves.png   - cross-entropy loss + train/test accuracy
  person_encoding.png   - 6 x 24 heatmap of the person-encoding layer,
                          rows annotated with nationality / generation / branch
  per_unit_bars.png     - one bar chart per encoding unit, grouped by
                          nationality / generation / branch (the headline
                          interpretable-axes view)
  pca_scatter.png       - 24 people projected to PC1 / PC2 of the 6-D
                          encoding, colored by each of the three attributes
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from family_trees import (
    FamilyTreeMLP, ALL_PEOPLE, ENGLISH_PEOPLE, ITALIAN_PEOPLE,
    build_triples, aggregate_facts, split_train_test,
    facts_to_arrays, train, inspect_person_encoding, attribute_table,
)


# Colour scheme: keep nationality / generation / branch palettes distinct.
NATIONALITY_COLORS = {0: "#1f77b4", 1: "#d62728"}            # eng / ita
NATIONALITY_LABEL = {0: "English", 1: "Italian"}
GEN_COLORS = {1: "#2ca02c", 2: "#ff7f0e", 3: "#9467bd"}      # 3 generations
BRANCH_COLORS = {0: "#17becf", 1: "#bcbd22", 2: "#7f7f7f"}   # left / right / outsider
BRANCH_LABEL = {0: "left (Christopher / Roberto)",
                1: "right (Andrew / Pierro)",
                2: "outsider (married in)"}


# ----------------------------------------------------------------------
# Training-curve panel
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)
    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")
    ax.set_title("Training loss")

    ax = axes[1]
    ax.plot(epochs, np.array(history["train_acc"]) * 100,
            color="#1f77b4", label="train")
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#d62728", label="test")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right")
    ax.set_title("Train / test accuracy (argmax in valid set)")

    fig.suptitle("Family trees — backprop training", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Heatmap of the 24 x 6 person encoding
# ----------------------------------------------------------------------

def plot_person_encoding(codes: np.ndarray, attrs: dict, out_path: str):
    """24 x 6 heatmap, rows sorted by (nationality, generation, branch)."""
    order = np.lexsort((attrs["branch"], attrs["generation"], attrs["nationality"]))
    sorted_codes = codes[order]
    sorted_names = [ALL_PEOPLE[i] for i in order]
    sorted_nat = attrs["nationality"][order]
    sorted_gen = attrs["generation"][order]
    sorted_branch = attrs["branch"][order]

    fig, ax = plt.subplots(figsize=(7.5, 8.5), dpi=140)
    im = ax.imshow(sorted_codes, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"unit {j}" for j in range(6)])
    ax.set_yticks(range(24))
    # Annotate the row labels with the attribute combo
    row_labels = [f"{n:11s}  [{NATIONALITY_LABEL[sorted_nat[i]][:3]}, "
                  f"g{sorted_gen[i]}, b{sorted_branch[i]}]"
                  for i, n in enumerate(sorted_names)]
    ax.set_yticklabels(row_labels, fontsize=8, family="monospace")
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02, label="tanh activation")

    # Horizontal separators between nationalities
    for boundary in np.where(np.diff(sorted_nat) != 0)[0]:
        ax.axhline(boundary + 0.5, color="black", linewidth=0.8)

    ax.set_title("6-D person encoding — rows sorted by nationality, "
                 "generation, branch", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-unit bar chart grouped by attribute (THE headline panel)
# ----------------------------------------------------------------------

def _bar_panel(ax, codes: np.ndarray, group_ids: np.ndarray,
               group_colors: dict, title: str, ylabel: str | None,
               group_labels: dict | None = None, show_legend: bool = False):
    n_people, n_units = codes.shape
    x = np.arange(n_people)
    colors = [group_colors[int(g)] for g in group_ids]
    for j in range(n_units):
        # Each bar is colored by group
        ax.bar(x + j * (n_people + 2), codes[:, j], color=colors,
               edgecolor="black", linewidth=0.2)
    # Boundaries between unit blocks
    for j in range(1, n_units):
        ax.axvline(j * (n_people + 2) - 1, color="lightgray",
                   linewidth=0.5, linestyle="--")
    ax.set_xticks([j * (n_people + 2) + (n_people - 1) / 2
                   for j in range(n_units)])
    ax.set_xticklabels([f"unit {j}" for j in range(n_units)])
    if ylabel:
        ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_ylim(-1.05, 1.05)
    ax.grid(axis="y", alpha=0.3)
    if show_legend and group_labels is not None:
        ax.legend(handles=[Patch(facecolor=group_colors[k], edgecolor="black",
                                 label=group_labels[k])
                           for k in sorted(group_labels)],
                  loc="lower right", fontsize=8, framealpha=0.95)


def plot_per_unit_bars(codes: np.ndarray, attrs: dict, out_path: str):
    fig, axes = plt.subplots(3, 1, figsize=(13, 9.5), dpi=130)
    _bar_panel(axes[0], codes, attrs["nationality"], NATIONALITY_COLORS,
               "Person encoding — bars grouped by nationality "
               "(units that fire ~equally on each side encode shared structure)",
               ylabel="tanh activation",
               group_labels=NATIONALITY_LABEL, show_legend=True)
    _bar_panel(axes[1], codes, attrs["generation"], GEN_COLORS,
               "Bars grouped by generation (1 = grandparents, 2 = parents, "
               "3 = grandchildren)",
               ylabel="tanh activation",
               group_labels={i: f"generation {i}" for i in (1, 2, 3)},
               show_legend=True)
    _bar_panel(axes[2], codes, attrs["branch"], BRANCH_COLORS,
               "Bars grouped by family branch", ylabel="tanh activation",
               group_labels=BRANCH_LABEL, show_legend=True)
    axes[2].set_xlabel("Each block = one of the 6 person-encoding units; "
                       "each block has 24 bars, one per person")
    fig.suptitle("6-unit person-encoding layer — interpretable axes "
                 "(Hinton 1986)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# 2-D PCA scatter
# ----------------------------------------------------------------------

def _pca_2d(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return (proj, explained_variance_ratio[:2])."""
    mean = codes.mean(axis=0, keepdims=True)
    centered = codes - mean
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt.T[:, :2]
    var = (s ** 2) / max(len(codes) - 1, 1)
    return proj, var[:2] / var.sum()


def plot_pca_scatter(codes: np.ndarray, attrs: dict, out_path: str):
    proj, var_ratio = _pca_2d(codes)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5.0), dpi=130)
    titles = [
        ("nationality", attrs["nationality"], NATIONALITY_COLORS, NATIONALITY_LABEL),
        ("generation", attrs["generation"], GEN_COLORS,
         {i: f"generation {i}" for i in (1, 2, 3)}),
        ("branch", attrs["branch"], BRANCH_COLORS, BRANCH_LABEL),
    ]
    for ax, (title, group_ids, color_map, label_map) in zip(axes, titles):
        for k in sorted(set(int(x) for x in group_ids)):
            mask = group_ids == k
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=color_map[k], s=110, edgecolors="black",
                       linewidths=0.6, label=label_map[k], zorder=3)
        for i, (px, py) in enumerate(proj):
            ax.annotate(ALL_PEOPLE[i][:3], (px, py),
                        xytext=(0, 0), textcoords="offset points",
                        ha="center", va="center", fontsize=6,
                        weight="bold", color="white", zorder=4)
        ax.set_xlabel(f"PC1 ({var_ratio[0]*100:.0f}% var)")
        ax.set_ylabel(f"PC2 ({var_ratio[1]*100:.0f}% var)")
        ax.set_title(f"colored by {title}")
        ax.legend(loc="best", fontsize=8, framealpha=0.95)
        ax.grid(alpha=0.3)
    fig.suptitle("PCA of the 6-D person encoding "
                 "(top two principal components)", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--weight-decay", type=float, default=0.0)
    p.add_argument("--n-test", type=int, default=4)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    triples = build_triples()
    facts = aggregate_facts(triples)
    train_facts, test_facts = split_train_test(facts,
                                               n_test=args.n_test,
                                               seed=args.seed)
    X_train, Y_train = facts_to_arrays(train_facts)
    X_test, Y_test = facts_to_arrays(test_facts)

    print(f"Training {args.epochs} epochs (seed={args.seed})...")
    model = FamilyTreeMLP(seed=args.seed, init_scale=args.init_scale)
    history = train(model, X_train, Y_train, X_test, Y_test,
                    n_sweeps=args.epochs, lr=args.lr,
                    momentum=0.9, weight_decay=args.weight_decay,
                    verbose=False)
    print(f"  final train acc: {history['train_acc'][-1]*100:.1f}%   "
          f"test acc: {history['test_acc'][-1]*100:.1f}%")

    codes = inspect_person_encoding(model)
    attrs = attribute_table()

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_person_encoding(codes, attrs,
                         os.path.join(args.outdir, "person_encoding.png"))
    plot_per_unit_bars(codes, attrs,
                       os.path.join(args.outdir, "per_unit_bars.png"))
    plot_pca_scatter(codes, attrs,
                     os.path.join(args.outdir, "pca_scatter.png"))


if __name__ == "__main__":
    main()
