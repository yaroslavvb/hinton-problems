"""
Static visualizations for T-C discrimination weight-tied conv net.

Outputs (in `viz/`):
  patterns.png            — the 8 input patterns (T x4 + C x4) on the retina.
  training_curves.png     — loss + accuracy + weight norm over training.
  filters.png             — THE HEADLINE: each discovered 3x3 kernel rendered
                            as a heatmap, annotated by detector taxonomy
                            (compactness / bar / on-centre / off-centre /
                            mixed).
  feature_maps.png        — for each pattern, the K post-conv feature maps,
                            laid out as a (n_patterns, K) grid of heatmaps.
                            Shows where each detector fires.
  predictions.png         — bar chart: model output vs target for the 8
                            patterns.
  taxonomy_sweep.png      — bar chart of detector-type counts across N seeds.
"""

from __future__ import annotations
import argparse
import os
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt

from t_c_discrimination import (
    WeightTiedConvNet, train, make_dataset,
    visualize_filters, taxonomize_filter, filter_taxonomy,
)


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

def plot_patterns(retina_size: int, out_path: str):
    """Render the 8 input patterns as a 2 x 4 grid of binary images."""
    X, y, names = make_dataset(retina_size, augment_positions=False)
    fig, axes = plt.subplots(2, 4, figsize=(8, 4.5), dpi=130)
    for idx, ax in enumerate(axes.flat):
        ax.imshow(X[idx], cmap="gray_r", vmin=0, vmax=1)
        ax.set_title(names[idx], fontsize=10)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_edgecolor("#888")
        ax.set_xlabel("T (target=0)" if y[idx, 0] == 0 else "C (target=1)",
                      fontsize=8)
    fig.suptitle(f"8 patterns on a {retina_size}x{retina_size} retina  "
                  f"(top: T x 4 rotations,  bottom: C x 4 rotations)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    for ax, key, color, lab in [
            (axes[0], "loss", "#9467bd", "MSE  (0.5 * mean (o - y)^2)"),
            (axes[1], "accuracy", "#1f77b4", "accuracy"),
            (axes[2], "weight_norm", "#ff7f0e", r"$\|W\|_F$"),
    ]:
        vals = history[key]
        if key == "accuracy":
            vals = np.array(vals) * 100
        ax.plot(epochs, vals, color=color)
        if converged_at is not None:
            ax.axvline(converged_at, color="green", linestyle="--",
                       linewidth=0.9, alpha=0.8,
                       label=f"converged @ {converged_at}")
            ax.legend(fontsize=9)
        ax.set_xlabel("epoch")
        ax.set_ylabel(lab)
        ax.grid(alpha=0.3)
        if key == "accuracy":
            ax.set_ylim(-5, 105)
            ax.set_title("Classification accuracy")
        elif key == "loss":
            ax.set_title("Training loss")
        else:
            ax.set_title("Weight norm")

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# THE HEADLINE — discovered-filters viz
# ----------------------------------------------------------------------

# Color per detector type — used in legend + filter borders.
_TAX_COLOR = {
    "bar":          "#d62728",
    "compactness":  "#2ca02c",
    "on-centre":    "#1f77b4",
    "off-centre":   "#ff7f0e",
    "mixed":        "#888888",
    "dead":         "#cccccc",
    "unknown":      "#cccccc",
}


def plot_filters(model: WeightTiedConvNet, out_path: str,
                 title_extra: str = ""):
    """Render the K discovered kernels as heatmaps + their taxonomy.

    Each kernel gets a panel: 3x3 weight heatmap (red=positive, blue=negative,
    saturation by magnitude), colored border by detector type, and the type
    label as the panel title.
    """
    filters = visualize_filters(model)              # (K, 3, 3)
    types = filter_taxonomy(model)
    K = filters.shape[0]
    max_abs = max(np.abs(filters).max(), 1e-3)

    n_cols = K
    fig, axes = plt.subplots(1, n_cols, figsize=(3 * n_cols, 3.6), dpi=130)
    if n_cols == 1:
        axes = [axes]
    for k in range(K):
        ax = axes[k]
        W = filters[k]
        im = ax.imshow(W, cmap="RdBu_r", vmin=-max_abs, vmax=max_abs)
        for i in range(3):
            for j in range(3):
                v = W[i, j]
                txt_col = "white" if abs(v) > 0.55 * max_abs else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=9, color=txt_col)
        col = _TAX_COLOR.get(types[k], "#888")
        for spine in ax.spines.values():
            spine.set_edgecolor(col)
            spine.set_linewidth(2.5)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"kernel {k + 1}\n[{types[k]}]",
                      fontsize=11, color=col)

    counts = Counter(types)
    summary = ", ".join(f"{c} {t}"
                         for t, c in counts.most_common())
    sup = "Discovered 3x3 detectors  —  " + summary
    if title_extra:
        sup = title_extra + "  —  " + summary
    fig.suptitle(sup, fontsize=12, y=1.03)
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.04, shrink=0.8)
    cbar.set_label("weight value (red +, blue -)", fontsize=9)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Feature maps
# ----------------------------------------------------------------------

def plot_feature_maps(model: WeightTiedConvNet, out_path: str):
    """For every pattern, show the K post-sigmoid feature maps.

    Reveals which kernel is firing at which spatial position for each input.
    """
    X, y, names = make_dataset(model.R, augment_positions=False)
    h, _, _, _, _ = model.forward(X)        # (B, K, M, M)
    types = filter_taxonomy(model)
    n_p, K, M, _ = h.shape

    fig, axes = plt.subplots(n_p, K + 1,
                              figsize=(1.5 * (K + 1), 1.5 * n_p), dpi=130,
                              gridspec_kw={"width_ratios": [1.2] + [1] * K})
    for p in range(n_p):
        ax = axes[p, 0]
        ax.imshow(X[p], cmap="gray_r", vmin=0, vmax=1)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_ylabel(names[p], rotation=0, ha="right", va="center",
                       fontsize=8)
        if p == 0:
            ax.set_title("input", fontsize=9)
        for k in range(K):
            ax = axes[p, k + 1]
            im = ax.imshow(h[p, k], cmap="viridis", vmin=0, vmax=1)
            ax.set_xticks([])
            ax.set_yticks([])
            if p == 0:
                ax.set_title(f"k{k + 1}\n[{types[k]}]", fontsize=8,
                              color=_TAX_COLOR.get(types[k], "#888"))
    fig.suptitle("Feature maps — sigmoid(conv) at every spatial position",
                  fontsize=11, y=1.005)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------

def plot_predictions(model: WeightTiedConvNet, out_path: str):
    """Bar chart of per-pattern output vs target."""
    X, y, names = make_dataset(model.R, augment_positions=False)
    o = model.predict(X).ravel()
    y = y.ravel()

    fig, ax = plt.subplots(figsize=(8.5, 3.8), dpi=130)
    xpos = np.arange(len(X))
    bar_colors = ["#1f77b4" if y[i] == 0 else "#d62728"
                   for i in range(len(X))]
    ax.bar(xpos, o, color=bar_colors, edgecolor="black", linewidth=0.5,
           alpha=0.85)
    ax.scatter(xpos, y, marker="x", s=80, color="black",
                label="target", zorder=3)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(xpos)
    ax.set_xticklabels(names, fontsize=8, rotation=20)
    ax.set_ylabel("output / target")
    ax.set_ylim(0, 1.1)
    ax.set_title("Per-pattern outputs  (blue = T, red = C)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Multi-seed taxonomy bar chart
# ----------------------------------------------------------------------

def plot_taxonomy_sweep(taxonomies: list[list[str]], out_path: str,
                         n_seeds: int):
    """Bar chart: how often each detector type was discovered across seeds."""
    flat = [t for tax in taxonomies for t in tax]
    counts = Counter(flat)
    types_order = ["bar", "compactness", "on-centre", "off-centre", "mixed",
                    "dead"]
    types = [t for t in types_order if t in counts]
    n = sum(counts.values())
    fig, ax = plt.subplots(figsize=(7.5, 3.8), dpi=130)
    bars = ax.bar(types, [counts[t] for t in types],
                   color=[_TAX_COLOR.get(t, "#888") for t in types],
                   edgecolor="black", linewidth=0.5)
    for bar, t in zip(bars, types):
        c = counts[t]
        pct = 100 * c / max(n, 1)
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.2,
                f"{c}  ({pct:.0f}%)", ha="center", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(f"Discovered detector types across {n_seeds} seeds  "
                  f"(K kernels each, n = {n} kernels total)")
    ax.set_ylim(0, max(counts.values()) * 1.20 + 1)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--retina-size", type=int, default=6)
    p.add_argument("--kernel-size", type=int, default=3)
    p.add_argument("--n-kernels", type=int, default=4)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=0.5)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", type=int, default=10,
                   help="Seeds for the taxonomy_sweep.png bar chart "
                        "(0 to skip).")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Generating patterns viz...")
    plot_patterns(args.retina_size, os.path.join(args.outdir, "patterns.png"))

    print(f"Training seed={args.seed}, max_epochs={args.max_epochs}...")
    model, history = train(retina_size=args.retina_size,
                            kernel_size=args.kernel_size,
                            n_kernels=args.n_kernels, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']}, "
          f"final acc {history['accuracy'][-1] * 100:.0f}%")

    title = (f"seed={args.seed},  K={args.n_kernels}  kernels,  "
             f"converged @ {history['converged_epoch']}")
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"),
                         title=title)
    plot_filters(model, os.path.join(args.outdir, "filters.png"),
                  title_extra=f"seed={args.seed}")
    plot_feature_maps(model, os.path.join(args.outdir, "feature_maps.png"))
    plot_predictions(model, os.path.join(args.outdir, "predictions.png"))

    if args.sweep > 0:
        print(f"\nRunning {args.sweep}-seed sweep for taxonomy bar chart...")
        taxonomies = []
        for s in range(args.sweep):
            m, _ = train(retina_size=args.retina_size,
                          kernel_size=args.kernel_size,
                          n_kernels=args.n_kernels, lr=args.lr,
                          momentum=args.momentum,
                          init_scale=args.init_scale,
                          max_epochs=args.max_epochs, seed=s, verbose=False)
            taxonomies.append(filter_taxonomy(m))
        plot_taxonomy_sweep(taxonomies,
                             os.path.join(args.outdir, "taxonomy_sweep.png"),
                             n_seeds=args.sweep)


if __name__ == "__main__":
    main()
