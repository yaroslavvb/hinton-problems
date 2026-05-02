"""
Visualize the ellipse-world dataset, the trained eGLOM-lite, and island
formation in the embedding space.

Outputs (under --outdir, default ./viz):
    examples_<class>.png      one example grid per class
    ambiguity_grid.png        same class at ambiguity = 0, 0.4, 0.8, 1.2
    training_curves.png       loss + accuracy curves
    confusion_matrix.png      val-set confusion matrix
    iters_ablation.png        accuracy as a function of T at inference
    islands_<class>.png       per-iteration cosine-sim heatmap (one per class)

Usage:
    python3 visualize_ellipse_world.py --seed 0 --ambiguity 0.4 --outdir viz
"""

from __future__ import annotations
import argparse
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from ellipse_world import (
    CLASSES, N_CLASSES, F_PER_LOC,
    LAYOUTS,
    sample_ellipse_layout, apply_affine, random_affine, add_ambiguity,
    render_grid, generate_dataset,
    train, accuracy, forward, visualize_islands,
)


# ----------------------------------------------------------------------
# Visual helpers
# ----------------------------------------------------------------------

def draw_grid(ax, layout, grid_size, mask, title=""):
    """Plot ellipses + cell grid lines."""
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)

    cell = 2.0 / grid_size
    # cell grid
    for k in range(grid_size + 1):
        x = -1 + k * cell
        ax.plot([x, x], [-1, 1], "k-", alpha=0.10, lw=0.5)
        ax.plot([-1, 1], [x, x], "k-", alpha=0.10, lw=0.5)

    # highlight occupied cells
    cell_coords = -1 + (np.arange(grid_size) + 0.5) * cell
    if mask is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if mask[idx] > 0:
                    ax.add_patch(plt.Rectangle(
                        (cell_coords[j] - cell / 2, cell_coords[i] - cell / 2),
                        cell, cell, facecolor="orange", alpha=0.10,
                        edgecolor="none"))

    # ellipses
    for cx, cy, a, b, theta in layout:
        e = Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(theta),
                    facecolor="#3060a0", edgecolor="black",
                    alpha=0.75, lw=1.0)
        ax.add_patch(e)

    ax.set_xticks([])
    ax.set_yticks([])


# ----------------------------------------------------------------------
# Examples per class
# ----------------------------------------------------------------------

def plot_examples_per_class(outdir, grid_size, ambiguity, seed):
    rng = np.random.default_rng(seed)
    fig, axes = plt.subplots(1, N_CLASSES, figsize=(2.6 * N_CLASSES, 2.8))
    for k, cls in enumerate(CLASSES):
        layout = sample_ellipse_layout(cls)
        layout = apply_affine(layout, random_affine(rng))
        layout = add_ambiguity(layout, ambiguity, rng)
        feats, mask = render_grid(layout, grid_size)
        draw_grid(axes[k], layout, grid_size, mask,
                  title=f"{cls}  (n_cells={int(mask.sum())})")
    fig.suptitle(
        f"Example layouts at ambiguity={ambiguity:.2f}",
        fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "examples_per_class.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_ambiguity_grid(outdir, grid_size, seed, cls="sheep"):
    """Same class at increasing ambiguity, fixed affine seed."""
    levels = [0.0, 0.4, 0.8, 1.2]
    fig, axes = plt.subplots(1, len(levels), figsize=(2.6 * len(levels), 2.8))
    base_rng = np.random.default_rng(seed)
    aff = random_affine(base_rng)
    for k, amb in enumerate(levels):
        rng = np.random.default_rng(seed + 100 + k)
        layout = sample_ellipse_layout(cls)
        layout = apply_affine(layout, aff)
        layout = add_ambiguity(layout, amb, rng)
        _, mask = render_grid(layout, grid_size)
        draw_grid(axes[k], layout, grid_size, mask,
                  title=f"ambiguity={amb}")
    fig.suptitle(f"'{cls}' across ambiguity levels (same affine)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    path = os.path.join(outdir, "ambiguity_grid.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ----------------------------------------------------------------------
# Training-time visualizations
# ----------------------------------------------------------------------

def plot_training_curves(outdir, history):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5))
    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], label="train", color="tab:blue")
    ax.plot(epochs, history["val_loss"], label="val", color="tab:orange")
    ax.set_xlabel("epoch"); ax.set_ylabel("cross-entropy loss")
    ax.set_title("loss")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, [a * 100 for a in history["train_acc"]],
            label="train", color="tab:blue")
    ax.plot(epochs, [a * 100 for a in history["val_acc"]],
            label="val (T=2)", color="tab:orange")
    ax.plot(epochs, [a * 100 for a in history["val_acc_t0"]],
            label="val (T=0)", color="tab:gray", linestyle="--")
    ax.plot(epochs, [a * 100 for a in history["val_acc_t3"]],
            label="val (T=3)", color="tab:green", linestyle=":")
    ax.axhline(100.0 / N_CLASSES, color="red", linestyle="-",
               alpha=0.4, label=f"chance = {100.0/N_CLASSES:.0f}%")
    ax.set_xlabel("epoch"); ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("classification accuracy")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    path = os.path.join(outdir, "training_curves.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_confusion(outdir, X_va, M_va, y_va, params, n_iters):
    logits = forward(X_va, M_va, params, n_iters=n_iters)
    pred = logits.argmax(-1)
    cm = np.zeros((N_CLASSES, N_CLASSES), dtype=np.int32)
    for t, p in zip(y_va, pred):
        cm[t, p] += 1
    cm_n = cm / np.maximum(cm.sum(axis=1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    im = ax.imshow(cm_n, vmin=0, vmax=1, cmap="Blues")
    fig.colorbar(im, fraction=0.045)
    ax.set_xticks(range(N_CLASSES))
    ax.set_yticks(range(N_CLASSES))
    ax.set_xticklabels(CLASSES, rotation=30)
    ax.set_yticklabels(CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title(f"validation confusion matrix (T={n_iters})")
    for i in range(N_CLASSES):
        for j in range(N_CLASSES):
            ax.text(j, i, f"{cm_n[i, j]:.2f}",
                    ha="center", va="center",
                    color="white" if cm_n[i, j] > 0.5 else "black",
                    fontsize=9)
    fig.tight_layout()
    path = os.path.join(outdir, "confusion_matrix.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def plot_iters_ablation(outdir, X_va, M_va, y_va, params, max_iters=6):
    accs = []
    Ts = list(range(0, max_iters + 1))
    for t in Ts:
        accs.append(accuracy(X_va, M_va, y_va, params, n_iters=t) * 100)
    fig, ax = plt.subplots(figsize=(5, 3.4))
    ax.plot(Ts, accs, "o-", color="tab:purple")
    ax.axhline(100.0 / N_CLASSES, color="red", linestyle="--",
               alpha=0.4, label="chance")
    ax.set_xlabel("attention iterations T at inference")
    ax.set_ylabel("val accuracy (%)")
    ax.set_title("how many GLOM iterations help?")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    for t, a in zip(Ts, accs):
        ax.annotate(f"{a:.1f}", (t, a), textcoords="offset points",
                    xytext=(0, 7), ha="center", fontsize=8)
    fig.tight_layout()
    path = os.path.join(outdir, "iters_ablation.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


# ----------------------------------------------------------------------
# Island formation
# ----------------------------------------------------------------------

def plot_islands(outdir, params, ambiguity, grid_size, seed,
                 n_iters=4, alpha=0.5):
    """For each class, show an example grid + the per-iteration similarity matrix.

    Restricts the heatmap to occupied cells so the islands are visible.
    """
    rng_master = np.random.default_rng(seed + 7)
    n_panels = 2 + n_iters     # grid + sim_t=0..n_iters
    fig, axes = plt.subplots(N_CLASSES, n_panels,
                             figsize=(2.4 * n_panels, 2.4 * N_CLASSES))
    for c, cls in enumerate(CLASSES):
        rng = np.random.default_rng(seed + 31 + c)
        layout = sample_ellipse_layout(cls)
        layout = apply_affine(layout, random_affine(rng))
        layout = add_ambiguity(layout, ambiguity, rng)
        feats, mask = render_grid(layout, grid_size)
        sims, _ = visualize_islands(feats, mask, params,
                                    n_iters=n_iters, alpha=alpha)

        # Panel 1: image
        draw_grid(axes[c, 0], layout, grid_size, mask, title=f"{cls}")

        # Panels 2..: sim heatmaps (occupied-only sub-matrix)
        occ = np.where(mask > 0)[0]
        for t, sim in enumerate(sims):
            ax = axes[c, 1 + t]
            sub = sim[np.ix_(occ, occ)]
            im = ax.imshow(sub, vmin=-1, vmax=1, cmap="RdBu_r")
            ax.set_title(f"sim t={t}", fontsize=9)
            ax.set_xticks(range(len(occ)))
            ax.set_yticks(range(len(occ)))
            ax.set_xticklabels([str(i) for i in range(len(occ))], fontsize=7)
            ax.set_yticklabels([str(i) for i in range(len(occ))], fontsize=7)

    fig.suptitle("Island formation: pairwise cosine similarity of "
                 "occupied-cell embeddings across attention iterations",
                 y=1.005, fontsize=11)
    fig.tight_layout()
    path = os.path.join(outdir, "islands.png")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {path}")


def measure_island_quality(params, ambiguity, grid_size, seed,
                           n_iters=3, alpha=0.5, n_samples=200):
    """Average pairwise cosine sim over OCCUPIED cells, t=0 vs t=n_iters.

    A higher value at t=n_iters than at t=0 means attention is pulling
    occupied-cell embeddings together — i.e., islands are forming.
    """
    rng = np.random.default_rng(seed + 991)
    sim0_avg = 0.0
    simT_avg = 0.0
    n_used = 0
    for _ in range(n_samples):
        cls = int(rng.integers(N_CLASSES))
        layout = sample_ellipse_layout(CLASSES[cls])
        layout = apply_affine(layout, random_affine(rng))
        layout = add_ambiguity(layout, ambiguity, rng)
        feats, mask = render_grid(layout, grid_size)
        if mask.sum() < 2:
            continue
        sims, _ = visualize_islands(feats, mask, params, n_iters=n_iters,
                                    alpha=alpha)
        occ = np.where(mask > 0)[0]
        sub0 = sims[0][np.ix_(occ, occ)]
        subT = sims[-1][np.ix_(occ, occ)]
        # mean off-diagonal
        off = np.ones_like(sub0) - np.eye(len(occ))
        sim0_avg += (sub0 * off).sum() / off.sum()
        simT_avg += (subT * off).sum() / off.sum()
        n_used += 1
    return sim0_avg / n_used, simT_avg / n_used


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ambiguity", type=float, default=0.4)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--n-iters", type=int, default=2)
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-val", type=int, default=500)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--outdir", default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # static dataset visualizations (no model needed)
    plot_examples_per_class(args.outdir, args.grid_size, args.ambiguity,
                            args.seed)
    plot_ambiguity_grid(args.outdir, args.grid_size, args.seed)

    # train
    rng = np.random.default_rng(args.seed)
    print(f"\n# training (ambiguity={args.ambiguity}, T={args.n_iters})")
    X_tr, M_tr, y_tr = generate_dataset(args.n_train, args.grid_size,
                                        args.ambiguity, rng)
    X_va, M_va, y_va = generate_dataset(args.n_val, args.grid_size,
                                        args.ambiguity, rng)
    params, history = train(
        X_tr, M_tr, y_tr, X_va, M_va, y_va,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_iters=args.n_iters, alpha=args.alpha, seed=args.seed,
        verbose=True,
    )

    # post-training visualizations
    plot_training_curves(args.outdir, history)
    plot_confusion(args.outdir, X_va, M_va, y_va, params, args.n_iters)
    plot_iters_ablation(args.outdir, X_va, M_va, y_va, params)
    plot_islands(args.outdir, params, args.ambiguity, args.grid_size,
                 args.seed, n_iters=4, alpha=args.alpha)

    sim0, simT = measure_island_quality(params, args.ambiguity,
                                        args.grid_size, args.seed)
    print(f"\nIsland quality (mean off-diag occupied-cell cosine sim):")
    print(f"  t=0 (no refinement):     {sim0:+.3f}")
    print(f"  t=3 (after refinement):  {simT:+.3f}")
    print(f"  delta:                   {simT - sim0:+.3f}")

    print(f"\nFinal val accuracies:")
    print(f"  T=0: {history['val_acc_t0'][-1]*100:.1f}%")
    print(f"  T=2: {history['val_acc'][-1]*100:.1f}%")
    print(f"  T=3: {history['val_acc_t3'][-1]*100:.1f}%")


if __name__ == "__main__":
    main()
