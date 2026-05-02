"""
Static visualizations for the trained constellations model.

Outputs (in `viz/`):
  training_curves.png       chamfer loss + recovery accuracy across epochs
  example_constellations.png  3x4 grid: ground-truth template colors,
                              predicted (capsule) colors, and decoded shapes.
  recovery_heatmap.png      per-point true-label x predicted-label confusion
                            matrix on a held-out batch.
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from constellations import (TEMPLATES, train, make_dataset,
                            part_capsule_recovery_accuracy,
                            _all_permutations)


# Deterministic colors -- one per template.
CAPSULE_COLORS = ["#d62728", "#1f77b4", "#2ca02c"]
TEMPLATE_NAMES = ["square (4)", "triangle+extra (4)", "triangle (3)"]


def _best_permutation(model, points: np.ndarray, labels: np.ndarray
                      ) -> np.ndarray:
    """Return a per-example permutation (B, K) such that
    relabelled = perm[b, pred_caps[b, n]] best matches labels[b].
    """
    decoded, point_capsule, _, _ = model.forward(points)
    diff = points[:, :, None, :] - decoded[:, None, :, :]
    sq = (diff ** 2).sum(axis=-1)
    nn = np.argmin(sq, axis=2)
    pred_caps = point_capsule[nn]
    K = model.K
    perms = _all_permutations(K)
    B = pred_caps.shape[0]
    best_perm = np.zeros((B, K), dtype=np.int32)
    best_score = -np.ones(B, dtype=np.float32)
    for perm in perms:
        perm_arr = np.array(perm, dtype=np.int32)
        relabelled = perm_arr[pred_caps]
        score = (relabelled == labels).mean(axis=1)
        better = score > best_score
        best_score = np.where(better, score, best_score)
        best_perm[better] = perm_arr
    return best_perm, pred_caps, decoded, point_capsule


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=120)
    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["loss"], label="train", color="#1f77b4")
    ax.plot(epochs, history["val_loss"], label="val", color="#ff7f0e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("symmetric chamfer")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    ax.set_title("Reconstruction loss")

    ax = axes[1]
    ax.plot(epochs, np.array(history["val_recovery"]) * 100, color="#2ca02c")
    ax.axhline(100 * 4 / 11, color="gray", linestyle="--", linewidth=0.7,
               label="chance (4/11 = 36.4%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("part-capsule recovery (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title("Per-point recovery (permutation-invariant)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_examples(model, points, labels, out_path: str, n_examples: int = 4):
    """3xN grid:
        row 0 -- input points colored by ground-truth template
        row 1 -- input points colored by best-matched capsule prediction
        row 2 -- decoded shapes (one color per capsule).
    """
    n_examples = min(n_examples, points.shape[0])
    pts = points[:n_examples]
    lbl = labels[:n_examples]
    perms, pred_caps, decoded, point_capsule = _best_permutation(model, pts, lbl)

    fig, axes = plt.subplots(3, n_examples, figsize=(3.0 * n_examples, 8.5),
                             dpi=120)
    if n_examples == 1:
        axes = axes[:, None]

    for b in range(n_examples):
        # Common axis range
        all_x = np.concatenate([pts[b, :, 0], decoded[b, :, 0]])
        all_y = np.concatenate([pts[b, :, 1], decoded[b, :, 1]])
        xmin, xmax = float(all_x.min()) - 0.5, float(all_x.max()) + 0.5
        ymin, ymax = float(all_y.min()) - 0.5, float(all_y.max()) + 0.5

        # Row 0: ground-truth coloring
        ax = axes[0, b]
        for k in range(3):
            mask = lbl[b] == k
            ax.scatter(pts[b, mask, 0], pts[b, mask, 1],
                       color=CAPSULE_COLORS[k], s=80, edgecolor="black",
                       linewidth=0.6, label=TEMPLATE_NAMES[k] if b == 0 else None)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"Example {b} -- ground truth", fontsize=10)
        if b == 0:
            ax.legend(loc="upper right", fontsize=7)

        # Row 1: predicted (after best permutation) coloring
        ax = axes[1, b]
        relabelled = perms[b][pred_caps[b]]              # (N,)
        per_point_correct = relabelled == lbl[b]
        n_correct = int(per_point_correct.sum())
        for k in range(3):
            mask = relabelled == k
            ax.scatter(pts[b, mask, 0], pts[b, mask, 1],
                       color=CAPSULE_COLORS[k], s=80, edgecolor="black",
                       linewidth=0.6)
        # Mark wrong points with an X
        for i in range(11):
            if not per_point_correct[i]:
                ax.scatter(pts[b, i, 0], pts[b, i, 1], color="black",
                           marker="x", s=70, linewidth=2.0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title(f"Predicted ({n_correct}/11 correct)", fontsize=10)

        # Row 2: decoded shapes
        ax = axes[2, b]
        for k in range(3):
            mask_dec = point_capsule == k
            ax.scatter(decoded[b, mask_dec, 0], decoded[b, mask_dec, 1],
                       color=CAPSULE_COLORS[k], s=70, marker="^",
                       edgecolor="black", linewidth=0.5)
            # Faint outline of the input cloud underneath
            ax.scatter(pts[b, :, 0], pts[b, :, 1], color="lightgray",
                       s=20, zorder=0)
        ax.set_xlim(xmin, xmax)
        ax.set_ylim(ymin, ymax)
        ax.set_aspect("equal")
        ax.grid(alpha=0.3)
        ax.set_title("Decoded (triangles) over input (gray)", fontsize=10)

    fig.suptitle("Constellations: input points -> per-template recovery",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recovery_heatmap(model, points, labels, out_path: str):
    """Confusion matrix between ground-truth template and best-matched
    predicted capsule, aggregated across the validation set.
    """
    perms, pred_caps, _, _ = _best_permutation(model, points, labels)
    K = model.K
    cm = np.zeros((K, K), dtype=np.int64)
    for b in range(points.shape[0]):
        relabelled = perms[b][pred_caps[b]]
        for n in range(points.shape[1]):
            cm[int(labels[b, n]), int(relabelled[n])] += 1

    cm_norm = cm / cm.sum(axis=1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.6, 4.0), dpi=140)
    im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(K))
    ax.set_yticks(range(K))
    ax.set_xticklabels(TEMPLATE_NAMES, rotation=20, ha="right")
    ax.set_yticklabels(TEMPLATE_NAMES)
    ax.set_xlabel("predicted template")
    ax.set_ylabel("true template")
    for i in range(K):
        for j in range(K):
            color = "white" if cm_norm[i, j] > 0.5 else "black"
            ax.text(j, i, f"{cm_norm[i, j]*100:.1f}%\n({cm[i, j]})",
                    ha="center", va="center", color=color, fontsize=9)
    ax.set_title(f"Recovery heatmap  "
                 f"(per-point acc = {(cm.diagonal().sum() / cm.sum())*100:.1f}%)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_epochs} epochs (seed={args.seed})...")
    model, history = train(n_epochs=args.n_epochs,
                           steps_per_epoch=args.steps_per_epoch,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           seed=args.seed,
                           verbose=False)
    print(f"  final recovery: {history['val_recovery'][-1]*100:.1f}%   "
          f"final val chamfer: {history['val_loss'][-1]:.3f}")

    # Held-out batch with a different RNG.
    eval_rng = np.random.default_rng(args.seed + 7)
    points, labels = make_dataset(256, eval_rng)
    final_acc = part_capsule_recovery_accuracy(model, (points, labels))
    print(f"  held-out recovery (256 ex): {final_acc*100:.1f}%")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_examples(model, points, labels,
                  os.path.join(args.outdir, "example_constellations.png"),
                  n_examples=4)
    plot_recovery_heatmap(model, points, labels,
                          os.path.join(args.outdir, "recovery_heatmap.png"))


if __name__ == "__main__":
    main()
