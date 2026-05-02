"""
Static visualizations for multi-level glimpse MNIST.

Outputs (in `viz/`):
  glimpse_overlay.png             One MNIST image with all 24 glimpse boxes
                                  drawn on top, numbered in visit order.
  training_curves.png             Loss + accuracy curves through training.
  fast_weights_evolution.png      A_t heatmap snapshots at every glimpse for
                                  one example. Shows the matrix building up
                                  outer-product traces and decaying.
  hidden_state_trace.png          h_t per glimpse (heatmap + norm).
  per_class_accuracy.png          Test accuracy bucketed by digit class.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

from multi_level_glimpse_mnist import (
    GlimpseFastWeightsRNN, build_glimpse_rnn_with_fast_weights,
    build_glimpse_inputs, generate_glimpse_sequence,
    train, evaluate, per_class_accuracy, load_mnist,
    GLIMPSE_OFFSETS, PATCH_SIZE, N_GLIMPSES, GLIMPSE_DIM, N_CLASSES,
)


def _ensure_viz_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_glimpse_overlay(image: np.ndarray, label: int, out_path: str) -> None:
    """Draw the 24 glimpse boxes overlaid on one MNIST image."""
    fig, ax = plt.subplots(figsize=(6.0, 6.0), dpi=130)
    ax.imshow(image, cmap="gray", interpolation="nearest")

    # Color: red for the 16 fine patches, blue for the 8 centre re-glimpses.
    for i, (r, c) in enumerate(GLIMPSE_OFFSETS):
        is_central = i >= 16
        edge = "#1565c0" if is_central else "#c62828"
        ls   = "--" if is_central else "-"
        # Slight jitter on the rectangle so overlapping boxes are distinguishable.
        jitter = 0.0 if not is_central else 0.18 * (i - 16)
        rect = mpatches.Rectangle(
            (c - 0.5 + jitter, r - 0.5 + jitter),
            PATCH_SIZE, PATCH_SIZE,
            linewidth=1.4, edgecolor=edge, facecolor="none",
            linestyle=ls, alpha=0.9)
        ax.add_patch(rect)
        # numeric label at top-left corner of the box
        ax.text(c - 0.4 + jitter, r - 0.4 + jitter, str(i),
                color=edge, fontsize=7, fontweight="bold",
                bbox=dict(facecolor="white", edgecolor="none",
                          pad=0.5, alpha=0.7))

    ax.set_xticks([0, 7, 14, 21, 27])
    ax.set_yticks([0, 7, 14, 21, 27])
    ax.grid(alpha=0.3)
    ax.set_title(
        f"24 hierarchical glimpses on a digit-{label} image\n"
        "red solid = 16 fine patches (4 coarse quadrants × 4 fine each)\n"
        "blue dashed = 8 centre re-glimpses",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
    steps = np.array(history["step"])
    test_steps = np.array(history["eval_test_step"])

    ax = axes[0]
    ax.plot(steps, history["train_loss"], color="#1f77b4", marker="o",
            markersize=3, label="train (mean over interval)")
    ax.plot(test_steps, history["test_loss"], color="#d62728", marker="s",
            markersize=4, label="test (full set)")
    ax.axhline(np.log(N_CLASSES), color="gray", linestyle=":",
               label=f"chance (log {N_CLASSES} = {np.log(N_CLASSES):.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, np.array(history["train_acc"]) * 100, color="#1f77b4",
            marker="o", markersize=3, label="train")
    ax.plot(test_steps, np.array(history["test_acc"]) * 100, color="#d62728",
            marker="s", markersize=4, label="test")
    ax.axhline(100.0 / N_CLASSES, color="gray", linestyle=":",
               label=f"chance ({100.0/N_CLASSES:.0f}%)")
    ax.set_xlabel("training step")
    ax.set_ylabel("classification accuracy (%)")
    ax.set_title("Accuracy")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Multi-level glimpse MNIST -- training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fast_weights_evolution(model: GlimpseFastWeightsRNN,
                                X_one: np.ndarray, label: int, pred: int,
                                out_path: str) -> None:
    """A_t heatmap snapshots at every glimpse for one image."""
    # X_one: (24, 73) -> add batch dim
    fwd = model.forward(X_one[None])
    A = fwd["A"][0]                                      # (T, H, H)
    T = A.shape[0]
    n_cols = T
    fig, axes = plt.subplots(1, n_cols, figsize=(0.85 * n_cols + 0.6, 1.4),
                             dpi=120)
    if n_cols == 1:
        axes = [axes]
    vmax = float(np.max(np.abs(A))) + 1e-6
    for t in range(T):
        ax = axes[t]
        ax.imshow(A[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.set_title(f"t={t}", fontsize=7)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Fast-weights matrix A_t per glimpse  "
        f"(label={label}, pred={pred})",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_trace(model: GlimpseFastWeightsRNN,
                      X_one: np.ndarray, label: int, pred: int,
                      out_path: str) -> None:
    fwd = model.forward(X_one[None])
    h = fwd["h"][0]                                      # (T+1, H)
    T = h.shape[0] - 1
    H = h.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(9.0, 5.5), dpi=120,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    im = ax.imshow(h[1:].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("glimpse index t")
    ax.set_ylabel("hidden unit")
    ax.set_xticks(range(T))
    ax.set_xticklabels([str(t) for t in range(T)], fontsize=7)
    ax.set_title(f"Hidden state h_t per glimpse  (label={label}, pred={pred})",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="h_t value", fraction=0.025, pad=0.01)

    ax = axes[1]
    norms = np.linalg.norm(h[1:], axis=1)
    ax.plot(range(T), norms, color="#3a7", marker="o", markersize=4)
    ax.set_xticks(range(T))
    ax.set_xticklabels([str(t) for t in range(T)], fontsize=7)
    ax.set_xlabel("glimpse index t")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden-state norm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_class_accuracy(per_class: np.ndarray, out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.6), dpi=130)
    xs = np.arange(N_CLASSES)
    bars = ax.bar(xs, per_class * 100, color="#3a7", edgecolor="black",
                  linewidth=0.4)
    for i, b in enumerate(bars):
        ax.text(i, per_class[i] * 100 + 1.0, f"{per_class[i]*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(100.0 / N_CLASSES, color="gray", linestyle=":",
               label=f"chance ({100.0/N_CLASSES:.0f}%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([str(c) for c in xs])
    ax.set_xlabel("digit class")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 110)
    ax.set_title("Per-class test accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Static visualizations for multi-level glimpse MNIST.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-hidden", type=int, default=128)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-train", type=int, default=20000,
                   help="0 = full 60k; smaller is faster but less accurate")
    p.add_argument("--out-dir", type=str, default="viz")
    args = p.parse_args()

    _ensure_viz_dir(args.out_dir)

    print(f"[viz] loading MNIST")
    train_x, train_y, test_x, test_y = load_mnist()
    if args.n_train and args.n_train < len(train_x):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_x))[:args.n_train]
        train_x = train_x[idx]
        train_y = train_y[idx]

    print(f"[viz] building glimpse inputs")
    train_X = build_glimpse_inputs(train_x)
    test_X  = build_glimpse_inputs(test_x)

    print(f"[viz] training  hidden={args.n_hidden}  epochs={args.n_epochs}  "
          f"n_train={len(train_x)}")
    model = build_glimpse_rnn_with_fast_weights(
        glimpse_dim=GLIMPSE_DIM, n_hidden=args.n_hidden, n_classes=N_CLASSES,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed)
    t0 = time.time()
    history = train(model, (train_X, train_y, test_X, test_y),
                    n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, grad_clip=5.0, eval_every=200,
                    seed=args.seed, verbose=False)
    print(f"[viz] training done in {time.time()-t0:.1f}s  "
          f"final test acc = {history['test_acc'][-1]*100:.2f}%")

    # Pick a clean test image for the per-step traces.
    rng = np.random.default_rng(args.seed + 99)
    pick = int(rng.integers(0, len(test_x)))
    img = test_x[pick]
    label = int(test_y[pick])
    X_one = build_glimpse_inputs(img[None])[0]            # (24, 73)
    pred = int(model.predict(X_one[None])[0])

    plot_glimpse_overlay(img, label,
                         os.path.join(args.out_dir, "glimpse_overlay.png"))
    print(f"[viz] wrote glimpse_overlay.png  (digit {label})")

    plot_training_curves(history,
                         os.path.join(args.out_dir, "training_curves.png"))
    print(f"[viz] wrote training_curves.png")

    plot_fast_weights_evolution(
        model, X_one, label, pred,
        os.path.join(args.out_dir, "fast_weights_evolution.png"))
    print(f"[viz] wrote fast_weights_evolution.png  (label={label}, pred={pred})")

    plot_hidden_trace(
        model, X_one, label, pred,
        os.path.join(args.out_dir, "hidden_state_trace.png"))
    print(f"[viz] wrote hidden_state_trace.png")

    per_class = per_class_accuracy(model, test_X, test_y,
                                   batch_size=args.batch_size)
    plot_per_class_accuracy(per_class,
                            os.path.join(args.out_dir, "per_class_accuracy.png"))
    print(f"[viz] wrote per_class_accuracy.png")
    print(f"[viz] all outputs in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
