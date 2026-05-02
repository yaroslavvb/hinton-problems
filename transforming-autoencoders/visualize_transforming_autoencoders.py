"""
Static visualizations for the trained transforming auto-encoder.

Outputs (in `viz/`):
  example_pairs.png     - input image, transformed image, dxdy label, reconstruction
  capsule_presence.png  - heatmap of presence p_c over a small validation set
  prediction_scatter.png - predicted vs true (dx, dy) on validation pairs
  reconstructions.png   - side-by-side targets vs reconstructions for several pairs
  training_curves.png   - per-epoch loss + R²(dx) + R²(dy)
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from transforming_autoencoders import (
    TransformingAutoencoder, train, load_mnist, translate, crop_center,
    _r_squared,
)


PRESENCE_CMAP = "viridis"


def _val_set(model: TransformingAutoencoder, n: int = 256, seed: int = 7):
    """Build a validation set of (image1, image2, dxdy)."""
    images = load_mnist("train")
    rng = np.random.default_rng(seed)
    idx = rng.permutation(images.shape[0])[:n]
    base = images[idx]
    t_in = rng.integers(-5, 6, size=(n, 2))
    dxdy = rng.integers(-5, 6, size=(n, 2)).astype(np.float32)
    img1 = np.stack([translate(base[i], int(t_in[i, 0]), int(t_in[i, 1]))
                     for i in range(n)])
    img2 = np.stack([translate(base[i],
                                int(t_in[i, 0] + dxdy[i, 0]),
                                int(t_in[i, 1] + dxdy[i, 1]))
                     for i in range(n)])
    return img1, img2, dxdy


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)

    ax = axes[0]
    ax.plot(history["epoch"], history["loss"], color="#1f77b4", label="train MSE")
    ax.plot(history["epoch"], history["val_mse"], color="#ff7f0e",
            linestyle="--", label="val MSE")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Reconstruction loss")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["dx_r2"], color="#2ca02c", label="R$^2$(dx)")
    ax.plot(history["epoch"], history["dy_r2"], color="#d62728", label="R$^2$(dy)")
    ax.axhline(0.0, color="gray", linewidth=0.5, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("R$^2$ on held-out pairs")
    ax.set_title("Disentanglement quality (top-3 capsules)")
    ax.set_ylim(-0.05, 1.0)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_example_pairs(model: TransformingAutoencoder, out_path: str,
                       n_examples: int = 6, seed: int = 7):
    img1, img2, dxdy = _val_set(model, n=n_examples, seed=seed)
    x1 = img1.reshape(n_examples, -1)
    recon, _ = model.forward(x1, dxdy)
    recon22 = recon.reshape(n_examples, 22, 22)
    target22 = crop_center(img2, 22)

    fig, axes = plt.subplots(4, n_examples, figsize=(2.0 * n_examples, 7.5), dpi=110)
    for i in range(n_examples):
        axes[0, i].imshow(img1[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"input\n(t_in random)", fontsize=9)
        axes[1, i].imshow(img2[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"target\n(dx, dy) = ({int(dxdy[i,0])}, {int(dxdy[i,1])})",
                              fontsize=9)
        axes[2, i].imshow(target22[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title("target 22x22", fontsize=9)
        axes[3, i].imshow(np.clip(recon22[i], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[3, i].set_title("reconstruction", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("Transforming auto-encoder: input -> (dx, dy) -> output",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_capsule_presence(model: TransformingAutoencoder, out_path: str,
                          n_examples: int = 24, seed: int = 7):
    img1, _, _ = _val_set(model, n=n_examples, seed=seed)
    x = img1.reshape(n_examples, -1)
    p, xy, _ = model._recognize(x)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5),
                              dpi=120, gridspec_kw={"width_ratios": [1.4, 1.0]})

    im = axes[0].imshow(p, aspect="auto", cmap=PRESENCE_CMAP, vmin=0, vmax=1)
    axes[0].set_xlabel("capsule index")
    axes[0].set_ylabel("validation example")
    axes[0].set_title("Capsule presence  $p_c$")
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # active-capsule histogram: how many capsules have p > 0.1 per input
    n_active = (p > 0.1).sum(axis=1)
    axes[1].hist(n_active, bins=range(0, model.n_capsules + 2), color="#1f77b4",
                 edgecolor="black", linewidth=0.4, align="left")
    axes[1].set_xlabel("# capsules with p > 0.1")
    axes[1].set_ylabel("# examples")
    axes[1].set_title(f"Active-capsule count\n(median = {int(np.median(n_active))})")
    axes[1].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_prediction_scatter(model: TransformingAutoencoder, out_path: str,
                            n_examples: int = 256, seed: int = 7):
    img1, img2, dxdy = _val_set(model, n=n_examples, seed=seed)
    pred = model.predict_transformation(img1, img2, top_k=3)
    r2_dx = _r_squared(dxdy[:, 0], pred[:, 0])
    r2_dy = _r_squared(dxdy[:, 1], pred[:, 1])

    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=120)
    for ax, axis_name, true_vals, pred_vals, r2 in [
        (axes[0], "dx", dxdy[:, 0], pred[:, 0], r2_dx),
        (axes[1], "dy", dxdy[:, 1], pred[:, 1], r2_dy),
    ]:
        # Add small jitter so integer grid points don't all stack.
        jitter = 0.18 * np.random.default_rng(0).standard_normal(true_vals.shape)
        ax.plot([-6, 6], [-6, 6], "k--", linewidth=0.7, alpha=0.5,
                label="$y = x$")
        ax.scatter(true_vals + jitter, pred_vals, alpha=0.4, s=18,
                   color="#1f77b4")
        ax.set_xlabel(f"true {axis_name}")
        ax.set_ylabel(f"predicted {axis_name}")
        ax.set_title(f"{axis_name}: R$^2$ = {r2:.3f}")
        ax.set_xlim(-6, 6)
        ax.set_ylim(min(-6, pred_vals.min() - 0.5),
                    max(6, pred_vals.max() + 0.5))
        ax.grid(alpha=0.3)
        ax.legend(loc="upper left", fontsize=9)
    fig.suptitle("Predicted (dx, dy) from a pair of images (top-3 capsule weighting)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(model: TransformingAutoencoder, out_path: str,
                         n_examples: int = 8, seed: int = 11):
    img1, img2, dxdy = _val_set(model, n=n_examples, seed=seed)
    x = img1.reshape(n_examples, -1)
    recon, _ = model.forward(x, dxdy)
    recon22 = np.clip(recon, 0, 1).reshape(n_examples, 22, 22)
    target22 = crop_center(img2, 22)
    mse = float(np.mean((recon22 - target22) ** 2))

    fig, axes = plt.subplots(2, n_examples, figsize=(1.6 * n_examples, 3.6), dpi=120)
    for i in range(n_examples):
        axes[0, i].imshow(target22[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"({int(dxdy[i,0])},{int(dxdy[i,1])})", fontsize=9)
        axes[1, i].imshow(recon22[i], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_ylabel("target")
    axes[1, 0].set_ylabel("recon")
    for ax in axes.flat:
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle(f"22x22 reconstructions  (mean MSE on this set = {mse:.4f})",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--n-capsules", type=int, default=30)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_epochs} epochs (seed={args.seed})...")
    model, history = train(n_epochs=args.n_epochs,
                           steps_per_epoch=args.steps_per_epoch,
                           n_capsules=args.n_capsules,
                           lr=args.lr,
                           seed=args.seed,
                           verbose=True)

    print(f"\nFinal val MSE: {history['val_mse'][-1]:.5f}")
    print(f"Final R2(dx):  {history['dx_r2'][-1]:.3f}")
    print(f"Final R2(dy):  {history['dy_r2'][-1]:.3f}")

    print("\nWriting visualizations...")
    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_example_pairs(model, os.path.join(args.outdir, "example_pairs.png"))
    plot_capsule_presence(model, os.path.join(args.outdir, "capsule_presence.png"))
    plot_prediction_scatter(model, os.path.join(args.outdir, "prediction_scatter.png"))
    plot_reconstructions(model, os.path.join(args.outdir, "reconstructions.png"))


if __name__ == "__main__":
    main()
