"""
Static visualizations for the MultiMNIST CapsNet.

Outputs (in `viz/`):
  example_pairs.png       - 6 MultiMNIST inputs with both source digits visible
  capsule_activations.png - heatmap of digit-capsule norms across validation set
  reconstructions.png     - composite | target_a | recon_a | target_b | recon_b
  training_curves.png     - margin loss / recon loss / test accuracy across epochs
"""
from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from multimnist_capsnet import (
    CapsNet, train, predict_top2,
)


def plot_training_curves(history: dict, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4), dpi=120)

    ax = axes[0]
    ax.plot(history["epoch"], history["margin"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("margin loss")
    ax.set_title("Margin loss (train)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["recon"], color="#ff7f0e", label="recon (train)")
    ax2 = ax.twinx()
    ax2.plot(history["epoch"], history["test_recon_mse"], color="#d62728",
             linestyle="--", label="recon MSE (test)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("sum-of-squares (train)", color="#ff7f0e")
    ax2.set_ylabel("MSE (test)", color="#d62728")
    ax.set_title("Reconstruction loss")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(history["epoch"], history["test_acc"], color="#2ca02c", marker="o")
    ax.axhline(1.0 / 45, color="gray", linestyle=":", linewidth=0.8,
               label=f"chance ({1/45:.3f})")
    ax.axhline(0.8, color="black", linestyle=":", linewidth=0.6,
               label="target 0.8")
    ax.set_xlabel("epoch")
    ax.set_ylabel("two-digit set accuracy (test)")
    ax.set_ylim(0, 1.0)
    ax.set_title("Two-digit identification accuracy")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_example_pairs(test_data: dict, out_path: str, n_examples: int = 6):
    Xte = test_data["Xte"]
    Ate = test_data["Ate"]
    Bte = test_data["Bte"]
    Lte = test_data["Lte"]

    fig, axes = plt.subplots(3, n_examples, figsize=(2.0 * n_examples, 6.0), dpi=110)
    for i in range(n_examples):
        axes[0, i].imshow(Xte[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"composite\n({Lte[i,0]}, {Lte[i,1]})", fontsize=9)
        axes[1, i].imshow(Ate[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"digit {Lte[i,0]}", fontsize=9)
        axes[2, i].imshow(Bte[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].set_title(f"digit {Lte[i,1]}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("MultiMNIST: 36x36 canvas, two digits each shifted +/-4 pixels (IoU >= 0.8)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_capsule_activations(model: CapsNet, test_data: dict, out_path: str,
                             n_examples: int = 24):
    Xte = test_data["Xte"]
    Lte = test_data["Lte"]
    v, _ = model.forward(Xte[:n_examples])
    norms = np.sqrt((v * v).sum(axis=-1) + 1e-8)                  # (n, 10)

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=120,
                             gridspec_kw={"width_ratios": [1.4, 1.0]})
    im = axes[0].imshow(norms, aspect="auto", cmap="viridis", vmin=0, vmax=1)
    axes[0].set_xlabel("digit class (capsule index)")
    axes[0].set_ylabel("validation example")
    axes[0].set_title("DigitCaps activation norms ||v_k||")
    axes[0].set_xticks(range(10))
    fig.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)

    # Mark the two ground-truth labels per row
    for i in range(n_examples):
        for k in (Lte[i, 0], Lte[i, 1]):
            axes[0].add_patch(plt.Rectangle((k - 0.5, i - 0.5), 1, 1,
                                            fill=False, edgecolor="red",
                                            linewidth=0.7))

    # Right panel: top-2 vs ground-truth correctness
    top2, _ = predict_top2(v)
    sorted_labels = np.sort(Lte[:n_examples], axis=1)
    correct = np.all(top2 == sorted_labels, axis=1)
    n_match_one = np.array([
        len(set(top2[i]) & set(sorted_labels[i])) for i in range(n_examples)
    ])
    counts = np.bincount(n_match_one, minlength=3)
    axes[1].bar([0, 1, 2], counts, color=["#d62728", "#ff7f0e", "#2ca02c"],
                edgecolor="black", linewidth=0.5)
    axes[1].set_xticks([0, 1, 2])
    axes[1].set_xticklabels(["miss both", "got 1", "got both"])
    axes[1].set_ylabel(f"# of {n_examples} examples")
    axes[1].set_title(f"Top-2 match  (full set acc = {correct.mean():.3f})")
    axes[1].grid(alpha=0.3, axis="y")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(model: CapsNet, test_data: dict, out_path: str,
                         n_examples: int = 6):
    Xte = test_data["Xte"]
    Ate = test_data["Ate"]
    Bte = test_data["Bte"]
    Lte = test_data["Lte"]
    canvas = model.canvas

    v, _ = model.forward(Xte[:n_examples])
    # Reconstruct using ground-truth labels (as paper does for MultiMNIST viz)
    ma = np.zeros((n_examples, 10), dtype=np.float32)
    ma[np.arange(n_examples), Lte[:n_examples, 0]] = 1.0
    mb = np.zeros((n_examples, 10), dtype=np.float32)
    mb[np.arange(n_examples), Lte[:n_examples, 1]] = 1.0
    recon_a, _ = model.decode(v, ma)
    recon_b, _ = model.decode(v, mb)
    recon_a = recon_a.reshape(n_examples, canvas, canvas)
    recon_b = recon_b.reshape(n_examples, canvas, canvas)

    fig, axes = plt.subplots(5, n_examples, figsize=(2.0 * n_examples, 9.0), dpi=110)
    row_titles = ["input", "target a", "recon a", "target b", "recon b"]
    for i in range(n_examples):
        axes[0, i].imshow(Xte[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"({Lte[i,0]}, {Lte[i,1]})", fontsize=9)
        axes[1, i].imshow(Ate[i], cmap="gray", vmin=0, vmax=1)
        axes[2, i].imshow(np.clip(recon_a[i], 0, 1), cmap="gray", vmin=0, vmax=1)
        axes[3, i].imshow(Bte[i], cmap="gray", vmin=0, vmax=1)
        axes[4, i].imshow(np.clip(recon_b[i], 0, 1), cmap="gray", vmin=0, vmax=1)
    for r, label in enumerate(row_titles):
        axes[r, 0].set_ylabel(label, fontsize=10)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Per-digit reconstruction by masking the corresponding DigitCaps",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_epochs} epochs on {args.n_train} pairs (seed={args.seed})...")
    model, history, test_data = train(
        n_epochs=args.n_epochs, n_train=args.n_train, n_test=args.n_test,
        seed=args.seed, verbose=True,
    )
    print(f"\nFinal test acc: {history['test_acc'][-1]:.3f}")

    print("\nWriting visualizations...")
    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_example_pairs(test_data, os.path.join(args.outdir, "example_pairs.png"))
    plot_capsule_activations(model, test_data,
                             os.path.join(args.outdir, "capsule_activations.png"))
    plot_reconstructions(model, test_data,
                         os.path.join(args.outdir, "reconstructions.png"))


if __name__ == "__main__":
    main()
