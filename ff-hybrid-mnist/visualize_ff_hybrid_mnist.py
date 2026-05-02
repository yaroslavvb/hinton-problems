"""
Static visualizations for the trained FF-hybrid-MNIST model.

Outputs (written to `viz/`):

    hybrid_examples.png        -- a 4 x 6 grid: rows = (digit_a, digit_b,
                                  random mask, hybrid). Shows the negative
                                  generation procedure.
    goodness_distributions.png -- per-layer histogram of mean-squared
                                  goodness for positives vs hybrids on
                                  the held-out test set.
    classifier_curves.png      -- softmax-head training/test error and
                                  per-layer FF training loss.
    weights_layer1.png         -- 4 x 4 grid of layer-1 receptive fields,
                                  reshaped to 28 x 28.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ff_hybrid_mnist import (
    build_ff_mlp,
    features_top_k,
    fit_softmax_on_top_layers,
    forward_all_layers,
    goodness,
    l2_normalize,
    load_mnist,
    make_hybrid_image,
    make_random_mask,
    train_unsupervised,
)


def plot_hybrid_examples(mnist: dict, out_path: str,
                         rng: np.random.Generator,
                         n_examples: int = 6, n_blur: int = 6):
    fig, axes = plt.subplots(4, n_examples, figsize=(1.6 * n_examples, 6.4),
                             dpi=120)
    n_train = mnist["train_images"].shape[0]
    for c in range(n_examples):
        i, j = rng.choice(n_train, size=2, replace=False)
        a = mnist["train_images"][i]
        b = mnist["train_images"][j]
        mask = make_random_mask(a.shape, rng, n_blur=n_blur)
        hybrid = mask * a + (1.0 - mask) * b
        for r, (img, title) in enumerate([
                (a, f"digit a ({mnist['train_labels'][i]})"),
                (b, f"digit b ({mnist['train_labels'][j]})"),
                (mask, "mask"),
                (hybrid, "hybrid")]):
            ax = axes[r, c]
            cmap = "gray_r" if r != 2 else "gray"
            ax.imshow(img, cmap=cmap, vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if c == 0:
                ax.set_ylabel(title.split()[0], fontsize=10)
            if r == 0 or r == 1:
                ax.set_title(title, fontsize=8)
    fig.suptitle("FF negatives: hybrid images via smoothly-thresholded "
                 "random mask", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_goodness_distributions(layers: list, mnist: dict, out_path: str,
                                rng: np.random.Generator, n_samples: int = 1000,
                                n_blur: int = 6):
    """Per-layer histogram of mean(h^2) for real digits vs hybrid images."""
    n_test = mnist["test_images"].shape[0]
    idx = rng.choice(n_test, size=min(n_samples, n_test), replace=False)
    pos = mnist["test_images"][idx].reshape(len(idx), -1)
    idx_b = rng.choice(n_test, size=len(idx), replace=False)
    neg = np.empty_like(pos)
    a_imgs = mnist["test_images"][idx]
    b_imgs = mnist["test_images"][idx_b]
    for k in range(len(idx)):
        neg[k] = make_hybrid_image(a_imgs[k], b_imgs[k], rng, n_blur=n_blur).reshape(-1)

    pos_acts = forward_all_layers(layers, pos)
    neg_acts = forward_all_layers(layers, neg)

    L = len(layers)
    fig, axes = plt.subplots(1, L, figsize=(3.0 * L, 3.0), dpi=120)
    if L == 1:
        axes = [axes]
    for li, ax in enumerate(axes):
        g_pos = goodness(pos_acts[li])
        g_neg = goodness(neg_acts[li])
        lo = float(min(g_pos.min(), g_neg.min()))
        hi = float(max(g_pos.max(), g_neg.max()))
        bins = np.linspace(lo, hi, 40)
        ax.hist(g_pos, bins=bins, alpha=0.55, label="real digit",
                color="#1f77b4")
        ax.hist(g_neg, bins=bins, alpha=0.55, label="hybrid",
                color="#d62728")
        ax.axvline(2.0, color="black", linestyle="--", linewidth=0.8,
                   label="threshold")
        ax.set_title(f"Layer {li+1}  (out_dim={layers[li].out_dim})")
        ax.set_xlabel("goodness  mean(h$^2$)")
        ax.set_ylabel("count")
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
        sep = (g_pos.mean() - g_neg.mean()) / max(1e-6, (g_pos.std() + g_neg.std()) / 2)
        ax.text(0.02, 0.95, f"sep={sep:.2f}σ",
                transform=ax.transAxes, fontsize=8, va="top",
                bbox=dict(boxstyle="round", facecolor="white", alpha=0.7))
    fig.suptitle("Goodness distributions: real digits vs hybrid negatives",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_classifier_curves(ff_history: dict, sm_history: dict,
                           out_path: str):
    L = sum(1 for k in ff_history if k.startswith("layer") and k.endswith("_loss"))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)

    ax = axes[0]
    for li in range(L):
        ax.plot(ff_history["epoch"], ff_history[f"layer{li+1}_loss"],
                label=f"layer {li+1}", linewidth=1.5)
    ax.set_xlabel("FF epoch")
    ax.set_ylabel("loss   softplus($\\theta$ - g)")
    ax.set_title("Per-layer FF unsupervised loss")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    err_train = [(1 - a) * 100 for a in sm_history["train_acc"]]
    err_test = [(1 - a) * 100 for a in sm_history["test_acc"]]
    ax.plot(sm_history["epoch"], err_train, label="train error",
            color="#1f77b4")
    ax.plot(sm_history["epoch"], err_test, label="test error",
            color="#d62728")
    ax.axhline(1.37, color="green", linestyle="--", linewidth=0.8,
               label="paper (1.37%)")
    ax.set_xlabel("softmax epoch")
    ax.set_ylabel("error (%)")
    ax.set_title("Linear softmax on top-3 FF layers")
    ax.set_ylim(0, max(20.0, max(err_test[:1] + [10.0]) + 1))
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_layer1_receptive_fields(layers: list, out_path: str,
                                 n_show: int = 16,
                                 rng: np.random.Generator | None = None):
    if rng is None:
        rng = np.random.default_rng(0)
    W = layers[0].W  # (784, hidden)
    sums = (W ** 2).sum(axis=0)
    top = np.argsort(-sums)[:n_show]
    fig, axes = plt.subplots(4, 4, figsize=(6, 6), dpi=120)
    for k, ax in enumerate(axes.flatten()):
        if k >= len(top):
            ax.axis("off")
            continue
        rf = W[:, top[k]].reshape(28, 28)
        m = max(abs(rf.max()), abs(rf.min()), 1e-3)
        ax.imshow(rf, cmap="seismic", vmin=-m, vmax=m)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"unit {top[k]}", fontsize=8)
    fig.suptitle("Layer-1 receptive fields (top by ||W_:,j||)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--layer-sizes", type=str,
                   default="784,1000,1000,1000,1000")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.03)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--n-blur", type=int, default=6)
    p.add_argument("--n-train", type=int, default=0)
    p.add_argument("--softmax-epochs", type=int, default=30)
    p.add_argument("--top-k", type=int, default=3)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print("loading MNIST...")
    mnist = load_mnist()

    if args.n_train > 0:
        idx = rng.permutation(mnist["train_images"].shape[0])[:args.n_train]
        mnist["train_images"] = mnist["train_images"][idx]
        mnist["train_labels"] = mnist["train_labels"][idx]

    layers = build_ff_mlp(layer_sizes, rng)
    print(f"training FF for {args.n_epochs} epochs...")
    ff_hist = train_unsupervised(layers, mnist["train_images"],
                                 n_epochs=args.n_epochs, lr=args.lr,
                                 batch_size=args.batch_size,
                                 threshold=args.threshold,
                                 rng=rng, n_blur=args.n_blur, verbose=True)
    print("fitting softmax head...")
    sm = fit_softmax_on_top_layers(
        layers, mnist, top_k=args.top_k, n_epochs=args.softmax_epochs,
        rng=np.random.default_rng(args.seed + 1), verbose=True)

    print("\nrendering figures...")
    plot_hybrid_examples(mnist, os.path.join(args.outdir,
                         "hybrid_examples.png"), rng, n_blur=args.n_blur)
    plot_goodness_distributions(layers, mnist, os.path.join(args.outdir,
                         "goodness_distributions.png"), rng,
                         n_blur=args.n_blur)
    plot_classifier_curves(ff_hist, sm["history"],
                           os.path.join(args.outdir,
                                        "classifier_curves.png"))
    plot_layer1_receptive_fields(layers, os.path.join(args.outdir,
                                  "weights_layer1.png"))
    print(f"\nfinal test error: {(1.0 - sm['history']['test_acc'][-1]) * 100:.2f}%")


if __name__ == "__main__":
    main()
