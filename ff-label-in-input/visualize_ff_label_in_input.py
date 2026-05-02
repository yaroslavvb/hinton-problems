"""
Static visualisations for the trained FF supervised-MNIST network.

Outputs (in `viz/`):
  example_images.png       4 sample images with the label encoded in the
                           first 10 pixels (top row).
  goodness_heatmap.png     For 6 test images: bar chart of summed goodness
                           across layers for each candidate label 0..9.
                           True label highlighted; argmax marked.
  training_curves.png      Per-layer FF loss + per-layer mean goodness for
                           pos / neg + train/test accuracy across epochs.
  weights_layer0.png       Random sample of 64 receptive fields from
                           layer 0 (784 -> 500), reshaped to 28x28.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from ff_label_in_input import (
    load_mnist, build_ff_mlp, train, TrainConfig, FFLayer, FFModel,
    encode_label_in_pixels, encode_label_in_pixels_batch,
    goodness_per_layer, predict_by_goodness_batch,
)


# ---------------------------------------------------------------------------
# Loading a saved run (from --save model.npz)
# ---------------------------------------------------------------------------

def load_saved_model(path: str) -> tuple[FFModel, dict]:
    npz = np.load(path)
    layer_sizes = tuple(int(x) for x in npz["layer_sizes"])
    threshold = float(npz["threshold"])
    seed = int(npz["seed"])
    model = build_ff_mlp(layer_sizes=layer_sizes, threshold=threshold, seed=seed)
    for i, layer in enumerate(model.layers):
        layer.W = npz[f"layer{i}_W"].astype(np.float32)
        layer.b = npz[f"layer{i}_b"].astype(np.float32)
    history = {
        "epoch": list(range(1, len(npz["history_test_acc"]) + 1)),
        "test_acc": list(npz["history_test_acc"]),
        "train_acc": list(npz["history_train_acc"]),
        "loss_per_layer": [list(row) for row in npz["history_loss_per_layer"]],
        "g_pos_per_layer": [list(row) for row in npz["history_g_pos"]],
        "g_neg_per_layer": [list(row) for row in npz["history_g_neg"]],
        "wallclock": list(npz["history_wallclock"]),
    }
    return model, history


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_example_images(images: np.ndarray, labels: np.ndarray, out_path: str,
                        n_classes: int = 10) -> None:
    """Show 4 images with one-hot label encoded in the first 10 pixels."""
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), dpi=120)
    for ax, img, y in zip(axes, images[:4], labels[:4]):
        encoded = encode_label_in_pixels(img, int(y), n_classes=n_classes)
        ax.imshow(encoded, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
        ax.axhline(0.5, color="cyan", linewidth=0.6, alpha=0.7)
        ax.axvline(n_classes - 0.5, color="cyan", linewidth=0.6, alpha=0.7)
        ax.add_patch(plt.Rectangle((-0.5, -0.5), n_classes, 1,
                                   facecolor="none", edgecolor="cyan",
                                   linewidth=1.2))
        ax.set_title(f"label = {y}", fontsize=10)
        ax.axis("off")
    fig.suptitle("Label encoded as one-hot in first 10 pixels (top row)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_goodness_heatmap(model: FFModel, images: np.ndarray, labels: np.ndarray,
                          out_path: str, n_classes: int = 10,
                          n_examples: int = 6, skip_first: bool = False) -> None:
    """For each example image, compute summed goodness for each candidate
    label and plot as a bar chart, with the true label and the prediction
    highlighted."""
    images = images[:n_examples]
    labels = labels[:n_examples]
    flat = images.reshape(images.shape[0], -1)
    expanded = np.repeat(flat, n_classes, axis=0).copy()
    label_vec = np.tile(np.arange(n_classes), images.shape[0])
    expanded[:, :n_classes] = 0.0
    expanded[np.arange(expanded.shape[0]), label_vec] = 1.0
    g = goodness_per_layer(model, expanded, skip_first=skip_first)
    summed = g.sum(axis=1).reshape(images.shape[0], n_classes)

    fig, axes = plt.subplots(2, n_examples, figsize=(2.0 * n_examples, 4.5),
                             dpi=120)
    for col in range(n_examples):
        ax_img = axes[0, col]
        ax_img.imshow(images[col], cmap="gray", vmin=0, vmax=1,
                      interpolation="nearest")
        ax_img.set_title(f"true: {labels[col]}", fontsize=10)
        ax_img.axis("off")

        ax_bar = axes[1, col]
        scores = summed[col]
        # Normalise per-example for visual comparability.
        s = scores - scores.min()
        s_norm = s / (s.max() + 1e-9)
        pred = int(scores.argmax())
        colors = ["#1f77b4"] * n_classes
        colors[int(labels[col])] = "#2ca02c"
        if pred != int(labels[col]):
            colors[pred] = "#d62728"
        ax_bar.bar(range(n_classes), s_norm, color=colors)
        ax_bar.set_xticks(range(n_classes))
        ax_bar.set_xticklabels([str(i) for i in range(n_classes)], fontsize=8)
        ax_bar.set_ylim(0, 1.05)
        ax_bar.set_yticks([])
        if col == 0:
            ax_bar.set_ylabel("summed\ngoodness\n(normalised)", fontsize=8)
        ax_bar.set_title(f"pred: {pred}",
                         fontsize=9,
                         color=("#2ca02c" if pred == int(labels[col]) else "#d62728"))
    fig.suptitle("Goodness for each candidate label  (green = true, red = wrong prediction)",
                 fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history: dict, out_path: str) -> None:
    epochs = history["epoch"]
    n_layers = len(history["loss_per_layer"])
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5), dpi=120)

    ax = axes[0, 0]
    for L in range(n_layers):
        ax.plot(epochs, history["loss_per_layer"][L],
                label=f"layer {L}", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("FF loss")
    ax.set_title("Per-layer FF loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[0, 1]
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5", "#ffbb78"]
    for L in range(n_layers):
        ax.plot(epochs, history["g_pos_per_layer"][L],
                color=colors_pos[L % len(colors_pos)],
                label=f"L{L} pos", linewidth=1.4)
        ax.plot(epochs, history["g_neg_per_layer"][L],
                color=colors_neg[L % len(colors_neg)],
                label=f"L{L} neg", linewidth=1.4, linestyle="--")
    ax.axhline(2.0, color="black", linewidth=0.7, linestyle=":",
               label=r"$\theta$ = 2.0")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"goodness  $\langle h^2 \rangle$")
    ax.set_title("Per-layer goodness (positive vs negative)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    ax = axes[1, 0]
    ax.plot(epochs, np.array(history["train_acc"]) * 100,
            color="#1f77b4", label="train", linewidth=1.5)
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#d62728", label="test", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Accuracy on the eval subset")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)

    ax = axes[1, 1]
    ax.plot(epochs, history["wallclock"], color="#7f7f7f", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("wallclock (s)")
    ax.set_title("Cumulative training time")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"FF supervised MNIST  —  test={history['test_acc'][-1]*100:.2f}%, "
        f"train={history['train_acc'][-1]*100:.2f}%",
        fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_layer0_weights(model: FFModel, out_path: str,
                        n_show: int = 64, seed: int = 0) -> None:
    """Random sample of layer-0 receptive fields, reshaped to 28x28."""
    W = model.layers[0].W  # (784, n_hidden)
    n_hidden = W.shape[1]
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n_hidden)[:n_show]

    n_cols = 8
    n_rows = n_show // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(n_cols, n_rows),
                             dpi=140)
    vmax = float(np.percentile(np.abs(W), 99))
    for i, ax in enumerate(axes.flat):
        if i >= n_show:
            ax.axis("off")
            continue
        rf = W[:, idx[i]].reshape(28, 28)
        ax.imshow(rf, cmap="seismic", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        ax.axis("off")
    fig.suptitle(f"Layer-0 receptive fields  ({n_show} of {n_hidden}, "
                 f"red = +, blue = −)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model.npz",
                   help="Path to a saved run (.npz). If missing, train from scratch.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--layer-sizes", type=str, default="784,500,500")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading MNIST...")
    data = load_mnist()
    train_x, train_y, test_x, test_y = data

    if args.model and os.path.exists(args.model):
        print(f"Loading saved model {args.model}")
        model, history = load_saved_model(args.model)
    else:
        layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
        cfg = TrainConfig(n_epochs=args.n_epochs, batch_size=128, lr=args.lr,
                          threshold=2.0, layer_sizes=layer_sizes,
                          seed=args.seed, eval_subset=2000)
        print(f"Training from scratch ({args.n_epochs} epochs)...")
        model = build_ff_mlp(layer_sizes=layer_sizes, threshold=2.0,
                             seed=args.seed)
        history = train(model, data, cfg, verbose=True)

    print("Generating visualisations...")
    plot_example_images(test_x, test_y,
                        os.path.join(args.outdir, "example_images.png"))
    plot_goodness_heatmap(model, test_x, test_y,
                          os.path.join(args.outdir, "goodness_heatmap.png"))
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_layer0_weights(model,
                        os.path.join(args.outdir, "weights_layer0.png"))


if __name__ == "__main__":
    main()
