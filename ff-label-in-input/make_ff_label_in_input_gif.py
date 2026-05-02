"""
Render an animated GIF showing the FF supervised-MNIST network learning to
separate positive (true label) from negative (wrong label) inputs.

Each frame combines:
  Top-left:  per-layer goodness over time (pos solid, neg dashed; threshold
             marked at 2.0).
  Top-right: bar chart of summed goodness across layers for each candidate
             label, on a fixed test image whose true label is highlighted.
  Bottom:    per-layer FF loss + test accuracy over time.

Usage:
    python3 make_ff_label_in_input_gif.py --epochs 30 --snapshot-every 1 --fps 6
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ff_label_in_input import (
    load_mnist, build_ff_mlp, train, TrainConfig, FFModel,
    encode_label_in_pixels, goodness_per_layer,
)


def render_frame(history: dict, model: FFModel,
                 demo_image: np.ndarray, demo_label: int,
                 epoch: int, total_epochs: int) -> Image.Image:
    n_layers = len(history["loss_per_layer"])
    epochs = history["epoch"]

    fig = plt.figure(figsize=(10, 5.4), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          width_ratios=[1.2, 0.8, 1.2],
                          hspace=0.45, wspace=0.40)

    # ---- top-left: per-layer goodness pos vs neg ----
    ax = fig.add_subplot(gs[0, 0])
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd", "#ff7f0e"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5", "#ffbb78"]
    for L in range(n_layers):
        ax.plot(epochs, history["g_pos_per_layer"][L],
                color=colors_pos[L % len(colors_pos)], linewidth=1.5,
                label=f"L{L} pos")
        ax.plot(epochs, history["g_neg_per_layer"][L],
                color=colors_neg[L % len(colors_neg)], linewidth=1.5,
                linestyle="--", label=f"L{L} neg")
    ax.axhline(2.0, color="black", linewidth=0.7, linestyle=":")
    ax.set_xlim(0, total_epochs)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel(r"goodness  $\langle h^2 \rangle$", fontsize=9)
    ax.set_title("Per-layer goodness", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="upper left")
    ax.grid(alpha=0.3)

    # ---- top-middle: the demo image ----
    ax = fig.add_subplot(gs[0, 1])
    encoded = encode_label_in_pixels(demo_image, int(demo_label))
    ax.imshow(encoded, cmap="gray", vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"demo image  (true: {demo_label})", fontsize=10)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), 10, 1,
                               facecolor="none", edgecolor="cyan",
                               linewidth=1.0))

    # ---- top-right: per-label summed goodness on the demo image ----
    ax = fig.add_subplot(gs[0, 2])
    flat = demo_image.reshape(-1)
    candidates = np.tile(flat, (10, 1))
    candidates[:, :10] = 0.0
    candidates[np.arange(10), np.arange(10)] = 1.0
    g = goodness_per_layer(model, candidates, skip_first=False)
    summed = g.sum(axis=1)
    pred = int(summed.argmax())
    bar_colors = ["#1f77b4"] * 10
    bar_colors[int(demo_label)] = "#2ca02c"
    if pred != int(demo_label):
        bar_colors[pred] = "#d62728"
    ax.bar(range(10), summed, color=bar_colors)
    ax.set_xticks(range(10))
    ax.set_xticklabels([str(i) for i in range(10)], fontsize=8)
    ax.set_xlabel("candidate label", fontsize=9)
    ax.set_ylabel("summed goodness", fontsize=9)
    title_color = "#2ca02c" if pred == int(demo_label) else "#d62728"
    ax.set_title(f"prediction: {pred}", fontsize=10, color=title_color)
    ax.grid(alpha=0.3, axis="y")

    # ---- bottom-left: per-layer loss ----
    ax = fig.add_subplot(gs[1, 0])
    for L in range(n_layers):
        ax.plot(epochs, history["loss_per_layer"][L],
                linewidth=1.5, label=f"layer {L}")
    ax.set_xlim(0, total_epochs)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("FF loss", fontsize=9)
    ax.set_title("Per-layer FF loss", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # ---- bottom-right: train/test accuracy ----
    ax = fig.add_subplot(gs[1, 1:])
    ax.plot(epochs, np.array(history["train_acc"]) * 100,
            color="#1f77b4", linewidth=1.7, label="train")
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#d62728", linewidth=1.7, label="test")
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, 100)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("accuracy (%)", fontsize=9)
    ax.set_title(f"Accuracy  (test: {history['test_acc'][-1]*100:.1f}%)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(f"FF supervised MNIST  —  epoch {epoch}/{total_epochs}",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE,
                                    colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--layer-sizes", type=str, default="784,500,500")
    p.add_argument("--train-subset", type=int, default=None,
                   help="Optionally subsample training data (speeds GIF rendering).")
    p.add_argument("--out", type=str, default="ff_label_in_input.gif")
    p.add_argument("--hold-final", type=int, default=12)
    p.add_argument("--max-size-kb", type=int, default=3072,
                   help="Soft limit on output GIF size in KB.")
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))

    print("Loading MNIST...")
    data = load_mnist()
    train_x, train_y, test_x, test_y = data
    rng = np.random.default_rng(args.seed)
    demo_idx = int(rng.integers(0, test_x.shape[0]))
    demo_image = test_x[demo_idx]
    demo_label = int(test_y[demo_idx])
    print(f"  demo image: index {demo_idx}, true label {demo_label}")

    model = build_ff_mlp(layer_sizes=layer_sizes, threshold=2.0,
                         seed=args.seed)
    cfg = TrainConfig(n_epochs=args.epochs, batch_size=128, lr=args.lr,
                      threshold=2.0, layer_sizes=layer_sizes,
                      seed=args.seed,
                      train_subset=args.train_subset,
                      eval_subset=2000)

    frames: list[Image.Image] = []

    def cb(epoch: int, m: FFModel, hist: dict) -> None:
        frame = render_frame(hist, m, demo_image, demo_label,
                             epoch + 1, args.epochs)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
              f"test={hist['test_acc'][-1]*100:.1f}%")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    train(model, data, cfg,
          snapshot_callback=cb,
          snapshot_every=args.snapshot_every,
          verbose=False)

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")
    if size_kb > args.max_size_kb:
        print(f"WARNING: gif size {size_kb:.0f} KB exceeds soft limit "
              f"{args.max_size_kb} KB. Consider lower --fps, fewer frames, or "
              f"a smaller palette.")


if __name__ == "__main__":
    main()
