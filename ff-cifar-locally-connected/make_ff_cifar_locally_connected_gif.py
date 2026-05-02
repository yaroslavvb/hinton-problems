"""
Render an animated GIF of the locally-connected FF model learning CIFAR-10.

Each frame shows:
  Top-left:  per-layer FF goodness (pos solid, neg dashed) over time, with
             the threshold theta marked.
  Top-mid:   the demo image (label-encoded) in centred-pixel space, scaled
             back to RGB for display. The cyan box marks the (LABEL_ROWS x
             LABEL_LEN) label slot.
  Top-right: bar chart of summed per-class goodness for the demo image
             (true label = green, prediction = red if wrong).
  Bottom:    train + test accuracy across epochs.

Usage:
    python3 make_ff_cifar_locally_connected_gif.py --epochs 8 --fps 4
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ff_cifar_locally_connected import (
    load_cifar10, FFModel, train_ff, TrainConfig,
    encode_label_in_image, goodness_per_layer,
    CIFAR_MEAN, CIFAR_CLASSES,
    LABEL_LEN, LABEL_ROWS, LABEL_OFF, LABEL_ON,
)


def _denormalize(img_centered: np.ndarray) -> np.ndarray:
    return np.clip(img_centered + CIFAR_MEAN, 0.0, 1.0)


def render_frame(history: dict, model: FFModel,
                 demo_image: np.ndarray, demo_label: int,
                 epoch: int, total_epochs: int,
                 skip_first: bool = True) -> Image.Image:
    n_layers = len(history["loss_per_layer"])
    epochs = history["epoch"]

    fig = plt.figure(figsize=(10.5, 5.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.0],
                          width_ratios=[1.2, 0.7, 1.2],
                          hspace=0.45, wspace=0.40)

    # ---- top-left: per-layer goodness ----
    ax = fig.add_subplot(gs[0, 0])
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5"]
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
    ax.set_ylabel(r"$\langle h^2 \rangle$", fontsize=9)
    ax.set_title("Per-layer goodness", fontsize=10)
    ax.legend(fontsize=7, ncol=2, loc="lower right")
    ax.grid(alpha=0.3)

    # ---- top-mid: demo image (label-encoded) ----
    ax = fig.add_subplot(gs[0, 1])
    encoded = encode_label_in_image(demo_image[None, ...],
                                    np.array([demo_label], dtype=np.int32))
    ax.imshow(_denormalize(encoded[0]), interpolation="nearest")
    ax.set_title(f"demo: {CIFAR_CLASSES[demo_label]}", fontsize=10)
    ax.axis("off")
    ax.add_patch(plt.Rectangle((-0.5, -0.5), LABEL_LEN, LABEL_ROWS,
                               facecolor="none", edgecolor="cyan",
                               linewidth=1.0))

    # ---- top-right: per-label summed goodness on demo image ----
    ax = fig.add_subplot(gs[0, 2])
    n = 10
    candidates = np.repeat(demo_image[None, ...], n, axis=0).copy()
    cand_labels = np.arange(n, dtype=np.int32)
    candidates[:, :LABEL_ROWS, :LABEL_LEN, :] = LABEL_OFF
    candidates[np.arange(n)[:, None], :LABEL_ROWS, cand_labels[:, None], :] = LABEL_ON
    g = goodness_per_layer(model, candidates, skip_first=skip_first)
    summed = g.sum(axis=1)
    pred = int(summed.argmax())
    bar_colors = ["#1f77b4"] * n
    bar_colors[demo_label] = "#2ca02c"
    if pred != demo_label:
        bar_colors[pred] = "#d62728"
    ax.bar(range(n), summed, color=bar_colors)
    ax.set_xticks(range(n))
    ax.set_xticklabels([c[:4] for c in CIFAR_CLASSES],
                       fontsize=7, rotation=30, ha="right")
    ax.set_ylabel("summed goodness", fontsize=9)
    title_color = "#2ca02c" if pred == demo_label else "#d62728"
    ax.set_title(f"prediction: {CIFAR_CLASSES[pred]}", fontsize=10,
                 color=title_color)
    ax.grid(alpha=0.3, axis="y")

    # ---- bottom: train/test accuracy ----
    ax = fig.add_subplot(gs[1, :])
    ax.plot(epochs, np.array(history["train_acc"]) * 100,
            color="#1f77b4", linewidth=1.7, label="train")
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#d62728", linewidth=1.7, label="test")
    ax.axhline(10, color="black", linewidth=0.7, linestyle=":",
               label="chance")
    ax.set_xlim(0, total_epochs)
    ymax = max(50, 5 + max(history["test_acc"]) * 100)
    ax.set_ylim(0, ymax)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("accuracy (%)", fontsize=9)
    ax.set_title(f"Accuracy  (test: {history['test_acc'][-1]*100:.1f}%)",
                 fontsize=10)
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"FF locally-connected on CIFAR-10  —  epoch {epoch}/{total_epochs}",
        fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P",
                                   palette=Image.Palette.ADAPTIVE, colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-layers", type=int, default=2, choices=(2, 3))
    p.add_argument("--train-subset", type=int, default=4000,
                   help="Smaller subset for GIF (default 4K) to keep "
                        "render time short.")
    p.add_argument("--eval-subset", type=int, default=500)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--out", type=str, default="ff_cifar_locally_connected.gif")
    p.add_argument("--hold-final", type=int, default=10)
    p.add_argument("--max-size-kb", type=int, default=3072,
                   help="Soft limit on output GIF size (KB). Warns if exceeded.")
    args = p.parse_args()

    if args.n_layers == 2:
        specs = ((11, 8), (5, 8))
    else:
        specs = ((11, 8), (5, 8), (5, 8))

    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = load_cifar10()
    train_x = train_x - CIFAR_MEAN
    test_x = test_x - CIFAR_MEAN

    rng = np.random.default_rng(args.seed)
    demo_idx = int(rng.integers(0, test_x.shape[0]))
    demo_image = test_x[demo_idx]
    demo_label = int(test_y[demo_idx])
    print(f"  demo image idx={demo_idx}, label={demo_label} "
          f"({CIFAR_CLASSES[demo_label]})")

    model = FFModel.init(list(specs), threshold=2.0, rng=rng, n_classes=10)
    cfg = TrainConfig(n_epochs=args.epochs, batch_size=64, lr=args.lr,
                      threshold=2.0, layer_specs=specs, seed=args.seed,
                      train_subset=args.train_subset,
                      eval_subset=args.eval_subset)

    frames: list[Image.Image] = []

    def cb(epoch: int, m: FFModel, hist: dict) -> None:
        frame = render_frame(hist, m, demo_image, demo_label,
                             epoch + 1, args.epochs)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch+1}  "
              f"test={hist['test_acc'][-1]*100:.1f}%")

    print(f"Training {args.epochs} epochs, snapshot every "
          f"{args.snapshot_every} ...")
    train_ff(model, (train_x, train_y, test_x, test_y), cfg,
             snapshot_callback=cb, snapshot_every=args.snapshot_every,
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
        print(f"WARNING: gif {size_kb:.0f} KB > soft limit "
              f"{args.max_size_kb} KB. Consider lower --fps, fewer frames, "
              f"or smaller palette.")


if __name__ == "__main__":
    main()
