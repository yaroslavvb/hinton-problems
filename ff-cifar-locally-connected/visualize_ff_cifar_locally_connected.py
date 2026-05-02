"""
Static visualisations for the trained FF locally-connected CIFAR-10 model.

Outputs (in `viz/`):
  example_images.png             4 raw CIFAR samples per class.
  example_label_encoded.png      Same images with the (3 x 10 x 3) label
                                 strip overlaid (one per class shown).
  receptive_fields_layer0.png    A grid of layer-0 receptive fields sampled
                                 from different spatial locations and
                                 channels — visualises the "no-weight-
                                 sharing" property by showing spatial
                                 variation of similar-channel filters.
  per_class_accuracy.png         Bar chart of FF and BP per-class test
                                 accuracy on the full 10K test set.
  ff_vs_bp_curves.png            Train + test accuracy across epochs for
                                 FF and BP, plus FF goodness gap and BP
                                 cross-entropy.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from ff_cifar_locally_connected import (
    load_cifar10, FFModel, BPModel, LCLayer,
    encode_label_in_image, train_ff, train_bp, TrainConfig,
    evaluate_ff, evaluate_bp,
    per_class_accuracy_ff, per_class_accuracy_bp,
    predict_by_goodness, bp_predict_batch,
    CIFAR_MEAN, CIFAR_CLASSES, LABEL_LEN, LABEL_ROWS, LABEL_OFF, LABEL_ON,
)


def _denormalize(images_centered: np.ndarray) -> np.ndarray:
    """Add per-channel mean back, clip to [0, 1] for display."""
    out = images_centered + CIFAR_MEAN
    return np.clip(out, 0.0, 1.0)


def plot_example_images(images_centered: np.ndarray, labels: np.ndarray,
                        out_path: str) -> None:
    """One sample per class in a 2x5 grid."""
    fig, axes = plt.subplots(2, 5, figsize=(10, 4.4), dpi=120)
    seen = {}
    for img, y in zip(images_centered, labels):
        c = int(y)
        if c not in seen:
            seen[c] = img
        if len(seen) == 10:
            break
    for c in range(10):
        ax = axes[c // 5, c % 5]
        if c in seen:
            ax.imshow(_denormalize(seen[c]), interpolation="nearest")
        ax.set_title(f"{CIFAR_CLASSES[c]}", fontsize=10)
        ax.axis("off")
    fig.suptitle("CIFAR-10 — one sample per class", fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_label_encoded(images_centered: np.ndarray, labels: np.ndarray,
                       out_path: str) -> None:
    """Show 4 images with the (LABEL_ROWS x LABEL_LEN x 3) label strip
    overlaid (true label encoded). The cyan box marks the label slot.
    """
    sel = images_centered[:4]
    sel_y = labels[:4]
    encoded = encode_label_in_image(sel, sel_y)
    fig, axes = plt.subplots(1, 4, figsize=(11, 3.0), dpi=120)
    for ax, img, y in zip(axes, encoded, sel_y):
        ax.imshow(_denormalize(img), interpolation="nearest")
        ax.add_patch(plt.Rectangle((-0.5, -0.5), LABEL_LEN, LABEL_ROWS,
                                   facecolor="none", edgecolor="cyan",
                                   linewidth=1.4))
        ax.set_title(f"label = {int(y)} ({CIFAR_CLASSES[int(y)]})",
                     fontsize=9)
        ax.axis("off")
    fig.suptitle(
        f"Label encoded as one-hot in the top-left {LABEL_ROWS} x "
        f"{LABEL_LEN} x 3 strip  (off={LABEL_OFF}, on={LABEL_ON})",
        fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_receptive_fields(model: FFModel, out_path: str,
                          n_locations: int = 6,
                          n_channels: int = 4,
                          seed: int = 0) -> None:
    """Sample n_locations spatial positions in layer 0 and show n_channels
    receptive fields per position. The point: the same channel index has
    *different* learned filters at different spatial positions — that's the
    "no weight sharing" property of locally-connected layers.

    Each tile is a (RF, RF, C_in) weight slice reshaped to (RF, RF, 3) and
    rendered as RGB after per-tile rescale.
    """
    L0 = model.layers[0]
    H_o, W_o, K, C_out = L0.W.shape
    RF = L0.RF
    assert K == RF * RF * 3, "expected layer 0 to consume 3-channel input"

    rng = np.random.default_rng(seed)
    # Pick `n_locations` distinct (i, j) pairs at random.
    coords = []
    while len(coords) < n_locations:
        i = int(rng.integers(0, H_o))
        j = int(rng.integers(0, W_o))
        if (i, j) not in coords:
            coords.append((i, j))

    # Pick n_channels random channels, used consistently across locations.
    channel_idx = rng.permutation(C_out)[:n_channels]

    fig, axes = plt.subplots(n_locations, n_channels,
                             figsize=(1.2 * n_channels, 1.2 * n_locations),
                             dpi=140)
    for r, (i, j) in enumerate(coords):
        for c, ch in enumerate(channel_idx):
            rf = L0.W[i, j, :, ch].reshape(RF, RF, 3)
            # Per-tile rescale to [0, 1] for visibility.
            mn, mx = rf.min(), rf.max()
            rng_ = (mx - mn) if mx > mn else 1.0
            disp = (rf - mn) / rng_
            ax = axes[r, c] if n_locations > 1 else axes[c]
            ax.imshow(disp, interpolation="nearest")
            if r == 0:
                ax.set_title(f"ch {int(ch)}", fontsize=8)
            if c == 0:
                ax.set_ylabel(f"({i},{j})", fontsize=8, rotation=0,
                              labelpad=18, va="center")
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"Layer-0 receptive fields ({RF}x{RF}x3)  —  "
        f"per-location filters at {n_locations} positions, {n_channels} channels each",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_class_accuracy(ff_acc: np.ndarray, bp_acc: np.ndarray,
                            out_path: str) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 4.2), dpi=120)
    x = np.arange(10)
    width = 0.40
    ax.bar(x - width / 2, ff_acc * 100, width, color="#1f77b4",
           label=f"FF  (mean {ff_acc.mean()*100:.1f}%)")
    ax.bar(x + width / 2, bp_acc * 100, width, color="#d62728",
           label=f"BP  (mean {bp_acc.mean()*100:.1f}%)")
    ax.axhline(10, color="black", linewidth=0.6, linestyle=":",
               label="chance (10%)")
    ax.set_xticks(x)
    ax.set_xticklabels(CIFAR_CLASSES, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.set_title("Per-class CIFAR-10 test accuracy "
                 "(FF vs end-to-end backprop, same locally-connected arch)",
                 fontsize=11)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(fontsize=9, loc="upper right")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_ff_vs_bp_curves(ff_history: dict, bp_history: dict,
                         out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)

    epochs_ff = ff_history["epoch"]
    epochs_bp = bp_history["epoch"]

    # Top-left: train + test accuracy
    ax = axes[0, 0]
    ax.plot(epochs_ff, np.array(ff_history["test_acc"]) * 100,
            color="#1f77b4", linewidth=1.6, label="FF test")
    ax.plot(epochs_ff, np.array(ff_history["train_acc"]) * 100,
            color="#1f77b4", linewidth=1.0, linestyle="--", label="FF train")
    ax.plot(epochs_bp, np.array(bp_history["test_acc"]) * 100,
            color="#d62728", linewidth=1.6, label="BP test")
    ax.plot(epochs_bp, np.array(bp_history["train_acc"]) * 100,
            color="#d62728", linewidth=1.0, linestyle="--", label="BP train")
    ax.axhline(10, color="black", linewidth=0.6, linestyle=":")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("FF vs BP accuracy over training")
    ax.set_ylim(0, max(60, 5 + max(
        max(ff_history["test_acc"]) * 100, max(bp_history["test_acc"]) * 100)))
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Top-right: per-layer FF goodness pos vs neg
    ax = axes[0, 1]
    n_layers = len(ff_history["g_pos_per_layer"])
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5"]
    for L in range(n_layers):
        ax.plot(epochs_ff, ff_history["g_pos_per_layer"][L],
                color=colors_pos[L % len(colors_pos)],
                label=f"L{L} pos", linewidth=1.4)
        ax.plot(epochs_ff, ff_history["g_neg_per_layer"][L],
                color=colors_neg[L % len(colors_neg)],
                label=f"L{L} neg", linewidth=1.4, linestyle="--")
    ax.axhline(2.0, color="black", linewidth=0.6, linestyle=":",
               label=r"$\theta = 2.0$")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"goodness  $\langle h^2 \rangle$")
    ax.set_title("FF per-layer goodness (pos vs neg)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, ncol=2)

    # Bottom-left: FF per-layer loss
    ax = axes[1, 0]
    for L in range(n_layers):
        ax.plot(epochs_ff, ff_history["loss_per_layer"][L],
                linewidth=1.5, label=f"layer {L}")
    ax.set_xlabel("epoch")
    ax.set_ylabel("FF loss")
    ax.set_title("FF per-layer loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    # Bottom-right: BP cross-entropy
    ax = axes[1, 1]
    ax.plot(epochs_bp, bp_history["loss"], color="#d62728", linewidth=1.5)
    ax.set_xlabel("epoch")
    ax.set_ylabel("softmax cross-entropy")
    ax.set_title("BP training loss")
    ax.grid(alpha=0.3)

    fig.suptitle(
        f"FF (test {ff_history['test_acc'][-1]*100:.1f}%) vs "
        f"BP (test {bp_history['test_acc'][-1]*100:.1f}%) on CIFAR-10  "
        f"(same locally-connected architecture)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Loading saved run
# ---------------------------------------------------------------------------

def load_saved(npz_path: str
               ) -> tuple[FFModel, dict, dict, BPModel, np.ndarray]:
    """Load FF model + FF / BP histories + BP model from the .npz produced
    by ff_cifar_locally_connected.py --save.

    Returns (ff_model, ff_history, bp_history, bp_model, layer_specs). Any
    of the BP fields may be None if the original run skipped --bp-baseline.
    """
    z = np.load(npz_path)
    layer_specs = [(int(rf), int(c)) for (rf, c) in z["layer_specs"]]
    threshold = float(z["threshold"])
    seed = int(z["seed"])

    rng = np.random.default_rng(seed)
    model = FFModel.init(layer_specs, threshold=threshold, rng=rng,
                         n_classes=10)
    for i, L in enumerate(model.layers):
        L.W = z[f"ff_layer{i}_W"].astype(np.float32)
        L.b = z[f"ff_layer{i}_b"].astype(np.float32)

    n_layers = len(layer_specs)
    ff_history = {
        "epoch": list(range(1, len(z["ff_test_acc"]) + 1)),
        "test_acc": list(z["ff_test_acc"]),
        "train_acc": list(z["ff_train_acc"]),
        "loss_per_layer": [list(z["ff_loss_per_layer"][L])
                           for L in range(n_layers)],
        "g_pos_per_layer": [list(z["ff_g_pos"][L]) for L in range(n_layers)],
        "g_neg_per_layer": [list(z["ff_g_neg"][L]) for L in range(n_layers)],
        "wallclock": list(z["ff_wallclock"]),
    }
    if "ff_full_test_acc" in z.files:
        ff_history["full_test_acc"] = float(z["ff_full_test_acc"])

    bp_history = None
    bp_model = None
    if "bp_test_acc" in z.files:
        bp_history = {
            "epoch": list(range(1, len(z["bp_test_acc"]) + 1)),
            "test_acc": list(z["bp_test_acc"]),
            "train_acc": list(z["bp_train_acc"]),
            "loss": list(z["bp_loss"]),
            "wallclock": list(z["bp_wallclock"]),
        }
        if "bp_full_test_acc" in z.files:
            bp_history["full_test_acc"] = float(z["bp_full_test_acc"])
        if "bp_W_out" in z.files:
            bp_rng = np.random.default_rng(seed + 100)
            bp_model = BPModel.init(layer_specs, n_classes=10, rng=bp_rng)
            for i, L in enumerate(bp_model.layers):
                L.W = z[f"bp_layer{i}_W"].astype(np.float32)
                L.b = z[f"bp_layer{i}_b"].astype(np.float32)
            bp_model.W_out = z["bp_W_out"].astype(np.float32)
            bp_model.b_out = z["bp_b_out"].astype(np.float32)
    return model, ff_history, bp_history, bp_model, np.array(layer_specs)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", type=str, default="model.npz",
                   help="Saved .npz from ff_cifar_locally_connected.py.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=10,
                   help="Used only if --model is missing -- trains from scratch.")
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--n-layers", type=int, default=2, choices=(2, 3))
    p.add_argument("--train-subset", type=int, default=10000)
    p.add_argument("--eval-subset", type=int, default=1000)
    p.add_argument("--bp-baseline", action="store_true", default=True)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--per-class-test-subset", type=int, default=10000,
                   help="Use this many test images for per-class accuracy "
                        "(prediction is expensive; default = full 10K).")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading CIFAR-10 ...")
    train_x, train_y, test_x, test_y = load_cifar10()
    train_x = train_x - CIFAR_MEAN
    test_x = test_x - CIFAR_MEAN
    print(f"  train {train_x.shape}, test {test_x.shape}")

    if os.path.exists(args.model):
        print(f"Loading saved run from {args.model}")
        ff_model, ff_history, bp_history, bp_model, _ = load_saved(args.model)
    else:
        print(f"No saved model at {args.model}; training from scratch ...")
        if args.n_layers == 2:
            specs = ((11, 8), (5, 8))
        else:
            specs = ((11, 8), (5, 8), (5, 8))
        cfg = TrainConfig(
            n_epochs=args.n_epochs, batch_size=64, lr=args.lr,
            threshold=2.0, layer_specs=specs, seed=args.seed,
            train_subset=args.train_subset, eval_subset=args.eval_subset)
        rng = np.random.default_rng(args.seed)
        ff_model = FFModel.init(list(specs), threshold=2.0, rng=rng,
                                n_classes=10)
        ff_history = train_ff(ff_model,
                              (train_x, train_y, test_x, test_y),
                              cfg, verbose=True)
        if args.bp_baseline:
            bp_rng = np.random.default_rng(args.seed + 100)
            bp_model = BPModel.init(list(specs), n_classes=10, rng=bp_rng)
            bp_history = train_bp(bp_model,
                                  (train_x, train_y, test_x, test_y),
                                  cfg, verbose=True)
        else:
            bp_model = None
            bp_history = None

    # Static plots that need only data + model
    plot_example_images(test_x, test_y,
                        os.path.join(args.outdir, "example_images.png"))
    plot_label_encoded(test_x, test_y,
                       os.path.join(args.outdir,
                                    "example_label_encoded.png"))
    plot_receptive_fields(ff_model,
                          os.path.join(args.outdir,
                                       "receptive_fields_layer0.png"))

    # Per-class accuracy: needs forward passes.
    n_eval = min(args.per_class_test_subset, test_x.shape[0])
    ev_x = test_x[:n_eval]; ev_y = test_y[:n_eval]
    print(f"Computing per-class accuracy on {n_eval} test images ...")
    ff_acc = per_class_accuracy_ff(ff_model, ev_x, ev_y, n_classes=10)
    if bp_history is not None and bp_model is None:
        # We have a saved BP history but no live BP model — skip BP per-class.
        print("  (no live BP model; skipping BP per-class bars)")
        bp_acc = np.full(10, np.nan)
    elif bp_model is not None:
        bp_acc = per_class_accuracy_bp(bp_model, ev_x, ev_y, n_classes=10)
    else:
        bp_acc = np.full(10, np.nan)
    plot_per_class_accuracy(ff_acc, bp_acc,
                            os.path.join(args.outdir,
                                         "per_class_accuracy.png"))

    if bp_history is not None:
        plot_ff_vs_bp_curves(
            ff_history, bp_history,
            os.path.join(args.outdir, "ff_vs_bp_curves.png"))


if __name__ == "__main__":
    main()
