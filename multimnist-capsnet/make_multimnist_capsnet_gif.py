"""
Render an animated GIF of the MultiMNIST CapsNet learning to disentangle two
overlapping digits via routing-by-agreement.

Each frame shows:
  - Top row: 4 fixed validation composites
  - Middle two rows: per-digit reconstructions (digit A, digit B)
  - Bottom: training curves (test 2-digit accuracy + recon MSE) up to current step.

Usage:
    python3 make_multimnist_capsnet_gif.py
    python3 make_multimnist_capsnet_gif.py --n-epochs 8 --snapshot-every 75 --fps 6
"""
from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from multimnist_capsnet import (
    CapsNet, train, generate_multimnist, load_mnist, evaluate,
)


def _fixed_pairs(n: int = 4, seed: int = 7, canvas: int = 36):
    images, labels = load_mnist("train")
    X, A, B, L = generate_multimnist(n, images, labels, canvas=canvas,
                                     max_shift=4, min_overlap=0.8, seed=seed)
    return X, A, B, L


def render_frame(model: CapsNet, history: dict, step: int,
                 X: np.ndarray, A: np.ndarray, B: np.ndarray, L: np.ndarray,
                 inline_metric: tuple[float, float] | None = None) -> Image.Image:
    n = X.shape[0]
    canvas = model.canvas
    v, _ = model.forward(X)
    ma = np.zeros((n, 10), dtype=np.float32)
    ma[np.arange(n), L[:, 0]] = 1.0
    mb = np.zeros((n, 10), dtype=np.float32)
    mb[np.arange(n), L[:, 1]] = 1.0
    recon_a, _ = model.decode(v, ma)
    recon_b, _ = model.decode(v, mb)
    recon_a = np.clip(recon_a, 0, 1).reshape(n, canvas, canvas)
    recon_b = np.clip(recon_b, 0, 1).reshape(n, canvas, canvas)

    fig = plt.figure(figsize=(10, 6.5), dpi=100)
    gs = fig.add_gridspec(4, n, height_ratios=[1.0, 0.9, 0.9, 1.4],
                          hspace=0.45, wspace=0.20)

    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(X[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_title(f"composite ({L[i,0]},{L[i,1]})", fontsize=8)
        if i == 0:
            ax.set_ylabel("input", fontsize=9)

        ax = fig.add_subplot(gs[1, i])
        ax.imshow(recon_a[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f"recon a", fontsize=9)

        ax = fig.add_subplot(gs[2, i])
        ax.imshow(recon_b[i], cmap="gray", vmin=0, vmax=1)
        ax.set_xticks([]); ax.set_yticks([])
        if i == 0:
            ax.set_ylabel(f"recon b", fontsize=9)

    ax_curve = fig.add_subplot(gs[3, :])
    if history["epoch"]:
        ax_curve.plot(history["epoch"], history["test_acc"], color="#2ca02c",
                      marker="o", linewidth=1.5, label="test 2-digit acc")
        ax_curve.set_ylabel("test acc", color="#2ca02c", fontsize=9)
        ax_curve.tick_params(axis="y", labelcolor="#2ca02c")
        ax_curve.set_xlabel("epoch", fontsize=9)
        ax_curve.set_ylim(0, 1.0)
        ax_curve.axhline(1/45, color="gray", linestyle=":", linewidth=0.6)
        ax_curve.grid(alpha=0.3)

        ax2 = ax_curve.twinx()
        ax2.plot(history["epoch"], history["test_recon_mse"], color="#d62728",
                 linestyle="--", linewidth=1.5, label="test recon MSE")
        ax2.set_ylabel("recon MSE", color="#d62728", fontsize=9)

        lines1, labels1 = ax_curve.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_curve.legend(lines1 + lines2, labels1 + labels2,
                        loc="center right", fontsize=8, framealpha=0.85)

    ep = history["epoch"][-1] if history["epoch"] else 0
    title = f"MultiMNIST CapsNet — step {step}, epoch {ep:.1f}"
    if inline_metric is not None:
        a, m = inline_metric
        title += f"   (test acc {a:.3f}, recon MSE {m:.4f})"
    fig.suptitle(title, fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    w, h = img.size
    img = img.resize((int(w * 0.80), int(h * 0.80)), Image.LANCZOS)
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=8)
    p.add_argument("--n-train", type=int, default=6000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--snapshot-every", type=int, default=75,
                   help="Snapshot every N training steps")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="multimnist_capsnet.gif")
    p.add_argument("--hold-final", type=int, default=8)
    args = p.parse_args()

    fixed_X, fixed_A, fixed_B, fixed_L = _fixed_pairs(n=4, seed=7, canvas=36)
    frames: list[Image.Image] = []

    def cb(step, model, history, Xte, Ate, Bte, Lte):
        # Quick-and-cheap eval for the inline metric (small subset)
        a, m = evaluate(model, Xte[:200], Ate[:200], Bte[:200], Lte[:200],
                        batch_size=32)
        # Build a synthetic running-history copy with the latest mid-epoch point.
        tmp = dict(history)
        if not tmp["epoch"] or tmp["step"][-1] < step:
            tmp = {
                "epoch": list(tmp["epoch"]) + [step / max(1, args.n_train // 32)],
                "step":  list(tmp["step"])  + [step],
                "margin": list(tmp.get("margin", [])) + [0.0],
                "recon":  list(tmp.get("recon", []))  + [0.0],
                "loss":   list(tmp.get("loss", []))   + [0.0],
                "test_acc": list(tmp["test_acc"]) + [a],
                "test_recon_mse": list(tmp["test_recon_mse"]) + [m],
            }
        frame = render_frame(model, tmp, step, fixed_X, fixed_A, fixed_B,
                             fixed_L, inline_metric=(a, m))
        frames.append(frame)
        print(f"  frame {len(frames):3d}  step {step}  test_acc={a:.3f}  recon_mse={m:.4f}")

    print(f"Training {args.n_epochs} epochs, snapshot every {args.snapshot_every} steps...")
    model, history, _ = train(
        n_epochs=args.n_epochs, n_train=args.n_train, n_test=args.n_test,
        seed=args.seed, snapshot_callback=cb,
        snapshot_every=args.snapshot_every, verbose=False,
    )

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
