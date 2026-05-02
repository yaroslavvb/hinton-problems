"""
Animate the gated three-way RBM as it learns transformation features.

Layout per frame:
  Top-left:  4 example pairs (x, y) with their transformation labels
             (these stay fixed across the run -- they show the *task*)
  Top-right: top factor filter pairs (W^x_f, W^y_f) at the current epoch
             (these change as the model learns)
  Bottom:    training curves up to current epoch
             (recon MSE on left axis, transform-classification accuracy on
             right axis)

Usage:
  python3 make_transforming_pairs_gif.py
  python3 make_transforming_pairs_gif.py --epochs 100 --snapshot-every 4 --fps 12
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from transforming_pairs import (
    GatedRBM,
    build_gated_rbm,
    generate_transformed_pairs,
    reconstruction_metrics,
    train,
    transform_classification_accuracy,
    transform_label,
    visualize_transformation_filters,
)

H_W = 13


def render_frame(model: GatedRBM,
                 history: dict,
                 epoch: int,
                 example_X: np.ndarray,
                 example_Y: np.ndarray,
                 example_pool_idx: np.ndarray,
                 pool: list,
                 n_factor_panels: int = 8) -> Image.Image:
    n_examples = 4
    n_cols = n_examples + n_factor_panels
    fig = plt.figure(figsize=(11, 6.0), dpi=100)
    gs = fig.add_gridspec(3, n_cols,
                          height_ratios=[1.4, 1.4, 1.0],
                          hspace=0.55, wspace=0.18)

    # Top row: 4 fixed example pairs
    for k in range(n_examples):
        col = k
        ax_x = fig.add_subplot(gs[0, col])
        ax_x.imshow(example_X[k].reshape(H_W, H_W), cmap="gray_r",
                     vmin=0, vmax=1, interpolation="nearest")
        ax_x.set_xticks([]); ax_x.set_yticks([])
        ax_x.set_title(f"$x_{k}$", fontsize=9, pad=2)
        ax_y = fig.add_subplot(gs[1, col])
        ax_y.imshow(example_Y[k].reshape(H_W, H_W), cmap="gray_r",
                     vmin=0, vmax=1, interpolation="nearest")
        ax_y.set_xticks([]); ax_y.set_yticks([])
        ax_y.set_title(f"$y_{k}$ ({transform_label(pool[int(example_pool_idx[k])])})",
                       fontsize=8, pad=2)

    # Top right: top factor filter pairs
    factors, Wx_imgs, Wy_imgs = visualize_transformation_filters(
        model, h_w=H_W, n_top=n_factor_panels)
    vmax = float(max(np.abs(Wx_imgs).max(), np.abs(Wy_imgs).max(), 1e-6))
    for k in range(n_factor_panels):
        col = n_examples + k
        ax_x = fig.add_subplot(gs[0, col])
        ax_x.imshow(Wx_imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
        ax_x.set_xticks([]); ax_x.set_yticks([])
        ax_x.set_title(f"$W^x_{{{int(factors[k])}}}$", fontsize=8, pad=2)
        ax_y = fig.add_subplot(gs[1, col])
        ax_y.imshow(Wy_imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
        ax_y.set_xticks([]); ax_y.set_yticks([])
        ax_y.set_title(f"$W^y_{{{int(factors[k])}}}$", fontsize=8, pad=2)

    # Bottom: training curves
    ax = fig.add_subplot(gs[2, :])
    if history["epoch"]:
        e = history["epoch"]
        ax.plot(e, history["recon_mse"], color="#1f77b4",
                label="recon MSE (train)", linewidth=1.4)
        ax.set_ylabel("recon MSE", fontsize=9, color="#1f77b4")
        ax.tick_params(axis="y", labelcolor="#1f77b4")
        ax.set_xlabel("epoch", fontsize=9)
        ax.grid(alpha=0.25)

        if "transform_acc" in history:
            ax2 = ax.twinx()
            eval_e = [ee for ee, v in zip(e, history["transform_acc"])
                      if v is not None]
            eval_v = [v * 100 for v in history["transform_acc"]
                      if v is not None]
            ax2.plot(eval_e, eval_v, "o-", color="#2ca02c",
                     label="transform classification (test)",
                     markersize=3, linewidth=1.2)
            ax2.set_ylabel("classification %", fontsize=9, color="#2ca02c")
            ax2.tick_params(axis="y", labelcolor="#2ca02c")
            ax2.set_ylim(0, 100)

    fig.suptitle(f"transforming-pairs gated 3-way RBM — epoch {epoch + 1}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transforms", type=str, default="shift")
    p.add_argument("--shift-max", type=int, default=1)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=600)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--n-factors", type=int, default=64)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--out", type=str, default="transforming_pairs.gif")
    p.add_argument("--hold-final", type=int, default=15)
    args = p.parse_args()

    transforms = tuple(t.strip() for t in args.transforms.split(","))
    rng = np.random.default_rng(args.seed)

    print(f"Generating data ...")
    X_tr, Y_tr, ids_tr, pool = generate_transformed_pairs(
        args.n_train, transforms=transforms, shift_max=args.shift_max, rng=rng)
    X_te, Y_te, ids_te, _ = generate_transformed_pairs(
        args.n_test, transforms=transforms, shift_max=args.shift_max, rng=rng)
    n_classes = len(pool)
    print(f"  {n_classes} transformation classes")

    # Pick 4 example pairs from distinct transforms for the static demo row.
    distinct = list(dict.fromkeys(int(t) for t in ids_te))
    pick = []
    for t_idx in distinct:
        cand = np.where(ids_te == t_idx)[0]
        if len(cand):
            pick.append(int(cand[0]))
        if len(pick) >= 4:
            break
    pick = np.array(pick)
    example_X = X_te[pick]
    example_Y = Y_te[pick]
    example_pool_idx = ids_te[pick]

    model = build_gated_rbm(169, 169, args.n_hidden, args.n_factors,
                             init_scale=args.init_scale, seed=args.seed)

    frames: list[Image.Image] = []

    def eval_fn(m, epoch):
        acc, _ = transform_classification_accuracy(
            m, X_te, Y_te, ids_te, n_classes,
            rng=np.random.default_rng(epoch))
        return dict(transform_acc=acc)

    def cb(epoch, m, history):
        frame = render_frame(m, history, epoch, example_X, example_Y,
                              example_pool_idx, pool)
        frames.append(frame)
        last_acc = (history["transform_acc"][-1]
                    if "transform_acc" in history
                    and history["transform_acc"][-1] is not None
                    else float("nan"))
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
              f"acc={last_acc * 100:.1f}%")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    history = train(model, X_tr, Y_tr,
                    n_epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size,
                    eval_fn=eval_fn,
                    eval_every=max(1, args.snapshot_every),
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


if __name__ == "__main__":
    main()
