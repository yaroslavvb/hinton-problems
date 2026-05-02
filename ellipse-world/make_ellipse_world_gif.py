"""
Render `ellipse_world.gif` — for each of the 5 classes, an example grid plus
the per-iteration cosine-similarity heatmap of occupied-cell embeddings.

The GIF cycles through attention iterations t = 0, 1, 2, 3 (and back to 0)
so the islands are visible "forming". Each frame is a row of 5 panels (one
per class), with the example grid on the left and the similarity heatmap
on the right.

Usage:
    python3 make_ellipse_world_gif.py --seed 0 --ambiguity 0.4 \
        --out ellipse_world.gif --fps 2

The companion `visualize_ellipse_world.py` script trains the model and saves
parameters to `viz/params.npz` if `--save-params` is set; otherwise this
script trains a fresh model in-memory.
"""

from __future__ import annotations
import argparse
import io
import os
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Ellipse

from PIL import Image

from ellipse_world import (
    CLASSES, N_CLASSES,
    sample_ellipse_layout, apply_affine, random_affine, add_ambiguity,
    render_grid, generate_dataset, train, visualize_islands,
)


def draw_grid(ax, layout, grid_size, mask, title=""):
    ax.set_xlim(-1.05, 1.05)
    ax.set_ylim(-1.05, 1.05)
    ax.set_aspect("equal")
    ax.set_title(title, fontsize=10)
    cell = 2.0 / grid_size
    for k in range(grid_size + 1):
        x = -1 + k * cell
        ax.plot([x, x], [-1, 1], "k-", alpha=0.10, lw=0.5)
        ax.plot([-1, 1], [x, x], "k-", alpha=0.10, lw=0.5)
    cell_coords = -1 + (np.arange(grid_size) + 0.5) * cell
    if mask is not None:
        for i in range(grid_size):
            for j in range(grid_size):
                idx = i * grid_size + j
                if mask[idx] > 0:
                    ax.add_patch(plt.Rectangle(
                        (cell_coords[j] - cell / 2, cell_coords[i] - cell / 2),
                        cell, cell, facecolor="orange", alpha=0.10,
                        edgecolor="none"))
    for cx, cy, a, b, theta in layout:
        e = Ellipse((cx, cy), 2 * a, 2 * b, angle=np.degrees(theta),
                    facecolor="#3060a0", edgecolor="black",
                    alpha=0.75, lw=1.0)
        ax.add_patch(e)
    ax.set_xticks([])
    ax.set_yticks([])


def fig_to_image(fig) -> Image.Image:
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110, bbox_inches="tight")
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--ambiguity", type=float, default=0.4)
    p.add_argument("--grid-size", type=int, default=8)
    p.add_argument("--epochs", type=int, default=15)
    p.add_argument("--n-train", type=int, default=1500)
    p.add_argument("--n-val", type=int, default=300)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--n-iters-train", type=int, default=2)
    p.add_argument("--n-iters-gif", type=int, default=4,
                   help="how many refinement iterations to render in the gif")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--out", default="ellipse_world.gif")
    p.add_argument("--fps", type=float, default=1.6)
    args = p.parse_args()

    # train
    rng = np.random.default_rng(args.seed)
    print(f"# training (ambiguity={args.ambiguity})")
    X_tr, M_tr, y_tr = generate_dataset(args.n_train, args.grid_size,
                                        args.ambiguity, rng)
    X_va, M_va, y_va = generate_dataset(args.n_val, args.grid_size,
                                        args.ambiguity, rng)
    params, _ = train(
        X_tr, M_tr, y_tr, X_va, M_va, y_va,
        n_epochs=args.epochs, batch_size=args.batch_size, lr=args.lr,
        n_iters=args.n_iters_train, alpha=args.alpha, seed=args.seed,
        verbose=False,
    )
    print("# trained.")

    # build one example per class with sims at every iteration
    rng = np.random.default_rng(args.seed + 17)
    examples = []
    for cls in CLASSES:
        layout = sample_ellipse_layout(cls)
        layout = apply_affine(layout, random_affine(rng))
        layout = add_ambiguity(layout, args.ambiguity, rng)
        feats, mask = render_grid(layout, args.grid_size)
        sims, _ = visualize_islands(feats, mask, params,
                                    n_iters=args.n_iters_gif,
                                    alpha=args.alpha)
        examples.append((cls, layout, mask, sims))

    # render frames: one frame per iteration t in [0..n_iters_gif]
    frames = []
    for t in range(args.n_iters_gif + 1):
        fig, axes = plt.subplots(2, N_CLASSES,
                                 figsize=(2.5 * N_CLASSES, 5.0))
        for c, (cls, layout, mask, sims) in enumerate(examples):
            ax_top = axes[0, c]
            ax_bot = axes[1, c]
            draw_grid(ax_top, layout, args.grid_size, mask, title=cls)
            occ = np.where(mask > 0)[0]
            sub = sims[t][np.ix_(occ, occ)]
            im = ax_bot.imshow(sub, vmin=-1, vmax=1, cmap="RdBu_r")
            ax_bot.set_title(f"sim  t={t}", fontsize=9)
            ax_bot.set_xticks(range(len(occ)))
            ax_bot.set_yticks(range(len(occ)))
            ax_bot.set_xticklabels([str(i) for i in range(len(occ))],
                                   fontsize=7)
            ax_bot.set_yticklabels([str(i) for i in range(len(occ))],
                                   fontsize=7)
        fig.suptitle(
            f"eGLOM-lite refinement, iteration t={t} "
            f"(ambiguity={args.ambiguity})",
            y=1.02, fontsize=12)
        fig.tight_layout()
        frames.append(fig_to_image(fig))
        plt.close(fig)

    # hold the last frame longer, then bounce back to t=0 for visual loop
    sequence = list(frames)
    sequence += [frames[-1]] * 2
    sequence += list(reversed(frames[:-1]))

    duration_ms = int(1000 / args.fps)
    sequence[0].save(
        args.out,
        save_all=True,
        append_images=sequence[1:],
        loop=0,
        duration=duration_ms,
        optimize=True,
    )
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"saved {args.out}  ({size_mb:.2f} MB, {len(sequence)} frames)")


if __name__ == "__main__":
    main()
