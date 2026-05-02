"""
Animate MNIST-2x5 subclass distillation training.

Each frame:
  - left panel: static MNIST sample grid grouped into super-class A (digits 0-4)
    and super-class B (digits 5-9) -- the problem definition
  - right panel: teacher 10x10 contingency at the current epoch

This makes the headline visible at a glance: starting from binary super-class
labels, the teacher discovers the latent 5x5 block-diagonal structure that
matches the original 10-way digit identity.

Usage:
  python3 make_mnist_2x5_subclass_gif.py --seed 0 --n-epochs-teacher 12 \
      --fps 4 --out mnist_2x5_subclass.gif
"""

from __future__ import annotations

import argparse
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from mnist_2x5_subclass import (
    MLP,
    auxiliary_distance_loss,
    build_teacher,
    load_mnist,
    relabel_to_superclass,
    superclass_logits_from_sublogits,
    teacher_subclass_contingency,
    teacher_super_loss_and_grad,
)


# ----------------------------------------------------------------------
# Static "problem definition" panel
# ----------------------------------------------------------------------

def make_problem_panel(x: np.ndarray, y: np.ndarray, n_per_class: int = 4,
                       seed: int = 0) -> np.ndarray:
    """Render a (10*n_per_class) sample grid as one image array.

    Rows: 10 digits (5 in super-A, 5 in super-B).
    Cols: n_per_class examples each.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for d in range(10):
        idx = np.where(y == d)[0]
        chosen = rng.choice(idx, size=n_per_class, replace=False)
        imgs = x[chosen].reshape(n_per_class, 28, 28)
        rows.append(np.concatenate(list(imgs), axis=1))
    full = np.concatenate(rows, axis=0)
    return full


def render_problem_axis(ax: plt.Axes, sample_grid: np.ndarray) -> None:
    ax.imshow(sample_grid, cmap="gray_r")
    ax.set_title("Problem: re-label digits 0-4 -> A, 5-9 -> B\n"
                 "(teacher sees ONLY the binary super-class label)",
                 fontsize=10)
    ax.set_xticks([])
    # y-ticks at the centre of each digit row
    rows_per_digit = sample_grid.shape[0] // 10
    ax.set_yticks([rows_per_digit * (d + 0.5) for d in range(10)])
    ax.set_yticklabels([f"{d}  (A)" if d < 5 else f"{d}  (B)"
                        for d in range(10)], fontsize=8)
    # super-class boundary
    ax.axhline(rows_per_digit * 5, color="red", linewidth=2)


def render_contingency_axis(ax: plt.Axes, cont: np.ndarray, epoch: int,
                            n_epochs: int, vmax: int) -> None:
    ax.imshow(cont, cmap="viridis", aspect="auto", vmin=0, vmax=vmax)
    ax.set_title(f"Teacher contingency  (epoch {epoch}/{n_epochs})\n"
                 "rows: cluster k = sub-logit argmax   |   cols: true digit",
                 fontsize=10)
    ax.set_xticks(range(10))
    ax.set_yticks(range(10))
    ax.axhline(4.5, color="white", linewidth=1.5, linestyle="--")
    for k in range(10):
        for c in range(10):
            v = int(cont[k, c])
            if v > 0:
                colour = "white" if v < 0.6 * vmax else "black"
                ax.text(c, k, str(v), ha="center", va="center",
                        color=colour, fontsize=6)


# ----------------------------------------------------------------------
# Frame loop
# ----------------------------------------------------------------------

def render_frame(sample_grid: np.ndarray, cont: np.ndarray, epoch: int,
                 n_epochs: int, vmax: int) -> Image.Image:
    fig, axes = plt.subplots(1, 2, figsize=(10, 5),
                             gridspec_kw={"width_ratios": [1, 1.2]})
    render_problem_axis(axes[0], sample_grid)
    render_contingency_axis(axes[1], cont, epoch, n_epochs, vmax)
    fig.suptitle("MNIST-2x5 subclass distillation  (Mueller, Kornblith & Hinton 2020)",
                 fontsize=11)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=110)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs-teacher", type=int, default=12)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.5)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--n-eval", type=int, default=2000,
                   help="how many test images to use when computing each "
                        "frame's contingency (smaller = faster, less noisy)")
    p.add_argument("--out", type=str, default="mnist_2x5_subclass.gif")
    args = p.parse_args()

    print("# loading MNIST")
    data = load_mnist()
    super_train = relabel_to_superclass(data["y_train"])

    # subset the test set used to compute each frame's contingency
    rng = np.random.default_rng(args.seed)
    eval_idx = rng.choice(len(data["x_test"]), size=args.n_eval, replace=False)
    x_eval = data["x_test"][eval_idx]
    y_eval = data["y_test"][eval_idx]

    print("# preparing problem-definition panel (static)")
    sample_grid = make_problem_panel(data["x_train"], data["y_train"],
                                     n_per_class=4, seed=args.seed)

    print("# building teacher")
    teacher = build_teacher(seed=args.seed, hidden=args.hidden, lr=args.lr)

    print("# training, snapshotting after each epoch")
    n = data["x_train"].shape[0]
    rng_train = np.random.default_rng(args.seed + 1)

    # epoch-0 snapshot (random init)
    cont0 = teacher_subclass_contingency(teacher, x_eval, y_eval)
    vmax = max(int(args.n_eval / 10), int(cont0.max()))
    frames = [render_frame(sample_grid, cont0, 0, args.n_epochs_teacher, vmax)]

    for epoch in range(args.n_epochs_teacher):
        order = rng_train.permutation(n)
        for i in range(0, n, args.batch_size):
            idx = order[i:i + args.batch_size]
            xb = data["x_train"][idx]
            yb = super_train[idx]
            logits = teacher.forward(xb)
            _, d_super = teacher_super_loss_and_grad(logits, yb)
            _, d_aux = auxiliary_distance_loss(logits, yb, sharpen=args.sharpen)
            grads = teacher.backward(d_super + args.aux_weight * d_aux)
            teacher.step(grads)
        cont = teacher_subclass_contingency(teacher, x_eval, y_eval)
        # rescale vmax up if needed (rarely)
        vmax = max(vmax, int(cont.max()))
        frames.append(render_frame(sample_grid, cont, epoch + 1,
                                   args.n_epochs_teacher, vmax))
        print(f"  epoch {epoch+1}/{args.n_epochs_teacher}  cont.max={cont.max()}")

    # write GIF
    out_path = Path(args.out)
    duration_ms = int(1000 / max(args.fps, 1))
    # hold first frame and last frame a little longer so the eye can catch them
    durations = [duration_ms] * len(frames)
    durations[0] = duration_ms * 3
    durations[-1] = duration_ms * 6
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True, disposal=2)
    size_kb = out_path.stat().st_size / 1024
    print(f"# wrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
