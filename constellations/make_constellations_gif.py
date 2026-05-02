"""
Render an animated GIF showing the constellations model learning to assign
points to template capsules.

Layout per frame (left to right):
  Left:    a single fixed validation example, points colored by best-matched
           predicted capsule. Decoded points overlaid as triangles.
  Middle:  recovery accuracy over training steps (chance line at 4/11).
  Right:   chamfer loss (train/val) over training steps.

Usage:
    python3 make_constellations_gif.py --n-epochs 30 --snapshot-every 100 --fps 10
"""
from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from constellations import (TEMPLATES, train, make_dataset,
                            part_capsule_recovery_accuracy,
                            _all_permutations)
from visualize_constellations import (CAPSULE_COLORS, TEMPLATE_NAMES,
                                       _best_permutation)


def render_frame(model, history: dict,
                 step: int,
                 fixed_example_x: np.ndarray,
                 fixed_example_y: np.ndarray,
                 max_step: int) -> Image.Image:
    """One example fixed across all frames, plus running training curves."""
    fig = plt.figure(figsize=(11, 4), dpi=100)
    gs = fig.add_gridspec(1, 3, width_ratios=[1.1, 1.0, 1.0],
                          wspace=0.30)

    # --- left: example with colored predictions ---
    ax = fig.add_subplot(gs[0, 0])
    perms, pred_caps, decoded, point_capsule = _best_permutation(
        model, fixed_example_x[None], fixed_example_y[None])
    relabelled = perms[0][pred_caps[0]]                       # (N,)
    correct = relabelled == fixed_example_y
    n_correct = int(correct.sum())

    # Decoded shape underneath, faint.
    for k in range(3):
        mask_d = point_capsule == k
        ax.scatter(decoded[0, mask_d, 0], decoded[0, mask_d, 1],
                   color=CAPSULE_COLORS[k], s=70, marker="^",
                   edgecolor="gray", linewidth=0.4, alpha=0.55, zorder=1)

    # Input points colored by predicted (best-permuted) capsule.
    for k in range(3):
        mask = relabelled == k
        ax.scatter(fixed_example_x[mask, 0], fixed_example_x[mask, 1],
                   color=CAPSULE_COLORS[k], s=110, edgecolor="black",
                   linewidth=0.6, zorder=3,
                   label=TEMPLATE_NAMES[k] if step == max_step else None)
    # Errors
    for n in range(11):
        if not correct[n]:
            ax.scatter(fixed_example_x[n, 0], fixed_example_x[n, 1],
                       color="black", marker="x", s=80, linewidth=2.0, zorder=4)

    all_x = np.concatenate([fixed_example_x[:, 0], decoded[0, :, 0]])
    all_y = np.concatenate([fixed_example_x[:, 1], decoded[0, :, 1]])
    ax.set_xlim(float(all_x.min()) - 0.5, float(all_x.max()) + 0.5)
    ax.set_ylim(float(all_y.min()) - 0.5, float(all_y.max()) + 0.5)
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title(f"Validation example  ({n_correct}/11 correct)", fontsize=10)
    ax.legend(loc="lower left", fontsize=7, framealpha=0.85)

    # --- middle: recovery curve ---
    ax = fig.add_subplot(gs[0, 1])
    if history["epoch"]:
        ax.plot(history["step"], np.array(history["val_recovery"]) * 100,
                color="#2ca02c", linewidth=1.8, marker="o", markersize=3)
    ax.axhline(100 * 4 / 11, color="gray", linestyle="--", linewidth=0.7,
               label="chance (36.4%)")
    ax.axvline(step, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlim(0, max_step)
    ax.set_ylim(0, 100)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("part-capsule recovery (%)", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Recovery accuracy", fontsize=10)

    # --- right: chamfer loss ---
    ax = fig.add_subplot(gs[0, 2])
    if history["epoch"]:
        ax.plot(history["step"], history["loss"], color="#1f77b4",
                linewidth=1.5, label="train")
        ax.plot(history["step"], history["val_loss"], color="#ff7f0e",
                linewidth=1.5, label="val")
    ax.axvline(step, color="black", linewidth=1.0, alpha=0.4)
    ax.set_xlim(0, max_step)
    ax.set_xlabel("step", fontsize=9)
    ax.set_ylabel("symmetric chamfer", fontsize=9)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper right", fontsize=8)
    ax.set_title("Reconstruction loss", fontsize=10)

    fig.suptitle(f"Constellations -- step {step}", fontsize=12, y=1.02)
    fig.tight_layout()

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--snapshot-every", type=int, default=100)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=str, default="constellations.gif")
    p.add_argument("--hold-final", type=int, default=15,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    # Pick a deterministic example to track across the animation.
    fixed_rng = np.random.default_rng(args.seed + 13)
    fixed_x_b, fixed_y_b = make_dataset(1, fixed_rng)
    fixed_x = fixed_x_b[0]
    fixed_y = fixed_y_b[0]

    max_step = args.n_epochs * args.steps_per_epoch
    frames = []

    def cb(step, model, history, val_x, val_y):
        # Periodic snapshot during training. We need the *current*
        # validation loss / recovery for the curves; since `train` only
        # logs at end-of-epoch, derive an instantaneous read here.
        live_history = dict(history)  # shallow copy is fine -- read only
        # Derive an instantaneous live point so the frame's curves don't
        # lag a full epoch behind the model.
        from constellations import ConstellationsModel as _CM  # noqa
        dec, _, _, _ = model.forward(val_x)
        L, _ = model.chamfer_loss_and_dparams(dec, val_x)
        live_recovery = part_capsule_recovery_accuracy(model, (val_x, val_y))
        live_history = {k: list(v) for k, v in history.items()}
        live_history["step"].append(step)
        live_history["epoch"].append(step / args.steps_per_epoch)
        live_history["loss"].append(history["loss"][-1] if history["loss"]
                                     else float(L))
        live_history["val_loss"].append(float(L))
        live_history["val_recovery"].append(float(live_recovery))

        frame = render_frame(model, live_history, step,
                             fixed_x, fixed_y, max_step)
        frames.append(frame)
        print(f"  frame {len(frames):3d}  step {step}  "
              f"recovery={live_recovery*100:.1f}%  chamfer={L:.3f}")

    print(f"Training {args.n_epochs} epochs, snapshot every {args.snapshot_every}...")
    model, history = train(n_epochs=args.n_epochs,
                           steps_per_epoch=args.steps_per_epoch,
                           batch_size=args.batch_size,
                           lr=args.lr,
                           seed=args.seed,
                           snapshot_callback=cb,
                           snapshot_every=args.snapshot_every,
                           verbose=False)
    print(f"Final recovery: {history['val_recovery'][-1]*100:.1f}%   "
          f"final chamfer: {history['val_loss'][-1]:.3f}")

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
