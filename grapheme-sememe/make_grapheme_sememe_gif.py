"""
Render an animated GIF of the 4-stage grapheme-sememe protocol.

Per frame the layout is:
  Top-left  : sememe activations of the 2 held-out words (target vs predicted)
  Top-right : sememe activations of 4 of the trained words (sample)
  Bottom    : 4-stage timeline of bit accuracy on trained-18 vs held-out-2
              (vertical line = current cycle; orange band = lesion stage)

Usage:
    python3 make_grapheme_sememe_gif.py
    python3 make_grapheme_sememe_gif.py --seed 0 --snapshot-every 25 --fps 8
"""

from __future__ import annotations
import argparse
import copy
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from grapheme_sememe import (
    GraphemeSememeMLP, generate_mapping, train, lesion, relearn_subset,
    accuracy_bitwise, accuracy_pattern, loss_bce,
)


COLOR_TRAIN = "#1f77b4"
COLOR_HELD = "#d62728"
COLOR_LESION = "#ff7f0e"


def render_frame(model: GraphemeSememeMLP,
                 X: np.ndarray, Y: np.ndarray,
                 held_idx: list[int],
                 sample_train_idx: list[int],
                 history: dict,
                 stage: str,
                 cycle: int,
                 lesion_cycle: int | None,
                 ) -> Image.Image:
    fig = plt.figure(figsize=(10, 6.5), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.1],
                          hspace=0.55, wspace=0.30)
    n_sememes = Y.shape[1]
    bits = np.arange(n_sememes)

    # ---- top-left: held-out 2 (target vs prediction) ----
    ax_h = fig.add_subplot(gs[0, 0])
    for r, idx in enumerate(held_idx):
        target = Y[idx]
        pred = model.predict(X[idx:idx+1])[0]
        ax_h.bar(bits + r * (n_sememes + 2), target, width=0.85,
                 color="#bbbbbb", edgecolor="#666666", linewidth=0.3,
                 label="target" if r == 0 else None)
        ax_h.bar(bits + r * (n_sememes + 2), pred, width=0.55,
                 color=COLOR_HELD, alpha=0.85,
                 label="prediction" if r == 0 else None)
        ax_h.text(r * (n_sememes + 2) + n_sememes / 2 - 0.5, 1.10,
                  f"held-out {idx}", ha="center", va="bottom", fontsize=9)
    ax_h.set_ylim(0, 1.20)
    ax_h.set_xticks([])
    ax_h.set_ylabel("activation")
    ax_h.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax_h.set_title("Held-out 2 sememes (never retrained after lesion)",
                   fontsize=10)

    # ---- top-right: 4 sample trained patterns ----
    ax_t = fig.add_subplot(gs[0, 1])
    for r, idx in enumerate(sample_train_idx):
        target = Y[idx]
        pred = model.predict(X[idx:idx+1])[0]
        ax_t.bar(bits + r * (n_sememes + 2), target, width=0.85,
                 color="#bbbbbb", edgecolor="#666666", linewidth=0.3,
                 label="target" if r == 0 else None)
        ax_t.bar(bits + r * (n_sememes + 2), pred, width=0.55,
                 color=COLOR_TRAIN, alpha=0.85,
                 label="prediction" if r == 0 else None)
        ax_t.text(r * (n_sememes + 2) + n_sememes / 2 - 0.5, 1.10,
                  f"word {idx}", ha="center", va="bottom", fontsize=9)
    ax_t.set_ylim(0, 1.20)
    ax_t.set_xticks([])
    ax_t.legend(loc="upper right", fontsize=8, framealpha=0.9, ncol=2)
    ax_t.set_title("Trained-18 sememes (sample of 4)", fontsize=10)

    # ---- bottom: 4-stage timeline ----
    ax = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        eps = np.array(history["epoch"], dtype=float)
        at = np.array(history["acc_bit_trained"]) * 100
        ah = np.array(history["acc_bit_held_out"]) * 100
        ax.plot(eps, at, color=COLOR_TRAIN, linewidth=1.5,
                label="trained 18 bit acc")
        ax.plot(eps, ah, color=COLOR_HELD, linewidth=1.5,
                label="held-out 2 bit acc")
    if lesion_cycle is not None:
        ax.axvline(lesion_cycle, color=COLOR_LESION,
                   linewidth=1.2, linestyle="--", label="lesion")
    ax.axvline(cycle, color="black", linewidth=1.0, alpha=0.5)

    # Stage banner
    txt_stage = {"train": "Stage 1: training on all 20",
                 "lesion": "Stage 2: lesion 50% of W1+W2",
                 "relearn": "Stage 3: relearning on 18 of 20"}
    ax.set_title(f"{txt_stage.get(stage, stage)}   |   "
                 f"cycle {cycle}", fontsize=10)

    ax.set_xlabel("training cycle")
    ax.set_ylabel("bit accuracy (%)")
    ax.set_ylim(40, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.9)

    fig.suptitle("Grapheme-sememe — 4-stage protocol "
                 "(Hinton & Sejnowski 1986)", fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lesion-fraction", type=float, default=0.5)
    p.add_argument("--n-train-cycles", type=int, default=600)
    p.add_argument("--n-relearn-cycles", type=int, default=80)
    p.add_argument("--snapshot-every", type=int, default=15)
    p.add_argument("--fps", type=int, default=8)
    p.add_argument("--out", type=str, default="grapheme_sememe.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    X, Y = generate_mapping(seed=args.seed)
    held_idx = list(range(len(X) - 2, len(X)))
    train_idx = [i for i in range(len(X)) if i not in held_idx]
    sample_train_idx = list(rng.choice(train_idx, size=4, replace=False))

    model = GraphemeSememeMLP(seed=args.seed, init_scale=0.5)
    history = {"phase": [], "epoch": [], "loss": [], "acc_bit": [],
               "acc_pattern": [], "weight_norm": [],
               "acc_pattern_trained": [], "acc_bit_trained": [],
               "acc_pattern_held_out": [], "acc_bit_held_out": []}
    history_eval = {"trained": (X[train_idx], Y[train_idx]),
                    "held_out": (X[held_idx], Y[held_idx])}

    frames: list[Image.Image] = []

    # ---- Stage 1: train on all 20 ----
    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                                ("W2", model.W2), ("b2", model.b2)]}
    from grapheme_sememe import train_step
    for cycle in range(args.n_train_cycles):
        train_step(model, X, Y, velocities, lr=0.3, momentum=0.5,
                   weight_decay=1e-3)
        # log every cycle (cheap; small history)
        history["phase"].append("train")
        history["epoch"].append(cycle + 1)
        history["loss"].append(loss_bce(model, X, Y))
        history["acc_bit"].append(accuracy_bitwise(model, X, Y))
        history["acc_pattern"].append(accuracy_pattern(model, X, Y))
        history["weight_norm"].append(
            float(np.linalg.norm(model.W1)) + float(np.linalg.norm(model.W2)))
        history["acc_pattern_trained"].append(
            accuracy_pattern(model, X[train_idx], Y[train_idx]))
        history["acc_bit_trained"].append(
            accuracy_bitwise(model, X[train_idx], Y[train_idx]))
        history["acc_pattern_held_out"].append(
            accuracy_pattern(model, X[held_idx], Y[held_idx]))
        history["acc_bit_held_out"].append(
            accuracy_bitwise(model, X[held_idx], Y[held_idx]))

        if (cycle % args.snapshot_every == 0
                or cycle == args.n_train_cycles - 1):
            frames.append(render_frame(
                model, X, Y, held_idx, sample_train_idx, history,
                "train", cycle + 1, lesion_cycle=None))
            print(f"  frame {len(frames):3d}  stage=train cycle={cycle+1}  "
                  f"acc_held={history['acc_bit_held_out'][-1]*100:.1f}%")

    # ---- Stage 2: lesion (single instantaneous event; render a few frames
    # holding the lesion for emphasis) ----
    mask = lesion(model, fraction=args.lesion_fraction, seed=args.seed + 1)
    lesion_cycle = history["epoch"][-1] if history["epoch"] else 0
    history["phase"].append("lesion")
    history["epoch"].append(lesion_cycle)
    history["loss"].append(loss_bce(model, X, Y))
    history["acc_bit"].append(accuracy_bitwise(model, X, Y))
    history["acc_pattern"].append(accuracy_pattern(model, X, Y))
    history["weight_norm"].append(
        float(np.linalg.norm(model.W1)) + float(np.linalg.norm(model.W2)))
    history["acc_pattern_trained"].append(
        accuracy_pattern(model, X[train_idx], Y[train_idx]))
    history["acc_bit_trained"].append(
        accuracy_bitwise(model, X[train_idx], Y[train_idx]))
    history["acc_pattern_held_out"].append(
        accuracy_pattern(model, X[held_idx], Y[held_idx]))
    history["acc_bit_held_out"].append(
        accuracy_bitwise(model, X[held_idx], Y[held_idx]))
    for _ in range(6):
        frames.append(render_frame(
            model, X, Y, held_idx, sample_train_idx, history,
            "lesion", lesion_cycle, lesion_cycle=lesion_cycle))
    print(f"  frames {len(frames)-5}..{len(frames)}  stage=lesion "
          f"acc_held={history['acc_bit_held_out'][-1]*100:.1f}%")

    # ---- Stage 3: relearn on 18 ----
    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                                ("W2", model.W2), ("b2", model.b2)]}
    Xs, Ys = X[train_idx], Y[train_idx]
    for cycle in range(args.n_relearn_cycles):
        train_step(model, Xs, Ys, velocities, lr=0.3, momentum=0.5,
                   weight_decay=1e-3, weight_mask=mask)
        epoch = lesion_cycle + cycle + 1
        history["phase"].append("relearn")
        history["epoch"].append(epoch)
        history["loss"].append(loss_bce(model, X, Y))
        history["acc_bit"].append(accuracy_bitwise(model, X, Y))
        history["acc_pattern"].append(accuracy_pattern(model, X, Y))
        history["weight_norm"].append(
            float(np.linalg.norm(model.W1)) + float(np.linalg.norm(model.W2)))
        history["acc_pattern_trained"].append(
            accuracy_pattern(model, X[train_idx], Y[train_idx]))
        history["acc_bit_trained"].append(
            accuracy_bitwise(model, X[train_idx], Y[train_idx]))
        history["acc_pattern_held_out"].append(
            accuracy_pattern(model, X[held_idx], Y[held_idx]))
        history["acc_bit_held_out"].append(
            accuracy_bitwise(model, X[held_idx], Y[held_idx]))

        if (cycle % max(args.snapshot_every // 4, 1) == 0
                or cycle == args.n_relearn_cycles - 1):
            frames.append(render_frame(
                model, X, Y, held_idx, sample_train_idx, history,
                "relearn", epoch, lesion_cycle=lesion_cycle))
            print(f"  frame {len(frames):3d}  stage=relearn cycle={epoch}  "
                  f"acc_held={history['acc_bit_held_out'][-1]*100:.1f}%")

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
