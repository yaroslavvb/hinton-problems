"""
Animated GIF showing the 25-sequence look-up RNN learning to associate
letter sequences with 3-bit codes.

Layout per frame:
  Top-left:    bar chart of held-out test predictions vs target. Each
                 of 5 bars shows the 3-bit signed output; correct bits
                 in green, wrong in red. Per-sequence and overall test
                 accuracy in the title.
  Top-right:   training-loss curve (log scale).
  Bottom-left: train + test accuracy curves.
  Bottom-right: heatmap of W_hh (recurrent matrix), so the viewer can
                 watch the recurrence settle as training progresses.
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sequence_lookup_25 import (SequenceLookupRNN, train, generate_dataset,
                                 ALPHABET_SIZE, SEQ_LEN, N_OUT)


def _seq_label(letters_row, letter_names="ABCDE"):
    return "".join(letter_names[int(l)] for l in letters_row)


def render_frame(model: SequenceLookupRNN, history: dict, data: dict,
                 sweep: int, max_sweep: int, variable_timing: bool,
                 letter_names: str = "ABCDE") -> Image.Image:
    fig = plt.figure(figsize=(10.5, 6.4), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                           hspace=0.55, wspace=0.32)

    # --- top-left: per-test-sequence prediction vs target ---
    ax_pred = fig.add_subplot(gs[0, 0])
    test_idx = data["test_idx"]
    targets = data["targets"][test_idx]
    if variable_timing:
        inp = [data["variable_inputs"][i] for i in test_idx]
        y = np.stack(model.forward_variable(inp)["ys"])
    else:
        y = model.forward(data["one_hot"][test_idx])["y"]
    pred = np.sign(y); pred[pred == 0] = 1.0

    bar_w = 0.27
    x_pos = np.arange(len(test_idx))
    for b in range(N_OUT):
        for j, idx in enumerate(test_idx):
            v = float(y[j, b])
            tgt = float(targets[j, b])
            correct = (np.sign(v) == tgt) or (v == 0 and tgt == 1)
            color = "#2ca02c" if correct else "#d62728"
            ax_pred.bar(j + (b - 1) * bar_w, v, bar_w * 0.9,
                         color=color, edgecolor="black", linewidth=0.4)
            # mark target with a black tick
            ax_pred.plot([j + (b - 1) * bar_w - bar_w * 0.45,
                          j + (b - 1) * bar_w + bar_w * 0.45],
                          [tgt, tgt], color="black", linewidth=1.2)
    ax_pred.axhline(0, color="black", linewidth=0.5)
    ax_pred.set_xticks(x_pos)
    ax_pred.set_xticklabels([_seq_label(data["letters"][i], letter_names)
                              for i in test_idx], fontsize=9)
    ax_pred.set_ylim(-1.25, 1.25)
    ax_pred.set_ylabel("output (tanh)")
    correct_each = np.all(pred == targets, axis=-1)
    n_correct = int(correct_each.sum())
    ax_pred.set_title(f"held-out: {n_correct}/{len(test_idx)} sequences correct  "
                       f"(b0 left, b1 mid, b2 right; black tick = target)",
                       fontsize=10)

    # --- top-right: loss curve ---
    ax_loss = fig.add_subplot(gs[0, 1])
    sw = np.array(history["sweep"])
    if len(sw) > 0:
        ax_loss.plot(sw, history["loss"], color="#9467bd", linewidth=1.3)
    ax_loss.set_xlim(1, max_sweep)
    ax_loss.set_yscale("log")
    ax_loss.set_xlabel("sweep")
    ax_loss.set_ylabel("MSE loss")
    ax_loss.set_title("training loss")
    ax_loss.grid(alpha=0.3, which="both")

    # --- bottom-left: train + test accuracy ---
    ax_acc = fig.add_subplot(gs[1, 0])
    if len(sw) > 0:
        ax_acc.plot(sw, np.array(history["train_acc"]) * 100,
                     color="#1f77b4", label="train (20)", linewidth=1.5)
        ax_acc.plot(sw, np.array(history["test_acc"]) * 100,
                     color="#d62728", label="held-out (5)", linewidth=1.5)
    ax_acc.axhline(80, color="grey", linestyle=":", linewidth=0.7)
    ax_acc.set_xlim(1, max_sweep)
    ax_acc.set_ylim(-3, 105)
    ax_acc.set_xlabel("sweep")
    ax_acc.set_ylabel("all-3-bits accuracy (%)")
    ax_acc.set_title("accuracy")
    ax_acc.legend(fontsize=9, loc="lower right")
    ax_acc.grid(alpha=0.3)

    # --- bottom-right: W_hh heatmap ---
    ax_w = fig.add_subplot(gs[1, 1])
    vmax = max(abs(model.W_hh).max(), 1e-6)
    ax_w.imshow(model.W_hh, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 interpolation="nearest", aspect="auto")
    ax_w.set_title(f"W_hh ({model.n_hidden}x{model.n_hidden}) "
                    f"|max|={vmax:.2f}", fontsize=10)
    ax_w.set_xticks([]); ax_w.set_yticks([])

    fig.suptitle(f"25-sequence look-up RNN  -- sweep {sweep}/{max_sweep}",
                  fontsize=12)

    buf = BytesIO()
    plt.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-seed", type=int, default=0)
    p.add_argument("--variable-timing", action="store_true")
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--n-sweeps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--max-timing", type=int, default=2)
    p.add_argument("--snapshot-every", type=int, default=None)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="sequence_lookup_25.gif")
    args = p.parse_args()

    if args.n_hidden is None:
        args.n_hidden = 60 if args.variable_timing else 30
    if args.n_sweeps is None:
        args.n_sweeps = 2000 if args.variable_timing else 800
    if args.lr is None:
        args.lr = 0.02 if args.variable_timing else 0.05
    if args.init_scale is None:
        args.init_scale = 0.2 if args.variable_timing else 0.5
    if args.grad_clip is None:
        args.grad_clip = 1.0 if args.variable_timing else 5.0
    if args.snapshot_every is None:
        # Aim for ~60 frames
        args.snapshot_every = max(1, args.n_sweeps // 60)

    frames = []

    def cb(sweep_idx, model, history, data):
        frames.append(render_frame(model, history, data,
                                    sweep=sweep_idx + 1,
                                    max_sweep=args.n_sweeps,
                                    variable_timing=args.variable_timing))

    print(f"# training with snapshot_every={args.snapshot_every} "
          f"(sweeps={args.n_sweeps}, hidden={args.n_hidden}, seed={args.seed})")
    model, hist, data = train(n_hidden=args.n_hidden, n_sweeps=args.n_sweeps,
                               lr=args.lr, init_scale=args.init_scale,
                               grad_clip=args.grad_clip, seed=args.seed,
                               dataset_seed=args.dataset_seed,
                               variable_timing=args.variable_timing,
                               max_timing=args.max_timing,
                               snapshot_callback=cb,
                               snapshot_every=args.snapshot_every,
                               verbose=False)

    if not frames:
        raise SystemExit("No frames captured; check snapshot_every")
    print(f"# captured {len(frames)} frames; writing {args.out}")

    # Hold the final frame for emphasis
    hold = max(args.fps, 5)
    frames_to_save = frames + [frames[-1]] * hold

    duration_ms = int(1000.0 / max(args.fps, 1))
    frames_to_save[0].save(args.out, save_all=True,
                            append_images=frames_to_save[1:],
                            duration=duration_ms, loop=0, optimize=True)
    size_mb = os.path.getsize(args.out) / 1e6
    print(f"# wrote {args.out}  ({size_mb:.2f} MB, {len(frames_to_save)} frames "
          f"@ {args.fps} fps)")


if __name__ == "__main__":
    main()
