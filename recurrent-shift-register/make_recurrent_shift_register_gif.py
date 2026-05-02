"""
Render an animated GIF showing the shift-matrix structure emerging in W_hh.

Layout per frame:
  Top-left:    heatmap of W_hh, with the *currently* discovered chain
                 outlined in lime. As training progresses, the random
                 dense matrix collapses to N - 1 strong entries.
  Top-right:   bar chart of |W_hh| entries pooled, sorted descending.
                 At convergence: ~N - 1 entries with magnitude > 1, the
                 rest at zero (the L1 plateau is visible).
  Bottom:      training curves -- masked MSE loss, overall accuracy, and
                 the sparsity ratio (off-chain / chain).

Usage:
    python3 make_recurrent_shift_register_gif.py --n-units 3 --seed 0
    python3 make_recurrent_shift_register_gif.py --n-units 5 --seed 6
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from recurrent_shift_register import (ShiftRegisterRNN, train,
                                        shift_matrix_score)


def render_frame(model: ShiftRegisterRNN, history: dict,
                 epoch: int, max_epoch: int) -> Image.Image:
    fig = plt.figure(figsize=(10.5, 6.4), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                           hspace=0.45, wspace=0.30)

    sm = shift_matrix_score(model.W_hh)
    N = model.n_units

    # ---- top-left: W_hh heatmap with chain outline ----
    ax_w = fig.add_subplot(gs[0, 0])
    vmax = max(abs(model.W_hh).max(), 0.4)
    im = ax_w.imshow(model.W_hh, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                      interpolation="nearest")
    for r, c in sm["chain_positions"]:
        ax_w.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                      fill=False, edgecolor="lime",
                                      linewidth=2.0))
    s = sm["input_stage"]
    ax_w.add_patch(plt.Rectangle((-0.5, s - 0.5), N, 1.0,
                                  fill=False, edgecolor="gray",
                                  linewidth=0.8, linestyle="--"))
    for i in range(N):
        for j in range(N):
            v = model.W_hh[i, j]
            if abs(v) > 0.3:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax_w.text(j, i, f"{v:+.2f}", ha="center", va="center",
                           fontsize=8, color=color)
    ax_w.set_xticks(range(N))
    ax_w.set_yticks(range(N))
    ax_w.set_xticklabels([f"h[{j}]" for j in range(N)], fontsize=8)
    ax_w.set_yticklabels([f"h[{i}]" for i in range(N)], fontsize=8)
    ax_w.set_title(f"$W_{{hh}}$  "
                    f"(chain |w|={sm['shift_diag_mean']:.2f}, "
                    f"leak={sm['shift_leak_max']:.2f})",
                    fontsize=10)
    fig.colorbar(im, ax=ax_w, fraction=0.046, pad=0.04)

    # ---- top-right: sorted-magnitude bar chart ----
    ax_b = fig.add_subplot(gs[0, 1])
    flat = np.sort(np.abs(model.W_hh).ravel())[::-1]
    colors = ["#2ca02c"] * (N - 1) + ["#d62728"] * (len(flat) - (N - 1))
    ax_b.bar(np.arange(len(flat)), flat,
              color=colors, edgecolor="black", linewidth=0.4)
    ax_b.axhline(0, color="black", linewidth=0.5)
    ax_b.axvline(N - 1.5, color="gray", linestyle=":", linewidth=1.0)
    ax_b.text(N - 1.6, ax_b.get_ylim()[1] * 0.9,
               "  N - 1 chain entries  |  off-chain", fontsize=8,
               ha="left", color="dimgray")
    ax_b.set_xlabel("rank", fontsize=9)
    ax_b.set_ylabel("|w|", fontsize=9)
    ax_b.set_title("Sorted $|W_{hh}|$  (target: top N - 1 large, rest zero)",
                    fontsize=10)
    ax_b.set_xticks(range(0, N * N, max(1, N * N // 6)))
    ax_b.set_ylim(0, max(0.5, ax_b.get_ylim()[1]))
    ax_b.grid(alpha=0.3, axis="y")

    # ---- bottom: training curves ----
    ax_t = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["loss"],
                   color="#9467bd", linewidth=1.3, label="MSE loss")
        ax_t.set_yscale("log")
        if history["converged_epoch"] is not None:
            ax_t.axvline(history["converged_epoch"], color="green",
                          linestyle="--", linewidth=0.9, alpha=0.7)
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max_epoch)
    ax_t.set_xlabel("sweep", fontsize=9)
    ax_t.set_ylabel("MSE loss", fontsize=9, color="#9467bd")
    ax_t.tick_params(axis="y", colors="#9467bd")
    ax_t.grid(alpha=0.3)

    if history["epoch"]:
        ax_r = ax_t.twinx()
        acc = np.array(history["accuracy"]) * 100
        sr = np.array(history["sparsity_ratio"])
        ax_r.plot(history["epoch"], acc,
                   color="#1f77b4", linewidth=1.2, alpha=0.85,
                   label="accuracy %")
        ax_r.plot(history["epoch"], 100 * sr,
                   color="#d62728", linewidth=1.2, alpha=0.85,
                   label="sparsity ratio (% x 100)")
        ax_r.axhline(20, color="gray", linestyle=":", linewidth=0.9,
                      alpha=0.7)
        ax_r.set_ylim(0, 105)
        ax_r.set_ylabel("accuracy / sparsity-ratio", fontsize=9)
        ax_r.legend(fontsize=8, loc="lower right")

    acc_now = history["accuracy"][-1] * 100 if history["accuracy"] else 0
    fig.suptitle(f"recurrent shift register, N = {N}  |  "
                  f"sweep {epoch + 1}  |  acc {acc_now:.0f}%  |  "
                  f"is shift matrix? {sm['is_shift_matrix']}",
                  fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-units", type=int, default=3, choices=[3, 5, 4, 6, 7, 8])
    p.add_argument("--seed", type=int, default=0,
                    help="N=3 default seed=0 (conv@89); N=5 try seed=6 (conv@121)")
    p.add_argument("--n-sweeps", type=int, default=300)
    p.add_argument("--sequence-len", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--l1-W-hh", type=float, default=0.05)
    p.add_argument("--init-scale", type=float, default=0.2)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--fps", type=int, default=14)
    p.add_argument("--out", type=str, default="recurrent_shift_register.gif")
    p.add_argument("--hold-final", type=int, default=20)
    p.add_argument("--max-frame-side", type=int, default=900)
    args = p.parse_args()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, args.n_sweeps)
        if max(frame.size) > args.max_frame_side:
            scale = args.max_frame_side / max(frame.size)
            new_size = (int(frame.size[0] * scale),
                         int(frame.size[1] * scale))
            frame = frame.resize(new_size, Image.LANCZOS)
        frames.append(frame)

    print(f"Training N={args.n_units} shift register, "
           f"{args.n_sweeps} sweeps, seed={args.seed}, "
           f"snapshot every {args.snapshot_every}...")
    model, history = train(n_units=args.n_units, n_sweeps=args.n_sweeps,
                            batch_size=args.batch_size,
                            sequence_len=args.sequence_len,
                            lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            l1_W_hh=args.l1_W_hh,
                            init_scale=args.init_scale,
                            seed=args.seed,
                            snapshot_callback=cb,
                            snapshot_every=args.snapshot_every,
                            verbose=False)
    print(f"  converged @ sweep {history['converged_epoch']},  "
           f"final acc {history['accuracy'][-1]*100:.0f}%,  "
           f"frames captured: {len(frames)}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    palette_frame = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    frames_q = [f.quantize(colors=128, method=Image.MEDIANCUT,
                            palette=palette_frame) for f in frames]
    frames_q[0].save(args.out, save_all=True, append_images=frames_q[1:],
                      duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
