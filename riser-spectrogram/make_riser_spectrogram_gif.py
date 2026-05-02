"""
Render an animated GIF for the synthetic-spectrogram riser/non-riser task.

Layout per frame:
  Top-left:    Six example noisy inputs (3 rising, 3 falling) with the
               clean track overlaid -- shows the *problem*.
  Top-right:   The 24 hidden-unit filters as 6 x 9 (freq x time) grids.
               Watches them sharpen into oriented, time-frequency edges
               that the network uses to score "rising" vs "falling".
  Bottom:      Train + test accuracy vs epoch, with the Bayes-optimal
               ceiling drawn as a dashed horizontal line.

Usage:
    python3 make_riser_spectrogram_gif.py
    python3 make_riser_spectrogram_gif.py --epochs 200 --snapshot-every 4 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from riser_spectrogram import (
    N_FREQ, N_TIME, N_HIDDEN, DEFAULT_NOISE_STD,
    sample_rising_tracks, sample_falling_tracks, tracks_to_specs,
    train, accuracy, bayes_optimal_accuracy,
)


def _example_inputs(noise_std: float, seed: int):
    rng = np.random.default_rng(seed)
    rise = sample_rising_tracks(rng, 3)
    fall = sample_falling_tracks(rng, 3)
    tracks = np.concatenate([rise, fall], axis=0)
    cleans = tracks_to_specs(tracks)
    noise = noise_std * rng.standard_normal(cleans.shape)
    return tracks, cleans + noise


def render_frame(model, history, epoch, examples_imgs, examples_tracks,
                 bayes_acc, max_epochs):
    fig = plt.figure(figsize=(11, 6.5), dpi=92)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.6, 1.0],
                          width_ratios=[1.0, 1.6],
                          hspace=0.45, wspace=0.25)

    # ---- top-left: example inputs ----
    ax_ex = fig.add_subplot(gs[0, 0])
    n_ex = examples_imgs.shape[0]
    panel_h, panel_w = N_FREQ + 1, N_TIME + 1
    rows, cols = 2, 3
    grid = np.full((rows * panel_h, cols * panel_w), np.nan)
    for k in range(n_ex):
        r, c = k // cols, k % cols
        grid[r * panel_h:r * panel_h + N_FREQ,
             c * panel_w:c * panel_w + N_TIME] = examples_imgs[k]
    vmax = float(np.nanmax(examples_imgs))
    vmin = float(np.nanmin(examples_imgs))
    ax_ex.imshow(grid, aspect="equal", origin="lower",
                 vmin=vmin, vmax=vmax, cmap="magma",
                 interpolation="nearest")
    for k in range(n_ex):
        r, c = k // cols, k % cols
        x_off = c * panel_w
        y_off = r * panel_h
        ax_ex.plot(x_off + np.arange(N_TIME), y_off + examples_tracks[k],
                   color="cyan", linewidth=0.8, alpha=0.7)
        label = "rising" if k < 3 else "falling"
        ax_ex.text(x_off + N_TIME / 2, y_off - 0.6, label,
                   color="white", fontsize=7, ha="center",
                   bbox=dict(facecolor="black", alpha=0.5,
                             pad=1.2, edgecolor="none"))
    ax_ex.set_xticks([])
    ax_ex.set_yticks([])
    ax_ex.set_title("Inputs (clean track in cyan)", fontsize=10)

    # ---- top-right: hidden filters ----
    ax_h = fig.add_subplot(gs[0, 1])
    n_h = model.W1.shape[0]
    n_cols = 6
    n_rows = (n_h + n_cols - 1) // n_cols
    cell_h = N_FREQ + 1
    cell_w = N_TIME + 1
    big = np.full((n_rows * cell_h, n_cols * cell_w), 0.0)
    vmax_w = max(float(np.abs(model.W1).max()), 1e-3)
    for i in range(n_h):
        r, c = i // n_cols, i % n_cols
        big[r * cell_h:r * cell_h + N_FREQ,
            c * cell_w:c * cell_w + N_TIME] = (
                model.W1[i].reshape(N_FREQ, N_TIME))
    ax_h.imshow(big, aspect="equal", origin="lower",
                vmin=-vmax_w, vmax=vmax_w, cmap="RdBu_r",
                interpolation="nearest")
    ax_h.set_xticks([])
    ax_h.set_yticks([])
    ax_h.set_title(
        f"24 hidden filters (6 freq x 9 time)  |W1|={np.linalg.norm(model.W1):.1f}",
        fontsize=10)

    # ---- bottom: training curves ----
    ax_acc = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_acc.plot(history["epoch"],
                    np.array(history["train_acc"]) * 100,
                    color="#1f77b4", linewidth=1.4, label="train")
        ax_acc.plot(history["epoch"],
                    np.array(history["test_acc"]) * 100,
                    color="#d62728", linewidth=1.4, label="test")
        ax_acc.axhline(bayes_acc * 100, color="black",
                       linestyle="--", linewidth=1.0, alpha=0.7,
                       label=f"Bayes-opt ({bayes_acc*100:.2f}%)")
        ax_acc.axvline(epoch + 1, color="gray", linewidth=0.7, alpha=0.4)
    ax_acc.set_xlim(0, max_epochs)
    ax_acc.set_ylim(85, 101)
    ax_acc.set_xlabel("epoch", fontsize=9)
    ax_acc.set_ylabel("accuracy (%)", fontsize=9)
    ax_acc.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_acc.grid(alpha=0.3)

    fig.suptitle(f"Riser-spectrogram (P&H 1987)  -  epoch {epoch + 1}",
                 fontsize=12, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=92, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE,
                                    colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--epochs", type=int, default=160)
    p.add_argument("--snapshot-every", type=int, default=4)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--out", type=str, default="riser_spectrogram.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat last frame this many times.")
    args = p.parse_args()

    print("Computing Bayes-optimal ceiling...")
    bayes_acc = bayes_optimal_accuracy(args.noise_std,
                                        n_samples=30_000,
                                        seed=args.seed + 7919)
    print(f"  bayes={bayes_acc*100:.2f}%")

    examples_tracks, examples_imgs = _example_inputs(args.noise_std,
                                                      args.seed + 1)

    frames = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch,
                              examples_imgs, examples_tracks,
                              bayes_acc, args.epochs)
        frames.append(frame)
        if len(frames) % 5 == 0 or len(frames) <= 3:
            print(f"  frame {len(frames):3d}  epoch {epoch+1}  "
                  f"test={history['test_acc'][-1]*100:.2f}%")

    print(f"Training {args.epochs} epochs, snapshot every "
          f"{args.snapshot_every}...")
    model, hist, _, _, X_test, y_test = train(
        n_train=args.n_train,
        n_test=args.n_test,
        n_sweeps=args.epochs,
        noise_std=args.noise_std,
        seed=args.seed,
        snapshot_callback=cb,
        snapshot_every=args.snapshot_every,
        verbose=False,
    )
    final_acc = accuracy(model, X_test, y_test)
    print(f"Final test acc: {final_acc*100:.2f}%  "
          f"(Bayes {bayes_acc*100:.2f}%, gap {(bayes_acc-final_acc)*100:+.2f} pp)")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True,
                   append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
