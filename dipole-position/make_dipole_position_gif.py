"""
Render an animated GIF showing the dipole-position population code learning
its 2D implicit space.

Layout per frame:
  Top-left:   one example dipole image
  Top-right:  the population activations a (100 units, displayed as a 10x10
              implicit-space grid). The bump-shaped lit region is the code.
  Bottom-left:  scatter of the bottleneck p over the full 56-image universe,
                coloured by true x. As training progresses the cloud spreads.
  Bottom-right: MDL trajectory up to the current epoch.

Usage:
    python3 make_dipole_position_gif.py
    python3 make_dipole_position_gif.py --n-epochs 4000 --snapshot-every 80 \
                                          --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dipole_position import (
    PopulationCoder,
    train,
    all_dipole_positions,
    render_dipole_flat,
    implicit_alignment_r2,
)


def render_frame(model: PopulationCoder, history: dict, epoch: int,
                  example_idx: int, universe: np.ndarray,
                  universe_imgs: np.ndarray) -> Image.Image:
    fig = plt.figure(figsize=(10, 7), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                           hspace=0.45, wspace=0.30)

    # ---- top-left: example dipole image -------------------------------
    ax_img = fig.add_subplot(gs[0, 0])
    img = universe_imgs[example_idx].reshape(model.image_h, model.image_w)
    ax_img.imshow(img, cmap="RdBu_r", vmin=-1.0, vmax=1.0)
    x, y = universe[example_idx]
    ax_img.set_title(f"Example dipole at (x={x}, y={y})", fontsize=10)
    ax_img.set_xticks([]); ax_img.set_yticks([])

    # ---- top-right: population code as 10x10 grid in implicit space ----
    ax_pop = fig.add_subplot(gs[0, 1])
    out = model._encode_internal(universe_imgs[example_idx:example_idx + 1])
    a = out["a"][0]
    p = out["p"][0]
    side = int(round(np.sqrt(model.n_hidden)))
    if side * side == model.n_hidden:
        # Reshape according to grid layout (mu was built with indexing="ij")
        a_grid = a.reshape(side, side)
        ax_pop.imshow(a_grid.T, cmap="hot", vmin=0.0, vmax=1.0,
                       origin="lower", extent=(0, 1, 0, 1))
        ax_pop.plot(p[0], p[1], "c+", markersize=18, markeredgewidth=2.5)
    else:
        sc = ax_pop.scatter(model.mu[:, 0], model.mu[:, 1], c=a,
                              cmap="hot", vmin=0, vmax=1, s=50)
        ax_pop.plot(p[0], p[1], "c+", markersize=18, markeredgewidth=2.5)
    ax_pop.set_title(f"Population code   $p$ = ({p[0]:.2f}, {p[1]:.2f})",
                      fontsize=10)
    ax_pop.set_xlabel("$\\mu_0$"); ax_pop.set_ylabel("$\\mu_1$")
    ax_pop.set_aspect("equal")

    # ---- bottom-left: implicit scatter ---------------------------------
    ax_sc = fig.add_subplot(gs[1, 0])
    p_all = model.encode_position(universe_imgs)
    sc = ax_sc.scatter(p_all[:, 0], p_all[:, 1], c=universe[:, 0],
                         cmap="viridis", s=40, edgecolors="black",
                         linewidths=0.3)
    # Highlight the active example
    ax_sc.plot(p_all[example_idx, 0], p_all[example_idx, 1], "o",
                markerfacecolor="none", markeredgecolor="cyan",
                markersize=15, markeredgewidth=2)
    r2 = (history["implicit_r2"][-1] if history["implicit_r2"] else 0.0)
    ax_sc.set_xlabel("$p_0$"); ax_sc.set_ylabel("$p_1$")
    ax_sc.set_xlim(-0.05, 1.05); ax_sc.set_ylim(-0.05, 1.05)
    ax_sc.grid(alpha=0.3)
    ax_sc.set_title(f"Implicit space (colour = true x).  "
                     f"$R^2$ for linear $p \\to (x, y)$: {r2:.3f}",
                     fontsize=10)

    # ---- bottom-right: MDL trajectory ----------------------------------
    ax_mdl = fig.add_subplot(gs[1, 1])
    if history["epoch"]:
        ax_mdl.plot(history["epoch"], history["mdl_bits"], color="#1f77b4",
                     label="MDL (bits)")
        ax_mdl.plot(history["epoch"],
                     [r * (max(history["mdl_bits"]) or 1.0) for r in history["implicit_r2"]],
                     color="#9467bd", linestyle=":",
                     label="$R^2 \\times$ MDL_max")
        ax_mdl.axvline(epoch, color="black", linewidth=0.8, alpha=0.4)
    ax_mdl.set_xlabel("epoch")
    ax_mdl.set_yscale("log")
    ax_mdl.legend(loc="upper right", fontsize=9)
    ax_mdl.grid(alpha=0.3, which="both")
    ax_mdl.set_title("MDL trajectory", fontsize=10)

    fig.suptitle(f"Dipole-position population coder — epoch {epoch}",
                  fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=4000)
    p.add_argument("--snapshot-every", type=int, default=80)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--out", type=str, default="dipole_position.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    args = p.parse_args()

    universe = all_dipole_positions()
    universe_imgs = render_dipole_flat(universe)

    # Pick a fixed example to highlight in the per-frame visualisation
    rng = np.random.default_rng(args.seed)
    example_idx = int(rng.integers(0, len(universe)))

    frames = []

    def cb(epoch, model, history):
        frame = render_frame(model, history, epoch, example_idx,
                              universe, universe_imgs)
        frames.append(frame)
        r2 = history["implicit_r2"][-1] if history["implicit_r2"] else 0.0
        mdl = history["mdl_bits"][-1] if history["mdl_bits"] else 0.0
        print(f"  frame {len(frames):3d}  epoch {epoch:5d}  "
              f"R^2={r2:.3f}  MDL={mdl:.2f} bits")

    print(f"Training {args.n_epochs} epochs, snapshot every "
           f"{args.snapshot_every}...")
    model, history = train(n_epochs=args.n_epochs, seed=args.seed,
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
