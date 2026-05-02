"""
Render an animated GIF showing the implicit space self-organising during
training of the what / where population coder.

Layout per frame:
  Top-left:    a few sample inputs (h-bars and v-bars)
  Top-right:   2-D scatter of the implicit-space code z, coloured by
               orientation (red = horizontal, blue = vertical).
  Bottom:      training loss curves (recon, MDL) up to the current epoch.

Usage:
    python3 make_dipole_what_where_gif.py
    python3 make_dipole_what_where_gif.py --n-epochs 200 --snapshot-every 4 --fps 12
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from dipole_what_where import (train, visualize_implicit_space,
                                WhatWhereCoder)


# ----------------------------------------------------------------------
# Frame rendering
# ----------------------------------------------------------------------

def _sample_panel(eval_data: dict, n_each: int = 4) -> np.ndarray:
    """Pick a fixed sample of n_each h-bars + n_each v-bars, return as
    one composite image (8x8 patches in a 2 x n_each grid).
    """
    o = eval_data["orient"]
    images = eval_data["images"]
    h_idx = np.where(o == 0)[0][:n_each]
    v_idx = np.where(o == 1)[0][:n_each]
    rows = []
    for idxs in [h_idx, v_idx]:
        row = np.concatenate([images[i].reshape(8, 8) for i in idxs], axis=1)
        rows.append(row)
    return np.concatenate(rows, axis=0)


def render_frame(model: WhatWhereCoder, history: dict, epoch: int,
                 eval_data: dict) -> Image.Image:
    fig = plt.figure(figsize=(10, 6.0), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0], hspace=0.35,
                          wspace=0.30)

    # ---- top-left: sample inputs ----
    ax_s = fig.add_subplot(gs[0, 0])
    panel = _sample_panel(eval_data, n_each=4)
    ax_s.imshow(panel, cmap="gray_r", vmin=0, vmax=1, aspect="equal")
    ax_s.set_xticks([])
    ax_s.set_yticks([])
    ax_s.set_title("Inputs (top: h-bars, bottom: v-bars)", fontsize=10)

    # ---- top-right: implicit-space scatter ----
    ax_z = fig.add_subplot(gs[0, 1])
    info = visualize_implicit_space(model, eval_data["images"],
                                    eval_data["orient"])
    z = info["z"]
    o = info["orient"]
    ax_z.scatter(z[o == 0, 0], z[o == 0, 1], c="#cc3333", s=8, alpha=0.6,
                 edgecolor="none", label="horizontal")
    ax_z.scatter(z[o == 1, 0], z[o == 1, 1], c="#1f77b4", s=8, alpha=0.6,
                 edgecolor="none", label="vertical")
    ax_z.set_xlim(-3.2, 3.2)
    ax_z.set_ylim(-3.2, 3.2)
    ax_z.set_aspect("equal")
    ax_z.set_xlabel("$z_1$", fontsize=9)
    ax_z.set_ylabel("$z_2$", fontsize=9)
    ax_z.grid(alpha=0.3)
    ax_z.legend(loc="upper right", fontsize=8, framealpha=0.85)
    ax_z.set_title(f"Implicit space  "
                   f"(angle={info['axis_angle_deg']:.0f}deg, "
                   f"lin_acc={info['linear_separability']:.2f})",
                   fontsize=10)

    # ---- bottom: training curves ----
    ax_l = fig.add_subplot(gs[1, :])
    ax_l.plot(history["epoch"], history["recon"], color="#cc3333",
              label="reconstruction", linewidth=1.4)
    ax_l.plot(history["epoch"], history["mdl"], color="#1f77b4",
              label="MDL code length", linewidth=1.4)
    ax_l.plot(history["epoch"], history["loss"], color="#444444",
              label="total", linewidth=1.0, alpha=0.8)
    ax_l.axvline(epoch + 1, color="black", linewidth=0.9, alpha=0.4)
    ax_l.set_xlim(0, max(history["epoch"][-1], 1))
    ax_l.set_xlabel("epoch", fontsize=9)
    ax_l.set_ylabel("loss", fontsize=9)
    ax_l.set_ylim(0, 1.6)
    ax_l.grid(alpha=0.3)
    ax_l.legend(loc="upper right", fontsize=8, framealpha=0.85)

    fig.suptitle(f"Dipole what / where — epoch {epoch + 1}", fontsize=11,
                 y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-epochs", type=int, default=150)
    p.add_argument("--snapshot-every", type=int, default=3)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--lambda-mdl", type=float, default=0.05)
    p.add_argument("--sigma-z", type=float, default=0.5)
    p.add_argument("--out", type=str, default="dipole_what_where.gif")
    p.add_argument("--hold-final", type=int, default=18,
                   help="Repeat the last frame this many times.")
    p.add_argument("--max-frames", type=int, default=80,
                   help="Cap on number of frames (subsamples uniformly).")
    args = p.parse_args()

    frames = []

    def cb(epoch: int, model: WhatWhereCoder, history: dict,
           eval_images: np.ndarray, eval_orient: np.ndarray,
           eval_pos: np.ndarray) -> None:
        eval_data = {"images": eval_images, "orient": eval_orient,
                     "position": eval_pos}
        frame = render_frame(model, history, epoch, eval_data)
        frames.append(frame)

    print(f"Training {args.n_epochs} epochs (seed={args.seed}, "
          f"snapshot every {args.snapshot_every})...")
    model, history, eval_data = train(n_epochs=args.n_epochs,
                                      seed=args.seed,
                                      lambda_mdl=args.lambda_mdl,
                                      sigma_z=args.sigma_z,
                                      snapshot_callback=cb,
                                      snapshot_every=args.snapshot_every,
                                      verbose=False)

    # Subsample frames if too many
    if len(frames) > args.max_frames:
        idx = np.linspace(0, len(frames) - 1, args.max_frames).astype(int)
        frames = [frames[i] for i in idx]
    info = visualize_implicit_space(model, eval_data["images"],
                                    eval_data["orient"])
    print(f"Final lin_sep={info['linear_separability']:.2f}  "
          f"axis_angle={info['axis_angle_deg']:.0f}deg")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
