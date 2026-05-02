"""
Render an animated GIF showing two Imax-trained modules learning to agree
on a disparity readout.

Layout per frame:
  Top-left:    one example random-dot stereo pair (left + right strip),
               with the ground-truth disparity printed
  Top-right:   scatter of (y_a, y_b) over a small held-out batch, colored
               by the ground-truth disparity
  Bottom:      training curves (Imax + corr(y_a, y_b)) up to current epoch
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from random_dot_stereograms import (
    build_two_module_net, train, generate_batch, generate_stereo_pair,
)


def render_frame(mod_a, mod_b, history: dict, epoch: int,
                  example_left: np.ndarray, example_right: np.ndarray,
                  example_d: float,
                  scatter_x_a, scatter_x_b, scatter_d) -> Image.Image:
    fig = plt.figure(figsize=(10, 6), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.0],
                          hspace=0.55, wspace=0.30)

    # ---- top-left: example stereogram ----
    ax_s = fig.add_subplot(gs[0, 0])
    img = np.stack([example_left, example_right], axis=0)
    ax_s.imshow(img, cmap="gray", vmin=-1.2, vmax=1.2,
                aspect="auto", interpolation="nearest")
    ax_s.set_yticks([0, 1])
    ax_s.set_yticklabels(["L", "R"], fontsize=10)
    ax_s.set_xlabel("pixel", fontsize=9)
    ax_s.set_title(f"Example stereo pair  (true disparity d = {example_d:+.2f})",
                   fontsize=10)

    # ---- top-right: y_a vs y_b scatter, colored by d ----
    ax_y = fig.add_subplot(gs[0, 1])
    y_a, _, _ = mod_a.forward(scatter_x_a)
    y_b, _, _ = mod_b.forward(scatter_x_b)
    sc = ax_y.scatter(y_a, y_b, c=scatter_d, cmap="coolwarm",
                       s=12, alpha=0.7, vmin=-3, vmax=3)
    lim = max(1.0, max(abs(y_a).max(), abs(y_b).max())) * 1.05
    ax_y.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.5, alpha=0.4)
    ax_y.set_xlim(-lim, lim)
    ax_y.set_ylim(-lim, lim)
    ax_y.set_xlabel(r"$y_a$", fontsize=9)
    ax_y.set_ylabel(r"$y_b$", fontsize=9)
    cb = fig.colorbar(sc, ax=ax_y, fraction=0.046, pad=0.04)
    cb.set_label("d", fontsize=8)
    cb.ax.tick_params(labelsize=7)
    if len(y_a) > 1 and y_a.std() > 1e-6 and y_b.std() > 1e-6:
        corr_ab = float(np.corrcoef(y_a, y_b)[0, 1])
    else:
        corr_ab = 0.0
    ax_y.set_title(f"Module outputs  (corr(y_a, y_b) = {corr_ab:+.3f})",
                   fontsize=10)
    ax_y.set_aspect("equal")
    ax_y.grid(alpha=0.3)

    # ---- bottom: training curves ----
    ax_t = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax_t.plot(history["epoch"], history["imax"], color="#1f77b4",
                  linewidth=1.5, label="Imax (nats)")
        ax_t.plot(history["epoch"], history["corr_ab"], color="#2ca02c",
                  linewidth=1.5, label=r"corr$(y_a, y_b)$")
        ax_t.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    ax_t.set_xlim(0, max(history["epoch"][-1] if history["epoch"] else 1, 1))
    ax_t.set_ylim(-0.2, 1.6)
    ax_t.set_xlabel("epoch", fontsize=9)
    ax_t.legend(loc="lower right", fontsize=8, framealpha=0.9)
    ax_t.grid(alpha=0.3)

    fig.suptitle(f"Random-dot stereograms (Imax) — epoch {epoch + 1}",
                 fontsize=12, y=0.99)
    # tight_layout is incompatible with the colorbar; use subplots_adjust
    # instead to silence the (cosmetic) warning.
    fig.subplots_adjust(left=0.07, right=0.97, top=0.92, bottom=0.08,
                        hspace=0.55, wspace=0.30)

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--strip-width", type=int, default=10)
    p.add_argument("--max-disparity", type=float, default=3.0)
    p.add_argument("--n-hidden", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--snapshot-every", type=int, default=20)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", type=str, default="random_dot_stereograms.gif")
    p.add_argument("--hold-final", type=int, default=20)
    args = p.parse_args()

    # Pre-allocate a fixed scatter batch + a fixed example stereogram so the
    # animation has a stable reference (only the modules update).
    scatter_rng = np.random.default_rng(args.seed + 91_017)
    scatter_x_a, scatter_x_b, scatter_d = generate_batch(
        scatter_rng, 256, strip_width=args.strip_width,
        max_disparity=args.max_disparity, continuous=True)

    ex_rng = np.random.default_rng(args.seed + 42)
    example_left, example_right, example_d = generate_stereo_pair(
        ex_rng, strip_width=args.strip_width,
        max_disparity=args.max_disparity, disparity=2.0, continuous=True)

    mod_a, mod_b = build_two_module_net(strip_width=args.strip_width,
                                         n_hidden=args.n_hidden,
                                         seed=args.seed,
                                         init_scale=0.5)

    frames = []

    def cb(epoch, m_a, m_b, history):
        frame = render_frame(m_a, m_b, history, epoch,
                              example_left, example_right, example_d,
                              scatter_x_a, scatter_x_b, scatter_d)
        frames.append(frame)
        if len(frames) % 5 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
                  f"Imax={history['imax'][-1]:.3f}")

    print(f"Training {args.n_epochs} epochs, snapshot every "
          f"{args.snapshot_every}...")
    history = train(mod_a, mod_b,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    strip_width=args.strip_width,
                    max_disparity=args.max_disparity,
                    continuous=True,
                    lr=args.lr,
                    momentum=0.9,
                    weight_decay=1e-5,
                    seed=args.seed,
                    snapshot_callback=cb,
                    snapshot_every=args.snapshot_every,
                    verbose=False)

    final_imax = history["imax"][-1]
    final_corr = history["corr_ab"][-1]
    print(f"\nFinal Imax: {final_imax:.3f}   corr(y_a,y_b): {final_corr:+.3f}")

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
