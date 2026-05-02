"""
Static visualisations for the trained what / where population coder.

Outputs (in `viz/`):
  example_images.png    - 8 horizontal and 8 vertical bar samples
  implicit_space.png    - 2-D scatter of z, coloured by orientation
  mdl_trajectory.png    - reconstruction loss + MDL code length per epoch
  decoder_grid.png      - decoder reconstructions on a regular grid in z
  weights.png           - encoder W1 receptive fields (10x10 panel of 8x8)
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from dipole_what_where import (train, generate_bars, visualize_implicit_space,
                                WhatWhereCoder)


# ----------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------

def plot_example_images(out_path: str, seed: int = 0, h: int = 8, w: int = 8,
                        n_each: int = 8) -> None:
    rng = np.random.default_rng(seed)
    images_h, _, pos_h = generate_bars(n_each, rng=rng)  # likely mixed
    # Force balanced examples by drawing each orientation separately.
    rng_h = np.random.default_rng(seed)
    rng_v = np.random.default_rng(seed + 1)
    h_imgs = np.zeros((n_each, h * w), dtype=np.float32)
    v_imgs = np.zeros((n_each, h * w), dtype=np.float32)
    h_pos = np.zeros(n_each, dtype=np.float32)
    v_pos = np.zeros(n_each, dtype=np.float32)
    for i in range(n_each):
        # horizontal
        ys = float(rng_h.uniform(0.0, h - 1.0))
        prof = np.exp(-0.5 * ((np.arange(h) - ys) / 0.7) ** 2)
        h_imgs[i] = (prof[:, None] * np.ones(w, dtype=np.float32)).ravel()
        h_pos[i] = ys
        # vertical
        xs = float(rng_v.uniform(0.0, w - 1.0))
        prof = np.exp(-0.5 * ((np.arange(w) - xs) / 0.7) ** 2)
        v_imgs[i] = (np.ones(h, dtype=np.float32)[:, None] * prof[None, :]).ravel()
        v_pos[i] = xs

    fig, axes = plt.subplots(2, n_each, figsize=(n_each * 1.0, 2.2), dpi=140)
    for j in range(n_each):
        axes[0, j].imshow(h_imgs[j].reshape(h, w), cmap="gray_r",
                          vmin=0, vmax=1)
        axes[0, j].set_title(f"H y={h_pos[j]:.1f}", fontsize=7)
        axes[0, j].axis("off")
        axes[1, j].imshow(v_imgs[j].reshape(h, w), cmap="gray_r",
                          vmin=0, vmax=1)
        axes[1, j].set_title(f"V x={v_pos[j]:.1f}", fontsize=7)
        axes[1, j].axis("off")
    fig.suptitle("Sample inputs: 8 horizontal bars (top) / 8 vertical bars (bottom)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_implicit_space(model: WhatWhereCoder, eval_data: dict,
                        out_path: str) -> None:
    info = visualize_implicit_space(model, eval_data["images"],
                                    eval_data["orient"])
    z = info["z"]
    o = info["orient"]
    pos = eval_data["position"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 5.0), dpi=140)

    ax = axes[0]
    ax.scatter(z[o == 0, 0], z[o == 0, 1], c="#cc3333", s=14,
               alpha=0.7, edgecolor="none", label="horizontal bar")
    ax.scatter(z[o == 1, 0], z[o == 1, 1], c="#1f77b4", s=14,
               alpha=0.7, edgecolor="none", label="vertical bar")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.legend(loc="best", framealpha=0.85)
    ax.set_title(f"Implicit space coloured by orientation\n"
                 f"axis angle = {info['axis_angle_deg']:.0f} deg, "
                 f"linear-probe acc = {info['linear_separability']:.2f}",
                 fontsize=10)

    ax = axes[1]
    sc = ax.scatter(z[:, 0], z[:, 1], c=pos, cmap="viridis", s=14, alpha=0.8,
                    edgecolor="none")
    ax.set_xlabel("$z_1$")
    ax.set_ylabel("$z_2$")
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)
    ax.set_title("Same scatter coloured by within-class position", fontsize=10)
    fig.colorbar(sc, ax=ax, shrink=0.7, label="bar centre (y for H, x for V)")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mdl_trajectory(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)

    ax = axes[0]
    ax.plot(history["epoch"], history["recon"], color="#cc3333",
            label="reconstruction (mean BCE / pixel)")
    ax.plot(history["epoch"], history["mdl"], color="#1f77b4",
            label="MDL code length (0.5 ||z||^2 per dim)")
    ax.plot(history["epoch"], history["loss"], color="#444444",
            label="total loss", linewidth=0.9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.set_title("Description-length trajectory")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(history["epoch"], history["linear_separability"], color="#2ca02c")
    ax.axhline(0.5, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("ridge linear-probe acc")
    ax.set_ylim(0.4, 1.0)
    ax.set_title("Linear separability of orientation")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(history["epoch"], history["axis_angle_deg"], color="#9467bd")
    ax.axhline(90, color="grey", linestyle=":", linewidth=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("angle between H and V principal axes (deg)")
    ax.set_ylim(0, 95)
    ax.set_title("What / where axis orthogonality")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_decoder_grid(model: WhatWhereCoder, out_path: str,
                      grid_n: int = 9, span: float = 2.5) -> None:
    """Sweep z over a `grid_n`x`grid_n` lattice and decode each point.

    Visualises what each region of implicit space "looks like" once
    decoded. With successful what/where structure, one axis should
    morph through the horizontal-bar family and the other through
    the vertical-bar family.
    """
    coords = np.linspace(-span, span, grid_n, dtype=np.float32)
    zs = np.array([[x, y] for y in coords[::-1] for x in coords],
                  dtype=np.float32)
    x_hat, _, _ = model.decode(zs)
    fig, axes = plt.subplots(grid_n, grid_n, figsize=(grid_n, grid_n), dpi=120)
    for k in range(grid_n * grid_n):
        ax = axes.ravel()[k]
        ax.imshow(x_hat[k].reshape(8, 8), cmap="gray_r", vmin=0, vmax=1)
        ax.axis("off")
    fig.suptitle(f"Decoder sweep over implicit space "
                 f"(z in [-{span}, +{span}]^2)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_encoder_weights(model: WhatWhereCoder, out_path: str) -> None:
    """Show the encoder's first-layer receptive fields as 8x8 patches."""
    W = model.W1                                      # (64, 100)
    side = int(round(np.sqrt(W.shape[1])))
    fig, axes = plt.subplots(side, side, figsize=(side, side), dpi=120)
    vmax = float(abs(W).max() + 1e-6)
    for k in range(side * side):
        ax = axes.ravel()[k]
        ax.imshow(W[:, k].reshape(8, 8), cmap="seismic",
                  vmin=-vmax, vmax=vmax)
        ax.axis("off")
    fig.suptitle("Encoder W1 receptive fields (one 8x8 panel per hidden unit)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Driver
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-epochs", type=int, default=150)
    p.add_argument("--lambda-mdl", type=float, default=0.05)
    p.add_argument("--sigma-z", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training {args.n_epochs} epochs (seed={args.seed}, "
          f"lambda_mdl={args.lambda_mdl}, sigma_z={args.sigma_z})...")
    model, history, eval_data = train(n_epochs=args.n_epochs,
                                      lambda_mdl=args.lambda_mdl,
                                      sigma_z=args.sigma_z,
                                      seed=args.seed,
                                      verbose=False)
    info = visualize_implicit_space(model, eval_data["images"],
                                    eval_data["orient"])
    print(f"  final lin_sep={info['linear_separability']:.2f}  "
          f"axis_angle={info['axis_angle_deg']:.0f}deg  "
          f"sep={info['cluster_separation']:.2f}")

    plot_example_images(os.path.join(args.outdir, "example_images.png"),
                        seed=args.seed)
    plot_implicit_space(model, eval_data,
                        os.path.join(args.outdir, "implicit_space.png"))
    plot_mdl_trajectory(history,
                        os.path.join(args.outdir, "mdl_trajectory.png"))
    plot_decoder_grid(model, os.path.join(args.outdir, "decoder_grid.png"))
    plot_encoder_weights(model, os.path.join(args.outdir, "weights.png"))


if __name__ == "__main__":
    main()
