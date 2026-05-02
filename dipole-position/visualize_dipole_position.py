"""
Static visualizations for the dipole-position population coder.

Outputs (in `viz/`):
  example_dipoles.png           — 16 sample dipole images, labelled by (x, y)
  implicit_space_scatter.png    — 2D scatter of bottleneck p coloured by true (x, y)
  mdl_trajectory.png            — MDL bits / example over training
  receptive_fields.png          — top decoder rows reshaped as 8x8 receptive fields
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from dipole_position import (
    PopulationCoder,
    train,
    all_dipole_positions,
    render_dipole_flat,
    implicit_alignment_r2,
)


def plot_example_dipoles(out_path: str, h: int = 8, w: int = 8) -> None:
    universe = all_dipole_positions(h, w)
    rng = np.random.default_rng(7)
    sample_idx = rng.choice(len(universe), size=16, replace=False)
    sample = universe[sample_idx]
    images = render_dipole_flat(sample, h, w).reshape(-1, h, w)

    fig, axes = plt.subplots(4, 4, figsize=(7, 7), dpi=120)
    for k, ax in enumerate(axes.flat):
        ax.imshow(images[k], cmap="RdBu_r", vmin=-1.0, vmax=1.0)
        ax.set_title(f"(x={sample[k, 0]}, y={sample[k, 1]})", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("16 dipole training images (8x8, +1 / -1 horizontal pairs)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_implicit_scatter(model: PopulationCoder, out_path: str) -> None:
    universe = all_dipole_positions(model.image_h, model.image_w)
    universe_imgs = render_dipole_flat(universe, model.image_h, model.image_w)
    p = model.encode_position(universe_imgs)
    r2, fit = implicit_alignment_r2(p, universe.astype(np.float32))

    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=140)
    # Colour by x for left panel, by y for right panel
    sc1 = axes[0].scatter(p[:, 0], p[:, 1], c=universe[:, 0],
                            cmap="viridis", s=70, edgecolors="black",
                            linewidths=0.4)
    axes[0].set_xlabel("$p_0$ (implicit dim 0)")
    axes[0].set_ylabel("$p_1$ (implicit dim 1)")
    axes[0].set_title(f"Implicit space coloured by true x  "
                        f"(R$^2$ for linear $p\\to (x,y)$ = {r2:.3f})",
                        fontsize=10)
    axes[0].grid(alpha=0.3)
    cbar1 = fig.colorbar(sc1, ax=axes[0])
    cbar1.set_label("true x")

    sc2 = axes[1].scatter(p[:, 0], p[:, 1], c=universe[:, 1],
                            cmap="plasma", s=70, edgecolors="black",
                            linewidths=0.4)
    axes[1].set_xlabel("$p_0$ (implicit dim 0)")
    axes[1].set_ylabel("$p_1$ (implicit dim 1)")
    axes[1].set_title("Implicit space coloured by true y", fontsize=10)
    axes[1].grid(alpha=0.3)
    cbar2 = fig.colorbar(sc2, ax=axes[1])
    cbar2.set_label("true y")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mdl_trajectory(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), dpi=120)

    epochs = history["epoch"]
    axes[0, 0].plot(epochs, history["mdl_bits"], color="#1f77b4")
    axes[0, 0].set_xlabel("epoch")
    axes[0, 0].set_ylabel("MDL (bits / example)")
    axes[0, 0].set_yscale("log")
    axes[0, 0].grid(alpha=0.3, which="both")
    axes[0, 0].set_title("Total description length")

    axes[0, 1].plot(epochs, history["recon_bits_per_pixel"], color="#d62728",
                     label="recon (bits/px)")
    axes[0, 1].plot(epochs, history["code_bits_per_unit"], color="#2ca02c",
                     label="code (bits/unit)")
    axes[0, 1].set_xlabel("epoch")
    axes[0, 1].set_ylabel("DL component")
    axes[0, 1].set_yscale("log")
    axes[0, 1].grid(alpha=0.3, which="both")
    axes[0, 1].legend(loc="upper right")
    axes[0, 1].set_title("Recon vs code DL")

    axes[1, 0].plot(epochs, history["implicit_r2"], color="#9467bd")
    axes[1, 0].set_xlabel("epoch")
    axes[1, 0].set_ylabel(r"$R^2$ for $p \to (x, y)$")
    axes[1, 0].set_ylim(-0.05, 1.05)
    axes[1, 0].grid(alpha=0.3)
    axes[1, 0].set_title("2D-implicit-space alignment with true (x, y)")

    axes[1, 1].plot(epochs, history["p_std_x"], color="#1f77b4",
                     label="std($p_0$)")
    axes[1, 1].plot(epochs, history["p_std_y"], color="#ff7f0e",
                     label="std($p_1$)")
    axes[1, 1].set_xlabel("epoch")
    axes[1, 1].set_ylabel("standard deviation")
    axes[1, 1].grid(alpha=0.3)
    axes[1, 1].legend(loc="lower right")
    axes[1, 1].set_title("Spread of implicit-space coordinates")

    fig.suptitle("Dipole-position population coder — training", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_receptive_fields(model: PopulationCoder, out_path: str,
                           grid_side: int = 6) -> None:
    """Show decoded image at each implicit-space grid point.

    For p on a grid in [0, 1]^2, render the decoder output `decode(bump(p))`
    as an image. The result shows what each implicit-space location "means"
    in terms of the dipole picture. If the 2D implicit space is faithful,
    these images form a smooth spatial layout of dipoles.
    """
    grid_xs = np.linspace(0, 1, grid_side)
    grid_ys = np.linspace(0, 1, grid_side)
    fig, axes = plt.subplots(grid_side, grid_side,
                               figsize=(8, 8), dpi=120)
    h, w = model.image_h, model.image_w
    # vmax across all decoded panels for a shared colour scale
    decoded = np.zeros((grid_side, grid_side, h, w), dtype=np.float32)
    for r, gy in enumerate(grid_ys):
        for c, gx in enumerate(grid_xs):
            p = np.array([[gx, gy]], dtype=np.float32)
            bump = model.expected_bump(p)              # (1, N)
            x_hat = model.decode(bump)[0].reshape(h, w)
            decoded[r, c] = x_hat
    vmax = max(abs(decoded).max(), 1e-6)
    for r in range(grid_side):
        for c in range(grid_side):
            ax = axes[grid_side - 1 - r, c]   # flip y so origin lower-left
            ax.imshow(decoded[r, c], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_title(f"p=({grid_xs[c]:.2f}, {grid_ys[r]:.2f})", fontsize=8)
            ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Decoded image at each implicit-space position p  "
                  "(decoder(bump(p)))", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=4000)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Plotting example dipoles...")
    plot_example_dipoles(os.path.join(args.outdir, "example_dipoles.png"))

    print(f"Training {args.n_epochs} epochs (seed={args.seed})...")
    model, history = train(n_epochs=args.n_epochs, seed=args.seed,
                            verbose=False)
    print(f"  final R^2(p <-> xy): {history['implicit_r2'][-1]:.3f}  "
          f"MDL = {history['mdl_bits'][-1]:.2f} bits/example")

    print("Plotting implicit-space scatter...")
    plot_implicit_scatter(model,
                            os.path.join(args.outdir, "implicit_space_scatter.png"))
    print("Plotting MDL trajectory...")
    plot_mdl_trajectory(history,
                         os.path.join(args.outdir, "mdl_trajectory.png"))
    print("Plotting decoder receptive fields...")
    plot_receptive_fields(model,
                           os.path.join(args.outdir, "receptive_fields.png"))


if __name__ == "__main__":
    main()
