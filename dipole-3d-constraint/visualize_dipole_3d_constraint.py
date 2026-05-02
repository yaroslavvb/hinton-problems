"""
Static visualisations for the dipole 3D-constraint population code.

Outputs (in `viz/`):
  example_images.png     - 8 example dipoles with their (x, y, theta) labels
  training_curves.png    - loss / MSE / DL / R^2 curves
  implicit_space_2d.png  - three 2D projections of m_hat coloured by x, y, theta
  implicit_space_3d.png  - 3D scatter of m_hat coloured by theta
  reconstructions.png    - 8 (input, reconstruction) pairs
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np

from dipole_3d_constraint import (
    build_population_coder,
    forward,
    generate_dipole_images,
    implicit_space_recovery,
    train,
)


def plot_example_images(out_path: str, seed: int = 0) -> None:
    X, params = generate_dipole_images(8, seed=seed + 7)
    fig, axes = plt.subplots(1, 8, figsize=(13, 2.0), dpi=130)
    vmax = float(np.max(np.abs(X)))
    for k, ax in enumerate(axes):
        ax.imshow(X[k].reshape(8, 8), cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        x, y, t = params[k]
        ax.set_title(f"({x:.1f},{y:.1f})\n$\\theta$={np.degrees(t):.0f}°",
                     fontsize=8)
        ax.set_xticks([])
        ax.set_yticks([])
    fig.suptitle("8x8 dipole images at random (x, y, theta)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=120)
    ep = history["epoch"]

    ax = axes[0, 0]
    ax.plot(ep, history["loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training loss (0.5 MSE)")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(ep, history["recon_mse"], color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("reconstruction MSE")
    ax.set_title("Reconstruction MSE")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(ep, history["dl_bits"], color="#ff7f0e")
    ax.set_xlabel("epoch")
    ax.set_ylabel("description length (bits)")
    ax.set_title("MDL proxy (lower = more compressed)")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(ep, history["r2_mean"], color="#9467bd")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"linear $R^2$ to (x, y, cos$2\theta$, sin$2\theta$)")
    ax.set_title("Implicit-space recovery (linear fit)")
    ax.grid(alpha=0.3)
    ax.set_ylim(-0.05, 1.05)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_implicit_space_2d(model, X, params, out_path: str) -> None:
    fwd = forward(model, X)
    M = fwd["m_hat"]
    x, y, t = params[:, 0], params[:, 1], params[:, 2]

    fig, axes = plt.subplots(3, 3, figsize=(11, 10), dpi=120)
    pairs = [(0, 1), (0, 2), (1, 2)]
    pair_names = [("dim 1", "dim 2"), ("dim 1", "dim 3"), ("dim 2", "dim 3")]
    colourings = [("x", x, "viridis"),
                  ("y", y, "viridis"),
                  (r"$\theta$", t, "twilight")]

    for row, (cname, cvals, cmap) in enumerate(colourings):
        for col, ((i, j), (xn, yn)) in enumerate(zip(pairs, pair_names)):
            ax = axes[row, col]
            sc = ax.scatter(M[:, i], M[:, j], c=cvals, cmap=cmap, s=8,
                            alpha=0.75, edgecolors="none")
            ax.set_xlabel(f"m_hat[{i}]")
            ax.set_ylabel(f"m_hat[{j}]")
            if row == 0:
                ax.set_title(f"{xn} vs {yn}")
            if col == 2:
                cb = fig.colorbar(sc, ax=ax)
                cb.set_label(cname)
    fig.suptitle("Implicit space m_hat coloured by true (x, y, theta)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_implicit_space_3d(model, X, params, out_path: str) -> None:
    fwd = forward(model, X)
    M = fwd["m_hat"]
    t = params[:, 2]

    fig = plt.figure(figsize=(7, 6), dpi=120)
    ax = fig.add_subplot(111, projection="3d")
    sc = ax.scatter(M[:, 0], M[:, 1], M[:, 2], c=t, cmap="twilight",
                    s=10, alpha=0.85, edgecolors="none")
    ax.set_xlabel("m_hat[0]")
    ax.set_ylabel("m_hat[1]")
    ax.set_zlabel("m_hat[2]")
    cb = fig.colorbar(sc, ax=ax, shrink=0.7)
    cb.set_label(r"true $\theta$ (rad)")
    ax.set_title("3D implicit space coloured by orientation")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_reconstructions(model, out_path: str, seed: int = 0) -> None:
    X, params = generate_dipole_images(8, seed=seed + 17)
    fwd = forward(model, X)
    Xh = fwd["x_hat"]

    fig, axes = plt.subplots(2, 8, figsize=(13, 3.5), dpi=130)
    vmax = float(max(np.abs(X).max(), np.abs(Xh).max()))
    for k in range(8):
        axes[0, k].imshow(X[k].reshape(8, 8), cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        axes[0, k].set_xticks([]); axes[0, k].set_yticks([])
        axes[1, k].imshow(Xh[k].reshape(8, 8), cmap="RdBu_r",
                          vmin=-vmax, vmax=vmax)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
        x, y, t = params[k]
        axes[0, k].set_title(f"({x:.1f},{y:.1f})\n$\\theta$={np.degrees(t):.0f}°",
                             fontsize=8)
    axes[0, 0].set_ylabel("input", fontsize=10)
    axes[1, 0].set_ylabel("recon", fontsize=10)
    fig.suptitle("Reconstruction through the 3D implicit-space bottleneck",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=200)
    parser.add_argument("--n-train", type=int, default=2000)
    parser.add_argument("--n-hidden", type=int, default=225)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--outdir", type=str, default="viz")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    np.random.seed(args.seed)

    print("Generating training data...")
    X, params = generate_dipole_images(args.n_train, seed=args.seed)
    Xt, params_t = generate_dipole_images(800, seed=args.seed + 1000)

    print("Training population coder...")
    model = build_population_coder(n_hidden=args.n_hidden,
                                   sigma=args.sigma,
                                   seed=args.seed)
    out = train(model, X,
                n_epochs=args.n_epochs, lr=args.lr,
                params=params, seed=args.seed,
                eval_every=max(1, args.n_epochs // 50))
    history = out["history"]

    print("Plotting...")
    plot_example_images(os.path.join(args.outdir, "example_images.png"),
                        seed=args.seed)
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_implicit_space_2d(model, Xt, params_t,
                           os.path.join(args.outdir, "implicit_space_2d.png"))
    plot_implicit_space_3d(model, Xt, params_t,
                           os.path.join(args.outdir, "implicit_space_3d.png"))
    plot_reconstructions(model, os.path.join(args.outdir, "reconstructions.png"),
                         seed=args.seed)

    rec3 = implicit_space_recovery(model, Xt, params_t, degree=3)
    print(f"Cubic R^2 mean: {rec3['r2_mean']:.3f}")
    print(f"  x={rec3['r2_x']:.3f}  y={rec3['r2_y']:.3f}  "
          f"cos2theta={rec3['r2_cos2theta']:.3f}  sin2theta={rec3['r2_sin2theta']:.3f}")
    print(f"Wrote PNGs to {args.outdir}/")


if __name__ == "__main__":
    main()
