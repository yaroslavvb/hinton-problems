"""
Static visualizations for spline-image factorial VQ (Hinton & Zemel 1994).

Outputs (in `viz/`):
  example_splines.png         — 16 sample spline images with control-point overlays
  dl_comparison.png           — bar chart of total / recon / code DL across the four models
  dl_trajectory.png           — bits-per-example over training for the three trainable models
  factor_codebooks.png        — for the factorial VQ, the 6 codewords of each of the 4 dims
  per_factor_receptive.png    — what each of the 4 factor-dims contributes (mean codeword)
  factorial_reconstructions.png — original vs reconstruction for 8 held-out splines
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from spline_images_factorial_vq import (
    IMAGE_H, IMAGE_W,
    generate_spline_images,
    build_baseline_vq, build_separate_vq, build_factorial_vq,
    PCAModel, train_vq, run_comparison,
)


# ----------------------------------------------------------------------
# Example spline images
# ----------------------------------------------------------------------

def plot_example_splines(out_path: str, n_show: int = 16, seed: int = 7) -> None:
    images, controls = generate_spline_images(n_samples=n_show, seed=seed)
    fig, axes = plt.subplots(4, 4, figsize=(8, 6.5), dpi=120)
    for k, ax in enumerate(axes.flat):
        img = images[k].reshape(IMAGE_H, IMAGE_W)
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        # overlay the 5 control points
        xs = np.linspace(0, IMAGE_W - 1, controls.shape[1])
        ax.plot(xs, controls[k], "ro", markersize=3, alpha=0.8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("16 spline images (8x12, intrinsic dim 5)\n"
                  "red dots = the 5 y-control points", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Description-length bar chart
# ----------------------------------------------------------------------

def plot_dl_comparison(results: dict, n_dims: int, n_units_per_dim: int,
                        out_path: str) -> None:
    n_total = n_dims * n_units_per_dim
    keys = ["baseline_vq", "separate_vq", "factorial_vq", "pca"]
    labels = [f"Standard\n{n_total}-VQ",
                f"Four\nseparate\n{n_dims}x{n_units_per_dim}",
                f"Factorial\n{n_dims}x{n_units_per_dim}",
                "PCA\n(5 comps)"]
    recon = np.array([float(results[k]["dl"]["recon_nats"].mean() / np.log(2.0))
                       for k in keys])
    code = np.array([float(results[k]["dl"]["code_nats"].mean() / np.log(2.0))
                       for k in keys])
    total = recon + code

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=120)
    xs = np.arange(len(keys))
    bw = 0.7
    bars_r = ax.bar(xs, recon, bw, label="reconstruction (bits)",
                     color="#d62728", alpha=0.85)
    bars_c = ax.bar(xs, code, bw, bottom=recon, label="code KL (bits)",
                     color="#2ca02c", alpha=0.85)
    for x, r, c in zip(xs, recon, code):
        ax.text(x, r + c + max(total) * 0.02, f"{r + c:.1f}",
                 ha="center", fontsize=10, fontweight="bold")
        if r > max(total) * 0.05:
            ax.text(x, r * 0.5, f"{r:.1f}", ha="center", color="white",
                     fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("description length (bits / example)")
    ax.set_title("Bits-back description length on spline images\n"
                  "(lower is better; factorial VQ wins by a wide margin)")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# DL trajectory (training curves)
# ----------------------------------------------------------------------

def plot_dl_trajectory(results: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=120)
    color = {"baseline_vq": "#1f77b4",
              "separate_vq": "#ff7f0e",
              "factorial_vq": "#2ca02c"}
    label = {"baseline_vq": "Standard 24-VQ",
              "separate_vq": "Four separate VQs",
              "factorial_vq": "Factorial 4x6 VQ"}
    for k in ["baseline_vq", "separate_vq", "factorial_vq"]:
        h = results[k]["history"]
        axes[0].plot(h["epoch"], h["total_bits"], color=color[k],
                      label=label[k])
        axes[1].plot(h["epoch"], h["recon_bits"], color=color[k],
                      linestyle="-", label=label[k] + " recon")
        axes[1].plot(h["epoch"], h["code_bits"], color=color[k],
                      linestyle=":", alpha=0.7,
                      label=label[k] + " code")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("total DL (bits / example)")
    axes[0].set_yscale("log"); axes[0].grid(alpha=0.3, which="both")
    axes[0].legend(); axes[0].set_title("Total description length over training")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("bits / example")
    axes[1].set_yscale("log"); axes[1].grid(alpha=0.3, which="both")
    axes[1].legend(fontsize=8, loc="upper right")
    axes[1].set_title("Recon (solid) vs code-KL (dotted)")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-dim codebooks of the factorial VQ
# ----------------------------------------------------------------------

def plot_factor_codebooks(model, out_path: str) -> None:
    n_dims = model.n_dims
    K = model.n_codes_per_dim
    fig, axes = plt.subplots(n_dims, K, figsize=(K * 1.3, n_dims * 1.3),
                                dpi=120)
    for d in range(n_dims):
        Cd = model.factors[d].C       # (K, D)
        vmax = max(abs(Cd).max(), 1e-6)
        for k in range(K):
            ax = axes[d, k] if n_dims > 1 else axes[k]
            img = Cd[k].reshape(IMAGE_H, IMAGE_W)
            ax.imshow(img, cmap="RdBu_r", vmin=-vmax, vmax=vmax)
            ax.set_xticks([]); ax.set_yticks([])
            if d == 0:
                ax.set_title(f"k={k}", fontsize=8)
            if k == 0:
                ax.set_ylabel(f"dim {d}", fontsize=9)
    fig.suptitle("Factorial-VQ codebooks: 4 factor-dims x 6 codewords each\n"
                  "Each row is one factor's contribution; the 4 are summed.",
                  fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_factor_receptive(model, images: np.ndarray,
                                out_path: str) -> None:
    """Mean reconstruction per factor: show what each factor "explains" by
    averaging its mean-field contribution over the training set, plus the
    standard deviation per factor (how much each factor varies)."""
    n_dims = model.n_dims
    out = model._forward(images)
    fig, axes = plt.subplots(2, n_dims, figsize=(n_dims * 2.2, 4.5), dpi=120)
    for d in range(n_dims):
        contrib = out["contributions"][d]    # (B, D)
        mu = contrib.mean(axis=0).reshape(IMAGE_H, IMAGE_W)
        sd = contrib.std(axis=0).reshape(IMAGE_H, IMAGE_W)
        vmax_mu = max(abs(mu).max(), 1e-6)
        vmax_sd = max(sd.max(), 1e-6)
        axes[0, d].imshow(mu, cmap="RdBu_r", vmin=-vmax_mu, vmax=vmax_mu)
        axes[0, d].set_title(f"dim {d} mean", fontsize=9)
        axes[0, d].set_xticks([]); axes[0, d].set_yticks([])
        axes[1, d].imshow(sd, cmap="viridis", vmin=0, vmax=vmax_sd)
        axes[1, d].set_title(f"dim {d} std (over data)", fontsize=9)
        axes[1, d].set_xticks([]); axes[1, d].set_yticks([])
    fig.suptitle("Per-factor mean and std of additive contribution\n"
                  "(mean = average codeword used; std = how much the factor varies)",
                  fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(results: dict, images: np.ndarray,
                           out_path: str, n_show: int = 8) -> None:
    """Compare originals vs each model's reconstruction on a sample."""
    rng = np.random.default_rng(11)
    idx = rng.choice(len(images), size=n_show, replace=False)
    sample = images[idx]

    rows = [("original", sample),
             ("Standard 24-VQ",
                results["baseline_vq"]["model"].decode(
                    results["baseline_vq"]["model"].encode(sample))),
             ("Four separate", results["separate_vq"]["model"]._forward(sample)["x_hat"]),
             ("Factorial",
                results["factorial_vq"]["model"]._forward(sample)["x_hat"]),
             ("PCA",
                results["pca"]["model"].decode(
                    results["pca"]["model"].encode(sample)))]

    fig, axes = plt.subplots(len(rows), n_show,
                                figsize=(n_show * 1.2, len(rows) * 1.2),
                                dpi=120)
    for i, (label, recs) in enumerate(rows):
        for j in range(n_show):
            ax = axes[i, j]
            ax.imshow(recs[j].reshape(IMAGE_H, IMAGE_W),
                       cmap="gray", vmin=0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if j == 0:
                ax.set_ylabel(label, fontsize=10)
    fig.suptitle("Reconstructions on 8 held-out samples", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-samples", type=int, default=200)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--n-dims", type=int, default=4)
    p.add_argument("--n-units-per-dim", type=int, default=6)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Plotting example spline images...")
    plot_example_splines(os.path.join(args.outdir, "example_splines.png"))

    print(f"Training all four models (seed={args.seed}, n_epochs={args.n_epochs})...")
    results = run_comparison(n_samples=args.n_samples, n_epochs=args.n_epochs,
                              n_dims=args.n_dims,
                              n_units_per_dim=args.n_units_per_dim,
                              seed=args.seed, verbose=True)

    print("\nPlotting DL comparison bar chart...")
    plot_dl_comparison(results, args.n_dims, args.n_units_per_dim,
                        os.path.join(args.outdir, "dl_comparison.png"))
    print("Plotting DL trajectory...")
    plot_dl_trajectory(results,
                        os.path.join(args.outdir, "dl_trajectory.png"))
    print("Plotting factor codebooks...")
    plot_factor_codebooks(results["factorial_vq"]["model"],
                            os.path.join(args.outdir, "factor_codebooks.png"))
    print("Plotting per-factor receptive contributions...")
    plot_per_factor_receptive(results["factorial_vq"]["model"],
                                results["images"],
                                os.path.join(args.outdir,
                                              "per_factor_receptive.png"))
    print("Plotting reconstructions...")
    plot_reconstructions(results, results["images"],
                          os.path.join(args.outdir,
                                        "factorial_reconstructions.png"))


if __name__ == "__main__":
    main()
