"""
Static visualizations for the trained sunspot-prediction networks.

Outputs (in `viz/`):
  series.png            - the raw Wolfer time-series with train/test split
  predictions.png       - predicted vs observed sunspot counts on the test
                           set, all three methods overlaid
  test_mse_bars.png     - bar chart of final test MSE for each method
                           (mean +/- std over multiple seeds)
  weight_histograms.png - distribution of W1+W2 weights for all three
                           methods, side-by-side. The MoG histogram shows
                           the cluster structure (multiple peaks); decay
                           is a single Gaussian at zero; vanilla is wide
                           and unstructured.
  mog_components.png    - the K MoG component density curves drawn over
                           the histogram of MoG weights, showing how the
                           prior carved up weight space
  training_curves.png   - train/test MSE over epochs for all three methods
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from sunspots import (load_wolfer, weigend_split, compare_methods, train_one,
                       MLP, MoG)


METHOD_COLORS = {"vanilla": "#1f77b4", "decay": "#2ca02c", "mog": "#d62728"}
METHOD_NAMES = {"vanilla": "vanilla (no prior)",
                 "decay": "weight decay (Gaussian prior)",
                 "mog": "soft weight-sharing (MoG prior)"}


def plot_series(years: np.ndarray, counts: np.ndarray, train_end: int,
                 test_end: int, out_path: str):
    fig, ax = plt.subplots(figsize=(10, 3.6), dpi=130)
    ax.plot(years, counts, color="#444", linewidth=1.0)
    ax.fill_between(years, 0, counts,
                     where=(years <= train_end),
                     color="#1f77b4", alpha=0.20, label="train (1700-1920)")
    ax.fill_between(years, 0, counts,
                     where=(years > train_end) & (years <= test_end),
                     color="#d62728", alpha=0.30, label="test (1921-1955)")
    ax.axvline(train_end + 0.5, color="black", linewidth=0.7, linestyle="--")
    ax.set_xlabel("year")
    ax.set_ylabel("Wolfer / SILSO yearly sunspot number")
    ax.set_title("Yearly sunspot count (Weigend benchmark)")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_predictions(results: dict, data: dict, out_path: str):
    """Predicted vs observed, on the test years."""
    fig, ax = plt.subplots(figsize=(10, 4.2), dpi=130)
    # observed
    obs_years = data["test_years"]
    obs = data["y_test"].ravel() * data["norm"]
    ax.plot(obs_years, obs, color="black", linewidth=1.6,
             marker="o", markersize=4, label="observed")
    for name in ["vanilla", "decay", "mog"]:
        m = results[name]["model"]
        pred_norm = m.predict(data["X_test"]).ravel()
        pred = pred_norm * data["norm"]
        test_mse = results[name]["final_test_mse"]
        ax.plot(obs_years, pred, color=METHOD_COLORS[name], linewidth=1.4,
                 marker=".", markersize=4, alpha=0.85,
                 label=f"{METHOD_NAMES[name]}  (test MSE = {test_mse:.5f})")
    ax.set_xlabel("year")
    ax.set_ylabel("yearly sunspot count")
    ax.set_title(f"Test-set predictions ({obs_years[0]}-{obs_years[-1]})")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_test_mse_bars(per_seed_results: dict, out_path: str):
    """Bar chart with mean +/- std over seeds."""
    methods = ["vanilla", "decay", "mog"]
    means = [np.mean(per_seed_results[m]) for m in methods]
    stds = [np.std(per_seed_results[m]) for m in methods]
    fig, ax = plt.subplots(figsize=(6, 4), dpi=130)
    xs = np.arange(3)
    bars = ax.bar(xs, means, yerr=stds, color=[METHOD_COLORS[m] for m in methods],
                   edgecolor="black", linewidth=0.6, capsize=6)
    for b, mu, sd in zip(bars, means, stds):
        ax.text(b.get_x() + b.get_width() / 2, mu + sd + 0.0002,
                 f"{mu:.5f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(xs)
    ax.set_xticklabels(["vanilla", "weight decay", "MoG soft sharing"])
    ax.set_ylabel("test MSE (normalised counts)")
    n_seeds = len(per_seed_results["vanilla"])
    ax.set_title(f"Final test MSE  (mean +/- std over {n_seeds} seeds)")
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_histograms(results: dict, out_path: str):
    """Side-by-side weight histograms for the three methods."""
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=130, sharey=True)
    methods = ["vanilla", "decay", "mog"]
    # shared x range
    all_W = np.concatenate([results[m]["model"].all_W() for m in methods])
    rng = max(np.abs(all_W).max(), 1e-3) * 1.05
    bins = np.linspace(-rng, rng, 41)

    for ax, name in zip(axes, methods):
        W = results[name]["model"].all_W()
        ax.hist(W, bins=bins, color=METHOD_COLORS[name],
                 edgecolor="black", linewidth=0.4, alpha=0.85)
        ax.axvline(0, color="black", linewidth=0.6)
        ax.set_title(f"{METHOD_NAMES[name]}\n"
                     f"|W|={np.linalg.norm(W):.2f}  range=[{W.min():.2f}, {W.max():.2f}]",
                     fontsize=10)
        ax.set_xlabel("weight value")
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("count")
    fig.suptitle("Weight distributions after training", fontsize=12, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_mog_components(results: dict, out_path: str):
    """Histogram of MoG weights + the K Gaussian component density curves."""
    mog: MoG = results["mog"]["mog"]
    model: MLP = results["mog"]["model"]
    W = model.all_W()

    fig, ax = plt.subplots(figsize=(9, 4.2), dpi=130)
    rng = max(np.abs(W).max(), 1e-3) * 1.05
    bins = np.linspace(-rng, rng, 60)
    counts, _, _ = ax.hist(W, bins=bins, color="#999999", edgecolor="black",
                              linewidth=0.4, alpha=0.65, label="MoG weights")

    # overlay component densities, scaled to max histogram height
    xs = np.linspace(-rng, rng, 400)
    pi = mog.pi()
    sig = mog.sigma()
    mu = mog.mu

    height = max(counts) if max(counts) > 0 else 1.0

    palette = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728",
                "#17becf", "#bcbd22", "#e377c2"]
    densities = []
    for k in range(mog.K):
        d = pi[k] * np.exp(-0.5 * ((xs - mu[k]) / sig[k]) ** 2) \
              / (np.sqrt(2 * np.pi) * sig[k])
        densities.append(d)
    max_d = max(d.max() for d in densities)
    scale = height / max(max_d, 1e-6)

    for k in range(mog.K):
        ax.plot(xs, densities[k] * scale, color=palette[k % len(palette)],
                 linewidth=1.5,
                 label=(f"comp {k}: pi={pi[k]:.2f}, "
                        f"mu={mu[k]:+.2f}, sigma={sig[k]:.2f}"
                        + ("  (pinned)" if k == 0 else "")))
    total = sum(densities) * scale
    ax.plot(xs, total, color="black", linewidth=1.2, linestyle="--",
             label="mixture total")

    ax.set_xlabel("weight value")
    ax.set_ylabel("count / scaled density")
    ax.set_title("MoG components fitted to the network's weight distribution")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(results: dict, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), dpi=130)
    for name in ["vanilla", "decay", "mog"]:
        h = results[name]["history"]
        axes[0].plot(h["epoch"], h["train_mse"], color=METHOD_COLORS[name],
                      linewidth=1.2, label=METHOD_NAMES[name])
        axes[1].plot(h["epoch"], h["test_mse"], color=METHOD_COLORS[name],
                      linewidth=1.2, label=METHOD_NAMES[name])
    for ax, title in zip(axes, ["train MSE", "test MSE"]):
        ax.set_xlabel("epoch")
        ax.set_ylabel(title)
        ax.set_title(title)
        ax.set_yscale("log")
        ax.grid(alpha=0.3, which="both")
        ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-seeds", type=int, default=5,
                    help="number of seeds for the bar-chart sweep")
    p.add_argument("--epochs", type=int, default=12000)
    p.add_argument("--n-hidden", type=int, default=16)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--lam-decay", type=float, default=0.01)
    p.add_argument("--lam-mog", type=float, default=0.0005)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading Wolfer data...")
    years, counts = load_wolfer()
    data = weigend_split(years, counts, n_lags=12)
    print(f"  {len(years)} years, train n={len(data['y_train'])}, "
          f"test n={len(data['y_test'])}")

    plot_series(years, counts, train_end=1920, test_end=1955,
                 out_path=os.path.join(args.outdir, "series.png"))

    print(f"\nTraining all three methods at seed={args.seed}...")
    results = compare_methods(data, n_epochs=args.epochs, seed=args.seed,
                                lam_decay=args.lam_decay,
                                lam_mog=args.lam_mog,
                                n_components=args.n_components, lr=args.lr,
                                n_hidden=args.n_hidden)
    for name in ["vanilla", "decay", "mog"]:
        r = results[name]
        print(f"  {name:10s}  train={r['final_train_mse']:.5f}  "
              f"test={r['final_test_mse']:.5f}  "
              f"best={r['best_test_mse']:.5f}")

    plot_predictions(results, data,
                       os.path.join(args.outdir, "predictions.png"))
    plot_weight_histograms(results,
                              os.path.join(args.outdir, "weight_histograms.png"))
    plot_mog_components(results,
                          os.path.join(args.outdir, "mog_components.png"))
    plot_training_curves(results,
                            os.path.join(args.outdir, "training_curves.png"))

    print(f"\nMulti-seed bar chart  ({args.n_seeds} seeds)...")
    per_seed = {"vanilla": [], "decay": [], "mog": []}
    for s in range(args.n_seeds):
        r = compare_methods(data, n_epochs=args.epochs, seed=s,
                              lam_decay=args.lam_decay,
                              lam_mog=args.lam_mog,
                              n_components=args.n_components, lr=args.lr,
                              n_hidden=args.n_hidden)
        for name in per_seed:
            per_seed[name].append(r[name]["final_test_mse"])
        print(f"  seed {s}: van={r['vanilla']['final_test_mse']:.5f}  "
              f"dec={r['decay']['final_test_mse']:.5f}  "
              f"mog={r['mog']['final_test_mse']:.5f}")
    plot_test_mse_bars(per_seed,
                          os.path.join(args.outdir, "test_mse_bars.png"))


if __name__ == "__main__":
    main()
