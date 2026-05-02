"""
Static visualizations for the riser-spectrogram experiment.

Outputs (in `viz/`):
  example_inputs.png    -- 6 example noisy spectrograms (3 rising, 3 falling)
                           with the underlying clean track overlaid.
  training_curves.png   -- train loss, train/test accuracy, and the
                           Bayes-optimal ceiling on the same axes.
  hidden_filters.png    -- the 24 hidden-unit input weights, one
                           filter per panel, displayed as 6 x 9
                           (frequency x time) grids.
  bayes_vs_net.png      -- accuracy gap to Bayes-optimal across a
                           noise-level sweep.
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from riser_spectrogram import (
    N_FREQ, N_TIME, N_HIDDEN, DEFAULT_NOISE_STD,
    generate_dataset, sample_rising_tracks, sample_falling_tracks,
    tracks_to_specs, train, accuracy, bayes_optimal_accuracy,
    bayes_log_posterior,
)


def plot_example_inputs(out_path: str, noise_std: float = DEFAULT_NOISE_STD,
                        seed: int = 0) -> None:
    rng = np.random.default_rng(seed)
    rise = sample_rising_tracks(rng, 3)
    fall = sample_falling_tracks(rng, 3)

    fig, axes = plt.subplots(2, 3, figsize=(8.5, 5.0), dpi=130)
    titles = ["rising", "rising", "rising", "falling", "falling", "falling"]
    tracks = np.concatenate([rise, fall], axis=0)
    cleans = tracks_to_specs(tracks)
    noise = noise_std * rng.standard_normal(cleans.shape)
    noisy = cleans + noise

    vmax = float(noisy.max()); vmin = float(noisy.min())
    for k in range(6):
        ax = axes[k // 3, k % 3]
        ax.imshow(noisy[k], aspect="auto", origin="lower",
                  vmin=vmin, vmax=vmax, cmap="magma")
        # overlay the clean track
        ax.plot(np.arange(N_TIME), tracks[k], color="cyan",
                linewidth=1.0, alpha=0.7)
        ax.scatter(np.arange(N_TIME), tracks[k], color="cyan",
                   s=14, alpha=0.8, edgecolors="black", linewidths=0.4)
        ax.set_title(f"{titles[k]}  (sigma={noise_std})", fontsize=9)
        ax.set_xlabel("time")
        ax.set_ylabel("frequency")
        ax.set_xticks(range(N_TIME))
        ax.set_yticks(range(N_FREQ))
    fig.suptitle("Example synthetic spectrograms (clean track overlaid)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_training_curves(history: dict, out_path: str,
                         bayes_acc: float | None = None) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6), dpi=130)

    epochs = history["epoch"]

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, np.array(history["train_acc"]) * 100,
            color="#1f77b4", label="train", linewidth=1.4)
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#d62728", label="test", linewidth=1.4)
    if bayes_acc is not None:
        ax.axhline(bayes_acc * 100, color="black", linestyle="--",
                   linewidth=1.0, alpha=0.7,
                   label=f"Bayes-optimal ({bayes_acc*100:.2f}%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(90, 100.5)
    ax.set_title("Train / test accuracy")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_filters(model, out_path: str) -> None:
    """The 24 hidden-unit W1 rows, each shown as a 6 x 9 image."""
    n_h = model.W1.shape[0]
    n_cols = 6
    n_rows = (n_h + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(11, 1.8 * n_rows),
                             dpi=130)
    vmax = float(np.abs(model.W1).max())
    for i in range(n_h):
        ax = axes[i // n_cols, i % n_cols]
        filt = model.W1[i].reshape(N_FREQ, N_TIME)
        ax.imshow(filt, aspect="auto", origin="lower",
                  vmin=-vmax, vmax=vmax, cmap="RdBu_r")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(f"h{i}  |w|={np.linalg.norm(filt):.2f}", fontsize=7)
    for i in range(n_h, n_rows * n_cols):
        axes[i // n_cols, i % n_cols].axis("off")
    fig.suptitle("Hidden-unit input weights (6 freq x 9 time)  red = +, blue = -",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_bayes_vs_net(out_path: str, base_kw: dict,
                      noise_levels=(0.4, 0.5, 0.6, 0.7, 0.8),
                      n_train: int = 2000, n_test: int = 4000,
                      n_sweeps: int = 200, seed: int = 0,
                      bayes_samples: int = 30_000) -> None:
    """Sweep noise level, plot network test acc and Bayes ceiling."""
    bayes_accs = []
    net_accs = []
    for sigma in noise_levels:
        bayes = bayes_optimal_accuracy(sigma, n_samples=bayes_samples,
                                       seed=seed + 10_000)
        kw = dict(base_kw)
        kw["noise_std"] = sigma
        kw["n_sweeps"] = n_sweeps
        kw["n_train"] = n_train
        kw["n_test"] = n_test
        kw["verbose"] = False
        kw["seed"] = seed
        kw["snapshot_callback"] = None
        model, _, _, _, X_test, y_test = train(**kw)
        net = accuracy(model, X_test, y_test)
        bayes_accs.append(bayes)
        net_accs.append(net)
        print(f"  sigma={sigma:.2f}  net={net*100:5.2f}%  "
              f"bayes={bayes*100:5.2f}%  gap={(bayes-net)*100:+.2f}pp")

    fig, ax = plt.subplots(figsize=(7.0, 4.2), dpi=130)
    ax.plot(noise_levels, np.array(bayes_accs) * 100, "o-",
            color="black", label="Bayes-optimal", linewidth=1.6)
    ax.plot(noise_levels, np.array(net_accs) * 100, "s-",
            color="#d62728", label="54-24-2 MLP (test)", linewidth=1.6)
    ax.set_xlabel("noise std (sigma)")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(50, 102)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left")
    ax.set_title(f"Bayes-vs-net accuracy across noise levels  "
                 f"(P&H 1987 reports 97.8% / 98.8% at one sigma)",
                 fontsize=10)
    for s, b, n in zip(noise_levels, bayes_accs, net_accs):
        ax.annotate(f"{(b - n)*100:+.1f}pp", xy=(s, n * 100 - 0.5),
                    fontsize=8, ha="center", va="top", color="#666")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--n-hidden", type=int, default=N_HIDDEN)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--skip-sweep", action="store_true",
                   help="skip the noise-level sweep (faster)")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Plotting example spectrograms...")
    plot_example_inputs(os.path.join(args.outdir, "example_inputs.png"),
                        noise_std=args.noise_std, seed=args.seed)

    print(f"Training {args.epochs} epochs (seed={args.seed}, "
          f"sigma={args.noise_std})...")
    model, hist, _, _, X_test, y_test = train(
        n_train=args.n_train,
        n_test=args.n_test,
        n_sweeps=args.epochs,
        lr=args.lr,
        momentum=args.momentum,
        init_scale=args.init_scale,
        n_hidden=args.n_hidden,
        noise_std=args.noise_std,
        seed=args.seed,
        verbose=False,
    )
    net_acc = accuracy(model, X_test, y_test)
    print(f"  final test acc: {net_acc*100:.2f}%")

    print("Computing Bayes-optimal accuracy...")
    bayes_acc = bayes_optimal_accuracy(args.noise_std, n_samples=50_000,
                                       seed=args.seed + 7919)
    print(f"  bayes={bayes_acc*100:.2f}%  gap={(bayes_acc-net_acc)*100:+.2f}pp")

    plot_training_curves(hist,
                          os.path.join(args.outdir, "training_curves.png"),
                          bayes_acc=bayes_acc)
    plot_hidden_filters(model,
                         os.path.join(args.outdir, "hidden_filters.png"))

    if not args.skip_sweep:
        print("Sweeping noise levels for Bayes-vs-net plot...")
        base_kw = dict(lr=args.lr, momentum=args.momentum,
                       init_scale=args.init_scale,
                       n_hidden=args.n_hidden,
                       batch_size=100)
        plot_bayes_vs_net(os.path.join(args.outdir, "bayes_vs_net.png"),
                          base_kw=base_kw, seed=args.seed,
                          n_sweeps=args.epochs,
                          n_train=args.n_train, n_test=args.n_test)


if __name__ == "__main__":
    main()
