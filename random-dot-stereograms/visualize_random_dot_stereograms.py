"""
Static visualizations for the trained Imax stereo network.

Outputs (in `viz/`):
  example_stereograms.png   four left/right strip pairs at different disparities
  outputs_vs_disparity.png  module outputs (y_a, y_b) vs ground-truth disparity
  imax_trajectory.png       Imax + module-agreement + variance curves
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from random_dot_stereograms import (
    build_two_module_net, train, evaluate, generate_stereo_pair,
    generate_batch,
)


def plot_example_stereograms(rng, strip_width: int, max_disparity: float,
                              out_path: str, n_examples: int = 4):
    """Show n_examples random-dot stereo pairs at varying disparities."""
    fig, axes = plt.subplots(n_examples, 2, figsize=(8, 0.8 * n_examples + 0.5),
                             dpi=140)
    if n_examples == 1:
        axes = axes[None, :]

    # Pick disparities spanning the full range, plus 0
    target_d = np.linspace(-max_disparity, max_disparity, n_examples)
    for i in range(n_examples):
        left, right, d = generate_stereo_pair(
            rng, strip_width=strip_width,
            max_disparity=max_disparity,
            disparity=float(target_d[i]),
            continuous=True,
        )
        # Show as 1D heatmap
        for ax, view, label in [(axes[i, 0], left, "left eye"),
                                 (axes[i, 1], right, "right eye")]:
            ax.imshow(view[None, :], cmap="gray", vmin=-1.2, vmax=1.2,
                      aspect="auto", interpolation="nearest")
            ax.set_yticks([])
            if i == n_examples - 1:
                ax.set_xlabel("pixel")
            else:
                ax.set_xticklabels([])
            if i == 0:
                ax.set_title(label)
            ax.set_ylabel(f"d = {d:+.2f}", fontsize=9)
    fig.suptitle("Random-dot stereo pairs (sub-pixel disparity)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_outputs_vs_disparity(mod_a, mod_b, rng, strip_width: int,
                               max_disparity: float, out_path: str,
                               n_eval: int = 1024):
    x_a, x_b, d = generate_batch(rng, n_eval, strip_width=strip_width,
                                  max_disparity=max_disparity, continuous=True)
    y_a, _, _ = mod_a.forward(x_a)
    y_b, _, _ = mod_b.forward(x_b)

    corr_a = float(np.corrcoef(y_a, d)[0, 1])
    corr_b = float(np.corrcoef(y_b, d)[0, 1])
    corr_a_abs = float(np.corrcoef(y_a, np.abs(d))[0, 1])
    corr_b_abs = float(np.corrcoef(y_b, np.abs(d))[0, 1])

    fig, axes = plt.subplots(1, 3, figsize=(13, 4), dpi=140)

    ax = axes[0]
    ax.scatter(d, y_a, s=6, alpha=0.4, color="#1f77b4", label="module A")
    ax.scatter(d, y_b, s=6, alpha=0.4, color="#d62728", label="module B")
    ax.axhline(0, color="gray", linewidth=0.5)
    ax.axvline(0, color="gray", linewidth=0.5)
    ax.set_xlabel("ground-truth disparity d")
    ax.set_ylabel("module output y")
    ax.set_title(f"Outputs vs signed d\n"
                 f"corr(y_a,d)={corr_a:+.3f}  corr(y_b,d)={corr_b:+.3f}",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.scatter(np.abs(d), y_a, s=6, alpha=0.4, color="#1f77b4", label="module A")
    ax.scatter(np.abs(d), y_b, s=6, alpha=0.4, color="#d62728", label="module B")
    ax.set_xlabel("|d| (unsigned disparity magnitude)")
    ax.set_ylabel("module output y")
    ax.set_title(f"Outputs vs |d|\n"
                 f"corr(y_a,|d|)={corr_a_abs:+.3f}  "
                 f"corr(y_b,|d|)={corr_b_abs:+.3f}", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.scatter(y_a, y_b, s=6, alpha=0.4, color="#2ca02c")
    lim = max(abs(y_a).max(), abs(y_b).max()) * 1.05
    ax.plot([-lim, lim], [-lim, lim], "k--", linewidth=0.6, alpha=0.4)
    ax.set_xlabel(r"$y_a$  (module A output)")
    ax.set_ylabel(r"$y_b$  (module B output)")
    corr_ab = float(np.corrcoef(y_a, y_b)[0, 1])
    ax.set_title(f"Module-A vs Module-B agreement\n"
                 f"corr(y_a, y_b) = {corr_ab:+.3f}", fontsize=10)
    ax.grid(alpha=0.3)
    ax.set_aspect("equal")

    fig.suptitle("Imax-trained modules agree on a disparity-related readout",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_imax_trajectory(history: dict, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(10, 6), dpi=130)

    epochs = history["epoch"]

    ax = axes[0, 0]
    ax.plot(epochs, history["imax"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("Imax (nats)")
    ax.set_title("Mutual information between modules")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(epochs, history["corr_ab"], color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("corr(y_a, y_b)")
    ax.set_title("Module-output agreement")
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, np.abs(history["corr_a_d"]),
            color="#1f77b4", label=r"$|\mathrm{corr}(y_a, d)|$")
    ax.plot(epochs, np.abs(history["corr_b_d"]),
            color="#d62728", label=r"$|\mathrm{corr}(y_b, d)|$", linestyle="--")
    ax.plot(epochs, np.abs(history["corr_a_absd"]),
            color="#1f77b4", label=r"$|\mathrm{corr}(y_a, |d|)|$",
            alpha=0.5, linewidth=0.8)
    ax.plot(epochs, np.abs(history["corr_b_absd"]),
            color="#d62728", label=r"$|\mathrm{corr}(y_b, |d|)|$",
            alpha=0.5, linewidth=0.8, linestyle="--")
    ax.set_xlabel("epoch")
    ax.set_ylabel("|correlation|")
    ax.set_title("Module-vs-disparity correlation\n(sign-invariant)",
                 fontsize=10)
    ax.legend(fontsize=8, loc="lower right")
    ax.set_ylim(0, 1.0)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(epochs, history["var_a"], color="#1f77b4", label=r"$\mathrm{var}(y_a)$")
    ax.plot(epochs, history["var_b"], color="#d62728",
            label=r"$\mathrm{var}(y_b)$", linestyle="--")
    ax.plot(epochs, history["var_diff"], color="#9467bd",
            label=r"$\mathrm{var}(y_a - y_b)$")
    ax.set_xlabel("epoch")
    ax.set_ylabel("variance")
    ax.set_title("Output variances\n(Imax = 0.5*log((va+vb)/vd))",
                 fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.suptitle("Imax training trajectory", fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=800)
    p.add_argument("--strip-width", type=int, default=10)
    p.add_argument("--max-disparity", type=float, default=3.0)
    p.add_argument("--n-hidden", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Training {args.n_epochs} epochs (seed={args.seed})...")
    mod_a, mod_b = build_two_module_net(strip_width=args.strip_width,
                                         n_hidden=args.n_hidden,
                                         seed=args.seed,
                                         init_scale=0.5)
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
                    verbose=False)
    print(f"  final Imax: {history['imax'][-1]:.3f}")

    rng = np.random.default_rng(args.seed + 7)
    plot_example_stereograms(rng, args.strip_width, args.max_disparity,
                              os.path.join(args.outdir, "example_stereograms.png"))
    rng2 = np.random.default_rng(args.seed + 13)
    plot_outputs_vs_disparity(mod_a, mod_b, rng2, args.strip_width,
                               args.max_disparity,
                               os.path.join(args.outdir,
                                            "outputs_vs_disparity.png"))
    plot_imax_trajectory(history,
                          os.path.join(args.outdir, "imax_trajectory.png"))


if __name__ == "__main__":
    main()
