"""
Static visualizations for the trained bars Helmholtz machine.

Outputs (in `viz/`):
  training_curves.png   -- KL[p_data || p_model] and NLL trajectories
  generated_samples.png -- 64 samples drawn from the trained generative net
  recognition_codes.png -- recognition activations on each of the 30 unique
                           support images (sorted by orientation)
  weights_generative.png-- Hinton diagram of W_hv (8 hidden x 16 visible)
  weights_recognition.png- Hinton diagram of R_vh
  hidden_specialization.png- per-hidden-unit "receptive field" (the W_hv row
                            reshaped to 4x4 + the corresponding b_v offset)
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import bars
from _train_canonical import load_model, train_and_save


# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------

def _hinton_diagram(ax, M: np.ndarray, max_abs: float | None = None,
                    sz_scale: float = 0.42) -> None:
    if max_abs is None:
        max_abs = max(abs(M).max(), 1e-6)
    rows, cols = M.shape
    for i in range(rows):
        for j in range(cols):
            w = M[i, j]
            sz = sz_scale * 2.0 * (abs(w) / max_abs) ** 0.5
            color = "#cc0000" if w > 0 else "#003366"
            ax.add_patch(Rectangle((j - sz / 2, i - sz / 2), sz, sz,
                                   facecolor=color, edgecolor="black",
                                   linewidth=0.3))
    ax.set_xlim(-0.6, cols - 0.4)
    ax.set_ylim(-0.6, rows - 0.4)
    ax.invert_yaxis()
    ax.set_aspect("equal")


def _grid_image(ax, imgs: np.ndarray, n_rows: int, n_cols: int,
                pad: int = 1) -> None:
    """Plot imgs (N, 16) as a tiled n_rows x n_cols grid of 4x4 images."""
    H, W = bars.H, bars.W
    canvas = np.full((n_rows * (H + pad) + pad, n_cols * (W + pad) + pad),
                     0.5, dtype=np.float32)
    for k in range(min(len(imgs), n_rows * n_cols)):
        r, c = divmod(k, n_cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = imgs[k].reshape(H, W)
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0, interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str) -> None:
    samples = np.array(history["samples"])
    kl = np.array(history["kl_bits"])
    nll = np.array(history["neg_log_lik"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

    ax = axes[0]
    ax.plot(samples / 1e6, kl, color="#1f77b4")
    ax.set_xlabel("samples (millions)")
    ax.set_ylabel("KL[p_data || p_model]  (bits)")
    ax.set_title("Asymmetric KL trajectory")
    ax.grid(alpha=0.3)
    ax.set_yscale("log")

    ax = axes[1]
    ax.plot(samples / 1e6, nll, color="#9467bd", label="NLL of $p_\\mathrm{model}$")
    h_data = nll[-1] - kl[-1]
    ax.axhline(h_data, color="black", linestyle="--", linewidth=0.8,
               label=f"$H(p_\\mathrm{{data}})$ = {h_data:.3f} bits")
    ax.set_xlabel("samples (millions)")
    ax.set_ylabel("bits per image")
    ax.legend(fontsize=9)
    ax.set_title("Cross-entropy under the model")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generated_samples(model: bars.HelmholtzMachine, out_path: str,
                           n: int = 64, seed: int = 99) -> None:
    rng = np.random.default_rng(seed)
    saved_rng = model.rng
    model.rng = rng
    v, h, t = model.generate(n)
    model.rng = saved_rng
    fig, ax = plt.subplots(figsize=(8, 8), dpi=130)
    _grid_image(ax, v, n_rows=8, n_cols=8)
    ax.set_title(f"Samples from $p_\\mathrm{{model}}(v)$ "
                 f"(top=1: {int(t.sum())}/{n}; expected ~{int(2*n/3)})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recognition_codes(model: bars.HelmholtzMachine, out_path: str
                           ) -> None:
    """Show the recognition net's hidden activations on each of the 30
    support images of p_data, sorted by the recognition net's top-unit
    probability so vertical and horizontal images cluster.
    """
    p_h = bars.sigmoid(bars.DATA_IMAGES @ model.R_vh + model.c_h)   # (30, 8)
    p_t = bars.sigmoid(p_h @ model.R_ht + model.c_top).squeeze()    # (30,)

    order = np.argsort(p_t)
    fig, axes = plt.subplots(2, 1, figsize=(10, 7), dpi=130,
                             gridspec_kw=dict(height_ratios=[1.0, 1.6]))

    ax = axes[0]
    H, W = bars.H, bars.W
    pad = 1
    canvas = np.full((H + 2 * pad, len(order) * (W + pad) + pad), 0.5,
                     dtype=np.float32)
    for k, idx in enumerate(order):
        x0 = pad + k * (W + pad)
        canvas[pad:pad + H, x0:x0 + W] = bars.DATA_IMAGES[idx].reshape(H, W)
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0,
              interpolation="nearest", aspect="auto")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title("30 support images, sorted by recognised $q(t=1 | v)$")

    ax = axes[1]
    code_matrix = p_h[order].T  # (8, 30)
    im = ax.imshow(code_matrix, cmap="viridis", vmin=0.0, vmax=1.0,
                   aspect="auto", interpolation="nearest")
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h[{j}]" for j in range(model.n_hidden)])
    ax.set_xticks(range(0, len(order), 2))
    ax.set_xticklabels([f"{p_t[order[k]]:.2f}" for k in range(0, len(order), 2)],
                       rotation=45, fontsize=8)
    ax.set_xlabel("image (labelled by $q(t=1 | v)$)")
    fig.colorbar(im, ax=ax, label="$q(h_j = 1 | v)$", fraction=0.025)
    ax.set_title("Recognition activations $q(h | v)$ on each support image")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_hidden_specialization(model: bars.HelmholtzMachine, out_path: str
                               ) -> None:
    """For each hidden unit, plot the marginal generative effect on each pixel:
    p(v_i = 1 | h_j = 1, all others off, top inferred from W_th).

    This visualises which bar each hidden unit has specialised to.
    """
    # one-hot hidden activations, take p(v | h) under the corresponding top
    # value that h_j most likely belongs to (sign of W_th[0, j])
    n_h = model.n_hidden
    fig, axes = plt.subplots(2, n_h, figsize=(n_h * 1.3, 3.0), dpi=140,
                             gridspec_kw=dict(hspace=0.2, wspace=0.05))
    for j in range(n_h):
        # generative effect with only h_j on
        h_one = np.zeros((1, n_h), dtype=np.float32)
        h_one[0, j] = 1.0
        z_v = h_one @ model.W_hv + model.b_v
        p_v = bars.sigmoid(z_v).reshape(bars.H, bars.W)
        p_v_baseline = bars.sigmoid(model.b_v).reshape(bars.H, bars.W)
        delta = p_v - p_v_baseline

        ax_top = axes[0, j]
        ax_top.imshow(p_v, cmap="binary", vmin=0, vmax=1, interpolation="nearest")
        ax_top.set_xticks([])
        ax_top.set_yticks([])
        top_w = float(model.W_th[0, j])
        ax_top.set_title(f"h[{j}]\n$W_{{th}}$={top_w:+.2f}", fontsize=9)

        ax_bot = axes[1, j]
        m = max(abs(delta).max(), 1e-3)
        ax_bot.imshow(delta, cmap="seismic", vmin=-m, vmax=m,
                      interpolation="nearest")
        ax_bot.set_xticks([])
        ax_bot.set_yticks([])
        if j == 0:
            ax_bot.set_ylabel("$\\Delta$ vs baseline", fontsize=9)

    axes[0, 0].set_ylabel("$p(v | h_j=1)$", fontsize=9)
    fig.suptitle("Generative receptive field of each hidden unit",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_diagrams(model: bars.HelmholtzMachine, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), dpi=130)

    ax = axes[0]
    _hinton_diagram(ax, model.W_hv)
    ax.set_xticks(range(model.n_visible))
    ax.set_xticklabels([f"{i}" for i in range(model.n_visible)], fontsize=7)
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h[{j}]" for j in range(model.n_hidden)], fontsize=8)
    ax.set_xlabel("visible pixel")
    ax.set_title(f"Generative weights $W_{{hv}}$  "
                 f"($\\|W\\|_F$ = {np.linalg.norm(model.W_hv):.2f})",
                 fontsize=10)

    ax = axes[1]
    _hinton_diagram(ax, model.R_vh.T)
    ax.set_xticks(range(model.n_visible))
    ax.set_xticklabels([f"{i}" for i in range(model.n_visible)], fontsize=7)
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h[{j}]" for j in range(model.n_hidden)], fontsize=8)
    ax.set_xlabel("visible pixel")
    ax.set_title(f"Recognition weights $R_{{vh}}^T$  "
                 f"($\\|R\\|_F$ = {np.linalg.norm(model.R_vh):.2f})",
                 fontsize=10)

    fig.suptitle("Wake-sleep weights (red = +, blue = -, area $\\propto \\sqrt{|w|}$)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=2)
    p.add_argument("--n-steps", type=int, default=2_000_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--snapshot-every", type=int, default=50_000)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--reuse", action="store_true",
                   help="if set, load viz/canonical_model.npz instead of "
                        "re-training")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    if args.reuse and os.path.exists(os.path.join(args.outdir,
                                                  "canonical_model.npz")):
        print(f"Loading trained model from {args.outdir}...")
        model = load_model(args.outdir)
        a = np.load(os.path.join(args.outdir, "canonical_history.npz"))
        history = {"step": a["step"].tolist(),
                   "samples": a["samples"].tolist(),
                   "kl_bits": a["kl_bits"].tolist(),
                   "neg_log_lik": a["neg_log_lik"].tolist()}
    else:
        print(f"Training from scratch (seed={args.seed}, n_steps={args.n_steps})")
        model, history, _snap = train_and_save(
            args.seed, args.n_steps, args.lr, args.batch_size,
            eval_every=20_000, snapshot_every=args.snapshot_every,
            outdir=args.outdir)

    print(f"Final KL: {history['kl_bits'][-1]:.4f} bits")

    plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    plot_generated_samples(model,
                           os.path.join(args.outdir, "generated_samples.png"))
    plot_recognition_codes(model,
                           os.path.join(args.outdir, "recognition_codes.png"))
    plot_hidden_specialization(model,
                               os.path.join(args.outdir,
                                            "hidden_specialization.png"))
    plot_weight_diagrams(model,
                         os.path.join(args.outdir, "weights.png"))


if __name__ == "__main__":
    main()
