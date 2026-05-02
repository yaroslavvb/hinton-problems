"""
Static visualizations for the trained Helmholtz-shifter machine.

Outputs (in `viz/`):
  training_curves.png         -- IS-NLL and direction-recovery trajectories
  generated_samples.png       -- 64 fantasies drawn from the trained net
  layer3_selectivity.png      -- per-top-unit shift-direction selectivity bars
                                 + fantasies grouped by which top unit is on
  layer2_receptive_fields.png -- generative receptive field of each hidden unit
                                 (shows shifted bit-pair detection)
  reconstructions.png         -- input v + reconstructed v from recognise->generate
  weights.png                 -- Hinton-style diagram of W_hv and R_vh
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

import helmholtz_shifter as hs
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


def _draw_image_grid(ax, imgs: np.ndarray, n_rows: int, n_cols: int,
                     pad: int = 1, title: str | None = None) -> None:
    H, W = hs.H, hs.W
    canvas = np.full((n_rows * (H + pad) + pad, n_cols * (W + pad) + pad),
                     0.5, dtype=np.float32)
    for k in range(min(len(imgs), n_rows * n_cols)):
        r, c = divmod(k, n_cols)
        y0 = pad + r * (H + pad)
        x0 = pad + c * (W + pad)
        canvas[y0:y0 + H, x0:x0 + W] = imgs[k].reshape(H, W)
    ax.imshow(canvas, cmap="binary", vmin=0.0, vmax=1.0,
              interpolation="nearest")
    ax.set_xticks([])
    ax.set_yticks([])
    if title is not None:
        ax.set_title(title, fontsize=10)


# ------------------------------------------------------------------
# Plots
# ------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str) -> None:
    samples = np.array(history["samples"])
    nll = np.array(history["is_nll_bits"])
    dir_acc = np.array(history["dir_acc"])
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=130)

    ax = axes[0]
    ax.plot(samples / 1e6, nll, color="#9467bd")
    ax.set_xlabel("samples (millions)")
    ax.set_ylabel("IS-NLL  (bits/image)")
    ax.set_title("Negative log-likelihood (importance-sampled)")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(samples / 1e6, dir_acc, color="#1f77b4")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="chance (0.5)")
    ax.set_xlabel("samples (millions)")
    ax.set_ylabel("recognition accuracy")
    ax.set_ylim(0.45, 1.02)
    ax.set_title("Direction recovery from $q(t \\,|\\, v)$")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generated_samples(model: hs.HelmholtzMachine, out_path: str,
                           n: int = 64, seed: int = 99) -> None:
    rng = np.random.default_rng(seed)
    saved_rng = model.rng
    model.rng = rng
    v, _, _ = model.generate(n)
    model.rng = saved_rng
    fig, ax = plt.subplots(figsize=(8, 6), dpi=130)
    sig = hs._shift_signature(v)
    n_right = int((sig == +1).sum())
    n_left = int((sig == -1).sum())
    n_other = n - n_right - n_left
    _draw_image_grid(ax, v, n_rows=8, n_cols=8)
    ax.set_title(f"Samples from $p_\\mathrm{{model}}(v)$  "
                 f"(right-shift pattern: {n_right}/{n}, "
                 f"left: {n_left}/{n}, "
                 f"other: {n_other}/{n})",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_layer3_selectivity(model: hs.HelmholtzMachine, out_path: str,
                            n_fantasy: int = 2048, seed: int = 7) -> None:
    """Bar chart of per-top-unit selectivity, plus a small grid of fantasies
    conditioned on each one-hot top configuration."""
    rng = np.random.default_rng(seed)
    inspect = hs.inspect_layer3_units(model, n_fantasy=n_fantasy, rng=rng)

    n_top = model.n_top
    fig = plt.figure(figsize=(11, 5.5), dpi=130)
    gs = fig.add_gridspec(1, 2, width_ratios=[1.0, 1.6])

    # ---- left: bars ----------------------------------------------------
    ax = fig.add_subplot(gs[0, 0])
    p_right = inspect["p_right_given_top_on"]
    p_left = inspect["p_left_given_top_on"]
    x = np.arange(n_top)
    width = 0.35
    ax.bar(x - width / 2, p_right, width, label="P(right shift | $t_k=1$)",
           color="#cc6633")
    ax.bar(x + width / 2, p_left, width, label="P(left shift | $t_k=1$)",
           color="#3366cc")
    ax.set_xticks(x)
    ax.set_xticklabels([f"$t_{k}$" for k in range(n_top)])
    ax.set_ylabel("fraction of fantasies")
    ax.set_ylim(0, 1)
    ax.legend(fontsize=9, loc="upper right")
    best = inspect["best_unit"]
    ax.set_title("Layer-3 unit shift-direction selectivity\n"
                 f"(best unit: $t_{{{best}}}$, |sel| = "
                 f"{inspect['best_unit_selectivity']:.2f})",
                 fontsize=10)
    ax.grid(alpha=0.3, axis="y")

    # ---- right: fantasies grouped by which top unit is on --------------
    ax = fig.add_subplot(gs[0, 1])
    H, W = hs.H, hs.W
    pad = 1
    rows_per_unit = 4
    cols_per_unit = 8
    grid_h = n_top * (rows_per_unit * (H + pad) + pad) + (n_top - 1) * 2
    grid_w = cols_per_unit * (W + pad) + pad
    canvas = np.full((grid_h, grid_w), 0.5, dtype=np.float32)
    saved_rng = model.rng
    model.rng = rng
    y_cursor = 0
    for k in range(n_top):
        v_k, _ = model.generate_conditional(rows_per_unit * cols_per_unit,
                                            top_value=0)  # not used
        # one-hot conditioning
        t_on = np.zeros((rows_per_unit * cols_per_unit, n_top),
                        dtype=np.float32)
        t_on[:, k] = 1.0
        p_h = hs.sigmoid(t_on @ model.W_th + model.b_h)
        h = hs._bernoulli(p_h, model.rng)
        p_v = hs.sigmoid(h @ model.W_hv + model.b_v)
        v_k = hs._bernoulli(p_v, model.rng)
        for s in range(rows_per_unit * cols_per_unit):
            r, c = divmod(s, cols_per_unit)
            y0 = y_cursor + pad + r * (H + pad)
            x0 = pad + c * (W + pad)
            canvas[y0:y0 + H, x0:x0 + W] = v_k[s].reshape(H, W)
        y_cursor += rows_per_unit * (H + pad) + pad + 2
    model.rng = saved_rng
    ax.imshow(canvas, cmap="binary", vmin=0, vmax=1, interpolation="nearest",
              aspect="equal")
    ax.set_xticks([])
    yticks = []
    yticklabels = []
    block = rows_per_unit * (H + pad) + pad + 2
    for k in range(n_top):
        yticks.append(k * block + (rows_per_unit * (H + pad) + pad) / 2)
        yticklabels.append(f"$t_{k}=1$ only")
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels, fontsize=9)
    ax.set_title("Fantasies conditioned on each top unit (one-hot)",
                 fontsize=10)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_layer2_receptive_fields(model: hs.HelmholtzMachine, out_path: str
                                 ) -> None:
    """For each hidden unit: p(v | h_j=1, others off) - p(v | all h off),
    reshaped to 4x8 and plotted as a heat map.

    Also draws the recognition counterpart (R_vh column for unit j, reshaped to
    4x8) below each generative field, so you can see the shifted bit-pair
    structure on both sides.
    """
    fields = hs.inspect_layer2_units(model)         # (n_hidden, H, W)
    n_h = model.n_hidden
    n_cols = 8
    n_rows = (n_h + n_cols - 1) // n_cols
    fig, axes = plt.subplots(2 * n_rows, n_cols,
                             figsize=(n_cols * 1.4, n_rows * 2.6),
                             dpi=140,
                             gridspec_kw=dict(hspace=0.25, wspace=0.10))
    if n_rows == 1:
        axes = axes.reshape(2, n_cols)
    max_abs_gen = max(abs(fields).max(), 1e-3)
    rec = model.R_vh.T.reshape(n_h, hs.H, hs.W)
    max_abs_rec = max(abs(rec).max(), 1e-3)
    for j in range(n_h):
        gr, gc = divmod(j, n_cols)
        ax_top = axes[2 * gr, gc]
        ax_top.imshow(fields[j], cmap="seismic",
                      vmin=-max_abs_gen, vmax=max_abs_gen,
                      interpolation="nearest")
        ax_top.set_xticks([]); ax_top.set_yticks([])
        wth = ", ".join(f"{model.W_th[k, j]:+.1f}"
                        for k in range(model.n_top))
        ax_top.set_title(f"h[{j}]\n$W_{{th}}$=[{wth}]", fontsize=7)
        ax_bot = axes[2 * gr + 1, gc]
        ax_bot.imshow(rec[j], cmap="seismic",
                      vmin=-max_abs_rec, vmax=max_abs_rec,
                      interpolation="nearest")
        ax_bot.set_xticks([]); ax_bot.set_yticks([])
        if gc == 0:
            ax_top.set_ylabel("$\\Delta p(v | h_j=1)$", fontsize=8)
            ax_bot.set_ylabel("$R_{vh}^T$", fontsize=8)

    # hide empty slots if any
    for j in range(n_h, n_rows * n_cols):
        gr, gc = divmod(j, n_cols)
        axes[2 * gr, gc].set_visible(False)
        axes[2 * gr + 1, gc].set_visible(False)

    fig.suptitle("Layer-2 (hidden) generative receptive fields (top row) "
                 "and recognition rows (bottom row)\n"
                 "Each unit detects a shifted bit-pair: row 0 pixel "
                 "+ row 3 pixel at offset $\\pm 1$.",
                 fontsize=11, y=1.0)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(model: hs.HelmholtzMachine, out_path: str,
                         n: int = 16, seed: int = 13) -> None:
    """Show n input shifter images alongside their recognise -> generate
    reconstructions.

    For each input v: sample (h, t) ~ q(.|v), then sample v_recon ~ p(v|h)
    using the *mean* (not bernoulli) so the reconstruction shows the model's
    top-down belief about each pixel.
    """
    rng = np.random.default_rng(seed)
    v, dirs = hs.generate_shifter(n, p_on=model.p_on, rng=rng)
    saved_rng = model.rng
    model.rng = rng
    h, t = model.recognize(v)
    model.rng = saved_rng
    p_v_recon = hs.sigmoid(h @ model.W_hv + model.b_v)

    fig, axes = plt.subplots(2, 1, figsize=(11, 3.5), dpi=130,
                             gridspec_kw=dict(hspace=0.2))
    H, W = hs.H, hs.W
    pad = 1
    canvas_w = n * (W + pad) + pad
    canvas_h = H + 2 * pad
    canvas_in = np.full((canvas_h, canvas_w), 0.5, dtype=np.float32)
    canvas_out = np.full((canvas_h, canvas_w), 0.5, dtype=np.float32)
    for k in range(n):
        x0 = pad + k * (W + pad)
        canvas_in[pad:pad + H, x0:x0 + W] = v[k].reshape(H, W)
        canvas_out[pad:pad + H, x0:x0 + W] = p_v_recon[k].reshape(H, W)
    axes[0].imshow(canvas_in, cmap="binary", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    axes[0].set_title("Input shifter images (top row + duplicate; "
                      "bottom row + duplicate)", fontsize=10)
    axes[0].set_xticks([]); axes[0].set_yticks([])

    axes[1].imshow(canvas_out, cmap="binary", vmin=0, vmax=1,
                   interpolation="nearest", aspect="auto")
    axes[1].set_title("Reconstruction $p(v \\,|\\, h)$  "
                      "where $h \\sim q(h \\,|\\, v)$", fontsize=10)
    axes[1].set_xticks([]); axes[1].set_yticks([])

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_diagrams(model: hs.HelmholtzMachine, out_path: str) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), dpi=130,
                             gridspec_kw=dict(hspace=0.4))
    ax = axes[0]
    _hinton_diagram(ax, model.W_hv)
    ax.set_xticks(range(model.n_visible))
    ax.set_xticklabels([f"{i}" for i in range(model.n_visible)], fontsize=6)
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h[{j}]" for j in range(model.n_hidden)], fontsize=8)
    ax.set_xlabel("visible pixel (row-major over 4x8 image)")
    ax.set_title(f"Generative $W_{{hv}}$  "
                 f"($\\|W\\|_F$ = {np.linalg.norm(model.W_hv):.2f})",
                 fontsize=10)

    ax = axes[1]
    _hinton_diagram(ax, model.R_vh.T)
    ax.set_xticks(range(model.n_visible))
    ax.set_xticklabels([f"{i}" for i in range(model.n_visible)], fontsize=6)
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h[{j}]" for j in range(model.n_hidden)], fontsize=8)
    ax.set_xlabel("visible pixel")
    ax.set_title(f"Recognition $R_{{vh}}^T$  "
                 f"($\\|R\\|_F$ = {np.linalg.norm(model.R_vh):.2f})",
                 fontsize=10)

    fig.suptitle("Wake-sleep weights "
                 "(red = +, blue = -, area $\\propto \\sqrt{|w|}$)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--n-passes", type=int, default=1_500_000)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--snapshot-every", type=int, default=50_000)
    p.add_argument("--p-on", type=float, default=hs.P_ON)
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
                   "is_nll_bits": a["is_nll_bits"].tolist(),
                   "dir_acc": a["dir_acc"].tolist()}
    else:
        print(f"Training from scratch (seed={args.seed}, "
              f"n_passes={args.n_passes})")
        model, history, _snap = train_and_save(
            args.seed, args.n_passes, args.lr, args.batch_size,
            eval_every=25_000, snapshot_every=args.snapshot_every,
            outdir=args.outdir, p_on=args.p_on)

    print(f"Final IS-NLL: {history['is_nll_bits'][-1]:.4f} bits")
    print(f"Final dir-acc: {history['dir_acc'][-1]:.4f}")

    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_generated_samples(model,
                           os.path.join(args.outdir, "generated_samples.png"))
    plot_layer3_selectivity(model,
                            os.path.join(args.outdir,
                                         "layer3_selectivity.png"))
    plot_layer2_receptive_fields(model,
                                 os.path.join(args.outdir,
                                              "layer2_receptive_fields.png"))
    plot_reconstructions(model,
                         os.path.join(args.outdir, "reconstructions.png"))
    plot_weight_diagrams(model,
                         os.path.join(args.outdir, "weights.png"))


if __name__ == "__main__":
    main()
