"""
Static visualizations for N-bit parity backprop.

Outputs (in `viz/`):
  training_curves.png         — loss + classification accuracy + |W|
  weights.png                 — Hinton diagram of W1, W2, biases
  thermometer_code.png        — THE KEY VIZ: mean hidden-unit activation
                                grouped by input bit-count. Reveals whether
                                the network learned the thermometer code
                                (monotonic step pattern across bit-counts).
  hidden_activations_full.png — heatmap of every hidden unit's response to
                                every input pattern, sorted by bit-count.
  predictions.png             — model output vs target for all 2**N patterns,
                                sorted by parity then by bit-count.
  sweep_thermometer.png       — bar chart of monotonicity scores per seed
                                (only generated if --sweep > 0).
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from n_bit_parity import (
    ParityMLP, train, make_parity_data, thermometer_score,
    bit_count_for_inputs, sweep,
)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str, title: str = ""):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    ax = axes[0]
    ax.plot(epochs, history["loss"], color="#9467bd")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8,
                   label=f"converged @ {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE  (0.5 · mean (o-y)²)")
    ax.set_title("Training loss")
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(epochs, np.array(history["accuracy"]) * 100, color="#1f77b4")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.set_title("Classification accuracy")
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                   linewidth=0.9, alpha=0.8)
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|W\|_F$")
    ax.set_title("Weight norm")
    ax.grid(alpha=0.3)

    if title:
        fig.suptitle(title, fontsize=11, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Hinton-diagram weight viz
# ----------------------------------------------------------------------

def _hinton_rect(ax, x: float, y: float, w: float, max_abs: float,
                 max_size: float = 0.85):
    sz = max_size * (abs(w) / max_abs) ** 0.5
    color = "#cc0000" if w > 0 else "#003366"
    ax.add_patch(Rectangle((x - sz / 2, y - sz / 2), sz, sz,
                           facecolor=color, edgecolor="black", linewidth=0.4))


def plot_weights(model: ParityMLP, out_path: str):
    """Hinton diagram of W1 (rows = hidden, cols = inputs+bias) and W2."""
    n_hidden = model.n_hidden
    n_bits = model.n_bits

    fig, axes = plt.subplots(1, 2, figsize=(8, 0.6 * n_hidden + 2.5),
                              dpi=130,
                              gridspec_kw={"width_ratios": [n_bits + 1, 2]})
    # ---- W1: n_hidden x (n_bits + bias) ----
    ax = axes[0]
    W = np.column_stack([model.W1, model.b1[:, None]])
    max_abs = max(abs(W).max(), 1e-3)
    for i in range(n_hidden):
        for j in range(n_bits + 1):
            _hinton_rect(ax, j, i, W[i, j], max_abs)
    ax.set_xlim(-0.7, n_bits + 0.7)
    ax.set_ylim(-0.7, n_hidden - 0.3)
    ax.invert_yaxis()
    ax.set_xticks(range(n_bits + 1))
    ax.set_xticklabels([f"x{i+1}" for i in range(n_bits)] + ["bias"],
                        fontsize=9)
    ax.set_yticks(range(n_hidden))
    ax.set_yticklabels([f"h{i+1}" for i in range(n_hidden)], fontsize=10)
    ax.set_aspect("equal")
    ax.set_title(f"W1: input → hidden  ({n_hidden}×{n_bits})")

    # ---- W2: 1 x (n_hidden + bias) ----
    ax = axes[1]
    W = np.column_stack([model.W2, model.b2[:, None]])
    max_abs = max(abs(W).max(), 1e-3)
    for j in range(n_hidden + 1):
        _hinton_rect(ax, j, 0, W[0, j], max_abs)
    ax.set_xlim(-0.7, n_hidden + 0.7)
    ax.set_ylim(-0.7, 0.7)
    ax.invert_yaxis()
    ax.set_xticks(range(n_hidden + 1))
    ax.set_xticklabels([f"h{i+1}" for i in range(n_hidden)] + ["bias"],
                        fontsize=9, rotation=45)
    ax.set_yticks([0]); ax.set_yticklabels(["o"], fontsize=10)
    ax.set_aspect("equal")
    ax.set_title("W2: hidden → output")

    fig.suptitle(f"Final weights  (red = +,  blue = −,  size ∝ √|w|)",
                  fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# THERMOMETER-CODE VIZ — the centrepiece
# ----------------------------------------------------------------------

def plot_thermometer_code(model: ParityMLP, out_path: str):
    """Mean hidden-unit activation grouped by input bit-count.

    A perfect thermometer code has each hidden unit firing when the
    input bit-count exceeds a unique threshold k = 1..N. Visually that
    is a step ladder: hidden unit k is off for bit-count < k and on for
    bit-count >= k. We plot one curve per hidden unit; a thermometer
    code shows up as N parallel sigmoidal steps shifted along the
    bit-count axis. Negative-polarity units (firing when many bits are
    *off*) are flipped for display so they overlay the positive ones.
    """
    score = thermometer_score(model)
    levels = score["levels"]               # 0..N
    mean_by_level = score["mean_by_level"] # (H, N+1)
    polarities = score["polarities"]
    thresholds = score["thresholds"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), dpi=130,
                              gridspec_kw={"width_ratios": [1.3, 1]})

    # ---- left: line plot of mean activation vs bit-count ----
    ax = axes[0]
    cmap = plt.get_cmap("viridis", model.n_hidden)
    for hi in range(model.n_hidden):
        row = mean_by_level[hi]
        marker = "o" if polarities[hi] == 1 else "s"
        label = (f"h{hi+1}  thresh≈{thresholds[hi]}  "
                 f"polarity={'+' if polarities[hi]==1 else '−'}")
        ax.plot(levels, row, marker=marker, color=cmap(hi),
                linewidth=2.0, markersize=7, label=label)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
    ax.set_xticks(levels)
    ax.set_xlabel("input bit-count  (number of '+1' bits)")
    ax.set_ylabel("mean hidden-unit activation")
    ax.set_ylim(-0.05, 1.10)
    ax.set_title(f"Hidden-unit activation by input bit-count\n"
                  f"(N = {model.n_bits})")
    ax.legend(fontsize=8, loc="center right")
    ax.grid(alpha=0.3)

    # ---- right: heatmap of hidden activation, sorted by polarity then thresh ----
    ax = axes[1]
    # Sort hidden units so the thermometer ladder is visually obvious.
    # Flip negative-polarity units so they look like positive thresholds.
    display = mean_by_level.copy()
    for hi in range(model.n_hidden):
        if polarities[hi] == -1:
            display[hi] = display[hi][::-1]   # reverse along bit-count axis
    eff_thresh = []
    for hi in range(model.n_hidden):
        on_levels = np.where(display[hi] > 0.5)[0]
        eff_thresh.append(int(on_levels[0]) if on_levels.size > 0
                          else model.n_bits + 1)
    order = np.argsort(eff_thresh)
    im = ax.imshow(display[order], cmap="viridis", aspect="auto",
                    vmin=0, vmax=1)
    ax.set_xticks(range(len(levels)))
    ax.set_xticklabels(levels)
    ax.set_yticks(range(model.n_hidden))
    ax.set_yticklabels([f"h{order[i]+1}" + (" (−)" if polarities[order[i]] == -1 else "")
                         for i in range(model.n_hidden)], fontsize=9)
    ax.set_xlabel("input bit-count" + ("  (negative-polarity units flipped)"
                                         if any(p == -1 for p in polarities)
                                         else ""))
    ax.set_title("Sorted by effective threshold\n"
                  "(thermometer ladder if monotonic)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    is_thermo = score["is_thermometer"]
    mono = score["mean_monotonicity"]
    fig.suptitle(
        f"Thermometer-code analysis  —  "
        f"perfect thermometer: {'YES' if is_thermo else 'partial'}  ·  "
        f"mean monotonicity = {mono:.2f}",
        fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Per-pattern hidden activation heatmap
# ----------------------------------------------------------------------

def plot_hidden_activations_full(model: ParityMLP, out_path: str):
    """Hidden activation for every input pattern, sorted by bit-count."""
    X, y = make_parity_data(model.n_bits, bipolar=model.bipolar)
    h, _ = model.forward(X)                      # (2^N, H)
    bit_counts = bit_count_for_inputs(X)
    order = np.lexsort((np.arange(len(bit_counts)), bit_counts))
    h_sorted = h[order]
    bc_sorted = bit_counts[order]
    y_sorted = y[order, 0]

    fig, ax = plt.subplots(figsize=(8, 0.18 * len(X) + 1.5), dpi=130)
    im = ax.imshow(h_sorted, cmap="viridis", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(model.n_hidden))
    ax.set_xticklabels([f"h{i+1}" for i in range(model.n_hidden)])
    ylabels = [f"bc={bc}, parity={int(yi)}"
               for bc, yi in zip(bc_sorted, y_sorted)]
    ax.set_yticks(range(len(X)))
    ax.set_yticklabels(ylabels, fontsize=7)
    ax.set_xlabel("hidden unit")
    ax.set_ylabel("input pattern  (sorted by bit-count, then index)")
    ax.set_title(f"Hidden activations for all {len(X)} input patterns "
                  f"(N={model.n_bits})")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="activation")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Predictions
# ----------------------------------------------------------------------

def plot_predictions(model: ParityMLP, out_path: str):
    """Bar chart: model output vs target for every input pattern."""
    X, y = make_parity_data(model.n_bits, bipolar=model.bipolar)
    o = model.predict(X).ravel()
    y = y.ravel()
    bit_counts = bit_count_for_inputs(X)
    order = np.lexsort((np.arange(len(bit_counts)), bit_counts))

    fig, ax = plt.subplots(figsize=(0.45 * len(X) + 1.5, 4), dpi=130)
    xpos = np.arange(len(X))
    bar_colors = ["#1f77b4" if y[order][i] == 0 else "#d62728"
                  for i in range(len(X))]
    ax.bar(xpos, o[order], color=bar_colors, edgecolor="black",
           linewidth=0.4, alpha=0.85)
    ax.scatter(xpos, y[order], marker="x", s=70, color="black",
                label="target", zorder=3)
    ax.axhline(0.5, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xticks(xpos)
    bc = bit_counts[order]
    ax.set_xticklabels([str(b) for b in bc], fontsize=8)
    ax.set_xlabel("input pattern  (x-axis: bit-count)")
    ax.set_ylabel("output / target")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Predictions on all 2^{model.n_bits} = {len(X)} patterns "
                  f"(blue = target 0, red = target 1)")
    ax.legend(loc="upper left", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Multi-seed sweep viz (optional)
# ----------------------------------------------------------------------

def plot_sweep_thermometer(seed_summaries: list[dict], n_bits: int,
                            out_path: str):
    """Bar chart of monotonicity score per seed (only converged seeds)."""
    fig, ax = plt.subplots(figsize=(7, 3.5), dpi=130)
    seeds = [s["seed"] for s in seed_summaries]
    monos = [s["mean_monotonicity"] for s in seed_summaries]
    converged = [s["converged"] for s in seed_summaries]
    colors = ["#2ca02c" if c else "#888" for c in converged]
    ax.bar(seeds, monos, color=colors, edgecolor="black")
    ax.axhline(1.0, color="green", linestyle="--", linewidth=0.8,
               label="perfect monotonicity")
    ax.set_xticks(seeds)
    ax.set_xlabel("seed")
    ax.set_ylabel("mean monotonicity score")
    ax.set_ylim(0, 1.1)
    ax.set_title(f"Per-seed thermometer-likeness  (N = {n_bits})\n"
                  "green = converged, gray = did not converge")
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-bits", type=int, default=4)
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--max-epochs", type=int, default=30000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sweep", type=int, default=0,
                   help="If > 0, also produce sweep_thermometer.png "
                        "across this many seeds.")
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training N={args.n_bits} parity, seed={args.seed}, "
          f"max_epochs={args.max_epochs}...")
    model, history = train(n_bits=args.n_bits, n_hidden=args.n_hidden,
                            lr=args.lr, momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            verbose=False)
    print(f"  converged @ epoch {history['converged_epoch']},  "
          f"final acc {history['accuracy'][-1]*100:.0f}%")

    title = (f"N={args.n_bits} parity, seed={args.seed}, "
             f"converged={history['converged_epoch']}")
    plot_training_curves(history,
                         os.path.join(args.outdir, "training_curves.png"),
                         title=title)
    plot_weights(model, os.path.join(args.outdir, "weights.png"))
    plot_thermometer_code(model,
                          os.path.join(args.outdir, "thermometer_code.png"))
    plot_hidden_activations_full(
        model, os.path.join(args.outdir, "hidden_activations_full.png"))
    plot_predictions(model,
                     os.path.join(args.outdir, "predictions.png"))

    if args.sweep > 0:
        print(f"\nRunning {args.sweep}-seed sweep for monotonicity bar chart...")
        seed_summaries = []
        for s in range(args.sweep):
            m, h = train(n_bits=args.n_bits, n_hidden=args.n_hidden,
                         lr=args.lr, momentum=args.momentum,
                         init_scale=args.init_scale,
                         max_epochs=args.max_epochs, seed=s, verbose=False)
            score = thermometer_score(m)
            seed_summaries.append({
                "seed": s,
                "converged": h["converged_epoch"] is not None,
                "mean_monotonicity": score["mean_monotonicity"],
                "is_thermometer": score["is_thermometer"],
            })
        plot_sweep_thermometer(seed_summaries, args.n_bits,
                                os.path.join(args.outdir,
                                             "sweep_thermometer.png"))


if __name__ == "__main__":
    main()
