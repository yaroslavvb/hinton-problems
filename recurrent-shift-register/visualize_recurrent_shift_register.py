"""
Static visualizations for the trained shift-register RNN.

Outputs (in `viz/`):
  training_curves.png   - loss + accuracy + W_hh sparsity dynamics
  weights_W_hh.png      - heatmap of the recurrent matrix at convergence,
                            with the discovered chain overlaid (the headline).
  weights_io.png        - W_xh (input -> hidden) and W_hy (hidden -> output)
                            as bar charts, showing which units the input
                            writes to and which units the delay-d output
                            reads from.
  state_evolution.png   - hidden activations on a fresh test sequence,
                            making the shift-register dynamics visible: the
                            input bit propagates diagonally across (unit, t).
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

from recurrent_shift_register import (ShiftRegisterRNN, train,
                                        make_sequence, shift_matrix_score)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, n_units: int, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    epochs = history["epoch"]
    converged_at = history["converged_epoch"]

    ax = axes[0, 0]
    ax.plot(epochs, history["loss"], color="#9467bd", linewidth=1.4)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8,
                    label=f"converged @ sweep {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("sweep")
    ax.set_ylabel("masked MSE loss")
    ax.set_yscale("log")
    ax.set_title(f"Training loss  (N = {n_units} units)")
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    acc = np.array(history["accuracy"])
    ax.plot(epochs, acc * 100, color="#1f77b4", linewidth=1.4,
            label="overall (avg over delays)")
    per_d = np.array(history["per_delay_accuracy"])      # (T, N - 1)
    if per_d.size:
        for d in range(per_d.shape[1]):
            ax.plot(epochs, per_d[:, d] * 100, linewidth=0.9, alpha=0.5,
                    label=f"delay {d + 1}")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("sign-match accuracy (%)")
    ax.set_ylim(0, 105)
    ax.set_title("Accuracy by delay channel")
    ax.legend(fontsize=8, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(epochs, history["weight_norm"], color="#ff7f0e", linewidth=1.4,
            label=r"$\|W_{hh}\|_F$")
    ax.plot(epochs, history["shift_diag_mean"], color="#2ca02c",
            linewidth=1.2, alpha=0.9, label="chain mean |w|")
    ax.plot(epochs, history["shift_leak_max"], color="#d62728",
            linewidth=1.2, alpha=0.9, label="off-chain max |w|")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("|w|")
    ax.set_title(r"$W_{hh}$ entries: chain vs off-chain")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    sr = np.array(history["sparsity_ratio"])
    ax.plot(epochs, sr, color="#9467bd", linewidth=1.4)
    ax.axhline(0.2, color="gray", linestyle=":", linewidth=1.0,
                label="shift-matrix threshold (0.2)")
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.8)
    ax.set_xlabel("sweep")
    ax.set_ylabel("sparsity ratio  (off-chain max / chain mean)")
    ax.set_ylim(0, max(1.05, 1.05 * sr.max() if sr.size else 1.05))
    ax.set_title("Shift-matrix sparsity (lower = cleaner)")
    ax.legend(fontsize=9, loc="upper right")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# W_hh heatmap with chain overlay (the headline)
# ----------------------------------------------------------------------

def plot_W_hh(model: ShiftRegisterRNN, out_path: str):
    W = model.W_hh
    sm = shift_matrix_score(W)
    N = W.shape[0]

    fig, ax = plt.subplots(figsize=(5.5, 5.0), dpi=140)
    vmax = max(abs(W).max(), 1e-3)
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")

    # numeric overlay
    for i in range(N):
        for j in range(N):
            v = W[i, j]
            if abs(v) > 0.05:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=9, color=color)

    # chain overlay: green outlines around the discovered chain entries
    for r, c in sm["chain_positions"]:
        ax.add_patch(plt.Rectangle((c - 0.45, r - 0.45), 0.9, 0.9,
                                    fill=False, edgecolor="lime",
                                    linewidth=2.5))
    # mark the silent (input) row with a gray dashed outline
    s = sm["input_stage"]
    ax.add_patch(plt.Rectangle((-0.5, s - 0.5), N, 1.0,
                                fill=False, edgecolor="gray",
                                linewidth=1.0, linestyle="--"))
    ax.text(N - 0.4, s, "  input\n  stage", fontsize=8, va="center",
             color="dimgray")

    ax.set_xticks(range(N))
    ax.set_yticks(range(N))
    ax.set_xticklabels([f"h[{j}]" for j in range(N)])
    ax.set_yticklabels([f"h[{i}]" for i in range(N)])
    ax.set_xlabel(r"reads from previous timestep's column ($W_{hh}\, h_{t-1}$)")
    ax.set_ylabel(r"writes into row ($h_t$)")

    title = (f"$W_{{hh}}$ at convergence  (N = {N})\n"
              f"chain mean |w| = {sm['shift_diag_mean']:.2f}, "
              f"off-chain max = {sm['shift_leak_max']:.2f}, "
              f"sparsity = {sm['sparsity_ratio']:.2f}\n"
              f"is shift matrix? {sm['is_shift_matrix']}")
    ax.set_title(title, fontsize=10)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# W_xh and W_hy bar charts
# ----------------------------------------------------------------------

def plot_io_weights(model: ShiftRegisterRNN, out_path: str):
    fig, axes = plt.subplots(1, 2, figsize=(11, 3.8), dpi=130,
                              gridspec_kw={"width_ratios": [1, 1.3]})

    # ---- W_xh: input -> hidden ----
    ax = axes[0]
    N = model.n_units
    bars = ax.bar(np.arange(N), model.W_xh,
                   color=["#1f77b4" if v > 0 else "#d62728" for v in model.W_xh],
                   edgecolor="black", linewidth=0.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"h[{i}]" for i in range(N)])
    ax.set_ylabel(r"$w$")
    ax.set_title(r"$W_{xh}$: input $\to$ hidden  "
                  "(strongest = input register)")
    ax.grid(alpha=0.3, axis="y")

    # ---- W_hy: hidden -> output[delay 1..N-1] ----
    ax = axes[1]
    n_out = model.n_out
    vmax = max(abs(model.W_hy).max(), 1e-3)
    im = ax.imshow(model.W_hy, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    aspect="auto", interpolation="nearest")
    for i in range(n_out):
        for j in range(N):
            v = model.W_hy[i, j]
            if abs(v) > 0.1:
                color = "white" if abs(v) > 0.5 * vmax else "black"
                ax.text(j, i, f"{v:+.2f}", ha="center", va="center",
                         fontsize=9, color=color)
    ax.set_xticks(range(N))
    ax.set_xticklabels([f"h[{i}]" for i in range(N)])
    ax.set_yticks(range(n_out))
    ax.set_yticklabels([f"y[delay {d + 1}]" for d in range(n_out)])
    ax.set_title(r"$W_{hy}$: hidden $\to$ delay output  "
                  "(each row should pick one column)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# State evolution on a fresh sequence
# ----------------------------------------------------------------------

def plot_state_evolution(model: ShiftRegisterRNN, out_path: str,
                          n_units: int, sequence_len: int = 24, seed: int = 7):
    rng = np.random.default_rng(seed)
    x, target, mask = make_sequence(n_units, sequence_len, rng)
    fwd = model.forward(x[None, :])
    h = fwd["h"][0]                          # (T+1, N)
    y = fwd["y"][0]                          # (T, N - 1)

    fig, axes = plt.subplots(3, 1, figsize=(11, 5.5), dpi=130,
                              gridspec_kw={"height_ratios": [0.4, 1.4, 1.2]})

    # ---- input strip ----
    ax = axes[0]
    ax.imshow(x[None, :], cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
              interpolation="nearest")
    for t, v in enumerate(x):
        ax.text(t, 0, f"{int(v):+d}", ha="center", va="center", fontsize=8,
                color="white" if abs(v) > 0.5 else "black")
    ax.set_yticks([0]); ax.set_yticklabels(["input"])
    ax.set_xticks([])
    ax.set_title(f"Input stream + hidden state evolution (N = {n_units})",
                  fontsize=11)

    # ---- hidden state ----
    ax = axes[1]
    ax.imshow(h[1:].T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
              interpolation="nearest")
    ax.set_yticks(range(n_units))
    ax.set_yticklabels([f"h[{i}]" for i in range(n_units)])
    ax.set_xticks([])
    ax.set_ylabel("hidden unit")

    # diagonal arrows showing the shift register pathway, derived from the
    # chain links of the converged matrix
    sm = shift_matrix_score(model.W_hh)
    chain = sm["chain_positions"]
    chain_map = {c: r for (r, c) in chain}      # source -> dest
    input_stage = sm["input_stage"]
    # walk forward from the input stage
    walk = [input_stage]
    cursor = input_stage
    while cursor in chain_map and chain_map[cursor] not in walk:
        cursor = chain_map[cursor]
        walk.append(cursor)
    # annotate the walk on the right side
    if len(walk) == n_units:
        ax.text(sequence_len * 1.02, walk[0], "input",
                fontsize=8, va="center", color="black")
        for k, u in enumerate(walk):
            ax.text(sequence_len * 1.02, u, f"  delay {k}",
                    fontsize=8, va="center", color="black")

    # ---- output / target ----
    ax = axes[2]
    n_out = model.n_out
    # show predicted vs target side by side as a single "stripe" per delay,
    # with the predicted value drawn and the target marker on top.
    pred = np.sign(y); pred[pred == 0] = 1
    ax.imshow(y.T, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto",
              interpolation="nearest")
    # mark mismatches with a black 'x'
    for d in range(n_out):
        for t in range(sequence_len):
            if mask[t, d] > 0 and pred[t, d] != target[t, d]:
                ax.text(t, d, "x", ha="center", va="center", fontsize=9,
                         color="black", weight="bold")
    ax.set_yticks(range(n_out))
    ax.set_yticklabels([f"y[delay {d + 1}]" for d in range(n_out)])
    ax.set_xticks(range(sequence_len))
    ax.set_xticklabels([str(t) for t in range(sequence_len)], fontsize=7)
    ax.set_xlabel("timestep t")
    ax.set_ylabel("delay output")

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-units", type=int, default=3, choices=[3, 5, 4, 6, 7, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-sweeps", type=int, default=300)
    p.add_argument("--sequence-len", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--l1-W-hh", type=float, default=0.05)
    p.add_argument("--init-scale", type=float, default=0.2)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    print(f"Training {args.n_units}-unit shift register, "
          f"seed={args.seed}, sweeps={args.n_sweeps}...")
    model, history = train(n_units=args.n_units, n_sweeps=args.n_sweeps,
                            batch_size=args.batch_size,
                            sequence_len=args.sequence_len,
                            lr=args.lr, momentum=args.momentum,
                            weight_decay=args.weight_decay,
                            l1_W_hh=args.l1_W_hh,
                            init_scale=args.init_scale,
                            seed=args.seed, verbose=False)
    sm = shift_matrix_score(model.W_hh)
    print(f"  converged @ sweep {history['converged_epoch']},  "
          f"final acc {history['accuracy'][-1]*100:.1f}%,  "
          f"is shift matrix = {sm['is_shift_matrix']}")

    plot_training_curves(history, args.n_units,
                          os.path.join(args.outdir,
                                       f"training_curves_N{args.n_units}.png"))
    plot_W_hh(model, os.path.join(args.outdir,
                                    f"weights_W_hh_N{args.n_units}.png"))
    plot_io_weights(model, os.path.join(args.outdir,
                                          f"weights_io_N{args.n_units}.png"))
    plot_state_evolution(model, os.path.join(args.outdir,
                                                f"state_evolution_N{args.n_units}.png"),
                          n_units=args.n_units)


if __name__ == "__main__":
    main()
