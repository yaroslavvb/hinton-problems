"""
Static visualizations for the trained 25-sequence look-up RNN.

Outputs (in `viz/` by default):
  training_curves.png       - train + test accuracy + loss + per-bit acc
  weights_W_xh.png          - heatmap: hidden unit x input letter
  weights_W_hh.png          - heatmap: recurrent matrix
  weights_W_hy.png          - heatmap: 3 output bits x hidden units
  state_evolution.png       - hidden activations across timesteps for the
                                5 held-out test sequences (one panel each)
  generalization_summary.png - bar chart: per-sequence train + test correctness
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from sequence_lookup_25 import (SequenceLookupRNN, train, generate_dataset,
                                 test_generalization, ALPHABET_SIZE, SEQ_LEN,
                                 N_OUT)


def _seq_label(letters_row, letter_names="ABCDE"):
    return "".join(letter_names[int(l)] for l in letters_row)


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, n_hidden: int, out_path: str):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6.5), dpi=120)
    sweeps = history["sweep"]
    converged_at = history["converged_sweep"]

    ax = axes[0, 0]
    ax.plot(sweeps, history["loss"], color="#9467bd", linewidth=1.4)
    if converged_at:
        ax.axvline(converged_at, color="green", linestyle="--",
                    linewidth=0.9, alpha=0.7,
                    label=f"train converged @ {converged_at}")
        ax.legend(fontsize=9)
    ax.set_xlabel("sweep")
    ax.set_ylabel("MSE loss")
    ax.set_yscale("log")
    ax.set_title(f"Training loss (hidden={n_hidden})")
    ax.grid(alpha=0.3, which="both")

    ax = axes[0, 1]
    ax.plot(sweeps, np.array(history["train_acc"]) * 100,
            color="#1f77b4", linewidth=1.6, label="train (20 sequences)")
    ax.plot(sweeps, np.array(history["test_acc"]) * 100,
            color="#d62728", linewidth=1.6, label="held-out test (5)")
    ax.axhline(80, color="grey", linestyle=":", linewidth=0.8,
                label="4/5 = 80%")
    ax.axhline(100, color="grey", linestyle=":", linewidth=0.8,
                alpha=0.4)
    ax.set_xlabel("sweep")
    ax.set_ylabel("all-3-bits accuracy (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("Train + held-out accuracy")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    train_pb = np.array(history["train_per_bit"])  # (n_sweeps, 3)
    for b in range(N_OUT):
        ax.plot(sweeps, train_pb[:, b] * 100, linewidth=1.3,
                label=f"bit {b}")
    ax.set_xlabel("sweep")
    ax.set_ylabel("train per-bit acc (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("Per-bit accuracy on train set")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    test_pb = np.array(history["test_per_bit"])
    for b in range(N_OUT):
        ax.plot(sweeps, test_pb[:, b] * 100, linewidth=1.3,
                label=f"bit {b}")
    ax.set_xlabel("sweep")
    ax.set_ylabel("test per-bit acc (%)")
    ax.set_ylim(-3, 105)
    ax.set_title("Per-bit accuracy on held-out set")
    ax.legend(fontsize=9, loc="lower right")
    ax.grid(alpha=0.3)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Weight visualizations
# ----------------------------------------------------------------------

def plot_weight_matrix(W: np.ndarray, out_path: str, title: str,
                       row_label: str, col_label: str,
                       letter_names: str = "ABCDE"):
    H, K = W.shape
    fig, ax = plt.subplots(figsize=(max(5, K * 0.35),
                                      max(4, H * 0.18)),
                           dpi=120)
    vmax = max(abs(W).max(), 1e-6)
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest", aspect="auto")
    ax.set_xlabel(col_label)
    ax.set_ylabel(row_label)
    ax.set_title(title)
    if K == ALPHABET_SIZE and "letter" in col_label.lower():
        ax.set_xticks(range(K))
        ax.set_xticklabels(list(letter_names[:K]))
    plt.colorbar(im, ax=ax, fraction=0.04)
    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Hidden state evolution
# ----------------------------------------------------------------------

def plot_state_evolution(model: SequenceLookupRNN, data: dict,
                         variable_timing: bool, out_path: str,
                         letter_names: str = "ABCDE"):
    """Show hidden activations across time for the 5 held-out sequences."""
    test_idx = data["test_idx"]
    fig, axes = plt.subplots(len(test_idx), 1,
                              figsize=(8.5, 1.7 * len(test_idx)), dpi=120,
                              sharex=False)
    if len(test_idx) == 1:
        axes = [axes]

    for ax_i, seq_i in enumerate(test_idx):
        if variable_timing:
            x_i = data["variable_inputs"][seq_i][None]      # not used; do scalar
            x_seq = data["variable_inputs"][seq_i]
            T_i = x_seq.shape[0]
            h_i = np.zeros((T_i + 1, model.n_hidden))
            for t in range(T_i):
                z_h = (x_seq[t] @ model.W_xh.T
                       + h_i[t] @ model.W_hh.T
                       + model.b_h)
                h_i[t + 1] = np.tanh(z_h)
            states = h_i[1:].T   # (n_hidden, T_i)
            xtick_labels = []
            timings = data["timings"][seq_i]
            cursor = 0
            for k, tmg in enumerate(timings):
                for r in range(int(tmg)):
                    xtick_labels.append(letter_names[
                        int(data["letters"][seq_i, k])])
                    cursor += 1
        else:
            x_i = data["one_hot"][seq_i:seq_i + 1]
            fwd = model.forward(x_i)
            states = fwd["h"][0, 1:, :].T   # (n_hidden, T)
            xtick_labels = [letter_names[int(l)]
                              for l in data["letters"][seq_i]]

        ax = axes[ax_i]
        vmax = max(abs(states).max(), 1e-6)
        ax.imshow(states, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                   aspect="auto", interpolation="nearest")
        ax.set_xticks(range(states.shape[1]))
        ax.set_xticklabels(xtick_labels, fontsize=9)
        ax.set_ylabel(f"seq #{seq_i}\n{_seq_label(data['letters'][seq_i], letter_names)}",
                       fontsize=9)
        if ax_i == 0:
            ax.set_title("Hidden state evolution on held-out sequences")
        if ax_i == len(test_idx) - 1:
            ax.set_xlabel("input letter at each timestep")

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Generalization summary bar chart
# ----------------------------------------------------------------------

def plot_generalization_summary(model: SequenceLookupRNN, data: dict,
                                 variable_timing: bool, out_path: str,
                                 letter_names: str = "ABCDE"):
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]
    targets = data["targets"]

    if variable_timing:
        train_inputs = [data["variable_inputs"][i] for i in train_idx]
        test_inputs = [data["variable_inputs"][i] for i in test_idx]
        y_train = np.stack(model.forward_variable(train_inputs)["ys"])
        y_test = np.stack(model.forward_variable(test_inputs)["ys"])
    else:
        y_train = model.forward(data["one_hot"][train_idx])["y"]
        y_test = model.forward(data["one_hot"][test_idx])["y"]

    pred_tr = np.sign(y_train); pred_tr[pred_tr == 0] = 1.0
    pred_te = np.sign(y_test); pred_te[pred_te == 0] = 1.0
    correct_tr = np.all(pred_tr == targets[train_idx], axis=-1)
    correct_te = np.all(pred_te == targets[test_idx], axis=-1)

    fig, axes = plt.subplots(1, 2, figsize=(13, 4), dpi=120,
                              gridspec_kw={"width_ratios": [4, 1]})

    ax = axes[0]
    n_tr = len(train_idx)
    bar_x = np.arange(n_tr)
    colors = ["#2ca02c" if c else "#d62728" for c in correct_tr]
    ax.bar(bar_x, np.ones(n_tr), color=colors, edgecolor="black",
            linewidth=0.4)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([_seq_label(data["letters"][i], letter_names)
                          for i in train_idx],
                         rotation=70, fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Train: {int(correct_tr.sum())}/{n_tr} correct  "
                  f"(green = all 3 bits right, red = wrong)")
    ax.set_ylim(0, 1)

    ax = axes[1]
    n_te = len(test_idx)
    bar_x = np.arange(n_te)
    colors = ["#2ca02c" if c else "#d62728" for c in correct_te]
    ax.bar(bar_x, np.ones(n_te), color=colors, edgecolor="black",
            linewidth=0.4)
    ax.set_xticks(bar_x)
    ax.set_xticklabels([_seq_label(data["letters"][i], letter_names)
                          for i in test_idx],
                         rotation=70, fontsize=8)
    ax.set_yticks([])
    ax.set_title(f"Held-out: {int(correct_te.sum())}/{n_te} correct")
    ax.set_ylim(0, 1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dataset-seed", type=int, default=0)
    p.add_argument("--variable-timing", action="store_true")
    p.add_argument("--n-hidden", type=int, default=None)
    p.add_argument("--n-sweeps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None)
    p.add_argument("--max-timing", type=int, default=2)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    if args.n_hidden is None:
        args.n_hidden = 60 if args.variable_timing else 30
    if args.n_sweeps is None:
        args.n_sweeps = 2000 if args.variable_timing else 800
    if args.lr is None:
        args.lr = 0.02 if args.variable_timing else 0.05
    if args.init_scale is None:
        args.init_scale = 0.2 if args.variable_timing else 0.5
    if args.grad_clip is None:
        args.grad_clip = 1.0 if args.variable_timing else 5.0

    os.makedirs(args.outdir, exist_ok=True)

    print(f"# training (variable_timing={args.variable_timing}, "
          f"hidden={args.n_hidden}, sweeps={args.n_sweeps}, seed={args.seed})")
    model, hist, data = train(n_hidden=args.n_hidden, n_sweeps=args.n_sweeps,
                               lr=args.lr, init_scale=args.init_scale,
                               grad_clip=args.grad_clip, seed=args.seed,
                               dataset_seed=args.dataset_seed,
                               variable_timing=args.variable_timing,
                               max_timing=args.max_timing,
                               verbose=False)

    suffix = "_variable" if args.variable_timing else ""

    plot_training_curves(hist, args.n_hidden,
                          os.path.join(args.outdir,
                                        f"training_curves{suffix}.png"))
    plot_weight_matrix(model.W_xh,
                        os.path.join(args.outdir, f"weights_W_xh{suffix}.png"),
                        title=f"W_xh: hidden ({args.n_hidden}) x letter (5)",
                        row_label="hidden unit",
                        col_label="input letter")
    plot_weight_matrix(model.W_hh,
                        os.path.join(args.outdir, f"weights_W_hh{suffix}.png"),
                        title=f"W_hh: hidden x hidden ({args.n_hidden} x "
                               f"{args.n_hidden})",
                        row_label="to hidden", col_label="from hidden")
    plot_weight_matrix(model.W_hy,
                        os.path.join(args.outdir, f"weights_W_hy{suffix}.png"),
                        title="W_hy: 3 output bits x hidden",
                        row_label="output bit", col_label="hidden unit")
    plot_state_evolution(model, data, args.variable_timing,
                          os.path.join(args.outdir,
                                        f"state_evolution{suffix}.png"))
    plot_generalization_summary(model, data, args.variable_timing,
                                  os.path.join(args.outdir,
                                                f"generalization_summary{suffix}.png"))

    gen = test_generalization(model, data, variable_timing=args.variable_timing)
    print(f"# final train_acc={hist['train_acc'][-1]*100:.1f}%  "
          f"test_correct={gen['n_correct']}/{gen['n_total']}")


if __name__ == "__main__":
    main()
