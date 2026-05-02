"""
Static visualizations for the fast-weights associative-retrieval model.

Outputs (in `viz/`):
  training_curves.png             Training + eval loss/accuracy through training.
  fast_weights_evolution.png      A_t heatmap snapshots at each timestep of one
                                  example sequence. Shows the matrix building up
                                  outer-product traces and decaying.
  per_pair_slot_acc.png           Accuracy broken down by which slot the query
                                  letter occupied (slot 0 = oldest, slot n-1 =
                                  most recent).
  hidden_state_trace.png          h_t and z_t trajectories across the timesteps
                                  of an example sequence; highlights how the
                                  query and trailing-read steps differ from the
                                  bulk of the sequence.
"""

from __future__ import annotations

import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from fast_weights_associative_retrieval import (
    FastWeightsRNN, build_fast_weights_rnn, generate_sample, generate_batch,
    train, per_position_accuracy,
    VOCAB_SIZE, N_LETTERS, N_DIGITS, SEP_INDEX, _token_str,
)


def _ensure_viz_dir(out_dir: str) -> None:
    os.makedirs(out_dir, exist_ok=True)


def plot_training_curves(history: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0), dpi=120)
    steps = np.array(history["step"])
    ax = axes[0]
    ax.plot(steps, history["train_loss"], color="#1f77b4", marker="o",
            markersize=3, label="train (mean over interval)")
    ax.plot(steps, history["eval_loss"], color="#d62728", marker="s",
            markersize=3, label="eval (256 samples)")
    ax.axhline(np.log(N_DIGITS), color="gray", linestyle=":",
               label=f"chance (log {N_DIGITS} = {np.log(N_DIGITS):.2f})")
    ax.set_xlabel("training step")
    ax.set_ylabel("cross-entropy loss")
    ax.set_title("Loss")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    ax = axes[1]
    ax.plot(steps, np.array(history["train_acc"]) * 100, color="#1f77b4",
            marker="o", markersize=3, label="train")
    ax.plot(steps, np.array(history["eval_acc"]) * 100, color="#d62728",
            marker="s", markersize=3, label="eval")
    ax.axhline(100.0 / N_DIGITS, color="gray", linestyle=":",
               label=f"chance ({100.0/N_DIGITS:.0f}%)")
    ax.set_xlabel("training step")
    ax.set_ylabel("retrieval accuracy (%)")
    ax.set_title("Accuracy")
    ax.set_ylim(-2, 102)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)

    fig.suptitle("Fast-weights associative retrieval -- training", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_fast_weights_evolution(model: FastWeightsRNN, seq: np.ndarray,
                                target: int, text: str,
                                out_path: str) -> None:
    """Heatmap snapshots of A_t at every step of the sequence."""
    fwd = model.forward_sequence(seq, keep_trace=True)
    As = fwd["A_trace"]                              # (T, H, H)
    pred = int(np.argmax(fwd["logits"]))
    T = As.shape[0]
    n_cols = T
    fig, axes = plt.subplots(1, n_cols, figsize=(1.2 * n_cols + 0.6, 1.6),
                             dpi=130)
    if n_cols == 1:
        axes = [axes]
    vmax = float(np.max(np.abs(As))) + 1e-6
    for t in range(T):
        ax = axes[t]
        ax.imshow(As[t], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                  interpolation="nearest")
        tok = _token_str(int(seq[t]))
        ax.set_title(f"t={t}\n'{tok}'", fontsize=8)
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle(
        f"A_t evolution for sequence  '{text}'   "
        f"target={target}  pred={pred}",
        fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.85))
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_per_slot_accuracy(per_slot: np.ndarray, out_path: str) -> None:
    n_pairs = len(per_slot)
    fig, ax = plt.subplots(figsize=(5, 3.4), dpi=130)
    xs = np.arange(n_pairs)
    bars = ax.bar(xs, per_slot * 100, color="#3a7", edgecolor="black",
                  linewidth=0.4)
    for i, (x, b) in enumerate(zip(xs, bars)):
        ax.text(x, per_slot[i] * 100 + 1.5, f"{per_slot[i]*100:.1f}%",
                ha="center", va="bottom", fontsize=8)
    ax.axhline(100.0 / N_DIGITS, color="gray", linestyle=":",
               label=f"chance ({100.0/N_DIGITS:.0f}%)")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"slot {i}\n(time {2*i}-{2*i+1})" for i in xs],
                       fontsize=8)
    ax.set_ylim(0, 110)
    ax.set_ylabel("retrieval accuracy (%)")
    ax.set_title("Accuracy by which (key, value) slot was queried")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_trace(model: FastWeightsRNN, seq: np.ndarray, target: int,
                      text: str, out_path: str) -> None:
    fwd = model.forward_sequence(seq, keep_trace=True)
    hs = fwd["h_trace"]                              # (T+1, H)
    T = int(seq.shape[0])
    H = hs.shape[1]

    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.5), dpi=120,
                             gridspec_kw={"height_ratios": [3, 1]})
    ax = axes[0]
    im = ax.imshow(hs[1:].T, aspect="auto", cmap="RdBu_r", vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_xlabel("timestep")
    ax.set_ylabel("hidden unit index")
    ax.set_xticks(range(T))
    ax.set_xticklabels([_token_str(int(seq[t])) for t in range(T)], fontsize=9)
    ax.set_title(f"Hidden state h_t per step  '{text}'  target={target}",
                 fontsize=10)
    fig.colorbar(im, ax=ax, label="h_t value", fraction=0.025, pad=0.01)

    ax = axes[1]
    norms = np.linalg.norm(hs[1:], axis=1)
    ax.plot(range(T), norms, color="#3a7", marker="o", markersize=4)
    ax.set_xticks(range(T))
    ax.set_xticklabels([_token_str(int(seq[t])) for t in range(T)], fontsize=9)
    ax.set_xlabel("timestep")
    ax.set_ylabel("||h_t||")
    ax.set_title("Hidden-state norm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    p = argparse.ArgumentParser(
        description="Generate static visualizations for fast-weights "
                    "associative retrieval.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--n-hidden", type=int, default=80)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--out-dir", type=str, default="viz")
    args = p.parse_args()

    _ensure_viz_dir(args.out_dir)

    print(f"[viz] training fast-weights RNN  n_pairs={args.n_pairs}  "
          f"hidden={args.n_hidden}  steps={args.n_steps}  seed={args.seed}")
    t0 = time.time()
    model = build_fast_weights_rnn(
        n_in=VOCAB_SIZE, n_hidden=args.n_hidden, n_out=N_DIGITS,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed)
    history = train(model, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_pairs=args.n_pairs,
                    lr=args.lr, eval_every=args.eval_every,
                    eval_batch=256, grad_clip=5.0,
                    seed=args.seed, verbose=False)
    print(f"[viz] training done in {time.time()-t0:.1f}s  "
          f"final eval acc = {history['eval_acc'][-1]*100:.1f}%")

    plot_training_curves(history,
                         os.path.join(args.out_dir, "training_curves.png"))
    print(f"[viz] wrote training_curves.png")

    per_slot = per_position_accuracy(model, n_pairs=args.n_pairs,
                                     n_eval=2000, seed=args.seed + 10000)
    plot_per_slot_accuracy(per_slot,
                           os.path.join(args.out_dir, "per_pair_slot_acc.png"))
    print(f"[viz] wrote per_pair_slot_acc.png")

    rng = np.random.default_rng(args.seed + 1234)
    seq, tgt, text = generate_sample(n_pairs=args.n_pairs, rng=rng)
    plot_fast_weights_evolution(
        model, seq, tgt, text,
        os.path.join(args.out_dir, "fast_weights_evolution.png"))
    print(f"[viz] wrote fast_weights_evolution.png  for sequence '{text}'")

    plot_hidden_trace(
        model, seq, tgt, text,
        os.path.join(args.out_dir, "hidden_state_trace.png"))
    print(f"[viz] wrote hidden_state_trace.png")

    print(f"[viz] all outputs in '{args.out_dir}/'")


if __name__ == "__main__":
    main()
