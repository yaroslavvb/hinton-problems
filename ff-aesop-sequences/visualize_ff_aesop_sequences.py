"""
Static visualisations for the FF-on-Aesop run.

Produces (in ``viz/``):
  per_char_accuracy_curves.png   per-epoch accuracy for both negative
                                 variants + random-fixed-hidden + unigram.
  per_position_accuracy.png      accuracy at each predicted character index
                                 (0..89) for both variants vs unigram.
  generated_samples.png          a sample autoregressive rollout from each
                                 variant + the corresponding ground truth.
  goodness_layers.png            per-layer goodness (pos vs neg) over training.

If ``--model-tf`` and ``--model-sg`` point to existing .npz files (saved with
``ff_aesop_sequences.py --save``) they are loaded; otherwise both variants are
trained from scratch using the CLI args.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt

from ff_aesop_sequences import (
    ALPHABET, N_SYMBOLS, WINDOW, INPUT_DIM,
    load_aesop_strings, build_ff_seq_model, train, TrainConfig,
    save_run, load_saved_model,
    evaluate_per_char_accuracy,
    random_fixed_hidden_baseline, unigram_baseline,
    make_negatives_self_generated, predict_next_char_batch,
)


# ---------------------------------------------------------------------------
# Train (or load) both variants
# ---------------------------------------------------------------------------

def train_or_load(variant: str, indices: np.ndarray, args, cache_path: str
                  ) -> tuple:
    """Either load a saved run or train from scratch and save."""
    if cache_path and os.path.exists(cache_path):
        print(f"  loading cached {variant} run: {cache_path}")
        return load_saved_model(cache_path)

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    cfg = TrainConfig(n_epochs=args.n_epochs,
                      batch_size=args.batch_size,
                      steps_per_epoch=args.steps_per_epoch,
                      lr=args.lr,
                      threshold=2.0,
                      layer_sizes=layer_sizes,
                      seed=args.seed,
                      negatives=variant,
                      eval_every=args.eval_every,
                      rollout_every=args.rollout_every,
                      rollout_temperature=args.rollout_temperature)
    model = build_ff_seq_model(seed=args.seed, threshold=2.0,
                               layer_sizes=layer_sizes)
    print(f"\nTraining variant={variant}, layers={layer_sizes}, "
          f"epochs={cfg.n_epochs}, lr={cfg.lr} ...")
    history = train(model, indices, cfg, verbose=True)
    if cache_path:
        save_run(cache_path, model, history, cfg)
        print(f"  saved -> {cache_path}")
    return model, history, variant


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_accuracy_curves(hist_tf: dict, hist_sg: dict,
                         rand_acc: float, uni_acc: float,
                         out_path: str) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 5.0), dpi=120)
    ax.plot(hist_tf["epoch"], np.array(hist_tf["test_acc"]) * 100,
            color="#1f77b4", linewidth=1.8, label="teacher-forcing negatives")
    ax.plot(hist_sg["epoch"], np.array(hist_sg["test_acc"]) * 100,
            color="#d62728", linewidth=1.8, label="self-generated negatives")
    ax.axhline(uni_acc * 100, color="#2ca02c", linewidth=1.0, linestyle="--",
               label=f"unigram baseline ({uni_acc*100:.1f}%)")
    ax.axhline(rand_acc * 100, color="#7f7f7f", linewidth=1.0, linestyle=":",
               label=f"random fixed hidden ({rand_acc*100:.1f}%)")
    ax.axhline(100.0 / N_SYMBOLS, color="#bbbbbb", linewidth=0.7, linestyle=":",
               label=f"chance (1/30 = {100.0/N_SYMBOLS:.2f}%)")
    ax.set_xlabel("epoch")
    ax.set_ylabel("per-character accuracy (%)")
    ax.set_title("FF on Aesop's Fables: per-char accuracy under both negative variants")
    ax.set_ylim(0, max(60.0, max(hist_tf["test_acc"] + hist_sg["test_acc"]) * 100 + 5))
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_per_position_accuracy(model_tf, model_sg, indices: np.ndarray,
                               uni_per_pos: np.ndarray, out_path: str) -> None:
    """Accuracy as a function of predicted-character index 10..99."""
    _, per_pos_tf = evaluate_per_char_accuracy(model_tf, indices)
    _, per_pos_sg = evaluate_per_char_accuracy(model_sg, indices)
    positions = np.arange(WINDOW, WINDOW + per_pos_tf.shape[0])

    fig, ax = plt.subplots(figsize=(9.0, 4.5), dpi=120)
    ax.plot(positions, per_pos_tf * 100, color="#1f77b4", linewidth=1.4,
            label="teacher-forcing")
    ax.plot(positions, per_pos_sg * 100, color="#d62728", linewidth=1.4,
            label="self-generated")
    ax.plot(positions, uni_per_pos * 100, color="#2ca02c", linewidth=1.0,
            linestyle="--", label="unigram")
    ax.set_xlabel("character position in string (1-indexed)")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Per-position prediction accuracy across the 90 predicted characters")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_generated_samples(model_tf, model_sg, indices: np.ndarray,
                           strings: list, out_path: str,
                           seed_idx: int = 0) -> None:
    """Render a sample rollout from each variant against the ground truth."""
    real_string = strings[seed_idx]
    real_indices = indices[seed_idx:seed_idx + 1]
    rollout_tf = make_negatives_self_generated(model_tf, real_indices,
                                               temperature=0.0)
    rollout_sg = make_negatives_self_generated(model_sg, real_indices,
                                               temperature=0.0)
    text_tf = "".join(ALPHABET[i] for i in rollout_tf[0])
    text_sg = "".join(ALPHABET[i] for i in rollout_sg[0])

    fig = plt.figure(figsize=(11.0, 3.6), dpi=120)
    ax = fig.add_subplot(1, 1, 1)
    ax.axis("off")
    txt = (
        f"seed (10 real chars):  '{real_string[:WINDOW]}'\n\n"
        f"GROUND TRUTH (chars 1-100):\n  '{real_string}'\n\n"
        f"TEACHER-FORCING rollout  (last 90 chars argmax-generated):\n  '{text_tf}'\n\n"
        f"SELF-GENERATED rollout   (last 90 chars argmax-generated):\n  '{text_sg}'\n"
    )
    ax.text(0.0, 1.0, txt,
            ha="left", va="top", family="monospace", fontsize=8.6,
            transform=ax.transAxes)
    fig.suptitle("Sample autoregressive rollouts (one string)")
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_goodness_curves(hist_tf: dict, hist_sg: dict, out_path: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.5), dpi=120, sharey=True)
    n_layers = len(hist_tf["loss_per_layer"])
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5"]
    for ax, hist, title in [(axes[0], hist_tf, "teacher-forcing"),
                             (axes[1], hist_sg, "self-generated")]:
        epochs = hist["epoch"]
        for L in range(n_layers):
            ax.plot(epochs, hist["g_pos_per_layer"][L],
                    color=colors_pos[L % len(colors_pos)], linewidth=1.5,
                    label=f"L{L} pos")
            ax.plot(epochs, hist["g_neg_per_layer"][L],
                    color=colors_neg[L % len(colors_neg)], linewidth=1.5,
                    linestyle="--", label=f"L{L} neg")
        ax.axhline(2.0, color="black", linewidth=0.7, linestyle=":")
        ax.set_xlabel("epoch")
        ax.set_title(f"goodness — {title}")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
    axes[0].set_ylabel(r"goodness  $\langle h^2 \rangle$")
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---------------------------------------------------------------------------
# Entry
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-tf", type=str, default="model_tf.npz",
                   help="Path to .npz for the teacher-forcing run.")
    p.add_argument("--model-sg", type=str, default="model_sg.npz",
                   help="Path to .npz for the self-generated run.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--layer-sizes", type=str, default="300,500,500,500")
    p.add_argument("--eval-every", type=int, default=2)
    p.add_argument("--rollout-every", type=int, default=1)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("Loading Aesop corpus...")
    strings, indices = load_aesop_strings()
    print(f"  {indices.shape[0]} strings of {indices.shape[1]} chars")

    # Train (or load) both variants.
    model_tf, hist_tf, _ = train_or_load("teacher_forcing", indices, args,
                                         args.model_tf)
    model_sg, hist_sg, _ = train_or_load("self_generated", indices, args,
                                         args.model_sg)

    # Baselines.
    print("\nComputing baselines...")
    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    rand_acc, _ = random_fixed_hidden_baseline(
        indices, n_hidden=layer_sizes[1],
        n_layers=len(layer_sizes) - 1, seed=args.seed)
    uni_acc, uni_per_pos = unigram_baseline(indices)
    print(f"  random fixed hidden: {rand_acc*100:.2f}%")
    print(f"  unigram            : {uni_acc*100:.2f}%")
    print(f"  teacher-forcing FF : {hist_tf['test_acc'][-1]*100:.2f}%")
    print(f"  self-generated FF  : {hist_sg['test_acc'][-1]*100:.2f}%")

    # Plots.
    print("\nGenerating visualisations...")
    plot_accuracy_curves(hist_tf, hist_sg, rand_acc, uni_acc,
                         os.path.join(args.outdir, "per_char_accuracy_curves.png"))
    plot_per_position_accuracy(model_tf, model_sg, indices, uni_per_pos,
                               os.path.join(args.outdir, "per_position_accuracy.png"))
    plot_generated_samples(model_tf, model_sg, indices, strings,
                           os.path.join(args.outdir, "generated_samples.png"))
    plot_goodness_curves(hist_tf, hist_sg,
                         os.path.join(args.outdir, "goodness_layers.png"))


if __name__ == "__main__":
    main()
