"""
Render an animated GIF showing the FF model learning to predict Aesop
characters under both teacher-forcing and self-generated negatives.

Each frame combines:
  Top:    per-char accuracy curves for both variants over training, with the
          unigram + random-fixed baselines marked.
  Middle: per-layer goodness (pos solid, neg dashed) for both variants.
  Bottom: a sample autoregressive rollout from each current model on a
          fixed seed.

Two training runs are interleaved: at each "snapshot" epoch we record a
snapshot of teacher-forcing's history+model and self-generated's
history+model. To keep the rendering tractable we only re-train on demand
(no caching here -- pass --use-cache to load from disk if available).

Usage:
    python3 make_ff_aesop_sequences_gif.py --epochs 30 --fps 6
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from ff_aesop_sequences import (
    ALPHABET, N_SYMBOLS, WINDOW,
    load_aesop_strings, build_ff_seq_model, train, TrainConfig, FFModel,
    random_fixed_hidden_baseline, unigram_baseline,
    make_negatives_self_generated,
)


def render_frame(hist_tf: dict, hist_sg: dict,
                 model_tf: FFModel, model_sg: FFModel,
                 demo_indices: np.ndarray, demo_string: str,
                 epoch: int, total_epochs: int,
                 rand_acc: float, uni_acc: float) -> Image.Image:
    fig = plt.figure(figsize=(10.0, 6.6), dpi=100)
    gs = fig.add_gridspec(3, 2, height_ratios=[1.2, 1.0, 0.9],
                          hspace=0.65, wspace=0.30)

    # ---- top: accuracy curves ----
    ax = fig.add_subplot(gs[0, :])
    epochs_tf = hist_tf["epoch"]
    epochs_sg = hist_sg["epoch"]
    if epochs_tf:
        ax.plot(epochs_tf, np.array(hist_tf["test_acc"]) * 100,
                color="#1f77b4", linewidth=1.8, label="teacher-forcing")
    if epochs_sg:
        ax.plot(epochs_sg, np.array(hist_sg["test_acc"]) * 100,
                color="#d62728", linewidth=1.8, label="self-generated")
    ax.axhline(uni_acc * 100, color="#2ca02c", linewidth=1.0, linestyle="--",
               label=f"unigram ({uni_acc*100:.1f}%)")
    ax.axhline(rand_acc * 100, color="#7f7f7f", linewidth=1.0, linestyle=":",
               label=f"random fixed ({rand_acc*100:.1f}%)")
    ax.set_xlim(0, total_epochs)
    ax.set_ylim(0, 60)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("per-char accuracy (%)", fontsize=9)
    ax.set_title("Accuracy on the 248 x 90 next-char predictions",
                 fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right", ncol=2)

    # ---- middle: per-layer goodness ----
    colors_pos = ["#1f77b4", "#2ca02c", "#9467bd"]
    colors_neg = ["#aec7e8", "#98df8a", "#c5b0d5"]
    for col_idx, (hist, title) in enumerate(
            [(hist_tf, "teacher-forcing"), (hist_sg, "self-generated")]):
        ax = fig.add_subplot(gs[1, col_idx])
        if hist["epoch"]:
            n_layers = len(hist["g_pos_per_layer"])
            for L in range(n_layers):
                ax.plot(hist["epoch"], hist["g_pos_per_layer"][L],
                        color=colors_pos[L % len(colors_pos)], linewidth=1.4,
                        label=f"L{L} pos")
                ax.plot(hist["epoch"], hist["g_neg_per_layer"][L],
                        color=colors_neg[L % len(colors_neg)], linewidth=1.4,
                        linestyle="--", label=f"L{L} neg")
            ax.axhline(2.0, color="black", linewidth=0.6, linestyle=":")
        ax.set_xlim(0, total_epochs)
        ax.set_xlabel("epoch", fontsize=9)
        if col_idx == 0:
            ax.set_ylabel(r"goodness $\langle h^2 \rangle$", fontsize=9)
        ax.set_title(f"goodness ({title})", fontsize=10)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=7, ncol=2, loc="upper right")

    # ---- bottom: text samples ----
    rollout_tf = make_negatives_self_generated(model_tf, demo_indices,
                                               temperature=0.0)
    rollout_sg = make_negatives_self_generated(model_sg, demo_indices,
                                               temperature=0.0)
    text_tf = "".join(ALPHABET[i] for i in rollout_tf[0])
    text_sg = "".join(ALPHABET[i] for i in rollout_sg[0])

    ax = fig.add_subplot(gs[2, :])
    ax.axis("off")
    txt = (
        f"seed:  '{demo_string[:WINDOW]}'\n"
        f"truth: '{demo_string}'\n"
        f"TF:    '{text_tf}'\n"
        f"SG:    '{text_sg}'\n"
    )
    ax.text(0.0, 1.0, txt, ha="left", va="top",
            family="monospace", fontsize=7.5, transform=ax.transAxes)

    fig.suptitle(f"FF on Aesop's Fables  —  epoch {epoch}/{total_epochs}",
                 fontsize=12, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.Palette.ADAPTIVE,
                                    colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--steps-per-epoch", type=int, default=120,
                   help="Reduced from 200 to keep GIF render time short.")
    p.add_argument("--layer-sizes", type=str, default="300,400,400,400")
    p.add_argument("--rollout-every", type=int, default=1)
    p.add_argument("--rollout-temperature", type=float, default=1.0)
    p.add_argument("--out", type=str, default="ff_aesop_sequences.gif")
    p.add_argument("--hold-final", type=int, default=10)
    p.add_argument("--max-size-kb", type=int, default=3072)
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))

    print("Loading Aesop...")
    strings, indices = load_aesop_strings()
    rng = np.random.default_rng(args.seed)
    demo_idx = int(rng.integers(0, indices.shape[0]))
    demo_indices = indices[demo_idx:demo_idx + 1]
    demo_string = strings[demo_idx]
    print(f"  demo string idx={demo_idx}: '{demo_string[:60]}...'")

    print("Computing baselines...")
    rand_acc, _ = random_fixed_hidden_baseline(
        indices, n_hidden=layer_sizes[1],
        n_layers=len(layer_sizes) - 1, seed=args.seed)
    uni_acc, _ = unigram_baseline(indices)

    # Two parallel runs -- we collect snapshots from each at every epoch.
    # We train them sequentially (interleaved is more natural with
    # snapshot_callback per run).
    cfg_tf = TrainConfig(n_epochs=args.epochs,
                         batch_size=128,
                         steps_per_epoch=args.steps_per_epoch,
                         lr=args.lr,
                         threshold=2.0,
                         layer_sizes=layer_sizes,
                         seed=args.seed,
                         negatives="teacher_forcing",
                         eval_every=1)
    cfg_sg = TrainConfig(n_epochs=args.epochs,
                         batch_size=128,
                         steps_per_epoch=args.steps_per_epoch,
                         lr=args.lr,
                         threshold=2.0,
                         layer_sizes=layer_sizes,
                         seed=args.seed,
                         negatives="self_generated",
                         eval_every=1,
                         rollout_every=args.rollout_every,
                         rollout_temperature=args.rollout_temperature)

    print("Training teacher-forcing variant (collecting snapshots)...")
    model_tf = build_ff_seq_model(seed=args.seed, threshold=2.0,
                                  layer_sizes=layer_sizes)
    snaps_tf = []   # list of (epoch_idx_1based, copy of model, copy of history)

    def cb_tf(epoch_idx: int, m: FFModel, hist: dict) -> None:
        # Deep-copy a model: in numpy land that's just the weight arrays.
        snap_model = build_ff_seq_model(seed=args.seed, threshold=2.0,
                                        layer_sizes=layer_sizes)
        for sl, ml in zip(snap_model.layers, m.layers):
            sl.W = ml.W.copy()
            sl.b = ml.b.copy()
        snap_hist = {k: (list(v) if isinstance(v, list) and v and isinstance(v[0], (int, float))
                         else [list(row) for row in v] if isinstance(v, list) and v and isinstance(v[0], list)
                         else v)
                     for k, v in hist.items()}
        snaps_tf.append((epoch_idx + 1, snap_model, snap_hist))

    train(model_tf, indices, cfg_tf,
          snapshot_callback=cb_tf, snapshot_every=args.snapshot_every,
          verbose=False)

    print("Training self-generated variant (collecting snapshots)...")
    model_sg = build_ff_seq_model(seed=args.seed, threshold=2.0,
                                  layer_sizes=layer_sizes)
    snaps_sg = []

    def cb_sg(epoch_idx: int, m: FFModel, hist: dict) -> None:
        snap_model = build_ff_seq_model(seed=args.seed, threshold=2.0,
                                        layer_sizes=layer_sizes)
        for sl, ml in zip(snap_model.layers, m.layers):
            sl.W = ml.W.copy()
            sl.b = ml.b.copy()
        snap_hist = {k: (list(v) if isinstance(v, list) and v and isinstance(v[0], (int, float))
                         else [list(row) for row in v] if isinstance(v, list) and v and isinstance(v[0], list)
                         else v)
                     for k, v in hist.items()}
        snaps_sg.append((epoch_idx + 1, snap_model, snap_hist))

    train(model_sg, indices, cfg_sg,
          snapshot_callback=cb_sg, snapshot_every=args.snapshot_every,
          verbose=False)

    # Render frames -- pair up by snapshot index.
    n_frames = min(len(snaps_tf), len(snaps_sg))
    print(f"Rendering {n_frames} frames...")
    frames: list[Image.Image] = []
    for i in range(n_frames):
        ep, m_tf, h_tf = snaps_tf[i]
        _, m_sg, h_sg = snaps_sg[i]
        frame = render_frame(h_tf, h_sg, m_tf, m_sg,
                             demo_indices, demo_string,
                             ep, args.epochs, rand_acc, uni_acc)
        frames.append(frame)
        if (i + 1) % max(1, n_frames // 5) == 0:
            print(f"  frame {i + 1:3d}/{n_frames}")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")
    if size_kb > args.max_size_kb:
        print(f"WARNING: gif size {size_kb:.0f} KB exceeds soft limit "
              f"{args.max_size_kb} KB. Consider lower --fps, fewer frames, "
              f"or smaller --layer-sizes.")


if __name__ == "__main__":
    main()
