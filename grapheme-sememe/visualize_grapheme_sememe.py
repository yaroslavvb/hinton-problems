"""
Static visualizations for the trained grapheme-sememe net.

Outputs (in `viz/`):
  training_curves.png        - 4-stage timeline (train, lesion, relearn) with
                               trained-18 vs held-out-2 accuracy tracked
                               separately
  weights.png                - W1 and W2 as a heatmap; lesioned weights are
                               drawn as black squares
  reconstructions.png        - per-pattern target vs. predicted sememe bits at
                               the end of stage 1, end of stage 2, end of
                               stage 4
  spontaneous_recovery.png   - bar chart: bit accuracy on held-out 2 across
                               the 4 stages
"""

from __future__ import annotations
import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from grapheme_sememe import (
    GraphemeSememeMLP, run_protocol, accuracy_bitwise, accuracy_pattern,
)


COLOR_TRAIN = "#1f77b4"
COLOR_HELD = "#d62728"
COLOR_LESION = "#ff7f0e"


def plot_training_curves(history: dict, summary: dict, out_path: str):
    """4-stage timeline of bit accuracy on the trained-18 and held-out-2 sets."""
    epochs = np.array(history["epoch"], dtype=float)
    phases = np.array(history["phase"])
    acc_train = np.array(history["acc_bit_trained"])
    acc_held = np.array(history["acc_bit_held_out"])
    pat_train = np.array(history["acc_pattern_trained"])
    pat_held = np.array(history["acc_pattern_held_out"])

    # Find phase boundaries
    train_end = (phases == "train").sum() - 1
    lesion_idx = np.where(phases == "lesion")[0]
    relearn_start = (phases == "lesion").sum() + (phases == "train").sum()

    fig, axes = plt.subplots(2, 1, figsize=(10, 6.5), dpi=120, sharex=True)

    ax = axes[0]
    ax.plot(epochs, acc_train * 100, color=COLOR_TRAIN, linewidth=1.5,
            label="trained 18 (bit acc)")
    ax.plot(epochs, acc_held * 100, color=COLOR_HELD, linewidth=1.5,
            label="held-out 2 (bit acc)")
    if len(lesion_idx):
        ax.axvline(epochs[lesion_idx[0]], color=COLOR_LESION,
                   linewidth=1.2, linestyle="--",
                   label=f"lesion ({summary['lesion_fraction']*100:.0f}%)")
    ax.set_ylabel("bit accuracy (%)")
    ax.set_ylim(40, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title("Bit accuracy across the 4-stage protocol")

    ax = axes[1]
    ax.plot(epochs, pat_train * 100, color=COLOR_TRAIN, linewidth=1.5,
            label="trained 18 (pattern acc)")
    ax.plot(epochs, pat_held * 100, color=COLOR_HELD, linewidth=1.5,
            label="held-out 2 (pattern acc)")
    if len(lesion_idx):
        ax.axvline(epochs[lesion_idx[0]], color=COLOR_LESION,
                   linewidth=1.2, linestyle="--",
                   label="lesion")
    ax.set_xlabel("training cycle")
    ax.set_ylabel("pattern accuracy (%)")
    ax.set_ylim(-5, 105)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)
    ax.set_title("Pattern accuracy (all 30 sememe bits correct)")

    fig.suptitle(
        f"grapheme-sememe (seed={summary['seed']})  —  "
        f"lesion {summary['lesion_fraction']*100:.0f}% of W1+W2  —  "
        f"relearn on 18 of {summary['n_words']} for "
        f"{summary['n_relearn_cycles']} cycles",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weights(model: GraphemeSememeMLP,
                 mask: dict[str, np.ndarray],
                 out_path: str):
    """Visualize W1 and W2; lesioned (zeroed) entries as black squares."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 5), dpi=130,
                             gridspec_kw={"width_ratios": [3, 3]})

    for ax, (name, W, M) in zip(axes, [("W1 (hidden ← grapheme)",
                                       model.W1, mask["W1"]),
                                       ("W2 (sememe ← hidden)",
                                       model.W2, mask["W2"])]):
        amax = max(np.abs(W).max(), 1e-3)
        im = ax.imshow(W, cmap="RdBu_r", vmin=-amax, vmax=amax,
                       aspect="auto", interpolation="nearest")
        # overlay black on lesioned entries
        lesioned = (M == 0)
        ax.imshow(np.where(lesioned, 1, np.nan),
                  cmap="gray_r", vmin=0, vmax=1, aspect="auto",
                  interpolation="nearest", alpha=0.85)
        ax.set_title(name + f"  shape={W.shape}", fontsize=10)
        fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)

    fig.suptitle("Weights after the 4-stage protocol "
                 "(black = lesioned entries forced to zero)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_reconstructions(model_at_stages: dict,
                         X: np.ndarray, Y: np.ndarray,
                         held_out: list[int],
                         out_path: str):
    """For each held-out word, show target sememe vs. prediction at 3 stages."""
    fig, axes = plt.subplots(len(held_out), 1, figsize=(10, 2.5 * len(held_out)),
                             dpi=130, squeeze=False)
    bit_xs = np.arange(Y.shape[1])

    for r, idx in enumerate(held_out):
        ax = axes[r, 0]
        target = Y[idx]
        ax.bar(bit_xs - 0.30, target, width=0.18,
               color="#444444", label="target")
        offset = -0.10
        for stage, model in model_at_stages.items():
            pred = model.predict(X[idx:idx+1])[0]
            ax.bar(bit_xs + offset, pred, width=0.18, label=stage,
                   alpha=0.85)
            offset += 0.20
        ax.set_xlim(-0.6, Y.shape[1] - 0.4)
        ax.set_ylim(-0.05, 1.10)
        ax.set_ylabel(f"held-out {idx}")
        ax.grid(axis="y", alpha=0.3)
        if r == 0:
            ax.legend(loc="upper right", ncol=4, fontsize=8, framealpha=0.9)
        if r == len(held_out) - 1:
            ax.set_xlabel("sememe bit index")

    fig.suptitle("Held-out sememe predictions across stages "
                 "(bars: target = gray; predictions = colored)",
                 fontsize=10)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_spontaneous_recovery(summary: dict, out_path: str):
    """Bar chart of bit accuracy on the held-out 2 across the 4 stages."""
    stages = ["pre-train\n(random init)",
              "stage 1\npost-train",
              "stage 2\npost-lesion",
              "stage 4\npost-relearn"]
    # pre-train is chance for random sigmoid output ~ 50%; we don't actually
    # measure it (depends on init). Use a simple expected baseline.
    chance = 50.0
    values = [chance,
              summary["pre_lesion_bit_held_out"] * 100,
              summary["post_lesion_bit_held_out"] * 100,
              summary["post_relearn_bit_held_out"] * 100]
    colors = ["#9e9e9e", COLOR_TRAIN, COLOR_LESION, COLOR_HELD]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=140)
    bars = ax.bar(stages, values, color=colors, edgecolor="black", linewidth=0.6)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + 1.5, f"{v:.1f}%",
                ha="center", va="bottom", fontsize=10)

    ax.axhline(chance, linestyle=":", color="gray", linewidth=0.8)
    ax.set_ylabel("bit accuracy on held-out 2 (%)")
    ax.set_ylim(0, 110)
    ax.grid(axis="y", alpha=0.3)
    delta = (summary["post_relearn_bit_held_out"]
             - summary["post_lesion_bit_held_out"]) * 100
    sign = "+" if delta >= 0 else ""
    ax.set_title(f"Spontaneous recovery: held-out 2 bits  "
                 f"({sign}{delta:.1f} pp from post-lesion to post-relearn)",
                 fontsize=11)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lesion-fraction", type=float, default=0.5)
    p.add_argument("--n-train-cycles", type=int, default=1500)
    p.add_argument("--n-relearn-cycles", type=int, default=50)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Running 4-stage protocol seed={args.seed}, "
          f"lesion={args.lesion_fraction}, "
          f"relearn={args.n_relearn_cycles} cycles ...")

    # First pass: capture model snapshots at end of each stage.
    # Cleanest way: run protocol once for the timeline (gets history), and
    # once again with a callback that snapshots after each stage.
    out = run_protocol(seed=args.seed,
                       lesion_fraction=args.lesion_fraction,
                       n_train_cycles=args.n_train_cycles,
                       n_relearn_cycles=args.n_relearn_cycles,
                       history_every=1,
                       verbose=False)
    summary = out["summary"]
    history = out["history"]
    model = out["model"]
    mask = out["mask"]
    X = out["data"]["X"]
    Y = out["data"]["Y"]
    held_out = out["data"]["held_out_idx"]

    print(f"  pre-lesion bit acc (held-out 2): "
          f"{summary['pre_lesion_bit_held_out']*100:.1f}%")
    print(f"  post-lesion bit acc (held-out 2): "
          f"{summary['post_lesion_bit_held_out']*100:.1f}%")
    print(f"  post-relearn bit acc (held-out 2): "
          f"{summary['post_relearn_bit_held_out']*100:.1f}%")
    print(f"  spontaneous recovery: "
          f"{summary['spontaneous_recovery_bits']*100:+.2f} pp")

    plot_training_curves(history, summary,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_weights(model, mask,
                 os.path.join(args.outdir, "weights.png"))
    plot_spontaneous_recovery(summary,
                              os.path.join(args.outdir,
                                           "spontaneous_recovery.png"))

    # Second pass: re-run to capture model state at end of each stage.
    # We do a simpler version that re-runs train -> snapshot -> lesion ->
    # snapshot -> relearn -> snapshot. To keep this consistent with the
    # primary run, we use the same RNG seeds (lesion seed = seed+1).
    import copy
    from grapheme_sememe import (
        generate_mapping, train, lesion, relearn_subset,
    )
    Xs, Ys = generate_mapping(seed=args.seed)
    held_idx = list(range(len(Xs) - 2, len(Xs)))
    train_idx = [i for i in range(len(Xs)) if i not in held_idx]

    m = GraphemeSememeMLP(seed=args.seed, init_scale=0.5)
    train(m, Xs, Ys, n_cycles=args.n_train_cycles, lr=0.3,
          momentum=0.5, weight_decay=1e-3, history_every=10**6)
    m_after_train = copy.deepcopy(m)
    mask2 = lesion(m, fraction=args.lesion_fraction, seed=args.seed + 1)
    m_after_lesion = copy.deepcopy(m)
    relearn_subset(m, Xs, Ys, indices=train_idx,
                   n_cycles=args.n_relearn_cycles, lr=0.3,
                   momentum=0.5, weight_decay=1e-3,
                   weight_mask=mask2, history_eval=None)
    m_after_relearn = copy.deepcopy(m)

    plot_reconstructions({"post-train": m_after_train,
                          "post-lesion": m_after_lesion,
                          "post-relearn": m_after_relearn},
                         Xs, Ys, held_idx,
                         os.path.join(args.outdir, "reconstructions.png"))


if __name__ == "__main__":
    main()
