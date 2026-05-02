"""
Static visualizations for the fast-weights-rehearsal protocol.

Outputs (in `viz/`):
  training_curves.png         Bit/pattern accuracy on A and B across the
                              4 phases (init / learn-A / learn-B /
                              rehearse-subset / test). The headline plot.
  weight_decomposition.png    Heatmaps of W_slow vs W_fast at the end of
                              each phase, with W_fast norm shrinking
                              between phases as it decays.
  recovery_bars.png           Per-pair bit accuracy on A: rehearsed (gold)
                              vs unrehearsed (gray), at end of phase 2
                              (post-B) and phase 3 (post-rehearsal).
  recall_distribution.png     Histogram of bit-accuracy across A pairs at
                              each phase, splitting rehearsed vs not.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

from fast_weights_rehearsal import (
    FastWeightsAssociator, run_protocol, recall_per_pair,
)


COLOR_A = "#1f77b4"
COLOR_B = "#d62728"
COLOR_REHEARSED = "#d4a017"
COLOR_UNREHEARSED = "#888888"


def _phase_boundaries(phases):
    """Return list of (label, idx_start, idx_end) inclusive, in plot order."""
    out = []
    cur_label = phases[0]
    cur_start = 0
    for i, p in enumerate(phases[1:], start=1):
        if p != cur_label:
            out.append((cur_label, cur_start, i - 1))
            cur_label = p
            cur_start = i
    out.append((cur_label, cur_start, len(phases) - 1))
    return out


PHASE_FACECOLOR = {"init": "#eeeeee", "learn_A": "#cfe7ff",
                   "learn_B": "#ffd6cf", "rehearse": "#fff4c2",
                   "test": "#e6e6e6"}
PHASE_LABEL = {"init": "init", "learn_A": "Phase 1: learn A",
               "learn_B": "Phase 2: learn B",
               "rehearse": "Phase 3: rehearse subset of A",
               "test": "Phase 4: test"}


def plot_training_curves(history: dict, summary: dict, out_path: str) -> None:
    """4-phase timeline: bit + pattern accuracy on A and B."""
    phases = history["phase"]
    n = len(phases)
    xs = np.arange(n)
    bit_A = np.array(history["acc_bit_A"]) * 100
    bit_B = np.array(history["acc_bit_B"]) * 100
    pat_A = np.array(history["acc_pattern_A"]) * 100
    pat_B = np.array(history["acc_pattern_B"]) * 100
    fast_norm = np.array(history["fast_norm"])
    slow_norm = np.array(history["slow_norm"])

    bands = _phase_boundaries(phases)

    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), dpi=120, sharex=True)

    # ---- panel 1: bit accuracy ----
    ax = axes[0]
    for label, i0, i1 in bands:
        ax.axvspan(i0 - 0.5, i1 + 0.5,
                   facecolor=PHASE_FACECOLOR.get(label, "#eeeeee"),
                   alpha=0.55, zorder=0)
    ax.plot(xs, bit_A, color=COLOR_A, linewidth=1.8,
            marker="o", markersize=3, label="recall A (bit)")
    ax.plot(xs, bit_B, color=COLOR_B, linewidth=1.8,
            marker="s", markersize=3, label="recall B (bit)")
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.8,
               label="chance (50%)")
    ax.set_ylim(35, 105)
    ax.set_ylabel("bit accuracy (%)")
    ax.set_title("Bit-wise recall on A and B across the 4-phase protocol",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)
    # phase labels at top
    for label, i0, i1 in bands:
        if label == "init":
            continue
        ax.text((i0 + i1) / 2, 102, PHASE_LABEL.get(label, label),
                ha="center", va="bottom", fontsize=8, color="black")

    # ---- panel 2: pattern accuracy ----
    ax = axes[1]
    for label, i0, i1 in bands:
        ax.axvspan(i0 - 0.5, i1 + 0.5,
                   facecolor=PHASE_FACECOLOR.get(label, "#eeeeee"),
                   alpha=0.55, zorder=0)
    ax.plot(xs, pat_A, color=COLOR_A, linewidth=1.8,
            marker="o", markersize=3, label="recall A (pattern)")
    ax.plot(xs, pat_B, color=COLOR_B, linewidth=1.8,
            marker="s", markersize=3, label="recall B (pattern)")
    ax.set_ylim(-5, 105)
    ax.set_ylabel("pattern accuracy (%)")
    ax.set_title("Pattern-wise recall (all bits correct)", fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", framealpha=0.95, fontsize=9)

    # ---- panel 3: weight norms ----
    ax = axes[2]
    for label, i0, i1 in bands:
        ax.axvspan(i0 - 0.5, i1 + 0.5,
                   facecolor=PHASE_FACECOLOR.get(label, "#eeeeee"),
                   alpha=0.55, zorder=0)
    ax.plot(xs, slow_norm, color="#2ca02c", linewidth=1.8,
            marker="o", markersize=3, label="||W_slow||")
    ax.plot(xs, fast_norm, color="#9467bd", linewidth=1.8,
            marker="s", markersize=3, label="||W_fast||")
    ax.set_xlabel("phase tick (one tick = one full sweep through current phase's data)")
    ax.set_ylabel("Frobenius norm")
    ax.set_title("Slow vs fast weight norms (fast decays between phases)",
                 fontsize=11)
    ax.grid(alpha=0.3)
    ax.legend(loc="upper left", framealpha=0.95, fontsize=9)

    fig.suptitle(
        f"fast-weights rehearsal (Hinton & Plaut 1987)  —  "
        f"seed={summary['seed']}, dim={summary['dim']}, "
        f"n_pairs={summary['n_pairs']}, rehearse {summary['n_rehearse']}\n"
        f"slow_lr={summary['slow_lr']}, fast_lr={summary['fast_lr']}, "
        f"fast_decay={summary['fast_decay']}",
        fontsize=10.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_weight_decomposition(model_after_A: FastWeightsAssociator,
                              model_after_B: FastWeightsAssociator,
                              model_after_R: FastWeightsAssociator,
                              out_path: str) -> None:
    """3 columns x 2 rows = (after-A, after-B, after-rehearse) x (slow, fast)."""
    fig, axes = plt.subplots(2, 3, figsize=(11, 6.5), dpi=130)
    cols = [(model_after_A, "after Phase 1\n(learn A)"),
            (model_after_B, "after Phase 2\n(learn B)"),
            (model_after_R, "after Phase 3\n(rehearse)")]
    rows = [("W_slow", lambda m: m.W_slow),
            ("W_fast", lambda m: m.W_fast)]

    # consistent color scale per row
    for r_idx, (rname, getter) in enumerate(rows):
        amax = max(np.abs(getter(c[0])).max() for c in cols)
        amax = max(amax, 1e-6)
        for c_idx, (m, title) in enumerate(cols):
            ax = axes[r_idx, c_idx]
            W = getter(m)
            im = ax.imshow(W, cmap="RdBu_r", vmin=-amax, vmax=amax,
                           aspect="equal", interpolation="nearest")
            ax.set_xticks([]); ax.set_yticks([])
            if r_idx == 0:
                ax.set_title(title, fontsize=10)
            if c_idx == 0:
                ax.set_ylabel(rname, fontsize=11, fontweight="bold")
            ax.text(0.02, 0.98, f"||·||={np.linalg.norm(W):.2f}",
                    transform=ax.transAxes, fontsize=8, va="top",
                    color="black",
                    bbox=dict(boxstyle="round,pad=0.2",
                              facecolor="white", edgecolor="none",
                              alpha=0.85))
        fig.colorbar(im, ax=axes[r_idx, :], fraction=0.025, pad=0.02)

    fig.suptitle(
        "Slow plastic vs fast elastic-decaying weights at the end of each phase\n"
        "(W_fast decays multiplicatively each presentation; the rehearsal phase "
        "rebuilds it on the subset)",
        fontsize=10.5, y=0.99)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recovery_bars(per_pair_after_B: np.ndarray,
                       per_pair_after_R: np.ndarray,
                       rehearsed_mask: np.ndarray,
                       out_path: str) -> None:
    """Per-pair bit accuracy on A: post-B vs post-rehearsal, split by rehearsed."""
    n = len(per_pair_after_B)
    xs = np.arange(n)
    width = 0.40

    fig, ax = plt.subplots(figsize=(10, 4.6), dpi=140)
    # post-B bars (lighter, behind)
    ax.bar(xs - width / 2, per_pair_after_B * 100, width=width,
           color="#bbbbbb", edgecolor="black", linewidth=0.4,
           label="after Phase 2 (post-B)")
    colors_R = [COLOR_REHEARSED if rehearsed_mask[i] else COLOR_UNREHEARSED
                for i in range(n)]
    ax.bar(xs + width / 2, per_pair_after_R * 100, width=width,
           color=colors_R, edgecolor="black", linewidth=0.4,
           label="after Phase 3 (post-rehearsal)")

    ax.axhline(50, linestyle=":", color="gray", linewidth=0.8)
    ax.axhline(100, linestyle="-", color="black", linewidth=0.4, alpha=0.4)

    ax.set_xticks(xs)
    ax.set_xticklabels([str(i) for i in xs], fontsize=8)
    ax.set_xlabel("A-pair index (gold = rehearsed)")
    ax.set_ylabel("bit accuracy on this A pair (%)")
    ax.set_ylim(40, 108)
    ax.grid(axis="y", alpha=0.3)

    # custom legend
    handles = [
        Patch(facecolor="#bbbbbb", edgecolor="black",
              label="after Phase 2 (post-B)"),
        Patch(facecolor=COLOR_REHEARSED, edgecolor="black",
              label="after Phase 3 — rehearsed pair"),
        Patch(facecolor=COLOR_UNREHEARSED, edgecolor="black",
              label="after Phase 3 — unrehearsed pair"),
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=8.5,
              framealpha=0.95)
    n_reh = int(rehearsed_mask.sum())
    ax.set_title(
        f"Per-pair recovery after rehearsing {n_reh} of {n} A pairs:\n"
        "rehearsed pairs jump to 100%; unrehearsed pairs barely move "
        "(random patterns share no structure)",
        fontsize=10.5)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_recall_distribution(per_pair_phases: dict[str, np.ndarray],
                             rehearsed_mask: np.ndarray,
                             out_path: str) -> None:
    """Histograms of per-pair bit accuracy at each phase, split by group."""
    phase_names = list(per_pair_phases.keys())
    n_phases = len(phase_names)
    fig, axes = plt.subplots(1, n_phases, figsize=(3.5 * n_phases, 3.4),
                             dpi=130, sharey=True)
    if n_phases == 1:
        axes = [axes]
    bins = np.linspace(0.4, 1.001, 21) * 100

    for ax, name in zip(axes, phase_names):
        v = per_pair_phases[name] * 100
        v_reh = v[rehearsed_mask]
        v_un = v[~rehearsed_mask]
        ax.hist(v_un, bins=bins, color=COLOR_UNREHEARSED,
                edgecolor="black", linewidth=0.3, alpha=0.85,
                label=f"unrehearsed ({len(v_un)})")
        ax.hist(v_reh, bins=bins, color=COLOR_REHEARSED,
                edgecolor="black", linewidth=0.3, alpha=0.95,
                label=f"rehearsed ({len(v_reh)})")
        ax.set_title(name, fontsize=10)
        ax.set_xlabel("bit acc on A pair (%)")
        ax.grid(axis="y", alpha=0.3)
        ax.legend(fontsize=8, loc="upper left", framealpha=0.95)
    axes[0].set_ylabel("# of A pairs")
    fig.suptitle("Distribution of per-pair bit accuracy on A across phases",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--n-rehearse", type=int, default=None)
    p.add_argument("--n-a-sweeps", type=int, default=30)
    p.add_argument("--n-b-sweeps", type=int, default=30)
    p.add_argument("--n-rehearse-sweeps", type=int, default=5)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"Running 4-phase protocol seed={args.seed}, dim={args.dim}, "
          f"n_pairs={args.n_pairs} ...")
    out = run_protocol(seed=args.seed,
                       dim=args.dim, n_pairs=args.n_pairs,
                       n_rehearse=args.n_rehearse,
                       n_a_sweeps=args.n_a_sweeps,
                       n_b_sweeps=args.n_b_sweeps,
                       n_rehearse_sweeps=args.n_rehearse_sweeps,
                       snapshot_every=1, verbose=False)

    summary = out["summary"]
    history = out["history"]
    print(f"  recall_A bit acc (post-A / post-B / post-R): "
          f"{summary['after_A_bit_A']*100:.1f}% / "
          f"{summary['after_B_bit_A']*100:.1f}% / "
          f"{summary['after_R_bit_A']*100:.1f}%")
    print(f"  rehearsed-pair recovery: "
          f"{summary['rehearsed_pair_recovery_bits']*100:+.2f} pp")
    print(f"  unrehearsed-pair recovery: "
          f"{summary['unrehearsed_pair_recovery_bits']*100:+.2f} pp")

    plot_training_curves(history, summary,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_weight_decomposition(out["model_after_A"],
                              out["model_after_B"],
                              out["model_after_R"],
                              os.path.join(args.outdir,
                                           "weight_decomposition.png"))

    rehearsed_mask = out["data"]["rehearsed_mask"]
    per_pair_after_B = out["data"]["per_pair_A_after_B"]
    per_pair_after_R = out["data"]["per_pair_A_after_R"]

    plot_recovery_bars(per_pair_after_B, per_pair_after_R, rehearsed_mask,
                       os.path.join(args.outdir, "recovery_bars.png"))

    # Distribution: also include "after A" snapshot for context
    per_pair_after_A = recall_per_pair(out["model_after_A"], out["data"]["A"])
    plot_recall_distribution({"after Phase 1\n(learn A)": per_pair_after_A,
                              "after Phase 2\n(learn B)": per_pair_after_B,
                              "after Phase 3\n(rehearsal)": per_pair_after_R},
                             rehearsed_mask,
                             os.path.join(args.outdir,
                                          "recall_distribution.png"))


if __name__ == "__main__":
    main()
