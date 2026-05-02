"""
Render an animated GIF of the 4-phase fast-weights-rehearsal protocol.

Layout per frame:
  Top-left   : per-pair bit accuracy on A (rehearsed pairs in gold,
               unrehearsed in gray)
  Top-right  : per-pair bit accuracy on B (red)
  Bottom     : 4-phase timeline of mean recall on A and B; vertical line
               is the current sweep, color bands mark the phases.

Usage:
    python3 make_fast_weights_rehearsal_gif.py
    python3 make_fast_weights_rehearsal_gif.py --seed 0 --fps 8
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fast_weights_rehearsal import (
    build_model, generate_two_sets,
    recall_accuracy, recall_pattern_accuracy, recall_per_pair,
    fast_weight_norm, slow_weight_norm,
)


COLOR_A = "#1f77b4"
COLOR_B = "#d62728"
COLOR_REHEARSED = "#d4a017"
COLOR_UNREHEARSED = "#888888"
PHASE_COLORS = {"learn_A": "#cfe7ff", "learn_B": "#ffd6cf",
                "rehearse": "#fff4c2", "test": "#e6e6e6"}
PHASE_TITLE = {"learn_A": "Phase 1: learn set A",
               "learn_B": "Phase 2: learn set B (interferes with A)",
               "rehearse": "Phase 3: rehearse subset of A "
                           "(fast weights deblur the rest)",
               "test": "Phase 4: test (no further updates)"}


def render_frame(model,
                 A: tuple[np.ndarray, np.ndarray],
                 B: tuple[np.ndarray, np.ndarray],
                 rehearsed_mask: np.ndarray,
                 history: dict,
                 phase: str,
                 sweep_global: int,
                 phase_x_bounds: list[tuple[str, int, int]],
                 ) -> Image.Image:
    fig = plt.figure(figsize=(10, 6.4), dpi=100)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.55, wspace=0.30)

    # ---- top-left: per-pair acc on A ----
    ax_a = fig.add_subplot(gs[0, 0])
    pp_A = recall_per_pair(model, A) * 100
    n = len(pp_A)
    xs = np.arange(n)
    colors = [COLOR_REHEARSED if rehearsed_mask[i] else COLOR_UNREHEARSED
              for i in range(n)]
    ax_a.bar(xs, pp_A, color=colors, edgecolor="black", linewidth=0.3)
    ax_a.axhline(50, color="gray", linestyle=":", linewidth=0.7)
    ax_a.axhline(100, color="black", linestyle="-", linewidth=0.4, alpha=0.4)
    ax_a.set_ylim(0, 110)
    ax_a.set_xticks(xs[::max(1, n // 10)])
    ax_a.set_xlabel("A-pair index")
    ax_a.set_ylabel("bit acc (%)")
    mean_A = float(pp_A.mean())
    ax_a.set_title(f"Set A per-pair recall   "
                   f"(mean = {mean_A:.1f}%, gold = rehearsed)",
                   fontsize=10)

    # ---- top-right: per-pair acc on B ----
    ax_b = fig.add_subplot(gs[0, 1])
    pp_B = recall_per_pair(model, B) * 100
    ax_b.bar(np.arange(len(pp_B)), pp_B,
             color=COLOR_B, alpha=0.85,
             edgecolor="black", linewidth=0.3)
    ax_b.axhline(50, color="gray", linestyle=":", linewidth=0.7)
    ax_b.axhline(100, color="black", linestyle="-", linewidth=0.4, alpha=0.4)
    ax_b.set_ylim(0, 110)
    ax_b.set_xticks(np.arange(len(pp_B))[::max(1, len(pp_B) // 10)])
    ax_b.set_xlabel("B-pair index")
    mean_B = float(pp_B.mean())
    ax_b.set_title(f"Set B per-pair recall   (mean = {mean_B:.1f}%)",
                   fontsize=10)

    # ---- bottom: timeline ----
    ax = fig.add_subplot(gs[1, :])
    if history["sweep_global"]:
        sg = np.array(history["sweep_global"])
        bA = np.array(history["acc_bit_A"]) * 100
        bB = np.array(history["acc_bit_B"]) * 100
        # phase shading
        for label, x0, x1 in phase_x_bounds:
            ax.axvspan(x0 - 0.5, x1 + 0.5,
                       facecolor=PHASE_COLORS.get(label, "#eeeeee"),
                       alpha=0.55, zorder=0)
        ax.plot(sg, bA, color=COLOR_A, linewidth=1.6,
                marker="o", markersize=2.5, label="recall A (bit)")
        ax.plot(sg, bB, color=COLOR_B, linewidth=1.6,
                marker="s", markersize=2.5, label="recall B (bit)")
    ax.axvline(sweep_global, color="black", linewidth=1.0, alpha=0.7)
    ax.axhline(50, color="gray", linestyle=":", linewidth=0.7)
    ax.set_ylim(35, 105)
    ax.set_xlabel("phase tick (one tick = one full sweep through current phase's data)")
    ax.set_ylabel("bit accuracy (%)")
    ax.grid(alpha=0.3)
    ax.legend(loc="lower left", fontsize=9, framealpha=0.95)
    ax.set_title(f"{PHASE_TITLE.get(phase, phase)}   |   "
                 f"||W_slow||={slow_weight_norm(model):.2f}   "
                 f"||W_fast||={fast_weight_norm(model):.2f}",
                 fontsize=10)

    fig.suptitle("Fast weights to deblur old memories "
                 "(Hinton & Plaut 1987)",
                 fontsize=11, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.965))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--dim", type=int, default=50)
    p.add_argument("--n-pairs", type=int, default=20)
    p.add_argument("--n-rehearse", type=int, default=None)
    p.add_argument("--slow-lr", type=float, default=0.1)
    p.add_argument("--fast-lr", type=float, default=0.5)
    p.add_argument("--fast-decay", type=float, default=0.9)
    p.add_argument("--n-a-sweeps", type=int, default=18)
    p.add_argument("--n-b-sweeps", type=int, default=18)
    p.add_argument("--n-rehearse-sweeps", type=int, default=8)
    p.add_argument("--snapshot-every", type=int, default=1,
                   help="render a frame every Nth sweep")
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--hold-final", type=int, default=12,
                   help="repeat the last frame N times")
    p.add_argument("--out", type=str, default="fast_weights_rehearsal.gif")
    args = p.parse_args()

    if args.n_rehearse is None:
        args.n_rehearse = max(2, args.n_pairs // 4)

    A, B = generate_two_sets(n_pairs=args.n_pairs, dim=args.dim,
                             seed=args.seed)
    rehearsed_mask = np.zeros(args.n_pairs, dtype=bool)
    rehearsed_mask[:args.n_rehearse] = True
    subset_idx = list(range(args.n_rehearse))

    model = build_model(dim=args.dim,
                        slow_lr=args.slow_lr, fast_lr=args.fast_lr,
                        fast_decay=args.fast_decay)

    history: dict = {"phase": [], "sweep_global": [],
                     "fast_norm": [], "slow_norm": [],
                     "acc_bit_A": [], "acc_pattern_A": [],
                     "acc_bit_B": [], "acc_pattern_B": []}

    rng = np.random.default_rng(args.seed + 7919)

    frames: list[Image.Image] = []
    phase_x_bounds: list[tuple[str, int, int]] = []
    sweep_global = 0

    def _record(phase_name: str) -> None:
        nonlocal sweep_global
        sweep_global += 1
        history["phase"].append(phase_name)
        history["sweep_global"].append(sweep_global)
        history["fast_norm"].append(fast_weight_norm(model))
        history["slow_norm"].append(slow_weight_norm(model))
        history["acc_bit_A"].append(recall_accuracy(model, A))
        history["acc_pattern_A"].append(recall_pattern_accuracy(model, A))
        history["acc_bit_B"].append(recall_accuracy(model, B))
        history["acc_pattern_B"].append(recall_pattern_accuracy(model, B))

    # Initial frame (no training yet)
    _record("init")
    frames.append(render_frame(
        model, A, B, rehearsed_mask, history, "learn_A",
        sweep_global, phase_x_bounds))

    def _train_phase(phase_name: str, data, idx_subset, n_sweeps: int):
        nonlocal sweep_global
        x_start = sweep_global + 1
        Xd, Yd = data
        if idx_subset is not None:
            Xd, Yd = Xd[idx_subset], Yd[idx_subset]
        order = np.arange(len(Xd))
        for sweep in range(n_sweeps):
            rng.shuffle(order)
            for i in order:
                model.step(Xd[i], Yd[i])
            _record(phase_name)
            if (sweep % args.snapshot_every == 0) or (sweep == n_sweeps - 1):
                frames.append(render_frame(
                    model, A, B, rehearsed_mask, history,
                    phase_name, sweep_global, phase_x_bounds))
        x_end = sweep_global
        phase_x_bounds.append((phase_name, x_start, x_end))

    _train_phase("learn_A", A, None, args.n_a_sweeps)
    print(f"  end Phase 1: A_bit={history['acc_bit_A'][-1]*100:.1f}%  "
          f"frames so far: {len(frames)}")
    _train_phase("learn_B", B, None, args.n_b_sweeps)
    print(f"  end Phase 2: A_bit={history['acc_bit_A'][-1]*100:.1f}%  "
          f"B_bit={history['acc_bit_B'][-1]*100:.1f}%  "
          f"frames so far: {len(frames)}")
    _train_phase("rehearse", A, subset_idx, args.n_rehearse_sweeps)
    print(f"  end Phase 3: A_bit={history['acc_bit_A'][-1]*100:.1f}%  "
          f"frames so far: {len(frames)}")

    # Phase 4: test (no updates) — render a single test frame
    _record("test")
    phase_x_bounds.append(("test", sweep_global, sweep_global))
    frames.append(render_frame(
        model, A, B, rehearsed_mask, history, "test",
        sweep_global, phase_x_bounds))

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
