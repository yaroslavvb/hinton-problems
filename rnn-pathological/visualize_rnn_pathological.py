"""
Static visualizations for the pathological-RNN ortho-vs-random comparison.

Reads `results.json` (produced by `python3 rnn_pathological.py --all`) and
emits four PNGs in `viz/`:

  ortho_vs_random.png    -- per-task training curves overlaid for both
                              inits (the headline figure)
  summary_table.png      -- final-metric comparison as a colour-coded grid
  spectrum_W_hh.png      -- singular-value spectrum of W_hh under the two
                              inits, before and after training (explains
                              *why* ortho works)
  task_examples.png      -- one input/target pair per task, showing what
                              the network actually has to remember

If `results.json` is missing the script will train fresh.
"""

from __future__ import annotations
import argparse
import json
import os
import sys

import numpy as np
import matplotlib.pyplot as plt

from rnn_pathological import (
    RNN, TASK_SPEC, TASKS, generate_dataset, train_with_momentum,
    chance_baseline, _ortho_matrix,
)


HEADLINE_TASKS = ("addition", "temporal_order", "3bit_memorization")


# ----------------------------------------------------------------------
# Headline: per-task training curves, ortho vs random
# ----------------------------------------------------------------------

def plot_ortho_vs_random(results: dict, out_path: str) -> None:
    histories = results["histories"]
    rows = results["rows"]
    n_tasks = len(HEADLINE_TASKS)

    fig, axes = plt.subplots(2, n_tasks, figsize=(4.2 * n_tasks, 6.5),
                              dpi=130, sharex="col")
    if n_tasks == 1:
        axes = axes[:, None]

    for col, task in enumerate(HEADLINE_TASKS):
        h_o = histories[f"{task}__ortho"]
        h_r = histories[f"{task}__random"]
        chance = next(r["chance"] for r in rows if r["task"] == task)

        # ---- top: loss ----
        ax = axes[0, col]
        ax.plot(h_o["epoch"], h_o["loss"], color="#1f77b4", linewidth=1.6,
                 label="orthogonal init")
        ax.plot(h_r["epoch"], h_r["loss"], color="#d62728", linewidth=1.6,
                 label="random init")
        ax.set_yscale("log")
        ax.set_title(f"{task}  (T = {h_o['sequence_len']})", fontsize=11)
        ax.set_ylabel("training loss")
        ax.grid(alpha=0.3, which="both")
        if col == 0:
            ax.legend(fontsize=9, loc="upper right")

        # ---- bottom: metric (accuracy or MSE), with chance line ----
        ax = axes[1, col]
        m_name = h_o["metric_name"]
        if m_name == "accuracy":
            ax.plot(h_o["epoch"], np.array(h_o["metric"]) * 100,
                     color="#1f77b4", linewidth=1.6, label="orthogonal")
            ax.plot(h_r["epoch"], np.array(h_r["metric"]) * 100,
                     color="#d62728", linewidth=1.6, label="random")
            ax.axhline(chance * 100, color="gray", linestyle=":",
                        linewidth=1.0, label=f"chance ({chance*100:.0f}%)")
            ax.set_ylim(0, 105)
            ax.set_ylabel("accuracy (%)")
        else:
            ax.plot(h_o["epoch"], h_o["metric"],
                     color="#1f77b4", linewidth=1.6, label="orthogonal")
            ax.plot(h_r["epoch"], h_r["metric"],
                     color="#d62728", linewidth=1.6, label="random")
            ax.axhline(chance, color="gray", linestyle=":",
                        linewidth=1.0, label=f"chance ({chance:.3f})")
            ax.set_ylabel("MSE")

        if h_o["solved_epoch"]:
            ax.axvline(h_o["solved_epoch"], color="#1f77b4",
                        linestyle="--", linewidth=0.9, alpha=0.6)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8, loc="best")

    fig.suptitle("Pathological RNN tasks: orthogonal vs random recurrent init",
                  fontsize=13, y=1.00)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Summary table as a colour-coded heatmap
# ----------------------------------------------------------------------

def plot_summary_table(results: dict, out_path: str) -> None:
    rows = results["rows"]
    fig, ax = plt.subplots(figsize=(11.5, 1.8 + 0.5 * len(HEADLINE_TASKS)),
                            dpi=140)
    ax.axis("off")

    headers = ["task", "T", "metric",
               "chance", "ortho final", "random final",
               "ortho solved@", "random solved@"]
    data = []
    cell_colors = []
    for task in HEADLINE_TASKS:
        o = next(r for r in rows if r["task"] == task and r["init"] == "ortho")
        r = next(rr for rr in rows if rr["task"] == task and rr["init"] == "random")
        # row
        data.append([
            task, str(o["T"]), o["metric_name"],
            f"{o['chance']:.3f}",
            f"{o['final_metric']:.3f}",
            f"{r['final_metric']:.3f}",
            str(o["solved_epoch"]) if o["solved_epoch"] else "—",
            str(r["solved_epoch"]) if r["solved_epoch"] else "—",
        ])
        # green for ortho if solved, red for random if failed
        is_acc = o["metric_name"] == "accuracy"
        ortho_solved = o["solved_epoch"] is not None
        rand_failed = r["solved_epoch"] is None
        rc = ["white"] * len(headers)
        if ortho_solved:
            rc[4] = "#c8e6c9"; rc[6] = "#c8e6c9"
        if rand_failed:
            rc[5] = "#ffcdd2"; rc[7] = "#ffcdd2"
        cell_colors.append(rc)

    tbl = ax.table(cellText=data, colLabels=headers, loc="center",
                    cellColours=cell_colors,
                    colColours=["#eceff1"] * len(headers),
                    cellLoc="center")
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(9.5)
    tbl.scale(1.0, 1.6)
    ax.set_title("Headline: orthogonal init solves; random init fails at the same T",
                  fontsize=11, pad=12)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Spectrum of W_hh (the structural reason ortho works)
# ----------------------------------------------------------------------

def plot_spectrum(results: dict, out_path: str, n_hidden: int = 64,
                  retrain_for_after: bool = True) -> None:
    """Plot singular-value spectra of W_hh:
      (a) at init for both ortho and random
      (b) after training (one task) for both inits

    The "ortho" spectrum stays close to 1, which is why gradients flow.
    The "random" spectrum has a tail of small values that produces
    vanishing gradients through long unrolls.
    """
    rng = np.random.default_rng(0)

    # init-time matrices
    W_ortho_init = _ortho_matrix(n_hidden, rng)
    W_rand_init = 0.1 * rng.standard_normal((n_hidden, n_hidden))

    # post-training matrices, taken from a fresh quick run so the
    # comparison uses the same task and same epochs
    if retrain_for_after:
        m_o, _ = train_with_momentum(
            "addition", sequence_len=30, n_hidden=n_hidden, init="ortho",
            n_epochs=results["config"]["n_epochs"],
            batch_size=results["config"]["batch_size"],
            batches_per_epoch=results["config"]["batches_per_epoch"],
            lr=results["config"]["lr"], momentum=results["config"]["momentum"],
            clip=results["config"]["clip"],
            seed=results["config"]["seed"], verbose=False)
        m_r, _ = train_with_momentum(
            "addition", sequence_len=30, n_hidden=n_hidden, init="random",
            n_epochs=results["config"]["n_epochs"],
            batch_size=results["config"]["batch_size"],
            batches_per_epoch=results["config"]["batches_per_epoch"],
            lr=results["config"]["lr"], momentum=results["config"]["momentum"],
            clip=results["config"]["clip"],
            seed=results["config"]["seed"], verbose=False)
        W_ortho_after = m_o.W_hh
        W_rand_after = m_r.W_hh
    else:
        W_ortho_after = W_ortho_init
        W_rand_after = W_rand_init

    sv_oi = np.linalg.svd(W_ortho_init, compute_uv=False)
    sv_ri = np.linalg.svd(W_rand_init, compute_uv=False)
    sv_oa = np.linalg.svd(W_ortho_after, compute_uv=False)
    sv_ra = np.linalg.svd(W_rand_after, compute_uv=False)

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.0), dpi=130, sharey=True)

    ax = axes[0]
    idx = np.arange(1, n_hidden + 1)
    ax.plot(idx, sv_oi, color="#1f77b4", linewidth=1.5,
             label="orthogonal init")
    ax.plot(idx, sv_ri, color="#d62728", linewidth=1.5,
             label="random init  (N(0, 0.1))")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("singular-value index")
    ax.set_ylabel(r"singular value of $W_{hh}$")
    ax.set_title(r"$W_{hh}$ spectrum at *init*")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    ax = axes[1]
    ax.plot(idx, sv_oa, color="#1f77b4", linewidth=1.5,
             label="orthogonal init  (after training on addition)")
    ax.plot(idx, sv_ra, color="#d62728", linewidth=1.5,
             label="random init  (after training on addition)")
    ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8)
    ax.set_xlabel("singular-value index")
    ax.set_title(r"$W_{hh}$ spectrum after training")
    ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="best")

    fig.suptitle(
        r"Why ortho wins: singular values of $W_{hh}$ stay clustered near 1 "
        "(every direction propagates equally through $T$ tanh layers); "
        "random init has a long tail of small singular values (those "
        "directions vanish), and the tail persists through training.",
        fontsize=10.5, y=1.02)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# Task examples: one (x, y) per task, visualised
# ----------------------------------------------------------------------

def plot_task_examples(out_path: str, T_addition: int = 30,
                       T_temporal: int = 60, T_3bit: int = 60) -> None:
    rng = np.random.default_rng(42)

    fig, axes = plt.subplots(3, 1, figsize=(11, 7.0), dpi=130)

    # ---- addition ----
    ax = axes[0]
    x, y = generate_dataset("addition", T_addition, 1, rng)
    val = x[0, :, 0]; mark = x[0, :, 1]
    t = np.arange(T_addition)
    ax.bar(t, val, color="#90caf9", edgecolor="black", linewidth=0.3,
            label="value (channel 0)")
    mk = np.where(mark > 0)[0]
    for ti in mk:
        ax.bar(ti, val[ti], color="#1565c0", edgecolor="black",
                linewidth=0.6, width=0.85)
        ax.text(ti, val[ti] + 0.04, f"{val[ti]:.2f}",
                 ha="center", fontsize=9, color="#0d47a1")
    ax.set_ylim(0, 1.2)
    ax.set_ylabel("value")
    ax.set_xlabel("timestep")
    ax.set_title(f"addition  (T = {T_addition}). "
                  f"Sum the two marked values.  target = {y[0]:.3f}",
                  fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    ax.set_xticks(range(0, T_addition, max(1, T_addition // 10)))
    ax.grid(alpha=0.25, axis="y")

    # ---- temporal_order ----
    ax = axes[1]
    x, y = generate_dataset("temporal_order", T_temporal, 1, rng)
    syms = "ABCDEF"
    seq = np.argmax(x[0], axis=1)
    colors = ["#e57373" if s < 2 else "#bdbdbd" for s in seq]
    t = np.arange(T_temporal)
    ax.scatter(t, np.zeros(T_temporal), c=colors, s=80,
                edgecolors="black", linewidths=0.4)
    for ti in range(T_temporal):
        ax.text(ti, 0.07, syms[seq[ti]],
                 ha="center", va="bottom", fontsize=8.5,
                 color="black" if seq[ti] >= 2 else "#b71c1c",
                 weight="bold" if seq[ti] < 2 else "normal")
    ax.set_xlim(-1, T_temporal)
    ax.set_ylim(-0.4, 0.45)
    ax.set_yticks([])
    ax.set_xlabel("timestep")
    pair = "AABBAB"[y[0]]
    pair = ["AA", "AB", "BA", "BB"][y[0]]
    ax.set_title(f"temporal_order  (T = {T_temporal}, vocab=6). "
                  f"Two cued positions in red carry A or B; the rest are "
                  f"distractors.  target = {pair} (class {y[0]})",
                  fontsize=10)
    ax.set_xticks(range(0, T_temporal, max(1, T_temporal // 10)))

    # ---- 3bit_memorization ----
    ax = axes[2]
    x, y = generate_dataset("3bit_memorization", T_3bit, 1, rng)
    seq = np.argmax(x[0], axis=1)
    labels = ["bit=0", "bit=1", "QUERY", "noise", "noise"]
    colors = ["#1565c0", "#1565c0", "#fbc02d", "#bdbdbd", "#bdbdbd"]
    t = np.arange(T_3bit)
    for ti in range(T_3bit):
        ax.bar(ti, 1.0, color=colors[seq[ti]], edgecolor="black",
                linewidth=0.3, width=0.9)
        if seq[ti] in (0, 1):
            ax.text(ti, 0.5, str(seq[ti]), ha="center", va="center",
                     color="white", fontsize=10, weight="bold")
        elif seq[ti] == 2:
            ax.text(ti, 0.5, "?", ha="center", va="center",
                     color="black", fontsize=11, weight="bold")
    ax.set_xlim(-1, T_3bit)
    ax.set_ylim(0, 1.2)
    ax.set_yticks([])
    ax.set_xlabel("timestep")
    bits = format(y[0], "03b")
    ax.set_title(f"3bit_memorization  (T = {T_3bit}, vocab=5). "
                  f"First 3 timesteps carry bits to memorise (blue); the "
                  f"rest is noise; final query timestep cues recall.  "
                  f"target = {bits} (class {y[0]})",
                  fontsize=10)
    ax.set_xticks(range(0, T_3bit, max(1, T_3bit // 10)))

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", type=str, default="results.json",
                   help="JSON dumped by `rnn_pathological.py --all`")
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--n-hidden", type=int, default=64)
    args = p.parse_args()

    if not os.path.exists(args.results):
        print(f"{args.results} not found; running `rnn_pathological.py --all` first")
        os.system(f"{sys.executable} rnn_pathological.py --all")
    with open(args.results) as f:
        results = json.load(f)

    os.makedirs(args.outdir, exist_ok=True)
    plot_ortho_vs_random(results, os.path.join(args.outdir,
                                                  "ortho_vs_random.png"))
    plot_summary_table(results, os.path.join(args.outdir, "summary_table.png"))
    plot_spectrum(results, os.path.join(args.outdir, "spectrum_W_hh.png"),
                   n_hidden=args.n_hidden)
    plot_task_examples(os.path.join(args.outdir, "task_examples.png"))


if __name__ == "__main__":
    main()
