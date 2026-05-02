"""
Visualisations for MNIST-2x5 subclass distillation.

Generates four PNGs in `viz/`:
  - teacher_contingency.png    (teacher 10x10: sub-logit-argmax vs true digit;
                                should show 5x5 block diagonal structure)
  - student_contingency.png    (student 10x10 vs true digit, after distillation
                                from a teacher that only saw super-class labels)
  - accuracy_bars.png          (super-class acc / subclass-recovery acc /
                                super-only baseline / chance)
  - training_curves.png        (teacher super loss + aux loss + super accuracy;
                                student distillation loss)

Usage:
  python3 visualize_mnist_2x5_subclass.py --seed 0
  python3 visualize_mnist_2x5_subclass.py --results-json viz/results.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from mnist_2x5_subclass import (
    main as run_main,
    relabel_to_superclass,
)


def _ensure_results(args) -> dict:
    if args.results_json and Path(args.results_json).exists():
        with open(args.results_json) as f:
            return json.load(f)
    # else run end-to-end, then read the json the run wrote
    run_args = [
        "--seed", str(args.seed),
        "--n-epochs-teacher", str(args.n_epochs_teacher),
        "--n-epochs-student", str(args.n_epochs_student),
        "--temperature", str(args.temperature),
        "--aux-weight", str(args.aux_weight),
        "--sharpen", str(args.sharpen),
        "--out", str(Path(args.outdir) / "results.json"),
    ]
    if args.quiet:
        run_args.append("--quiet")
    run_main(run_args)
    with open(Path(args.outdir) / "results.json") as f:
        return json.load(f)


# ----------------------------------------------------------------------
# Confusion / contingency plots
# ----------------------------------------------------------------------

def plot_contingency(cont: np.ndarray, title: str, path: Path,
                     xlabel: str = "true digit", ylabel: str = "cluster (sub-logit argmax)") -> None:
    cont = np.asarray(cont, dtype=np.int64)
    n_clusters, n_classes = cont.shape
    fig, ax = plt.subplots(figsize=(6.0, 5.5))
    im = ax.imshow(cont, cmap="viridis", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_xticks(range(n_classes))
    ax.set_yticks(range(n_clusters))

    # super-class boundary marker (cluster 5)
    ax.axhline(4.5, color="white", linewidth=1.5, linestyle="--")

    # annotate each cell with its count
    vmax = cont.max() if cont.max() > 0 else 1
    for k in range(n_clusters):
        for c in range(n_classes):
            v = cont[k, c]
            if v > 0:
                colour = "white" if v < 0.6 * vmax else "black"
                ax.text(c, k, str(v), ha="center", va="center",
                        color=colour, fontsize=7)
    fig.colorbar(im, ax=ax, label="count")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_accuracy_bars(results: dict, path: Path) -> None:
    """Compare: teacher super-acc / student super-acc / subclass recovery /
    super-only baseline (50%) / random chance (10%)."""
    vals = [
        ("chance (10-way)", 0.10),
        ("super-only baseline\n(predict super-class for whole super-class)", 0.50),
        ("subclass recovery\n(any-mapping)", results["subclass_recovery_acc"]),
        ("subclass recovery\n(1-to-1 matching)", results["subclass_recovery_acc_1to1"]),
        ("teacher super-acc\n(supervised, binary)", results["teacher_super_acc_test"]),
        ("student super-acc\n(via 10-way logits)", results["super_acc_via_student"]),
    ]
    labels = [v[0] for v in vals]
    heights = [v[1] for v in vals]
    colours = ["#bbbbbb", "#888888", "#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd"]
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(range(len(vals)), heights, color=colours)
    ax.set_xticks(range(len(vals)))
    ax.set_xticklabels(labels, rotation=12, ha="right", fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("accuracy")
    ax.set_title("MNIST-2x5 subclass distillation: accuracy comparison")
    for b, h in zip(bars, heights):
        ax.text(b.get_x() + b.get_width() / 2, h + 0.012,
                f"{h*100:.1f}%", ha="center", va="bottom", fontsize=9)
    ax.axhline(0.5, color="grey", linestyle="--", linewidth=0.5)
    ax.axhline(0.1, color="grey", linestyle="--", linewidth=0.5)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


def plot_training_curves(results: dict, path: Path) -> None:
    th = results["teacher_history"]
    sh = results["student_history"]
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    ax = axes[0, 0]
    ax.plot(th["epoch"], th["super_loss"], "o-")
    ax.set_xlabel("epoch")
    ax.set_ylabel("super-class CE")
    ax.set_title("teacher: super-class cross-entropy")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(th["epoch"], th["aux_loss"], "o-", color="darkorange")
    ax.set_xlabel("epoch")
    ax.set_ylabel("aux loss (lower = better)")
    ax.set_title("teacher: auxiliary diversity+sharpness")
    ax.axhline(-np.log(5), color="grey", linestyle="--", linewidth=0.7,
               label=f"-H_max = -log 5 = {-np.log(5):.3f}")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(th["epoch"], [a * 100 for a in th["super_acc"]], "o-", color="green")
    ax.set_xlabel("epoch")
    ax.set_ylabel("super-class accuracy (train, %)")
    ax.set_title("teacher: super-class accuracy")
    ax.set_ylim(80, 101)
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.plot(sh["epoch"], sh["distill_loss"], "o-", color="purple")
    ax.set_xlabel("epoch")
    ax.set_ylabel("KL with teacher (T-scaled)")
    ax.set_title("student: distillation loss")
    ax.grid(alpha=0.3)

    fig.suptitle("MNIST-2x5 subclass distillation: training curves", fontsize=12)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    print(f"  wrote {path}")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs-teacher", type=int, default=12)
    p.add_argument("--n-epochs-student", type=int, default=12)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.5)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--results-json", type=str, default="",
                   help="reuse a results.json from a previous run "
                        "instead of re-training")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("# loading / running training")
    results = _ensure_results(args)

    print("# generating PNGs")
    teacher_cont = np.array(results["teacher_contingency"], dtype=np.int64)
    student_cont = np.array(results["student_contingency"], dtype=np.int64)
    plot_contingency(teacher_cont,
                     "Teacher 10x10 contingency: sub-logit argmax vs true digit\n"
                     "(teacher saw ONLY binary super-class labels)",
                     out / "teacher_contingency.png")
    plot_contingency(student_cont,
                     "Student 10x10 contingency: cluster id vs true digit\n"
                     "(student distilled from teacher's sub-logits, no 10-way labels)",
                     out / "student_contingency.png")
    plot_accuracy_bars(results, out / "accuracy_bars.png")
    plot_training_curves(results, out / "training_curves.png")
    print("done.")


if __name__ == "__main__":
    main()
