"""
Static visualizations for the affNIST robustness experiment.

Outputs (in `viz/`):
  example_pairs.png            -- translated-MNIST vs affNIST examples
  accuracy_bars.png            -- bar chart of in-distribution vs affNIST acc
  per_class_robustness.png     -- per-class affNIST accuracy + capsnet/cnn gap
  training_curves.png          -- val-acc vs step for both archs

All plots read from `results.json` written by `affnist.py --out results.json`.
If `results.json` is missing, the script trains both models with default args.
"""
from __future__ import annotations
import argparse
import json
import os
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from affnist import (
    TinyCapsNet, TinyCNN, train, evaluate,
    load_mnist, load_affnist_test, make_translated_mnist,
)


def _maybe_train_and_load(results_path: str, n_epochs: int, n_train: int,
                          n_test: int, seed: int):
    if Path(results_path).exists():
        print(f"  loading existing {results_path}")
        return json.loads(Path(results_path).read_text())
    print(f"  no {results_path}; running quick training")
    # Run the same path the CLI does, but in-process and only as a fallback.
    import subprocess, sys
    cmd = [sys.executable, "affnist.py", "--arch", "both",
           "--n-epochs", str(n_epochs), "--n-train", str(n_train),
           "--n-test", str(n_test), "--seed", str(seed),
           "--out", results_path]
    subprocess.check_call(cmd, cwd=os.path.dirname(__file__) or ".")
    return json.loads(Path(results_path).read_text())


def plot_example_pairs(out_path: str, n_examples: int = 6, seed: int = 0):
    test_imgs, test_lbls = load_mnist("test")
    rng = np.random.default_rng(seed)
    sel = rng.permutation(test_imgs.shape[0])[:n_examples]
    base = test_imgs[sel]
    tr = make_translated_mnist(base, max_shift=6, seed=seed)
    aff_x, aff_y, _ = load_affnist_test(n_synth=n_examples, seed=seed + 1)

    fig, axes = plt.subplots(2, n_examples, figsize=(2.0 * n_examples, 4.5), dpi=120)
    for i in range(n_examples):
        axes[0, i].imshow(tr[i], cmap="gray", vmin=0, vmax=1)
        axes[0, i].set_title(f"trans-MNIST\nlabel={int(test_lbls[sel[i]])}", fontsize=9)
        axes[1, i].imshow(aff_x[i], cmap="gray", vmin=0, vmax=1)
        axes[1, i].set_title(f"affNIST (synth)\nlabel={int(aff_y[i])}", fontsize=9)
    for ax in axes.flat:
        ax.set_xticks([]); ax.set_yticks([])
    fig.suptitle("Train distribution (top) vs test distribution (bottom)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_accuracy_bars(results: dict, out_path: str):
    archs = list(results["models"].keys())
    in_dist = [results["models"][a]["translated_mnist_acc"] for a in archs]
    aff = [results["models"][a]["affnist_acc"] for a in archs]
    x = np.arange(len(archs))
    w = 0.35

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    bars1 = ax.bar(x - w/2, in_dist, w, label="translated-MNIST (in dist.)",
                   color="#4c72b0")
    bars2 = ax.bar(x + w/2, aff, w, label="affNIST (out of dist.)",
                   color="#dd8452")
    for b, v in zip(bars1, in_dist):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9)
    for b, v in zip(bars2, aff):
        ax.text(b.get_x() + b.get_width()/2, v + 0.01, f"{v:.2f}",
                ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([a.upper() for a in archs])
    ax.set_ylabel("accuracy")
    ax.set_ylim(0, 1.05)
    ax.set_title("Generalization gap: train translated, test affine")
    ax.legend(loc="upper right")
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def _per_class_acc(model, x, y, n_classes=10, batch_size=64):
    preds = []
    for i in range(0, x.shape[0], batch_size):
        preds.append(model.predict(x[i:i + batch_size]))
    preds = np.concatenate(preds)
    accs = np.zeros(n_classes)
    for c in range(n_classes):
        m = (y == c)
        accs[c] = float((preds[m] == y[m]).mean()) if m.sum() else float("nan")
    return accs, preds


def plot_per_class_robustness(out_path: str, n_epochs: int, n_train: int,
                              n_test: int, seed: int):
    """Re-train and evaluate per-class. Avoids storing models in JSON."""
    print("  re-training models for per-class breakdown...")
    aff_x, aff_y, _ = load_affnist_test(n_synth=n_test, seed=seed)

    caps, _ = train(arch="capsnet", n_epochs=n_epochs, n_train=n_train, seed=seed,
                    verbose=False)
    cnn, _ = train(arch="cnn", n_epochs=n_epochs, n_train=n_train, seed=seed,
                   verbose=False)
    caps_acc, _ = _per_class_acc(caps, aff_x, aff_y)
    cnn_acc, _ = _per_class_acc(cnn, aff_x, aff_y)

    classes = np.arange(10)
    w = 0.35
    fig, ax = plt.subplots(figsize=(9, 4.5), dpi=120)
    ax.bar(classes - w/2, caps_acc, w, label="CapsNet", color="#4c72b0")
    ax.bar(classes + w/2, cnn_acc, w, label="CNN", color="#dd8452")
    ax.set_xticks(classes)
    ax.set_xlabel("digit")
    ax.set_ylabel("accuracy on affNIST")
    ax.set_title("Per-class affNIST accuracy")
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")
    return caps_acc, cnn_acc


def plot_training_curves(results: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    colors = {"capsnet": "#4c72b0", "cnn": "#dd8452"}
    for arch, info in results["models"].items():
        h = info["history"]
        if not h["step"]:
            continue
        ax.plot(h["step"], h["val_acc"], color=colors.get(arch, None),
                label=arch.upper(), marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("translated-MNIST val accuracy")
    ax.set_title("Validation accuracy through training")
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--results", default="results.json")
    p.add_argument("--outdir", default="viz")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--no-per-class", action="store_true",
                   help="skip the per-class plot (it re-trains)")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    results = _maybe_train_and_load(args.results, args.n_epochs, args.n_train,
                                    args.n_test, args.seed)
    plot_example_pairs(str(out / "example_pairs.png"), seed=args.seed)
    plot_accuracy_bars(results, str(out / "accuracy_bars.png"))
    plot_training_curves(results, str(out / "training_curves.png"))
    if not args.no_per_class:
        plot_per_class_robustness(str(out / "per_class_robustness.png"),
                                  n_epochs=args.n_epochs, n_train=args.n_train,
                                  n_test=args.n_test, seed=args.seed)


if __name__ == "__main__":
    main()
