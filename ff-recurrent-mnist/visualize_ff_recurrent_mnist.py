"""Visualize the recurrent FF MNIST model.

Produces three PNGs in the chosen out directory:

    training_curves.png — train loss and test error vs epoch
    iteration_goodness.png — goodness traces vs iteration for 10 candidate
        labels on a few test images. Each panel shows the goodness summed
        across hidden layers as a function of iteration t in 1..n_iters.
        The true label's curve is highlighted; ranking is decided by the
        goodness summed over the test-time iterations (default 3..5).
    state_evolution.png — per-iteration hidden-layer activation heatmaps
        for one image. Top row: the image and its label. Subsequent rows:
        hidden layer 1 and hidden layer 2 activations across t=1..8,
        rendered as 1D bar strips per iteration.
"""

from __future__ import annotations

import argparse
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ff_recurrent_mnist import (
    build_recurrent_ff,
    evaluate,
    init_states,
    l2_normalize,
    load_mnist,
    load_model,
    one_hot,
    predict_by_iteration_goodness,
    relu,
    train,
)


def per_iteration_goodness(model, image, label_oh, n_iters: int = 8):
    """Return goodness of every hidden layer at every iteration. Shape (n_iters, K)."""
    L = model["L"]
    damping = model["damping"]
    states = init_states(model, image, label_oh)
    out = np.zeros((n_iters, L - 2), dtype=np.float32)
    for loop_t in range(n_iters):
        new_states = list(states)
        for k in range(1, L - 1):
            up = l2_normalize(states[k - 1]) @ model["W_up"][k - 1]
            dn = l2_normalize(states[k + 1]) @ model["W_dn"][k]
            pre = up + dn + model["b"][k]
            new_states[k] = damping * relu(pre) + (1.0 - damping) * states[k]
        states = new_states
        for kk, k in enumerate(range(1, L - 1)):
            out[loop_t, kk] = float(np.mean(states[k] ** 2))
    return out


def state_history(model, image, label_oh, n_iters: int = 8):
    """Return hidden-layer states for every iteration. List[List[ndarray]] of
    length n_iters+1; each inner list has L state arrays (only hidden layers
    are interesting)."""
    L = model["L"]
    damping = model["damping"]
    states = init_states(model, image, label_oh)
    history = [list(states)]
    for _t in range(n_iters):
        new_states = list(states)
        for k in range(1, L - 1):
            up = l2_normalize(states[k - 1]) @ model["W_up"][k - 1]
            dn = l2_normalize(states[k + 1]) @ model["W_dn"][k]
            pre = up + dn + model["b"][k]
            new_states[k] = damping * relu(pre) + (1.0 - damping) * states[k]
        states = new_states
        history.append(list(states))
    return history


def plot_training_curves(history, path):
    epochs = history["epoch"]
    fig, ax1 = plt.subplots(figsize=(7, 4.5))
    ax1.plot(epochs, history["train_loss"], "C0-", label="train loss (BCE on goodness)")
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("train loss", color="C0")
    ax1.tick_params(axis="y", labelcolor="C0")

    test_err = [e for e in history["test_err"] if e is not None]
    test_epochs = [history["epoch"][i] for i, e in enumerate(history["test_err"]) if e is not None]
    if test_err:
        ax2 = ax1.twinx()
        ax2.plot(test_epochs, [e * 100 for e in test_err], "C3o-", label="test error (%)")
        ax2.set_ylabel("test error (%)", color="C3")
        ax2.tick_params(axis="y", labelcolor="C3")

    ax1.set_title("Recurrent FF on MNIST — training curves")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_iteration_goodness(model, X, y, n_iters: int, path, n_examples: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed + 7)
    idx = rng.choice(X.shape[0], size=n_examples, replace=False)
    n_classes = int(model["sizes"][-1])

    fig, axes = plt.subplots(2, n_examples, figsize=(3.0 * n_examples, 6),
                              gridspec_kw={"height_ratios": [1, 2]})
    if n_examples == 1:
        axes = np.array(axes).reshape(2, 1)

    for col, i in enumerate(idx):
        img = X[i:i + 1]
        true_label = int(y[i])

        axes[0, col].imshow(img.reshape(28, 28), cmap="gray_r")
        axes[0, col].set_title(f"true label = {true_label}")
        axes[0, col].axis("off")

        per_label_curves = np.zeros((n_classes, n_iters), dtype=np.float32)
        for c in range(n_classes):
            label_oh = one_hot(np.array([c]), n_classes)
            traces = per_iteration_goodness(model, img, label_oh, n_iters=n_iters)
            per_label_curves[c] = traces.sum(axis=1)  # sum across hidden layers

        ax = axes[1, col]
        for c in range(n_classes):
            color = "C3" if c == true_label else "0.7"
            lw = 2.4 if c == true_label else 0.9
            label = f"label {c}" + (" (true)" if c == true_label else "")
            ax.plot(np.arange(1, n_iters + 1), per_label_curves[c], color=color, lw=lw,
                    label=label if c == true_label else None)
        # mark accumulation window 3..5
        ax.axvspan(2.5, 5.5, alpha=0.10, color="C2",
                   label="accumulation window (iters 3-5)")
        ax.set_xlabel("iteration t")
        ax.set_ylabel("Σ goodness over hidden layers")
        ax.legend(loc="upper left", fontsize=7)
        ax.set_title(f"per-label goodness (#{int(i)})")

    fig.suptitle("Recurrent FF — per-iteration goodness for 10 candidate labels")
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def plot_state_evolution(model, X, y, n_iters: int, path, sample_idx: int = 0):
    img = X[sample_idx:sample_idx + 1]
    true_label = int(y[sample_idx])
    label_oh = one_hot(np.array([true_label]), int(model["sizes"][-1]))
    history = state_history(model, img, label_oh, n_iters=n_iters)

    L = model["L"]
    n_hidden = L - 2
    fig, axes = plt.subplots(n_hidden + 1, 1, figsize=(8, 1.6 * (n_hidden + 1)),
                              gridspec_kw={"height_ratios": [1.5] + [1.0] * n_hidden})
    if n_hidden + 1 == 1:
        axes = np.array([axes])

    axes[0].imshow(img.reshape(28, 28), cmap="gray_r")
    axes[0].set_title(f"input image (true label = {true_label}, candidate = {true_label})")
    axes[0].axis("off")

    for kk, k in enumerate(range(1, L - 1)):
        # stack iters x dim
        block = np.stack([history[t + 1][k][0] for t in range(n_iters)], axis=0)
        ax = axes[kk + 1]
        im = ax.imshow(block, cmap="viridis", aspect="auto")
        ax.set_xlabel(f"hidden unit (dim={block.shape[1]})")
        ax.set_ylabel("iter t")
        ax.set_yticks(range(n_iters))
        ax.set_yticklabels([str(t + 1) for t in range(n_iters)])
        ax.set_title(f"hidden layer {k} state across iterations")
        fig.colorbar(im, ax=ax, fraction=0.025)

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=10)
    p.add_argument("--n-iters", type=int, default=8)
    p.add_argument("--damping", type=float, default=0.7)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--threshold", type=float, default=1.0)
    p.add_argument("--n-train", type=int, default=15000)
    p.add_argument("--hidden", type=int, default=400)
    p.add_argument("--n-hidden-layers", type=int, default=2)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--outdir", type=str, default="viz")
    p.add_argument("--load-model", type=str, default=None,
                   help="Load weights from this .npz instead of training.")
    p.add_argument("--results-json", type=str, default=None,
                   help="If --load-model is unset, optionally read history from this JSON.")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print("[viz] loading MNIST...")
    Xtr, ytr, Xte, yte = load_mnist(verbose=False)

    if args.load_model:
        print(f"[viz] loading {args.load_model}")
        model = load_model(args.load_model)
        history = None
        if args.results_json and os.path.exists(args.results_json):
            import json
            with open(args.results_json) as f:
                blob = json.load(f)
            history = blob.get("history")
    else:
        if args.n_train < Xtr.shape[0]:
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(Xtr.shape[0])[:args.n_train]
            Xtr, ytr = Xtr[idx], ytr[idx]

        sizes = tuple([784] + [args.hidden] * args.n_hidden_layers + [10])
        print(f"[viz] training {sizes} for {args.n_epochs} epochs on {args.n_train} samples")
        model = build_recurrent_ff(layer_sizes=sizes, damping=args.damping, seed=args.seed,
                                    init_scale=args.init_scale)
        t0 = time.time()
        history = train(
            model, Xtr, ytr, Xte, yte,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            n_iters=args.n_iters, threshold=args.threshold, seed=args.seed,
            eval_test_subset=2000,
        )
        print(f"[viz] training: {time.time() - t0:.1f}s")

    if history is not None:
        print("[viz] training_curves.png")
        plot_training_curves(history, os.path.join(args.outdir, "training_curves.png"))
    else:
        print("[viz] (no history available; skipping training_curves.png)")
    print("[viz] iteration_goodness.png")
    plot_iteration_goodness(model, Xte, yte, args.n_iters,
                            os.path.join(args.outdir, "iteration_goodness.png"))
    print("[viz] state_evolution.png")
    plot_state_evolution(model, Xte, yte, args.n_iters,
                         os.path.join(args.outdir, "state_evolution.png"))

    final_acc = evaluate(model, Xte, yte, batch_size=512, n_iters=args.n_iters)
    print(f"[viz] final test acc {final_acc * 100:.2f}% (err {(1 - final_acc) * 100:.2f}%)")


if __name__ == "__main__":
    main()
