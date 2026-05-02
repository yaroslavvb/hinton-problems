"""Render an animated GIF of the recurrent FF inference loop on MNIST.

Each frame shows one iteration of the synchronous update for a single test
image, while the network runs through 8 iterations under each candidate
label. The frame layout, top-to-bottom:

    1. The input image, with the candidate label boxed.
    2. Hidden layer 1 activations (1D bar strip).
    3. Hidden layer 2 activations (1D bar strip).
    4. The 10 per-label goodness traces, growing one step at a time.
    5. The 10 per-label running predictions (current best in red).

The animation cycles through one row of test images and 8 iterations per
image, producing approximately n_examples * n_iters frames.
"""

from __future__ import annotations

import argparse
import os
import time

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ff_recurrent_mnist import (
    build_recurrent_ff,
    init_states,
    l2_normalize,
    load_mnist,
    load_model,
    one_hot,
    relu,
    train,
)


def render_frame(image, true_label, candidate, iter_idx, n_iters,
                 hidden_states, goodness_history, running_pred):
    """Build a single frame as a numpy uint8 RGB image."""
    n_hidden = len(hidden_states)
    fig = plt.figure(figsize=(10, 6.5), dpi=80)
    gs = fig.add_gridspec(2, 3, width_ratios=[1.2, 2.0, 1.2],
                           height_ratios=[1.0, 1.0], hspace=0.6, wspace=0.4)

    # 1) image
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image.reshape(28, 28), cmap="gray_r")
    if running_pred < 0:
        rp = "(waits for iter 3)"
        color = "0.4"
    elif running_pred == true_label:
        rp = f"running pred={running_pred}"
        color = "C2"
    else:
        rp = f"running pred={running_pred}"
        color = "C3"
    ax_img.set_title(f"true={true_label}, candidate={candidate}\n{rp}",
                     color=color, fontsize=10)
    ax_img.axis("off")

    # 2) hidden activations
    ax_h1 = fig.add_subplot(gs[0, 1])
    ax_h2 = fig.add_subplot(gs[1, 1])
    for ax, h, k in [(ax_h1, hidden_states[0], 1), (ax_h2, hidden_states[1], 2)]:
        # bar plot of first 80 units (so we can see structure)
        n_show = min(80, h.shape[0])
        ax.bar(np.arange(n_show), h[:n_show], color="C0")
        ax.set_xlim(-0.5, n_show - 0.5)
        ax.set_ylim(0, max(0.05, h.max() * 1.2 + 1e-6))
        ax.set_xlabel(f"hidden unit (showing {n_show} of {h.shape[0]})")
        ax.set_title(f"layer {k} activations at iter {iter_idx}")

    # 3) goodness curves
    ax_g = fig.add_subplot(gs[0, 2])
    n_classes = goodness_history.shape[0]
    ts = np.arange(1, iter_idx + 1)
    for c in range(n_classes):
        is_true = c == true_label
        is_cand = c == candidate
        if is_true:
            ax_g.plot(ts, goodness_history[c, :iter_idx], "C3-", lw=2.0,
                       label=f"true={c}")
        elif is_cand:
            ax_g.plot(ts, goodness_history[c, :iter_idx], "C0-", lw=1.6,
                       label=f"cand={c}")
        else:
            ax_g.plot(ts, goodness_history[c, :iter_idx], color="0.7", lw=0.7)
    ax_g.axvspan(2.5, 5.5, alpha=0.15, color="C2")
    ax_g.set_xlim(0.5, n_iters + 0.5)
    if iter_idx > 0:
        ax_g.set_ylim(0, max(1e-3, goodness_history[:, :iter_idx].max() * 1.1))
    ax_g.set_xlabel("iter")
    ax_g.set_ylabel("Σ goodness")
    ax_g.set_title("per-candidate goodness")
    ax_g.legend(fontsize=7, loc="upper left")

    # 4) running scoreboard - accumulate goodness only over iters 3..min(5, iter_idx)
    ax_score = fig.add_subplot(gs[1, 2])
    end = min(5, iter_idx)
    if iter_idx >= 3:
        accum = goodness_history[:, 2:end].sum(axis=1)
        argmax = int(np.argmax(accum))
    else:
        accum = np.zeros(n_classes, dtype=np.float32)
        argmax = -1
    colors = ["C0"] * n_classes
    if argmax >= 0:
        colors[argmax] = "C3"
    ax_score.bar(np.arange(n_classes), accum, color=colors)
    ax_score.set_xticks(range(n_classes))
    ax_score.set_xlabel("candidate label")
    ax_score.set_ylabel(f"Σ goodness over iters 3..{end if end >= 3 else '?'}")
    title = "scoreboard (red = current pick)" if iter_idx >= 3 else "scoreboard (waits for iter 3)"
    ax_score.set_title(title)

    fig.suptitle(f"Forward-Forward recurrent on MNIST — iter {iter_idx} / {n_iters}",
                 fontsize=12)

    fig.canvas.draw()
    # robustly grab RGB regardless of backend
    buf = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
    w, h = fig.canvas.get_width_height()
    img = buf.reshape(h, w, 4)[..., :3].copy()
    plt.close(fig)
    return img


def make_gif(model, X, y, n_iters: int, path: str,
             n_examples: int = 3, fps: int = 4, seed: int = 0):
    rng = np.random.default_rng(seed + 11)
    L = model["L"]
    damping = model["damping"]
    n_classes = int(model["sizes"][-1])

    # pick n_examples test images
    idx = rng.choice(X.shape[0], size=n_examples, replace=False)

    frames = []
    for i in idx:
        img = X[i:i + 1]
        true_label = int(y[i])

        # for each iteration, we want to show the dynamics for ONE candidate
        # label at a time, while computing all candidates' goodness in
        # parallel. We cycle through candidates by rendering them per frame.
        # To keep the GIF size small, we show only one candidate per image:
        # the true label (matches paper's clamped-label inference).
        candidate = true_label
        label_oh = one_hot(np.array([candidate]), n_classes)
        states = init_states(model, img, label_oh)

        # compute all candidates' goodness at every iter for the scoreboard
        all_g = np.zeros((n_classes, n_iters), dtype=np.float32)
        # we'll just precompute the running goodness for each candidate
        for c in range(n_classes):
            lo = one_hot(np.array([c]), n_classes)
            st = init_states(model, img, lo)
            for t in range(n_iters):
                new_st = list(st)
                for k in range(1, L - 1):
                    up = l2_normalize(st[k - 1]) @ model["W_up"][k - 1]
                    dn = l2_normalize(st[k + 1]) @ model["W_dn"][k]
                    pre = up + dn + model["b"][k]
                    new_st[k] = damping * relu(pre) + (1.0 - damping) * st[k]
                st = new_st
                all_g[c, t] = float(np.sum([np.mean(st[k] ** 2) for k in range(1, L - 1)]))

        # now redo for the chosen candidate, capturing per-iter hidden states
        for t in range(n_iters):
            new_states = list(states)
            for k in range(1, L - 1):
                up = l2_normalize(states[k - 1]) @ model["W_up"][k - 1]
                dn = l2_normalize(states[k + 1]) @ model["W_dn"][k]
                pre = up + dn + model["b"][k]
                new_states[k] = damping * relu(pre) + (1.0 - damping) * states[k]
            states = new_states
            iter_idx = t + 1
            # Running prediction uses the same partial accumulation as the
            # scoreboard so the title and bar plot agree.
            if iter_idx >= 3:
                running_accum = all_g[:, 2:min(5, iter_idx)].sum(axis=1)
                running_pred = int(np.argmax(running_accum))
            else:
                running_pred = -1
            hidden_states = [states[k][0] for k in range(1, L - 1)]
            frame = render_frame(
                img[0], true_label, candidate, iter_idx, n_iters,
                hidden_states, all_g, running_pred,
            )
            frames.append(frame)

    print(f"[gif] writing {len(frames)} frames at {fps} fps -> {path}")
    imageio.mimsave(path, frames, format="GIF", fps=fps, loop=0)
    size_kb = os.path.getsize(path) / 1024.0
    print(f"[gif] wrote {path} ({size_kb:.1f} KB)")


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
    p.add_argument("--n-examples", type=int, default=3)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", type=str, default="ff_recurrent_mnist.gif")
    p.add_argument("--load-model", type=str, default=None,
                   help="Load weights from this .npz instead of training.")
    args = p.parse_args()

    print("[gif] loading MNIST...")
    Xtr, ytr, Xte, yte = load_mnist(verbose=False)

    if args.load_model:
        print(f"[gif] loading {args.load_model}")
        model = load_model(args.load_model)
    else:
        if args.n_train < Xtr.shape[0]:
            rng = np.random.default_rng(args.seed)
            idx = rng.permutation(Xtr.shape[0])[:args.n_train]
            Xtr, ytr = Xtr[idx], ytr[idx]

        sizes = tuple([784] + [args.hidden] * args.n_hidden_layers + [10])
        print(f"[gif] training {sizes} for {args.n_epochs} epochs on {args.n_train} samples")
        model = build_recurrent_ff(layer_sizes=sizes, damping=args.damping, seed=args.seed,
                                    init_scale=args.init_scale)
        t0 = time.time()
        train(
            model, Xtr, ytr, Xte, yte,
            n_epochs=args.n_epochs, batch_size=args.batch_size, lr=args.lr,
            n_iters=args.n_iters, threshold=args.threshold, seed=args.seed,
            eval_test_subset=2000,
        )
        print(f"[gif] training: {time.time() - t0:.1f}s")

    make_gif(model, Xte, yte, args.n_iters, args.out,
             n_examples=args.n_examples, fps=args.fps, seed=args.seed)


if __name__ == "__main__":
    main()
