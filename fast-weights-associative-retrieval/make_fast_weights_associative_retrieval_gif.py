"""
Animated GIF for the fast-weights associative-retrieval task.

Each frame walks through one timestep of an example sequence and shows:

  Top    : the sequence so far, with the current step highlighted
  Middle : the fast-weights matrix A_t (heatmap)
  Bottom : the hidden-state vector h_t (one row of color cells)

The final frame also displays the predicted digit vs. the target.

Usage:
    python3 make_fast_weights_associative_retrieval_gif.py
    python3 make_fast_weights_associative_retrieval_gif.py --seed 0 --fps 2
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from fast_weights_associative_retrieval import (
    build_fast_weights_rnn, generate_sample, train,
    VOCAB_SIZE, N_DIGITS, _token_str,
)


def render_frame(seq: np.ndarray, target: int, text: str, t_now: int,
                 A_t: np.ndarray, h_t: np.ndarray,
                 logits: np.ndarray | None = None,
                 ) -> Image.Image:
    """One frame for timestep t_now (0-indexed)."""
    fig = plt.figure(figsize=(6.4, 4.6), dpi=110)
    gs = fig.add_gridspec(3, 1, height_ratios=[0.7, 3.2, 0.7],
                          hspace=0.45)

    T = int(seq.shape[0])
    H = h_t.shape[0]

    # ---- top: sequence with highlighted current step ----
    ax_top = fig.add_subplot(gs[0])
    ax_top.set_xlim(-0.5, T - 0.5)
    ax_top.set_ylim(-0.5, 1.2)
    ax_top.axis("off")
    for t in range(T):
        tok = _token_str(int(seq[t]))
        is_now = (t == t_now)
        face = "#fff4c2" if is_now else "#eeeeee"
        edge = "black" if is_now else "#aaaaaa"
        rect = plt.Rectangle((t - 0.4, 0.0), 0.8, 0.8,
                             facecolor=face, edgecolor=edge, linewidth=1.2)
        ax_top.add_patch(rect)
        ax_top.text(t, 0.4, tok, ha="center", va="center", fontsize=12,
                    fontweight=("bold" if is_now else "normal"))
        ax_top.text(t, -0.3, str(t), ha="center", va="center", fontsize=7,
                    color="#666")
    ax_top.text(T, 0.4, f"  ->  target = {target}",
                ha="left", va="center", fontsize=10, color="#444")
    ax_top.set_title(f"sequence  '{text}'   step {t_now}/{T-1}",
                     fontsize=10)

    # ---- middle: fast weights A_t heatmap ----
    ax_mid = fig.add_subplot(gs[1])
    vmax = float(np.max(np.abs(A_t))) + 1e-6
    im = ax_mid.imshow(A_t, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
    fro = float(np.linalg.norm(A_t))
    ax_mid.set_title(f"fast-weights matrix A_t   (||A||_F = {fro:.2f})",
                     fontsize=10)
    ax_mid.set_xticks([]); ax_mid.set_yticks([])
    fig.colorbar(im, ax=ax_mid, fraction=0.04, pad=0.02)

    # ---- bottom: hidden state h_t ----
    ax_bot = fig.add_subplot(gs[2])
    ax_bot.imshow(h_t.reshape(1, -1), cmap="RdBu_r", vmin=-1, vmax=1,
                  aspect="auto", interpolation="nearest")
    ax_bot.set_yticks([])
    ax_bot.set_xticks([])
    title = f"hidden state h_t (H = {H})"
    if logits is not None:
        pred = int(np.argmax(logits))
        ok = "OK" if pred == target else "WRONG"
        title += f"   |   pred = {pred}, target = {target}  [{ok}]"
    ax_bot.set_title(title, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=2000)
    p.add_argument("--n-hidden", type=int, default=80)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--fps", type=float, default=1.5)
    p.add_argument("--out", type=str,
                   default="fast_weights_associative_retrieval.gif")
    args = p.parse_args()

    print(f"[gif] training fast-weights RNN  n_pairs={args.n_pairs}  "
          f"hidden={args.n_hidden}  steps={args.n_steps}")
    model = build_fast_weights_rnn(
        n_in=VOCAB_SIZE, n_hidden=args.n_hidden, n_out=N_DIGITS,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed)
    history = train(model, n_steps=args.n_steps, batch_size=args.batch_size,
                    n_pairs=args.n_pairs, lr=args.lr, eval_every=500,
                    eval_batch=128, grad_clip=5.0, seed=args.seed,
                    verbose=False)
    print(f"[gif] training done  final eval acc = "
          f"{history['eval_acc'][-1]*100:.1f}%")

    rng = np.random.default_rng(args.seed + 1234)
    seq, target, text = generate_sample(n_pairs=args.n_pairs, rng=rng)
    print(f"[gif] sample sequence  '{text}'  target={target}")

    fwd = model.forward_sequence(seq, keep_trace=True)
    As = fwd["A_trace"]
    hs = fwd["h_trace"]
    T = int(seq.shape[0])

    frames = []
    for t in range(T):
        logits = fwd["logits"] if t == T - 1 else None
        # h_t shown is h_{t+1} (the hidden state AFTER processing step t)
        img = render_frame(seq, target, text, t_now=t,
                           A_t=As[t], h_t=hs[t + 1],
                           logits=logits)
        frames.append(img)

    duration_ms = int(1000.0 / args.fps)
    # Hold the final frame for an extra ~2s
    durations = [duration_ms] * (T - 1) + [duration_ms * 4]
    frames[0].save(args.out, save_all=True, append_images=frames[1:],
                   duration=durations, loop=0, optimize=True)
    print(f"[gif] wrote {args.out}  ({len(frames)} frames @ {args.fps} fps)")
    sz = os.path.getsize(args.out) / 1024
    print(f"[gif] size: {sz:.1f} KB")


if __name__ == "__main__":
    main()
