"""
Animated GIF for multi-level glimpse MNIST.

Each frame shows one glimpse step:
  Top-left   : the full MNIST image with the current glimpse box highlighted
               and previous glimpse boxes drawn faintly
  Top-right  : the 7x7 patch the network is reading at this step
  Middle     : the fast-weights matrix A_t (heatmap)
  Bottom     : the hidden-state vector h_t (one row of color cells)

The final frame also displays the predicted digit vs. the target.

Usage:
    python3 make_multi_level_glimpse_mnist_gif.py
    python3 make_multi_level_glimpse_mnist_gif.py --seed 0 --fps 2
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image

from multi_level_glimpse_mnist import (
    GlimpseFastWeightsRNN, build_glimpse_rnn_with_fast_weights,
    build_glimpse_inputs, train, load_mnist,
    GLIMPSE_OFFSETS, PATCH_SIZE, N_GLIMPSES, GLIMPSE_DIM, N_CLASSES,
)


def render_frame(image: np.ndarray, label: int, t_now: int,
                 X_one: np.ndarray, A_t: np.ndarray, h_t: np.ndarray,
                 logits: np.ndarray | None = None,
                 ) -> Image.Image:
    """One frame for glimpse step t_now (0..23)."""
    fig = plt.figure(figsize=(8.0, 6.0), dpi=90)
    gs = fig.add_gridspec(3, 2,
                          height_ratios=[2.6, 2.0, 0.55],
                          width_ratios=[1.5, 1.0],
                          hspace=0.50, wspace=0.30)

    H = h_t.shape[0]

    # ---- top-left: image with glimpse boxes ----
    ax_img = fig.add_subplot(gs[0, 0])
    ax_img.imshow(image, cmap="gray", interpolation="nearest")
    for t in range(t_now + 1):
        r, c = GLIMPSE_OFFSETS[t]
        is_now = (t == t_now)
        is_central = t >= 16
        if is_now:
            edge, lw, alpha = "#ffeb3b", 2.5, 1.0
        elif is_central:
            edge, lw, alpha = "#1565c0", 0.8, 0.45
        else:
            edge, lw, alpha = "#c62828", 0.8, 0.45
        rect = mpatches.Rectangle(
            (c - 0.5, r - 0.5), PATCH_SIZE, PATCH_SIZE,
            linewidth=lw, edgecolor=edge, facecolor="none", alpha=alpha)
        ax_img.add_patch(rect)
    ax_img.set_title(f"image (label = {label})   glimpse {t_now+1}/{N_GLIMPSES}",
                     fontsize=10)
    ax_img.set_xticks([0, 7, 14, 21, 27])
    ax_img.set_yticks([0, 7, 14, 21, 27])

    # ---- top-right: current 7x7 patch ----
    ax_patch = fig.add_subplot(gs[0, 1])
    patch = X_one[t_now, :PATCH_SIZE * PATCH_SIZE].reshape(PATCH_SIZE,
                                                           PATCH_SIZE)
    ax_patch.imshow(patch, cmap="gray", vmin=0.0, vmax=1.0,
                    interpolation="nearest")
    r, c = GLIMPSE_OFFSETS[t_now]
    role = "centre re-glimpse" if t_now >= 16 else "fine patch"
    ax_patch.set_title(f"glimpse {t_now}\noffset=({r},{c})  [{role}]",
                       fontsize=9)
    ax_patch.set_xticks([]); ax_patch.set_yticks([])

    # ---- middle: fast-weights matrix A_t ----
    ax_mid = fig.add_subplot(gs[1, :])
    vmax = float(np.max(np.abs(A_t))) + 1e-6
    im = ax_mid.imshow(A_t, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                       interpolation="nearest")
    fro = float(np.linalg.norm(A_t))
    ax_mid.set_title(f"fast-weights matrix A_t   (||A||_F = {fro:.2f})",
                     fontsize=10)
    ax_mid.set_xticks([]); ax_mid.set_yticks([])
    fig.colorbar(im, ax=ax_mid, fraction=0.04, pad=0.02)

    # ---- bottom: hidden state h_t ----
    ax_bot = fig.add_subplot(gs[2, :])
    ax_bot.imshow(h_t.reshape(1, -1), cmap="RdBu_r", vmin=-1, vmax=1,
                  aspect="auto", interpolation="nearest")
    ax_bot.set_yticks([])
    ax_bot.set_xticks([])
    title = f"hidden state h_t (H = {H})"
    if logits is not None:
        pred = int(np.argmax(logits))
        ok = "OK" if pred == label else "WRONG"
        title += f"   |   pred = {pred}, label = {label}  [{ok}]"
    ax_bot.set_title(title, fontsize=10)

    fig.tight_layout(rect=(0, 0, 1, 0.97))
    buf = BytesIO()
    fig.savefig(buf, format="png", bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    img = Image.open(buf).convert("RGB")
    # Downscale if huge — we want gif under 3 MB.
    return img


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=2)
    p.add_argument("--n-hidden", type=int, default=96)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--lr", type=float, default=2e-3)
    p.add_argument("--n-train", type=int, default=20000)
    p.add_argument("--fps", type=float, default=2.5)
    p.add_argument("--out", type=str,
                   default="multi_level_glimpse_mnist.gif")
    p.add_argument("--max-width", type=int, default=560,
                   help="downscale frames to this width (keep gif small)")
    args = p.parse_args()

    print(f"[gif] loading MNIST")
    train_x, train_y, test_x, test_y = load_mnist()
    if args.n_train and args.n_train < len(train_x):
        rng = np.random.default_rng(args.seed)
        idx = rng.permutation(len(train_x))[:args.n_train]
        train_x = train_x[idx]
        train_y = train_y[idx]

    print(f"[gif] training glimpse RNN  hidden={args.n_hidden}  "
          f"epochs={args.n_epochs}")
    train_X = build_glimpse_inputs(train_x)
    test_X  = build_glimpse_inputs(test_x)
    model = build_glimpse_rnn_with_fast_weights(
        glimpse_dim=GLIMPSE_DIM, n_hidden=args.n_hidden, n_classes=N_CLASSES,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed)
    history = train(model, (train_X, train_y, test_X, test_y),
                    n_epochs=args.n_epochs, batch_size=args.batch_size,
                    lr=args.lr, grad_clip=5.0, eval_every=500,
                    seed=args.seed, verbose=False)
    print(f"[gif] training done  test_acc = {history['test_acc'][-1]*100:.1f}%")

    # Pick a clean correctly-classified test image so the gif tells a clear story.
    rng = np.random.default_rng(args.seed + 1234)
    candidates = rng.permutation(len(test_x))
    chosen_i = None
    for i in candidates[:200]:
        x_i = build_glimpse_inputs(test_x[i:i+1])[0]
        if int(model.predict(x_i[None])[0]) == int(test_y[i]):
            chosen_i = int(i)
            break
    if chosen_i is None:
        chosen_i = int(candidates[0])
    image = test_x[chosen_i]
    label = int(test_y[chosen_i])
    X_one = build_glimpse_inputs(image[None])[0]
    print(f"[gif] sample image idx={chosen_i}  label={label}")

    fwd = model.forward(X_one[None])
    A = fwd["A"][0]                                      # (T, H, H)
    h = fwd["h"][0]                                      # (T+1, H)
    logits = fwd["logits"][0]
    T = A.shape[0]

    frames = []
    for t in range(T):
        last = (t == T - 1)
        frame = render_frame(image, label, t_now=t,
                             X_one=X_one, A_t=A[t], h_t=h[t + 1],
                             logits=logits if last else None)
        # downscale to keep GIF small
        if args.max_width > 0 and frame.width > args.max_width:
            ratio = args.max_width / frame.width
            new_size = (args.max_width, int(frame.height * ratio))
            frame = frame.resize(new_size, Image.LANCZOS)
        frames.append(frame)

    duration_ms = int(1000.0 / args.fps)
    durations = [duration_ms] * (T - 1) + [duration_ms * 4]
    # palette quantization to shrink size
    p_frames = [f.convert("P", palette=Image.ADAPTIVE, colors=128)
                for f in frames]
    p_frames[0].save(args.out, save_all=True, append_images=p_frames[1:],
                     duration=durations, loop=0, optimize=True)
    sz = os.path.getsize(args.out) / 1024
    print(f"[gif] wrote {args.out}  ({len(frames)} frames @ {args.fps} fps)")
    print(f"[gif] size: {sz:.1f} KB")


if __name__ == "__main__":
    main()
