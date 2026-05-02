"""Animate the gate's specialization emerging during MoE training.

Trains a fresh MoE from scratch, snapshotting parameters each epoch, then
renders a 2-panel GIF: left = argmax-gate partition over F1/F2 input space
(experts emerging), right = test-accuracy curve climbing.

Output: vowel_mixture_experts.gif (target <= 3 MB).
"""

from __future__ import annotations

import argparse
import io
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from PIL import Image

from vowel_mixture_experts import (
    MoE,
    MLP,
    accuracy,
    build_moe,
    load_peterson_barney,
    sgd_step,
    split_by_speaker,
    standardise,
    VOWEL_LABELS,
)
from visualize_vowel_mixture_experts import (
    CLASS_COLOURS,
    EXPERT_COLOURS,
    _grid_in_input_space,
)

try:
    import imageio.v2 as imageio  # type: ignore
    HAVE_IMAGEIO = True
except Exception:
    HAVE_IMAGEIO = False


def _frame(moe: MoE, mlp_acc_hist, moe_acc_hist, data, epoch: int, n_epochs: int) -> Image.Image:
    F1, F2, grid_std = _grid_in_input_space(data, n=180)
    _p, g, _pe = moe.predict(grid_std)
    expert_argmax = g.argmax(axis=1).reshape(F1.shape)
    cmap_e = ListedColormap(EXPERT_COLOURS[: moe.n_experts])
    Xtr = data["Xtr"] * data["feat_std"] + data["feat_mean"]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.6))

    ax = axes[0]
    ax.pcolormesh(F2, F1, expert_argmax, cmap=cmap_e, shading="auto", alpha=0.7,
                  vmin=-0.5, vmax=moe.n_experts - 0.5)
    for cls in range(4):
        m = data["ytr"] == cls
        ax.scatter(Xtr[m, 1], Xtr[m, 0], c=CLASS_COLOURS[cls], s=12,
                   edgecolor="black", linewidth=0.3, label=VOWEL_LABELS[cls])
    ax.invert_yaxis(); ax.invert_xaxis()
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
    g_train = moe.predict(data["Xtr"])[1]
    used = sorted(set(g_train.argmax(axis=1).tolist()))
    ax.set_title(f"Gate partition  (epoch {epoch+1}/{n_epochs}, active experts {used})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=8)

    ax = axes[1]
    epochs_plot = np.arange(1, len(moe_acc_hist) + 1)
    ax.plot(epochs_plot, moe_acc_hist, color="#1f77b4", lw=1.8, label="MoE")
    ax.plot(np.arange(1, len(mlp_acc_hist) + 1), mlp_acc_hist, color="#d62728", lw=1.8,
            label="Monolithic MLP")
    ax.axhline(0.90, color="grey", linestyle="--", lw=0.8)
    ax.axhline(0.25, color="grey", linestyle=":", lw=0.8)
    ax.set_xlim(1, n_epochs)
    ax.set_ylim(0.0, 1.05)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_title("Test accuracy")
    ax.legend(loc="lower right", framealpha=0.9, fontsize=9)

    fig.suptitle("MoE on Peterson-Barney vowels: experts specializing during training",
                 y=1.02)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("P", palette=Image.ADAPTIVE, colors=128)


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-experts", type=int, default=4)
    p.add_argument("--n-epochs", type=int, default=60)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", type=str, default="vowel_mixture_experts.gif")
    p.add_argument("--max-frames", type=int, default=40,
                   help="Cap on rendered frames to keep gif size small.")
    p.add_argument("--frame-ms", type=int, default=180)
    args = p.parse_args(argv)

    X, y, sp, _is_real = load_peterson_barney()
    Xtr_raw, ytr, Xte_raw, yte = split_by_speaker(X, y, sp, 0.75, seed=args.seed)
    Xtr, Xte, mu, sd = standardise(Xtr_raw, Xte_raw)
    data = dict(Xtr=Xtr, ytr=ytr, Xte=Xte, yte=yte, feat_mean=mu, feat_std=sd)

    moe = build_moe(n_experts=args.n_experts, n_in=2, n_out=4, seed=args.seed)
    n_hidden = max(2, int(round((moe.n_params - 4) / 7)))
    mlp = MLP.init(n_in=2, n_hidden=n_hidden, n_classes=4, seed=args.seed)

    rng = np.random.default_rng(args.seed)
    moe_acc, mlp_acc = [], []
    frames: list[Image.Image] = []
    # Pick frame indices uniformly so the gif stays under the size cap.
    frame_at = set(np.linspace(0, args.n_epochs - 1, args.max_frames).round().astype(int))
    N = Xtr.shape[0]

    for epoch in range(args.n_epochs):
        perm = rng.permutation(N)
        for start in range(0, N, args.batch_size):
            idx = perm[start:start + args.batch_size]
            xb, yb = Xtr[idx], ytr[idx]
            _, gm, _ = moe.loss_and_grads(xb, yb)
            sgd_step(moe, gm, args.lr)
            _, gp, _ = mlp.loss_and_grads(xb, yb)
            sgd_step(mlp, gp, args.lr)
        moe_acc.append(accuracy(moe, Xte, yte))
        mlp_acc.append(accuracy(mlp, Xte, yte))
        if epoch in frame_at:
            frames.append(_frame(moe, mlp_acc, moe_acc, data, epoch, args.n_epochs))

    out = Path(args.out)
    if HAVE_IMAGEIO:
        imageio.mimsave(out, [np.asarray(f.convert("RGB")) for f in frames],
                        duration=args.frame_ms / 1000.0, loop=0)
    else:
        # PIL-only fallback path.
        first, *rest = frames
        first.save(out, save_all=True, append_images=rest,
                   duration=args.frame_ms, loop=0, optimize=True)
    print(f"wrote {out}  ({out.stat().st_size/1024:.1f} KB, {len(frames)} frames)")
    print(f"final  MoE test acc {moe_acc[-1]:.3f}  MLP test acc {mlp_acc[-1]:.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
