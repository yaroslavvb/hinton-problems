"""
Render `affnist.gif`: an animation showing how CapsNet and CNN classify a fixed
panel of affNIST test digits as training progresses.

Each frame:
  - Top row: 6 fixed affNIST test images
  - Middle: per-image CapsNet prediction + tick/cross
  - Bottom: per-image CNN prediction + tick/cross
  - Right side: running val accuracy (translated MNIST) for both archs

Default budget: ~30 frames, ~1 minute of training per arch on a laptop.

Usage:
    python3 make_affnist_gif.py
    python3 make_affnist_gif.py --n-epochs 3 --snapshot-every 50 --fps 4
"""
from __future__ import annotations
import argparse
from io import BytesIO
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from affnist import (
    TinyCapsNet, TinyCNN, train, load_mnist, load_affnist_test,
    make_translated_mnist,
)


def _fixed_panel(seed: int = 7, n: int = 6):
    aff_x, aff_y, _ = load_affnist_test(n_synth=200, seed=seed)
    # take 6 with diverse labels
    chosen = []
    seen = set()
    for i in range(aff_x.shape[0]):
        if int(aff_y[i]) not in seen:
            chosen.append(i); seen.add(int(aff_y[i]))
        if len(chosen) >= n:
            break
    chosen = chosen[:n]
    return aff_x[chosen], aff_y[chosen]


def render_frame(panel_x, panel_y, caps_pred, cnn_pred,
                 caps_hist, cnn_hist, step):
    n = panel_x.shape[0]
    fig = plt.figure(figsize=(2.0 * n + 4.5, 5.5), dpi=110)
    gs = fig.add_gridspec(3, n + 2, width_ratios=[1] * n + [0.2, 1.4])

    # rows of images + predictions
    for i in range(n):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(panel_x[i], cmap="gray", vmin=0, vmax=1)
        ax.set_title(f"y={int(panel_y[i])}", fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

        ax = fig.add_subplot(gs[1, i])
        ok = caps_pred[i] == panel_y[i]
        ax.text(0.5, 0.5, f"CapsNet\n{int(caps_pred[i])} {'ok' if ok else 'X'}",
                ha="center", va="center", fontsize=11,
                color=("#2ca02c" if ok else "#d62728"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f0f0f0" if ok else "#fbeaea")

        ax = fig.add_subplot(gs[2, i])
        ok = cnn_pred[i] == panel_y[i]
        ax.text(0.5, 0.5, f"CNN\n{int(cnn_pred[i])} {'ok' if ok else 'X'}",
                ha="center", va="center", fontsize=11,
                color=("#2ca02c" if ok else "#d62728"))
        ax.set_xticks([]); ax.set_yticks([])
        ax.set_facecolor("#f0f0f0" if ok else "#fbeaea")

    # right-side training curve (spans all rows in last column)
    ax = fig.add_subplot(gs[:, n + 1])
    if caps_hist["step"]:
        ax.plot(caps_hist["step"], caps_hist["val_acc"], label="CapsNet",
                color="#4c72b0", marker="o", markersize=3)
    if cnn_hist["step"]:
        ax.plot(cnn_hist["step"], cnn_hist["val_acc"], label="CNN",
                color="#dd8452", marker="o", markersize=3)
    ax.set_xlabel("step")
    ax.set_ylabel("trans-MNIST val acc")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"step {step}", fontsize=10)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)

    fig.suptitle("Train: translated MNIST | Test: affNIST (synth) -- live",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=90)
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).copy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--panel-seed", type=int, default=7)
    p.add_argument("--n-epochs", type=int, default=3)
    p.add_argument("--n-train", type=int, default=3000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--snapshot-every", type=int, default=15)
    p.add_argument("--val-every", type=int, default=15)
    p.add_argument("--fps", type=int, default=4)
    p.add_argument("--out", default="affnist.gif")
    p.add_argument("--max-frames", type=int, default=40,
                   help="cap to keep gif under 3 MB")
    args = p.parse_args()

    panel_x, panel_y = _fixed_panel(seed=args.panel_seed)

    # We snapshot mid-training. Easiest is two passes (CapsNet, then CNN), each
    # with a snapshot callback that records (step, predictions). Then render
    # frames after both runs so the side-by-side curves can be drawn.
    caps_records = []
    cnn_records = []

    def caps_cb(step, model, hist):
        preds = model.predict(panel_x)
        caps_records.append(dict(step=step, preds=preds.copy(),
                                  hist={"step": list(hist["step"]),
                                        "val_acc": list(hist["val_acc"])}))

    def cnn_cb(step, model, hist):
        preds = model.predict(panel_x)
        cnn_records.append(dict(step=step, preds=preds.copy(),
                                 hist={"step": list(hist["step"]),
                                       "val_acc": list(hist["val_acc"])}))

    print("# training CapsNet (with snapshots)")
    train(arch="capsnet", n_epochs=args.n_epochs, n_train=args.n_train,
          batch_size=args.batch_size, seed=args.seed,
          snapshot_callback=caps_cb,
          val_every_steps=args.val_every,
          val_n=500, verbose=True)
    print("# training CNN (with snapshots)")
    train(arch="cnn", n_epochs=args.n_epochs, n_train=args.n_train,
          batch_size=args.batch_size, seed=args.seed,
          snapshot_callback=cnn_cb,
          val_every_steps=args.val_every,
          val_n=500, verbose=True)

    # Pair up snapshots by index. If lengths differ, take min.
    n_pairs = min(len(caps_records), len(cnn_records))
    if n_pairs == 0:
        raise RuntimeError("no snapshots collected; lower --snapshot-every")

    # Subsample to fit max_frames
    if n_pairs > args.max_frames:
        sel = np.linspace(0, n_pairs - 1, args.max_frames).astype(int)
    else:
        sel = np.arange(n_pairs)

    print(f"# rendering {len(sel)} frames...")
    frames = []
    for k in sel:
        cr = caps_records[k]
        nr = cnn_records[k]
        frame = render_frame(panel_x, panel_y, cr["preds"], nr["preds"],
                             cr["hist"], nr["hist"], step=max(cr["step"], nr["step"]))
        frames.append(frame)

    # Hold final frame for emphasis
    for _ in range(args.fps * 2):
        frames.append(frames[-1])

    out = Path(args.out)
    duration_ms = int(1000 / args.fps)
    frames[0].save(out, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    print(f"  wrote {out} ({out.stat().st_size / 1024:.1f} KB, {len(frames)} frames)")


if __name__ == "__main__":
    main()
