"""
Animated GIF: ortho-init vs random-init training, side by side, on the
3-bit memorization task.

Why 3-bit memorization? It is the cleanest task where ortho solves
within a minute and random fails completely (chance = 12.5%, ortho
->100%, random stuck near chance). Watching the two accuracy curves
diverge is the visual statement of the headline.

Each frame:
  Top row    : per-init training-loss curves up to current epoch
  Bottom row : per-init accuracy curves with chance-line baseline
  Overlay    : current epoch number; ortho's "solved" sweep is marked
                 once it's reached.

Usage:
    python3 make_rnn_pathological_gif.py
    python3 make_rnn_pathological_gif.py --task addition --sequence-len 30
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from rnn_pathological import (
    RNN, TASK_SPEC, generate_dataset, _grad_clip, chance_baseline,
)


def _train_with_snapshots(task: str, sequence_len: int, n_hidden: int,
                          init: str, n_epochs: int, batch_size: int,
                          batches_per_epoch: int, lr: float, momentum: float,
                          clip: float, seed: int):
    """A copy of train_with_momentum that records the full per-epoch
    history including per-step batch metrics, so we can render frames."""
    spec = TASK_SPEC[task]
    rng_data = np.random.default_rng(seed + 7919)
    m = RNN(n_in=spec["n_in"], n_hidden=n_hidden, n_out=spec["n_out"],
            init=init, seed=seed)
    velos = {k: np.zeros_like(getattr(m, k))
             for k in ["W_ih", "W_hh", "b_h", "W_hy", "b_y"]}

    history = {"epoch": [], "loss": [], "metric": [],
               "metric_name": spec["metric_name"], "loss_kind": spec["loss"],
               "solved_epoch": None}

    solved_threshold = {
        "addition": ("mse", 0.05),
        "xor": ("acc", 0.95),
        "temporal_order": ("acc", 0.90),
        "3bit_memorization": ("acc", 0.90),
    }[task]

    for ep in range(n_epochs):
        bls, bms = [], []
        for _ in range(batches_per_epoch):
            x, y = generate_dataset(task, sequence_len, batch_size, rng_data)
            fwd = m.forward(x)
            grads = m.backward(x, y, fwd, loss=spec["loss"])
            grads = _grad_clip(grads, clip)
            for k in velos:
                velos[k] = momentum * velos[k] - lr * grads[k]
                setattr(m, k, getattr(m, k) + velos[k])
            if spec["loss"] == "mse":
                bls.append(RNN.loss_mse(fwd["logits"], y))
                bms.append(bls[-1])
            else:
                bls.append(RNN.loss_ce(fwd["logits"], y))
                bms.append(RNN.accuracy_ce(fwd["logits"], y))
        loss_val = float(np.mean(bls)); met_val = float(np.mean(bms))
        history["epoch"].append(ep + 1)
        history["loss"].append(loss_val)
        history["metric"].append(met_val)
        kind, thr = solved_threshold
        solved = (met_val < thr) if kind == "mse" else (met_val > thr)
        if solved and history["solved_epoch"] is None:
            history["solved_epoch"] = ep + 1

    return history


def _render_frame(ortho_hist: dict, rand_hist: dict, current_ep: int,
                  task: str, T: int, chance: float,
                  total_epochs: int) -> Image.Image:
    fig, axes = plt.subplots(2, 1, figsize=(8.5, 5.6), dpi=110,
                              sharex=True)

    # ---- top: loss ----
    ax = axes[0]
    e_o = ortho_hist["epoch"][:current_ep]
    l_o = ortho_hist["loss"][:current_ep]
    e_r = rand_hist["epoch"][:current_ep]
    l_r = rand_hist["loss"][:current_ep]

    ax.plot(e_o, l_o, color="#1f77b4", linewidth=2.0, label="orthogonal init")
    ax.plot(e_r, l_r, color="#d62728", linewidth=2.0, label="random init")
    ax.set_yscale("log")
    ax.set_ylabel("training loss")
    ax.grid(alpha=0.3, which="both")
    ax.legend(fontsize=9, loc="upper right")
    ax.set_xlim(0, total_epochs)

    # set y-range to the union so it doesn't jump around
    all_l = ortho_hist["loss"] + rand_hist["loss"]
    if all_l:
        ax.set_ylim(min(all_l) * 0.7, max(all_l) * 1.4)

    # ---- bottom: metric ----
    ax = axes[1]
    name = ortho_hist["metric_name"]
    if name == "accuracy":
        ax.plot(e_o, np.array(l_o) * 0 + 0,  # placeholder so legend ordering matches
                 alpha=0)
        m_o = np.array(ortho_hist["metric"][:current_ep]) * 100
        m_r = np.array(rand_hist["metric"][:current_ep]) * 100
        ax.plot(e_o, m_o, color="#1f77b4", linewidth=2.0)
        ax.plot(e_r, m_r, color="#d62728", linewidth=2.0)
        ax.axhline(chance * 100, color="gray", linestyle=":",
                    linewidth=1.0, label=f"chance ({chance*100:.1f}%)")
        ax.set_ylim(0, 105)
        ax.set_ylabel("accuracy (%)")
    else:
        m_o = ortho_hist["metric"][:current_ep]
        m_r = rand_hist["metric"][:current_ep]
        ax.plot(e_o, m_o, color="#1f77b4", linewidth=2.0)
        ax.plot(e_r, m_r, color="#d62728", linewidth=2.0)
        ax.axhline(chance, color="gray", linestyle=":",
                    linewidth=1.0, label=f"chance ({chance:.3f})")
        ax.set_ylabel("MSE")

    if (ortho_hist["solved_epoch"] is not None
            and current_ep >= ortho_hist["solved_epoch"]):
        ax.axvline(ortho_hist["solved_epoch"], color="#1f77b4",
                    linestyle="--", linewidth=1.0, alpha=0.7,
                    label=f"ortho solved @ {ortho_hist['solved_epoch']}")
    ax.set_xlabel("epoch")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="best")
    ax.set_xlim(0, total_epochs)

    cur_o = ortho_hist["metric"][current_ep - 1] if current_ep > 0 else 0
    cur_r = rand_hist["metric"][current_ep - 1] if current_ep > 0 else 0
    cur_o_str = f"{cur_o*100:.1f}%" if name == "accuracy" else f"{cur_o:.3f}"
    cur_r_str = f"{cur_r*100:.1f}%" if name == "accuracy" else f"{cur_r:.3f}"
    fig.suptitle(
        f"task = {task},  T = {T},  epoch {current_ep}/{total_epochs}\n"
        f"orthogonal-init {name} = {cur_o_str}    "
        f"random-init {name} = {cur_r_str}",
        fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--task", type=str, default="3bit_memorization")
    p.add_argument("--sequence-len", type=int, default=60)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--n-epochs", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=50)
    p.add_argument("--batches-per-epoch", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--clip", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot-every", type=int, default=2,
                   help="render a frame every N epochs")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--hold-final", type=int, default=12)
    p.add_argument("--max-frame-side", type=int, default=900)
    p.add_argument("--out", type=str, default="rnn_pathological.gif")
    args = p.parse_args()

    print(f"Training {args.task} (T={args.sequence_len}) under both inits, "
           f"{args.n_epochs} epochs each...")

    h_o = _train_with_snapshots(args.task, args.sequence_len, args.n_hidden,
                                "ortho", args.n_epochs, args.batch_size,
                                args.batches_per_epoch, args.lr, args.momentum,
                                args.clip, args.seed)
    h_r = _train_with_snapshots(args.task, args.sequence_len, args.n_hidden,
                                "random", args.n_epochs, args.batch_size,
                                args.batches_per_epoch, args.lr, args.momentum,
                                args.clip, args.seed)
    print(f"  ortho   : final {h_o['metric_name']}={h_o['metric'][-1]:.3f}  "
           f"solved@{h_o['solved_epoch']}")
    print(f"  random  : final {h_r['metric_name']}={h_r['metric'][-1]:.3f}  "
           f"solved@{h_r['solved_epoch']}")

    chance = chance_baseline(args.task, args.sequence_len)

    # build frame list at every `snapshot_every` epochs
    frames: list[Image.Image] = []
    snap_eps = list(range(args.snapshot_every, args.n_epochs + 1,
                          args.snapshot_every))
    if snap_eps[-1] != args.n_epochs:
        snap_eps.append(args.n_epochs)
    for cur in snap_eps:
        frame = _render_frame(h_o, h_r, cur, args.task, args.sequence_len,
                               chance, args.n_epochs)
        if max(frame.size) > args.max_frame_side:
            scale = args.max_frame_side / max(frame.size)
            frame = frame.resize((int(frame.size[0] * scale),
                                   int(frame.size[1] * scale)), Image.LANCZOS)
        frames.append(frame)

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    palette_frame = frames[0].quantize(colors=128, method=Image.MEDIANCUT)
    frames_q = [f.quantize(colors=128, method=Image.MEDIANCUT,
                            palette=palette_frame) for f in frames]
    frames_q[0].save(args.out, save_all=True, append_images=frames_q[1:],
                      duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(args.out) / 1024
    print(f"\nWrote {args.out}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
