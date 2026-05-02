"""Generate ``bouncing_balls_3.gif`` -- the README headline animation.

Per frame:
   left half  : a ground-truth bouncing-balls sequence
   right half : a free rollout from the RTRBM, conditioned on the first
                `warmup` frames of the same sequence

The GIF stitches together rollouts at several training checkpoints
(epochs 1, 5, 10, 25, 50 by default), so a single animation shows BOTH the
data distribution (the simulator) AND the model's learning dynamics
(the rollout improving as training progresses), which is the per-spec
requirement (issue #1, "GIF illustrates both the problem definition AND
the learning dynamics").
"""
from __future__ import annotations

import argparse
import os
import time

import imageio.v2 as imageio
import numpy as np

from bouncing_balls_3 import (
    build_rtrbm,
    make_dataset,
    rollout,
    train,
)


def _to_uint8(img: np.ndarray) -> np.ndarray:
    return (np.clip(img, 0.0, 1.0) * 255.0).astype(np.uint8)


def _scale_up(img: np.ndarray, scale: int = 4) -> np.ndarray:
    """Pixel-replicate scale up. Keeps things crisp."""
    return np.repeat(np.repeat(img, scale, axis=0), scale, axis=1)


def _draw_label(canvas: np.ndarray, text: str, top: int, left: int) -> None:
    """Render `text` as a tiny pixel font in `canvas` (in-place).

    Lightweight 5x7 bitmap font for ASCII letters/digits used here. We only
    need a handful of characters, so we hardcode them. Falls back to a
    simple line of '|' bars if a character isn't in the table.
    """
    # 5x7 glyphs (rows top-to-bottom).
    glyphs = {
        "T": ["11111", "00100", "00100", "00100", "00100", "00100", "00100"],
        "R": ["11110", "10001", "10001", "11110", "10100", "10010", "10001"],
        "U": ["10001", "10001", "10001", "10001", "10001", "10001", "01110"],
        "H": ["10001", "10001", "10001", "11111", "10001", "10001", "10001"],
        "M": ["10001", "11011", "10101", "10101", "10001", "10001", "10001"],
        "O": ["01110", "10001", "10001", "10001", "10001", "10001", "01110"],
        "D": ["11110", "10001", "10001", "10001", "10001", "10001", "11110"],
        "E": ["11111", "10000", "10000", "11110", "10000", "10000", "11111"],
        "L": ["10000", "10000", "10000", "10000", "10000", "10000", "11111"],
        "P": ["11110", "10001", "10001", "11110", "10000", "10000", "10000"],
        "I": ["11111", "00100", "00100", "00100", "00100", "00100", "11111"],
        "C": ["01110", "10001", "10000", "10000", "10000", "10001", "01110"],
        "N": ["10001", "11001", "10101", "10011", "10001", "10001", "10001"],
        "A": ["01110", "10001", "10001", "11111", "10001", "10001", "10001"],
        "G": ["01110", "10001", "10000", "10111", "10001", "10001", "01110"],
        "F": ["11111", "10000", "10000", "11110", "10000", "10000", "10000"],
        "Y": ["10001", "10001", "01010", "00100", "00100", "00100", "00100"],
        "S": ["01111", "10000", "10000", "01110", "00001", "00001", "11110"],
        "0": ["01110", "10001", "10011", "10101", "11001", "10001", "01110"],
        "1": ["00100", "01100", "00100", "00100", "00100", "00100", "01110"],
        "2": ["01110", "10001", "00001", "00010", "00100", "01000", "11111"],
        "3": ["11110", "00001", "00001", "01110", "00001", "00001", "11110"],
        "4": ["00010", "00110", "01010", "10010", "11111", "00010", "00010"],
        "5": ["11111", "10000", "11110", "00001", "00001", "10001", "01110"],
        "6": ["01110", "10000", "10000", "11110", "10001", "10001", "01110"],
        "7": ["11111", "00001", "00010", "00100", "01000", "01000", "01000"],
        "8": ["01110", "10001", "10001", "01110", "10001", "10001", "01110"],
        "9": ["01110", "10001", "10001", "01111", "00001", "00001", "01110"],
        " ": ["00000", "00000", "00000", "00000", "00000", "00000", "00000"],
        "=": ["00000", "00000", "11111", "00000", "11111", "00000", "00000"],
        ".": ["00000", "00000", "00000", "00000", "00000", "00000", "00100"],
        ":": ["00000", "00100", "00000", "00000", "00000", "00100", "00000"],
        "+": ["00000", "00100", "00100", "11111", "00100", "00100", "00000"],
        "-": ["00000", "00000", "00000", "11111", "00000", "00000", "00000"],
        "/": ["00001", "00010", "00010", "00100", "01000", "01000", "10000"],
    }
    char_w, char_h, gap = 5, 7, 1
    x = left
    for ch in text.upper():
        glyph = glyphs.get(ch, glyphs[" "])
        for r in range(char_h):
            for c in range(char_w):
                if glyph[r][c] == "1":
                    yy = top + r
                    xx = x + c
                    if (0 <= yy < canvas.shape[0]
                            and 0 <= xx < canvas.shape[1]):
                        canvas[yy, xx] = 255
        x += char_w + gap


def _compose_frame(truth_frame: np.ndarray,
                   pred_frame: np.ndarray,
                   h: int, w: int,
                   scale: int,
                   step_idx: int,
                   warmup: int,
                   epoch_label: str | None) -> np.ndarray:
    """Build a single combined frame: truth on the left, prediction on the
    right, separated by a 2-pixel band, plus header strips for labels.
    """
    truth_img = _scale_up(truth_frame.reshape(h, w), scale=scale)
    pred_img = _scale_up(pred_frame.reshape(h, w), scale=scale)

    # both grayscale; concatenate horizontally with a separator
    sep = np.full((truth_img.shape[0], 4), 0.4, dtype=np.float32)
    body = np.concatenate([truth_img, sep, pred_img], axis=1)

    # Two header rows so labels never overlap. Top row: epoch label
    # (centered). Second row: TRUTH / WARMUP|PRED labels (per-half).
    header_h = 22
    canvas = np.zeros((header_h + body.shape[0], body.shape[1]),
                      dtype=np.uint8)
    canvas[header_h:, :] = _to_uint8(body)

    if epoch_label:
        # center horizontally
        text_w = len(epoch_label) * 6
        _draw_label(canvas, epoch_label, top=2,
                    left=max(2, (body.shape[1] - text_w) // 2))
    _draw_label(canvas, "TRUTH", top=12, left=2)
    pred_x = truth_img.shape[1] + 4 + 2
    if step_idx < warmup:
        _draw_label(canvas, "WARMUP", top=12, left=pred_x)
    else:
        _draw_label(canvas, f"PRED +{step_idx - warmup + 1}",
                    top=12, left=pred_x)
    return canvas


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="bouncing-balls-3 GIF builder")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h", type=int, default=30)
    p.add_argument("--w", type=int, default=30)
    p.add_argument("--n-balls", type=int, default=3)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--n-sequences", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=100)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-epochs", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--rollout-warmup", type=int, default=10)
    p.add_argument("--rollout-future", type=int, default=30)
    p.add_argument("--n-gibbs", type=int, default=25)
    p.add_argument("--scale", type=int, default=4,
                   help="pixel-replication scale for the GIF")
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--checkpoints", type=str, default="1,5,10,25,50",
                   help="comma-separated epoch checkpoints for rollouts "
                        "(must all be <= --n-epochs)")
    p.add_argument("--out", type=str, default="bouncing_balls_3.gif")
    return p.parse_args()


def main() -> None:
    args = _parse_args()
    n_visible = args.h * args.w

    sequences = make_dataset(n_sequences=args.n_sequences,
                             seq_len=args.seq_len,
                             n_balls=args.n_balls,
                             h=args.h, w=args.w,
                             radius=args.radius, speed=args.speed,
                             seed=args.seed)
    model = build_rtrbm(n_visible=n_visible, n_hidden=args.n_hidden,
                        seed=args.seed)
    data_mean = np.clip(sequences.reshape(-1, n_visible).mean(axis=0),
                        1e-3, 1 - 1e-3)
    model.b_v[:] = np.log(data_mean / (1.0 - data_mean)).astype(np.float32)

    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    if any(c > args.n_epochs for c in checkpoints):
        raise ValueError(
            f"checkpoint > n_epochs: {checkpoints} vs {args.n_epochs}")
    if any(c < 1 for c in checkpoints):
        raise ValueError(f"checkpoint < 1 not allowed: {checkpoints}")

    val = make_dataset(n_sequences=1, seq_len=args.seq_len,
                       n_balls=args.n_balls,
                       h=args.h, w=args.w,
                       radius=args.radius, speed=args.speed,
                       seed=args.seed + 9999)[0]
    n_future = min(args.rollout_future,
                   args.seq_len - args.rollout_warmup)

    snapshots: list[tuple[int, np.ndarray]] = []

    def snapshot(epoch: int, m, history) -> None:
        # Called *after* finishing epoch (epoch+1 in user terms).
        e = epoch + 1
        if e in checkpoints:
            pred = rollout(m, val[:args.rollout_warmup],
                           n_future=n_future,
                           n_gibbs=args.n_gibbs,
                           sample_visible=False, sample_hidden=True)
            snapshots.append((e, pred.copy()))
            print(f"  -> snapshot at epoch {e}: "
                  f"recon_mse={history['recon_mse'][-1]:.4f}")

    t0 = time.time()
    history = train(model, sequences,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    verbose=True,
                    snapshot_callback=snapshot,
                    snapshot_every=1)
    print(f"# trained in {time.time() - t0:.2f}s, final recon MSE = "
          f"{history['recon_mse'][-1]:.4f}")

    if not snapshots:
        # safety net: at least show the final state
        pred = rollout(model, val[:args.rollout_warmup],
                       n_future=n_future,
                       n_gibbs=args.n_gibbs,
                       sample_visible=False, sample_hidden=True)
        snapshots.append((args.n_epochs, pred))

    # ---- assemble frames ----
    frames = []
    n_show = args.rollout_warmup + n_future
    for epoch, pred in snapshots:
        for k in range(n_show):
            frame = _compose_frame(val[k], pred[k],
                                   h=args.h, w=args.w,
                                   scale=args.scale,
                                   step_idx=k,
                                   warmup=args.rollout_warmup,
                                   epoch_label=f"EPOCH {epoch}")
            frames.append(frame)
        # hold a final pause-frame for half a second so the viewer can
        # register the segment break
        for _ in range(args.fps // 2):
            frames.append(frames[-1])

    print(f"# writing {args.out} ({len(frames)} frames, "
          f"{frames[0].shape[1]}x{frames[0].shape[0]})")
    imageio.mimsave(args.out, frames, fps=args.fps, loop=0)
    sz = os.path.getsize(args.out)
    print(f"# {args.out} size: {sz/1024:.1f} KB "
          f"({len(snapshots)} epoch checkpoints)")


if __name__ == "__main__":
    main()
