"""
Animate EM convergence of flow capsules on a Geo frame pair.

Layout per frame (single example pair):
  Top row:    frame1, frame2, GT flow, GT segmentation
  Middle row: per-capsule responsibility maps + background
  Bottom row: reconstruction MSE curve (extending up to current iter)

Frames advance one EM iteration each, then hold for a few frames at the end.

Usage:
  python3 make_geo_flow_capsules_gif.py
  python3 make_geo_flow_capsules_gif.py --seed 0 --n-iters 30
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from geo_flow_capsules import (
    _fit_flow_capsules_once,
    generate_geo_pair,
    part_segmentation_iou,
)
from visualize_geo_flow_capsules import flow_to_rgb, _segmentation_rgb


def render_frame(pair: dict, snapshot: np.ndarray,
                 history_so_far: list[dict],
                 iter_idx: int, max_iter: int,
                 final_mse: float | None,
                 mean_iou_at_iter: float) -> Image.Image:
    """Render one GIF frame."""
    K = snapshot.shape[-1] - 1  # last channel is background
    K_gt = len(pair["masks1"])
    fig = plt.figure(figsize=(11.5, 6.4), dpi=100)
    gs = fig.add_gridspec(3, max(4, K + 1),
                          height_ratios=[1.0, 1.0, 0.55],
                          hspace=0.5, wspace=0.18)

    # Top row: frame1, frame2, GT flow, GT segmentation
    titles_top = ["frame 1", "frame 2", "GT flow", "GT segmentation"]
    contents = []
    contents.append(("gray", pair["frame1"], dict(vmin=0, vmax=1)))
    contents.append(("gray", pair["frame2"], dict(vmin=0, vmax=1)))
    contents.append(("rgb", flow_to_rgb(pair["flow"]), {}))
    gt_label = -np.ones(pair["frame1"].shape, dtype=np.int64)
    for s in range(K_gt):
        gt_label[pair["masks1"][s]] = s
    contents.append(("rgb", _segmentation_rgb(gt_label, K_gt), {}))
    for c in range(4):
        ax = fig.add_subplot(gs[0, c])
        kind, img, kw = contents[c]
        if kind == "gray":
            ax.imshow(img, cmap="gray", **kw)
        else:
            ax.imshow(img, **kw)
        ax.set_title(titles_top[c], fontsize=10)
        ax.set_xticks([]); ax.set_yticks([])

    # Middle row: capsule + background responsibility maps
    for k in range(K):
        ax = fig.add_subplot(gs[1, k])
        ax.imshow(snapshot[..., k], cmap="magma", vmin=0, vmax=1)
        ax.set_title(f"capsule {k}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    ax = fig.add_subplot(gs[1, K])
    ax.imshow(snapshot[..., K], cmap="magma", vmin=0, vmax=1)
    ax.set_title("background", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # Bottom row: MSE curve
    ax = fig.add_subplot(gs[2, :])
    iters = [h["iter"] for h in history_so_far]
    mses = [h["mse"] for h in history_so_far]
    ax.plot(iters, mses, "o-", color="#1f77b4", markersize=3, linewidth=1.2)
    ax.set_xlim(-0.5, max_iter - 0.5)
    if mses:
        ax.set_ylim(0, max(mses) * 1.1 + 1e-3)
    ax.set_xlabel("EM iteration", fontsize=9)
    ax.set_ylabel("flow recon MSE", fontsize=9)
    ax.grid(alpha=0.3)

    iou_str = (f"  mean IoU = {mean_iou_at_iter:.2f}"
               if mean_iou_at_iter is not None else "")
    fig.suptitle(f"Geo flow capsules — EM iter {iter_idx + 1}/{max_iter}"
                 f"  recon MSE = {mses[-1]:.3f}{iou_str}",
                 fontsize=12, y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=4)
    p.add_argument("--n-shapes", type=int, default=3)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--n-iters", type=int, default=24)
    p.add_argument("--sigma-flow", type=float, default=0.8)
    p.add_argument("--sigma-xy-init", type=float, default=14.0)
    p.add_argument("--max-translation", type=float, default=5.0)
    p.add_argument("--max-rotation", type=float, default=0.20)
    p.add_argument("--scale-jitter", type=float, default=0.10)
    p.add_argument("--fps", type=int, default=6)
    p.add_argument("--hold-final", type=int, default=10)
    p.add_argument("--out", type=str, default="geo_flow_capsules.gif")
    p.add_argument("--max-bytes", type=int, default=3_000_000)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)
    pair = generate_geo_pair(args.resolution, args.resolution,
                              args.n_shapes, rng=rng,
                              max_translation=args.max_translation,
                              max_rotation=args.max_rotation,
                              scale_jitter=args.scale_jitter)
    print(f"Running EM ({args.n_iters} iters) on seed {args.seed} ...")
    fit = _fit_flow_capsules_once(
        pair["flow"], K=args.n_shapes,
        n_iters=args.n_iters,
        sigma_flow=args.sigma_flow,
        sigma_xy_init=args.sigma_xy_init,
        rng=np.random.default_rng(args.seed),
        foreground_threshold=0.25,
    )
    final_ev = part_segmentation_iou(fit["responsibilities"], pair["masks1"])
    print(f"  final IoU = {final_ev['mean_iou']:.3f}, "
          f"per-shape = {[round(v, 3) for v in final_ev['per_shape_iou']]}")

    snapshots = fit["snapshots"]
    history = fit["history"]
    final_mse = history[-1]["mse"]

    print(f"Rendering {len(snapshots)} frames ...")
    frames = []
    for i, snap in enumerate(snapshots):
        if i == len(snapshots) - 1:
            iou_now = final_ev["mean_iou"]
        else:
            iou_now = None
        f = render_frame(pair, snap, history[: i + 1], i,
                         max_iter=len(snapshots),
                         final_mse=final_mse,
                         mean_iou_at_iter=iou_now)
        frames.append(f)

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out

    # Iteratively reduce size if needed.
    cur_frames = frames
    cur_size_mult = 1.0
    while True:
        cur_frames[0].save(out_path, save_all=True,
                            append_images=cur_frames[1:],
                            duration=duration_ms, loop=0, optimize=True)
        sz = os.path.getsize(out_path)
        print(f"  wrote {out_path}  ({len(cur_frames)} frames, "
              f"{sz/1024:.0f} KB)")
        if sz <= args.max_bytes:
            break
        # Resize to 90% and retry.
        cur_size_mult *= 0.85
        new_w = int(frames[0].width * cur_size_mult)
        new_h = int(frames[0].height * cur_size_mult)
        if new_w < 200:
            print("  cannot shrink further; giving up")
            break
        print(f"  size > {args.max_bytes/1024:.0f} KB; resizing to "
              f"{new_w}x{new_h}")
        cur_frames = [f.resize((new_w, new_h), Image.LANCZOS) for f in frames]
        if args.hold_final > 0:
            cur_frames = cur_frames + [cur_frames[-1]] * 0  # already added


if __name__ == "__main__":
    main()
