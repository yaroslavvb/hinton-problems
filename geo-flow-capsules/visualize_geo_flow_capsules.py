"""
Static visualizations for the Geo flow-capsule fit.

Outputs (in `viz/`):
  example_pairs.png         - frame1, frame2, GT flow, GT segmentation
  capsule_attention.png     - per-capsule responsibility maps for one pair
  segmentation_iou_bar.png  - per-shape IoU bar chart over the test set
  iou_distribution.png      - histogram of mean IoU across the test set
  em_convergence.png        - reconstruction MSE vs EM iteration
  results.json              - canonical run summary
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors

from geo_flow_capsules import (
    build_flow_capsule_net,
    env_info,
    fit_flow_capsules,
    generate_geo_pair,
    part_segmentation_iou,
)


# ----------------------------------------------------------------------
# Flow visualization (HSV: hue = direction, saturation/value = magnitude)
# ----------------------------------------------------------------------

def flow_to_rgb(flow: np.ndarray, max_mag: float | None = None) -> np.ndarray:
    """Convert (h, w, 2) flow to (h, w, 3) RGB using HSV color wheel."""
    h, w, _ = flow.shape
    fx = flow[..., 0]
    fy = flow[..., 1]
    mag = np.hypot(fx, fy)
    if max_mag is None:
        max_mag = max(mag.max(), 1e-6)
    hue = (np.arctan2(fy, fx) + np.pi) / (2.0 * np.pi)  # in [0, 1]
    sat = np.clip(mag / max_mag, 0.0, 1.0)
    val = np.ones_like(hue)
    hsv = np.stack([hue, sat, val], axis=-1)
    rgb = mcolors.hsv_to_rgb(hsv)
    return rgb


def _segmentation_rgb(label: np.ndarray, K: int) -> np.ndarray:
    """label is (h, w) int in {-1 (bg), 0..K-1}. Returns (h, w, 3) RGB."""
    cmap = plt.get_cmap("tab10")
    out = np.zeros((label.shape[0], label.shape[1], 3), dtype=np.float32)
    for k in range(K):
        out[label == k] = cmap(k)[:3]
    out[label == -1] = [0.05, 0.05, 0.05]
    return out


# ----------------------------------------------------------------------
# Plots
# ----------------------------------------------------------------------

def plot_example_pairs(pairs: list[dict], fits: list[dict], evals: list[dict],
                       out_path: str, n_examples: int = 4):
    """Grid of (frame1, frame2, GT flow, GT segmentation, predicted segmentation)."""
    n = min(n_examples, len(pairs))
    fig, axes = plt.subplots(n, 5, figsize=(11, 2.2 * n), dpi=140)
    if n == 1:
        axes = axes[None, :]
    col_titles = ["frame 1", "frame 2", "ground-truth flow",
                  "GT segmentation", "predicted (EM)"]
    max_mag = max(np.linalg.norm(p["flow"], axis=-1).max() for p in pairs[:n])
    K_gt = len(pairs[0]["masks1"])
    for r in range(n):
        d = pairs[r]
        fit = fits[r]
        ev = evals[r]
        axes[r, 0].imshow(d["frame1"], cmap="gray", vmin=0, vmax=1)
        axes[r, 1].imshow(d["frame2"], cmap="gray", vmin=0, vmax=1)
        axes[r, 2].imshow(flow_to_rgb(d["flow"], max_mag=max_mag))
        # Ground-truth segmentation
        gt_label = -np.ones(d["frame1"].shape, dtype=np.int64)
        for s in range(K_gt):
            gt_label[d["masks1"][s]] = s
        axes[r, 3].imshow(_segmentation_rgb(gt_label, K_gt))
        # Predicted: remap via pairs to align colors with GT shapes.
        pred_label = ev["pred_label"]
        remap = -np.ones(d["frame1"].shape, dtype=np.int64)
        for p_idx, g_idx in ev["pairs"]:
            remap[pred_label == p_idx] = g_idx
        axes[r, 4].imshow(_segmentation_rgb(remap, K_gt))
        axes[r, 4].set_xlabel(f"IoU = {ev['mean_iou']:.2f}",
                              fontsize=9)
        for c in range(5):
            axes[r, c].set_xticks([])
            axes[r, c].set_yticks([])
        axes[r, 0].set_ylabel(f"pair {r}", rotation=0, ha="right",
                              labelpad=24, fontsize=10)
    for c in range(5):
        axes[0, c].set_title(col_titles[c], fontsize=10, pad=4)
    fig.suptitle("Geo frame pairs and EM-fit flow capsule segmentation",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_capsule_attention(pair: dict, fit: dict, ev: dict, out_path: str):
    """Per-capsule responsibility maps for one example pair."""
    K = fit["K"]
    resp = fit["responsibilities"]  # (h, w, K+1)
    fig, axes = plt.subplots(2, K + 1, figsize=(2.4 * (K + 1), 4.8), dpi=140)
    axes[0, 0].imshow(pair["frame1"], cmap="gray", vmin=0, vmax=1)
    axes[0, 0].set_title("frame 1", fontsize=10)
    axes[0, 1].imshow(pair["frame2"], cmap="gray", vmin=0, vmax=1)
    axes[0, 1].set_title("frame 2", fontsize=10)
    axes[0, 2].imshow(flow_to_rgb(pair["flow"]))
    axes[0, 2].set_title("GT flow", fontsize=10)
    if K + 1 > 3:
        # GT segmentation in the remaining slot
        gt_label = -np.ones(pair["frame1"].shape, dtype=np.int64)
        for s in range(len(pair["masks1"])):
            gt_label[pair["masks1"][s]] = s
        axes[0, 3].imshow(_segmentation_rgb(gt_label, len(pair["masks1"])))
        axes[0, 3].set_title("GT segmentation", fontsize=10)
    for c in range(K + 1):
        axes[0, c].set_xticks([])
        axes[0, c].set_yticks([])

    # Bottom row: per-capsule + background responsibility
    for k in range(K):
        axes[1, k].imshow(resp[..., k], cmap="magma", vmin=0, vmax=1)
        # Annotate with capsule->shape match if available
        match = next((g for p, g in ev["pairs"] if p == k), None)
        match_str = f" -> shape {match}" if match is not None else ""
        axes[1, k].set_title(f"capsule {k}{match_str}", fontsize=9)
        axes[1, k].set_xticks([]); axes[1, k].set_yticks([])
    axes[1, K].imshow(resp[..., K], cmap="magma", vmin=0, vmax=1)
    axes[1, K].set_title("background", fontsize=9)
    axes[1, K].set_xticks([]); axes[1, K].set_yticks([])

    fig.suptitle(f"Per-capsule responsibility (mean IoU = {ev['mean_iou']:.2f})",
                 fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_segmentation_iou_bar(per_shape_iou: list[list[float]],
                              out_path: str):
    """Per-shape mean IoU bars, plus error bars across pairs."""
    arr = np.array(per_shape_iou)  # (n_pairs, K)
    K = arr.shape[1]
    means = arr.mean(axis=0)
    stds = arr.std(axis=0)
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
    xs = np.arange(K)
    bars = ax.bar(xs, means, yerr=stds, capsize=4,
                  color=[plt.get_cmap("tab10")(k) for k in range(K)],
                  alpha=0.85)
    for k, m in enumerate(means):
        ax.text(k, m + 0.02, f"{m:.2f}", ha="center", fontsize=10)
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="target")
    chance_iou = 1.0 / (2 * K - 1)
    ax.axhline(chance_iou, color="red", linestyle=":", linewidth=0.8,
               label=f"~chance ({chance_iou:.2f})")
    ax.set_xticks(xs)
    ax.set_xticklabels([f"shape {k}" for k in range(K)])
    ax.set_ylabel("IoU")
    ax.set_ylim(0, 1.05)
    ax.set_title(f"Per-shape segmentation IoU "
                 f"(mean over {arr.shape[0]} test pairs)")
    ax.legend(loc="lower right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_iou_distribution(per_pair_iou: list[float], out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.4), dpi=140)
    arr = np.array(per_pair_iou)
    ax.hist(arr, bins=30, color="#1f77b4", alpha=0.85, edgecolor="white")
    ax.axvline(arr.mean(), color="black", linestyle="-", linewidth=1.2,
               label=f"mean = {arr.mean():.3f}")
    ax.axvline(0.5, color="gray", linestyle="--", linewidth=0.8,
               label="target")
    ax.set_xlabel("mean per-pair IoU (over 3 shapes)")
    ax.set_ylabel("count")
    ax.set_title(f"Distribution of mean IoU across {len(arr)} test pairs")
    ax.legend(fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_em_convergence(fit: dict, out_path: str):
    fig, ax = plt.subplots(figsize=(5.5, 3.2), dpi=140)
    iters = [h["iter"] for h in fit["history"]]
    mses = [h["mse"] for h in fit["history"]]
    ax.plot(iters, mses, "o-", color="#1f77b4", markersize=3)
    ax.set_xlabel("EM iteration")
    ax.set_ylabel("flow reconstruction MSE")
    ax.set_title("EM convergence on a single Geo pair")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-shapes", type=int, default=3)
    p.add_argument("--resolution", type=int, default=64)
    p.add_argument("--n-test", type=int, default=120)
    p.add_argument("--n-iters", type=int, default=30)
    p.add_argument("--n-restarts", type=int, default=3)
    p.add_argument("--sigma-flow", type=float, default=0.8)
    p.add_argument("--sigma-xy-init", type=float, default=14.0)
    p.add_argument("--max-translation", type=float, default=5.0)
    p.add_argument("--max-rotation", type=float, default=0.20)
    p.add_argument("--scale-jitter", type=float, default=0.10)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    print(f"Generating {args.n_test} test pairs (seed={args.seed}) ...")
    test = []
    for _ in range(args.n_test):
        test.append(generate_geo_pair(
            args.resolution, args.resolution,
            args.n_shapes, rng=rng,
            max_translation=args.max_translation,
            max_rotation=args.max_rotation,
            scale_jitter=args.scale_jitter,
        ))

    model = build_flow_capsule_net(
        K=args.n_shapes,
        n_iters=args.n_iters,
        sigma_flow=args.sigma_flow,
        sigma_xy_init=args.sigma_xy_init,
        n_restarts=args.n_restarts,
    )

    print(f"Fitting {args.n_test} flow capsule decompositions ...")
    fits = []
    evals = []
    for d in test:
        sub_rng = np.random.default_rng(int(rng.integers(2**31 - 1)))
        fit = fit_flow_capsules(
            d["flow"], K=args.n_shapes,
            n_iters=args.n_iters,
            sigma_flow=args.sigma_flow,
            sigma_xy_init=args.sigma_xy_init,
            n_restarts=args.n_restarts,
            rng=sub_rng,
        )
        ev = part_segmentation_iou(fit["responsibilities"], d["masks1"])
        fits.append(fit)
        evals.append(ev)

    per_pair_iou = [ev["mean_iou"] for ev in evals]
    per_shape_iou = [ev["per_shape_iou"] for ev in evals]
    per_shape_arr = np.array(per_shape_iou)
    print(f"  mean IoU: {np.mean(per_pair_iou):.3f}")
    print(f"  per-shape IoU mean: "
          f"{[round(v, 3) for v in per_shape_arr.mean(axis=0).tolist()]}")

    # Pick example pairs that span the IoU distribution (best-2, median, worst-1)
    order = np.argsort(per_pair_iou)
    pick_idx = [int(order[-1]), int(order[len(order) * 3 // 4]),
                int(order[len(order) // 2]), int(order[0])]
    pick_pairs = [test[i] for i in pick_idx]
    pick_fits = [fits[i] for i in pick_idx]
    pick_evals = [evals[i] for i in pick_idx]

    plot_example_pairs(pick_pairs, pick_fits, pick_evals,
                       os.path.join(args.outdir, "example_pairs.png"),
                       n_examples=len(pick_idx))
    plot_capsule_attention(pick_pairs[0], pick_fits[0], pick_evals[0],
                            os.path.join(args.outdir, "capsule_attention.png"))
    plot_segmentation_iou_bar(per_shape_iou,
                               os.path.join(args.outdir,
                                            "segmentation_iou_bar.png"))
    plot_iou_distribution(per_pair_iou,
                           os.path.join(args.outdir, "iou_distribution.png"))
    plot_em_convergence(pick_fits[0],
                         os.path.join(args.outdir, "em_convergence.png"))

    summary = dict(
        config=vars(args),
        mean_iou=float(np.mean(per_pair_iou)),
        median_iou=float(np.median(per_pair_iou)),
        per_shape_iou_mean=per_shape_arr.mean(axis=0).tolist(),
        per_shape_iou_std=per_shape_arr.std(axis=0).tolist(),
        n_pairs=args.n_test,
        env=env_info(),
    )
    with open(os.path.join(args.outdir, "results.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(f"  wrote {os.path.join(args.outdir, 'results.json')}")


if __name__ == "__main__":
    main()
