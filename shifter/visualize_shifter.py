"""
Visualize what the shifter RBM has learned.

Outputs (all under `--outdir`):
    figure3.png             — Hinton-diagram weight viz of the 24 hidden units
                               (matches the layout convention of Fig 3 from
                               Hinton & Sejnowski 1986). Each panel: top-left
                               threshold + top-right [L,N,R] output weights;
                               below them V1 row and V2 row, with white = +,
                               black = -, square area proportional to |w|.
    weights_heatmap.png     — full visible x hidden weight matrix as heatmap.
    training_curves.png     — recon-MSE and accuracy vs epoch.
    accuracy_vs_v1_bits.png — recognition accuracy as a function of how many
                               V1 bits are on (the paper reports 50-89%
                               varying with this).
    confusion.png           — 3x3 confusion matrix on the full 768 cases.

The Hinton diagram is the headline figure — it reveals the position-pair
detectors. A unit preferring "shift left" tends to show a strong
white-on-white pair at V1[i] and V2[(i-1) mod N], plus a strong white
output square in the L column.
"""

from __future__ import annotations
import argparse
import os
import time

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Rectangle

import shifter as sh


# ----------------------------------------------------------------------
# Hinton-diagram (Figure 3 reproduction)
# ----------------------------------------------------------------------

def hinton_unit_panel(ax, w_v1, w_v2, w_y, w_thresh, vmax,
                      cell: float = 1.0):
    """Draw one unit's weights in PDP Fig-3 style.

    Layout (rows top-to-bottom):
        row 0: [thresh] ........ [y_L y_N y_R]
        row 1: blank
        row 2: V1 weights (N cells)
        row 3: V2 weights (N cells)

    White square = positive weight, black = negative, side proportional
    to |w| / vmax. Background mid-gray to match the original look.
    """
    N = len(w_v1)
    ncols = max(N, 7)
    nrows = 4

    ax.add_patch(Rectangle((0, 0), ncols, nrows, facecolor="#c8c8c8",
                            edgecolor="black", lw=0.7, zorder=0))

    def draw_cell(col, row, w):
        if np.isnan(w) or w == 0:
            return
        s = min(0.95, abs(w) / vmax) * cell
        cx = col + 0.5
        cy = (nrows - 1 - row) + 0.5
        color = "white" if w > 0 else "black"
        ax.add_patch(Rectangle((cx - s / 2, cy - s / 2), s, s,
                                facecolor=color, edgecolor="none",
                                zorder=2))

    # threshold (top-left)
    draw_cell(0, 0, w_thresh)
    # output trio (top-right)
    y_start = ncols - 3
    for k, w in enumerate(w_y):
        draw_cell(y_start + k, 0, w)
    # V1 row
    for i, w in enumerate(w_v1):
        draw_cell(i, 2, w)
    # V2 row
    for i, w in enumerate(w_v2):
        draw_cell(i, 3, w)

    # thin separators
    ax.plot([0, ncols], [nrows - 1, nrows - 1], color="black", lw=0.3,
            zorder=1)
    ax.plot([0, ncols], [2, 2], color="black", lw=0.3, zorder=1)

    ax.set_xlim(0, ncols)
    ax.set_ylim(0, nrows)
    ax.set_aspect("equal")
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def render_figure3(rbm: sh.ShifterRBM, N: int, grid=(4, 6),
                    out_path: str = "figure3.png") -> tuple[list[int], np.ndarray]:
    """Render all 24 hidden units in a 4-row x 6-col grid (Fig 3 layout)."""
    if rbm.nh != grid[0] * grid[1]:
        raise ValueError(f"rbm.nh={rbm.nh} != grid {grid[0]*grid[1]}")

    W = rbm.W
    bh = rbm.bh
    nv_y = 3
    W_v1 = W[:N, :]
    W_v2 = W[N:2 * N, :]
    W_y = W[2 * N:2 * N + nv_y, :]

    pref = np.argmax(W_y, axis=0)
    strength = W_y.max(axis=0) - W_y.min(axis=0)
    order = sorted(range(rbm.nh), key=lambda j: (pref[j], -strength[j]))

    # Use 97th percentile of |w| so small weights remain visible
    vmax = float(np.percentile(
        np.abs(np.concatenate([W.ravel(), bh.ravel()])), 97))

    rows, cols = grid
    fig, axes = plt.subplots(rows, cols,
                              figsize=(cols * 1.7, rows * 1.3))
    fig.suptitle(
        "Shifter hidden-unit weights (24 units, Hinton-diagram style)\n"
        "top-left: threshold   .   top-right trio: [L, N, R] output weights\n"
        "bottom two rows: V1 and V2 receptive fields   .   "
        "white = +, black = -, side proportional to |w|",
        fontsize=9,
    )

    pref_label = {0: "L", 1: "N", 2: "R"}
    for k, j in enumerate(order):
        ax = axes.flat[k]
        hinton_unit_panel(ax,
                          w_v1=W_v1[:, j], w_v2=W_v2[:, j],
                          w_y=W_y[:, j], w_thresh=bh[j], vmax=vmax)
        ax.set_title(f"unit {j} -> {pref_label[pref[j]]}",
                     fontsize=7, pad=2)

    plt.subplots_adjust(left=0.02, right=0.98, top=0.86, bottom=0.02,
                        wspace=0.15, hspace=0.3)
    plt.savefig(out_path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {out_path}")

    # report the most interpretable position-pair detector
    best = None
    for j in range(rbm.nh):
        v1w = W_v1[:, j]
        v2w = W_v2[:, j]
        score_pos = np.outer(np.maximum(v1w, 0), np.maximum(v2w, 0))
        score_neg = np.outer(-np.minimum(v1w, 0), -np.minimum(v2w, 0))
        score = np.maximum(score_pos, score_neg)
        idx = np.unravel_index(np.argmax(score), score.shape)
        if best is None or score[idx] > best[2]:
            best = (j, idx, score[idx])
    j, (i1, i2), _ = best
    offset = (i2 - i1) % N
    if offset == N - 1:
        slabel = "left (-1)"
    elif offset == 1:
        slabel = "right (+1)"
    elif offset == 0:
        slabel = "none (0)"
    else:
        slabel = f"offset +{offset}"
    print(f"\nMost interpretable position-pair detector: unit {j}")
    print(f"  Strongest V1<->V2 pair: V1[{i1}] <-> V2[{i2}]  "
          f"(offset {offset}, consistent with shift-{slabel})")
    print(f"  Output preference:     L={W_y[0,j]:+.2f}  "
          f"N={W_y[1,j]:+.2f}  R={W_y[2,j]:+.2f}")
    print(f"  Threshold:             {bh[j]:+.2f}")
    return order, pref


# ----------------------------------------------------------------------
# Heatmap of full visible x hidden weight matrix
# ----------------------------------------------------------------------

def plot_weights_heatmap(rbm: sh.ShifterRBM, N: int,
                         order: list[int],
                         out_path: str = "weights_heatmap.png"):
    W = rbm.W[:, order]
    nv = W.shape[0]
    fig, ax = plt.subplots(figsize=(max(8, rbm.nh * 0.22), 6))
    vmax = float(np.abs(W).max())
    im = ax.imshow(W, cmap="RdBu_r", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.axhline(N - 0.5, color="black", lw=1)
    ax.axhline(2 * N - 0.5, color="black", lw=1)
    ax.set_yticks(range(nv))
    ax.set_yticklabels([f"v1[{i}]" for i in range(N)] +
                       [f"v2[{i}]" for i in range(N)] +
                       ["v3_left", "v3_none", "v3_right"], fontsize=8)
    ax.set_xlabel("hidden unit (sorted by preferred shift)")
    ax.set_ylabel("visible unit")
    ax.set_title("Shifter RBM weight matrix W (visible x hidden)")
    plt.colorbar(im, ax=ax, label="weight")
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, out_path: str = "training_curves.png"):
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    axes[0].plot(history["epoch"], history["recon_mse"],
                 color="#3a78a8", lw=1.5)
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("reconstruction MSE (visible)")
    axes[0].set_title("Reconstruction error")
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["eval_epoch"],
                 [a * 100 for a in history["acc"]],
                 marker="o", color="#a83232", lw=1.5)
    axes[1].axhline(33.3, color="grey", lw=0.7, ls="--",
                    label="chance (33.3%)")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("recognition accuracy (%)")
    axes[1].set_title("Shift recognition accuracy")
    axes[1].set_ylim(0, 100)
    axes[1].grid(alpha=0.3)
    axes[1].legend(loc="lower right", fontsize=8)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Accuracy vs number of on-bits in V1
# ----------------------------------------------------------------------

def plot_accuracy_vs_v1_bits(rbm: sh.ShifterRBM, N: int,
                              out_path: str = "accuracy_vs_v1_bits.png",
                              n_gibbs: int = 150):
    buckets = sh.accuracy_vs_v1_activity(rbm, N=N, n_gibbs=n_gibbs)
    ks = list(buckets.keys())
    accs = [c / t * 100 for (c, t) in buckets.values()]
    totals = [t for (c, t) in buckets.values()]

    fig, ax = plt.subplots(figsize=(7, 3.6))
    bars = ax.bar(ks, accs, color="#3a78a8", edgecolor="#1f3f5e", alpha=0.85)
    for b, t in zip(bars, totals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1,
                f"n={t}", ha="center", va="bottom", fontsize=8,
                color="#444")
    ax.axhline(33.3, color="grey", lw=0.7, ls="--", label="chance")
    ax.axhspan(50, 89, color="#3a78a8", alpha=0.10,
               label="paper range (50-89%)")
    ax.set_xlabel("number of V1 bits on (k)")
    ax.set_ylabel("recognition accuracy (%)")
    ax.set_title("Shifter accuracy vs V1 activity\n"
                 "(per the original paper: 50-89% across V1 activity)")
    ax.set_ylim(0, 100)
    ax.set_xticks(ks)
    ax.grid(alpha=0.3, axis="y")
    ax.legend(loc="lower center", fontsize=8)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}")


# ----------------------------------------------------------------------
# Confusion matrix
# ----------------------------------------------------------------------

def plot_confusion(rbm: sh.ShifterRBM, N: int,
                   out_path: str = "confusion.png",
                   n_gibbs: int = 200):
    V1, V2, V3 = sh.make_shifter_data(N)
    clamp_mask = np.concatenate([np.ones(2 * N), np.zeros(3)]).astype(np.float32)
    cm = np.zeros((3, 3), dtype=int)
    for i in range(V1.shape[0]):
        v_init = np.concatenate([V1[i], V2[i],
                                 np.zeros(3, dtype=np.float32)])
        v_mean = rbm.conditional_fill(v_init, clamp_mask, n_gibbs=n_gibbs)
        pred = int(np.argmax(v_mean[2 * N:]))
        true = int(np.argmax(V3[i]))
        cm[true, pred] += 1
    classes = ["left (-1)", "none (0)", "right (+1)"]
    fig, ax = plt.subplots(figsize=(4.5, 4))
    im = ax.imshow(cm, cmap="Blues")
    for i in range(3):
        for j in range(3):
            color = "white" if cm[i, j] > cm.max() / 2 else "black"
            ax.text(j, i, str(cm[i, j]), ha="center", va="center",
                    color=color)
    ax.set_xticks(range(3)); ax.set_xticklabels(classes)
    ax.set_yticks(range(3)); ax.set_yticklabels(classes)
    ax.set_xlabel("predicted shift"); ax.set_ylabel("true shift")
    overall = np.trace(cm) / cm.sum()
    ax.set_title(f"Shifter confusion matrix\noverall acc = {overall*100:.1f}%")
    plt.colorbar(im, ax=ax, fraction=0.046)
    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path}  ({cm.sum()} cases, acc {overall*100:.1f}%)")


# ----------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--outdir", default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"# Training shifter RBM: N={args.N}, hidden={args.hidden}, "
          f"epochs={args.epochs}, seed={args.seed}")
    history: dict = {}
    t0 = time.time()
    rbm = sh.train(N=args.N, n_hidden=args.hidden, n_epochs=args.epochs,
                   lr=args.lr, momentum=args.momentum,
                   batch_size=args.batch, seed=args.seed,
                   eval_every=10, eval_gibbs=80, history=history,
                   verbose=False)
    print(f"# train took {time.time()-t0:.1f}s")

    order, pref = render_figure3(rbm, args.N, grid=(4, 6),
                                 out_path=os.path.join(args.outdir,
                                                       "figure3.png"))
    plot_weights_heatmap(rbm, args.N, order,
                         out_path=os.path.join(args.outdir,
                                               "weights_heatmap.png"))
    plot_training_curves(history,
                         out_path=os.path.join(args.outdir,
                                               "training_curves.png"))
    plot_accuracy_vs_v1_bits(rbm, args.N,
                             out_path=os.path.join(args.outdir,
                                                   "accuracy_vs_v1_bits.png"))
    plot_confusion(rbm, args.N,
                   out_path=os.path.join(args.outdir, "confusion.png"))

    np.savez(os.path.join(args.outdir, "rbm_weights.npz"),
             W=rbm.W, bv=rbm.bv, bh=rbm.bh)
    print(f"saved {args.outdir}/rbm_weights.npz")

    classes = ["left", "none", "right"]
    counts = np.bincount(pref, minlength=3)
    print("\nHidden-unit shift preference distribution:")
    for k, c in enumerate(classes):
        print(f"  {c:5s}: {counts[k]} units")
