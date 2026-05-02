"""
Render an animated GIF showing the family-trees network learning, with
the 6-D person encoding's interpretable axes emerging over training.

Layout per frame:
  Top:    PCA scatter of the 24 people, colored by nationality.
  Middle: Heatmap of the 6-unit person encoding (24 rows sorted by
          nationality / generation, 6 columns).
  Bottom: Train / test accuracy curves up to the current epoch.

Usage:
    python3 make_family_trees_gif.py
    python3 make_family_trees_gif.py --epochs 10000 --snapshot-every 200 --fps 10
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from family_trees import (
    FamilyTreeMLP, ALL_PEOPLE,
    build_triples, aggregate_facts, split_train_test,
    facts_to_arrays, train, inspect_person_encoding, attribute_table,
)


NATIONALITY_COLORS = {0: "#1f77b4", 1: "#d62728"}
GEN_MARKERS = {1: "o", 2: "s", 3: "D"}


def _pca_2d(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = codes.mean(axis=0, keepdims=True)
    centered = codes - mean
    U, s, Vt = np.linalg.svd(centered, full_matrices=False)
    proj = centered @ Vt.T[:, :2]
    return proj, Vt


def render_frame(history: dict, epoch: int, codes: np.ndarray,
                 attrs: dict) -> Image.Image:
    fig = plt.figure(figsize=(9.5, 7.5), dpi=85)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.4, 1.0],
                          width_ratios=[1.2, 1.0],
                          hspace=0.45, wspace=0.30)

    # ---- top-left: PCA scatter ----
    ax = fig.add_subplot(gs[0, 0])
    proj, _ = _pca_2d(codes)
    nat = attrs["nationality"]
    gen = attrs["generation"]
    for k_nat in (0, 1):
        for k_gen in (1, 2, 3):
            mask = (nat == k_nat) & (gen == k_gen)
            if not mask.any():
                continue
            ax.scatter(proj[mask, 0], proj[mask, 1],
                       c=NATIONALITY_COLORS[k_nat],
                       marker=GEN_MARKERS[k_gen],
                       s=110, edgecolors="black", linewidths=0.5)
    for i, (px, py) in enumerate(proj):
        ax.annotate(ALL_PEOPLE[i][:3], (px, py), xytext=(0, 0),
                    textcoords="offset points", ha="center", va="center",
                    fontsize=6.5, color="white", weight="bold")
    ax.set_title("PCA of 6-D person encoding "
                 "(blue=Eng, red=Ita; o/s/D = gen 1/2/3)", fontsize=10)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    ax.grid(alpha=0.3)
    # Fix axes so the scatter doesn't jitter frame-to-frame
    lim = max(abs(proj).max(), 1.0) * 1.15
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_aspect("equal")

    # ---- top-right: encoding heatmap ----
    ax = fig.add_subplot(gs[0, 1])
    order = np.lexsort((attrs["branch"], attrs["generation"], attrs["nationality"]))
    sorted_codes = codes[order]
    im = ax.imshow(sorted_codes, aspect="auto", cmap="RdBu_r",
                   vmin=-1, vmax=1)
    ax.set_xticks(range(6))
    ax.set_xticklabels([f"u{j}" for j in range(6)], fontsize=8)
    ax.set_yticks(range(24))
    ax.set_yticklabels([ALL_PEOPLE[i][:6] for i in order], fontsize=6)
    nat_sorted = attrs["nationality"][order]
    for boundary in np.where(np.diff(nat_sorted) != 0)[0]:
        ax.axhline(boundary + 0.5, color="black", linewidth=0.6)
    fig.colorbar(im, ax=ax, fraction=0.04, pad=0.02)
    ax.set_title("6-unit person encoding\n(rows sorted by nat/gen/branch)",
                 fontsize=10)

    # ---- bottom: training curves ----
    ax = fig.add_subplot(gs[1, :])
    if history["epoch"]:
        ax.plot(history["epoch"], np.array(history["train_acc"]) * 100,
                color="#1f77b4", linewidth=1.5, label="train accuracy")
        ax.plot(history["epoch"], np.array(history["test_acc"]) * 100,
                color="#d62728", linewidth=1.5, label="test accuracy")
        ax.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.4)
    final_x = history["epoch"][-1] if history["epoch"] else 1
    ax.set_xlim(0, final_x)
    ax.set_ylim(0, 105)
    ax.set_xlabel("epoch", fontsize=9)
    ax.set_ylabel("accuracy (%)", fontsize=9)
    ax.legend(loc="lower right", fontsize=8, framealpha=0.95)
    ax.grid(alpha=0.3)

    fig.suptitle(f"Family trees — epoch {epoch + 1} "
                 f"(Hinton 1986; backprop, tanh hidden, softmax CE)",
                 fontsize=11, y=0.99)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=85, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=10000)
    p.add_argument("--snapshot-every", type=int, default=250)
    p.add_argument("--fps", type=int, default=10)
    p.add_argument("--seed", type=int, default=6)
    p.add_argument("--out", type=str, default="family_trees.gif")
    p.add_argument("--hold-final", type=int, default=20,
                   help="Repeat the last frame this many times.")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--init-scale", type=float, default=1.0)
    args = p.parse_args()

    triples = build_triples()
    facts = aggregate_facts(triples)
    train_facts, test_facts = split_train_test(facts, n_test=4, seed=args.seed)
    X_train, Y_train = facts_to_arrays(train_facts)
    X_test, Y_test = facts_to_arrays(test_facts)
    attrs = attribute_table()

    frames: list[Image.Image] = []

    def cb(epoch, model, history):
        codes = inspect_person_encoding(model)
        frame = render_frame(history, epoch, codes, attrs)
        frames.append(frame)
        if len(frames) % 5 == 0:
            print(f"  frame {len(frames):3d}  epoch {epoch + 1:5d}  "
                  f"train={history['train_acc'][-1]*100:.0f}%  "
                  f"test={history['test_acc'][-1]*100:.0f}%")

    print(f"Training {args.epochs} epochs, snapshot every {args.snapshot_every}...")
    model = FamilyTreeMLP(seed=args.seed, init_scale=args.init_scale)
    history = train(model, X_train, Y_train, X_test, Y_test,
                    n_sweeps=args.epochs, lr=args.lr,
                    momentum=0.9, weight_decay=0.0,
                    snapshot_callback=cb,
                    snapshot_every=args.snapshot_every,
                    verbose=False)

    print(f"\nFinal: train={history['train_acc'][-1]*100:.0f}%  "
          f"test={history['test_acc'][-1]*100:.0f}%   "
          f"({len(frames)} frames)")

    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(out_path, save_all=True, append_images=frames[1:],
                   duration=duration_ms, loop=0, optimize=True)
    size_kb = os.path.getsize(out_path) / 1024
    print(f"Wrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")


if __name__ == "__main__":
    main()
