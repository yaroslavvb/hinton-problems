"""
Static visualizations for the trained gated three-way RBM
(transforming-pairs).

Outputs (in `viz/`):
  example_pairs.png         - 4x4 grid of (x, y) example pairs with labels
  filter_pairs.png          - top factors as input/output filter image pairs
  transformation_profile.png - per-transform mean hidden activation heatmap
  training_curves.png       - recon MSE, transform classification accuracy
  transfer_examples.png     - transformation-transfer demo
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from transforming_pairs import (
    GatedRBM,
    build_gated_rbm,
    generate_transformed_pairs,
    per_unit_transform_profile,
    reconstruction_metrics,
    train,
    transform_classification_accuracy,
    transform_label,
    transform_specificity_score,
    visualize_transformation_filters,
)

H_W = 13


# ----------------------------------------------------------------------
# helpers
# ----------------------------------------------------------------------

def _imshow_pair(ax_x, ax_y, x_img, y_img, *, cmap="gray_r",
                 vmin=None, vmax=None):
    ax_x.imshow(x_img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax_y.imshow(y_img, cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    for ax in (ax_x, ax_y):
        ax.set_xticks([])
        ax.set_yticks([])


# ----------------------------------------------------------------------
# example pairs
# ----------------------------------------------------------------------

def plot_example_pairs(X, Y, ids, pool, out_path: str, n_pairs: int = 8):
    """Show 8 (x, y) example pairs labeled with their transformation."""
    fig, axes = plt.subplots(2, n_pairs, figsize=(1.3 * n_pairs, 2.8), dpi=140)
    for k in range(n_pairs):
        i = k
        x = X[i].reshape(H_W, H_W)
        y = Y[i].reshape(H_W, H_W)
        _imshow_pair(axes[0, k], axes[1, k], x, y, vmin=0, vmax=1)
        axes[0, k].set_title(transform_label(pool[int(ids[i])]),
                             fontsize=9, pad=2)
    axes[0, 0].set_ylabel("$x$", rotation=0, labelpad=12, fontsize=11)
    axes[1, 0].set_ylabel("$y$", rotation=0, labelpad=12, fontsize=11)
    fig.suptitle("Example pairs $(x, y)$ where $y = \\mathrm{transform}(x)$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# factor filter pairs
# ----------------------------------------------------------------------

def plot_filter_pairs(model: GatedRBM, out_path: str, n_top: int = 16):
    """Top factors visualized as (W^x_f, W^y_f) image pairs."""
    factors, Wx_imgs, Wy_imgs = visualize_transformation_filters(
        model, h_w=H_W, n_top=n_top)
    n_cols = 8
    n_rows = max(1, (n_top + n_cols - 1) // n_cols)
    fig, axes = plt.subplots(2 * n_rows, n_cols,
                             figsize=(1.4 * n_cols, 1.5 * 2 * n_rows),
                             dpi=140)
    if 2 * n_rows == 1:
        axes = axes[None, :]
    vmax = float(max(np.abs(Wx_imgs).max(), np.abs(Wy_imgs).max(), 1e-6))
    for k in range(n_top):
        r = k // n_cols
        c = k % n_cols
        ax_x = axes[2 * r, c]
        ax_y = axes[2 * r + 1, c]
        ax_x.imshow(Wx_imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
        ax_y.imshow(Wy_imgs[k], cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                    interpolation="nearest")
        for ax in (ax_x, ax_y):
            ax.set_xticks([])
            ax.set_yticks([])
        ax_x.set_title(f"f{int(factors[k])}", fontsize=8, pad=2)
    # Hide unused panels
    for k in range(n_top, n_rows * n_cols):
        r = k // n_cols
        c = k % n_cols
        axes[2 * r, c].axis("off")
        axes[2 * r + 1, c].axis("off")
    for r in range(n_rows):
        axes[2 * r, 0].set_ylabel("$W^x_f$", rotation=0, labelpad=15,
                                  fontsize=10)
        axes[2 * r + 1, 0].set_ylabel("$W^y_f$", rotation=0, labelpad=15,
                                      fontsize=10)
    fig.suptitle("Top factor filter pairs  $(W^x_f \\rightarrow W^y_f)$",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# per-transform profile
# ----------------------------------------------------------------------

def plot_transformation_profile(profile: np.ndarray, pool: list, out_path: str,
                                  min_specificity: float = 0.5):
    """Heatmap with rows = transformations, cols = hidden units (sorted).

    Drops "always on" / "always off" units (specificity below
    `min_specificity`) so the structure is visible. The fraction kept is
    reported in the title.
    """
    n_t, n_h = profile.shape
    eps = 1e-6
    means = profile.mean(axis=0) + eps
    maxes = profile.max(axis=0)
    spec = (maxes - means) / means
    # Keep units that respond peakily to one transform AND fire enough to be
    # visible in the heatmap.
    keep = (spec >= min_specificity) & (maxes >= 0.05)
    if keep.sum() < 4:
        # Fallback: take the 16 most selective units.
        keep_idx = np.argsort(-spec)[:min(16, n_h)]
    else:
        keep_idx = np.where(keep)[0]
    profile_k = profile[:, keep_idx]
    spec_k = spec[keep_idx]
    argmax_k = np.argmax(profile_k, axis=0)
    # Sort kept units by preferred transform, then by peak height.
    order = np.lexsort((-profile_k.max(axis=0), argmax_k))
    profile_s = profile_k[:, order]

    fig, ax = plt.subplots(figsize=(max(6, 0.22 * len(order) + 2),
                                    0.45 * n_t + 1.0),
                           dpi=140)
    im = ax.imshow(profile_s, aspect="auto", cmap="viridis",
                   interpolation="nearest")
    ax.set_yticks(range(n_t))
    ax.set_yticklabels([transform_label(t) for t in pool], fontsize=8)
    ax.set_xlabel(f"selective hidden unit (sorted by preferred transform)  "
                  f"-- {len(order)} of {n_h} shown")
    ax.set_ylabel("transformation")
    ax.set_title("Mean hidden activation per (transform, unit)  "
                 f"[median specificity = {np.median(spec):.2f}, "
                 f"shown median = {np.median(spec_k):.2f}]")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# training curves
# ----------------------------------------------------------------------

def plot_training_curves(history: dict, n_classes: int, out_path: str):
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.4), dpi=140)
    e = history["epoch"]

    ax = axes[0]
    ax.plot(e, history["recon_mse"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("training MSE")
    ax.set_title("Reconstruction MSE")
    ax.grid(alpha=0.3)

    ax = axes[1]
    if "transform_acc" in history:
        eval_e = [ee for ee, v in zip(e, history["transform_acc"])
                  if v is not None]
        eval_v = [v * 100 for v in history["transform_acc"] if v is not None]
        ax.plot(eval_e, eval_v, "o-", color="#2ca02c", markersize=4)
        ax.axhline(100.0 / n_classes, color="gray", linestyle="--",
                   linewidth=0.8, label="chance")
        ax.legend(fontsize=8)
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.set_title("Transformation classification (held-out)")
    ax.set_ylim(0, 105)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(e, history["weight_norm"], color="#ff7f0e")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$\|W^x\|_F + \|W^y\|_F + \|W^h\|_F$")
    ax.set_title("Combined weight norm")
    ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# transformation transfer
# ----------------------------------------------------------------------

def plot_transfer_examples(model: GatedRBM, X, Y, ids, pool,
                            out_path: str, rng: np.random.Generator,
                            n_examples: int = 5):
    """Pick `n_examples` distinct transforms; for each, take a (x_ref, y_ref)
    pair, then show several (x_query, predicted y_query, true y_query)."""
    distinct_transforms = list(set(int(t) for t in ids))[:n_examples]
    fig, axes = plt.subplots(n_examples, 7,
                             figsize=(11, 1.6 * n_examples), dpi=140)
    if n_examples == 1:
        axes = axes[None, :]
    for row, t_idx in enumerate(distinct_transforms):
        mask = (ids == t_idx)
        idxs = np.where(mask)[0]
        rng.shuffle(idxs)
        ref = idxs[0]
        queries = idxs[1:3]
        x_ref = X[ref:ref + 1]
        y_ref = Y[ref:ref + 1]
        # Plot reference pair
        axes[row, 0].imshow(X[ref].reshape(H_W, H_W), cmap="gray_r",
                             vmin=0, vmax=1, interpolation="nearest")
        axes[row, 1].imshow(Y[ref].reshape(H_W, H_W), cmap="gray_r",
                             vmin=0, vmax=1, interpolation="nearest")
        for k in range(2):
            axes[row, k].set_xticks([])
            axes[row, k].set_yticks([])
        # Place "given" arrow label in middle
        if row == 0:
            axes[row, 0].set_title("$x_\\mathrm{ref}$", fontsize=9)
            axes[row, 1].set_title("$y_\\mathrm{ref}$", fontsize=9)
        # Two queries: (x_q, predicted y_q, true y_q)
        for q, qi in enumerate(queries):
            x_q = X[qi:qi + 1]
            y_pred = model.transfer(x_q, x_ref, y_ref, n_gibbs=2)
            base = 2 + q * 3 - (1 if q else 0)
            # Layout: cols 2..3, 4..5 are query blocks; col 6 is unused on
            # purpose since we want a small visual gap.
            cx = 2 + q * 2 + (1 if q else 0)
            ax_xq = axes[row, cx]
            ax_yp = axes[row, cx + 1]
            ax_xq.imshow(X[qi].reshape(H_W, H_W), cmap="gray_r",
                         vmin=0, vmax=1, interpolation="nearest")
            ax_yp.imshow(y_pred.reshape(H_W, H_W), cmap="gray_r",
                         vmin=0, vmax=1, interpolation="nearest")
            for ax in (ax_xq, ax_yp):
                ax.set_xticks([]); ax.set_yticks([])
            if row == 0:
                ax_xq.set_title(f"$x_{{q{q+1}}}$", fontsize=9)
                ax_yp.set_title(f"$\\hat{{y}}_{{q{q+1}}}$", fontsize=9)
        # Hide gap column
        axes[row, 4].axis("off")
        axes[row, 0].set_ylabel(transform_label(pool[t_idx]),
                                rotation=0, ha="right", labelpad=20, fontsize=9)
    fig.suptitle("Transformation transfer  "
                 "($x_q$ pushed through $h(x_\\mathrm{ref}, y_\\mathrm{ref})$)",
                 fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ----------------------------------------------------------------------
# main
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--transforms", type=str, default="shift")
    p.add_argument("--shift-max", type=int, default=1)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--n-train", type=int, default=4000)
    p.add_argument("--n-test", type=int, default=1000)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--n-factors", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    transforms = tuple(t.strip() for t in args.transforms.split(","))
    rng = np.random.default_rng(args.seed)

    print(f"Generating data (seed={args.seed}) ...")
    X_tr, Y_tr, ids_tr, pool = generate_transformed_pairs(
        args.n_train, transforms=transforms, shift_max=args.shift_max,
        rng=rng)
    X_te, Y_te, ids_te, _ = generate_transformed_pairs(
        args.n_test, transforms=transforms, shift_max=args.shift_max,
        rng=rng)
    n_classes = len(pool)

    print(f"Training gated 3-way RBM for {args.epochs} epochs ...")
    model = build_gated_rbm(169, 169, args.n_hidden, args.n_factors,
                             init_scale=args.init_scale, seed=args.seed)

    def eval_fn(m, epoch):
        acc, _ = transform_classification_accuracy(
            m, X_te, Y_te, ids_te, n_classes,
            rng=np.random.default_rng(epoch))
        recon = reconstruction_metrics(m, X_te, Y_te)
        return dict(transform_acc=acc, test_recon_mse=recon["recon_mse"])

    history = train(model, X_tr, Y_tr,
                    n_epochs=args.epochs, lr=args.lr,
                    batch_size=args.batch_size,
                    eval_fn=eval_fn,
                    eval_every=max(1, args.epochs // 10),
                    verbose=False)

    final_acc, _ = transform_classification_accuracy(
        model, X_te, Y_te, ids_te, n_classes,
        rng=np.random.default_rng(args.seed + 1))
    profile = per_unit_transform_profile(model, X_te, Y_te, ids_te, n_classes)
    spec = transform_specificity_score(profile)
    print(f"  final transform-classification: {final_acc * 100:.1f}%   "
          f"chance: {100.0 / n_classes:.1f}%   specificity: {spec:.2f}")

    plot_example_pairs(X_te, Y_te, ids_te, pool,
                       os.path.join(args.outdir, "example_pairs.png"))
    plot_filter_pairs(model, os.path.join(args.outdir, "filter_pairs.png"),
                      n_top=16)
    plot_transformation_profile(profile, pool,
                                 os.path.join(args.outdir,
                                              "transformation_profile.png"))
    plot_training_curves(history, n_classes,
                         os.path.join(args.outdir, "training_curves.png"))
    plot_transfer_examples(model, X_te, Y_te, ids_te, pool,
                            os.path.join(args.outdir, "transfer_examples.png"),
                            rng=np.random.default_rng(args.seed + 2),
                            n_examples=min(5, n_classes))


if __name__ == "__main__":
    main()
