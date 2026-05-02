"""Visualizations for the Peterson-Barney mixture-of-experts experiment.

Reads `results.json` and the companion `results.npz` written by
`vowel_mixture_experts.py --results results.json` and emits four PNGs into
`viz/`:

  data_scatter.png        : F1/F2 scatter coloured by class.
  expert_partitioning.png : argmax-gate partition over a 2-D grid, with the
                            decision boundary of the dominant expert per
                            region overlaid.
  training_curves.png     : MoE vs monolithic-MLP loss and test accuracy.
  comparison_table.png    : final-metric grid (numeric summary).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from vowel_mixture_experts import (
    MoE,
    MLP,
    VOWEL_LABELS,
    softmax,
)

CLASS_COLOURS = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]  # 4 vowels
EXPERT_COLOURS = [
    "#a6cee3", "#fdbf6f", "#b2df8a", "#fb9a99",
    "#cab2d6", "#ffff99", "#1f78b4", "#33a02c",
]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def load_run(results_path: Path):
    """Return (results dict, models dict, data dict)."""
    res = json.loads(results_path.read_text())
    npz = np.load(results_path.with_suffix(".npz"))
    cfg = res["config"]
    moe = MoE(
        n_in=2,
        n_classes=4,
        n_experts=cfg["n_experts"],
        W_e=npz["W_e"], b_e=npz["b_e"],
        W_g=npz["W_g"], b_g=npz["b_g"],
    )
    mlp = MLP(
        n_in=2,
        n_hidden=cfg["n_hidden_baseline"],
        n_classes=4,
        W1=npz["W1"], b1=npz["b1"], W2=npz["W2"], b2=npz["b2"],
    )
    data = dict(
        Xtr=npz["X_train"], ytr=npz["y_train"],
        Xte=npz["X_test"], yte=npz["y_test"],
        feat_mean=npz["feat_mean"], feat_std=npz["feat_std"],
    )
    return res, dict(moe=moe, mlp=mlp), data


# ---------------------------------------------------------------------------
# 1. F1/F2 scatter coloured by class
# ---------------------------------------------------------------------------

def plot_scatter(data: dict, path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    Xtr = data["Xtr"] * data["feat_std"] + data["feat_mean"]
    Xte = data["Xte"] * data["feat_std"] + data["feat_mean"]
    for cls in range(4):
        m_tr = data["ytr"] == cls
        m_te = data["yte"] == cls
        ax.scatter(
            Xtr[m_tr, 1], Xtr[m_tr, 0],
            c=CLASS_COLOURS[cls], s=18, alpha=0.6,
            label=VOWEL_LABELS[cls], edgecolor="none",
        )
        ax.scatter(
            Xte[m_te, 1], Xte[m_te, 0],
            c=CLASS_COLOURS[cls], s=40, alpha=0.95,
            edgecolor="black", linewidth=0.7, marker="^",
        )
    # Phonetic vowel chart convention: F1 increases downward, F2 decreases
    # rightward.  This makes [i] sit top-right and [a] bottom-left.
    ax.invert_yaxis()
    ax.invert_xaxis()
    ax.set_xlabel("F2 (Hz)")
    ax.set_ylabel("F1 (Hz)")
    ax.set_title("Peterson-Barney F1/F2 by vowel\n(circles = train, triangles = test)")
    ax.legend(loc="upper left", framealpha=0.9)
    ax.grid(True, linestyle=":", alpha=0.4)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 2. Expert partitioning -- argmax-gate over a grid
# ---------------------------------------------------------------------------

def _grid_in_input_space(data: dict, n: int = 220):
    Xtr = data["Xtr"] * data["feat_std"] + data["feat_mean"]
    f1_lo, f1_hi = Xtr[:, 0].min() - 50, Xtr[:, 0].max() + 50
    f2_lo, f2_hi = Xtr[:, 1].min() - 50, Xtr[:, 1].max() + 50
    f1 = np.linspace(f1_lo, f1_hi, n)
    f2 = np.linspace(f2_lo, f2_hi, n)
    F1, F2 = np.meshgrid(f1, f2, indexing="ij")  # (n, n)
    grid_raw = np.stack([F1.ravel(), F2.ravel()], axis=1)
    grid_std = (grid_raw - data["feat_mean"]) / data["feat_std"]
    return F1, F2, grid_std


def plot_partitioning(model: MoE, data_or_X, y_or_path, path: str | None = None) -> None:
    """Argmax of the gate over a 2-D F1/F2 grid.

    Two call signatures:
      * plot_partitioning(model, data_dict, path)        -- preferred
      * plot_partitioning(model, X, y, path)             -- spec-compatible
    """
    if isinstance(data_or_X, dict):
        # Preferred form: data_dict carries already-split / standardised data.
        data = data_or_X
        path = y_or_path
    else:
        # Spec-compatible form: model, X (raw F1/F2 in Hz), y, path.
        X = np.asarray(data_or_X, dtype=np.float64)
        y = np.asarray(y_or_path, dtype=np.int64)
        if path is None:
            path = "expert_partitioning.png"
        feat_mean = X.mean(0)
        feat_std = X.std(0) + 1e-9
        data = dict(
            Xtr=(X - feat_mean) / feat_std,
            ytr=y,
            Xte=np.empty((0, 2)),
            yte=np.empty(0, dtype=int),
            feat_mean=feat_mean, feat_std=feat_std,
        )
    F1, F2, grid_std = _grid_in_input_space(data)
    p_mix, g, p_e = model.predict(grid_std)
    expert_argmax = g.argmax(axis=1).reshape(F1.shape)
    class_argmax = p_mix.argmax(axis=1).reshape(F1.shape)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    cmap_e = ListedColormap(EXPERT_COLOURS[: model.n_experts])
    cmap_c = ListedColormap(CLASS_COLOURS)
    Xtr = data["Xtr"] * data["feat_std"] + data["feat_mean"]

    # Left panel: which expert dominates which input region.
    ax = axes[0]
    ax.pcolormesh(F2, F1, expert_argmax, cmap=cmap_e, shading="auto", alpha=0.7,
                  vmin=-0.5, vmax=model.n_experts - 0.5)
    for cls in range(4):
        m = data["ytr"] == cls
        ax.scatter(Xtr[m, 1], Xtr[m, 0], c=CLASS_COLOURS[cls], s=14,
                   edgecolor="black", linewidth=0.4, label=VOWEL_LABELS[cls])
    ax.invert_yaxis(); ax.invert_xaxis()
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
    # Report which experts the gate actually uses on the data.
    g_data = model.predict(data["Xtr"])[1]
    used = sorted(set(g_data.argmax(axis=1).tolist()))
    ax.set_title(f"Gate argmax over input space\n(active experts on data: {used})")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    # Right panel: mixture's predicted class.
    ax = axes[1]
    ax.pcolormesh(F2, F1, class_argmax, cmap=cmap_c, shading="auto", alpha=0.4,
                  vmin=-0.5, vmax=3.5)
    for cls in range(4):
        m = data["ytr"] == cls
        ax.scatter(Xtr[m, 1], Xtr[m, 0], c=CLASS_COLOURS[cls], s=14,
                   edgecolor="black", linewidth=0.4, label=VOWEL_LABELS[cls])
    ax.invert_yaxis(); ax.invert_xaxis()
    ax.set_xlabel("F2 (Hz)"); ax.set_ylabel("F1 (Hz)")
    ax.set_title("Mixture predicted class")
    ax.legend(loc="upper left", framealpha=0.9, fontsize=9)

    fig.suptitle("Mixture-of-experts: input partitioning + decision regions", y=1.02)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 3. Training curves
# ---------------------------------------------------------------------------

def plot_training_curves(res: dict, path: Path) -> None:
    moe = res["moe_history"]
    mlp = res["mlp_history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    ax = axes[0]
    ax.plot(moe["epoch"], moe["train_loss"], label="MoE", color="#1f77b4", lw=1.8)
    ax.plot(mlp["epoch"], mlp["train_loss"], label="Monolithic MLP", color="#d62728", lw=1.8)
    ax.set_yscale("log")
    ax.set_xlabel("epoch"); ax.set_ylabel("train cross-entropy (log scale)")
    ax.set_title("Training loss")
    ax.legend(framealpha=0.9); ax.grid(True, linestyle=":", alpha=0.4)

    ax = axes[1]
    ax.plot(moe["epoch"], moe["test_acc"], label="MoE", color="#1f77b4", lw=1.8)
    ax.plot(mlp["epoch"], mlp["test_acc"], label="Monolithic MLP", color="#d62728", lw=1.8)
    ax.axhline(0.90, color="grey", linestyle="--", lw=0.8, label="90% threshold")
    ax.axhline(0.25, color="grey", linestyle=":", lw=0.8, label="chance (25%)")
    if res["summary"]["moe_epochs_to_90"] is not None:
        ax.axvline(res["summary"]["moe_epochs_to_90"], color="#1f77b4", linestyle=":", lw=0.8)
    if res["summary"]["mlp_epochs_to_90"] is not None:
        ax.axvline(res["summary"]["mlp_epochs_to_90"], color="#d62728", linestyle=":", lw=0.8)
    ax.set_xlabel("epoch"); ax.set_ylabel("test accuracy")
    ax.set_ylim(0.0, 1.05)
    ax.set_title("Speaker-independent test accuracy")
    ax.legend(framealpha=0.9, loc="lower right"); ax.grid(True, linestyle=":", alpha=0.4)

    fig.suptitle(
        f"MoE (K={res['config']['n_experts']}, {res['config']['moe_params']} params) vs "
        f"monolithic MLP (H={res['config']['n_hidden_baseline']}, "
        f"{res['config']['mlp_params']} params)",
        y=1.02,
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# 4. Comparison table (rendered as a figure)
# ---------------------------------------------------------------------------

def plot_comparison_table(res: dict, path: Path) -> None:
    s = res["summary"]; c = res["config"]
    rows = [
        ("model", "params", "test acc", "epochs->90%", "wallclock (s)"),
        (
            f"MoE (K={c['n_experts']})",
            str(c["moe_params"]),
            f"{s['moe_test_acc']:.3f}",
            "--" if s["moe_epochs_to_90"] is None else str(s["moe_epochs_to_90"]),
            f"{s['moe_wallclock_s']:.2f}",
        ),
        (
            f"MLP (H={c['n_hidden_baseline']})",
            str(c["mlp_params"]),
            f"{s['mlp_test_acc']:.3f}",
            "--" if s["mlp_epochs_to_90"] is None else str(s["mlp_epochs_to_90"]),
            f"{s['mlp_wallclock_s']:.2f}",
        ),
    ]
    fig, ax = plt.subplots(figsize=(8, 2.4))
    ax.axis("off")
    table = ax.table(
        cellText=[list(r) for r in rows[1:]],
        colLabels=list(rows[0]),
        loc="center",
        cellLoc="center",
        colColours=["#dddddd"] * 5,
    )
    table.auto_set_font_size(False); table.set_fontsize(11); table.scale(1.0, 1.6)
    ax.set_title(
        "Headline summary: MoE vs monolithic backprop\n"
        f"(seed={c['seed']}, {c['n_train']} train / {c['n_test']} test, "
        f"{'real Peterson-Barney' if c['is_real_data'] else 'synthetic fallback'})",
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------

def main(argv=None) -> int:
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results", type=str, default="results.json")
    p.add_argument("--out-dir", type=str, default="viz")
    args = p.parse_args(argv)
    results_path = Path(args.results)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    res, models, data = load_run(results_path)
    plot_scatter(data, out_dir / "data_scatter.png")
    plot_partitioning(models["moe"], data, out_dir / "expert_partitioning.png")
    plot_training_curves(res, out_dir / "training_curves.png")
    plot_comparison_table(res, out_dir / "comparison_table.png")
    print(f"wrote {sorted(out_dir.glob('*.png'))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
