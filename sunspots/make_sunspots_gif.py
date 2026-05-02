"""
Animated GIF: the three regularisers' weight distributions evolving as
training progresses.

Layout per frame:
  Top:      yearly Wolfer counts in grey + each method's running test
             prediction across the test years
  Bottom:   three side-by-side weight histograms (vanilla, decay, mog),
             with the K MoG component density curves overlaid on the mog
             panel. The clusters appear and tighten over time -- the
             headline result.

Usage:
    python3 make_sunspots_gif.py             # sunspots.gif at defaults
    python3 make_sunspots_gif.py --fps 12 --snapshot-every 200
"""

from __future__ import annotations
import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from sunspots import (load_wolfer, weigend_split, train_one, MLP, MoG)


METHOD_COLORS = {"vanilla": "#1f77b4", "decay": "#2ca02c", "mog": "#d62728"}
METHOD_NAMES = {"vanilla": "vanilla", "decay": "weight decay",
                "mog": "MoG soft sharing"}


def _snapshot_pred(model: MLP, X: np.ndarray, norm: float) -> np.ndarray:
    return model.predict(X).ravel() * norm


def render_frame(epoch: int, max_epoch: int, snaps: dict, data: dict,
                 hist_bins: np.ndarray, mog_density_state: dict,
                 ) -> Image.Image:
    fig = plt.figure(figsize=(11, 6.6), dpi=100)
    gs = fig.add_gridspec(2, 3, height_ratios=[1.0, 1.1],
                           hspace=0.55, wspace=0.30)

    # --- top: predictions over the test years ---
    ax_p = fig.add_subplot(gs[0, :])
    obs_years = data["test_years"]
    obs = data["y_test"].ravel() * data["norm"]
    ax_p.plot(obs_years, obs, color="black", linewidth=1.6,
               marker="o", markersize=3.5, label="observed")
    for name, snap in snaps.items():
        pred = _snapshot_pred(snap["model"], data["X_test"], data["norm"])
        ax_p.plot(obs_years, pred, color=METHOD_COLORS[name], linewidth=1.4,
                   alpha=0.85, label=f"{METHOD_NAMES[name]}  "
                                       f"(test MSE={snap['test_mse']:.4f})")
    ax_p.set_xlabel("year", fontsize=9)
    ax_p.set_ylabel("yearly sunspot count", fontsize=9)
    ax_p.set_title(f"Test-set predictions  -  epoch {epoch + 1} / {max_epoch}",
                    fontsize=10)
    ax_p.legend(fontsize=8, loc="upper right")
    ax_p.grid(alpha=0.3)
    ax_p.set_ylim(-20, max(220, obs.max() * 1.15))

    # --- bottom three: weight histograms ---
    methods = ["vanilla", "decay", "mog"]
    for idx, name in enumerate(methods):
        ax = fig.add_subplot(gs[1, idx])
        W = snaps[name]["model"].all_W()
        ax.hist(W, bins=hist_bins, color=METHOD_COLORS[name],
                 edgecolor="black", linewidth=0.3, alpha=0.85)
        ax.set_xlabel("weight value", fontsize=9)
        ax.set_xlim(hist_bins[0], hist_bins[-1])
        ax.set_ylim(0, mog_density_state["ymax"])
        ax.axvline(0, color="black", linewidth=0.5)
        ax.set_title(f"{METHOD_NAMES[name]}  (|W|={np.linalg.norm(W):.2f})",
                      fontsize=10)
        ax.grid(alpha=0.3, axis="y")
        if name == "mog" and snaps[name].get("mog_state") is not None:
            mog_state = snaps[name]["mog_state"]
            xs = np.linspace(hist_bins[0], hist_bins[-1], 300)
            pi = mog_state["pi"]; sig = mog_state["sigma"]; mu = mog_state["mu"]
            palette = ["#9467bd", "#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]
            densities = []
            for k in range(len(pi)):
                d = pi[k] * np.exp(-0.5 * ((xs - mu[k]) / sig[k]) ** 2) \
                      / (np.sqrt(2 * np.pi) * sig[k])
                densities.append(d)
            scale = mog_density_state["density_scale"]
            for k, d in enumerate(densities):
                ax.plot(xs, d * scale, color=palette[k % len(palette)],
                         linewidth=1.0, alpha=0.85)
            ax.plot(xs, sum(densities) * scale, color="black",
                     linewidth=1.0, linestyle="--", alpha=0.7)

    fig.suptitle("Sunspots: vanilla / weight-decay / MoG soft weight-sharing  "
                  f"(Nowlan & Hinton 1992)", fontsize=11.5, y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.96))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def collect_snapshots(method: str, data: dict, n_epochs: int, n_hidden: int,
                       lam: float, n_components: int, lr: float, seed: int,
                       snapshot_every: int) -> list[dict]:
    """Train and return a list of {epoch, model_snapshot, mog_state, test_mse}
    sampled every `snapshot_every` epochs."""
    snaps = []

    def cb(epoch, model, history, mog):
        snaps.append({
            "epoch": epoch + 1,
            "model_snapshot": model.snapshot(),
            "mog_state": (None if mog is None
                            else {"pi": mog.pi().copy(),
                                   "mu": mog.mu.copy(),
                                   "sigma": mog.sigma().copy()}),
            "test_mse": history["test_mse"][-1],
        })

    train_one(method, data, n_epochs=n_epochs, n_hidden=n_hidden,
                lr=lr, lam=lam, n_components=n_components, seed=seed,
                snapshot_callback=cb, snapshot_every=snapshot_every)
    return snaps


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=12000)
    p.add_argument("--n-hidden", type=int, default=16)
    p.add_argument("--n-components", type=int, default=5)
    p.add_argument("--lam-decay", type=float, default=0.01)
    p.add_argument("--lam-mog", type=float, default=0.0005)
    p.add_argument("--lr", type=float, default=0.0005)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--snapshot-every", type=int, default=300)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--out", type=str, default="sunspots.gif")
    p.add_argument("--hold-final", type=int, default=18)
    p.add_argument("--max-frame-side", type=int, default=900)
    args = p.parse_args()

    print("Loading Wolfer data...")
    years, counts = load_wolfer()
    data = weigend_split(years, counts, n_lags=12)

    print(f"\nTraining + collecting snapshots (every {args.snapshot_every} ep)...")
    series = {}
    for name, lam in [("vanilla", 0.0), ("decay", args.lam_decay),
                       ("mog", args.lam_mog)]:
        print(f"  {name}...")
        snaps = collect_snapshots(
            name, data, n_epochs=args.epochs, n_hidden=args.n_hidden,
            lam=lam, n_components=args.n_components, lr=args.lr,
            seed=args.seed, snapshot_every=args.snapshot_every,
        )
        series[name] = snaps
        print(f"    {len(snaps)} snapshots, final test MSE = "
              f"{snaps[-1]['test_mse']:.5f}")

    n_frames = min(len(s) for s in series.values())
    print(f"\nRendering {n_frames} frames...")

    # Determine x-range and y-max for stable histograms across frames
    final_W = []
    for s in series.values():
        snap = s[-1]
        snapped_W = np.concatenate([
            snap["model_snapshot"]["W1"].ravel(),
            snap["model_snapshot"]["W2"].ravel(),
        ])
        final_W.append(snapped_W)
    rng = max(np.abs(np.concatenate(final_W)).max(), 1e-3) * 1.1
    bins = np.linspace(-rng, rng, 41)

    # estimate hist y-max across all snapshots: take max over final frames
    ymax = 0
    for s in series.values():
        for snap in s:
            snapped_W = np.concatenate([
                snap["model_snapshot"]["W1"].ravel(),
                snap["model_snapshot"]["W2"].ravel(),
            ])
            counts_, _ = np.histogram(snapped_W, bins=bins)
            ymax = max(ymax, counts_.max())
    ymax = ymax * 1.1
    density_scale = ymax * 0.45     # so the density curves don't overshoot

    frames: list[Image.Image] = []
    for i in range(n_frames):
        snaps_for_frame = {}
        for name, s in series.items():
            snap = s[i]
            # rebuild a dummy MLP from the snapshot for predict()
            m = MLP(n_in=12, n_hidden=args.n_hidden, n_out=1,
                     init_scale=0.0, seed=0)
            m.W1 = snap["model_snapshot"]["W1"]
            m.b1 = snap["model_snapshot"]["b1"]
            m.W2 = snap["model_snapshot"]["W2"]
            m.b2 = snap["model_snapshot"]["b2"]
            snaps_for_frame[name] = {
                "model": m,
                "mog_state": snap["mog_state"],
                "test_mse": snap["test_mse"],
            }
        frame = render_frame(snap["epoch"] - 1, args.epochs, snaps_for_frame,
                              data, bins,
                              {"ymax": ymax, "density_scale": density_scale})
        if max(frame.size) > args.max_frame_side:
            scale = args.max_frame_side / max(frame.size)
            new_size = (int(frame.size[0] * scale),
                         int(frame.size[1] * scale))
            frame = frame.resize(new_size, Image.LANCZOS)
        frames.append(frame)

    if args.hold_final > 0:
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
