"""
Render the animated GIF for the dipole 3D-constraint population code.

Layout per frame:
  Top-left:    8 example dipole inputs at random (x, y, theta)
  Top-right:   their reconstructions through the 3D bottleneck
  Bottom-left: 3D scatter of m_hat for a held-out batch, coloured by theta
  Bottom-right: training curves (MSE + linear R^2 to (x, y, cos2t, sin2t))

Snapshots at evenly-spaced training epochs.
"""

from __future__ import annotations

import argparse
import os

import imageio.v2 as imageio
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

from dipole_3d_constraint import (
    build_population_coder,
    forward,
    generate_dipole_images,
    implicit_space_recovery,
    loss_and_grads,
    sgd_update,
    description_length_loss,
)


def render_frame(model, X_show, params_show, X_held, params_held,
                 history: dict, epoch: int) -> Image.Image:
    fwd_show = forward(model, X_show)
    fwd_held = forward(model, X_held)
    Xh_show = fwd_show["x_hat"]
    M_held = fwd_held["m_hat"]
    t_held = params_held[:, 2]

    fig = plt.figure(figsize=(11, 6.5), dpi=110)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05],
                          hspace=0.4, wspace=0.32)

    # ---- top-left: input strip ----
    ax_in = fig.add_subplot(gs[0, 0])
    n = X_show.shape[0]
    strip = np.concatenate([X_show[k].reshape(8, 8) for k in range(n)],
                           axis=1)
    vmax = float(max(np.abs(X_show).max(), np.abs(Xh_show).max(), 1e-3))
    ax_in.imshow(strip, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 aspect="equal")
    ax_in.set_xticks([4 + 8 * k for k in range(n)])
    ax_in.set_xticklabels([f"$\\theta$={np.degrees(params_show[k, 2]):.0f}°"
                           for k in range(n)],
                          fontsize=7)
    ax_in.set_yticks([])
    ax_in.set_title("inputs (8 dipoles, random x,y,theta)", fontsize=10)

    # ---- top-right: reconstruction strip ----
    ax_re = fig.add_subplot(gs[0, 1])
    strip_h = np.concatenate([Xh_show[k].reshape(8, 8) for k in range(n)],
                             axis=1)
    ax_re.imshow(strip_h, cmap="RdBu_r", vmin=-vmax, vmax=vmax,
                 aspect="equal")
    ax_re.set_xticks([4 + 8 * k for k in range(n)])
    ax_re.set_xticklabels([f"({params_show[k,0]:.1f},{params_show[k,1]:.1f})"
                           for k in range(n)],
                          fontsize=7)
    ax_re.set_yticks([])
    ax_re.set_title("reconstructions through 3D m_hat bottleneck",
                    fontsize=10)

    # ---- bottom-left: 3D scatter of m_hat ----
    ax_sc = fig.add_subplot(gs[1, 0], projection="3d")
    sc = ax_sc.scatter(M_held[:, 0], M_held[:, 1], M_held[:, 2],
                       c=t_held, cmap="twilight", s=8,
                       alpha=0.85, edgecolors="none")
    ax_sc.set_xlim(0, 1); ax_sc.set_ylim(0, 1); ax_sc.set_zlim(0, 1)
    ax_sc.set_xlabel("m_hat[0]", fontsize=8)
    ax_sc.set_ylabel("m_hat[1]", fontsize=8)
    ax_sc.set_zlabel("m_hat[2]", fontsize=8)
    ax_sc.set_title(f"3D implicit space (colour = $\\theta$)\n"
                    f"epoch {epoch}", fontsize=10)
    ax_sc.tick_params(labelsize=7)

    # ---- bottom-right: training curves ----
    ax_cu = fig.add_subplot(gs[1, 1])
    if history["epoch"]:
        ep_arr = np.asarray(history["epoch"])
        ax_cu.plot(ep_arr, history["recon_mse"], color="#2ca02c",
                   label="recon MSE")
        ax_cu.set_xlabel("epoch")
        ax_cu.set_ylabel("MSE", color="#2ca02c")
        ax_cu.tick_params(axis="y", labelcolor="#2ca02c")
        ax_cu.grid(alpha=0.3)
        ax_cu.set_xlim(0, max(ep_arr.max(), 1))

        ax_r2 = ax_cu.twinx()
        ax_r2.plot(ep_arr, history["r2_mean"], color="#9467bd",
                   label="linear $R^2$")
        ax_r2.set_ylabel(r"linear $R^2$ to (x,y,cos$2\theta$,sin$2\theta$)",
                         color="#9467bd")
        ax_r2.tick_params(axis="y", labelcolor="#9467bd")
        ax_r2.set_ylim(-0.05, 1.05)
    ax_cu.set_title("training curves", fontsize=10)

    fig.suptitle("Dipole 3D-constraint population code (Zemel & Hinton 1995)",
                 fontsize=12)
    fig.tight_layout()

    fig.canvas.draw()
    img = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()
    plt.close(fig)
    return Image.fromarray(img)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-epochs", type=int, default=120)
    parser.add_argument("--n-train", type=int, default=1500)
    parser.add_argument("--n-hidden", type=int, default=225)
    parser.add_argument("--n-enc-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=0.18)
    parser.add_argument("--snapshot-every", type=int, default=4)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--out", type=str, default="dipole_3d_constraint.gif")
    parser.add_argument("--max-size-mb", type=float, default=3.0)
    args = parser.parse_args()

    np.random.seed(args.seed)

    matplotlib.rcParams["font.size"] = 9

    print("Generating data...")
    X, params = generate_dipole_images(args.n_train, seed=args.seed)
    X_show, params_show = generate_dipole_images(8, seed=args.seed + 17)
    X_held, params_held = generate_dipole_images(400, seed=args.seed + 1000)

    print("Building model...")
    model = build_population_coder(n_hidden=args.n_hidden,
                                   n_enc_hidden=args.n_enc_hidden,
                                   sigma=args.sigma,
                                   seed=args.seed)

    history = {"epoch": [], "loss": [], "recon_mse": [],
               "dl_bits": [], "r2_mean": []}

    rng = np.random.default_rng(args.seed)
    N = X.shape[0]
    batch = 64
    frames = []

    # initial frame
    frames.append(render_frame(model, X_show, params_show,
                               X_held, params_held, history, 0))

    for epoch in range(args.n_epochs):
        idx = rng.permutation(N)
        epoch_loss = 0.0
        for s in range(0, N, batch):
            xb = X[idx[s:s + batch]]
            loss, grads, _ = loss_and_grads(model, xb)
            epoch_loss += float(loss) * xb.shape[0] / N
            sgd_update(model, grads, args.lr)

        # eval each epoch (cheap on 1500 samples)
        fwd = forward(model, X)
        mse = float(((fwd["x_hat"] - X) ** 2).mean())
        rec = implicit_space_recovery(model, X, params, degree=1)
        history["epoch"].append(epoch)
        history["loss"].append(epoch_loss)
        history["recon_mse"].append(mse)
        history["dl_bits"].append(description_length_loss(model, X))
        history["r2_mean"].append(rec["r2_mean"])

        if (epoch + 1) % args.snapshot_every == 0 or epoch == args.n_epochs - 1:
            frames.append(render_frame(model, X_show, params_show,
                                       X_held, params_held, history,
                                       epoch + 1))
            print(f"epoch {epoch + 1}: mse={mse:.4f} "
                  f"r2_lin={rec['r2_mean']:.3f}")

    # hold final frame a bit by repeating
    frames.extend([frames[-1]] * max(2, args.fps // 2))

    print(f"Writing {args.out} ({len(frames)} frames @ {args.fps} fps)...")
    imageio.mimsave(args.out, [np.asarray(f) for f in frames],
                    fps=args.fps, loop=0)

    size_mb = os.path.getsize(args.out) / 1e6
    print(f"GIF size: {size_mb:.2f} MB")
    if size_mb > args.max_size_mb:
        print(f"WARNING: GIF exceeds {args.max_size_mb} MB target.")


if __name__ == "__main__":
    main()
