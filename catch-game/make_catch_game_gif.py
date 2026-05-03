"""
Animated GIF of a trained catch-game agent playing one episode.

Each frame shows three panels side-by-side:
  1. true world state (ball + paddle, always rendered)
  2. agent's input (blanked once t > blank_after)
  3. fast-weights matrix A_t

Trains a quick model, picks a deterministic high-stakes spawn (ball far from
the paddle's start so the agent must move + remember), then animates.

Output: catch_game.gif at the project root.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import patches as mpatches

from catch_game import CatchEnv, build_a3c_policy, train_a3c, _evaluate
from visualize_catch_game import _record_episode

try:
    import imageio.v2 as imageio  # tolerant import; pillow fallback below
except ImportError:
    imageio = None


def _frame_to_array(fig) -> np.ndarray:
    """Render a matplotlib figure into an HxWx3 uint8 numpy array."""
    fig.canvas.draw()
    rgba = np.asarray(fig.canvas.buffer_rgba())
    return rgba[..., :3].copy()


def _render_frame(rec: dict, t: int, vmax_A: float) -> np.ndarray:
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7),
                             gridspec_kw={"wspace": 0.25})
    # 1. true world
    ax = axes[0]
    ax.imshow(rec["grids_visible"][t], cmap="gray_r",
              vmin=0, vmax=1, interpolation="nearest")
    ax.set_title(f"true world (t={t})", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 2. agent input
    ax = axes[1]
    if rec["grids_blanked"][t]:
        ax.imshow(np.ones_like(rec["grids_obs"][t]) * 0.3,
                  cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
        ax.text(rec["size"] / 2 - 0.5, rec["size"] / 2 - 0.5,
                "BLANK", ha="center", va="center",
                fontsize=11, color="white", fontweight="bold")
    else:
        ax.imshow(rec["grids_obs"][t], cmap="gray_r",
                  vmin=0, vmax=1, interpolation="nearest")
    a_t = rec["actions"][t]
    a_str = ["LEFT", "STAY", "RIGHT"][a_t]
    ax.set_title(f"agent input -- action: {a_str}", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    # 3. fast-weights matrix
    ax = axes[2]
    A = rec["As"][t]
    ax.imshow(A, cmap="RdBu_r", vmin=-vmax_A, vmax=vmax_A,
              interpolation="nearest")
    ax.set_title("A_t (fast-weights memory)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    arr = _frame_to_array(fig)
    plt.close(fig)
    return arr


def _make_outcome_frame(rec: dict, vmax_A: float) -> np.ndarray:
    """A final summary frame held for ~1 second."""
    T = rec["T"]
    last = T - 1
    fig, axes = plt.subplots(1, 3, figsize=(7.6, 2.7),
                             gridspec_kw={"wspace": 0.25})
    ax = axes[0]
    ax.imshow(rec["grids_visible"][last], cmap="gray_r",
              vmin=0, vmax=1, interpolation="nearest")
    outcome = "CAUGHT" if rec["caught"] else "MISSED"
    color = "tab:green" if rec["caught"] else "tab:red"
    ax.set_title(f"final state -- {outcome}", fontsize=10, color=color)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[1]
    ax.imshow(np.ones_like(rec["grids_obs"][last]) * 0.3,
              cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
    ax.text(rec["size"] / 2 - 0.5, rec["size"] / 2 - 0.5,
            "BLANK", ha="center", va="center", fontsize=11,
            color="white", fontweight="bold")
    ax.set_title(f"reward = {sum(rec['rewards']):+.0f}", fontsize=9, color=color)
    ax.set_xticks([]); ax.set_yticks([])

    ax = axes[2]
    A = rec["As"][last]
    ax.imshow(A, cmap="RdBu_r", vmin=-vmax_A, vmax=vmax_A,
              interpolation="nearest")
    ax.set_title("A_T (final fast weights)", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])

    arr = _frame_to_array(fig)
    plt.close(fig)
    return arr


def _save_gif(frames: list[np.ndarray], path: str, fps: int = 3) -> None:
    duration_ms = int(1000 / fps)
    if imageio is not None:
        # imageio.mimsave wants list of arrays. duration in ms recently switched.
        imageio.mimsave(path, frames, duration=duration_ms, loop=0)
        return
    # Pillow fallback
    from PIL import Image
    pil_imgs = [Image.fromarray(f) for f in frames]
    pil_imgs[0].save(path, save_all=True, append_images=pil_imgs[1:],
                     duration=duration_ms, loop=0, optimize=True)


def main() -> None:
    p = argparse.ArgumentParser(description="Animated catch-game agent.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=24)
    p.add_argument("--blank-after", type=int, default=8)
    p.add_argument("--n-episodes", type=int, default=12000,
                   help="how long to train the model used for the GIF; "
                        "matches the headline run; ~60-90s wallclock")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--batch-episodes", type=int, default=16)
    p.add_argument("--out", type=str, default="catch_game.gif")
    p.add_argument("--fps", type=int, default=3)
    args = p.parse_args()

    print(f"# size={args.size}  blank_after={args.blank_after}  "
          f"hidden={args.hidden}  episodes={args.n_episodes}")

    print("[1/2] training the FW model ...", flush=True)
    obs_dim = args.size * args.size
    model = build_a3c_policy(obs_dim=obs_dim, n_actions=3,
                             hidden=args.hidden, use_fast_weights=True,
                             seed=args.seed)
    history = train_a3c(CatchEnv, model,
                        n_episodes=args.n_episodes, lr=args.lr,
                        size=args.size, blank_after=args.blank_after,
                        batch_episodes=args.batch_episodes,
                        eval_every=500, eval_episodes=50,
                        seed=args.seed, verbose=False)
    final_eval = _evaluate(model, CatchEnv, size=args.size,
                           blank_after=args.blank_after,
                           n_episodes=200, seed=args.seed + 50000,
                           greedy=True)
    print(f"  trained {args.n_episodes} eps in {history['wallclock']:.1f}s "
          f"-> greedy eval {final_eval*100:.1f}%")

    print("[2/2] picking a high-stakes episode + rendering frames ...",
          flush=True)
    # try a few seeds; keep the FIRST CAUGHT one with the ball spawning far
    # from the paddle's start (i.e., the agent had to move several columns
    # under blank).
    rec = None
    for trial in range(50):
        cand = _record_episode(model, args.size, args.blank_after,
                               seed=args.seed + 9000 + trial,
                               spawn_col=args.size - 2,
                               greedy=True)
        if cand["caught"]:
            rec = cand
            break
    if rec is None:
        # fall back to any episode (caught or not)
        rec = _record_episode(model, args.size, args.blank_after,
                              seed=args.seed + 9000,
                              spawn_col=args.size - 2,
                              greedy=True)

    vmax_A = float(np.max(np.abs(rec["As"])) + 1e-9)
    frames = []
    for t in range(rec["T"]):
        frames.append(_render_frame(rec, t, vmax_A))
    # hold the final frame a few extra ticks
    final = _make_outcome_frame(rec, vmax_A)
    for _ in range(args.fps):
        frames.append(final)

    print(f"  rendered {len(frames)} frames")

    out_path = args.out
    _save_gif(frames, out_path, fps=args.fps)
    sz_kb = os.path.getsize(out_path) / 1024.0
    print(f"\n=== wrote {out_path} ({sz_kb:.0f} KB) ===")


if __name__ == "__main__":
    main()
