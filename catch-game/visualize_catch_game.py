"""
Static visualizations for catch-game (Ba et al. 2016 fast-weights RL).

Produces:
  viz/example_episode.png       -- frame strip of one episode (ball + paddle,
                                   blanked frames marked with a hatched mask)
  viz/training_curves.png       -- training reward, eval catch rate, loss
  viz/with_vs_without.png       -- with-FW vs without-FW catch-rate curves
                                   (the headline result)
  viz/fast_weights_evolution.png -- A_t heatmap snapshots across one episode
  viz/hidden_state_trace.png    -- h_t over time for one episode

Trains its own quick model (fewer episodes than the headline run) so this
script can run end-to-end in roughly 30 seconds. Override --n-episodes for
a closer match to the README headline.
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from catch_game import (
    CatchEnv, build_a3c_policy, train_a3c, _evaluate,
)


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _ensure_outdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _train_one(size: int, blank_after: int, n_episodes: int,
               hidden: int, lr: float, batch_episodes: int,
               beta_ent: float, value_coef: float, seed: int,
               use_fast_weights: bool, eval_every: int = 250
               ) -> tuple[object, dict]:
    obs_dim = size * size
    model = build_a3c_policy(obs_dim=obs_dim, n_actions=3, hidden=hidden,
                             use_fast_weights=use_fast_weights, seed=seed)
    history = train_a3c(CatchEnv, model,
                        n_episodes=n_episodes, lr=lr,
                        size=size, blank_after=blank_after,
                        beta_ent=beta_ent, value_coef=value_coef,
                        batch_episodes=batch_episodes,
                        eval_every=eval_every,
                        eval_episodes=100,
                        seed=seed,
                        verbose=False)
    return model, history


def _record_episode(model, size: int, blank_after: int, seed: int,
                    spawn_col: int | None = None,
                    greedy: bool = True) -> dict:
    """Run one episode and capture per-step state for plotting / gif."""
    env = CatchEnv(size=size, blank_after=blank_after, seed=seed)
    env.reset(seed=seed)
    if spawn_col is not None:
        # Force the ball spawn column for a deterministic demo.
        env._ball_col = int(spawn_col)
    rng = np.random.default_rng(seed + 42)

    T_max = env.episode_len
    H = model.H
    grids_visible = []
    grids_obs = []
    grids_blanked = []
    actions = []
    rewards = []
    paddle_xs = []
    ball_cols = []
    hs_arr = np.zeros((T_max + 1, H))
    As_arr = np.zeros((T_max, H, H))
    A_prev = np.zeros((H, H))

    obs = env.render_visible()
    for t in range(T_max):
        x_t = obs.reshape(-1)
        h_prev = hs_arr[t]
        h_t, A_t, _, _, _ = model._step_forward(x_t, h_prev, A_prev)
        hs_arr[t + 1] = h_t
        As_arr[t] = A_t
        probs, _ = model.policy_value(h_t)
        a_t = int(np.argmax(probs)) if greedy else int(rng.choice(3, p=probs))

        # record state BEFORE step
        grids_visible.append(env.render_visible().copy())
        grids_obs.append(obs.copy())
        grids_blanked.append(env.is_blanked())
        paddle_xs.append(env.state()["paddle_x"])
        ball_cols.append(env.state()["ball_col"])

        obs, r, done = env.step(a_t)
        actions.append(a_t)
        rewards.append(r)
        A_prev = A_t
        if done:
            break
    T = t + 1

    return {"T": T, "grids_visible": grids_visible,
            "grids_obs": grids_obs, "grids_blanked": grids_blanked,
            "actions": actions, "rewards": rewards,
            "paddle_xs": paddle_xs, "ball_cols": ball_cols,
            "hs": hs_arr[:T + 1], "As": As_arr[:T],
            "size": size, "blank_after": blank_after,
            "caught": float(np.sum(rewards)) > 0}


# --------------------------------------------------------------------------
# Plots
# --------------------------------------------------------------------------

def plot_example_episode(rec: dict, outpath: str) -> None:
    """Frame strip showing both the visible state and the agent's
    (possibly blanked) input, side by side per frame."""
    T = rec["T"]
    n_cols = T
    fig, axes = plt.subplots(2, n_cols, figsize=(1.0 * n_cols, 2.4),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    for t in range(T):
        ax_top = axes[0, t]
        ax_top.imshow(rec["grids_visible"][t], cmap="gray_r",
                      vmin=0, vmax=1, interpolation="nearest")
        ax_top.set_xticks([]); ax_top.set_yticks([])
        ax_top.set_title(f"t={t}", fontsize=7)

        ax_bot = axes[1, t]
        if rec["grids_blanked"][t]:
            # show blanked input as solid hatched grey
            ax_bot.imshow(np.ones_like(rec["grids_obs"][t]) * 0.3,
                          cmap="gray_r", vmin=0, vmax=1, interpolation="nearest")
            ax_bot.text(rec["size"] / 2 - 0.5, rec["size"] / 2 - 0.5,
                        "blank", ha="center", va="center",
                        fontsize=7, color="white")
        else:
            ax_bot.imshow(rec["grids_obs"][t], cmap="gray_r",
                          vmin=0, vmax=1, interpolation="nearest")
        ax_bot.set_xticks([]); ax_bot.set_yticks([])

    axes[0, 0].set_ylabel("true", fontsize=8)
    axes[1, 0].set_ylabel("agent input", fontsize=8)

    outcome = "CAUGHT" if rec["caught"] else "missed"
    fig.suptitle(f"Catch (size={rec['size']}, blank_after={rec['blank_after']}) "
                 f"-- one episode -- {outcome}",
                 fontsize=10)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_training_curves(history: dict, outpath: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    ep = history["episode"]
    axes[0].plot(ep, history["reward"], color="C0")
    axes[0].axhline(0, color="k", linewidth=0.5, linestyle=":")
    axes[0].set_xlabel("episode"); axes[0].set_ylabel("mean reward (per ep)")
    axes[0].set_title("training reward")
    axes[0].grid(alpha=0.3)

    axes[1].plot(ep, [c * 100 for c in history["eval_catch_rate"]], color="C2")
    axes[1].set_xlabel("episode"); axes[1].set_ylabel("eval catch rate (%)")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("eval catch rate (greedy, n=100)")
    axes[1].grid(alpha=0.3)

    axes[2].plot(ep, history["loss"], color="C3")
    axes[2].set_xlabel("episode"); axes[2].set_ylabel("REINFORCE loss")
    axes[2].set_title("loss")
    axes[2].grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_with_vs_without(history_fw: dict, history_no: dict,
                         final_fw: float, final_no: float,
                         chance: float, outpath: str) -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.5))
    ep1 = history_fw["episode"]
    ep0 = history_no["episode"]
    ax.plot(ep1, [c * 100 for c in history_fw["eval_catch_rate"]],
            color="C2", linewidth=1.8, label="with fast weights")
    ax.plot(ep0, [c * 100 for c in history_no["eval_catch_rate"]],
            color="C1", linewidth=1.8, label="vanilla RNN (no fast weights)")
    ax.axhline(chance * 100, color="k", linewidth=0.8, linestyle="--",
               label=f"random-paddle chance ({chance * 100:.1f}%)")
    ax.set_xlabel("episode")
    ax.set_ylabel("eval catch rate (greedy, %)")
    ax.set_ylim(0, 100)
    ax.set_title(f"Fast weights vs vanilla RNN -- final greedy catch rate "
                 f"(n=500): with={final_fw*100:.1f}%, "
                 f"without={final_no*100:.1f}%")
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_fast_weights_evolution(rec: dict, outpath: str,
                                n_panels: int = 6) -> None:
    """A_t snapshots at evenly spaced timesteps."""
    T = rec["T"]
    if T < n_panels:
        n_panels = T
    idxs = np.linspace(0, T - 1, n_panels).astype(int)

    fig, axes = plt.subplots(1, n_panels, figsize=(2.0 * n_panels, 2.3))
    if n_panels == 1:
        axes = [axes]
    vmax = float(np.max(np.abs(rec["As"])) + 1e-9)
    for ax, ti in zip(axes, idxs):
        im = ax.imshow(rec["As"][ti], cmap="RdBu_r", vmin=-vmax, vmax=vmax)
        tag = " (blank)" if rec["grids_blanked"][ti] else ""
        ax.set_title(f"A_{ti}{tag}", fontsize=9)
        ax.set_xticks([]); ax.set_yticks([])
    cbar = fig.colorbar(im, ax=axes, fraction=0.025, pad=0.02)
    cbar.set_label("A_t entry", fontsize=8)
    fig.suptitle("Per-episode fast-weights matrix A_t (LRU-style decay+outer-product)",
                 fontsize=10)
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


def plot_hidden_trace(rec: dict, outpath: str) -> None:
    """h_t evolution over the episode."""
    T = rec["T"]
    H = rec["hs"].shape[1]
    fig, ax = plt.subplots(figsize=(1.0 + 0.35 * T, 0.05 * H + 1.5))
    im = ax.imshow(rec["hs"][1:].T, cmap="RdBu_r", vmin=-1, vmax=1,
                   aspect="auto", interpolation="nearest")
    blanked_t = [t for t in range(T) if rec["grids_blanked"][t]]
    if blanked_t:
        ax.axvspan(min(blanked_t) - 0.5, max(blanked_t) + 0.5,
                   color="black", alpha=0.10)
    ax.set_xlabel("timestep")
    ax.set_ylabel("hidden unit")
    ax.set_title("hidden state h_t (shaded region = obs blanked)")
    fig.colorbar(im, ax=ax, fraction=0.02, pad=0.02)
    fig.tight_layout()
    fig.savefig(outpath, dpi=130, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def main() -> None:
    p = argparse.ArgumentParser(description="Visualize catch-game results.")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=24)
    p.add_argument("--blank-after", type=int, default=8)
    p.add_argument("--n-episodes", type=int, default=12000,
                   help="trains TWO models (FW and no-FW); ~2-3 minutes "
                        "wallclock at the default")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--beta-ent", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--batch-episodes", type=int, default=16)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    _ensure_outdir(args.outdir)
    print(f"# size={args.size}  blank_after={args.blank_after}  "
          f"hidden={args.hidden}  episodes={args.n_episodes}")

    print("[1/2] training fast-weights model ...", flush=True)
    fw_model, fw_hist = _train_one(args.size, args.blank_after,
                                    args.n_episodes, args.hidden, args.lr,
                                    args.batch_episodes, args.beta_ent,
                                    args.value_coef, args.seed,
                                    use_fast_weights=True, eval_every=250)
    print(f"  done. final eval (history-tail) = {fw_hist['eval_catch_rate'][-1]*100:.1f}%  "
          f"({fw_hist['wallclock']:.1f}s)")

    print("[2/2] training vanilla-RNN baseline ...", flush=True)
    no_model, no_hist = _train_one(args.size, args.blank_after,
                                    args.n_episodes, args.hidden, args.lr,
                                    args.batch_episodes, args.beta_ent,
                                    args.value_coef, args.seed,
                                    use_fast_weights=False, eval_every=250)
    print(f"  done. final eval (history-tail) = {no_hist['eval_catch_rate'][-1]*100:.1f}%  "
          f"({no_hist['wallclock']:.1f}s)")

    # Final greedy eval on a larger sample for the comparison plot
    print("[final eval] n=500 greedy each model ...", flush=True)
    final_fw = _evaluate(fw_model, CatchEnv, size=args.size,
                         blank_after=args.blank_after,
                         n_episodes=500, seed=args.seed + 50000, greedy=True)
    final_no = _evaluate(no_model, CatchEnv, size=args.size,
                         blank_after=args.blank_after,
                         n_episodes=500, seed=args.seed + 50000, greedy=True)
    chance = 3.0 / args.size
    print(f"  with FW   : {final_fw*100:.1f}%")
    print(f"  vanilla   : {final_no*100:.1f}%")
    print(f"  chance    : {chance*100:.1f}%  (3-cell paddle / N columns)")

    print("[plot] training curves ...")
    plot_training_curves(fw_hist, os.path.join(args.outdir, "training_curves.png"))
    plot_with_vs_without(fw_hist, no_hist, final_fw, final_no, chance,
                         os.path.join(args.outdir, "with_vs_without.png"))

    print("[plot] example episode + A_t + h_t for the FW model ...")
    # Pick a spawn column far from the paddle's start so the agent must
    # actually move (and remember) to catch.
    rec = _record_episode(fw_model, args.size, args.blank_after,
                          seed=args.seed + 4242,
                          spawn_col=args.size - 2,
                          greedy=True)
    plot_example_episode(rec, os.path.join(args.outdir, "example_episode.png"))
    plot_fast_weights_evolution(rec, os.path.join(args.outdir,
                                                   "fast_weights_evolution.png"))
    plot_hidden_trace(rec, os.path.join(args.outdir, "hidden_state_trace.png"))

    print(f"\n=== done. wrote {len(os.listdir(args.outdir))} files into {args.outdir}/ ===")


if __name__ == "__main__":
    main()
