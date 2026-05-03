"""
Catch with delayed-blank observations (Ba, Hinton, Mnih, Leibo, Ionescu 2016).

Source:
    J. Ba, G. Hinton, V. Mnih, J. Z. Leibo, C. Ionescu (2016),
    "Using Fast Weights to Attend to the Recent Past", NIPS.
    https://arxiv.org/abs/1610.06258  (section 5: "Reinforcement learning")

Problem:
    A ball drops one pixel per step from the top of an NxN binary grid towards
    a 3-wide paddle on the bottom row. The agent picks one of {left, stay,
    right} at every step. After `blank_after` steps the observation is blanked
    (all zeros) for the rest of the episode -- the agent must REMEMBER the
    ball's column and trajectory and steer the paddle to intercept. Reward is
    +1 on catch, -1 on miss, 0 otherwise. Episode length is exactly N-1 steps
    (ball needs N-1 falls to reach the paddle row).

    With observation always visible the task is trivial. With early blanking
    the agent must hold a state across many "blind" steps. Vanilla RNNs of
    small hidden width forget; fast-weights RNNs hold the trajectory in the
    per-episode matrix A_t and catch reliably.

Architecture (Ba et al. with the same single-LayerNorm body used by the two
sibling fast-weights stubs in this wave):

    A_t  = lambda_decay * A_{t-1} + eta * outer(h_{t-1}, h_{t-1})    (A_0 = 0)
    z_t  = W_h h_{t-1} + W_x x_t + b + A_t @ h_{t-1}
    zn_t = LayerNorm(z_t)         # mean-0 std-1 over H, no learnable affine
    h_t  = tanh(zn_t)
    pi_t = softmax(W_pi h_t + b_pi)        # 3-way policy head
    V_t  = W_v h_t + b_v                    # scalar value baseline

    LayerNorm is required to stop A_t @ h_{t-1} from blowing up (Ba et al.
    "Layer Normalization is critical"). Same finding holds here.

REINFORCE with baseline (deviation from the paper's full A3C):
    The paper runs A3C, which needs distributed actor-learners. We use the
    on-policy single-actor simplification:

        L = sum_t  - advantage_t.detach() * log pi_t[a_t]
                  + 0.5 * (V_t - G_t)^2
                  - beta_ent * H(pi_t)

    with G_t = sum_{k>=t} r_k (gamma=1; episodes are short and bounded) and
    advantage_t = G_t - V_t. This is the classic REINFORCE-with-baseline /
    actor-critic loss; the only thing missing vs A3C is asynchronous parallel
    workers, which buys wall-clock not capability.

BPTT with fast weights and per-timestep loss (slight extension of the
final-step-only siblings):
    Same recurrent backward pass as the fast-weights-associative-retrieval
    sibling, but loss gradient on h_t is injected at EVERY timestep instead
    of only at t=T. dh_t accumulates the gradient flowing back from t+1 plus
    the local gradient from policy + value + entropy heads at t.

CLI:
    python3 catch_game.py --seed 0 --size 24 --blank-after 8 --n-episodes 4000
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np


# ============================================================
# Environment
# ============================================================

class CatchEnv:
    """Drop-the-ball partial-observability catch game.

    Grid is `size x size`. Ball spawns on row 0 at a random column and falls
    one row per step. Paddle is 3 cells wide centered at `paddle_x`, lives on
    row size-1, moves +/-1 per step, clipped to [1, size-2].

    Episode ends when the ball reaches the bottom row (the agent's chosen
    action that step lands the paddle, then catch is checked). Reward is +1
    on catch, -1 on miss, 0 on every interior step. Episode length is
    `size - 1` (steps t=0..size-2; at t=size-2 the ball is on row size-1
    and the catch check fires).

    Observation at step t is an `(size, size)` float32 binary grid with the
    ball pixel and the 3 paddle pixels set to 1.0 -- UNLESS t > blank_after,
    in which case the observation is all zeros. The reward signal is always
    delivered at the end.

    Actions: 0 = left, 1 = stay, 2 = right.
    """

    PADDLE_HALF = 1   # paddle spans [paddle_x-1, paddle_x, paddle_x+1]

    def __init__(self, size: int = 24, blank_after: int = 8,
                 seed: int | None = None):
        self.size = int(size)
        self.blank_after = int(blank_after)
        self.rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._ball_row = 0
        self._ball_col = 0
        self._paddle_x = self.size // 2
        self._done = True

    # --- public api ---

    @property
    def obs_dim(self) -> int:
        return self.size * self.size

    @property
    def n_actions(self) -> int:
        return 3

    @property
    def episode_len(self) -> int:
        return self.size - 1

    def reset(self, seed: int | None = None) -> np.ndarray:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self._step_idx = 0
        self._ball_row = 0
        # ball can spawn anywhere in [0, size-1]; paddle starts in middle
        self._ball_col = int(self.rng.integers(0, self.size))
        self._paddle_x = self.size // 2
        self._done = False
        return self._make_obs()

    def step(self, action: int) -> tuple[np.ndarray, float, bool]:
        """Returns (obs, reward, done). Obs blanked once t > blank_after."""
        if self._done:
            raise RuntimeError("step() called on a done episode; call reset()")

        # 1. apply action to paddle
        a = int(action)
        if a == 0:
            self._paddle_x -= 1
        elif a == 2:
            self._paddle_x += 1
        # action 1 = stay
        lo, hi = self.PADDLE_HALF, self.size - 1 - self.PADDLE_HALF
        if self._paddle_x < lo:
            self._paddle_x = lo
        elif self._paddle_x > hi:
            self._paddle_x = hi

        # 2. drop ball
        self._ball_row += 1

        self._step_idx += 1

        # 3. check terminal (ball reached bottom row)
        reward = 0.0
        if self._ball_row >= self.size - 1:
            self._done = True
            if abs(self._ball_col - self._paddle_x) <= self.PADDLE_HALF:
                reward = 1.0
            else:
                reward = -1.0

        return self._make_obs(), reward, self._done

    # --- introspection helpers used by visualisation ---

    def render_visible(self) -> np.ndarray:
        """Always-visible render of current state (ignores blanking).

        Used only for plotting / GIFs. The agent never sees this.
        """
        grid = np.zeros((self.size, self.size), dtype=np.float32)
        if 0 <= self._ball_row < self.size:
            grid[self._ball_row, self._ball_col] = 1.0
        for dx in (-1, 0, 1):
            x = self._paddle_x + dx
            if 0 <= x < self.size:
                grid[self.size - 1, x] = 1.0
        return grid

    def state(self) -> dict:
        return {"ball_row": self._ball_row, "ball_col": self._ball_col,
                "paddle_x": self._paddle_x, "step_idx": self._step_idx,
                "done": self._done}

    def is_blanked(self) -> bool:
        return self._step_idx > self.blank_after

    # --- internal ---

    def _make_obs(self) -> np.ndarray:
        if self._step_idx > self.blank_after:
            return np.zeros((self.size, self.size), dtype=np.float32)
        return self.render_visible()


# ============================================================
# Model: fast-weights RNN with policy + value heads
# ============================================================

class FastWeightsActorCritic:
    """RNN with per-episode fast-weights matrix, policy + value heads.

    Slow params (learned by BPTT):
      W_h  : (H, H)       recurrent
      W_x  : (H, D)       input (D = obs_dim = size*size)
      b    : (H,)         hidden bias
      W_pi : (A, H)       policy logits
      b_pi : (A,)
      W_v  : (1, H)       value head
      b_v  : (1,)

    Fast weights A_t (H, H) are reset to zero at the start of every episode.
    Set `use_fast_weights=False` (or eta=0) to ablate the fast-weights term;
    that gives the vanilla-RNN baseline used in the with/without comparison.
    """

    def __init__(self, obs_dim: int, hidden: int, n_actions: int,
                 lambda_decay: float = 0.95, eta: float = 0.5,
                 use_fast_weights: bool = True,
                 seed: int = 0):
        self.D = int(obs_dim)
        self.H = int(hidden)
        self.A = int(n_actions)
        self.lambda_decay = float(lambda_decay)
        self.eta = float(eta) if use_fast_weights else 0.0
        self.use_fast_weights = bool(use_fast_weights)
        rng = np.random.default_rng(seed)

        s = 1.0 / np.sqrt(self.H)
        # IRNN-style: W_h identity *0.5 (Le, Jaitly, Hinton 2015), LayerNorm
        # rescales so it doesn't blow up. Same init used in sibling stubs.
        self.W_h = np.eye(self.H) * 0.5
        # W_x init small because input is binary 0/1 with up to ~4 ones set
        self.W_x = rng.standard_normal((self.H, self.D)) * s
        self.b = np.zeros(self.H)
        self.W_pi = rng.standard_normal((self.A, self.H)) * s
        self.b_pi = np.zeros(self.A)
        self.W_v = rng.standard_normal((1, self.H)) * s
        self.b_v = np.zeros(1)

    def n_params(self) -> int:
        return (self.W_h.size + self.W_x.size + self.b.size
                + self.W_pi.size + self.b_pi.size
                + self.W_v.size + self.b_v.size)

    def params(self) -> dict[str, np.ndarray]:
        return {"W_h": self.W_h, "W_x": self.W_x, "b": self.b,
                "W_pi": self.W_pi, "b_pi": self.b_pi,
                "W_v": self.W_v, "b_v": self.b_v}

    # --- forward step (one timestep) ---

    @staticmethod
    def _layernorm(z: np.ndarray, eps: float = 1e-5
                   ) -> tuple[np.ndarray, np.ndarray, float]:
        """Returns (zn, mu_subtracted z, sigma)."""
        mu = z.mean()
        zc = z - mu
        sigma = float(np.sqrt((zc * zc).mean() + eps))
        zn = zc / sigma
        return zn, zn, sigma  # zn returned twice for consistency with caller

    def _step_forward(self, x_t: np.ndarray, h_prev: np.ndarray,
                      A_prev: np.ndarray
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, float]:
        """One RNN step. Returns (h_t, A_t, z_t, zn_t, sigma_t)."""
        if self.eta == 0.0:
            A_t = A_prev  # stays zero throughout
        else:
            A_t = self.lambda_decay * A_prev + self.eta * np.outer(h_prev, h_prev)
        z_t = self.W_h @ h_prev + self.W_x @ x_t + self.b + A_t @ h_prev
        mu = z_t.mean()
        zc = z_t - mu
        sigma = float(np.sqrt((zc * zc).mean() + 1e-5))
        zn_t = zc / sigma
        h_t = np.tanh(zn_t)
        return h_t, A_t, z_t, zn_t, sigma

    def policy_value(self, h_t: np.ndarray) -> tuple[np.ndarray, float]:
        """Returns (probs over actions, scalar value estimate)."""
        logits = self.W_pi @ h_t + self.b_pi
        m = float(logits.max())
        exp = np.exp(logits - m)
        probs = exp / exp.sum()
        v = float((self.W_v @ h_t + self.b_v)[0])
        return probs, v

    # --- run an episode (forward, sample actions) ---

    def rollout(self, env: CatchEnv,
                rng: np.random.Generator | None = None,
                greedy: bool = False,
                reset: bool = True,
                ) -> dict:
        """Full episode. Stores everything needed for BPTT.

        If `reset` is True (default), env.reset() is called with no seed and
        thus uses the env's internal rng. If False, the caller is expected to
        have already reset the env -- pass reset=False if you want a specific
        spawn seed to take effect.
        """
        if rng is None:
            rng = np.random.default_rng()

        H, A_dim = self.H, self.A
        T_max = env.episode_len
        if reset:
            obs = env.reset()
        else:
            # Caller has already called env.reset(seed=...). At t=0 the obs
            # is always the visible grid (blanking only kicks in for
            # step_idx > blank_after >= 0).
            obs = env.render_visible()

        xs = np.zeros((T_max, self.D))
        hs = np.zeros((T_max + 1, H))
        zs = np.zeros((T_max, H))
        zns = np.zeros((T_max, H))
        sigs = np.zeros(T_max)
        As = np.zeros((T_max, H, H))
        probs_all = np.zeros((T_max, A_dim))
        values = np.zeros(T_max)
        actions = np.zeros(T_max, dtype=np.int64)
        rewards = np.zeros(T_max)
        blanked = np.zeros(T_max, dtype=bool)

        A_prev = np.zeros((H, H))
        t = 0
        for t in range(T_max):
            x_t = obs.reshape(-1)
            xs[t] = x_t
            blanked[t] = bool(np.all(x_t == 0.0))

            h_prev = hs[t]
            h_t, A_t, z_t, zn_t, sigma = self._step_forward(x_t, h_prev, A_prev)
            hs[t + 1] = h_t
            As[t] = A_t
            zs[t] = z_t
            zns[t] = zn_t
            sigs[t] = sigma

            probs, v = self.policy_value(h_t)
            probs_all[t] = probs
            values[t] = v
            if greedy:
                a_t = int(np.argmax(probs))
            else:
                a_t = int(rng.choice(A_dim, p=probs))
            actions[t] = a_t

            obs, r, done = env.step(a_t)
            rewards[t] = r
            A_prev = A_t
            if done:
                break

        T = t + 1
        # truncate to actual length
        return {"T": T,
                "xs": xs[:T], "hs": hs[:T + 1],
                "zs": zs[:T], "zns": zns[:T], "sigs": sigs[:T],
                "As": As[:T], "probs": probs_all[:T],
                "values": values[:T], "actions": actions[:T],
                "rewards": rewards[:T], "blanked": blanked[:T]}

    # --- backward (REINFORCE with baseline) ---

    def loss_and_grads(self, ep: dict, gamma: float = 1.0,
                       beta_ent: float = 0.01,
                       value_coef: float = 0.5
                       ) -> tuple[float, float, dict]:
        """One episode's REINFORCE-with-baseline loss + grads on slow params.

        loss = sum_t [ - advantage_t.detach() * log pi_t[a_t]
                       + 0.5 * value_coef * (V_t - G_t)^2
                       - beta_ent * H(pi_t) ]
        """
        T = ep["T"]
        H = self.H
        A_dim = self.A

        rewards = ep["rewards"]
        # discounted returns G_t = sum_{k>=t} gamma^(k-t) r_k
        G = np.zeros(T)
        running = 0.0
        for t in range(T - 1, -1, -1):
            running = rewards[t] + gamma * running
            G[t] = running

        values = ep["values"]
        probs = ep["probs"]
        actions = ep["actions"]

        adv = G - values   # used both as detached weight (policy) and live (value)

        # --- per-step gradient ON h_t from the heads ---
        # Policy:    L_pi_t = -adv_t.detach() * log p_t[a_t]
        #   d L_pi_t / d logits_k = -adv_t.detach() * (1[k==a_t] - p_k)
        # Value:     L_v_t = 0.5 * value_coef * (V_t - G_t)^2
        #   d L_v_t / d V_t = value_coef * (V_t - G_t)
        # Entropy:   L_ent_t = -beta_ent * H(p) = beta_ent * sum_k p_k log p_k
        #   d L_ent_t / d logits_k = beta_ent * p_k * (log p_k - sum_j p_j log p_j)

        d_logits = np.zeros((T, A_dim))   # gradient wrt policy logits per step
        d_v = np.zeros(T)                 # gradient wrt V_t per step
        loss_pi = 0.0
        loss_v = 0.0
        loss_ent = 0.0

        # init param grads
        dW_pi = np.zeros_like(self.W_pi)
        db_pi = np.zeros_like(self.b_pi)
        dW_v = np.zeros_like(self.W_v)
        db_v = np.zeros_like(self.b_v)

        log_probs_clip = np.log(probs + 1e-12)
        for t in range(T):
            p = probs[t]
            a = int(actions[t])
            adv_t = float(adv[t])
            # policy gradient (detach advantage)
            grad_pi_logits = -adv_t * (-p.copy())     # = adv * p  baseline
            grad_pi_logits[a] += -adv_t * (1.0)        # then add 1 at a (with sign)
            # The two lines above implement: -adv_t * (one_hot(a) - p) ; so it
            # expands to: -adv_t * one_hot(a) + adv_t * p. Verify:
            #   grad_pi_logits = adv_t * p  (vector)
            #   grad_pi_logits[a] += -adv_t   -- net: adv_t*p[a] - adv_t at index a
            # which equals -adv_t * (1 - p[a]) at index a, +adv_t * p[k] elsewhere.
            # That is exactly -adv_t * (one_hot(a) - p). Good.
            loss_pi += -adv_t * float(log_probs_clip[t, a])

            # value gradient
            d_v_t = value_coef * (float(values[t]) - float(G[t]))
            d_v[t] = d_v_t
            loss_v += 0.5 * value_coef * (float(values[t]) - float(G[t])) ** 2

            # entropy gradient
            #  H = -sum p log p
            #  d H / d logits = - p * ( log p - sum p log p ) ?  derive:
            # softmax derivative: d p_k / d logits_j = p_k (delta_kj - p_j)
            # H = -sum_k p_k log p_k
            # dH/d logits_j = -sum_k (dp_k * log p_k + p_k * dp_k / p_k)
            #              = -sum_k dp_k (log p_k + 1)
            # plug dp_k/dlogits_j = p_k(delta - p_j):
            #   = -sum_k p_k(delta_kj - p_j)(log p_k + 1)
            #   = -p_j(log p_j + 1) + p_j sum_k p_k(log p_k + 1)
            #   = -p_j(log p_j + 1) + p_j (sum p_k log p_k + 1)
            #   = -p_j log p_j - p_j + p_j sum p_k log p_k + p_j
            #   = -p_j log p_j + p_j sum_k p_k log p_k
            #   = p_j ( - log p_j + sum_k p_k log p_k )
            #   = -p_j ( log p_j - sum_k p_k log p_k )
            # We minimize -beta_ent*H, so gradient on logits is -beta_ent * dH/dlogits:
            #   d (-beta_ent H) / d logits_j = beta_ent * p_j * (log p_j - sum_k p_k log p_k)
            mean_log = float(np.sum(p * log_probs_clip[t]))
            grad_ent_logits = beta_ent * p * (log_probs_clip[t] - mean_log)
            loss_ent += -beta_ent * (-float(np.sum(p * log_probs_clip[t])))  # = beta_ent * sum p log p

            d_logits[t] = grad_pi_logits + grad_ent_logits

            # accumulate output-head param grads
            h_t = ep["hs"][t + 1]
            dW_pi += np.outer(d_logits[t], h_t)
            db_pi += d_logits[t]
            dW_v[0] += d_v_t * h_t
            db_v[0] += d_v_t

        loss = loss_pi + loss_v + loss_ent

        # gradient on h_t from each head:
        #   policy: dh_t += W_pi.T @ d_logits[t]
        #   value:  dh_t += W_v.T * d_v[t]   (W_v is (1,H), so W_v[0] * d_v[t])
        dh_local = (d_logits @ self.W_pi) + d_v[:, None] * self.W_v[0][None, :]   # (T, H)

        # --- recurrent BPTT ---
        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db = np.zeros_like(self.b)

        dh = np.zeros(H)
        dA_running = np.zeros((H, H))
        for t in range(T - 1, -1, -1):
            x_t = ep["xs"][t]
            h_prev = ep["hs"][t]
            h_now = ep["hs"][t + 1]
            A_t = ep["As"][t]

            # local injection from heads at this timestep
            dh = dh + dh_local[t]

            # tanh backward
            dzn = dh * (1.0 - h_now * h_now)

            # LayerNorm backward (no learnable affine):
            # zn = (z - mean(z)) / sigma  =>
            # dz = (dzn - dzn.mean() - zn * (dzn * zn).mean()) / sigma
            sigma = float(ep["sigs"][t])
            zn = ep["zns"][t]
            dz = (dzn - dzn.mean() - zn * (dzn * zn).mean()) / sigma

            # parameter grads
            dW_h += np.outer(dz, h_prev)
            dW_x += np.outer(dz, x_t)
            db += dz

            # backprop through z = W_h h_{t-1} + W_x x + b + A_t h_{t-1}
            dh_prev_partial = self.W_h.T @ dz + A_t.T @ dz
            dA_t_local = np.outer(dz, h_prev)

            dA_t_total = dA_running + dA_t_local

            if self.eta == 0.0:
                # A_t == A_prev == 0, nothing to chain through outer products
                dh = dh_prev_partial
                dA_running = dA_t_total  # propagated as zero anyway
            else:
                # backprop through A_t = lambda A_{t-1} + eta outer(h_{t-1}, h_{t-1})
                dh_prev = (dh_prev_partial
                           + self.eta * (dA_t_total + dA_t_total.T) @ h_prev)
                dh = dh_prev
                dA_running = self.lambda_decay * dA_t_total

        grads = {"W_h": dW_h, "W_x": dW_x, "b": db,
                 "W_pi": dW_pi, "b_pi": db_pi,
                 "W_v": dW_v, "b_v": db_v}
        return float(loss), float(np.sum(rewards)), grads


def build_a3c_policy(obs_dim: int, n_actions: int = 3,
                     hidden: int = 64, lambda_decay: float = 0.95,
                     eta: float = 0.5, use_fast_weights: bool = True,
                     seed: int = 0) -> FastWeightsActorCritic:
    """Per-stub spec factory. We use REINFORCE-with-baseline (single actor),
    documented as a deviation from full A3C in the README."""
    return FastWeightsActorCritic(obs_dim=obs_dim, hidden=hidden,
                                  n_actions=n_actions,
                                  lambda_decay=lambda_decay, eta=eta,
                                  use_fast_weights=use_fast_weights,
                                  seed=seed)


# ============================================================
# Optimizer
# ============================================================

class Adam:
    def __init__(self, params: dict[str, np.ndarray],
                 lr: float = 5e-3, beta1: float = 0.9,
                 beta2: float = 0.999, eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            mhat = self.m[k] / (1.0 - self.beta1 ** self.t)
            vhat = self.v[k] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ============================================================
# Training loop
# ============================================================

def train_a3c(env_cls, model: FastWeightsActorCritic,
              n_episodes: int = 4000, lr: float = 5e-3,
              size: int = 24, blank_after: int = 8,
              gamma: float = 1.0,
              beta_ent: float = 0.01,
              value_coef: float = 0.5,
              grad_clip: float = 5.0,
              batch_episodes: int = 8,
              eval_every: int = 100,
              eval_episodes: int = 100,
              seed: int = 0,
              verbose: bool = True
              ) -> dict:
    """Train via REINFORCE-with-baseline; gradients averaged over a small
    batch of episodes per Adam update.

    `env_cls` is the env class (CatchEnv) so we can spawn fresh envs with
    different seeds. Naming kept as `train_a3c` to match the per-stub spec.
    """
    train_env = env_cls(size=size, blank_after=blank_after, seed=seed + 1)
    rng = np.random.default_rng(seed + 2)
    optim = Adam(model.params(), lr=lr)

    history = {"episode": [], "reward": [], "catch_rate": [],
               "loss": [], "eval_catch_rate": []}

    t0 = time.time()
    running_R = 0.0
    running_loss = 0.0
    running_n = 0

    ep_idx = 0
    while ep_idx < n_episodes:
        # gather a batch
        agg = {k: np.zeros_like(v) for k, v in model.params().items()}
        batch_R = 0.0
        batch_loss = 0.0
        batch_n = min(batch_episodes, n_episodes - ep_idx)
        for _ in range(batch_n):
            train_env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            ep = model.rollout(train_env, rng=rng, greedy=False, reset=False)
            loss, R, grads = model.loss_and_grads(
                ep, gamma=gamma, beta_ent=beta_ent, value_coef=value_coef)
            for k in agg:
                agg[k] += grads[k]
            batch_R += R
            batch_loss += loss
            ep_idx += 1
        for k in agg:
            agg[k] /= batch_n

        # global-norm gradient clip
        gnorm = float(np.sqrt(sum(float(np.sum(g * g)) for g in agg.values())))
        if grad_clip is not None and gnorm > grad_clip:
            scale = grad_clip / (gnorm + 1e-12)
            for k in agg:
                agg[k] *= scale

        optim.step(agg)

        running_R += batch_R
        running_loss += batch_loss
        running_n += batch_n

        if (ep_idx % eval_every == 0) or (ep_idx >= n_episodes):
            mean_R = running_R / max(1, running_n)
            mean_loss = running_loss / max(1, running_n)
            running_R = 0.0; running_loss = 0.0; running_n = 0

            eval_cr = _evaluate(model, env_cls, size=size,
                                blank_after=blank_after,
                                n_episodes=eval_episodes,
                                seed=seed + 99000)
            history["episode"].append(ep_idx)
            history["reward"].append(mean_R)
            history["catch_rate"].append(0.5 * (mean_R + 1.0))
            history["loss"].append(mean_loss)
            history["eval_catch_rate"].append(eval_cr)

            if verbose:
                elapsed = time.time() - t0
                print(f"  ep {ep_idx:5d}  meanR={mean_R:+.3f}  "
                      f"train_catch~{50*(mean_R+1):4.1f}%  "
                      f"eval_catch={eval_cr*100:4.1f}%  "
                      f"loss={mean_loss:+.3f}  ({elapsed:5.1f}s)")

    history["wallclock"] = time.time() - t0
    return history


def _evaluate(model: FastWeightsActorCritic, env_cls,
              size: int, blank_after: int,
              n_episodes: int = 200, seed: int = 12345,
              greedy: bool = True) -> float:
    env = env_cls(size=size, blank_after=blank_after, seed=seed)
    rng = np.random.default_rng(seed + 7)
    n_caught = 0
    for i in range(n_episodes):
        env.reset(seed=seed + i + 1)
        ep = model.rollout(env, rng=rng, greedy=greedy, reset=False)
        if float(ep["rewards"].sum()) > 0:
            n_caught += 1
    return n_caught / max(1, n_episodes)


# ============================================================
# CLI
# ============================================================

def _print_environment() -> None:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL,
                                         text=True).strip()[:10]
    except Exception:
        commit = "unknown"
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}  git {commit}")


def main() -> None:
    p = argparse.ArgumentParser(description="Catch with delayed-blank "
                                            "observations (Ba et al. 2016).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--size", type=int, default=24,
                   help="grid edge length (per-stub spec default 24)")
    p.add_argument("--blank-after", type=int, default=8,
                   help="observation is blanked once step_idx > blank_after "
                        "(per-stub spec default 8)")
    p.add_argument("--n-episodes", type=int, default=12000,
                   help="REINFORCE has high variance; 12k episodes lets the "
                        "FW agent reliably exceed chance at size=24. For a "
                        "much faster sanity check try --size 10 "
                        "--blank-after 4 --n-episodes 1500.")
    p.add_argument("--hidden", type=int, default=64)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--lr", type=float, default=3e-3)
    p.add_argument("--gamma", type=float, default=1.0)
    p.add_argument("--beta-ent", type=float, default=0.01)
    p.add_argument("--value-coef", type=float, default=0.5)
    p.add_argument("--batch-episodes", type=int, default=16)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--eval-every", type=int, default=200)
    p.add_argument("--eval-episodes", type=int, default=200)
    p.add_argument("--no-fast-weights", action="store_true",
                   help="ablation: zero out the fast-weights term")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    _print_environment()
    print(f"# config: size={args.size}  blank_after={args.blank_after}  "
          f"hidden={args.hidden}  lambda={args.lambda_decay}  eta={args.eta}  "
          f"lr={args.lr}  episodes={args.n_episodes}  seed={args.seed}  "
          f"fast_weights={not args.no_fast_weights}")

    obs_dim = args.size * args.size
    model = build_a3c_policy(obs_dim=obs_dim, n_actions=3,
                             hidden=args.hidden,
                             lambda_decay=args.lambda_decay, eta=args.eta,
                             use_fast_weights=not args.no_fast_weights,
                             seed=args.seed)
    print(f"# n_params: {model.n_params():,}")

    print("\n=== Training ===")
    history = train_a3c(CatchEnv, model,
                        n_episodes=args.n_episodes, lr=args.lr,
                        size=args.size, blank_after=args.blank_after,
                        gamma=args.gamma, beta_ent=args.beta_ent,
                        value_coef=args.value_coef,
                        batch_episodes=args.batch_episodes,
                        grad_clip=args.grad_clip,
                        eval_every=args.eval_every,
                        eval_episodes=args.eval_episodes,
                        seed=args.seed,
                        verbose=not args.quiet)

    final_eval = _evaluate(model, CatchEnv, size=args.size,
                           blank_after=args.blank_after,
                           n_episodes=500, seed=args.seed + 50000,
                           greedy=True)
    print("\n=== Final ===")
    print(f"  eval catch rate (n=500, greedy): {final_eval*100:5.1f}%")
    print(f"  train wallclock                : {history['wallclock']:.1f}s")
    print(f"  n_params                       : {model.n_params():,}")


if __name__ == "__main__":
    main()
