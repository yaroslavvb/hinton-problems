"""
Bouncing balls (2 balls, TRBM) — reproduction of the synthetic-video benchmark
from Sutskever & Hinton, *"Learning multilevel distributed representations for
high-dimensional sequences"*, AISTATS 2007.

Problem:
  A short video of two balls bouncing around in a box with elastic wall
  collisions. The balls overlap (no ball-ball collisions, per the original
  paper's simplification). Each frame is rendered as a soft-blob occupancy
  map, so pixel intensities are real-valued in [0, 1] and the visible units
  of the RBM are conditionally Bernoulli with these probabilities.

Model:
  Temporal Restricted Boltzmann Machine (TRBM). At every time step t the
  visible / hidden conditional has its biases shifted by directed connections
  from previous hidden state h_{t-1} **and** the previous N visible frames
  V_past = [v_{t-1}; v_{t-2}; ...; v_{t-N}] (the Conditional-RBM family
  Sutskever & Hinton subsume under "TRBM" in the 2007 paper):

      P(h_t = 1 | v_t, V_past, h_{t-1})
          = sigma( W^T v_t  +  b_h  +  W_hh^T h_{t-1}  +  W_vh^T V_past )
      P(v_t = 1 | h_t, V_past, h_{t-1})
          = sigma( W   h_t  +  b_v  +  W_hv^T h_{t-1}  +  W_vv^T V_past )

  W_hh : H x H        (prev hidden    ->  current hidden)
  W_hv : H x V        (prev hidden    ->  current visible)
  W_vh : (N*V) x H    (prev N visibles -> current hidden)   -- velocity
  W_vv : (N*V) x V    (prev N visibles -> current visible)  -- autoregressive

  Two visible lags (N = 2) is the minimum that lets the model infer ball
  velocity. Without it, mean-field rollout collapses to the mean-frame
  marginal; with it, the model can extrapolate one step of motion.

  Training is one-step contrastive divergence (CD-1) per frame, conditional
  on (V_past, h_{t-1}) (treated as given). The hidden state used as h_{t-1}
  for the next frame is the positive-phase mean.

Rollout:
  Given a few seed frames we condition forward through them to get h_seed.
  For each future step we treat the TRBM as a vanilla RBM with biases
  shifted by h_{t-1}, run a short Gibbs chain initialised at the previous
  predicted frame, and emit the mean visible activation.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Physics simulation
# ----------------------------------------------------------------------

def simulate_balls(n_steps: int,
                   n_balls: int = 2,
                   h: int = 16,
                   w: int = 16,
                   ball_radius: float = 1.5,
                   speed: float = 1.0,
                   seed: int = 0,
                   binarise: bool = True) -> np.ndarray:
    """Render T frames of n_balls bouncing in an [0, w] x [0, h] box.

    Returns an array of shape (n_steps, h, w) with values in [0, 1].
    With `binarise=True` (default, matching Sutskever & Hinton 2007) each
    pixel is 1 if it lies within `ball_radius` of any ball centre, else 0.
    With `binarise=False` we render a soft Gaussian blob — useful for
    debugging the physics but harder for the TRBM to track because the
    autoregressive feedback signal smears.

    Wall collisions are perfectly elastic (specular reflection). Ball-ball
    collisions are ignored — the balls pass through one another, matching
    Sutskever & Hinton's original synthetic dataset.
    """
    rng = np.random.default_rng(seed)
    pos = np.empty((n_balls, 2), dtype=np.float64)
    pos[:, 0] = rng.uniform(ball_radius, w - ball_radius, size=n_balls)
    pos[:, 1] = rng.uniform(ball_radius, h - ball_radius, size=n_balls)

    angles = rng.uniform(0.0, 2.0 * np.pi, size=n_balls)
    vel = np.stack([np.cos(angles), np.sin(angles)], axis=-1) * speed

    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)

    frames = np.zeros((n_steps, h, w), dtype=np.float32)
    for t in range(n_steps):
        if binarise:
            f = np.zeros((h, w), dtype=np.float64)
            for b in range(n_balls):
                d2 = (xx - pos[b, 0]) ** 2 + (yy - pos[b, 1]) ** 2
                f = np.maximum(f, (d2 <= ball_radius ** 2).astype(np.float64))
        else:
            f = np.zeros((h, w), dtype=np.float64)
            for b in range(n_balls):
                d2 = (xx - pos[b, 0]) ** 2 + (yy - pos[b, 1]) ** 2
                f += np.exp(-d2 / (ball_radius ** 2))
            f = np.clip(f, 0.0, 1.0)
        frames[t] = f.astype(np.float32)

        # update positions
        pos += vel

        # wall collisions (specular): reflect both position and velocity
        for b in range(n_balls):
            if pos[b, 0] < ball_radius:
                pos[b, 0] = 2.0 * ball_radius - pos[b, 0]
                vel[b, 0] = -vel[b, 0]
            if pos[b, 0] > w - ball_radius:
                pos[b, 0] = 2.0 * (w - ball_radius) - pos[b, 0]
                vel[b, 0] = -vel[b, 0]
            if pos[b, 1] < ball_radius:
                pos[b, 1] = 2.0 * ball_radius - pos[b, 1]
                vel[b, 1] = -vel[b, 1]
            if pos[b, 1] > h - ball_radius:
                pos[b, 1] = 2.0 * (h - ball_radius) - pos[b, 1]
                vel[b, 1] = -vel[b, 1]

    return frames


def make_dataset(n_sequences: int,
                 seq_len: int,
                 n_balls: int = 2,
                 h: int = 16,
                 w: int = 16,
                 ball_radius: float = 1.5,
                 speed: float = 1.0,
                 seed: int = 0,
                 binarise: bool = True) -> np.ndarray:
    """Generate a batch of bouncing-ball sequences. Shape: (n_sequences, T, h*w)."""
    seqs = np.empty((n_sequences, seq_len, h * w), dtype=np.float32)
    for i in range(n_sequences):
        frames = simulate_balls(seq_len, n_balls=n_balls, h=h, w=w,
                                ball_radius=ball_radius, speed=speed,
                                seed=seed + i, binarise=binarise)
        seqs[i] = frames.reshape(seq_len, -1)
    return seqs


# ----------------------------------------------------------------------
# TRBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class TRBM:
    """Temporal RBM with directed h_{t-1} -> h_t and h_{t-1} -> v_t connections.

    Sampling for the visible units is mean-field (Bernoulli with the conditional
    probability), which is the standard choice for binary-pixel image RBMs.
    Hidden units are stochastically sampled in the negative phase (CD-1).
    """

    def __init__(self,
                 n_visible: int,
                 n_hidden: int,
                 n_lag: int = 2,
                 init_scale: float = 0.01,
                 seed: int = 0):
        self.n_v = n_visible
        self.n_h = n_hidden
        self.n_lag = n_lag
        self.rng = np.random.default_rng(seed)
        self.W = (init_scale * self.rng.standard_normal((n_visible, n_hidden))
                  ).astype(np.float32)
        self.W_hh = (init_scale * self.rng.standard_normal((n_hidden, n_hidden))
                     ).astype(np.float32)
        self.W_hv = (init_scale * self.rng.standard_normal((n_hidden, n_visible))
                     ).astype(np.float32)
        # Autoregressive: previous N visibles concatenated, then mapped
        # to current hidden / visible.
        self.W_vh = (init_scale * self.rng.standard_normal(
            (n_lag * n_visible, n_hidden))).astype(np.float32)
        self.W_vv = (init_scale * self.rng.standard_normal(
            (n_lag * n_visible, n_visible))).astype(np.float32)
        self.b_v = np.zeros(n_visible, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)

    # ---- per-frame conditionals ----------------------------------------

    def hidden_prob(self, v: np.ndarray, v_past: np.ndarray,
                    h_prev: np.ndarray) -> np.ndarray:
        """P(h_t = 1 | v_t, V_past, h_{t-1}).

        v: (B, V); v_past: (B, n_lag*V); h_prev: (B, H).
        """
        return sigmoid(v @ self.W + self.b_h
                       + h_prev @ self.W_hh + v_past @ self.W_vh)

    def visible_prob(self, h: np.ndarray, v_past: np.ndarray,
                     h_prev: np.ndarray) -> np.ndarray:
        """P(v_t = 1 | h_t, V_past, h_{t-1}).

        h: (B, H); v_past: (B, n_lag*V); h_prev: (B, H).
        """
        return sigmoid(h @ self.W.T + self.b_v
                       + h_prev @ self.W_hv + v_past @ self.W_vv)

    # ---- one-sequence CD-1 step ----------------------------------------

    def cd1_sequence(self, sequence: np.ndarray) -> tuple[float, np.ndarray]:
        """Run CD-1 over a batch of sequences, return (mean recon MSE, mean h_T).

        sequence: (B, T, V) float32, values in [0, 1].
        Returns the per-frame reconstruction MSE averaged over B and T,
        and accumulates gradients into self._d* for an external optimizer.
        """
        B, T, V = sequence.shape
        H = self.n_h
        N = self.n_lag

        h_prev = np.zeros((B, H), dtype=np.float32)
        # v_past holds the most recent N frames concatenated:
        # [v_{t-1}, v_{t-2}, ..., v_{t-N}] in time-order (most recent first).
        v_past = np.zeros((B, N * V), dtype=np.float32)
        dW = np.zeros_like(self.W)
        dWhh = np.zeros_like(self.W_hh)
        dWhv = np.zeros_like(self.W_hv)
        dWvh = np.zeros_like(self.W_vh)
        dWvv = np.zeros_like(self.W_vv)
        dbv = np.zeros_like(self.b_v)
        dbh = np.zeros_like(self.b_h)

        recon_err = 0.0
        for t in range(T):
            v_pos = sequence[:, t, :]                                 # (B, V)

            # ---- positive phase ----
            h_prob_pos = self.hidden_prob(v_pos, v_past, h_prev)      # (B, H)
            h_pos = (self.rng.random((B, H)).astype(np.float32)
                     < h_prob_pos).astype(np.float32)

            # ---- negative phase: one Gibbs step ----
            v_neg = self.visible_prob(h_pos, v_past, h_prev)          # mean-field
            h_prob_neg = self.hidden_prob(v_neg, v_past, h_prev)

            # ---- gradients (averaged over batch) ----
            dW   += (v_pos.T  @ h_prob_pos - v_neg.T  @ h_prob_neg) / B
            dWhh += (h_prev.T @ h_prob_pos - h_prev.T @ h_prob_neg) / B
            dWvh += (v_past.T @ h_prob_pos - v_past.T @ h_prob_neg) / B
            dWhv += (h_prev.T @ v_pos      - h_prev.T @ v_neg)      / B
            dWvv += (v_past.T @ v_pos      - v_past.T @ v_neg)      / B
            dbv  += (v_pos - v_neg).mean(axis=0)
            dbh  += (h_prob_pos - h_prob_neg).mean(axis=0)

            recon_err += float(np.mean((v_pos - v_neg) ** 2))

            # ---- propagate state: shift v_past, append v_pos at the front ----
            h_prev = h_prob_pos
            if N > 0:
                v_past = np.concatenate([v_pos, v_past[:, :(N - 1) * V]],
                                        axis=1).astype(np.float32)

        self._dW   = dW   / T
        self._dWhh = dWhh / T
        self._dWhv = dWhv / T
        self._dWvh = dWvh / T
        self._dWvv = dWvv / T
        self._dbv  = dbv  / T
        self._dbh  = dbh  / T
        return recon_err / T, h_prev

    # ---- rollout ------------------------------------------------------

    def rollout(self,
                init_frames: np.ndarray,
                n_future: int,
                k_gibbs: int = 5,
                stochastic_v: bool = False,
                feedback: str = "binarise") -> np.ndarray:
        """Predict the next `n_future` frames given `init_frames`.

        Uses the standard one-step mean-field inference for a CRBM:
        given (V_past, h_{t-1}) we get the *prior* hidden mean,
        feed it through the visible bias for an initial v_t guess, and
        refine with `k_gibbs` mean-field passes.

        `feedback` controls how the predicted v_t is folded back into
        V_past for the next step:
          "mean":     feed the soft (mean-field) probability   - smears fast
          "binarise": threshold at 0.5                         - sharp, default
          "sample":   sample Bernoulli(v_prob)                 - sharp, stochastic
        Binarising the feedback empirically extends the rollout horizon by
        several frames before the prediction collapses to the mean-frame,
        because the autoregressive matrices W_vv / W_vh were trained on
        binary v_{t-1}.

        init_frames: (T_init, V) float32
        Returns rollout of shape (n_future, V) — the *soft* prediction at
        each step (the binarisation is only used for feedback).
        """
        V = init_frames.shape[1]
        H = self.n_h
        N = self.n_lag
        h_prev = np.zeros(H, dtype=np.float32)
        v_past = np.zeros(N * V, dtype=np.float32)

        # Run forward through the seed to update (v_past, h_prev)
        for v in init_frames:
            h_prev = self.hidden_prob(v[None, :], v_past[None, :],
                                      h_prev[None, :])[0]
            v_past = np.concatenate(
                [v.astype(np.float32), v_past[:(N - 1) * V]])

        rollout = np.empty((n_future, V), dtype=np.float32)
        for step in range(n_future):
            h_prev_b = h_prev[None, :]
            v_past_b = v_past[None, :]

            # Prior on h_t given (V_past, h_{t-1}) — no v_t yet
            h_prior = sigmoid(self.b_h
                              + h_prev_b @ self.W_hh
                              + v_past_b @ self.W_vh)

            # Initial v_t mean-field guess from h_prior
            v = sigmoid(h_prior @ self.W.T + self.b_v
                        + h_prev_b @ self.W_hv
                        + v_past_b @ self.W_vv)

            # Refine with a few mean-field passes
            for _ in range(k_gibbs):
                h_prob = self.hidden_prob(v, v_past_b, h_prev_b)
                if stochastic_v:
                    h = (self.rng.random(h_prob.shape).astype(np.float32)
                         < h_prob).astype(np.float32)
                else:
                    h = h_prob
                v = self.visible_prob(h, v_past_b, h_prev_b)

            v_curr = v[0]
            rollout[step] = v_curr
            # Pick the version of v that gets fed back
            if feedback == "binarise":
                v_feedback = (v_curr > 0.5).astype(np.float32)
            elif feedback == "sample":
                v_feedback = (self.rng.random(v_curr.shape).astype(np.float32)
                              < v_curr).astype(np.float32)
            else:
                v_feedback = v_curr.astype(np.float32)
            # Update (v_past, h_prev) using the predicted frame
            h_prev = self.hidden_prob(v_feedback[None, :], v_past_b, h_prev_b)[0]
            v_past = np.concatenate(
                [v_feedback, v_past[:(N - 1) * V]])
        return rollout


def build_trbm(n_visible: int, n_hidden: int, n_lag: int = 2,
               init_scale: float = 0.01, seed: int = 0) -> TRBM:
    return TRBM(n_visible=n_visible, n_hidden=n_hidden, n_lag=n_lag,
                init_scale=init_scale, seed=seed)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(model: TRBM,
          sequences: np.ndarray,
          n_epochs: int = 30,
          lr: float = 0.05,
          momentum: float = 0.5,
          weight_decay: float = 1e-4,
          batch_size: int = 10,
          verbose: bool = True) -> dict:
    """SGD with momentum on the per-sequence CD-1 gradients."""
    B_total = sequences.shape[0]

    vW = np.zeros_like(model.W)
    vWhh = np.zeros_like(model.W_hh)
    vWhv = np.zeros_like(model.W_hv)
    vWvh = np.zeros_like(model.W_vh)
    vWvv = np.zeros_like(model.W_vv)
    vbv = np.zeros_like(model.b_v)
    vbh = np.zeros_like(model.b_h)

    history = {"epoch": [], "recon_mse": [], "weight_norm": []}

    if verbose:
        print(f"# TRBM training: V={model.n_v}, H={model.n_h}, "
              f"sequences={B_total}, T={sequences.shape[1]}, "
              f"epochs={n_epochs}, lr={lr}, batch_size={batch_size}")

    for epoch in range(n_epochs):
        t0 = time.time()
        order = model.rng.permutation(B_total)
        epoch_err = 0.0
        n_batches = 0
        for start in range(0, B_total, batch_size):
            idx = order[start:start + batch_size]
            batch = sequences[idx]
            err, _ = model.cd1_sequence(batch)
            epoch_err += err
            n_batches += 1

            vW   = momentum * vW   + lr * (model._dW   - weight_decay * model.W)
            vWhh = momentum * vWhh + lr * (model._dWhh - weight_decay * model.W_hh)
            vWhv = momentum * vWhv + lr * (model._dWhv - weight_decay * model.W_hv)
            vWvh = momentum * vWvh + lr * (model._dWvh - weight_decay * model.W_vh)
            vWvv = momentum * vWvv + lr * (model._dWvv - weight_decay * model.W_vv)
            vbv  = momentum * vbv  + lr * model._dbv
            vbh  = momentum * vbh  + lr * model._dbh

            model.W    += vW
            model.W_hh += vWhh
            model.W_hv += vWhv
            model.W_vh += vWvh
            model.W_vv += vWvv
            model.b_v  += vbv
            model.b_h  += vbh

        epoch_err /= max(n_batches, 1)
        history["epoch"].append(epoch + 1)
        history["recon_mse"].append(float(epoch_err))
        history["weight_norm"].append(float(np.linalg.norm(model.W)))

        if verbose and (epoch % max(1, n_epochs // 20) == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  recon_mse={epoch_err:.5f}  "
                  f"|W|={np.linalg.norm(model.W):.3f}  "
                  f"({time.time()-t0:.2f}s)")
    return history


# ----------------------------------------------------------------------
# Rollout helper for CLI
# ----------------------------------------------------------------------

def rollout(model: TRBM, init_frames: np.ndarray, n_future: int,
            k_gibbs: int = 5, feedback: str = "sample") -> np.ndarray:
    return model.rollout(init_frames, n_future, k_gibbs=k_gibbs,
                         feedback=feedback)


def evaluate_rollout(model: TRBM,
                     test_sequence: np.ndarray,
                     n_seed: int,
                     n_future: int,
                     k_gibbs: int = 5,
                     feedback: str = "sample") -> tuple[float, np.ndarray]:
    """Return (mean MSE, rolled-out frames) for one held-out sequence."""
    seed = test_sequence[:n_seed]
    truth = test_sequence[n_seed:n_seed + n_future]
    pred = model.rollout(seed, n_future, k_gibbs=k_gibbs, feedback=feedback)
    mse = float(np.mean((pred - truth) ** 2))
    return mse, pred


# ----------------------------------------------------------------------
# Environment / reproducibility
# ----------------------------------------------------------------------

def env_info() -> dict:
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL).decode().strip()
    except Exception:
        commit = "unknown"
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor(),
        "git_commit": commit,
    }


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-balls", type=int, default=2)
    p.add_argument("--h", type=int, default=16)
    p.add_argument("--w", type=int, default=16)
    p.add_argument("--ball-radius", type=float, default=1.5)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--n-sequences", type=int, default=40)
    p.add_argument("--seq-len", type=int, default=40)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-lag", type=int, default=2,
                   help="Number of past visible frames in the conditioning context.")
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--batch-size", type=int, default=10)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--decay", type=float, default=1e-4)
    p.add_argument("--init-scale", type=float, default=0.01)
    p.add_argument("--n-seed-frames", type=int, default=10,
                   help="Number of seed frames before rollout (eval).")
    p.add_argument("--n-future", type=int, default=20,
                   help="Number of frames to roll out (eval).")
    p.add_argument("--k-gibbs", type=int, default=5)
    p.add_argument("--feedback", type=str, default="sample",
                   choices=["mean", "binarise", "sample"],
                   help="How predicted v_t is fed back into V_past for the next step.")
    p.add_argument("--results-json", type=str, default=None)
    args = p.parse_args()

    np.random.seed(args.seed)

    print(f"Generating {args.n_sequences} training sequences "
          f"({args.h}x{args.w}, T={args.seq_len})...")
    train_seqs = make_dataset(args.n_sequences, args.seq_len,
                              n_balls=args.n_balls,
                              h=args.h, w=args.w,
                              ball_radius=args.ball_radius,
                              speed=args.speed,
                              seed=args.seed)

    test_seq = make_dataset(1, args.n_seed_frames + args.n_future,
                            n_balls=args.n_balls,
                            h=args.h, w=args.w,
                            ball_radius=args.ball_radius,
                            speed=args.speed,
                            seed=args.seed + 9999)[0]

    n_visible = args.h * args.w
    model = build_trbm(n_visible, args.n_hidden, n_lag=args.n_lag,
                       init_scale=args.init_scale, seed=args.seed)

    t0 = time.time()
    history = train(model, train_seqs,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.decay,
                    batch_size=args.batch_size)
    train_secs = time.time() - t0
    print(f"\nTraining wall-clock: {train_secs:.1f} s")

    print(f"\nEvaluating rollout on held-out sequence...")
    rollout_mse, _ = evaluate_rollout(model, test_seq,
                                      n_seed=args.n_seed_frames,
                                      n_future=args.n_future,
                                      k_gibbs=args.k_gibbs,
                                      feedback=args.feedback)
    print(f"Rollout MSE ({args.n_future} steps): {rollout_mse:.5f}")

    # 20-sequence average for a more honest number
    n_eval = 20
    eval_mses = []
    mean_baseline_mses = []
    last_baseline_mses = []
    for s in range(n_eval):
        ts = make_dataset(1, args.n_seed_frames + args.n_future,
                          n_balls=args.n_balls,
                          h=args.h, w=args.w,
                          ball_radius=args.ball_radius,
                          speed=args.speed,
                          seed=args.seed + 9000 + s)[0]
        gt = ts[args.n_seed_frames:args.n_seed_frames + args.n_future]
        model.rng = np.random.default_rng(args.seed + 7000 + s)
        rp = model.rollout(ts[:args.n_seed_frames], args.n_future,
                           k_gibbs=args.k_gibbs, feedback=args.feedback)
        eval_mses.append(float(np.mean((rp - gt) ** 2)))
        mean_baseline_mses.append(
            float(np.mean((train_seqs.reshape(-1, n_visible).mean(axis=0)
                           - gt) ** 2)))
        last_baseline_mses.append(
            float(np.mean((ts[args.n_seed_frames - 1] - gt) ** 2)))
    print(f"Rollout MSE ({n_eval}-seq avg):     "
          f"{np.mean(eval_mses):.5f}  +/- {np.std(eval_mses):.5f}")

    # Baselines on the same 20 sequences
    print(f"Baseline mean-frame ({n_eval}-seq avg): "
          f"{np.mean(mean_baseline_mses):.5f}")
    print(f"Baseline copy-last  ({n_eval}-seq avg): "
          f"{np.mean(last_baseline_mses):.5f}")
    mean_baseline_mse = float(np.mean(mean_baseline_mses))
    last_baseline_mse = float(np.mean(last_baseline_mses))

    if args.results_json:
        result = {
            "config": vars(args),
            "env": env_info(),
            "train_secs": train_secs,
            "rollout_mse_single_seq": rollout_mse,
            "rollout_mse_20seq_avg": float(np.mean(eval_mses)),
            "rollout_mse_20seq_std": float(np.std(eval_mses)),
            "mean_frame_baseline_mse": mean_baseline_mse,
            "last_frame_baseline_mse": last_baseline_mse,
            "final_recon_mse": history["recon_mse"][-1],
        }
        with open(args.results_json, "w") as f:
            json.dump(result, f, indent=2)
        print(f"\nWrote results to {args.results_json}")
