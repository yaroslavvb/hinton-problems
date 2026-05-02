"""
40-10-40 encoder -- Boltzmann-machine reproduction of the experiment from
Ackley, Hinton & Sejnowski, "A learning algorithm for Boltzmann machines",
Cognitive Science 9 (1985).

Problem:
  Two groups of 40 visible binary units (V1, V2) connected through 10 hidden
  binary units (H). Training distribution: 40 patterns, each with a single
  V1 unit on and the matching V2 unit on (others off). The 10 hidden units
  must self-organize into a code that maps the 40 patterns to 40 distinct
  hidden states drawn from the 2^10 = 1024 corners of {0,1}^10.

  Unlike 8-3-8 (zero slack: 8 patterns -> 8 of 8 corners) this version has
  generous slack (40 patterns / 1024 corners). The headline results from
  the 1985 paper for this scale are:

    1. Asymptotic reconstruction accuracy ~98.6% with sufficient Gibbs sweeps.
    2. Graceful speed/accuracy tradeoff at retrieval: accuracy degrades
       smoothly as the Gibbs-sweep budget shrinks.

  We measure both: `evaluate_exact` for the asymptotic limit (computed by
  enumerating the 1024 hidden states), and `speed_accuracy_curve` for the
  sampled-Gibbs version at varying budgets.

Architecture:
  Bipartite (V <-> H only), 80 visible + 10 hidden = 90 units total.
  Indices 0..39 = V1, 40..79 = V2, 80..89 = H. V1 and V2 communicate only
  through H.

Learning:
  CD-k (Hinton 2002), the standard fast surrogate for Boltzmann learning
  on bipartite networks. Same gradient form as the 1985 rule:

      Delta w_ij  proportional to  <v_i h_j>_data  -  <v_i h_j>_model

  but the model expectation is taken from k Gibbs steps instead of full
  simulated annealing. We add a sparsity penalty (target 50% activation
  per hidden unit) and a plateau-restart loop, both lifted from the
  sibling 8-3-8 recipe.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Constants
# ----------------------------------------------------------------------

N_PATTERNS = 40
N_GROUP = 40        # |V1| = |V2| = 40
N_HIDDEN = 10       # 2^10 = 1024 corners; 40 patterns leave generous slack


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_encoder_data() -> np.ndarray:
    """Return 40 patterns of length 80 = (V1, V2). Pattern i has V1[i]=V2[i]=1."""
    data = np.zeros((N_PATTERNS, 2 * N_GROUP), dtype=np.float32)
    for i in range(N_PATTERNS):
        data[i, i] = 1.0
        data[i, N_GROUP + i] = 1.0
    return data


# ----------------------------------------------------------------------
# RBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def softplus(x: np.ndarray) -> np.ndarray:
    return np.log1p(np.exp(np.clip(x, -50, 50)))


class EncoderRBM:
    """80 visible <-> 10 hidden bipartite Boltzmann machine, trained with CD-k."""

    def __init__(self, n_visible: int = 2 * N_GROUP, n_hidden: int = N_HIDDEN,
                 n_group: int = N_GROUP,
                 init_scale: float = 0.1, seed: int = 0):
        if n_visible != 2 * n_group:
            raise ValueError(f"n_visible ({n_visible}) must equal 2 * n_group "
                             f"({n_group}).")
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_group = n_group
        self.rng = np.random.default_rng(seed)
        self.W = (init_scale * self.rng.standard_normal((n_visible, n_hidden))
                  ).astype(np.float32)
        self.b_v = np.zeros(n_visible, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)
        # Cache the 2^H corner table -- it's used in every exact eval and
        # never changes.
        self._h_table_cache = self._build_h_state_table()

    # ---- conditional sampling -------------------------------------------

    def hidden_prob(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.b_h)

    def visible_prob(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.b_v)

    def sample_h_given_v(self, v: np.ndarray, T: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        prob = sigmoid((v @ self.W + self.b_h) / T)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    def sample_v_given_h(self, h: np.ndarray, T: float = 1.0
                         ) -> tuple[np.ndarray, np.ndarray]:
        prob = sigmoid((h @ self.W.T + self.b_v) / T)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    # ---- one CD-k learning step -----------------------------------------

    def cd_step(self, batch: np.ndarray, k: int = 1,
                anneal_schedule: tuple[float, ...] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gradient estimates (dW, db_v, db_h) from a CD-k pass.

        With `anneal_schedule = (T_0, ..., T_{k-1})`, the negative-phase Gibbs
        steps run at decreasing temperature -- a discrete simulated-annealing
        variant of CD-k. Closer to the 1985 procedure than vanilla CD-k.
        """
        if k < 1:
            raise ValueError(f"cd_step requires k >= 1 (got k={k})")
        if anneal_schedule is not None and len(anneal_schedule) != k:
            raise ValueError(f"anneal_schedule length ({len(anneal_schedule)}) "
                             f"must equal k ({k}).")
        h_prob_pos, h_pos = self.sample_h_given_v(batch)
        v_neg = batch
        h_prob_neg = h_prob_pos
        for step in range(k):
            T = 1.0 if anneal_schedule is None else anneal_schedule[step]
            _, v_neg = self.sample_v_given_h(h_pos, T=T)
            h_prob_neg, h_pos = self.sample_h_given_v(v_neg, T=T)

        n = batch.shape[0]
        dW = (batch.T @ h_prob_pos - v_neg.T @ h_prob_neg) / n
        db_v = (batch - v_neg).mean(axis=0)
        db_h = (h_prob_pos - h_prob_neg).mean(axis=0)
        return dW, db_v, db_h

    # ---- exact inference (marginalize over H) ---------------------------
    #
    # With 10 hidden units, H has 1024 states. p(H | V1) and p(V2 | V1) are
    # both exactly computable by enumeration. The closed form (V2
    # marginalized out of p(V1, V2, H)):
    #
    #     p(H | V1) propto exp(V1^T W_v1 H + b_h^T H) *
    #                      prod_i (1 + exp((W_v2 H + b_v2)_i))
    #
    # which we evaluate in log space.

    def _build_h_state_table(self) -> np.ndarray:
        return np.array([[(idx >> j) & 1 for j in range(self.n_hidden)]
                         for idx in range(2 ** self.n_hidden)],
                        dtype=np.float32)

    def _h_state_table(self) -> np.ndarray:
        return self._h_table_cache

    def hidden_posterior_exact_batch(self, v1_batch: np.ndarray) -> np.ndarray:
        """Exact p(H | V1) for a batch of patterns. Shape (B, 2^n_hidden).

        Vectorized over patterns -- much faster than calling the per-pattern
        version 40 times. Used by `evaluate_exact` and `accuracy_per_pattern`.
        """
        ng = self.n_group
        W1 = self.W[:ng]
        W2 = self.W[ng:]
        bv2 = self.b_v[ng:]
        H = self._h_state_table()                              # (S, h)
        S = H.shape[0]

        # V1 contribution per (pattern, state): (B, h) @ (h, S) -> (B, S)
        v1_input = v1_batch @ W1                               # (B, h)
        v1_state = v1_input @ H.T                              # (B, S)
        # Hidden bias per state: (S,)
        bh_state = H @ self.b_h                                # (S,)
        # V2 partition contribution per state: sum_i softplus((W2 H + bv2)_i)
        v2_in = H @ W2.T + bv2                                 # (S, ng)
        v2_state = softplus(v2_in).sum(axis=1)                 # (S,)
        log_p = v1_state + bh_state[None, :] + v2_state[None, :]  # (B, S)
        log_p -= log_p.max(axis=1, keepdims=True)
        p = np.exp(log_p)
        p = p / p.sum(axis=1, keepdims=True)
        return p.astype(np.float32)

    def hidden_posterior_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact p(H | V1) -- length 2^n_hidden."""
        return self.hidden_posterior_exact_batch(v1[None, :])[0]

    def hidden_code_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact marginal p(H_j = 1 | V1) for each hidden unit j."""
        p_h = self.hidden_posterior_exact(v1)
        H = self._h_state_table()
        return (p_h[:, None] * H).sum(axis=0)

    def dominant_code(self, v1: np.ndarray) -> tuple[int, ...]:
        p_h = self.hidden_posterior_exact(v1)
        H = self._h_state_table()
        return tuple(int(x) for x in H[int(np.argmax(p_h))])

    def reconstruct_exact_batch(self, v1_batch: np.ndarray) -> np.ndarray:
        """Exact marginal p(V2 | V1) for a batch of V1 patterns. Shape (B, ng).

        Each entry is sum over hidden states of p(H | V1) * sigmoid(W2 H + bv2).
        """
        ng = self.n_group
        W2 = self.W[ng:]
        bv2 = self.b_v[ng:]
        H = self._h_state_table()
        # State-wise V2 probabilities: (S, ng)
        v2_per_state = sigmoid(H @ W2.T + bv2)
        p_h = self.hidden_posterior_exact_batch(v1_batch)      # (B, S)
        return (p_h @ v2_per_state).astype(np.float32)         # (B, ng)

    def reconstruct_exact(self, v1: np.ndarray) -> np.ndarray:
        return self.reconstruct_exact_batch(v1[None, :])[0]

    # ---- sampled inference (the speed/accuracy headline) ----------------

    def reconstruct_sampled(self, v1_batch: np.ndarray, n_sweeps: int,
                            n_trials: int = 1,
                            T: float = 1.0,
                            seed: int | None = None
                            ) -> tuple[np.ndarray, np.ndarray]:
        """Sampled retrieval via `n_sweeps` Gibbs sweeps with V1 clamped.

        Returns `(v2_samples, v2_prob)`:
          - `v2_samples`: shape (B, n_trials, ng) -- the binary V2 samples
            from each chain at the end of the last sweep.
          - `v2_prob`: shape (B, ng) -- trial-averaged V2 conditional
            probability after the last H sample. Useful as a denoised
            retrieval; equivalent to running many independent chains and
            taking the mean of `p(V2 | H_last)`.

        Each "sweep" = (sample V given H, with V1 clamped) -> (sample H | V).

        `T` is the retrieval temperature (1.0 = the trained model). Higher
        `T` softens the conditional posteriors and makes single-chain
        recovery noisier; the speed/accuracy curve at `T > 1` shows a
        clearer "more sweeps = better accuracy" tradeoff than at `T = 1`
        where well-separated codes resolve in 1-2 sweeps.
        """
        ng = self.n_group
        B = v1_batch.shape[0]
        v1_tiled = np.repeat(v1_batch, n_trials, axis=0)       # (B*T_, ng)
        BT = v1_tiled.shape[0]

        local_rng = (self.rng if seed is None
                     else np.random.default_rng(seed))

        # Start V2 from independent uniform-bernoulli (mean 0.5).
        v2 = (local_rng.random((BT, ng)) < 0.5).astype(np.float32)
        v = np.concatenate([v1_tiled, v2], axis=1)
        prob = sigmoid((v @ self.W + self.b_h) / T)
        h = (local_rng.random(prob.shape) < prob).astype(np.float32)

        v2_sample_last = v2.copy()
        v2_prob_last = np.zeros((BT, ng), dtype=np.float32)
        for s in range(n_sweeps):
            v_prob_full = sigmoid((h @ self.W.T + self.b_v) / T)
            v2 = (local_rng.random((BT, ng)) < v_prob_full[:, ng:]).astype(np.float32)
            v[:, :ng] = v1_tiled
            v[:, ng:] = v2
            h_prob = sigmoid((v @ self.W + self.b_h) / T)
            h = (local_rng.random(h_prob.shape) < h_prob).astype(np.float32)
            if s == n_sweeps - 1:
                v_prob_full = sigmoid((h @ self.W.T + self.b_v) / T)
                v2_prob_last = v_prob_full[:, ng:]
                v2_sample_last = v2

        v2_prob = v2_prob_last.reshape(B, n_trials, ng).mean(axis=1)
        v2_samples = v2_sample_last.reshape(B, n_trials, ng)
        return v2_samples.astype(np.float32), v2_prob.astype(np.float32)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 2000,
          lr: float = 0.1,
          momentum: float = 0.5,
          weight_decay: float = 1e-4,
          k: int = 5,
          init_scale: float = 0.3,
          batch_repeats: int = 8,
          seed: int = 0,
          perturb_after: int = 250,
          max_restarts: int = 10,
          anneal_T_start: float = 1.0,
          anneal_T_end: float = 1.0,
          target_h_mean: float = 0.5,
          sparsity_weight: float = 5.0,
          eval_every: int = 25,
          snapshot_callback=None,
          snapshot_every: int = 50,
          verbose: bool = True) -> tuple[EncoderRBM, dict]:
    """Train the 40-10-40 encoder with CD-k.

    Differences from 8-3-8:
      * Larger weights (10 hidden units): default `sparsity_weight` is gentler
        because we don't need every corner used (40 / 1024 has slack).
      * Eval is throttled to every `eval_every` epochs because each exact
        eval enumerates 1024 hidden states across 40 patterns.
      * Plateau detector watches the *exact reconstruction accuracy* rather
        than n_distinct_codes: with 1024 corners, code distinctness saturates
        early but accuracy keeps climbing.
    """
    seed_seq = np.random.SeedSequence(seed)
    train_seed, *restart_seeds = seed_seq.spawn(max_restarts + 1)
    rbm = EncoderRBM(seed=int(train_seed.generate_state(1)[0]),
                     init_scale=init_scale)
    vW = np.zeros_like(rbm.W)
    vbv = np.zeros_like(rbm.b_v)
    vbh = np.zeros_like(rbm.b_h)

    data = make_encoder_data()
    history = {"epoch": [], "acc": [], "weight_norm": [],
               "code_separation": [], "reconstruction_error": [],
               "n_distinct_codes": [], "perturbations": []}

    if verbose:
        print(f"# 40-10-40 encoder: {N_PATTERNS} patterns, "
              f"{rbm.n_visible} visible + {rbm.n_hidden} hidden")

    if anneal_T_start <= 0 or anneal_T_end <= 0:
        raise ValueError("Anneal temperatures must be positive.")
    if k == 1:
        anneal_schedule = (anneal_T_end,)
    else:
        ratio = (anneal_T_end / anneal_T_start) ** (1.0 / (k - 1))
        anneal_schedule = tuple(anneal_T_start * (ratio ** s) for s in range(k))

    epochs_since_improvement = 0
    best_acc_this_attempt = 0.0
    last_acc = 0.0
    last_n_codes = 0
    last_sep = 0.0
    last_recon = 0.0

    for epoch in range(n_epochs):
        t0 = time.time()
        for _ in range(batch_repeats):
            order = rbm.rng.permutation(N_PATTERNS)
            batch = data[order]
            dW, dbv, dbh = rbm.cd_step(batch, k=k,
                                       anneal_schedule=anneal_schedule)
            if sparsity_weight > 0:
                h_prob = rbm.hidden_prob(batch)
                h_mean = h_prob.mean(axis=0)
                d_act = (target_h_mean - h_mean)               # (n_hidden,)
                grad_factor = h_prob * (1 - h_prob)            # (B, n_hidden)
                dW_sp = (batch.T @ grad_factor) / batch.shape[0] * d_act
                db_h_sp = grad_factor.mean(axis=0) * d_act
                dW = dW + sparsity_weight * dW_sp
                dbh = dbh + sparsity_weight * db_h_sp
            vW = momentum * vW + lr * (dW - weight_decay * rbm.W)
            vbv = momentum * vbv + lr * dbv
            vbh = momentum * vbh + lr * dbh
            rbm.W += vW
            rbm.b_v += vbv
            rbm.b_h += vbh

        # Throttled eval: skip enumeration on most epochs.
        do_eval = (epoch % eval_every == 0
                   or epoch == n_epochs - 1
                   or epoch < eval_every)
        if do_eval:
            v1 = data[:, :N_GROUP]
            p_h_batch = rbm.hidden_posterior_exact_batch(v1)
            v2_pred = rbm.reconstruct_exact_batch(v1)
            preds = np.argmax(v2_pred, axis=1)
            truth = np.argmax(data[:, N_GROUP:], axis=1)
            last_acc = float((preds == truth).mean())
            # Distinct dominant codes (weak ceiling at min(40, 1024)).
            H = rbm._h_state_table()
            dominants = H[np.argmax(p_h_batch, axis=1)]
            unique = {tuple(int(x) for x in row) for row in dominants}
            last_n_codes = len(unique)
            # Code separation -- mean L2 between exact hidden marginals.
            codes_marginal = p_h_batch @ H                     # (B, h)
            last_sep = mean_pairwise_distance(codes_marginal)
            # Reconstruction MSE.
            last_recon = float(((v2_pred - data[:, N_GROUP:]) ** 2).mean())

        history["epoch"].append(epoch + 1)
        history["acc"].append(last_acc)
        history["weight_norm"].append(float(np.linalg.norm(rbm.W)))
        history["code_separation"].append(last_sep)
        history["reconstruction_error"].append(last_recon)
        history["n_distinct_codes"].append(last_n_codes)

        # Plateau detector based on accuracy improvement.
        if last_acc > best_acc_this_attempt + 1e-6:
            best_acc_this_attempt = last_acc
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if last_acc >= 0.999:
            epochs_since_improvement = 0  # solved; no restart needed

        if epochs_since_improvement >= perturb_after and last_acc < 0.999:
            n_done = len(history["perturbations"])
            if n_done >= len(restart_seeds):
                if verbose:
                    print(f"epoch {epoch+1:4d}  *** restart budget exhausted ***")
                break
            restart_rng = np.random.default_rng(restart_seeds[n_done])
            rbm.W = (init_scale * restart_rng.standard_normal(rbm.W.shape)
                     ).astype(np.float32)
            rbm.b_v *= 0
            rbm.b_h *= 0
            rbm.rng = restart_rng
            vW *= 0; vbv *= 0; vbh *= 0
            history["perturbations"].append(epoch + 1)
            epochs_since_improvement = 0
            best_acc_this_attempt = 0.0
            if verbose:
                print(f"epoch {epoch+1:4d}  *** restart {n_done+1} from fresh "
                      f"independent init (acc={last_acc*100:.1f}%) ***")

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  acc={last_acc*100:5.1f}%  "
                  f"|W|={np.linalg.norm(rbm.W):.3f}  "
                  f"sep={last_sep:.3f}  recon={last_recon:.3f}  "
                  f"distinct_codes={last_n_codes}/{N_PATTERNS}  "
                  f"({time.time()-t0:.3f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, rbm, history)

    return rbm, history


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def evaluate_exact(rbm: EncoderRBM, data: np.ndarray | None = None) -> float:
    """Asymptotic accuracy: argmax of marginal p(V2 | V1) via 1024-state
    enumeration. This is the limit a sampled retrieval converges to.
    """
    if data is None:
        data = make_encoder_data()
    v1 = data[:, :rbm.n_group]
    v2_pred = rbm.reconstruct_exact_batch(v1)
    preds = np.argmax(v2_pred, axis=1)
    truth = np.argmax(data[:, rbm.n_group:], axis=1)
    return float((preds == truth).mean())


# Alias matching the stub signature (`speed_accuracy_curve(model, data)`).
def speed_accuracy_curve(rbm: EncoderRBM,
                         data: np.ndarray | None = None,
                         sweep_budgets: tuple[int, ...] = (
                             1, 2, 4, 8, 16, 32, 64, 128, 256, 512),
                         n_trials: int = 50,
                         T: float = 1.0,
                         seed: int = 0,
                         mode: str = "per_trial"
                         ) -> list[tuple[int, float]]:
    """Accuracy as a function of Gibbs-sweep budget at retrieval.

    For each `n_sweeps`, clamp V1, run `n_sweeps` alternating Gibbs sweeps
    on (V2, H) for `n_trials` independent chains. Compute accuracy two ways:

    - `mode="per_trial"` (default): each chain's final binary V2 sample is
      checked for "argmax matches truth." We report the fraction of chains
      that recover the right pattern. This is what a single Gibbs retrieval
      (no averaging) would deliver.
    - `mode="averaged"`: average the V2 conditional probabilities across
      chains and check the argmax against truth. This is the
      ensemble-of-chains accuracy and saturates near 100% quickly.

    `T` is the retrieval temperature (1.0 = the trained model). Higher
    `T` softens the conditional posteriors and lowers the plateau the
    sampled accuracy approaches.

    Returns a list of (n_sweeps, accuracy) tuples.

    The 1985 paper's headline for 40-10-40 is "98.6% asymptotic accuracy
    with sufficient sweeps." This curve is the empirical version: how
    accuracy approaches that asymptote as the sweep budget grows.
    """
    if data is None:
        data = make_encoder_data()
    v1 = data[:, :rbm.n_group]
    truth = np.argmax(data[:, rbm.n_group:], axis=1)            # (B,)
    out = []
    for n_sweeps in sweep_budgets:
        samples, prob = rbm.reconstruct_sampled(v1, n_sweeps=n_sweeps,
                                                n_trials=n_trials,
                                                T=T, seed=seed)
        if mode == "per_trial":
            # samples shape (B, n_trials, ng)
            preds = np.argmax(samples, axis=2)                  # (B, T)
            correct = (preds == truth[:, None])                 # (B, T)
            acc = float(correct.mean())
        elif mode == "averaged":
            preds = np.argmax(prob, axis=1)
            acc = float((preds == truth).mean())
        else:
            raise ValueError(f"unknown mode {mode!r}")
        out.append((int(n_sweeps), acc))
    return out


def n_distinct_codes(rbm: EncoderRBM, data: np.ndarray | None = None) -> int:
    """Distinct dominant H states across the patterns (max = N_PATTERNS)."""
    if data is None:
        data = make_encoder_data()
    v1 = data[:, :rbm.n_group]
    p_h = rbm.hidden_posterior_exact_batch(v1)
    H = rbm._h_state_table()
    dominants = H[np.argmax(p_h, axis=1)]
    return len({tuple(int(x) for x in row) for row in dominants})


def mean_pairwise_distance(codes: np.ndarray) -> float:
    n = codes.shape[0]
    if n < 2:
        return 0.0
    diffs = codes[:, None, :] - codes[None, :, :]
    dists = np.linalg.norm(diffs, axis=2)
    iu = np.triu_indices(n, k=1)
    return float(dists[iu].mean())


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=2000,
                   help="Training epochs (called 'cycles' in the 1985 paper).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Alias for --n-cycles.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--decay", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=5, help="CD-k Gibbs steps")
    p.add_argument("--repeats", type=int, default=8,
                   help="batches per epoch")
    p.add_argument("--init-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-after", type=int, default=250)
    p.add_argument("--max-restarts", type=int, default=10)
    p.add_argument("--sparsity-weight", type=float, default=5.0)
    p.add_argument("--gibbs-sweeps", type=int, default=200,
                   help="Sweeps used by the sampled retrieval reported at "
                        "the end of training.")
    p.add_argument("--print-curve", action="store_true",
                   help="Also print the speed/accuracy curve.")
    args = p.parse_args()

    n_epochs = args.epochs if args.epochs is not None else args.n_cycles

    t0 = time.time()
    rbm, history = train(n_epochs=n_epochs,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.decay,
                         k=args.k,
                         batch_repeats=args.repeats,
                         init_scale=args.init_scale,
                         seed=args.seed,
                         perturb_after=args.perturb_after,
                         max_restarts=args.max_restarts,
                         sparsity_weight=args.sparsity_weight)
    train_secs = time.time() - t0

    data = make_encoder_data()
    asy = evaluate_exact(rbm, data)
    samples, v2_prob = rbm.reconstruct_sampled(data[:, :N_GROUP],
                                               n_sweeps=args.gibbs_sweeps,
                                               n_trials=50, T=1.0,
                                               seed=args.seed + 1)
    truth = np.argmax(data[:, N_GROUP:], axis=1)
    per_trial_correct = (np.argmax(samples, axis=2) == truth[:, None])
    sampled_acc = float(per_trial_correct.mean())

    print(f"\nTraining wall-clock: {train_secs:.1f} s  "
          f"({n_epochs} epochs, restarts={history['perturbations']})")
    print(f"Asymptotic accuracy (exact, 1024-state enumeration): "
          f"{asy*100:.1f}%")
    print(f"Sampled accuracy ({args.gibbs_sweeps} sweeps x 50 trials): "
          f"{sampled_acc*100:.1f}%")
    print(f"Distinct dominant codes: {n_distinct_codes(rbm)}/{N_PATTERNS}")

    if args.print_curve:
        curve = speed_accuracy_curve(rbm, data, seed=args.seed + 1)
        print("\nSpeed/accuracy curve (sweeps -> accuracy):")
        for n, a in curve:
            print(f"  sweeps={n:4d}  acc={a*100:5.1f}%")


if __name__ == "__main__":
    main()
