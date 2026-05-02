"""
8-3-8 encoder — Boltzmann-machine reproduction of the experiment from
Ackley, Hinton & Sejnowski, "A learning algorithm for Boltzmann machines",
Cognitive Science 9 (1985).

Problem:
  Two groups of 8 visible binary units (V1, V2) connected through 3 hidden
  binary units (H). Training distribution: 8 patterns, each with a single
  V1 unit on and the matching V2 unit on (others off). The 3 hidden units
  must self-organize into a 3-bit code that maps the 8 patterns onto the
  8 corners of {0,1}^3.

  This is the *theoretical-minimum-capacity* version: 3 hidden bits = log2(8).
  The paper reports 16/20 (80%) successful runs; getting 8 distinct corners
  is harder than the 4-corner 4-2-4 case because every corner must be used.

Architecture:
  Bipartite (V <-> H only), 16 visible + 3 hidden = 19 units total.
  Indices 0..7 = V1, 8..15 = V2, 16..18 = H. With this layout, V1 and V2
  communicate only through H.

Learning:
  CD-k (Hinton 2002), the standard fast surrogate for Boltzmann learning
  on bipartite networks. Same gradient form as the 1985 rule:

      Delta w_ij  proportional to  <v_i h_j>_data  -  <v_i h_j>_model

  but the model expectation is taken from k Gibbs steps instead of full
  simulated annealing.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

N_PATTERNS = 8
N_GROUP = 8        # |V1| = |V2| = 8
N_HIDDEN = 3       # log2(8)


def make_encoder_data() -> np.ndarray:
    """Return 8 patterns of length 16 = (V1, V2). Pattern i has V1[i]=V2[i]=1."""
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


class EncoderRBM:
    """16 visible <-> 3 hidden bipartite Boltzmann machine, trained with CD-k."""

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

    # ---- conditional sampling -------------------------------------------

    def hidden_prob(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.b_h)

    def visible_prob(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.b_v)

    def sample_h_given_v(self, v: np.ndarray, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        prob = sigmoid((v @ self.W + self.b_h) / T)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    def sample_v_given_h(self, h: np.ndarray, T: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        prob = sigmoid((h @ self.W.T + self.b_v) / T)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    # ---- one CD-k learning step -----------------------------------------

    def cd_step(self, batch: np.ndarray, k: int = 1,
                anneal_schedule: tuple[float, ...] | None = None
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gradient estimates (dW, db_v, db_h) from a CD-k pass.

        With `anneal_schedule = (T_0, T_1, ..., T_{k-1})`, the negative-phase
        Gibbs steps are taken at decreasing temperature -- a discrete
        simulated-annealing variant of CD-k. This is closer to the 1985
        Boltzmann-machine procedure than vanilla CD-k (which always samples
        at T=1) and noticeably helps the 8-3-8 case escape local minima.
        """
        if k < 1:
            raise ValueError(f"cd_step requires k >= 1 (got k={k})")
        if anneal_schedule is not None and len(anneal_schedule) != k:
            raise ValueError(f"anneal_schedule length ({len(anneal_schedule)}) "
                             f"must equal k ({k}).")
        # Positive phase always uses T=1 (data temperature).
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
    # With only 3 hidden units, H has just 8 states, so p(H | V1) and
    # p(V2 | V1) are exactly computable by enumeration -- no Gibbs needed.
    # The closed form (V2 marginalized out of p(V1, V2, H)):
    #
    #     p(H | V1) propto exp(V1^T W_v1 H + b_h^T H) *
    #                      prod_i (1 + exp((W_v2 H + b_v2)_i))
    #
    # which we evaluate in log space.

    def _h_state_table(self) -> np.ndarray:
        return np.array([[(idx >> j) & 1 for j in range(self.n_hidden)]
                         for idx in range(2 ** self.n_hidden)],
                        dtype=np.float32)

    def hidden_posterior_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact p(H | V1) -- length 2^n_hidden. Marginalizes over V2."""
        ng = self.n_group
        W1 = self.W[:ng]
        W2 = self.W[ng:]
        bv2 = self.b_v[ng:]
        h_states = self._h_state_table()
        # per-h-unit input from V1
        v1_input = v1 @ W1                       # shape (n_hidden,)
        log_p = np.empty(len(h_states), dtype=np.float64)
        for s, h in enumerate(h_states):
            h_input = float(v1_input @ h + self.b_h @ h)
            v2_in = W2 @ h + bv2
            log_p[s] = h_input + np.log1p(np.exp(np.clip(v2_in, -50, 50))).sum()
        log_p -= log_p.max()
        p = np.exp(log_p)
        return (p / p.sum()).astype(np.float32)

    def hidden_code_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact marginal p(H_j = 1 | V1) for each hidden unit j."""
        p_h = self.hidden_posterior_exact(v1)
        h_states = self._h_state_table()
        return (p_h[:, None] * h_states).sum(axis=0)

    def dominant_code(self, v1: np.ndarray) -> tuple[int, ...]:
        """Argmax-of-posterior dominant H state as an int tuple."""
        p_h = self.hidden_posterior_exact(v1)
        h_states = self._h_state_table()
        return tuple(int(x) for x in h_states[int(np.argmax(p_h))])

    def reconstruct_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact marginal p(V2_i = 1 | V1) for each i, by enumerating H."""
        p_h = self.hidden_posterior_exact(v1)
        h_states = self._h_state_table()
        ng = self.n_group
        W2 = self.W[ng:]
        bv2 = self.b_v[ng:]
        v2 = np.zeros(ng, dtype=np.float32)
        for s, h in enumerate(h_states):
            v2 += p_h[s] * sigmoid(W2 @ h + bv2)
        return v2

    # ---- sampled inference (kept for animation) -------------------------

    def hidden_code(self, v1: np.ndarray, n_steps: int = 30) -> np.ndarray:
        """Clamp V1, run alternating Gibbs steps, return mean hidden activations.

        Stochastic; for evaluation prefer `hidden_code_exact`. This sampled
        version is kept for the per-frame animation, since it shows the
        chain converging in real time.
        """
        ng = self.n_group
        v = np.zeros(self.n_visible, dtype=np.float32)
        v[:ng] = v1
        v[ng:] = 0.5
        for t in range(n_steps):
            h_prob, h_sample = self.sample_h_given_v(v[None, :])
            h = h_sample if t < n_steps - 1 else h_prob
            v_prob, v_sample = self.sample_v_given_h(h)
            v = v_sample[0] if t < n_steps - 1 else v_prob[0]
            v[:ng] = v1
        return self.hidden_prob(v[None, :])[0]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 4000,
          lr: float = 0.1,
          momentum: float = 0.5,
          weight_decay: float = 1e-4,
          k: int = 5,
          init_scale: float = 0.3,
          batch_repeats: int = 16,
          seed: int = 0,
          perturb_after: int = 250,
          max_restarts: int = 20,
          anneal_T_start: float = 1.0,
          anneal_T_end: float = 1.0,
          target_h_mean: float = 0.5,
          sparsity_weight: float = 5.0,
          snapshot_callback=None,
          snapshot_every: int = 25,
          verbose: bool = True) -> tuple[EncoderRBM, dict]:
    """Train the 8-3-8 encoder with CD-k.

    The 8-3-8 encoder must use ALL 8 corners of {0,1}^3 -- there is zero
    slack. This is harder than the 4-2-4 case (4 of 4 corners) because the
    map from patterns to corners has to be exhaustive: any two patterns
    sharing a corner is a permanent failure. Local minima where two or
    more patterns share a code dominate, so we wrap CD-k with a
    plateau-restart loop similar to the 4-2-4 implementation.
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
        print(f"# 8-3-8 encoder: {N_PATTERNS} patterns, "
              f"{rbm.n_visible} visible + {rbm.n_hidden} hidden")

    # Geometric anneal schedule on the negative-phase Gibbs chain. The first
    # few negative samples explore the model distribution at high temperature
    # (helps escape local minima where two patterns share a code), then the
    # last step pulls back to T=1 so the gradient targets the model we will
    # eventually evaluate at.
    if anneal_T_start <= 0 or anneal_T_end <= 0:
        raise ValueError("Anneal temperatures must be positive.")
    if k == 1:
        anneal_schedule = (anneal_T_end,)
    else:
        ratio = (anneal_T_end / anneal_T_start) ** (1.0 / (k - 1))
        anneal_schedule = tuple(anneal_T_start * (ratio ** s) for s in range(k))

    epochs_since_improvement = 0
    best_codes_this_attempt = 0

    for epoch in range(n_epochs):
        t0 = time.time()
        for _ in range(batch_repeats):
            order = rbm.rng.permutation(N_PATTERNS)
            batch = data[order]
            dW, dbv, dbh = rbm.cd_step(batch, k=k,
                                       anneal_schedule=anneal_schedule)
            # Sparsity penalty pushing each hidden unit's data-phase mean
            # activation toward `target_h_mean`. With 8 patterns and 3
            # hidden bits, the natural target is 0.5 (4 patterns activate
            # each unit on average). Encourages all corners to be used.
            if sparsity_weight > 0:
                h_prob = rbm.hidden_prob(batch)
                h_mean = h_prob.mean(axis=0)
                # gradient of -0.5 * (h_mean - target)^2 w.r.t. W and b_h
                # via chain rule through sigmoid (derivative h*(1-h)).
                d_act = (target_h_mean - h_mean)         # shape (n_hidden,)
                # propagate through batch:
                # h_mean_j = mean_b sigmoid((batch @ W)_bj + b_h_j)
                # dh_mean_j / dW_ij = mean_b batch[b,i] * h_prob*(1-h_prob)
                grad_factor = h_prob * (1 - h_prob)      # (B, n_hidden)
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

        acc = evaluate(rbm, data)
        codes = np.array([rbm.hidden_code_exact(data[i, :N_GROUP])
                          for i in range(N_PATTERNS)])
        sep = mean_pairwise_distance(codes)
        recon = reconstruction_error(rbm, data)
        n_codes = n_distinct_codes(rbm, data)

        history["epoch"].append(epoch + 1)
        history["acc"].append(acc)
        history["weight_norm"].append(float(np.linalg.norm(rbm.W)))
        history["code_separation"].append(float(sep))
        history["reconstruction_error"].append(float(recon))
        history["n_distinct_codes"].append(int(n_codes))

        # Plateau = "no improvement in best-codes-seen-this-attempt for
        # `perturb_after` epochs". This is gentler than "any epoch below
        # 8 codes counts" because the network often climbs 1->4->5->6->7
        # before getting stuck, and we want to give the climbing phase room.
        if n_codes > best_codes_this_attempt:
            best_codes_this_attempt = n_codes
            epochs_since_improvement = 0
        else:
            epochs_since_improvement += 1

        if n_codes == N_PATTERNS:
            epochs_since_improvement = 0  # solved; keep training without restart

        if epochs_since_improvement >= perturb_after and n_codes < N_PATTERNS:
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
            best_codes_this_attempt = 0
            if verbose:
                print(f"epoch {epoch+1:4d}  *** restart {n_done+1} from fresh "
                      f"independent init (n_codes={n_codes}/{N_PATTERNS}) ***")

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  acc={acc*100:5.1f}%  "
                  f"|W|={np.linalg.norm(rbm.W):.3f}  "
                  f"sep={sep:.3f}  recon={recon:.3f}  "
                  f"distinct_codes={n_codes}/{N_PATTERNS}  "
                  f"({time.time()-t0:.3f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, rbm, history)

    return rbm, history


def evaluate(rbm: EncoderRBM, data: np.ndarray) -> float:
    """Exact reconstruction accuracy: argmax of marginal p(V2 | V1)."""
    correct = 0
    for v in data:
        v2_pred = rbm.reconstruct_exact(v[:rbm.n_group])
        if int(np.argmax(v2_pred)) == int(np.argmax(v[:rbm.n_group])):
            correct += 1
    return correct / len(data)


def n_distinct_codes(rbm: EncoderRBM, data: np.ndarray) -> int:
    """Count distinct dominant H states across the patterns.

    The encoder is "solved" iff all patterns map to distinct dominant
    hidden states. For 8-3-8 this means all 8 corners of {0,1}^3 are used.
    """
    dominants = [rbm.dominant_code(v[:rbm.n_group]) for v in data]
    return len(set(dominants))


def codes_used(rbm: EncoderRBM) -> int:
    """Headline metric: how many of the 2^H corners did the network adopt?

    For 8-3-8 the maximum (and target) is 8.
    """
    data = make_encoder_data()
    return n_distinct_codes(rbm, data)


def reconstruction_error(rbm: EncoderRBM, data: np.ndarray) -> float:
    """Mean squared error between marginal p(V2 | V1) and the true V2 one-hot."""
    err = 0.0
    for v in data:
        v2_pred = rbm.reconstruct_exact(v[:rbm.n_group])
        err += float(np.mean((v2_pred - v[rbm.n_group:]) ** 2))
    return err / len(data)


def mean_pairwise_distance(codes: np.ndarray) -> float:
    n = codes.shape[0]
    total = 0.0
    pairs = 0
    for i in range(n):
        for j in range(i + 1, n):
            total += float(np.linalg.norm(codes[i] - codes[j]))
            pairs += 1
    return total / max(pairs, 1)


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-cycles", type=int, default=4000,
                   help="Training epochs (called 'cycles' in the 1985 paper).")
    p.add_argument("--epochs", type=int, default=None,
                   help="Alias for --n-cycles.")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--decay", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=5, help="CD-k steps")
    p.add_argument("--repeats", type=int, default=16,
                   help="batches per epoch")
    p.add_argument("--init-scale", type=float, default=0.3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-after", type=int, default=250)
    p.add_argument("--max-restarts", type=int, default=20)
    p.add_argument("--sparsity-weight", type=float, default=5.0)
    args = p.parse_args()

    n_epochs = args.epochs if args.epochs is not None else args.n_cycles

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

    data = make_encoder_data()
    final_acc = evaluate(rbm, data)
    n_codes = codes_used(rbm)
    print(f"\nFinal accuracy: {final_acc*100:.1f}%")
    print(f"Distinct hidden codes: {n_codes}/{N_PATTERNS}  "
          f"(success = {n_codes == N_PATTERNS})")
    print("\nDominant H state per pattern:")
    for i in range(N_PATTERNS):
        dom = rbm.dominant_code(data[i, :N_GROUP])
        p_h = rbm.hidden_posterior_exact(data[i, :N_GROUP])
        print(f"  pattern {i}: H = {dom}  (prob {p_h.max():.3f})")


if __name__ == "__main__":
    main()
