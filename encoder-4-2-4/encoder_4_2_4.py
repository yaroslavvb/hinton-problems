"""
4-2-4 encoder — Boltzmann-machine reproduction of the experiment from
Ackley, Hinton & Sejnowski, "A learning algorithm for Boltzmann machines",
Cognitive Science 9 (1985).

Problem:
  Two groups of 4 visible binary units (V1, V2) connected through 2 hidden
  binary units (H). Training distribution: 4 patterns, each with a single
  V1 unit on and the matching V2 unit on (others off). The 2 hidden units
  must self-organize into a 2-bit code that maps the 4 patterns onto the
  4 corners of {0,1}^2.

Architecture:
  Bipartite (V <-> H only), 8 visible + 2 hidden = 10 units total.
  Indices 0..3 = V1, 4..7 = V2, 8..9 = H. With this layout, V1 and V2
  communicate only through H.

Learning:
  CD-1 (Hinton 2002), the standard fast surrogate for Boltzmann learning
  on bipartite networks. Same gradient form as the 1985 rule:

      Delta w_ij  proportional to  <v_i h_j>_data  -  <v_i h_j>_model

  but the model expectation is taken from a single Gibbs step instead of
  full simulated annealing.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_encoder_data() -> np.ndarray:
    """Return 4 patterns of length 8 = (V1, V2). Pattern i has V1[i]=V2[i]=1."""
    data = np.zeros((4, 8), dtype=np.float32)
    for i in range(4):
        data[i, i] = 1.0
        data[i, 4 + i] = 1.0
    return data


# ----------------------------------------------------------------------
# RBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class EncoderRBM:
    """8 visible <-> 2 hidden bipartite Boltzmann machine, trained with CD-1."""

    def __init__(self, n_visible: int = 8, n_hidden: int = 2,
                 init_scale: float = 0.05, seed: int = 0):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
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

    def sample_h_given_v(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob = self.hidden_prob(v)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    def sample_v_given_h(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        prob = self.visible_prob(h)
        sample = (self.rng.random(prob.shape) < prob).astype(np.float32)
        return prob, sample

    # ---- one CD-k learning step -----------------------------------------

    def cd_step(self, batch: np.ndarray, k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return gradient estimates (dW, db_v, db_h) from a CD-k pass."""
        if k < 1:
            raise ValueError(f"cd_step requires k >= 1 (got k={k})")
        h_prob_pos, h_pos = self.sample_h_given_v(batch)
        v_neg = batch
        h_prob_neg = h_prob_pos
        for _ in range(k):
            _, v_neg = self.sample_v_given_h(h_pos)
            h_prob_neg, h_pos = self.sample_h_given_v(v_neg)

        n = batch.shape[0]
        dW = (batch.T @ h_prob_pos - v_neg.T @ h_prob_neg) / n
        db_v = (batch - v_neg).mean(axis=0)
        db_h = (h_prob_pos - h_prob_neg).mean(axis=0)
        return dW, db_v, db_h

    # ---- exact inference (marginalize over H) ---------------------------
    #
    # With only 2 hidden units, H has just 4 states, so p(H | V1) and
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
        W1 = self.W[:4]
        W2 = self.W[4:]
        bv2 = self.b_v[4:]
        h_states = self._h_state_table()
        v1_input = float(v1 @ W1[:, 0]), float(v1 @ W1[:, 1])  # per-h-unit
        log_p = np.empty(len(h_states), dtype=np.float64)
        for s, h in enumerate(h_states):
            h_input = (v1_input[0] * h[0] + v1_input[1] * h[1]
                       + float(self.b_h @ h))
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

    def reconstruct_exact(self, v1: np.ndarray) -> np.ndarray:
        """Exact marginal p(V2_i = 1 | V1) for each i, by enumerating H."""
        p_h = self.hidden_posterior_exact(v1)
        h_states = self._h_state_table()
        W2 = self.W[4:]
        bv2 = self.b_v[4:]
        v2 = np.zeros(4, dtype=np.float32)
        for s, h in enumerate(h_states):
            v2 += p_h[s] * sigmoid(W2 @ h + bv2)
        return v2

    # ---- sampled inference (kept for reference / animation) -------------

    def hidden_code(self, v1: np.ndarray, n_steps: int = 30) -> np.ndarray:
        """Clamp V1, run alternating Gibbs steps, return mean hidden activations.

        Stochastic; for evaluation prefer `hidden_code_exact`. This sampled
        version is kept for the per-frame animation, since it shows the
        chain converging in real time.
        """
        v = np.zeros(self.n_visible, dtype=np.float32)
        v[:4] = v1
        v[4:] = 0.5
        for t in range(n_steps):
            h_prob, h_sample = self.sample_h_given_v(v[None, :])
            h = h_sample if t < n_steps - 1 else h_prob
            v_prob, v_sample = self.sample_v_given_h(h)
            v = v_sample[0] if t < n_steps - 1 else v_prob[0]
            v[:4] = v1
        return self.hidden_prob(v[None, :])[0]

    def reconstruct(self, v1: np.ndarray, n_steps: int = 30) -> np.ndarray:
        """Clamp V1, run alternating Gibbs, return mean V2 probabilities.

        Stochastic; prefer `reconstruct_exact` for evaluation.
        """
        v = np.zeros(self.n_visible, dtype=np.float32)
        v[:4] = v1
        v[4:] = 0.5
        for t in range(n_steps):
            h_prob, h_sample = self.sample_h_given_v(v[None, :])
            h = h_sample if t < n_steps - 1 else h_prob
            v_prob, v_sample = self.sample_v_given_h(h)
            v = v_sample[0] if t < n_steps - 1 else v_prob[0]
            v[:4] = v1
        return v[4:]


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 400,
          lr: float = 0.05,
          momentum: float = 0.5,
          weight_decay: float = 1e-4,
          k: int = 5,
          init_scale: float = 0.1,
          batch_repeats: int = 8,
          seed: int = 0,
          perturb_after: int = 80,
          snapshot_callback=None,
          snapshot_every: int = 10,
          verbose: bool = True) -> tuple[EncoderRBM, dict]:
    """Train the 4-2-4 encoder with CD-k.

    `batch_repeats` controls how many CD-k mini-batches per epoch we run. With
    only 4 distinct patterns, repeating the data a few times per epoch gives
    a less noisy gradient.

    The 4-2-4 encoder has well-known local minima where two patterns collapse
    onto the same hidden code. If accuracy stalls below 100% for
    `perturb_after` epochs, we re-initialize the weights and continue. The
    original 1985 paper reported 250/250 convergence under full simulated
    annealing; CD-k on a bipartite RBM is sloppier and benefits from this
    multi-restart wrapper.
    """
    # SeedSequence gives statistically independent child seeds for each
    # restart -- without this, the restart's W draw depends on however many
    # training RNG calls have happened beforehand, so a "bad" seed can keep
    # producing bad restart inits in lockstep.
    seed_seq = np.random.SeedSequence(seed)
    train_seed, *restart_seeds = seed_seq.spawn(64)
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
        print(f"# 4-2-4 encoder: 4 patterns, "
              f"{rbm.n_visible} visible + {rbm.n_hidden} hidden")

    epochs_at_plateau = 0

    for epoch in range(n_epochs):
        t0 = time.time()
        for _ in range(batch_repeats):
            order = rbm.rng.permutation(4)
            dW, dbv, dbh = rbm.cd_step(data[order], k=k)
            vW = momentum * vW + lr * (dW - weight_decay * rbm.W)
            vbv = momentum * vbv + lr * dbv
            vbh = momentum * vbh + lr * dbh
            rbm.W += vW
            rbm.b_v += vbv
            rbm.b_h += vbh

        acc = evaluate(rbm, data)
        codes = np.array([rbm.hidden_code_exact(data[i, :4]) for i in range(4)])
        sep = mean_pairwise_distance(codes)
        recon = reconstruction_error(rbm, data)
        n_codes = n_distinct_codes(rbm, data)

        history["epoch"].append(epoch + 1)
        history["acc"].append(acc)
        history["weight_norm"].append(float(np.linalg.norm(rbm.W)))
        history["code_separation"].append(float(sep))
        history["reconstruction_error"].append(float(recon))
        history["n_distinct_codes"].append(int(n_codes))

        # plateau detection -- use the binary "all 4 codes distinct" signal
        # rather than the (formerly noisy, now exact) accuracy.
        if n_codes < 4:
            epochs_at_plateau += 1
        else:
            epochs_at_plateau = 0

        if epochs_at_plateau >= perturb_after:
            n_done = len(history["perturbations"])
            if n_done >= len(restart_seeds):
                if verbose:
                    print(f"epoch {epoch+1:4d}  *** restart budget exhausted ***")
                break
            # Replace BOTH the weight init and the training RNG so that the
            # post-restart gradient trajectory is statistically independent
            # of the pre-restart one.
            restart_rng = np.random.default_rng(restart_seeds[n_done])
            rbm.W = (init_scale * restart_rng.standard_normal(rbm.W.shape)
                     ).astype(np.float32)
            rbm.b_v *= 0
            rbm.b_h *= 0
            rbm.rng = restart_rng
            vW *= 0; vbv *= 0; vbh *= 0
            history["perturbations"].append(epoch + 1)
            epochs_at_plateau = 0
            if verbose:
                print(f"epoch {epoch+1:4d}  *** restart {n_done+1} from fresh "
                      f"independent init (n_codes={n_codes}/4) ***")

        if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  acc={acc*100:5.1f}%  "
                  f"|W|={np.linalg.norm(rbm.W):.3f}  "
                  f"sep={sep:.3f}  recon={recon:.3f}  "
                  f"distinct_codes={n_codes}/4  "
                  f"({time.time()-t0:.3f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, rbm, history)

    return rbm, history


def evaluate(rbm: EncoderRBM, data: np.ndarray) -> float:
    """Exact reconstruction accuracy: argmax of marginal p(V2 | V1)."""
    correct = 0
    for v in data:
        v2_pred = rbm.reconstruct_exact(v[:4])
        if int(np.argmax(v2_pred)) == int(np.argmax(v[:4])):
            correct += 1
    return correct / len(data)


def n_distinct_codes(rbm: EncoderRBM, data: np.ndarray) -> int:
    """Count distinct dominant H states across the 4 patterns.

    The encoder is "solved" iff all 4 patterns map to distinct dominant
    hidden states (4-corner code). This is binary in expectation -- a
    cleaner plateau signal than a noisy accuracy estimate.
    """
    h_states = rbm._h_state_table()
    dominants = []
    for v in data:
        p_h = rbm.hidden_posterior_exact(v[:4])
        dominants.append(tuple(int(x) for x in h_states[int(np.argmax(p_h))]))
    return len(set(dominants))


def reconstruction_error(rbm: EncoderRBM, data: np.ndarray) -> float:
    """Mean squared error between marginal p(V2 | V1) and the true V2 one-hot."""
    err = 0.0
    for v in data:
        v2_pred = rbm.reconstruct_exact(v[:4])
        err += float(np.mean((v2_pred - v[4:]) ** 2))
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

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--decay", type=float, default=1e-4)
    p.add_argument("--k", type=int, default=5, help="CD-k steps")
    p.add_argument("--repeats", type=int, default=8,
                   help="batches per epoch")
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--perturb-after", type=int, default=80)
    args = p.parse_args()

    rbm, history = train(n_epochs=args.epochs,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.decay,
                         k=args.k,
                         batch_repeats=args.repeats,
                         init_scale=args.init_scale,
                         seed=args.seed,
                         perturb_after=args.perturb_after)

    data = make_encoder_data()
    print(f"\nFinal accuracy: {evaluate(rbm, data)*100:.1f}%")
    print(f"Distinct hidden codes: {n_distinct_codes(rbm, data)}/4")
    print("\nDominant H state per pattern (exact argmax over p(H | V1)):")
    h_states = rbm._h_state_table()
    for i in range(4):
        p_h = rbm.hidden_posterior_exact(data[i, :4])
        dom = h_states[int(np.argmax(p_h))]
        print(f"  pattern {i}: H = ({int(dom[0])}, {int(dom[1])})  "
              f"(prob {p_h.max():.3f})")
    print("\nWeight matrix W (visible x hidden):")
    print(rbm.W)
