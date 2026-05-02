"""
3-bit even-parity ensemble — Boltzmann-machine reproduction of the negative
result motivating the encoder problems in Ackley, Hinton & Sejnowski,
"A learning algorithm for Boltzmann machines", Cognitive Science 9 (1985).

Problem:
  Three visible binary units. Training distribution: 4 even-parity patterns
  at uniform p=0.25 -- {000, 011, 101, 110}. The other 4 patterns (odd
  parity) have target probability 0.

Why this matters:
  For the 3-bit even-parity ensemble:
    <v_i>_data           = 0.5  for every i
    <v_i v_j>_data       = 0.25 for every i != j
  These are exactly the moments of the *uniform* distribution over all 8
  patterns. A visible-only Boltzmann machine has ONLY first- and second-order
  parameters (biases + pairwise weights), so its model expectations match the
  data expectations exactly when p_model is uniform. The gradient
      Delta b_i  = <v_i>_data       - <v_i>_model
      Delta W_ij = <v_i v_j>_data   - <v_i v_j>_model
  drives the model to the uniform distribution and stops. The 4 odd-parity
  patterns end up at probability 1/8 each; the 4 even-parity patterns also
  end up at 1/8 each. Half the mass is on the wrong patterns.

  Adding hidden units breaks the symmetry: the joint p(v, h) carries third-
  order interactions when h is marginalized out, and these CAN distinguish
  parity. This file implements both variants and the comparison is the
  pedagogical payoff.

Architecture:
  --n-hidden 0  : pure visible Boltzmann, exact gradient over 2^3 = 8 states
  --n-hidden K  : bipartite RBM (3 visible <-> K hidden), trained with CD-k
                  (Hinton 2002). Evaluation enumerates the 2^(3+K) joint
                  states for an exact marginal p(v).
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

ALL_PATTERNS = np.array([[(idx >> b) & 1 for b in range(3)]
                         for idx in range(8)], dtype=np.float32)
# parity of pattern i = sum of bits mod 2; even-parity = parity 0
PARITY = ALL_PATTERNS.sum(axis=1).astype(int) % 2
EVEN_MASK = (PARITY == 0)
ODD_MASK = (PARITY == 1)


def make_parity_data() -> np.ndarray:
    """Return the 4 even-parity 3-bit patterns at uniform p=0.25."""
    return ALL_PATTERNS[EVEN_MASK].copy()


def target_distribution() -> np.ndarray:
    """Return target p(v) over all 8 patterns: 0.25 for even, 0 for odd."""
    p = np.zeros(8, dtype=np.float32)
    p[EVEN_MASK] = 0.25
    return p


# ----------------------------------------------------------------------
# Visible-only Boltzmann (the negative result)
# ----------------------------------------------------------------------

class VisibleBoltzmann:
    """Pure visible Boltzmann machine: 3 binary units, biases + pairwise W.

    Energy:
        E(v) = -b . v - sum_{i<j} W_ij v_i v_j

    The full distribution p(v) = exp(-E(v)) / Z is computed exactly by
    enumeration of 2^3 = 8 states, so the gradient is exact (no Gibbs
    sampling needed). This isolates the *representational* failure from any
    sampling noise: even with the optimal gradient, a visible-only model
    cannot match the parity ensemble.
    """

    def __init__(self, n_visible: int = 3, init_scale: float = 0.05,
                 seed: int = 0):
        self.n_visible = n_visible
        self.rng = np.random.default_rng(seed)
        self.b = (init_scale * self.rng.standard_normal(n_visible)
                  ).astype(np.float32)
        # symmetric, zero diagonal -- only n*(n-1)/2 free params
        W = init_scale * self.rng.standard_normal((n_visible, n_visible))
        W = 0.5 * (W + W.T)
        np.fill_diagonal(W, 0.0)
        self.W = W.astype(np.float32)

    def model_distribution(self) -> np.ndarray:
        """Exact p(v) over the 8 patterns."""
        v = ALL_PATTERNS  # (8, 3)
        # log p ~ b . v + 0.5 v^T W v
        bilinear = 0.5 * np.einsum("ni,ij,nj->n", v, self.W, v)
        log_p = v @ self.b + bilinear
        log_p -= log_p.max()
        p = np.exp(log_p)
        return (p / p.sum()).astype(np.float32)

    def fit_step(self, lr: float = 0.1) -> tuple[float, float]:
        """One full-batch exact-gradient step. Returns (KL, log-likelihood)."""
        target = target_distribution()
        model = self.model_distribution()

        # data expectations (over the 4 even-parity patterns)
        data_v = ALL_PATTERNS[EVEN_MASK].mean(axis=0)
        data_vv = (ALL_PATTERNS[EVEN_MASK].T @ ALL_PATTERNS[EVEN_MASK]) / EVEN_MASK.sum()

        # model expectations (weighted over all 8 patterns)
        model_v = (model[:, None] * ALL_PATTERNS).sum(axis=0)
        model_vv = ALL_PATTERNS.T @ (ALL_PATTERNS * model[:, None])

        db = data_v - model_v
        dW = data_vv - model_vv
        dW = 0.5 * (dW + dW.T)
        np.fill_diagonal(dW, 0.0)

        self.b += lr * db
        self.W += lr * dW

        kl = float(_kl(target, model))
        ll = float(np.sum(target * np.log(np.clip(model, 1e-12, 1.0))))
        return kl, ll


# ----------------------------------------------------------------------
# Hidden-unit RBM (the fix)
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class ParityRBM:
    """3 visible <-> K hidden bipartite Boltzmann machine, trained with CD-k.

    With even one hidden unit the joint p(v, h) can carry effective triple
    interactions in the marginal p(v). In practice K=4 is comfortable for
    parity-3.
    """

    def __init__(self, n_visible: int = 3, n_hidden: int = 4,
                 init_scale: float = 0.1, seed: int = 0):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.rng = np.random.default_rng(seed)
        self.W = (init_scale * self.rng.standard_normal((n_visible, n_hidden))
                  ).astype(np.float32)
        self.b_v = np.zeros(n_visible, dtype=np.float32)
        self.b_h = np.zeros(n_hidden, dtype=np.float32)

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

    def cd_step(self, batch: np.ndarray, k: int = 1
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (dW, db_v, db_h) from a CD-k pass."""
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

    def model_distribution(self) -> np.ndarray:
        """Exact marginal p(v) over the 8 patterns by enumerating hidden.

        log p(v) = -F(v) - log Z, where free energy
            F(v) = -b_v . v - sum_j softplus(W^T v + b_h)_j
        """
        v = ALL_PATTERNS
        wx = v @ self.W + self.b_h           # (8, K)
        # softplus, numerically stable
        sp = np.log1p(np.exp(-np.abs(wx))) + np.maximum(wx, 0)
        free_energy = -(v @ self.b_v) - sp.sum(axis=1)
        log_p = -free_energy
        log_p -= log_p.max()
        p = np.exp(log_p)
        return (p / p.sum()).astype(np.float32)


# ----------------------------------------------------------------------
# Training loops
# ----------------------------------------------------------------------

def train_visible(n_steps: int = 400, lr: float = 0.1, seed: int = 0,
                  snapshot_callback=None, snapshot_every: int = 10,
                  verbose: bool = True
                  ) -> tuple[VisibleBoltzmann, dict]:
    """Train a visible-only Boltzmann via exact gradient.

    The KL converges to log(8/4) = log(2) ≈ 0.693 because the model collapses
    to the uniform distribution: the irreducible loss equals KL(target ||
    uniform).
    """
    bm = VisibleBoltzmann(seed=seed)
    history = {"step": [], "kl": [], "log_lik": [],
               "p_even_total": [], "p_odd_total": []}

    if verbose:
        print(f"# visible-only Boltzmann (n_visible=3, n_hidden=0)")
        print(f"# irreducible KL toward uniform = log(2) ≈ {np.log(2):.4f}")

    for step in range(n_steps):
        kl, ll = bm.fit_step(lr=lr)
        p = bm.model_distribution()
        history["step"].append(step + 1)
        history["kl"].append(kl)
        history["log_lik"].append(ll)
        history["p_even_total"].append(float(p[EVEN_MASK].sum()))
        history["p_odd_total"].append(float(p[ODD_MASK].sum()))

        if verbose and (step % 50 == 0 or step == n_steps - 1):
            print(f"step {step+1:4d}  KL={kl:.4f}  "
                  f"p(even)={p[EVEN_MASK].sum():.3f}  "
                  f"p(odd)={p[ODD_MASK].sum():.3f}")

        if snapshot_callback is not None and (step % snapshot_every == 0
                                              or step == n_steps - 1):
            snapshot_callback(step, bm, history)

    return bm, history


def train_hidden(n_hidden: int = 4,
                 n_epochs: int = 800,
                 lr: float = 0.05,
                 momentum: float = 0.5,
                 weight_decay: float = 1e-4,
                 k: int = 5,
                 init_scale: float = 0.5,
                 batch_repeats: int = 16,
                 seed: int = 0,
                 snapshot_callback=None,
                 snapshot_every: int = 10,
                 verbose: bool = True
                 ) -> tuple[ParityRBM, dict]:
    """Train an RBM with CD-k on the 4 even-parity patterns."""
    rbm = ParityRBM(n_hidden=n_hidden, init_scale=init_scale, seed=seed)
    vW = np.zeros_like(rbm.W)
    vbv = np.zeros_like(rbm.b_v)
    vbh = np.zeros_like(rbm.b_h)

    data = make_parity_data()
    history = {"step": [], "kl": [], "log_lik": [],
               "p_even_total": [], "p_odd_total": []}

    if verbose:
        print(f"# parity RBM (n_visible=3, n_hidden={n_hidden})")

    target = target_distribution()

    for epoch in range(n_epochs):
        for _ in range(batch_repeats):
            order = rbm.rng.permutation(len(data))
            dW, dbv, dbh = rbm.cd_step(data[order], k=k)
            vW = momentum * vW + lr * (dW - weight_decay * rbm.W)
            vbv = momentum * vbv + lr * dbv
            vbh = momentum * vbh + lr * dbh
            rbm.W += vW
            rbm.b_v += vbv
            rbm.b_h += vbh

        p = rbm.model_distribution()
        kl = float(_kl(target, p))
        ll = float(np.sum(target * np.log(np.clip(p, 1e-12, 1.0))))

        history["step"].append(epoch + 1)
        history["kl"].append(kl)
        history["log_lik"].append(ll)
        history["p_even_total"].append(float(p[EVEN_MASK].sum()))
        history["p_odd_total"].append(float(p[ODD_MASK].sum()))

        if verbose and (epoch % 50 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  KL={kl:.4f}  "
                  f"p(even)={p[EVEN_MASK].sum():.3f}  "
                  f"p(odd)={p[ODD_MASK].sum():.3f}")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, rbm, history)

    return rbm, history


# ----------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------

def _kl(p: np.ndarray, q: np.ndarray) -> float:
    """KL(p || q), summed over the support of p only."""
    mask = p > 0
    return float(np.sum(p[mask] * (np.log(p[mask]) -
                                    np.log(np.clip(q[mask], 1e-12, 1.0)))))


def pattern_string(v: np.ndarray) -> str:
    return "".join(str(int(x)) for x in v)


def report(model, kind: str) -> None:
    p = model.model_distribution()
    target = target_distribution()
    kl = _kl(target, p)
    print(f"\n--- {kind} ---")
    print(f"KL(target || model) = {kl:.4f}  "
          f"(uniform baseline = {np.log(2):.4f})")
    print(f"p(even patterns) = {p[EVEN_MASK].sum():.3f}   "
          f"p(odd patterns) = {p[ODD_MASK].sum():.3f}")
    print(f"\n{'pattern':>8}  {'parity':>6}  {'target':>7}  {'model':>7}")
    for idx in range(8):
        v = ALL_PATTERNS[idx]
        par = "even" if PARITY[idx] == 0 else "odd"
        print(f"{pattern_string(v):>8}  {par:>6}  "
              f"{target[idx]:7.3f}  {p[idx]:7.3f}")


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n-hidden", type=int, default=0,
                   help="0 = visible-only Boltzmann (negative result); "
                        ">=1 = bipartite RBM with that many hidden units")
    p.add_argument("--steps", type=int, default=None,
                   help="training steps (visible) or epochs (RBM); "
                        "default 400 / 800")
    p.add_argument("--lr", type=float, default=None,
                   help="learning rate; default 0.1 (visible) / 0.05 (RBM)")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    t0 = time.time()
    if args.n_hidden == 0:
        steps = args.steps if args.steps is not None else 400
        lr = args.lr if args.lr is not None else 0.1
        model, history = train_visible(n_steps=steps, lr=lr, seed=args.seed)
        report(model, "visible-only Boltzmann")
    else:
        steps = args.steps if args.steps is not None else 800
        lr = args.lr if args.lr is not None else 0.05
        model, history = train_hidden(n_hidden=args.n_hidden,
                                      n_epochs=steps, lr=lr, seed=args.seed)
        report(model, f"RBM (n_hidden={args.n_hidden})")
    print(f"\nWall time: {time.time() - t0:.2f} s")
