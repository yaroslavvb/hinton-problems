"""
Shifter — reproduction of Hinton & Sejnowski (1986),
"Learning and relearning in Boltzmann machines", PDP Vol. 1, Ch. 7.

Two rings of N binary units V1 and V2 where V2 is V1 shifted by one of
{-1, 0, +1} positions (with wraparound). Three one-hot V3 units encode the
shift class. The network sees all 19 visible bits during training (V1 + V2
+ V3 = 8 + 8 + 3) and must infer V3 from V1+V2 at test time.

Pairwise statistics carry zero information about shift; the hidden units
must discover **third-order conjunctive (position-pair) features** —
"V1[i] on AND V2[(i+s) mod N] on" detectors for the relevant shift class.

Architecture: bipartite RBM with visible = [V1 | V2 | V3] and a layer of
hidden units. Training: Contrastive Divergence (CD-1, Hinton 2002) — same
positive-phase-minus-negative-phase gradient as the 1986 Boltzmann
learning rule, but with the efficient bipartite sampling structure.

This file is a lift of `shifter_rbm.py` from
`cybertronai/sutro-problems/wip-boltzmann-shifter/`, adapted to the
hinton-problems stub layout (default `--hidden 24` to match the spec /
original Figure 3) and with `shift_recognition_accuracy()` exposed as a
top-level function.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_shifter_data(N: int = 8) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All N-bit shifter cases: 2^N patterns x 3 shifts = full enumeration.

    Returns (V1, V2, V3) where each row of V3 is a one-hot indicator of
    the shift in {-1, 0, +1}. For N=8 the shape is (768, 8) / (768, 8) /
    (768, 3).
    """
    patterns = np.array(
        [[(p >> i) & 1 for i in range(N)] for p in range(2**N)],
        dtype=np.float32,
    )
    V1, V2, V3 = [], [], []
    for p in patterns:
        for k, s in enumerate([-1, 0, 1]):
            V1.append(p)
            V2.append(np.roll(p, s))
            y = np.zeros(3, dtype=np.float32)
            y[k] = 1.0
            V3.append(y)
    return np.array(V1), np.array(V2), np.array(V3)


def generate_dataset(n_samples: int | None = None,
                     p_on: float = 0.5,
                     n_bits: int = 8,
                     seed: int = 0):
    """Sampled-distribution dataset (matches the stub signature).

    If `n_samples` is None, returns the full enumeration via
    `make_shifter_data`. Otherwise samples `n_samples` patterns with each
    bit on with probability `p_on`, paired with a uniformly random shift.
    """
    if n_samples is None:
        return make_shifter_data(n_bits)
    rng = np.random.default_rng(seed)
    P = (rng.random((n_samples, n_bits)) < p_on).astype(np.float32)
    shifts = rng.integers(0, 3, size=n_samples)  # 0, 1, 2 -> -1, 0, +1
    V1 = P
    V2 = np.stack([np.roll(P[i], int(shifts[i]) - 1) for i in range(n_samples)])
    V3 = np.zeros((n_samples, 3), dtype=np.float32)
    V3[np.arange(n_samples), shifts] = 1.0
    return V1, V2, V3


# ----------------------------------------------------------------------
# RBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class ShifterRBM:
    """Bipartite Boltzmann machine: visible = V1 || V2 || V3, hidden = H."""

    def __init__(self, n_visible: int, n_hidden: int, rng=None,
                 init_scale: float = 0.01):
        self.nv = n_visible
        self.nh = n_hidden
        self.rng = rng or np.random.default_rng(0)
        self.W = (init_scale * self.rng.standard_normal((n_visible, n_hidden))
                  ).astype(np.float32)
        self.bv = np.zeros(n_visible, dtype=np.float32)
        self.bh = np.zeros(n_hidden, dtype=np.float32)

    def ph_given_v(self, v: np.ndarray) -> np.ndarray:
        return sigmoid(v @ self.W + self.bh)

    def pv_given_h(self, h: np.ndarray) -> np.ndarray:
        return sigmoid(h @ self.W.T + self.bv)

    def sample(self, p: np.ndarray) -> np.ndarray:
        return (self.rng.random(p.shape) < p).astype(np.float32)

    def cd1(self, v0: np.ndarray, lr: float, momentum: float,
            vW: np.ndarray, vbv: np.ndarray, vbh: np.ndarray):
        """One CD-1 step on minibatch v0: (B, nv). Returns updated momentum."""
        # positive phase
        ph0 = self.ph_given_v(v0)
        h0 = self.sample(ph0)

        # negative phase: one Gibbs step
        pv1 = self.pv_given_h(h0)
        v1 = self.sample(pv1)
        ph1 = self.ph_given_v(v1)

        # gradients (use probabilities on the hidden side for lower variance)
        B = v0.shape[0]
        dW = (v0.T @ ph0 - v1.T @ ph1) / B
        dbv = (v0 - v1).mean(axis=0)
        dbh = (ph0 - ph1).mean(axis=0)

        vW = momentum * vW + lr * dW
        vbv = momentum * vbv + lr * dbv
        vbh = momentum * vbh + lr * dbh
        self.W += vW
        self.bv += vbv
        self.bh += vbh
        return vW, vbv, vbh

    def conditional_fill(self, v_init: np.ndarray, clamp_mask: np.ndarray,
                         n_gibbs: int = 80) -> np.ndarray:
        """Clamp a subset of visible units; let the rest settle by Gibbs.

        Returns the mean visible state averaged over the second half of
        sampling (uses probabilities on the unclamped side for a
        lower-variance Rao-Blackwellised mean).
        """
        v = v_init.copy()
        accum = np.zeros_like(v)
        n_accum = 0
        for t in range(n_gibbs):
            ph = self.ph_given_v(v)
            h = self.sample(ph)
            pv = self.pv_given_h(h)
            v = self.sample(pv)
            v = v * (1 - clamp_mask) + v_init * clamp_mask
            if t >= n_gibbs // 2:
                v_mean = pv * (1 - clamp_mask) + v_init * clamp_mask
                accum += v_mean
                n_accum += 1
        return accum / n_accum


def build_model(n_visible: int = 19, n_hidden: int = 24,
                seed: int = 0, init_scale: float = 0.01) -> ShifterRBM:
    """Boltzmann machine with V1 (8) + V2 (8) + V3 (3) = 19 visible units."""
    rng = np.random.default_rng(seed)
    return ShifterRBM(n_visible=n_visible, n_hidden=n_hidden,
                      rng=rng, init_scale=init_scale)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(N: int = 8,
          n_hidden: int = 24,
          n_epochs: int = 200,
          lr: float = 0.05,
          momentum: float = 0.7,
          batch_size: int = 16,
          seed: int = 0,
          eval_every: int = 25,
          eval_gibbs: int = 80,
          history: dict | None = None,
          verbose: bool = True) -> ShifterRBM:
    """Train the shifter RBM with CD-1 on the full 768-case enumeration."""
    rng = np.random.default_rng(seed)
    V1, V2, V3 = make_shifter_data(N)
    X = np.concatenate([V1, V2, V3], axis=1)  # (M, 2N+3)
    M, nv = X.shape
    nv_inputs = 2 * N

    rbm = ShifterRBM(n_visible=nv, n_hidden=n_hidden, rng=rng)
    vW = np.zeros_like(rbm.W)
    vbv = np.zeros_like(rbm.bv)
    vbh = np.zeros_like(rbm.bh)

    clamp_eval_mask = np.concatenate(
        [np.ones(nv_inputs), np.zeros(3)]
    ).astype(np.float32)

    if verbose:
        print(f"# N={N}: {M} training cases, visible={nv}, hidden={n_hidden}")

    for epoch in range(n_epochs):
        t0 = time.time()
        idx = rng.permutation(M)
        recon_total = 0.0
        for i in range(0, M, batch_size):
            batch = X[idx[i:i + batch_size]]
            vW, vbv, vbh = rbm.cd1(batch, lr, momentum, vW, vbv, vbh)
            ph = rbm.ph_given_v(batch)
            pv = rbm.pv_given_h(ph)
            recon_total += float(np.mean((batch - pv) ** 2)) * batch.shape[0]
        recon_mse = recon_total / M

        if history is not None:
            history.setdefault("epoch", []).append(epoch + 1)
            history.setdefault("recon_mse", []).append(recon_mse)
            if (epoch + 1) % eval_every == 0 or epoch == n_epochs - 1:
                acc = shift_recognition_accuracy(rbm, V1, V2, V3, N=N,
                                                 n_gibbs=eval_gibbs)
                history.setdefault("eval_epoch", []).append(epoch + 1)
                history.setdefault("acc", []).append(acc)

        if verbose and ((epoch + 1) % eval_every == 0
                        or epoch == n_epochs - 1):
            acc = shift_recognition_accuracy(rbm, V1, V2, V3, N=N,
                                             n_gibbs=eval_gibbs)
            print(f"epoch {epoch+1:4d}  recon_mse={recon_mse:.4f}  "
                  f"acc={acc*100:5.1f}%  ({time.time()-t0:.2f}s/epoch)")

    return rbm


# ----------------------------------------------------------------------
# Inference
# ----------------------------------------------------------------------

def shift_recognition_accuracy(rbm: ShifterRBM,
                                V1: np.ndarray,
                                V2: np.ndarray,
                                V3: np.ndarray,
                                N: int = 8,
                                n_gibbs: int = 200) -> float:
    """Clamp V1+V2 on the visible layer; sample V3; argmax accuracy."""
    M = V1.shape[0]
    clamp_mask = np.concatenate([np.ones(2 * N), np.zeros(3)]).astype(np.float32)
    correct = 0
    for i in range(M):
        v_init = np.concatenate([V1[i], V2[i],
                                 np.zeros(3, dtype=np.float32)])
        v_mean = rbm.conditional_fill(v_init, clamp_mask, n_gibbs=n_gibbs)
        y_pred = v_mean[2 * N:]
        if int(np.argmax(y_pred)) == int(np.argmax(V3[i])):
            correct += 1
    return correct / M


def per_class_accuracy(rbm: ShifterRBM, N: int = 8,
                        n_gibbs: int = 200) -> dict[str, float]:
    """Per-shift-class accuracy on the full 768-case enumeration."""
    V1, V2, V3 = make_shifter_data(N)
    out = {}
    for k, name in enumerate(["left (-1)", "none (0)", "right (+1)"]):
        mask = V3[:, k] == 1
        out[name] = shift_recognition_accuracy(
            rbm, V1[mask], V2[mask], V3[mask], N=N, n_gibbs=n_gibbs
        )
    return out


def accuracy_vs_v1_activity(rbm: ShifterRBM, N: int = 8,
                             n_gibbs: int = 150) -> dict[int, tuple[int, int]]:
    """Bucket the 768 cases by number of on-bits in V1; return per-bucket
    (correct, total). The original paper reports 50-89% varying with this."""
    V1, V2, V3 = make_shifter_data(N)
    clamp_mask = np.concatenate([np.ones(2 * N), np.zeros(3)]).astype(np.float32)
    buckets: dict[int, list[int]] = {}
    for i in range(V1.shape[0]):
        n_on = int(V1[i].sum())
        v_init = np.concatenate([V1[i], V2[i],
                                 np.zeros(3, dtype=np.float32)])
        v_mean = rbm.conditional_fill(v_init, clamp_mask, n_gibbs=n_gibbs)
        ok = int(np.argmax(v_mean[2 * N:])) == int(np.argmax(V3[i]))
        buckets.setdefault(n_on, [0, 0])
        buckets[n_on][0] += int(ok)
        buckets[n_on][1] += 1
    return {k: (c, t) for k, (c, t) in sorted(buckets.items())}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--hidden", type=int, default=24)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.7)
    p.add_argument("--batch", type=int, default=16)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-gibbs", type=int, default=200)
    args = p.parse_args()

    t0 = time.time()
    rbm = train(N=args.N, n_hidden=args.hidden, n_epochs=args.epochs,
                lr=args.lr, momentum=args.momentum,
                batch_size=args.batch, seed=args.seed)
    train_secs = time.time() - t0

    V1, V2, V3 = make_shifter_data(args.N)
    acc = shift_recognition_accuracy(rbm, V1, V2, V3, N=args.N,
                                     n_gibbs=args.eval_gibbs)
    print(f"\nFinal accuracy (N={args.N}, {V1.shape[0]} cases, "
          f"{args.eval_gibbs} Gibbs sweeps): {acc*100:.2f}%")
    print(f"Training wallclock: {train_secs:.1f}s")

    print("\nPer-class accuracy:")
    for cname, cacc in per_class_accuracy(rbm, N=args.N,
                                          n_gibbs=args.eval_gibbs).items():
        print(f"  {cname:12s}  {cacc*100:5.1f}%")

    print("\nAccuracy vs V1 activity (bits on):")
    print(f"{'k':>3} {'acc':>7}  ({'correct':>3}/{'total':>3})")
    accs_by_k = []
    for k, (c, t) in accuracy_vs_v1_activity(rbm, N=args.N,
                                              n_gibbs=args.eval_gibbs).items():
        a = c / t if t else 0.0
        accs_by_k.append(a)
        print(f"{k:>3} {a*100:6.1f}%  ({c:>3}/{t:>3})")
    if accs_by_k:
        print(f"\nrange across V1-activity buckets: "
              f"{min(accs_by_k)*100:.1f}% - {max(accs_by_k)*100:.1f}%  "
              f"(paper: 50-89%)")
