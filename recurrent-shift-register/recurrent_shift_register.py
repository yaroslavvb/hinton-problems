"""
Recurrent shift register (Rumelhart, Hinton & Williams 1986, PDP Vol. 1, Ch. 8).

A small recurrent network with N hidden units learns to act as an N-stage
shift register: random binary bits arrive one at a time on a single input
line, and the network's job is to emit, on N - 1 separate output lines, the
bit that arrived 1, 2, ..., N - 1 timesteps earlier. The interesting
result is the *form* of the converged solution: the recurrent weight
matrix becomes (approximately) a permutation/shift matrix that passes
activations from one hidden unit to the next, while W_xh writes into the
first stage and the output projection reads each delayed bit from a
different stage. The mechanism is a literal hardware shift register
implemented in real-valued sigmoidal units.

Why multi-output? With a single output and delay = 1, the spec's "target
= input shifted by 1 timestep" is solvable by a single hidden unit and
the network has no incentive to use the other N - 1 units. To force all
N units to participate as a chain (so the structural shift-matrix
prediction can be checked), we predict the input at *every* delay from 1
through N - 1 simultaneously. Each output reads from a different hidden
unit, so the only efficient solution is a chain of N stages with W_hh ~
shift matrix.

This file: numpy-only RNN + Backpropagation Through Time (BPTT). Inputs
in {-1, +1}; tanh hidden + tanh output; MSE loss; weight decay + momentum.
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_sequence(n_units: int, sequence_len: int,
                  rng: np.random.Generator
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Random binary sequence in {-1, +1}.

    The target at time t is a vector of length n_units - 1, whose d-th
    entry (for d = 1, 2, ..., n_units - 1) is input[t - d] -- i.e. the
    input from 1, 2, ..., N - 1 timesteps earlier. The mask at time t
    for delay d is 1 iff t >= d.

    With N hidden units and N - 1 delay outputs, the network has just
    enough capacity to hold each delayed bit in a dedicated unit and
    pass them along a chain. With weight decay + L1 on W_hh, the
    converged W_hh becomes recognisably permutation-shaped.

    Returns (input, target, mask) with shapes:
      input  : (sequence_len,)
      target : (sequence_len, n_units - 1)
      mask   : (sequence_len, n_units - 1)
    """
    n_out = n_units - 1
    x = rng.choice([-1.0, 1.0], size=sequence_len)
    target = np.zeros((sequence_len, n_out))
    mask = np.zeros((sequence_len, n_out))
    for d in range(1, n_units):
        idx = d - 1
        if d < sequence_len:
            target[d:, idx] = x[:sequence_len - d]
            mask[d:, idx] = 1.0
    return x, target, mask


def make_batch(n_units: int, batch_size: int, sequence_len: int,
               rng: np.random.Generator
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Stack `batch_size` independent sequences. Shapes:
      x:      (B, T)
      target: (B, T, N)
      mask:   (B, T, N)
    """
    xs, ts, ms = [], [], []
    for _ in range(batch_size):
        x, t, m = make_sequence(n_units, sequence_len, rng)
        xs.append(x); ts.append(t); ms.append(m)
    return np.stack(xs), np.stack(ts), np.stack(ms)


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class ShiftRegisterRNN:
    """Single-input, N-output RNN with N tanh hidden units.

    Forward (per timestep t):
        z_h[t] = W_xh * x[t] + W_hh @ h[t-1] + b_h          # (N,)
        h[t]   = tanh(z_h[t])
        z_y[t] = W_hy @ h[t] + b_y                           # (N,)
        y[t]   = tanh(z_y[t])
    """

    def __init__(self, n_units: int, seed: int = 0, init_scale: float = 0.2):
        if n_units < 2:
            raise ValueError("need at least 2 units (1 stage of delay)")
        self.n_units = n_units
        self.n_out = n_units - 1
        self.rng = np.random.default_rng(seed)
        s = init_scale
        self.W_xh = s * (self.rng.random((n_units,)) - 0.5) * 2.0
        self.W_hh = s * (self.rng.random((n_units, n_units)) - 0.5) * 2.0
        self.b_h = s * (self.rng.random((n_units,)) - 0.5) * 2.0
        self.W_hy = s * (self.rng.random((self.n_out, n_units)) - 0.5) * 2.0
        self.b_y = s * (self.rng.random((self.n_out,)) - 0.5) * 2.0

    # ---- forward -----------------------------------------------------------

    def forward(self, x_batch: np.ndarray) -> dict:
        """x_batch: (B, T). Returns h: (B, T+1, N), y: (B, T, N - 1)."""
        B, T = x_batch.shape
        N = self.n_units
        h = np.zeros((B, T + 1, N))
        y = np.zeros((B, T, self.n_out))
        for t in range(T):
            z_h = (np.outer(x_batch[:, t], self.W_xh)
                   + h[:, t, :] @ self.W_hh.T
                   + self.b_h)
            h[:, t + 1, :] = np.tanh(z_h)
            z_y = h[:, t + 1, :] @ self.W_hy.T + self.b_y
            y[:, t, :] = np.tanh(z_y)
        return {"h": h, "y": y}

    # ---- loss / accuracy ---------------------------------------------------

    @staticmethod
    def loss(y: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
        m = mask.sum()
        if m < 1e-9:
            return 0.0
        return float(0.5 * np.sum(((y - target) ** 2) * mask) / m)

    @staticmethod
    def accuracy(y: np.ndarray, target: np.ndarray, mask: np.ndarray) -> float:
        """Sign-match accuracy, averaged over masked-in (timestep, delay) pairs."""
        m = mask.sum()
        if m < 1e-9:
            return 0.0
        pred = np.sign(y)
        pred[pred == 0] = 1.0
        correct = ((pred == target) * mask).sum()
        return float(correct / m)

    @staticmethod
    def per_delay_accuracy(y: np.ndarray, target: np.ndarray,
                            mask: np.ndarray) -> np.ndarray:
        """Sign-match accuracy broken down by delay channel.

        Returns (n_delays,) array.
        """
        pred = np.sign(y)
        pred[pred == 0] = 1.0
        per = []
        for d in range(y.shape[-1]):
            m = mask[..., d].sum()
            if m < 1e-9:
                per.append(0.0)
            else:
                per.append(float(((pred[..., d] == target[..., d])
                                  * mask[..., d]).sum() / m))
        return np.array(per)

    # ---- BPTT --------------------------------------------------------------

    def backward(self, x_batch: np.ndarray, target: np.ndarray,
                 mask: np.ndarray, fwd: dict) -> dict:
        """Backprop-through-time on the masked MSE loss."""
        B, T = x_batch.shape
        N = self.n_units
        h = fwd["h"]                        # (B, T+1, N)
        y = fwd["y"]                        # (B, T, n_out)

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        m_total = max(float(mask.sum()), 1.0)

        # carry-over gradient from t+1's hidden state
        dh_next = np.zeros((B, N))

        for t in reversed(range(T)):
            # ---- output head ----
            # dL/dy[t, :] = (y - target) / m_total * mask
            dy = (y[:, t, :] - target[:, t, :]) * mask[:, t, :] / m_total  # (B, n_out)
            # through tanh of output
            dz_y = dy * (1.0 - y[:, t, :] ** 2)                            # (B, n_out)
            dW_hy += dz_y.T @ h[:, t + 1, :]                               # (n_out, N)
            db_y += dz_y.sum(axis=0)                                       # (n_out,)
            # gradient flowing back into h[t+1] from output
            dh_from_y = dz_y @ self.W_hy                                   # (B, N)

            # combine with future-timestep contribution
            dh = dh_from_y + dh_next                                       # (B, N)

            # through tanh of hidden
            dz_h = dh * (1.0 - h[:, t + 1, :] ** 2)                        # (B, N)

            # parameter grads
            dW_xh += dz_h.T @ x_batch[:, t]                                # (N,)
            dW_hh += dz_h.T @ h[:, t, :]                                   # (N, N)
            db_h += dz_h.sum(axis=0)

            # propagate to previous timestep's hidden state
            dh_next = dz_h @ self.W_hh                                     # (B, N)

        return {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                "W_hy": dW_hy, "b_y": db_y}

    # ---- utility -----------------------------------------------------------

    def n_params(self) -> int:
        return (self.W_xh.size + self.W_hh.size + self.b_h.size
                + self.W_hy.size + self.b_y.size)

    def snapshot(self) -> dict:
        return {"W_xh": self.W_xh.copy(),
                "W_hh": self.W_hh.copy(),
                "b_h": self.b_h.copy(),
                "W_hy": self.W_hy.copy(),
                "b_y": self.b_y.copy()}


# ----------------------------------------------------------------------
# Permutation / shift-matrix metrics
# ----------------------------------------------------------------------

def shift_matrix_score(W_hh: np.ndarray) -> dict:
    """Quantify how close W_hh is to a shift matrix (the structure of a
    hardware shift register).

    A non-cyclic shift register has rank-(N - 1) connectivity: one
    "input" stage with all-zero incoming recurrent weights, plus N - 1
    chain links each carrying activation from one stage to the next.
    Equivalently, only N - 1 of the N rows have a strong recurrent
    connection. The hidden units are interchangeable, so the strong
    entries can land in any (row, col) positions provided the 1's trace
    out a chain of N - 1 directed edges that visits every unit exactly
    once.

    We compute:

      strong_entries     : positions of the N - 1 largest |W_hh| entries
                             (one per row, picked greedily); ties broken
                             arbitrarily.
      shift_diag_mean    : mean |W_hh[i, j]| over those N - 1 strong
                             entries (ideal target: > 1.0 so tanh
                             saturates and bits are preserved).
      shift_leak_max     : largest |W_hh[i, j]| outside the chosen
                             strong-entry set.
      sparsity_ratio     : shift_leak_max / shift_diag_mean (ideal: ~0).
      is_shift_matrix    : sparsity_ratio < 0.2 AND shift_diag_mean > 0.4.
    """
    N = W_hh.shape[0]

    # ---- pick the N - 1 strongest (row, col) entries, one per row ----
    # We select N - 1 rows to be "active" chain links; one row will be
    # left "silent" (the input stage). Try every choice of silent row
    # and assign each remaining row to its biggest column, then pick
    # whichever choice gives the largest summed magnitude.
    W_abs = np.abs(W_hh)
    best_total = -np.inf
    best_silent = 0
    best_assign = []
    for silent in range(N):
        active_rows = [r for r in range(N) if r != silent]
        # for each active row, pick its argmax column
        # we DON'T enforce uniqueness across columns because the chain
        # can fork; uniqueness is checked separately below.
        per_row = [(r, int(np.argmax(W_abs[r])), float(W_abs[r].max()))
                   for r in active_rows]
        total = sum(p[2] for p in per_row)
        if total > best_total:
            best_total = total
            best_silent = silent
            best_assign = per_row

    chain_positions = [(r, c) for (r, c, _) in best_assign]
    chain_mag = [W_hh[r, c] for (r, c) in chain_positions]
    shift_diag_mean = float(np.mean(np.abs(chain_mag))) if chain_mag else 0.0

    leak_mask = np.ones_like(W_hh, dtype=bool)
    leak_mask[best_silent, :] = True   # silent row counts as leak everywhere
    for r, c in chain_positions:
        leak_mask[r, c] = False
    shift_leak_max = float(np.max(np.abs(W_hh[leak_mask]))) if leak_mask.any() else 0.0
    sparsity_ratio = (shift_leak_max / shift_diag_mean
                       if shift_diag_mean > 1e-9 else 1.0)

    # ---- chain traversal: starting from the "input" row, can we walk
    # the chain and visit every hidden unit exactly once? ----
    # Build a forward map: row -> column it reads from (i.e. unit row's
    # value at next timestep equals the chosen column's value at this
    # timestep). The "input" row is best_silent.
    chain_map = {r: c for (r, c) in chain_positions}
    visited = [best_silent]
    cursor = best_silent
    chain_visits_all = True
    while len(visited) < N:
        # who reads from `cursor`? find the row whose chain_map[row] == cursor
        next_row = None
        for r, c in chain_positions:
            if c == cursor and r not in visited:
                next_row = r
                break
        if next_row is None:
            chain_visits_all = False
            break
        visited.append(next_row)
        cursor = next_row

    # ---- closest sign-permutation (full-rank version, for reporting) ----
    perm = _hungarian_max(W_abs)
    P = np.zeros_like(W_hh)
    for i, j in enumerate(perm):
        P[i, j] = 1.0 if W_hh[i, j] >= 0 else -1.0
    perm_dist = float(np.linalg.norm(W_hh - P, ord="fro"))

    return {
        "shift_diag_mean": shift_diag_mean,        # mean |w| of chain links
        "shift_leak_max": shift_leak_max,           # max |w| off-chain
        "sparsity_ratio": sparsity_ratio,           # leak / chain
        "input_stage": int(best_silent),            # row with no recurrent input
        "chain_positions": chain_positions,         # list of (row, col)
        "chain_visits_all": bool(chain_visits_all),
        "best_perm_distance": perm_dist,
        "best_perm_indices": perm.tolist(),
        "weight_matrix_norm": float(np.linalg.norm(W_hh)),
        "is_shift_matrix": bool(sparsity_ratio < 0.2
                                 and shift_diag_mean > 0.4
                                 and chain_visits_all),
    }


def _hungarian_max(C: np.ndarray) -> np.ndarray:
    """Maximum-weight perfect matching on a square matrix. Tiny problem
    sizes (N <= 8 here), so brute force / O(N!) is fine and avoids a
    scipy dependency."""
    from itertools import permutations
    N = C.shape[0]
    best = (-np.inf, None)
    for p in permutations(range(N)):
        s = float(sum(C[i, p[i]] for i in range(N)))
        if s > best[0]:
            best = (s, p)
    return np.array(best[1], dtype=np.int64)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_units: int = 3,
          n_sweeps: int = 200,
          batch_size: int = 16,
          sequence_len: int = 30,
          lr: float = 0.3,
          momentum: float = 0.9,
          weight_decay: float = 1e-3,
          l1_W_hh: float = 0.05,
          init_scale: float = 0.2,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 5,
          verbose: bool = True
          ) -> tuple[ShiftRegisterRNN, dict]:
    """Train the N-unit shift-register RNN with vanilla BPTT + momentum.

    Two regularisers push the converged solution toward a clean shift
    register:

      weight_decay (L2 on W_xh, W_hh, W_hy): shrinks the overall scale
      l1_W_hh (L1 on W_hh): encourages sparsity in the recurrent matrix
        so that off-cycle entries die out and the matrix becomes
        recognisably permutation-shaped.
    """
    model = ShiftRegisterRNN(n_units, seed=seed, init_scale=init_scale)
    rng = np.random.default_rng(seed + 10_000)

    velocities = {k: np.zeros_like(v) for k, v in
                  [("W_xh", model.W_xh), ("W_hh", model.W_hh),
                   ("b_h", model.b_h), ("W_hy", model.W_hy),
                   ("b_y", model.b_y)]}

    history = {"epoch": [], "loss": [], "accuracy": [],
               "per_delay_accuracy": [],
               "shift_diag_mean": [], "shift_leak_max": [],
               "sparsity_ratio": [],
               "best_perm_distance": [], "perm_indices": [],
               "weight_norm": [],
               "converged_epoch": None,
               "snapshots": []}

    if verbose:
        print(f"# {n_units}-unit shift-register RNN  params={model.n_params()}  "
              f"out_delays={list(range(1, n_units))}  batch={batch_size}  "
              f"seq_len={sequence_len}  lr={lr}  momentum={momentum}  "
              f"wd={weight_decay}  l1_W_hh={l1_W_hh}  seed={seed}")

    for epoch in range(n_sweeps):
        x_batch, target, mask = make_batch(n_units, batch_size,
                                            sequence_len, rng)

        fwd = model.forward(x_batch)
        grads = model.backward(x_batch, target, mask, fwd)
        loss_val = ShiftRegisterRNN.loss(fwd["y"], target, mask)
        acc = ShiftRegisterRNN.accuracy(fwd["y"], target, mask)
        per_delay = ShiftRegisterRNN.per_delay_accuracy(fwd["y"], target, mask)

        # SGD with momentum + weight decay (decoupled, applied after grads)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W_xh += velocities["W_xh"]
        model.W_hh += velocities["W_hh"]
        model.b_h += velocities["b_h"]
        model.W_hy += velocities["W_hy"]
        model.b_y += velocities["b_y"]
        if weight_decay > 0:
            shrink = (1.0 - lr * weight_decay)
            model.W_xh *= shrink
            model.W_hh *= shrink
            model.W_hy *= shrink
            # don't shrink biases
        if l1_W_hh > 0:
            # proximal step on |W_hh|: soft-threshold by lr * l1
            thr = lr * l1_W_hh
            model.W_hh = np.sign(model.W_hh) * np.maximum(
                np.abs(model.W_hh) - thr, 0.0)

        sm = shift_matrix_score(model.W_hh)
        history["epoch"].append(epoch + 1)
        history["loss"].append(loss_val)
        history["accuracy"].append(acc)
        history["per_delay_accuracy"].append(per_delay.tolist())
        history["shift_diag_mean"].append(sm["shift_diag_mean"])
        history["shift_leak_max"].append(sm["shift_leak_max"])
        history["sparsity_ratio"].append(sm["sparsity_ratio"])
        history["best_perm_distance"].append(sm["best_perm_distance"])
        history["perm_indices"].append(sm["best_perm_indices"])
        history["weight_norm"].append(sm["weight_matrix_norm"])

        if (history["converged_epoch"] is None
                and acc >= 0.99 and sm["is_shift_matrix"]):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at sweep {epoch + 1}  "
                      f"loss={loss_val:.4f}  acc={acc*100:.1f}%  "
                      f"chain_mag={sm['shift_diag_mean']:.2f}  "
                      f"leak={sm['shift_leak_max']:.2f}")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)
            history["snapshots"].append((epoch + 1, model.snapshot()))

        if verbose and (epoch % 20 == 0 or epoch == n_sweeps - 1):
            print(f"  sweep {epoch+1:4d}  loss={loss_val:.4f}  "
                  f"acc={acc*100:5.1f}%  "
                  f"chain_mag={sm['shift_diag_mean']:.2f}  "
                  f"leak={sm['shift_leak_max']:.2f}  "
                  f"sparsity={sm['sparsity_ratio']:.2f}  "
                  f"|W_hh|={sm['weight_matrix_norm']:.2f}")

    return model, history


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep_seeds(n_units: int, n_seeds: int, n_sweeps: int = 200, **kw) -> dict:
    out = {"seeds": [], "converged_epoch": [], "final_acc": [],
           "shift_diag_mean": [], "shift_leak_max": [],
           "sparsity_ratio": [], "is_shift_matrix": []}
    for s in range(n_seeds):
        model, hist = train(n_units=n_units, n_sweeps=n_sweeps,
                            seed=s, verbose=False, **kw)
        sm = shift_matrix_score(model.W_hh)
        out["seeds"].append(s)
        out["converged_epoch"].append(hist["converged_epoch"])
        out["final_acc"].append(hist["accuracy"][-1])
        out["shift_diag_mean"].append(sm["shift_diag_mean"])
        out["shift_leak_max"].append(sm["shift_leak_max"])
        out["sparsity_ratio"].append(sm["sparsity_ratio"])
        out["is_shift_matrix"].append(sm["is_shift_matrix"])
        print(f"  N={n_units}  seed {s:2d}  conv@{str(hist['converged_epoch']):>5}  "
              f"acc={hist['accuracy'][-1]*100:5.1f}%  "
              f"chain={sm['shift_diag_mean']:.2f}  "
              f"leak={sm['shift_leak_max']:.2f}  "
              f"sparse={sm['sparsity_ratio']:.2f}  "
              f"is_shift={sm['is_shift_matrix']}")
    return out


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n-units", type=int, default=3, choices=[3, 5, 4, 6, 7, 8])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--sequence-len", type=int, default=30)
    p.add_argument("--n-sweeps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--l1-W-hh", type=float, default=0.05,
                   help="L1 penalty on W_hh to encourage shift-matrix sparsity")
    p.add_argument("--init-scale", type=float, default=0.2)
    p.add_argument("--multi-seed", type=int, default=0,
                   help="if > 0, run a sweep over this many seeds and exit")
    args = p.parse_args()

    _print_environment()

    if args.multi_seed > 0:
        out = sweep_seeds(n_units=args.n_units, n_seeds=args.multi_seed,
                          n_sweeps=args.n_sweeps,
                          batch_size=args.batch_size,
                          sequence_len=args.sequence_len,
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          l1_W_hh=args.l1_W_hh,
                          init_scale=args.init_scale)
        n = len(out["seeds"])
        n_conv = sum(1 for c in out["converged_epoch"] if c is not None)
        n_shift = sum(out["is_shift_matrix"])
        conv_eps = [c for c in out["converged_epoch"] if c is not None]
        med = int(np.median(conv_eps)) if conv_eps else None
        print(f"\n{n_conv}/{n} converged in <{args.n_sweeps} sweeps; "
              f"{n_shift}/{n} learned a shift-matrix W_hh. "
              f"median converged sweep = {med}")
        return

    t0 = time.time()
    model, hist = train(n_units=args.n_units, n_sweeps=args.n_sweeps,
                        batch_size=args.batch_size,
                        sequence_len=args.sequence_len, lr=args.lr,
                        momentum=args.momentum,
                        weight_decay=args.weight_decay,
                        l1_W_hh=args.l1_W_hh,
                        init_scale=args.init_scale, seed=args.seed)
    wallclock = time.time() - t0

    sm = shift_matrix_score(model.W_hh)

    print("\n=== final ===")
    print(f"final accuracy : {hist['accuracy'][-1]*100:.1f}%")
    print(f"per-delay acc  : "
          + " ".join(f"d{d+1}={a*100:.1f}%"
                     for d, a in enumerate(hist['per_delay_accuracy'][-1])))
    print(f"final loss     : {hist['loss'][-1]:.5f}")
    print(f"converged sweep: {hist['converged_epoch']}")
    print(f"wallclock      : {wallclock:.3f}s")

    print("\n=== W_hh ===")
    with np.printoptions(precision=3, suppress=True):
        print(model.W_hh)
    print(f"\n  input stage (silent row)    : unit {sm['input_stage']}")
    print(f"  chain links (row -> col)    : "
          + "  ".join(f"h[{r}]<-h[{c}] (w={model.W_hh[r, c]:+.2f})"
                       for (r, c) in sm["chain_positions"]))
    print(f"  chain visits all hidden?    : {sm['chain_visits_all']}")
    print(f"  chain mean |w|              : {sm['shift_diag_mean']:.3f}")
    print(f"  off-chain max |w|           : {sm['shift_leak_max']:.3f}")
    print(f"  sparsity ratio (leak/chain) : {sm['sparsity_ratio']:.3f}")
    print(f"  is approximately a shift matrix? : {sm['is_shift_matrix']}")

    print("\n=== W_xh (input -> hidden) ===")
    with np.printoptions(precision=3, suppress=True):
        print(" ", model.W_xh)
    print("=== W_hy (delay output[1..N-1] <- hidden) ===")
    with np.printoptions(precision=3, suppress=True):
        print(model.W_hy)


if __name__ == "__main__":
    main()
