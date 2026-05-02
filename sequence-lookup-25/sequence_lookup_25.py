"""
25-sequence look-up RNN task (Rumelhart, Hinton & Williams 1986, PDP Vol. 1, Ch. 8).

A small recurrent network with 30 tanh hidden units learns to associate 25
five-letter sequences with their 3-bit target codes. Inputs are one-hot
encoded letters from a 5-letter alphabet, presented one letter per
timestep. After processing all 5 letters, the network emits a 3-bit code
on its three tanh output units (read at the final timestep only).

The interesting property: training on only 20 of the 25 sequences and
holding out 5 for test, the converged network correctly recalls 4 to 5
of the held-out sequences. This generalization is non-trivial -- with
arbitrary look-up labels there would be no basis for it. Here, the
labels come from a fixed *teacher function* that combines the letter
identities with their positions; the network discovers that compositional
structure during training and can therefore extrapolate.

Variable-timing variant: each letter is presented for a random number
of timesteps (1-3), and the net (now with 60 hidden units for added
capacity) must learn that the output depends on letter content and
order, not on timing. This is a small time-warp invariance challenge.

This file: numpy-only RNN + Backpropagation Through Time. Tanh hidden +
tanh output, MSE loss read at the final timestep, full-batch SGD with
momentum and weight decay.
"""

from __future__ import annotations
import argparse
import json
import os
import platform
import sys
import time
import numpy as np


# ----------------------------------------------------------------------
# Task constants
# ----------------------------------------------------------------------

ALPHABET_SIZE = 5  # input one-hot dimension (5 letters)
SEQ_LEN = 5        # letters per sequence
N_OUT = 3          # output bits (3-digit code)
N_TOTAL = 25       # total sequences (20 train + 5 test)
N_TRAIN = 20
N_TEST = 5
TEACHER_SEED = 1234


# ----------------------------------------------------------------------
# Teacher function (defines the targets)
# ----------------------------------------------------------------------

def make_teacher(seed: int = TEACHER_SEED, alphabet_size: int = ALPHABET_SIZE,
                 seq_len: int = SEQ_LEN, n_out: int = N_OUT):
    """A fixed, position-dependent linear teacher:

        logit_i(seq) = sum_t W_teacher[i, l_t] * pos_weight[t] + b_teacher[i]
        target_i(seq) = sign(logit_i(seq))

    With |W_teacher| ~ N(0, 1) and |pos_weight| draws from a smooth
    monotone curve plus a small random perturbation, the targets are
    a learnable function of *both* letter identity and position. A
    bag-of-letters classifier cannot solve it; the student RNN must
    encode position as well as content in its hidden state. This gives
    the held-out test set a basis for generalization.
    """
    rng = np.random.default_rng(seed)
    W_teacher = rng.standard_normal((n_out, alphabet_size))
    # smooth monotone decay 1.0 -> 0.2 across positions, plus small noise
    decay = np.linspace(1.0, 0.2, seq_len)
    pos_weight = decay + 0.15 * rng.standard_normal(seq_len)
    b_teacher = 0.2 * rng.standard_normal(n_out)
    return {"W": W_teacher, "pos_weight": pos_weight, "b": b_teacher}


def teacher_eval(letters: np.ndarray, teacher: dict) -> np.ndarray:
    """letters: (B, T) integer ids. Returns (B, n_out) targets in {-1, +1}."""
    B, T = letters.shape
    W = teacher["W"]              # (n_out, alphabet_size)
    pos = teacher["pos_weight"]    # (T,)
    b = teacher["b"]               # (n_out,)
    logits = np.zeros((B, W.shape[0]))
    for t in range(T):
        # W[:, letters[:, t]] is (n_out, B); transpose to (B, n_out)
        logits += pos[t] * W[:, letters[:, t]].T
    logits += b[None, :]
    targets = np.sign(logits)
    targets[targets == 0] = 1.0
    return targets


# ----------------------------------------------------------------------
# Dataset
# ----------------------------------------------------------------------

def _build_variable_input(letters_row: np.ndarray, timings_row: np.ndarray,
                          alphabet_size: int) -> np.ndarray:
    T_i = int(timings_row.sum())
    seq_input = np.zeros((T_i, alphabet_size))
    t = 0
    for k in range(len(letters_row)):
        for _ in range(int(timings_row[k])):
            seq_input[t, letters_row[k]] = 1.0
            t += 1
    return seq_input


def resample_timings(letters: np.ndarray, rng: np.random.Generator,
                     max_timing: int = 3,
                     alphabet_size: int = ALPHABET_SIZE) -> tuple:
    """Resample variable-timing inputs for the given letter sequences."""
    n_total, seq_len = letters.shape
    timings = []
    inputs = []
    for i in range(n_total):
        tmg = rng.integers(1, max_timing + 1, size=seq_len)
        timings.append(tmg)
        inputs.append(_build_variable_input(letters[i], tmg, alphabet_size))
    return inputs, timings


def generate_dataset(seed: int = 0, n_total: int = N_TOTAL,
                     n_train: int = N_TRAIN, variable_timing: bool = False,
                     alphabet_size: int = ALPHABET_SIZE,
                     seq_len: int = SEQ_LEN, max_timing: int = 3,
                     teacher_seed: int = TEACHER_SEED) -> dict:
    """Sample `n_total` distinct 5-letter sequences and compute targets.

    Returns a dict containing:
      letters       : (n_total, seq_len) int ids
      one_hot       : (n_total, seq_len, alphabet_size) fixed-timing input
      targets       : (n_total, n_out) {-1, +1}
      train_idx     : (n_train,) indices
      test_idx      : (n_total - n_train,) indices
      teacher       : the W / pos_weight / b dict
      variable_inputs : list of (T_i, alphabet_size) per-sequence inputs
                          (only present if variable_timing)
      timings         : list of (seq_len,) holds per letter
                          (only present if variable_timing)
    """
    rng = np.random.default_rng(seed)
    teacher = make_teacher(teacher_seed, alphabet_size, seq_len)

    # Sample distinct sequences
    seen = set()
    letters = np.zeros((n_total, seq_len), dtype=np.int64)
    for i in range(n_total):
        while True:
            cand = tuple(rng.integers(0, alphabet_size, size=seq_len).tolist())
            if cand not in seen:
                seen.add(cand)
                letters[i] = cand
                break

    # Fixed-timing one-hot input
    one_hot = np.zeros((n_total, seq_len, alphabet_size))
    for b in range(n_total):
        for t in range(seq_len):
            one_hot[b, t, letters[b, t]] = 1.0

    targets = teacher_eval(letters, teacher)

    # Train/test split
    perm = rng.permutation(n_total)
    train_idx = perm[:n_train]
    test_idx = perm[n_train:]

    out = {
        "letters": letters,
        "one_hot": one_hot,
        "targets": targets,
        "train_idx": train_idx,
        "test_idx": test_idx,
        "teacher": teacher,
        "alphabet_size": alphabet_size,
        "seq_len": seq_len,
    }

    if variable_timing:
        variable_inputs, timings = resample_timings(letters, rng, max_timing,
                                                     alphabet_size)
        out["timings"] = timings
        out["variable_inputs"] = variable_inputs
        out["max_timing"] = max_timing

    return out


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class SequenceLookupRNN:
    """One-layer tanh RNN, output emitted at the final timestep only."""

    def __init__(self, n_in: int = ALPHABET_SIZE, n_hidden: int = 30,
                 n_out: int = N_OUT, seed: int = 0, init_scale: float = 0.3):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.seed = seed
        self.init_scale = init_scale
        self.rng = np.random.default_rng(seed)
        s = init_scale
        self.W_xh = s * self.rng.standard_normal((n_hidden, n_in))
        # spectral-radius-friendly init for W_hh: small uniform
        self.W_hh = (s * 0.7 / np.sqrt(n_hidden)
                     * self.rng.standard_normal((n_hidden, n_hidden)))
        self.b_h = np.zeros(n_hidden)
        self.W_hy = s * self.rng.standard_normal((n_out, n_hidden))
        self.b_y = np.zeros(n_out)

    # ---- forward -----------------------------------------------------------

    def forward(self, x_batch: np.ndarray) -> dict:
        """x_batch: (B, T, n_in). Output read at timestep T (final).

        Returns h: (B, T+1, n_hidden), y: (B, n_out).
        """
        B, T, _ = x_batch.shape
        h = np.zeros((B, T + 1, self.n_hidden))
        for t in range(T):
            z_h = (x_batch[:, t, :] @ self.W_xh.T
                   + h[:, t, :] @ self.W_hh.T
                   + self.b_h[None, :])
            h[:, t + 1, :] = np.tanh(z_h)
        z_y = h[:, T, :] @ self.W_hy.T + self.b_y[None, :]
        y = np.tanh(z_y)
        return {"h": h, "y": y, "T": T}

    # ---- per-sequence variable-length forward (variable timing) -----------

    def forward_variable(self, inputs: list) -> dict:
        """`inputs`: list of B arrays, each (T_i, n_in). Output at last t."""
        ys = []
        hs = []
        for x_i in inputs:
            T_i = x_i.shape[0]
            h_i = np.zeros((T_i + 1, self.n_hidden))
            for t in range(T_i):
                z_h = (x_i[t] @ self.W_xh.T
                       + h_i[t] @ self.W_hh.T
                       + self.b_h)
                h_i[t + 1] = np.tanh(z_h)
            z_y = h_i[T_i] @ self.W_hy.T + self.b_y
            y_i = np.tanh(z_y)
            ys.append(y_i)
            hs.append(h_i)
        return {"ys": ys, "hs": hs}

    # ---- loss / accuracy ---------------------------------------------------

    @staticmethod
    def loss(y: np.ndarray, target: np.ndarray) -> float:
        return float(0.5 * np.mean(np.sum((y - target) ** 2, axis=-1)))

    @staticmethod
    def all_bits_accuracy(y: np.ndarray, target: np.ndarray) -> float:
        pred = np.sign(y)
        pred[pred == 0] = 1.0
        return float(np.mean(np.all(pred == target, axis=-1)))

    @staticmethod
    def per_bit_accuracy(y: np.ndarray, target: np.ndarray) -> np.ndarray:
        pred = np.sign(y)
        pred[pred == 0] = 1.0
        return np.mean(pred == target, axis=0)

    @staticmethod
    def n_correct(y: np.ndarray, target: np.ndarray) -> int:
        pred = np.sign(y)
        pred[pred == 0] = 1.0
        return int(np.sum(np.all(pred == target, axis=-1)))

    # ---- BPTT --------------------------------------------------------------

    def backward(self, x_batch: np.ndarray, target: np.ndarray,
                 fwd: dict) -> dict:
        """Backprop-through-time on MSE loss read at final timestep."""
        B, T, _ = x_batch.shape
        h = fwd["h"]
        y = fwd["y"]

        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)

        # output head
        dy = (y - target) / B                          # (B, n_out)
        dz_y = dy * (1.0 - y ** 2)                      # (B, n_out)
        dW_hy = dz_y.T @ h[:, T, :]                     # (n_out, n_hidden)
        db_y = dz_y.sum(axis=0)
        dh = dz_y @ self.W_hy                           # (B, n_hidden)

        for t in reversed(range(T)):
            dz_h = dh * (1.0 - h[:, t + 1, :] ** 2)     # (B, n_hidden)
            dW_xh += dz_h.T @ x_batch[:, t, :]
            dW_hh += dz_h.T @ h[:, t, :]
            db_h += dz_h.sum(axis=0)
            dh = dz_h @ self.W_hh

        return {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                "W_hy": dW_hy, "b_y": db_y}

    def backward_variable(self, inputs: list, targets: np.ndarray,
                          fwd: dict) -> dict:
        """Per-sequence BPTT for variable-length inputs.

        targets: (B, n_out). Loss is mean over batch of MSE at final t.
        """
        B = len(inputs)
        dW_xh = np.zeros_like(self.W_xh)
        dW_hh = np.zeros_like(self.W_hh)
        db_h = np.zeros_like(self.b_h)
        dW_hy = np.zeros_like(self.W_hy)
        db_y = np.zeros_like(self.b_y)

        for i, x_i in enumerate(inputs):
            T_i = x_i.shape[0]
            h_i = fwd["hs"][i]                         # (T_i+1, n_hidden)
            y_i = fwd["ys"][i]                         # (n_out,)
            tgt = targets[i]

            dy = (y_i - tgt) / B                       # (n_out,)
            dz_y = dy * (1.0 - y_i ** 2)               # (n_out,)
            dW_hy += np.outer(dz_y, h_i[T_i])
            db_y += dz_y
            dh = dz_y @ self.W_hy                      # (n_hidden,)

            for t in reversed(range(T_i)):
                dz_h = dh * (1.0 - h_i[t + 1] ** 2)   # (n_hidden,)
                dW_xh += np.outer(dz_h, x_i[t])
                dW_hh += np.outer(dz_h, h_i[t])
                db_h += dz_h
                dh = dz_h @ self.W_hh

        return {"W_xh": dW_xh, "W_hh": dW_hh, "b_h": db_h,
                "W_hy": dW_hy, "b_y": db_y}

    # ---- utility -----------------------------------------------------------

    def n_params(self) -> int:
        return (self.W_xh.size + self.W_hh.size + self.b_h.size +
                self.W_hy.size + self.b_y.size)

    def snapshot(self) -> dict:
        return {"W_xh": self.W_xh.copy(),
                "W_hh": self.W_hh.copy(),
                "b_h": self.b_h.copy(),
                "W_hy": self.W_hy.copy(),
                "b_y": self.b_y.copy()}


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_hidden: int = 30,
          n_sweeps: int = 1500,
          lr: float = 0.05,
          momentum: float = 0.9,
          weight_decay: float = 1e-4,
          init_scale: float = 0.5,
          grad_clip: float = 5.0,
          seed: int = 0,
          dataset_seed: int = 0,
          variable_timing: bool = False,
          max_timing: int = 3,
          snapshot_callback=None,
          snapshot_every: int = 25,
          verbose: bool = True) -> tuple:
    """Train the look-up RNN. Returns (model, history, dataset)."""
    data = generate_dataset(seed=dataset_seed, variable_timing=variable_timing,
                             max_timing=max_timing)
    targets = data["targets"]
    train_idx = data["train_idx"]
    test_idx = data["test_idx"]

    model = SequenceLookupRNN(n_in=ALPHABET_SIZE, n_hidden=n_hidden,
                              n_out=N_OUT, seed=seed, init_scale=init_scale)

    velocities = {k: np.zeros_like(v) for k, v in
                  [("W_xh", model.W_xh), ("W_hh", model.W_hh),
                   ("b_h", model.b_h), ("W_hy", model.W_hy),
                   ("b_y", model.b_y)]}

    history = {"sweep": [], "loss": [], "train_acc": [], "test_acc": [],
               "train_per_bit": [], "test_per_bit": [],
               "test_n_correct": [], "weight_norm": [],
               "converged_sweep": None, "snapshots": []}

    if verbose:
        print(f"# sequence-lookup-25  hidden={n_hidden}  variable_timing={variable_timing}  "
              f"params={model.n_params()}")
        print(f"# train_idx={list(train_idx)}  test_idx={list(test_idx)}")
        print(f"# lr={lr}  momentum={momentum}  weight_decay={weight_decay}  "
              f"init_scale={init_scale}  n_sweeps={n_sweeps}  seed={seed}")

    timing_rng = np.random.default_rng(seed + 99_991)
    for sweep in range(n_sweeps):
        if variable_timing:
            # Resample TRAIN timings each sweep -> teaches time-warp invariance.
            # Test timings are FIXED (the originals at dataset-gen time) so the
            # held-out generalization number is reproducible.
            train_inputs, _ = resample_timings(
                data["letters"][train_idx], timing_rng,
                max_timing=data.get("max_timing", 3),
                alphabet_size=ALPHABET_SIZE)
            test_inputs = [data["variable_inputs"][i] for i in test_idx]
            fwd_train = model.forward_variable(train_inputs)
            grads = model.backward_variable(train_inputs,
                                            targets[train_idx], fwd_train)
            y_train = np.stack(fwd_train["ys"])
            fwd_test = model.forward_variable(test_inputs)
            y_test = np.stack(fwd_test["ys"])
        else:
            x_train = data["one_hot"][train_idx]
            x_test = data["one_hot"][test_idx]
            fwd_train = model.forward(x_train)
            grads = model.backward(x_train, targets[train_idx], fwd_train)
            y_train = fwd_train["y"]
            fwd_test = model.forward(x_test)
            y_test = fwd_test["y"]

        # Gradient clipping by global L2 norm (stabilizes long BPTT chains)
        if grad_clip is not None and grad_clip > 0:
            gnorm = float(np.sqrt(sum(np.sum(g ** 2) for g in grads.values())))
            if gnorm > grad_clip:
                scale = grad_clip / (gnorm + 1e-12)
                for k in grads:
                    grads[k] = grads[k] * scale

        # SGD with momentum + weight decay
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

        # logging
        loss_val = SequenceLookupRNN.loss(y_train, targets[train_idx])
        train_acc = SequenceLookupRNN.all_bits_accuracy(y_train,
                                                          targets[train_idx])
        test_acc = SequenceLookupRNN.all_bits_accuracy(y_test,
                                                         targets[test_idx])
        train_pb = SequenceLookupRNN.per_bit_accuracy(y_train,
                                                       targets[train_idx])
        test_pb = SequenceLookupRNN.per_bit_accuracy(y_test,
                                                      targets[test_idx])
        n_correct_test = SequenceLookupRNN.n_correct(y_test,
                                                      targets[test_idx])
        wnorm = (np.linalg.norm(model.W_xh) + np.linalg.norm(model.W_hh)
                 + np.linalg.norm(model.W_hy))

        history["sweep"].append(sweep + 1)
        history["loss"].append(loss_val)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["train_per_bit"].append(train_pb.tolist())
        history["test_per_bit"].append(test_pb.tolist())
        history["test_n_correct"].append(int(n_correct_test))
        history["weight_norm"].append(float(wnorm))

        if (history["converged_sweep"] is None and train_acc >= 0.999):
            history["converged_sweep"] = sweep + 1
            if verbose:
                print(f"  train converged at sweep {sweep + 1}: "
                      f"train_acc=100%, test_correct={n_correct_test}/{len(test_idx)}")

        if snapshot_callback is not None and (sweep % snapshot_every == 0
                                              or sweep == n_sweeps - 1):
            snapshot_callback(sweep, model, history, data)
            history["snapshots"].append((sweep + 1, model.snapshot()))

        if verbose and (sweep % 100 == 0 or sweep == n_sweeps - 1):
            print(f"  sweep {sweep+1:5d}  loss={loss_val:.4f}  "
                  f"train_acc={train_acc*100:5.1f}%  "
                  f"test_acc={test_acc*100:5.1f}%  "
                  f"test_correct={n_correct_test}/{len(test_idx)}  "
                  f"|W|={wnorm:.2f}")

    return model, history, data


# ----------------------------------------------------------------------
# Generalization test
# ----------------------------------------------------------------------

def test_generalization(model: SequenceLookupRNN, data: dict,
                        variable_timing: bool = False) -> dict:
    """Evaluate on the held-out test sequences."""
    test_idx = data["test_idx"]
    targets = data["targets"][test_idx]
    if variable_timing:
        inputs = [data["variable_inputs"][i] for i in test_idx]
        fwd = model.forward_variable(inputs)
        y = np.stack(fwd["ys"])
    else:
        x = data["one_hot"][test_idx]
        fwd = model.forward(x)
        y = fwd["y"]
    pred = np.sign(y)
    pred[pred == 0] = 1.0
    n_total = len(test_idx)
    n_correct = int(np.sum(np.all(pred == targets, axis=-1)))
    per_bit = np.mean(pred == targets, axis=0)
    return {
        "n_correct": n_correct,
        "n_total": n_total,
        "fraction_correct": n_correct / max(n_total, 1),
        "per_bit_accuracy": per_bit.tolist(),
        "predictions": pred.tolist(),
        "targets": targets.tolist(),
        "test_idx": list(test_idx),
        "test_letters": data["letters"][test_idx].tolist(),
    }


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep_seeds(n_seeds: int, n_hidden: int = 30, n_sweeps: int = 1500,
                variable_timing: bool = False, dataset_seed: int = 0,
                **kw) -> dict:
    out = {"seeds": [], "train_acc": [], "test_correct": [],
           "test_fraction": [], "converged_sweep": []}
    for s in range(n_seeds):
        model, hist, data = train(n_hidden=n_hidden, n_sweeps=n_sweeps,
                                   variable_timing=variable_timing,
                                   dataset_seed=dataset_seed,
                                   seed=s, verbose=False, **kw)
        gen = test_generalization(model, data, variable_timing=variable_timing)
        out["seeds"].append(s)
        out["train_acc"].append(hist["train_acc"][-1])
        out["test_correct"].append(gen["n_correct"])
        out["test_fraction"].append(gen["fraction_correct"])
        out["converged_sweep"].append(hist["converged_sweep"])
        print(f"  seed {s:2d}  train_acc={hist['train_acc'][-1]*100:5.1f}%  "
              f"test_correct={gen['n_correct']}/{gen['n_total']}  "
              f"converged@{hist['converged_sweep']}")
    return out


# ----------------------------------------------------------------------
# Build helper (matches the stub spec)
# ----------------------------------------------------------------------

def build_rnn(n_in: int = ALPHABET_SIZE, n_hidden: int = 30, n_out: int = N_OUT,
              seed: int = 0) -> SequenceLookupRNN:
    return SequenceLookupRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out, seed=seed)


def train_bptt(model: SequenceLookupRNN, data: dict,
               n_sweeps: int = 1500, lr: float = 0.05,
               momentum: float = 0.9, weight_decay: float = 1e-4,
               variable_timing: bool = False, verbose: bool = True) -> dict:
    """Train an existing model on the dataset (used by the stub API)."""
    velocities = {k: np.zeros_like(v) for k, v in
                  [("W_xh", model.W_xh), ("W_hh", model.W_hh),
                   ("b_h", model.b_h), ("W_hy", model.W_hy),
                   ("b_y", model.b_y)]}
    targets = data["targets"]
    train_idx = data["train_idx"]
    history = {"sweep": [], "loss": [], "train_acc": [], "test_acc": []}
    for sweep in range(n_sweeps):
        if variable_timing:
            inp = [data["variable_inputs"][i] for i in train_idx]
            fwd = model.forward_variable(inp)
            grads = model.backward_variable(inp, targets[train_idx], fwd)
            y = np.stack(fwd["ys"])
        else:
            x = data["one_hot"][train_idx]
            fwd = model.forward(x)
            grads = model.backward(x, targets[train_idx], fwd)
            y = fwd["y"]
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W_xh += velocities["W_xh"]
        model.W_hh += velocities["W_hh"]
        model.b_h += velocities["b_h"]
        model.W_hy += velocities["W_hy"]
        model.b_y += velocities["b_y"]
        if weight_decay > 0:
            shrink = (1.0 - lr * weight_decay)
            model.W_xh *= shrink; model.W_hh *= shrink; model.W_hy *= shrink
        history["sweep"].append(sweep + 1)
        history["loss"].append(SequenceLookupRNN.loss(y, targets[train_idx]))
        history["train_acc"].append(
            SequenceLookupRNN.all_bits_accuracy(y, targets[train_idx]))
    return history


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0,
                   help="model init seed")
    p.add_argument("--dataset-seed", type=int, default=0,
                   help="seed for sequence sampling (and train/test split)")
    p.add_argument("--variable-timing", action="store_true",
                   help="run the variable-timing variant (60 hidden units)")
    p.add_argument("--n-hidden", type=int, default=None,
                   help="override hidden size (default: 30 fixed-timing, "
                        "60 variable-timing)")
    p.add_argument("--n-sweeps", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--init-scale", type=float, default=None)
    p.add_argument("--grad-clip", type=float, default=None,
                   help="clip global gradient L2 norm; <=0 to disable")
    p.add_argument("--max-timing", type=int, default=2,
                   help="variable-timing only: max repeats per letter (1..max)")
    p.add_argument("--multi-seed", type=int, default=0,
                   help="if >0, sweep that many seeds and exit")
    p.add_argument("--save-results", type=str, default=None,
                   help="optional path to dump a results JSON")
    args = p.parse_args()

    _print_environment()

    if args.n_hidden is None:
        args.n_hidden = 60 if args.variable_timing else 30
    if args.n_sweeps is None:
        args.n_sweeps = 2000 if args.variable_timing else 800
    if args.lr is None:
        args.lr = 0.02 if args.variable_timing else 0.05
    if args.init_scale is None:
        args.init_scale = 0.2 if args.variable_timing else 0.5
    if args.grad_clip is None:
        args.grad_clip = 1.0 if args.variable_timing else 5.0

    if args.multi_seed > 0:
        out = sweep_seeds(n_seeds=args.multi_seed, n_hidden=args.n_hidden,
                          n_sweeps=args.n_sweeps,
                          variable_timing=args.variable_timing,
                          dataset_seed=args.dataset_seed,
                          lr=args.lr, momentum=args.momentum,
                          weight_decay=args.weight_decay,
                          init_scale=args.init_scale,
                          grad_clip=args.grad_clip,
                          max_timing=args.max_timing)
        n = len(out["seeds"])
        n_train_solved = sum(1 for a in out["train_acc"] if a >= 0.999)
        n_geq_4 = sum(1 for c in out["test_correct"] if c >= 4)
        median_test = int(np.median(out["test_correct"]))
        print(f"\nseeds={n}  train_solved={n_train_solved}/{n}  "
              f"test>=4/5: {n_geq_4}/{n}  median_test={median_test}/5")
        return

    t0 = time.time()
    model, hist, data = train(n_hidden=args.n_hidden, n_sweeps=args.n_sweeps,
                               lr=args.lr, momentum=args.momentum,
                               weight_decay=args.weight_decay,
                               init_scale=args.init_scale, seed=args.seed,
                               dataset_seed=args.dataset_seed,
                               variable_timing=args.variable_timing,
                               grad_clip=args.grad_clip,
                               max_timing=args.max_timing)
    wallclock = time.time() - t0

    gen = test_generalization(model, data, variable_timing=args.variable_timing)

    print("\n=== final ===")
    print(f"final train accuracy : {hist['train_acc'][-1]*100:.1f}%")
    print(f"final test  accuracy : {gen['fraction_correct']*100:.1f}% "
          f"({gen['n_correct']}/{gen['n_total']} held-out sequences)")
    print(f"per-bit train acc    : "
          + " ".join(f"b{i}={a*100:.1f}%"
                     for i, a in enumerate(hist['train_per_bit'][-1])))
    print(f"per-bit test  acc    : "
          + " ".join(f"b{i}={a*100:.1f}%"
                     for i, a in enumerate(gen["per_bit_accuracy"])))
    print(f"final loss           : {hist['loss'][-1]:.5f}")
    print(f"converged sweep      : {hist['converged_sweep']}")
    print(f"wallclock            : {wallclock:.3f}s")

    if args.save_results:
        os.makedirs(os.path.dirname(args.save_results) or ".", exist_ok=True)
        with open(args.save_results, "w") as f:
            json.dump({
                "args": vars(args),
                "wallclock": wallclock,
                "final_train_acc": hist["train_acc"][-1],
                "final_test_acc": gen["fraction_correct"],
                "test_n_correct": gen["n_correct"],
                "test_n_total": gen["n_total"],
                "test_predictions": gen["predictions"],
                "test_targets": gen["targets"],
                "test_letters": gen["test_letters"],
                "converged_sweep": hist["converged_sweep"],
                "n_params": model.n_params(),
                "history_loss": hist["loss"],
                "history_train_acc": hist["train_acc"],
                "history_test_acc": hist["test_acc"],
                "git_commit": _git_commit(),
                "python": sys.version.split()[0],
                "numpy": np.__version__,
                "platform": platform.platform(),
            }, f, indent=2)
        print(f"# results saved to {args.save_results}")


def _git_commit() -> str:
    try:
        import subprocess
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "unknown"


if __name__ == "__main__":
    main()
