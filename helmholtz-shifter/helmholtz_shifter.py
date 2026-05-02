"""
Helmholtz-machine shifter (Dayan, Hinton, Neal & Zemel 1995, "The Helmholtz
machine", Neural Computation 7(5):889--904).

Problem:
  4 x 8 binary image. Row 0 (and its duplicate row 1) is uniform-Bernoulli with
  marginal p_on=0.2. Rows 2-3 (duplicates) are row 0 cyclically shifted by +1
  (right) or -1 (left), each with prior probability 1/2. The latent generative
  process therefore has two factors: the shift direction (1 bit, top latent
  unit) and the row-0 bit pattern (8 bits, captured by the hidden layer's
  shifted-pair detectors).

Architecture (sigmoid belief net, top-down generative + bottom-up recognition):

    v (32 visible) <-- W_hv -- h (n_hidden=16) <-- W_th -- t (1 top)
    v (32 visible)  -- R_vh --> h (n_hidden=16)  -- R_ht --> t (1 top)

  All units are binary stochastic; each layer's conditional is factorial.
  After wake-sleep, the single top unit becomes shift-direction selective
  (top=1 -> one direction, top=0 -> the other) and the n_hidden layer-2
  units specialise to individual shifted bit-pairs (one pixel in row 0 + one
  pixel in row 3 at a fixed offset).

Learning:
  Wake-sleep (Hinton, Dayan, Frey & Neal 1995). Two alternating phases:

  - Wake (data drives, generative weights learn):
      sample v from the data; pass it through the recognition net for a
      latent (h, t) sample; update each layer's generative weights to predict
      the layer below given the latents the recognition net inferred (delta
      rule: Delta W proportional to (target - sigma(predicted)) * latent^T).

  - Sleep (model drives, recognition weights learn):
      sample (t, h, v) "fantasy" tuple from the generative net; update
      recognition weights to invert the generative net (same delta rule).

  No backprop; both updates are local 1-step rules.
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Shifter distribution
# ----------------------------------------------------------------------

H = 4              # image height
W = 8              # image width
N_VIS = H * W      # 32 visible units
N_HID_DEFAULT = 16 # 8 positions x 2 directions
N_TOP_DEFAULT = 4  # multi-unit top layer (single-unit version cannot break the
                   # t -> 1 - t symmetry of wake-sleep on this task; with >=2
                   # top units, individual units become direction selective)
P_ON = 0.2         # marginal pixel-on probability for the source row
P_RIGHT = 0.5      # prior of right-shift direction


def generate_shifter(n_samples: int, p_on: float = P_ON, p_right: float = P_RIGHT,
                     rng: np.random.Generator | None = None
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Sample `n_samples` 4x8 shifter images.

    Returns:
      v          float32 array, shape (n_samples, 32). Pixel order: row-major
                 over the 4x8 image.
      direction  int8 array, shape (n_samples,). 1 = right shift, 0 = left
                 shift. Latent ground-truth label, used only for inspection.
    """
    if rng is None:
        rng = np.random.default_rng()
    v_top = (rng.random((n_samples, W)) < p_on).astype(np.float32)
    is_right = (rng.random(n_samples) < p_right).astype(np.int8)
    shifts = np.where(is_right == 1, 1, -1)                  # +1 right, -1 left
    # cyclic shift each row independently
    cols = np.arange(W)[None, :] - shifts[:, None]           # (n, W)
    cols = cols % W
    v_bot = np.take_along_axis(v_top, cols, axis=1)
    img = np.zeros((n_samples, H, W), dtype=np.float32)
    img[:, 0] = v_top
    img[:, 1] = v_top
    img[:, 2] = v_bot
    img[:, 3] = v_bot
    return img.reshape(n_samples, N_VIS), is_right


# ----------------------------------------------------------------------
# Helmholtz machine
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _log_sigmoid(x: np.ndarray) -> np.ndarray:
    """Numerically stable log sigmoid(x)."""
    return -np.log1p(np.exp(-np.clip(x, -50.0, 50.0)))


def _log_one_minus_sigmoid(x: np.ndarray) -> np.ndarray:
    """log (1 - sigmoid(x)) = log sigmoid(-x)."""
    return -np.log1p(np.exp(np.clip(x, -50.0, 50.0)))


def _bernoulli(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(p.shape) < p).astype(np.float32)


def _log_bernoulli(x: np.ndarray, logits: np.ndarray) -> np.ndarray:
    """log p(x | logits) for Bernoulli (factorial). Sums over the last axis."""
    log_p1 = _log_sigmoid(logits)
    log_p0 = _log_one_minus_sigmoid(logits)
    return (x * log_p1 + (1.0 - x) * log_p0).sum(axis=-1)


@dataclass
class HelmholtzMachine:
    """Three-layer sigmoid belief net (visible -- hidden -- top) trained by
    wake-sleep.

    Generative weights (top-down):
        b_top                        -- bias for the single top unit
        W_th  shape (1, n_hidden)    -- top -> hidden
        b_h   shape (n_hidden,)
        W_hv  shape (n_hidden, 32)   -- hidden -> visible
        b_v   shape (32,)

    Recognition weights (bottom-up):
        R_vh  shape (32, n_hidden)   -- visible -> hidden
        c_h   shape (n_hidden,)
        R_ht  shape (n_hidden, 1)    -- hidden -> top
        c_top shape (1,)
    """

    n_visible: int = N_VIS
    n_hidden: int = N_HID_DEFAULT
    n_top: int = N_TOP_DEFAULT
    init_scale: float = 0.1
    seed: int = 0
    init_visible_bias_to_marginal: bool = True
    p_on: float = P_ON

    # weights / biases populated in __post_init__
    W_th: np.ndarray = field(init=False)
    W_hv: np.ndarray = field(init=False)
    b_top: np.ndarray = field(init=False)
    b_h: np.ndarray = field(init=False)
    b_v: np.ndarray = field(init=False)
    R_vh: np.ndarray = field(init=False)
    R_ht: np.ndarray = field(init=False)
    c_h: np.ndarray = field(init=False)
    c_top: np.ndarray = field(init=False)
    rng: np.random.Generator = field(init=False)

    def __post_init__(self):
        self.rng = np.random.default_rng(self.seed)
        rs = self.init_scale
        self.W_th = (rs * self.rng.standard_normal((self.n_top, self.n_hidden))
                     ).astype(np.float32)
        self.W_hv = (rs * self.rng.standard_normal((self.n_hidden, self.n_visible))
                     ).astype(np.float32)
        self.b_top = np.zeros(self.n_top, dtype=np.float32)
        self.b_h = np.zeros(self.n_hidden, dtype=np.float32)
        if self.init_visible_bias_to_marginal:
            p_clip = float(np.clip(self.p_on, 1e-3, 1.0 - 1e-3))
            self.b_v = np.full(self.n_visible,
                               np.log(p_clip / (1.0 - p_clip)),
                               dtype=np.float32)
        else:
            self.b_v = np.zeros(self.n_visible, dtype=np.float32)
        self.R_vh = (rs * self.rng.standard_normal((self.n_visible, self.n_hidden))
                     ).astype(np.float32)
        self.R_ht = (rs * self.rng.standard_normal((self.n_hidden, self.n_top))
                     ).astype(np.float32)
        self.c_h = np.zeros(self.n_hidden, dtype=np.float32)
        self.c_top = np.zeros(self.n_top, dtype=np.float32)

    # ---- recognition (bottom-up) ---------------------------------------

    def recognize(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Sample (h, t) ~ q(h|v) q(t|h). v: (B, n_visible)."""
        p_h = sigmoid(v @ self.R_vh + self.c_h)
        h = _bernoulli(p_h, self.rng)
        p_t = sigmoid(h @ self.R_ht + self.c_top)
        t = _bernoulli(p_t, self.rng)
        return h, t

    def recognize_probs(self, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Return (q(h=1|v), q(t=1|h)) as deterministic probabilities, with h
        sampled in between (matches the stochastic recognition path)."""
        p_h = sigmoid(v @ self.R_vh + self.c_h)
        h = _bernoulli(p_h, self.rng)
        p_t = sigmoid(h @ self.R_ht + self.c_top)
        return p_h, p_t

    # ---- generation (top-down) -----------------------------------------

    def generate(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (v, h, t) from the generative model."""
        p_t = sigmoid(self.b_top)
        t = _bernoulli(np.broadcast_to(p_t, (batch_size, self.n_top)).copy(),
                       self.rng)
        p_h = sigmoid(t @ self.W_th + self.b_h)
        h = _bernoulli(p_h, self.rng)
        p_v = sigmoid(h @ self.W_hv + self.b_v)
        v = _bernoulli(p_v, self.rng)
        return v, h, t

    def generate_conditional(self, batch_size: int,
                             top_value: int) -> tuple[np.ndarray, np.ndarray]:
        """Sample (v, h) from p(v, h | top=top_value)."""
        t = np.full((batch_size, self.n_top), float(top_value), dtype=np.float32)
        p_h = sigmoid(t @ self.W_th + self.b_h)
        h = _bernoulli(p_h, self.rng)
        p_v = sigmoid(h @ self.W_hv + self.b_v)
        v = _bernoulli(p_v, self.rng)
        return v, h

    # ---- importance-sampled NLL ----------------------------------------

    def importance_log_prob(self, v: np.ndarray, n_samples: int = 50,
                            rng: np.random.Generator | None = None
                            ) -> np.ndarray:
        """Estimate log p_model(v) by importance sampling with the recognition
        net as proposal.

        v: (B, n_visible). Returns shape (B,) array of log-prob estimates.

        log p(v) = log E_{q(h,t|v)} [ p(v|h) p(h|t) p(t) / q(h|v) q(t|h) ]

        Uses M = n_samples samples per v.
        """
        if rng is None:
            rng = np.random.default_rng()
        B = v.shape[0]
        M = n_samples
        # tile v to (M, B, n_visible) by broadcasting, sample latents
        z_h_q = v @ self.R_vh + self.c_h                     # (B, n_hidden)
        # draw h_m ~ q(h | v) for each sample
        h_logits = np.broadcast_to(z_h_q, (M, B, self.n_hidden))
        h_samples = (rng.random(h_logits.shape)
                     < sigmoid(h_logits)).astype(np.float32)
        # log q(h | v)
        log_q_h = _log_bernoulli(h_samples, h_logits)         # (M, B)
        # draw t_m ~ q(t | h_m)
        t_logits = h_samples @ self.R_ht + self.c_top         # (M, B, n_top)
        t_samples = (rng.random(t_logits.shape)
                     < sigmoid(t_logits)).astype(np.float32)
        log_q_t = _log_bernoulli(t_samples, t_logits)         # (M, B)
        # log p(t)
        log_p_t = _log_bernoulli(t_samples,
                                 np.broadcast_to(self.b_top, t_logits.shape))
        # log p(h | t)
        h_logits_gen = t_samples @ self.W_th + self.b_h       # (M, B, n_hidden)
        log_p_h_given_t = _log_bernoulli(h_samples, h_logits_gen)
        # log p(v | h)
        v_logits = h_samples @ self.W_hv + self.b_v           # (M, B, n_visible)
        log_p_v_given_h = _log_bernoulli(
            np.broadcast_to(v, (M, B, self.n_visible)), v_logits
        )
        # log w_m = log p(v|h) + log p(h|t) + log p(t) - log q(h|v) - log q(t|h)
        log_w = log_p_v_given_h + log_p_h_given_t + log_p_t - log_q_h - log_q_t
        # log p(v) approx = log mean_m w_m = logsumexp(log_w, axis=0) - log M
        m_max = log_w.max(axis=0, keepdims=True)
        log_mean_w = (m_max.squeeze(0)
                      + np.log(np.exp(log_w - m_max).mean(axis=0)))
        return log_mean_w


# convenience constructor matching the stub spec
def build_helmholtz_machine(n_visible: int = N_VIS, n_hidden_2: int = N_HID_DEFAULT,
                            n_hidden_3: int = N_TOP_DEFAULT, *, seed: int = 0,
                            init_scale: float = 0.1
                            ) -> HelmholtzMachine:
    return HelmholtzMachine(n_visible=n_visible, n_hidden=n_hidden_2,
                            n_top=n_hidden_3, init_scale=init_scale, seed=seed)


# ----------------------------------------------------------------------
# Wake / sleep updates
# ----------------------------------------------------------------------

def _wake_update(net: HelmholtzMachine, v: np.ndarray, lr: float) -> None:
    """One wake update on a batch of data v: (B, n_visible).

    Generative weights learn to predict the layer below given the latents
    that the recognition net inferred.
    """
    h, t = net.recognize(v)
    B = v.shape[0]
    p_v_pred = sigmoid(h @ net.W_hv + net.b_v)
    p_h_pred = sigmoid(t @ net.W_th + net.b_h)
    p_t_pred = sigmoid(np.broadcast_to(net.b_top, (B, net.n_top)))
    err_v = v - p_v_pred
    err_h = h - p_h_pred
    err_t = t - p_t_pred
    net.W_hv += lr * (h.T @ err_v) / B
    net.b_v  += lr * err_v.mean(axis=0)
    net.W_th += lr * (t.T @ err_h) / B
    net.b_h  += lr * err_h.mean(axis=0)
    net.b_top += lr * err_t.mean(axis=0)


def _sleep_update(net: HelmholtzMachine, batch_size: int, lr: float) -> None:
    """One sleep update on a batch of size B sampled from the generative net."""
    v, h, t = net.generate(batch_size)
    p_h_pred = sigmoid(v @ net.R_vh + net.c_h)
    p_t_pred = sigmoid(h @ net.R_ht + net.c_top)
    err_h = h - p_h_pred
    err_t = t - p_t_pred
    net.R_vh += lr * (v.T @ err_h) / batch_size
    net.c_h  += lr * err_h.mean(axis=0)
    net.R_ht += lr * (h.T @ err_t) / batch_size
    net.c_top += lr * err_t.mean(axis=0)


def wake_sleep(model: HelmholtzMachine, data_rng: np.random.Generator,
               n_passes: int, lr: float = 0.1, batch_size: int = 1,
               eval_every: int = 0,
               eval_callback=None,
               eval_set: tuple[np.ndarray, np.ndarray] | None = None,
               eval_n_samples: int = 50) -> dict:
    """Run `n_passes` wake-sleep iterations on `model`.

    One pass = one wake update on a fresh data batch + one sleep update on a
    fresh fantasy batch. Total data samples consumed = n_passes * batch_size.

    `eval_set = (v_eval, dir_eval)` is a fixed held-out set used for the
    importance-sampled NLL and direction-recovery accuracy.

    `eval_callback(step, model, history)` is invoked every `eval_every` passes
    if both arguments are supplied.
    """
    history = {"step": [], "samples": [],
               "is_nll_bits": [], "dir_acc": []}
    eval_rng = np.random.default_rng(0)
    for step in range(n_passes):
        v_batch, _ = generate_shifter(batch_size, p_on=model.p_on, rng=data_rng)
        _wake_update(model, v_batch, lr)
        _sleep_update(model, batch_size, lr)
        if eval_every > 0 and (step + 1) % eval_every == 0:
            if eval_set is None:
                v_eval, dir_eval = generate_shifter(256, p_on=model.p_on,
                                                    rng=eval_rng)
            else:
                v_eval, dir_eval = eval_set
            log_p = model.importance_log_prob(v_eval, n_samples=eval_n_samples,
                                              rng=eval_rng)
            nll_bits = float(-log_p.mean() / np.log(2.0))
            dir_acc = float(direction_recovery(model, v_eval, dir_eval, n_draws=5,
                                               rng=eval_rng))
            history["step"].append(step + 1)
            history["samples"].append((step + 1) * batch_size)
            history["is_nll_bits"].append(nll_bits)
            history["dir_acc"].append(dir_acc)
            if eval_callback is not None:
                eval_callback(step + 1, model, history)
    return history


# ----------------------------------------------------------------------
# Inspection / evaluation
# ----------------------------------------------------------------------

def direction_recovery(model: HelmholtzMachine, v: np.ndarray,
                       direction_true: np.ndarray, n_draws: int = 5,
                       rng: np.random.Generator | None = None) -> float:
    """Recognise q(t | v) deterministically (via the sigmoid means, marginalising
    over the recognition path's h sample) and find the best linear projection of
    the n_top-dimensional top vector that predicts direction_true.

    Concretely: for each draw, sample h ~ q(h|v), compute p(t_k=1|h) for each k,
    average over draws to get a (B, n_top) score matrix, then for each candidate
    sign vector s in {-1, +1}^n_top compute the accuracy of (sum_k s_k * score_k > 0)
    against direction_true, and return the best (with sign-flip allowed).

    Falls back gracefully when n_top is small enough to enumerate (<=6)."""
    if rng is None:
        rng = np.random.default_rng()
    B = v.shape[0]
    z_h = v @ model.R_vh + model.c_h
    p_h = sigmoid(z_h)
    h_samples = (rng.random((n_draws, B, model.n_hidden))
                 < p_h).astype(np.float32)
    p_t = sigmoid(h_samples @ model.R_ht + model.c_top)      # (n_draws, B, n_top)
    t_score = p_t.mean(axis=0)                                # (B, n_top), in [0, 1]
    centred = t_score - t_score.mean(axis=0, keepdims=True)   # mean-zero per unit
    n_top = model.n_top
    best_acc = 0.5
    if n_top <= 6:
        for mask in range(1, 2 ** n_top):
            signs = np.array([+1 if (mask >> k) & 1 else -1
                              for k in range(n_top)], dtype=np.float32)
            proj = centred @ signs                             # (B,)
            pred = (proj > 0).astype(np.int8)
            agree = float((pred == direction_true).mean())
            best_acc = max(best_acc, agree, 1.0 - agree)
    else:
        # n_top large -- pick the single most-discriminative unit
        for k in range(n_top):
            pred = (centred[:, k] > 0).astype(np.int8)
            agree = float((pred == direction_true).mean())
            best_acc = max(best_acc, agree, 1.0 - agree)
    return best_acc


def _shift_signature(vs: np.ndarray) -> np.ndarray:
    """+1 if bottom row equals top row shifted right, -1 if shifted left,
    0 if neither or both (e.g. ambiguous all-zero / all-one row)."""
    imgs = (vs.reshape(-1, H, W) > 0.5)
    top_row = imgs[:, 0]
    bot_row = imgs[:, 3]
    is_right = (bot_row == np.roll(top_row, 1, axis=1)).all(axis=1)
    is_left = (bot_row == np.roll(top_row, -1, axis=1)).all(axis=1)
    sig = np.zeros(len(vs), dtype=np.int8)
    sig[is_right & ~is_left] = +1
    sig[is_left & ~is_right] = -1
    return sig


def inspect_layer3_units(model: HelmholtzMachine, n_fantasy: int = 1024,
                         rng: np.random.Generator | None = None) -> dict:
    """For each top unit k, generate fantasies under one-hot top=e_k vs
    top=0 and measure shift-direction selectivity.

    The `selectivity[k]` score is in [-1, 1]:
        P(right-shift fantasy | t_k=1) - P(right-shift fantasy | t_k=0).
    A unit with |selectivity| close to 1 is a clean shift-direction detector.

    Returns a dict with per-unit arrays and aggregate scores. The
    `best_unit_selectivity` is max_k |selectivity[k]| -- the most
    direction-selective unit's score.
    """
    if rng is None:
        rng = model.rng
    saved_rng = model.rng
    model.rng = rng

    n_top = model.n_top
    p_right = np.zeros(n_top, dtype=np.float32)
    p_left = np.zeros(n_top, dtype=np.float32)
    p_right_off = np.zeros(n_top, dtype=np.float32)
    p_left_off = np.zeros(n_top, dtype=np.float32)

    for k in range(n_top):
        # one-hot top with unit k on
        t_on = np.zeros((n_fantasy, n_top), dtype=np.float32)
        t_on[:, k] = 1.0
        p_h = sigmoid(t_on @ model.W_th + model.b_h)
        h = _bernoulli(p_h, model.rng)
        p_v = sigmoid(h @ model.W_hv + model.b_v)
        v_on = _bernoulli(p_v, model.rng)
        sig_on = _shift_signature(v_on)
        p_right[k] = float((sig_on == +1).mean())
        p_left[k] = float((sig_on == -1).mean())

        # one-hot complement (all top units off except not-k)
        t_off = np.zeros((n_fantasy, n_top), dtype=np.float32)
        p_h = sigmoid(t_off @ model.W_th + model.b_h)
        h = _bernoulli(p_h, model.rng)
        p_v = sigmoid(h @ model.W_hv + model.b_v)
        v_off = _bernoulli(p_v, model.rng)
        sig_off = _shift_signature(v_off)
        p_right_off[k] = float((sig_off == +1).mean())
        p_left_off[k] = float((sig_off == -1).mean())

    model.rng = saved_rng

    selectivity = p_right - p_right_off
    abs_sel = np.abs(selectivity)
    best_idx = int(abs_sel.argmax())
    return {
        "p_right_given_top_on": p_right,
        "p_left_given_top_on": p_left,
        "p_right_given_top_off": p_right_off,
        "p_left_given_top_off": p_left_off,
        "selectivity_per_unit": selectivity,
        "abs_selectivity_per_unit": abs_sel,
        "best_unit": best_idx,
        "best_unit_selectivity": float(abs_sel[best_idx]),
    }


def inspect_layer2_units(model: HelmholtzMachine) -> np.ndarray:
    """Return the per-hidden-unit generative receptive field, shape
    (n_hidden, H, W). The receptive field of unit j is
        p(v | h_j = 1, all others off)  -  p(v | all h off)
    so positive pixels are where the unit *adds* mass."""
    n_h = model.n_hidden
    fields = np.zeros((n_h, H, W), dtype=np.float32)
    p_v_baseline = sigmoid(model.b_v).reshape(H, W)
    for j in range(n_h):
        h = np.zeros((1, n_h), dtype=np.float32)
        h[0, j] = 1.0
        z_v = h @ model.W_hv + model.b_v
        p_v = sigmoid(z_v).reshape(H, W)
        fields[j] = p_v - p_v_baseline
    return fields


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Helmholtz-machine shifter "
                                            "(Dayan/Hinton/Neal/Zemel 1995).")
    p.add_argument("--n-passes", type=int, default=600_000,
                   help="number of wake-sleep iterations")
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--n-hidden", type=int, default=N_HID_DEFAULT)
    p.add_argument("--n-top", type=int, default=N_TOP_DEFAULT)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--p-on", type=float, default=P_ON,
                   help="row-marginal pixel-on probability")
    p.add_argument("--eval-every", type=int, default=10_000,
                   help="evaluate IS-NLL + dir-acc every N passes (0 = off)")
    return p


def main():
    args = _build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)
    model = HelmholtzMachine(n_hidden=args.n_hidden,
                             n_top=args.n_top,
                             init_scale=args.init_scale,
                             seed=args.seed,
                             p_on=args.p_on)
    print(f"# Helmholtz-shifter machine: {model.n_visible} visible + "
          f"{model.n_hidden} hidden + {model.n_top} top")

    eval_rng = np.random.default_rng(123)
    eval_set = generate_shifter(256, p_on=args.p_on, rng=eval_rng)
    log_p0 = model.importance_log_prob(eval_set[0], n_samples=50,
                                       rng=eval_rng)
    nll0 = float(-log_p0.mean() / np.log(2.0))
    print(f"# IS-NLL before training: {nll0:.4f} bits/image")

    t0 = time.time()
    last_print = [t0]

    def cb(step, m, history):
        now = time.time()
        if now - last_print[0] > 5.0 or step == args.n_passes:
            last_print[0] = now
            print(f"  step {step:>10d} / {args.n_passes:<10d}  "
                  f"samples={history['samples'][-1]:>11d}  "
                  f"IS-NLL={history['is_nll_bits'][-1]:.4f} bits  "
                  f"dir-acc={history['dir_acc'][-1]:.3f}  "
                  f"({now - t0:.1f}s elapsed)")

    history = wake_sleep(model, rng, n_passes=args.n_passes, lr=args.lr,
                         batch_size=args.batch_size,
                         eval_every=args.eval_every,
                         eval_callback=cb,
                         eval_set=eval_set)
    elapsed = time.time() - t0

    log_p_final = model.importance_log_prob(eval_set[0], n_samples=200,
                                            rng=eval_rng)
    nll_final = float(-log_p_final.mean() / np.log(2.0))
    inspect = inspect_layer3_units(model, n_fantasy=2048, rng=eval_rng)
    dir_acc = direction_recovery(model, eval_set[0], eval_set[1],
                                 n_draws=11, rng=eval_rng)

    print(f"\nFinal IS-NLL (M=200)        : {nll_final:.4f} bits/image")
    print(f"Direction recovery accuracy : {dir_acc:.3f}  (chance = 0.5)")
    print(f"Best top-unit selectivity   : {inspect['best_unit_selectivity']:.3f}  "
          f"(0 = no info, 1 = perfect)  -- unit {inspect['best_unit']}")
    for k in range(model.n_top):
        print(f"  unit {k}: P(right|t_k=1)={inspect['p_right_given_top_on'][k]:.3f}  "
              f"P(left|t_k=1)={inspect['p_left_given_top_on'][k]:.3f}  "
              f"selectivity={inspect['selectivity_per_unit'][k]:+.3f}")
    print(f"Wall-clock time             : {elapsed:.1f} sec")
    return model, history


if __name__ == "__main__":
    main()
