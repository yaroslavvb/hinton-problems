"""
Bars task — Helmholtz machine + wake-sleep reproduction of the experiment from
Hinton, Dayan, Frey & Neal, "The wake-sleep algorithm for unsupervised neural
networks", Science 268 (1995).

Problem:
  4x4 binary images. Each image has either *vertical* bars (with prior 2/3) or
  *horizontal* bars (with prior 1/3). Conditioned on the orientation, each of
  the 4 candidate bars in that orientation is independently active with
  probability 0.2; the visible image is the union (logical OR) of all active
  bars. There are 16 + 16 - 2 = 30 distinct images in the support.

Architecture:
  Sigmoid belief network with three layers, top-down generative weights and
  bottom-up recognition weights:

      v (16 visible) <-- W_hv -- h (8 hidden)  <-- W_th -- t (1 top-most)
      v (16 visible) -- R_vh --> h (8 hidden)   -- R_ht --> t (1 top-most)

  All units are binary stochastic. Each layer's conditional is factorial.
  The 8 first-hidden units are expected to specialise (after wake-sleep) to
  individual bars, while the single top-most unit picks up vertical vs
  horizontal orientation.

Learning:
  Wake-sleep (Hinton et al. 1995). Two alternating phases:

  - Wake (data drives, generative weights learn):
      sample v from the data; pass it through the recognition net to obtain
      a (h, t) latent sample; then update generative weights with the local
      delta rule  Delta W_hv  proportional to  (v - sigma(z_v_pred)) * h^T
      (and analogously for layers above), so each layer's generative
      predictor is taught to predict the layer below given the latents the
      recognition net inferred.

  - Sleep (model drives, recognition weights learn):
      sample (t, h, v) "fantasy" tuple from the generative net; update
      recognition weights with the same delta rule
      Delta R_vh  proportional to  (h - sigma(z_h_pred)) * v^T
      so the recognition net is taught to invert what the generative net
      just produced.

  No backprop; both updates are 1-step delta rules. Each pair (wake +
  sleep) consumes one batch from each direction.
"""

from __future__ import annotations
import argparse
import time
from dataclasses import dataclass, field

import numpy as np


# ----------------------------------------------------------------------
# Bars distribution
# ----------------------------------------------------------------------

H = 4   # image height
W = 4   # image width
N_VIS = H * W
N_BARS = H + W   # 4 vertical + 4 horizontal bar candidates
N_HID_DEFAULT = 8
P_VERTICAL = 2.0 / 3.0
P_BAR = 0.2


def _bar_template(idx: int) -> np.ndarray:
    """Return the 16-pixel template for bar `idx`.

    idx in 0..3   -> vertical bar in column idx (4 lit pixels per column).
    idx in 4..7   -> horizontal bar in row (idx - 4).
    """
    img = np.zeros((H, W), dtype=np.float32)
    if idx < W:
        img[:, idx] = 1.0
    else:
        img[idx - W, :] = 1.0
    return img.reshape(N_VIS)


# pre-compute a (8, 16) matrix of bar templates so we can OR them by maxing.
_BAR_MATRIX = np.stack([_bar_template(i) for i in range(N_BARS)])
_VERTICAL_BARS = np.arange(0, W)              # indices 0..3
_HORIZONTAL_BARS = np.arange(W, N_BARS)       # indices 4..7


def generate_bars(n_samples: int, p_vertical: float = P_VERTICAL,
                  p_bar: float = P_BAR, rng: np.random.Generator | None = None
                  ) -> np.ndarray:
    """Sample `n_samples` 16-pixel bars images from the hierarchical prior.

    Returns float32 array of shape (n_samples, 16) with values in {0, 1}.

    Procedure: for each sample, flip a biased coin for orientation (P=p_vertical
    for vertical, 1-p_vertical for horizontal). Then turn each of the 4 bars
    of that orientation on independently with probability p_bar. The image
    is the union (OR) of the active bars.
    """
    if rng is None:
        rng = np.random.default_rng()
    is_vertical = rng.random(n_samples) < p_vertical
    out = np.zeros((n_samples, N_VIS), dtype=np.float32)
    # vectorised bar activations per sample: shape (n_samples, 4)
    vert_active = (rng.random((n_samples, W)) < p_bar).astype(np.float32)
    horiz_active = (rng.random((n_samples, H)) < p_bar).astype(np.float32)
    # mask out the off-orientation bars
    vert_active = vert_active * is_vertical[:, None].astype(np.float32)
    horiz_active = horiz_active * (~is_vertical)[:, None].astype(np.float32)
    out += vert_active @ _BAR_MATRIX[_VERTICAL_BARS]
    out += horiz_active @ _BAR_MATRIX[_HORIZONTAL_BARS]
    np.minimum(out, 1.0, out=out)   # OR (saturate at 1)
    return out


# ----- exact data distribution (used for KL evaluation) -----------------

def _enumerate_bar_distribution() -> tuple[np.ndarray, np.ndarray]:
    """Return (images, probs): the exact discrete distribution over 4x4 images.

    `images`  -- float32 array, shape (N_unique, 16), each row a {0,1} image.
    `probs`   -- float64 array, shape (N_unique,), summing to 1.

    Enumerates 2^4 vertical configurations + 2^4 horizontal configurations and
    deduplicates (the all-zero and all-on patterns appear under both
    orientations).
    """
    image_to_prob: dict[bytes, float] = {}

    # vertical: 2^4 = 16 configs
    for mask in range(2 ** W):
        bits = np.array([(mask >> j) & 1 for j in range(W)], dtype=np.float32)
        img = (bits[None, :] * np.ones((H, 1), dtype=np.float32)).reshape(N_VIS)
        n_on = int(bits.sum())
        prob = P_VERTICAL * (P_BAR ** n_on) * ((1.0 - P_BAR) ** (W - n_on))
        key = img.astype(np.uint8).tobytes()
        image_to_prob[key] = image_to_prob.get(key, 0.0) + prob

    # horizontal: 2^4 = 16 configs
    for mask in range(2 ** H):
        bits = np.array([(mask >> i) & 1 for i in range(H)], dtype=np.float32)
        img = (bits[:, None] * np.ones((1, W), dtype=np.float32)).reshape(N_VIS)
        n_on = int(bits.sum())
        prob = ((1.0 - P_VERTICAL) * (P_BAR ** n_on) *
                ((1.0 - P_BAR) ** (H - n_on)))
        key = img.astype(np.uint8).tobytes()
        image_to_prob[key] = image_to_prob.get(key, 0.0) + prob

    images = np.array([np.frombuffer(k, dtype=np.uint8).astype(np.float32)
                       for k in image_to_prob.keys()])
    probs = np.array(list(image_to_prob.values()), dtype=np.float64)
    probs /= probs.sum()  # normalise (numerical hygiene; should be ~1)
    return images, probs


# Cached: 30 unique images and their data-distribution probabilities.
DATA_IMAGES, DATA_PROBS = _enumerate_bar_distribution()


# ----------------------------------------------------------------------
# Helmholtz machine
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def _bernoulli(p: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    return (rng.random(p.shape) < p).astype(np.float32)


@dataclass
class HelmholtzMachine:
    """Three-layer sigmoid belief net (visible -- hidden -- top) trained by
    wake-sleep.

    Generative weights (top-down):
        b_top                    -- bias for the single top unit
        W_th  shape (1, 8)       -- top -> hidden (each col is one h-unit's
                                     coupling to the top unit)
        b_h   shape (8,)
        W_hv  shape (8, 16)      -- hidden -> visible
        b_v   shape (16,)

    Recognition weights (bottom-up):
        R_vh  shape (16, 8)      -- visible -> hidden
        c_h   shape (8,)
        R_ht  shape (8, 1)       -- hidden -> top
        c_top shape (1,)
    """

    n_visible: int = N_VIS
    n_hidden: int = N_HID_DEFAULT
    n_top: int = 1
    init_scale: float = 0.1
    seed: int = 0
    init_visible_bias_to_marginal: bool = True

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
        # Optionally initialise b_v to the data-marginal logit so the network
        # starts already producing roughly the pixel marginal under the
        # all-hidden-off path. With b_v = 0 (symmetric init) every hidden
        # unit must learn a substantial negative contribution before any
        # pixel is "off by default"; the marginal init removes that dead
        # start without otherwise biasing the wake-sleep dynamics.
        if self.init_visible_bias_to_marginal:
            pixel_p = float(np.clip(
                (DATA_PROBS[:, None] * DATA_IMAGES).sum(axis=0).mean(),
                1e-3, 1 - 1e-3))
            self.b_v = (np.full(self.n_visible,
                                np.log(pixel_p / (1.0 - pixel_p)),
                                dtype=np.float32))
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

    # ---- generation (top-down) -----------------------------------------

    def generate(self, batch_size: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Sample (t, h, v) from the generative model."""
        p_t = sigmoid(self.b_top)
        t = _bernoulli(np.broadcast_to(p_t, (batch_size, self.n_top)).copy(),
                       self.rng)
        p_h = sigmoid(t @ self.W_th + self.b_h)
        h = _bernoulli(p_h, self.rng)
        p_v = sigmoid(h @ self.W_hv + self.b_v)
        v = _bernoulli(p_v, self.rng)
        return v, h, t

    def model_visible_prob(self, v_query: np.ndarray) -> np.ndarray:
        """Exact p_model(v) for each v in v_query, by enumerating the 2^9 = 512
        latent configurations (1 top * 8 hidden bits).

        v_query: (Q, n_visible). Returns (Q,) probabilities.
        """
        Q = v_query.shape[0]
        # enumerate all (top, h) configurations
        n_h = self.n_hidden
        h_states = np.array(
            [[(idx >> j) & 1 for j in range(n_h)] for idx in range(2 ** n_h)],
            dtype=np.float32,
        )  # (256, 8)
        log_probs = np.zeros((Q,), dtype=np.float64)
        out_probs = np.zeros((Q,), dtype=np.float64)
        for top_val in (0.0, 1.0):
            t = np.array([[top_val]], dtype=np.float32)              # (1,1)
            log_p_t = float(np.log(sigmoid(self.b_top))) if top_val == 1.0 \
                else float(np.log1p(-sigmoid(self.b_top)))
            # p(h | t) for all 8 components, broadcast across h_states
            z_h = (t @ self.W_th + self.b_h).reshape(n_h)            # (8,)
            log_p_h_one = -np.log1p(np.exp(-np.clip(z_h, -50, 50)))    # log sigmoid(z_h)
            log_p_h_zero = -np.log1p(np.exp(np.clip(z_h, -50, 50)))    # log (1 - sigmoid(z_h))
            # log p(h | t) for each of the 256 h_states:
            log_p_h_given_t = (h_states * log_p_h_one + (1 - h_states) * log_p_h_zero).sum(axis=1)
            # log p(v | h) for each (Q, 256) pair
            # z_v: (256, n_visible)
            z_v = h_states @ self.W_hv + self.b_v                    # (256, 16)
            # log_p_v_one[s, i] = log sigmoid(z_v[s, i])
            log_sig = -np.log1p(np.exp(-np.clip(z_v, -50, 50)))
            log_one_minus_sig = -np.log1p(np.exp(np.clip(z_v, -50, 50)))
            # for each query v_q, log p(v_q | h_s) = sum_i log p(v_q_i | h_s)
            # shape (Q, 256)
            log_p_v_given_h = (v_query @ log_sig.T
                               + (1.0 - v_query) @ log_one_minus_sig.T)
            # combine: log p(v_q, h | t) = log p(v_q | h) + log p(h | t)
            log_joint = log_p_v_given_h + log_p_h_given_t[None, :]   # (Q, 256)
            # marginalise over h: p(v_q | t) = sum_h exp(log_joint)
            m = log_joint.max(axis=1, keepdims=True)
            log_p_v_given_t = (m.squeeze(-1) +
                               np.log(np.exp(log_joint - m).sum(axis=1)))
            out_probs += np.exp(log_p_t + log_p_v_given_t)
        return out_probs


def build_helmholtz(n_visible: int = N_VIS, n_hidden_1: int = N_HID_DEFAULT,
                    n_hidden_2: int = 1, *, seed: int = 0,
                    init_scale: float = 0.1,
                    init_visible_bias_to_marginal: bool = True
                    ) -> HelmholtzMachine:
    """Factory matching the stub-spec name. `n_hidden_2` becomes `n_top`.

    Kept as a thin wrapper so callers that follow the spec verbatim
    (`build_helmholtz(16, 8, 1)`) work without touching the dataclass."""
    return HelmholtzMachine(n_visible=n_visible, n_hidden=n_hidden_1,
                            n_top=n_hidden_2, init_scale=init_scale,
                            seed=seed,
                            init_visible_bias_to_marginal=init_visible_bias_to_marginal)


# ----------------------------------------------------------------------
# Wake / sleep updates
# ----------------------------------------------------------------------

def _wake_update(net: HelmholtzMachine, v: np.ndarray, lr: float) -> None:
    """One wake update on a batch of data v: (B, n_visible).

    Generative weights learn to predict the layer below given the latents
    that the recognition net inferred.
    """
    h, t = net.recognize(v)                                          # (B, 8), (B, 1)
    B = v.shape[0]
    # predictions from generative net given sampled latents
    p_v_pred = sigmoid(h @ net.W_hv + net.b_v)
    p_h_pred = sigmoid(t @ net.W_th + net.b_h)
    p_t_pred = sigmoid(np.broadcast_to(net.b_top, (B, net.n_top)))
    # delta-rule errors
    err_v = v - p_v_pred                                             # (B, 16)
    err_h = h - p_h_pred                                             # (B, 8)
    err_t = t - p_t_pred                                             # (B, 1)
    # gradient updates
    net.W_hv += lr * (h.T @ err_v) / B
    net.b_v += lr * err_v.mean(axis=0)
    net.W_th += lr * (t.T @ err_h) / B
    net.b_h += lr * err_h.mean(axis=0)
    net.b_top += lr * err_t.mean(axis=0)


def _sleep_update(net: HelmholtzMachine, batch_size: int, lr: float) -> None:
    """One sleep update on a batch of size B sampled from the generative net.

    Recognition weights learn to invert what the generative net just produced.
    """
    v, h, t = net.generate(batch_size)
    p_h_pred = sigmoid(v @ net.R_vh + net.c_h)                       # q(h | v)
    p_t_pred = sigmoid(h @ net.R_ht + net.c_top)                     # q(t | h)
    err_h = h - p_h_pred
    err_t = t - p_t_pred
    net.R_vh += lr * (v.T @ err_h) / batch_size
    net.c_h += lr * err_h.mean(axis=0)
    net.R_ht += lr * (h.T @ err_t) / batch_size
    net.c_top += lr * err_t.mean(axis=0)


def wake_sleep(model: HelmholtzMachine, data_rng: np.random.Generator,
               n_steps: int, lr: float, batch_size: int = 1,
               eval_every: int = 0,
               eval_callback=None) -> dict:
    """Run `n_steps` wake-sleep iterations on `model`.

    One step = one wake update on a fresh data batch + one sleep update on a
    fresh fantasy batch. `n_steps * batch_size` is the total data-sample
    count.

    `eval_callback(step, model, history)` is invoked every `eval_every` steps
    if both arguments are supplied.
    """
    history = {"step": [], "samples": [], "kl_bits": [], "neg_log_lik": []}
    for step in range(n_steps):
        v_batch = generate_bars(batch_size, rng=data_rng)
        _wake_update(model, v_batch, lr)
        _sleep_update(model, batch_size, lr)
        if eval_every > 0 and (step + 1) % eval_every == 0:
            kl = asymmetric_kl(model)
            nll = neg_log_likelihood(model)
            history["step"].append(step + 1)
            history["samples"].append((step + 1) * batch_size)
            history["kl_bits"].append(float(kl))
            history["neg_log_lik"].append(float(nll))
            if eval_callback is not None:
                eval_callback(step + 1, model, history)
    return history


# ----------------------------------------------------------------------
# Evaluation
# ----------------------------------------------------------------------

def asymmetric_kl(model: HelmholtzMachine,
                  data_images: np.ndarray = DATA_IMAGES,
                  data_probs: np.ndarray = DATA_PROBS) -> float:
    """KL[p_data || p_model] in BITS, computed exactly.

    p_data is the discrete bars distribution over its 30-image support.
    p_model is the marginal over visible obtained by enumerating the 512
    latent configurations of the 8-hidden + 1-top network.

    Returns a non-negative scalar (bits).
    """
    p_model = model.model_visible_prob(data_images)
    # guard against numerical underflow
    p_model = np.clip(p_model, 1e-30, 1.0)
    p_data = np.clip(data_probs, 1e-30, 1.0)
    kl_nats = float(np.sum(p_data * (np.log(p_data) - np.log(p_model))))
    return kl_nats / np.log(2.0)


def neg_log_likelihood(model: HelmholtzMachine,
                       data_images: np.ndarray = DATA_IMAGES,
                       data_probs: np.ndarray = DATA_PROBS) -> float:
    """- E_{p_data} [log_2 p_model(v)]  (bits)."""
    p_model = np.clip(model.model_visible_prob(data_images), 1e-30, 1.0)
    return float(-np.sum(data_probs * np.log2(p_model)))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Bars Helmholtz machine "
                                            "(Hinton/Dayan/Frey/Neal 1995).")
    p.add_argument("--n-steps", type=int, default=2_000_000,
                   help="number of wake-sleep iterations "
                        "(each iteration = 1 wake + 1 sleep update)")
    p.add_argument("--lr", type=float, default=0.01,
                   help="learning rate for both wake and sleep updates")
    p.add_argument("--batch-size", type=int, default=20,
                   help="samples per wake-sleep iteration "
                        "(B=1 = pure online; larger B = effectively a "
                        "noisy minibatch wake-sleep)")
    p.add_argument("--n-hidden", type=int, default=N_HID_DEFAULT)
    p.add_argument("--init-scale", type=float, default=0.1)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--eval-every", type=int, default=10_000,
                   help="evaluate KL every N iterations (0 to disable)")
    return p


def main():
    args = _build_argparser().parse_args()
    rng = np.random.default_rng(args.seed)
    model = HelmholtzMachine(n_hidden=args.n_hidden,
                             init_scale=args.init_scale,
                             seed=args.seed)
    print(f"# Bars Helmholtz machine: {model.n_visible} visible + "
          f"{model.n_hidden} hidden + {model.n_top} top")
    print(f"# data support has {len(DATA_IMAGES)} unique images "
          f"(p_data normalisation = {DATA_PROBS.sum():.6f})")
    kl0 = asymmetric_kl(model)
    print(f"# KL[p_data || p_model] before training: {kl0:.4f} bits")

    t0 = time.time()
    last_print = [t0]

    def cb(step, m, history):
        now = time.time()
        if now - last_print[0] > 5.0 or step == args.n_steps:
            last_print[0] = now
            print(f"  step {step:>10d} / {args.n_steps:<10d}  "
                  f"samples={history['samples'][-1]:>11d}  "
                  f"KL={history['kl_bits'][-1]:.4f} bits  "
                  f"NLL={history['neg_log_lik'][-1]:.4f} bits  "
                  f"({now - t0:.1f}s elapsed)")

    history = wake_sleep(model, rng, n_steps=args.n_steps, lr=args.lr,
                         batch_size=args.batch_size,
                         eval_every=args.eval_every,
                         eval_callback=cb)
    elapsed = time.time() - t0
    kl_final = asymmetric_kl(model)
    nll_final = neg_log_likelihood(model)
    print(f"\nFinal KL[p_data || p_model]: {kl_final:.4f} bits")
    print(f"Final NLL                  : {nll_final:.4f} bits")
    print(f"Total samples              : {args.n_steps * args.batch_size}")
    print(f"Wall-clock time            : {elapsed:.1f} sec")
    return model, history


if __name__ == "__main__":
    main()
