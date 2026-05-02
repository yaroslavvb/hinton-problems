"""
Random-dot stereograms with the Imax mutual-information objective
(Becker & Hinton 1992, Nature).

Setup
  A 1-D "world" of dots is rendered into two retinal arrays (left and right
  eye). The right eye's view is the same dot pattern shifted horizontally by
  a per-example disparity drawn from a synthetic surface. Two adjacent
  rectangular receptive fields ("modules") each see one local strip of the
  left + right pair. Each module is a small backprop MLP that emits a single
  scalar.

Imax loss (under a Gaussian assumption, equal output dim = 1)
  I(y_a; y_b) = 0.5 * log( var(y_a) + var(y_b) )
              - 0.5 * log( var(y_a - y_b) )

  Maximizing I forces the two modules to agree on whatever the dot patches
  share (the disparity from the smooth surface) while ignoring whatever they
  do not share (the random dots).

Architecture
  Module: input (2 * strip_width) -> sigmoid hidden (n_hidden) -> linear (1)
  Two modules with independent weights, trained jointly.
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
import numpy as np


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def generate_stereo_pair(rng: np.random.Generator,
                         strip_width: int = 6,
                         max_disparity: float = 3.0,
                         disparity: float | None = None,
                         dot_density: float = 0.5,
                         continuous: bool = True,
                         ) -> tuple[np.ndarray, np.ndarray, float]:
    """Render ONE stereo strip (left + right) at a single disparity.

    `continuous=True`  -- real-valued disparity, sub-pixel rendering by linear
                          interpolation over a +/-1 dot field (a smooth
                          stand-in for the band-limited dot rendering used in
                          Becker & Hinton 1992).
    `continuous=False` -- integer disparity, hard pixel shift (simpler, but
                          gives a much flatter Imax landscape — kept as an
                          option for diagnostics).

    Returns (left, right, disparity). Each view is a 1-D float array of length
    `strip_width`. The right view is the left view shifted by `disparity`
    pixels:  right[i] = left[i - disparity].  The render buffer is padded so
    pixels that fall outside the strip on either eye come from independent
    random dots — so only the disparity itself is a stable cue.
    """
    pad = int(np.ceil(max_disparity)) + 1
    pattern = np.where(rng.random(strip_width + 2 * pad) < dot_density,
                       1.0, -1.0).astype(np.float32)
    if disparity is None:
        if continuous:
            disparity = float(rng.uniform(-max_disparity, max_disparity))
        else:
            disparity = float(rng.integers(-int(max_disparity),
                                            int(max_disparity) + 1))

    # Left view: integer indexing.
    left = pattern[pad : pad + strip_width]

    # Right view: shifted by `disparity` (pattern[i - d] for visible i).
    if continuous:
        # Linear interpolation between adjacent integer dots.
        positions = np.arange(strip_width, dtype=np.float32) - float(disparity)
        positions = positions + pad   # buffer offset
        i0 = np.floor(positions).astype(np.int64)
        frac = (positions - i0).astype(np.float32)
        # Clip indices safely to the buffer
        i0 = np.clip(i0, 0, len(pattern) - 2)
        right = (1.0 - frac) * pattern[i0] + frac * pattern[i0 + 1]
    else:
        d_int = int(disparity)
        right = pattern[pad - d_int : pad - d_int + strip_width]
    return left.astype(np.float32), right.astype(np.float32), float(disparity)


def generate_batch(rng: np.random.Generator,
                   batch_size: int,
                   strip_width: int = 6,
                   max_disparity: float = 3.0,
                   continuous: bool = True,
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate a training batch of two-module inputs.

    Two MODULES per example, each fed an INDEPENDENT random-dot stereo strip.
    Both strips share the SAME disparity — that is the only common signal
    across modules, by construction. (This eliminates pixel-level leakage
    between adjacent receptive fields, which would otherwise let the modules
    correlate on shared dots instead of on the disparity.)

    Returns:
        x_a: (batch, 2*strip_width)  module-A input = [left_a, right_a]
        x_b: (batch, 2*strip_width)  module-B input = [left_b, right_b]
        d:   (batch,)                ground-truth disparity per example
    """
    x_a = np.empty((batch_size, 2 * strip_width), dtype=np.float32)
    x_b = np.empty((batch_size, 2 * strip_width), dtype=np.float32)
    d = np.empty(batch_size, dtype=np.float32)
    for i in range(batch_size):
        if continuous:
            disp = float(rng.uniform(-max_disparity, max_disparity))
        else:
            disp = float(rng.integers(-int(max_disparity),
                                       int(max_disparity) + 1))
        left_a, right_a, _ = generate_stereo_pair(
            rng, strip_width=strip_width,
            max_disparity=max_disparity, disparity=disp,
            continuous=continuous)
        left_b, right_b, _ = generate_stereo_pair(
            rng, strip_width=strip_width,
            max_disparity=max_disparity, disparity=disp,
            continuous=continuous)
        x_a[i, :strip_width] = left_a
        x_a[i, strip_width:] = right_a
        x_b[i, :strip_width] = left_b
        x_b[i, strip_width:] = right_b
        d[i] = disp
    return x_a, x_b, d


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def featurize(x: np.ndarray, strip_width: int) -> np.ndarray:
    """Augment raw [left, right] with binocular cross-product features.

    A 2-layer sigmoid net, trained by Imax from a random init, almost never
    escapes the flat region around y_a, y_b uncorrelated -- the only useful
    stereo cue (cross-correlation between left and right pixels) is a
    multiplicative interaction that backprop is famously bad at discovering
    from raw pixels via additive features alone.

    We hand the network the natural "binocular simple cell" features:
    elementwise products  left[i] * right[i+k]  for shifts k in a small
    window. With these as input, a small MLP can easily learn to weight
    them into a disparity readout. This is the same trick used in modern
    stereo CNNs (cost-volume / correlation layer).

    Output features per module:
        [left, right, products at shifts -3..+3]   ->  9 * strip_width
    (2 raw views + 7 product channels.)
    """
    left = x[:, :strip_width]
    right = x[:, strip_width:]
    feats = [left, right]
    # Cross-product features at small integer shifts.
    # right shifted by k aligns right[i+k] under left[i]; we pad with zeros.
    for k in range(-3, 4):
        if k == 0:
            shifted = right
        elif k > 0:
            shifted = np.concatenate(
                [right[:, k:], np.zeros((right.shape[0], k), dtype=right.dtype)],
                axis=1)
        else:
            shifted = np.concatenate(
                [np.zeros((right.shape[0], -k), dtype=right.dtype), right[:, :k]],
                axis=1)
        feats.append(left * shifted)
    return np.concatenate(feats, axis=1)


class Module:
    """One module: featurize -> sigmoid hidden -> linear scalar output.

    The input to the trainable weights is the cross-product feature map (see
    `featurize`), so the hidden layer learns a weighted combination of
    binocular cross-correlations rather than discovering them from scratch.
    """

    def __init__(self, n_in_raw: int, strip_width: int, n_hidden: int,
                 rng: np.random.Generator, init_scale: float = 0.5):
        self.strip_width = strip_width
        # featurize produces 9 * strip_width features (left, right, 7 products)
        n_in = 9 * strip_width
        self.W1 = (init_scale * rng.standard_normal((n_in, n_hidden))
                   / np.sqrt(n_in)).astype(np.float32)
        self.b1 = np.zeros(n_hidden, dtype=np.float32)
        self.W2 = (init_scale * rng.standard_normal((n_hidden, 1))
                   / np.sqrt(n_hidden)).astype(np.float32)
        self.b2 = np.zeros(1, dtype=np.float32)

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        feat = featurize(x, self.strip_width)
        h = sigmoid(feat @ self.W1 + self.b1)
        y = (h @ self.W2 + self.b2).reshape(-1)
        return y, h, feat

    def backward(self, feat: np.ndarray, h: np.ndarray, dy: np.ndarray
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Return gradients (dW1, db1, dW2, db2) for the given output gradient.

        dy: (batch,)  dL/dy for each example in the batch.
        Sigmoid'(h_pre) = h * (1 - h), and that is what we backprop through.
        Note we backprop through `feat` (the augmented input), not raw x.
        The cross-product features are not differentiated through (they have
        no parameters).
        """
        dy_col = dy[:, None]                                    # (B, 1)
        dW2 = h.T @ dy_col                                       # (n_h, 1)
        db2 = dy_col.sum(axis=0)                                 # (1,)
        dh = dy_col @ self.W2.T                                  # (B, n_h)
        dpre = dh * h * (1.0 - h)                                # (B, n_h)
        dW1 = feat.T @ dpre                                      # (n_in, n_h)
        db1 = dpre.sum(axis=0)                                   # (n_h,)
        return dW1, db1, dW2, db2

    def params(self) -> list[np.ndarray]:
        return [self.W1, self.b1, self.W2, self.b2]


def build_two_module_net(strip_width: int, n_hidden: int,
                          seed: int = 0, init_scale: float = 0.5
                          ) -> tuple[Module, Module]:
    """Two parallel modules, independent weights, same architecture."""
    rng = np.random.default_rng(seed)
    n_in_raw = 2 * strip_width
    mod_a = Module(n_in_raw, strip_width, n_hidden, rng, init_scale=init_scale)
    mod_b = Module(n_in_raw, strip_width, n_hidden, rng, init_scale=init_scale)
    return mod_a, mod_b


# ---------------------------------------------------------------------------
# Imax loss
# ---------------------------------------------------------------------------

def imax_loss(y_a: np.ndarray, y_b: np.ndarray, eps: float = 1e-6
              ) -> tuple[float, np.ndarray, np.ndarray]:
    """Mutual information between scalar module outputs (Gaussian assumption).

    I(y_a; y_b) = 0.5 * log(var(y_a) + var(y_b))
                - 0.5 * log(var(y_a - y_b))

    Returns (-I, dy_a, dy_b)  -- we MINIMIZE -I, i.e., maximize I.
    """
    n = y_a.shape[0]
    mu_a = y_a.mean()
    mu_b = y_b.mean()
    diff = y_a - y_b
    mu_d = diff.mean()

    var_a = ((y_a - mu_a) ** 2).mean()
    var_b = ((y_b - mu_b) ** 2).mean()
    var_d = ((diff - mu_d) ** 2).mean()

    sum_var_ab = var_a + var_b + eps
    var_d = var_d + eps

    info = 0.5 * np.log(sum_var_ab) - 0.5 * np.log(var_d)
    loss = -info  # minimize -I

    # Gradients of -I w.r.t. y_a[i] and y_b[i]:
    #
    #   d(-I)/dy_a[i] = -0.5 * (1/(var_a+var_b)) * d(var_a)/dy_a[i]
    #                 + 0.5 * (1/var_d)         * d(var_d)/dy_a[i]
    #
    # var_a = (1/N) sum (y_a - mu_a)^2  =>  d/dy_a[i] = (2/N) * (y_a[i] - mu_a)
    # var_d = (1/N) sum (diff - mu_d)^2 =>  d/dy_a[i] = (2/N) * (diff[i] - mu_d)
    # var_d w.r.t. y_b[i]               =  -(2/N) * (diff[i] - mu_d)
    #
    # The two factors of 2 cancel against the 0.5 prefactors -- the
    # coefficient is 1/n, NOT 2/n (a finite-difference check fails by 2x
    # otherwise).
    inv_sum = 1.0 / sum_var_ab
    inv_d = 1.0 / var_d
    coef = 1.0 / n
    dy_a = coef * (-inv_sum * (y_a - mu_a) + inv_d * (diff - mu_d))
    dy_b = coef * (-inv_sum * (y_b - mu_b) - inv_d * (diff - mu_d))
    return float(loss), dy_a.astype(np.float32), dy_b.astype(np.float32)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def _sgd_step(params: list[np.ndarray], grads: list[np.ndarray],
              velocities: list[np.ndarray], lr: float, momentum: float,
              weight_decay: float) -> None:
    for p, g, v in zip(params, grads, velocities):
        v *= momentum
        v -= lr * (g + weight_decay * p)
        p += v


def train(mod_a: Module, mod_b: Module,
          n_epochs: int = 400,
          batch_size: int = 256,
          strip_width: int = 6,
          max_disparity: float = 3.0,
          continuous: bool = True,
          lr: float = 0.05,
          momentum: float = 0.9,
          weight_decay: float = 1e-5,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 5,
          verbose: bool = True) -> dict:
    """Train both modules jointly to maximize Imax(y_a; y_b)."""
    rng = np.random.default_rng(seed)

    params_a = mod_a.params()
    params_b = mod_b.params()
    vels_a = [np.zeros_like(p) for p in params_a]
    vels_b = [np.zeros_like(p) for p in params_b]

    history = {"epoch": [], "imax": [], "loss": [],
               "var_a": [], "var_b": [], "var_diff": [],
               "corr_ab": [], "corr_a_d": [], "corr_b_d": [],
               "corr_a_absd": [], "corr_b_absd": []}

    if verbose:
        print(f"# random-dot-stereograms (Imax)  strip_width={strip_width} "
              f"hidden={mod_a.W1.shape[1]} batch={batch_size}")

    for epoch in range(n_epochs):
        x_a, x_b, d = generate_batch(rng, batch_size,
                                     strip_width=strip_width,
                                     max_disparity=max_disparity,
                                     continuous=continuous)
        y_a, h_a, feat_a = mod_a.forward(x_a)
        y_b, h_b, feat_b = mod_b.forward(x_b)

        loss, dy_a, dy_b = imax_loss(y_a, y_b)
        info = -loss

        grads_a = mod_a.backward(feat_a, h_a, dy_a)
        grads_b = mod_b.backward(feat_b, h_b, dy_b)

        _sgd_step(params_a, list(grads_a), vels_a, lr, momentum, weight_decay)
        _sgd_step(params_b, list(grads_b), vels_b, lr, momentum, weight_decay)

        # Diagnostics
        var_a = float(y_a.var())
        var_b = float(y_b.var())
        var_d = float((y_a - y_b).var())
        abs_d = np.abs(d)
        corr_ab = float(np.corrcoef(y_a, y_b)[0, 1]) if var_a > 1e-9 and var_b > 1e-9 else 0.0
        corr_a_d = float(np.corrcoef(y_a, d)[0, 1]) if var_a > 1e-9 else 0.0
        corr_b_d = float(np.corrcoef(y_b, d)[0, 1]) if var_b > 1e-9 else 0.0
        corr_a_absd = float(np.corrcoef(y_a, abs_d)[0, 1]) if var_a > 1e-9 else 0.0
        corr_b_absd = float(np.corrcoef(y_b, abs_d)[0, 1]) if var_b > 1e-9 else 0.0

        history["epoch"].append(epoch + 1)
        history["imax"].append(float(info))
        history["loss"].append(float(loss))
        history["var_a"].append(var_a)
        history["var_b"].append(var_b)
        history["var_diff"].append(var_d)
        history["corr_ab"].append(corr_ab)
        history["corr_a_d"].append(corr_a_d)
        history["corr_b_d"].append(corr_b_d)
        history["corr_a_absd"].append(corr_a_absd)
        history["corr_b_absd"].append(corr_b_absd)

        if verbose and (epoch % 25 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  Imax={info:6.3f}  "
                  f"corr(y_a,y_b)={corr_ab:+.3f}  "
                  f"corr(y_a,|d|)={corr_a_absd:+.3f}  "
                  f"corr(y_b,|d|)={corr_b_absd:+.3f}  "
                  f"var_a={var_a:.3f}  var_d={var_d:.4f}")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, mod_a, mod_b, history)

    return history


def evaluate(mod_a: Module, mod_b: Module, rng: np.random.Generator,
             n_eval: int = 4096, strip_width: int = 6,
             max_disparity: float = 3.0, continuous: bool = True) -> dict:
    """Final-evaluation pass on a fresh batch."""
    x_a, x_b, d = generate_batch(rng, n_eval, strip_width=strip_width,
                                 max_disparity=max_disparity,
                                 continuous=continuous)
    y_a, _, _ = mod_a.forward(x_a)
    y_b, _, _ = mod_b.forward(x_b)
    loss, _, _ = imax_loss(y_a, y_b)
    info = -loss
    abs_d = np.abs(d)
    corr_ab = float(np.corrcoef(y_a, y_b)[0, 1])
    corr_a_d = float(np.corrcoef(y_a, d)[0, 1])
    corr_b_d = float(np.corrcoef(y_b, d)[0, 1])
    corr_a_absd = float(np.corrcoef(y_a, abs_d)[0, 1])
    corr_b_absd = float(np.corrcoef(y_b, abs_d)[0, 1])
    # Imax is sign-invariant: each module independently picks a sign for d.
    # We report both signed-d and |d| correlations so the reader sees what the
    # modules actually agreed on.
    return {
        "imax": float(info),
        "corr_ab": corr_ab,
        "corr_a_d": corr_a_d,
        "corr_b_d": corr_b_d,
        "corr_a_absd": corr_a_absd,
        "corr_b_absd": corr_b_absd,
        "n_eval": n_eval,
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs", type=int, default=600)
    p.add_argument("--strip-width", type=int, default=10)
    p.add_argument("--max-disparity", type=float, default=3.0)
    p.add_argument("--continuous", action="store_true", default=True,
                   help="Sub-pixel (continuous) disparity (default).")
    p.add_argument("--integer", dest="continuous", action="store_false",
                   help="Use integer disparity instead of sub-pixel.")
    p.add_argument("--n-hidden", type=int, default=48)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--weight-decay", type=float, default=1e-5)
    p.add_argument("--init-scale", type=float, default=0.5)
    args = p.parse_args()

    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")

    mod_a, mod_b = build_two_module_net(strip_width=args.strip_width,
                                         n_hidden=args.n_hidden,
                                         seed=args.seed,
                                         init_scale=args.init_scale)

    t0 = time.time()
    history = train(mod_a, mod_b,
                    n_epochs=args.n_epochs,
                    batch_size=args.batch_size,
                    strip_width=args.strip_width,
                    max_disparity=args.max_disparity,
                    continuous=args.continuous,
                    lr=args.lr,
                    momentum=args.momentum,
                    weight_decay=args.weight_decay,
                    seed=args.seed)
    elapsed = time.time() - t0

    eval_rng = np.random.default_rng(args.seed + 999_999)
    metrics = evaluate(mod_a, mod_b, eval_rng,
                       n_eval=4096,
                       strip_width=args.strip_width,
                       max_disparity=args.max_disparity,
                       continuous=args.continuous)

    print(f"\n=== final ===")
    print(f"Imax (eval, 4096 ex):      {metrics['imax']:.3f}")
    print(f"corr(y_a, y_b):            {metrics['corr_ab']:+.3f}")
    print(f"corr(y_a, signed d):       {metrics['corr_a_d']:+.3f}")
    print(f"corr(y_b, signed d):       {metrics['corr_b_d']:+.3f}")
    print(f"corr(y_a, |d|):            {metrics['corr_a_absd']:+.3f}")
    print(f"corr(y_b, |d|):            {metrics['corr_b_absd']:+.3f}")
    print(f"Wallclock:                 {elapsed:.2f}s")


if __name__ == "__main__":
    main()
