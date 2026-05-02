"""
Bars problem for RBM training -- the canonical Hinton 2000 / Foldiak 1990
sanity check for contrastive divergence.

Problem:
  Generate 4x4 binary images by independently activating each of 8 possible
  bars (4 horizontal rows + 4 vertical columns) with a small probability.
  Each image is the OR of its activated bars. After CD-1 training each
  hidden unit should specialize to a single bar -- the receptive-field
  visualization is the headline result.

Source:
  Hinton, G. E. (2000/2002). "Training products of experts by minimizing
  contrastive divergence." Neural Computation 14(8). The bars problem
  follows Foldiak (1990), "Forming sparse representations by local
  anti-Hebbian learning."

Architecture:
  16 visible binary units (4x4 image) <-> n_hidden binary units. Bipartite
  RBM. n_hidden = 8 is canonical (one per bar); n_hidden = 16 demonstrates
  what extra capacity does (duplicate detectors + dead units).

Learning:
  CD-1 (Hinton 2002).

      Delta w_ij  ~  <v_i h_j>_data  -  <v_i h_j>_recon

  where the reconstruction is one Gibbs step from the data: data -> h -> v.
"""

from __future__ import annotations
import argparse
import time
import numpy as np


# ----------------------------------------------------------------------
# Data
# ----------------------------------------------------------------------

def make_bar_templates(h: int = 4, w: int = 4) -> np.ndarray:
    """Return (n_bars, h, w) array of single-bar templates.

    First `h` are horizontal bars (one row each), next `w` are vertical
    bars (one column each).
    """
    n_bars = h + w
    templates = np.zeros((n_bars, h, w), dtype=np.float32)
    for i in range(h):
        templates[i, i, :] = 1.0
    for j in range(w):
        templates[h + j, :, j] = 1.0
    return templates


def generate_bars(n_samples: int,
                  h: int = 4,
                  w: int = 4,
                  p_bar: float = 0.125,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Generate `n_samples` flat (h*w,) bar images.

    Each of the (h + w) bars is independently activated with probability
    `p_bar`. Pixel value = OR of activated bars (so the result is binary
    in {0.0, 1.0}). Default p_bar = 1 / (h + w) gives ~1 bar per image
    on average.
    """
    if rng is None:
        rng = np.random.default_rng()
    templates = make_bar_templates(h, w)              # (n_bars, h, w)
    n_bars = templates.shape[0]
    activations = (rng.random((n_samples, n_bars)) < p_bar).astype(np.float32)
    # OR over bars: clip the sum to {0, 1}.
    images = np.einsum("sb,bhw->shw", activations, templates)
    images = np.minimum(images, 1.0)
    return images.reshape(n_samples, h * w)


# ----------------------------------------------------------------------
# RBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


class BarsRBM:
    """Bipartite binary RBM, trained with CD-1.

    Visible: h*w pixels. Hidden: n_hidden binary feature detectors. Each
    hidden unit's incoming weight slice W[:, j] reshaped as (h, w) is the
    "receptive field" plotted at the end.
    """

    def __init__(self,
                 n_visible: int = 16,
                 n_hidden: int = 8,
                 init_scale: float = 0.01,
                 seed: int = 0,
                 image_shape: tuple[int, int] = (4, 4)):
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.image_shape = image_shape
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


# ----------------------------------------------------------------------
# CD-1 step (module-level so the spec's `cd1_step()` symbol exists)
# ----------------------------------------------------------------------

def cd1_step(rbm: BarsRBM,
             batch: np.ndarray,
             lr: float = 0.1,
             weight_decay: float = 0.0,
             sparsity_target: float | None = None,
             sparsity_cost: float = 0.0,
             momentum: float = 0.0,
             velocity: dict | None = None
             ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One CD-1 update on `rbm` from `batch`.

    Returns the gradient pieces (dW, db_v, db_h). `rbm` is mutated in place.

    Optional knobs (off by default to keep the basic call simple):
      weight_decay      L2 penalty on W, applied as -wd * W
      sparsity_target   if set, push mean hidden prob toward this value
                        (Lee/Ekanadham/Ng 2008-style sparsity)
      sparsity_cost     coefficient on the sparsity gradient
      momentum          if > 0, use velocity dict to accumulate updates
      velocity          mutable dict with keys 'W', 'b_v', 'b_h' for momentum
    """
    n = batch.shape[0]

    # positive phase: data -> hidden probs. We use probs for the
    # gradient and binary samples for the negative phase, per the
    # Hinton 2010 practical guide.
    h_prob_pos, h_sample_pos = rbm.sample_h_given_v(batch)

    # negative phase (one Gibbs step): h_sample -> v_prob -> h_prob.
    # Using probabilities for the visible reconstruction reduces noise
    # without changing the gradient direction in expectation.
    v_recon = rbm.visible_prob(h_sample_pos)
    h_prob_neg = rbm.hidden_prob(v_recon)

    dW = (batch.T @ h_prob_pos - v_recon.T @ h_prob_neg) / n
    db_v = (batch - v_recon).mean(axis=0)
    db_h = (h_prob_pos - h_prob_neg).mean(axis=0)

    if weight_decay > 0:
        dW = dW - weight_decay * rbm.W

    if sparsity_target is not None and sparsity_cost > 0:
        # Penalize departure of mean hidden activation from the target
        # rate. Gradient w.r.t. b_h is (target - mean_h_prob); applied
        # per-unit. This is the cheap form used in most practical
        # implementations.
        mean_h = h_prob_pos.mean(axis=0)
        db_h = db_h + sparsity_cost * (sparsity_target - mean_h)

    if momentum > 0 and velocity is not None:
        velocity["W"] = momentum * velocity["W"] + lr * dW
        velocity["b_v"] = momentum * velocity["b_v"] + lr * db_v
        velocity["b_h"] = momentum * velocity["b_h"] + lr * db_h
        rbm.W += velocity["W"]
        rbm.b_v += velocity["b_v"]
        rbm.b_h += velocity["b_h"]
    else:
        rbm.W += lr * dW
        rbm.b_v += lr * db_v
        rbm.b_h += lr * db_h
    return dW, db_v, db_h


# ----------------------------------------------------------------------
# Filter inspection
# ----------------------------------------------------------------------

def per_unit_bar_purity(rbm: BarsRBM) -> dict:
    """For each hidden unit, find which bar template best matches its
    receptive field, and return a "purity" score.

    Purity is computed as:

        purity_j = max_b cos(W[:, j], template_b - mean(template_b))
                       , clipped to [0, 1]

    where the template is mean-centered to remove the d.c. component
    (a hidden unit that just outputs the data mean would otherwise score
    nonzero against every bar). Higher = cleaner single-bar detector.

    Returns a dict with:
      best_bar       (n_hidden,) int -- index of best-matching bar (0..7)
      purity         (n_hidden,) float -- cosine sim with that bar
      bars_covered   int -- number of distinct bars that some unit detects
                            with purity >= 0.5
      n_bars         int -- total bars (h + w)
      mean_purity    float -- mean over hidden units
    """
    h, w = rbm.image_shape
    templates = make_bar_templates(h, w).reshape(h + w, h * w)  # (n_bars, n_v)
    # mean-center each template so a constant filter scores ~0
    centered = templates - templates.mean(axis=1, keepdims=True)
    centered_norm = centered / (np.linalg.norm(centered, axis=1, keepdims=True)
                                + 1e-12)

    W = rbm.W.T  # (n_hidden, n_visible)
    W_centered = W - W.mean(axis=1, keepdims=True)
    W_norm = W_centered / (np.linalg.norm(W_centered, axis=1, keepdims=True)
                           + 1e-12)

    sims = W_norm @ centered_norm.T  # (n_hidden, n_bars)
    best_bar = sims.argmax(axis=1)
    purity = sims[np.arange(W.shape[0]), best_bar]
    purity = np.clip(purity, 0.0, 1.0)

    covered = set()
    for j in range(W.shape[0]):
        if purity[j] >= 0.5:
            covered.add(int(best_bar[j]))
    return {
        "best_bar": best_bar.astype(int),
        "purity": purity.astype(float),
        "bars_covered": len(covered),
        "n_bars": h + w,
        "mean_purity": float(purity.mean()),
    }


def visualize_filters(rbm: BarsRBM) -> str:
    """ASCII receptive-field plot. Each hidden unit is shown as its
    weight slice W[:, j] reshaped to (h, w), with `+` for above-mean,
    `-` for below-mean, ` ` for near-zero. Useful for sanity checks
    in CLI runs without matplotlib.
    """
    h, w = rbm.image_shape
    lines = []
    score = per_unit_bar_purity(rbm)
    bar_names = ([f"H{i}" for i in range(h)] + [f"V{j}" for j in range(w)])
    for j in range(rbm.n_hidden):
        rf = rbm.W[:, j].reshape(h, w)
        thresh = 0.5 * np.abs(rf).max()
        lines.append(f"hidden unit {j:2d}  best={bar_names[score['best_bar'][j]]}"
                     f"  purity={score['purity'][j]:.2f}")
        for r in range(h):
            row = []
            for c in range(w):
                v = rf[r, c]
                if v > thresh:
                    row.append("+")
                elif v < -thresh:
                    row.append("-")
                else:
                    row.append(".")
            lines.append("  " + " ".join(row))
        lines.append("")
    return "\n".join(lines)


# ----------------------------------------------------------------------
# Reconstruction error (sanity metric over training)
# ----------------------------------------------------------------------

def mean_recon_error(rbm: BarsRBM, data: np.ndarray) -> float:
    """One-step Gibbs reconstruction MSE."""
    h_prob = rbm.hidden_prob(data)
    v_recon = rbm.visible_prob(h_prob)
    return float(((data - v_recon) ** 2).mean())


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_epochs: int = 300,
          n_hidden: int = 8,
          n_train: int = 2000,
          batch_size: int = 20,
          lr: float = 0.1,
          weight_decay: float = 1e-4,
          momentum: float = 0.5,
          sparsity_target: float | None = None,
          sparsity_cost: float = 0.1,
          init_scale: float = 0.01,
          p_bar: float = 0.125,
          h: int = 4,
          w: int = 4,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 5,
          verbose: bool = True):
    """Train a BarsRBM with CD-1.

    Returns (rbm, history) where history has per-epoch lists for
    'epoch', 'recon_error', 'mean_purity', 'bars_covered'.

    `sparsity_target` defaults to `1 / n_hidden` if left as None and
    `sparsity_cost > 0`. Set `sparsity_cost = 0` to disable.
    """
    seed_seq = np.random.SeedSequence(seed)
    data_seed, model_seed = seed_seq.spawn(2)
    data_rng = np.random.default_rng(data_seed)

    data = generate_bars(n_train, h=h, w=w, p_bar=p_bar, rng=data_rng)
    rbm = BarsRBM(n_visible=h * w, n_hidden=n_hidden,
                  init_scale=init_scale,
                  seed=int(model_seed.generate_state(1)[0]),
                  image_shape=(h, w))

    # Initialize hidden bias so initial mean hidden activation matches the
    # target sparsity rate -- this stops dead/saturated units at startup.
    target = (sparsity_target if sparsity_target is not None
              else 1.0 / max(n_hidden, 1))
    if 0 < target < 1:
        rbm.b_h[:] = float(np.log(target / (1.0 - target)))
    # Initialize visible bias so initial reconstruction matches the data
    # mean -- this is the standard practical-guide trick.
    data_mean = np.clip(data.mean(axis=0), 1e-3, 1 - 1e-3)
    rbm.b_v[:] = np.log(data_mean / (1.0 - data_mean)).astype(np.float32)

    velocity = {"W": np.zeros_like(rbm.W),
                "b_v": np.zeros_like(rbm.b_v),
                "b_h": np.zeros_like(rbm.b_h)}

    history = {"epoch": [], "recon_error": [], "mean_purity": [],
               "bars_covered": []}

    if verbose:
        print(f"# bars-rbm: {h}x{w} images, {n_hidden} hidden, "
              f"{n_train} samples, batch={batch_size}, lr={lr}, "
              f"p_bar={p_bar}, momentum={momentum}, wd={weight_decay}, "
              f"sparsity_cost={sparsity_cost} target={target:.3f}")

    for epoch in range(n_epochs):
        t0 = time.time()
        perm = rbm.rng.permutation(n_train)
        for start in range(0, n_train, batch_size):
            idx = perm[start:start + batch_size]
            cd1_step(rbm, data[idx], lr=lr,
                     weight_decay=weight_decay,
                     sparsity_target=target if sparsity_cost > 0 else None,
                     sparsity_cost=sparsity_cost,
                     momentum=momentum,
                     velocity=velocity)

        recon = mean_recon_error(rbm, data)
        score = per_unit_bar_purity(rbm)
        history["epoch"].append(epoch + 1)
        history["recon_error"].append(recon)
        history["mean_purity"].append(score["mean_purity"])
        history["bars_covered"].append(score["bars_covered"])

        if verbose and (epoch % 20 == 0 or epoch == n_epochs - 1):
            print(f"epoch {epoch+1:4d}  recon_mse={recon:.4f}  "
                  f"mean_purity={score['mean_purity']:.3f}  "
                  f"bars_covered={score['bars_covered']}/{score['n_bars']}  "
                  f"({time.time()-t0:.3f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, rbm, history)

    return rbm, history


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Bars-RBM CD-1 training")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-hidden", type=int, default=8)
    p.add_argument("--n-epochs", type=int, default=200)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--batch-size", type=int, default=20)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--sparsity-cost", type=float, default=0.1,
                   help="0 disables sparsity penalty on hidden bias")
    p.add_argument("--p-bar", type=float, default=0.125)
    p.add_argument("--init-scale", type=float, default=0.01)
    p.add_argument("--h", type=int, default=4)
    p.add_argument("--w", type=int, default=4)
    p.add_argument("--ascii", action="store_true",
                   help="Print ASCII receptive fields after training")
    args = p.parse_args()

    t_start = time.time()
    rbm, history = train(n_epochs=args.n_epochs,
                         n_hidden=args.n_hidden,
                         n_train=args.n_train,
                         batch_size=args.batch_size,
                         lr=args.lr,
                         momentum=args.momentum,
                         weight_decay=args.weight_decay,
                         sparsity_cost=args.sparsity_cost,
                         init_scale=args.init_scale,
                         p_bar=args.p_bar,
                         h=args.h,
                         w=args.w,
                         seed=args.seed)
    elapsed = time.time() - t_start

    score = per_unit_bar_purity(rbm)
    print()
    print(f"Training time: {elapsed:.2f}s")
    print(f"Final reconstruction MSE: {history['recon_error'][-1]:.4f}")
    print(f"Mean per-unit purity: {score['mean_purity']:.3f}")
    print(f"Distinct bars detected (purity >= 0.5): "
          f"{score['bars_covered']}/{score['n_bars']}")
    bar_names = ([f"H{i}" for i in range(args.h)]
                 + [f"V{j}" for j in range(args.w)])
    print("\nPer-unit best-bar match:")
    for j in range(rbm.n_hidden):
        print(f"  unit {j:2d}: best={bar_names[score['best_bar'][j]]:>3}  "
              f"purity={score['purity'][j]:.3f}")

    if args.ascii:
        print("\nReceptive fields (ASCII):")
        print(visualize_filters(rbm))
