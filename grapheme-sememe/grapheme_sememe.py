"""
Grapheme-sememe synthetic word reading + lesion / relearning experiments.

Source:
    Hinton & Sejnowski (1986), "Learning and relearning in Boltzmann machines",
    in Rumelhart & McClelland (eds.), *Parallel Distributed Processing*, Vol. 1,
    Chapter 7.

Problem:
    20 random "words" mapping graphemes (3 letter positions x 10 letters,
    represented as 30 binary input units, with one-hot per position) to
    sememes (30 binary output units representing semantic micro-features).
    The 30->20->30 net learns the mapping; the experiment asks what happens
    when the trained net is damaged ("lesioned") and partially retrained.

Demonstrates:
    Distributed representations are damage-resistant. After randomly zeroing
    a fraction of weights and retraining on only 18 of the 20 associations,
    the network's accuracy on the 2 held-out associations recovers
    substantially — the famous "spontaneous recovery" result.

Algorithm:
    The 1986 paper used Boltzmann learning. We use backprop with momentum
    (Rumelhart, Hinton & Williams 1986). The spec for this stub explicitly
    permits either; backprop is simpler for a deterministic 30->20->30 net
    and the spontaneous-recovery effect is a property of distributed
    representations, not of the specific learning rule. See "Deviations
    from the original" in README.md.

Protocol (4-stage):
    1. Train on all 20 grapheme->sememe associations.
    2. Lesion: randomly zero a fraction of W1 and W2 weights.
    3. Relearn-subset: retrain on 18 of 20 patterns (the other 2 held out).
    4. Test held-out 2: report accuracy without ever showing them again.

This file: model + 4-stage protocol + CLI.
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

def generate_mapping(n_words: int = 20,
                     n_letters: int = 10,
                     n_positions: int = 3,
                     n_sememes: int = 30,
                     n_semantic_prototypes: int = 4,
                     prototypes_per_word: int = 2,
                     prototype_density: float = 0.35,
                     sememe_noise: float = 0.05,
                     seed: int = 0,
                     ) -> tuple[np.ndarray, np.ndarray]:
    """Generate `n_words` random grapheme->sememe associations with shared
    semantic micro-features.

    Returns
    -------
    graphemes : (n_words, n_positions * n_letters) float64, in {0, 1}
        Each row has exactly `n_positions` ones (one-hot within each
        letter-position block of width `n_letters`).
    sememes : (n_words, n_sememes) float64, in {0, 1}
        Each row is the union (logical OR) of `prototypes_per_word`
        randomly selected micro-feature prototypes drawn from a pool of
        `n_semantic_prototypes`. Words that share a prototype share
        ~`n_sememes / n_semantic_prototypes` sememe bits.

    Why prototypes: Hinton & Sejnowski's 1986 result relies on shared
    semantic micro-features between words (e.g. "BAT" and "RAT" both have
    the "ANIMAL" feature, "MAT" and "RAT" both have "OBJECT-IN-ROOM").
    Independent Bernoulli sememes have no such shared structure, so
    retraining 18 patterns gives nothing to transfer to the 2 held-out
    ones. With a small pool of prototypes, the held-out 2 inevitably
    share several prototypes with the 18 that ARE retrained, and the
    distributed representation learned for the trained set generalizes.
    See `docs/deviations` in the README.
    """
    rng = np.random.default_rng(seed)
    n_grapheme = n_positions * n_letters

    # Graphemes: one-hot per position. Sample a letter-index per position
    # for each word, ensuring no two words share the same (pos -> letter)
    # tuple in all positions (rejection sampling on duplicates).
    used: set[tuple[int, ...]] = set()
    grapheme_codes: list[tuple[int, ...]] = []
    for _ in range(n_words):
        for _attempt in range(1000):
            code = tuple(int(rng.integers(0, n_letters))
                         for _ in range(n_positions))
            if code not in used:
                used.add(code)
                grapheme_codes.append(code)
                break
        else:
            raise RuntimeError("could not draw a unique grapheme")

    graphemes = np.zeros((n_words, n_grapheme), dtype=np.float64)
    for w, code in enumerate(grapheme_codes):
        for pos, letter in enumerate(code):
            graphemes[w, pos * n_letters + letter] = 1.0

    # Build a pool of `n_semantic_prototypes` random binary "category"
    # vectors over the n_sememes micro-features. Each prototype is dense
    # (~50% bits on) so multiple prototypes overlap and combine
    # non-trivially; this is the "semantic micro-feature" structure.
    prototypes = (rng.random((n_semantic_prototypes, n_sememes))
                  < prototype_density).astype(np.float64)

    # Each word picks a random subset of `prototypes_per_word` prototypes
    # and ORs them, then flips each bit independently with probability
    # `sememe_noise` to break degeneracy when the prototype pool is small.
    # Reject duplicates / all-zero / all-one.
    sememes = np.zeros((n_words, n_sememes), dtype=np.float64)
    seen: set[tuple[int, ...]] = set()
    for w in range(n_words):
        for _attempt in range(1000):
            picks = rng.choice(n_semantic_prototypes,
                               size=prototypes_per_word, replace=False)
            s = np.clip(prototypes[picks].sum(axis=0), 0, 1)
            if sememe_noise > 0:
                flip = rng.random(n_sememes) < sememe_noise
                s = np.where(flip, 1 - s, s)
            tup = tuple(int(x) for x in s)
            if (1 <= int(s.sum()) <= n_sememes - 1) and tup not in seen:
                seen.add(tup)
                sememes[w] = s
                break
        else:
            raise RuntimeError("could not draw a unique sememe; try a "
                               "larger prototype pool, more noise, or "
                               "fewer words")

    return graphemes, sememes


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


class GraphemeSememeMLP:
    """30 -> 20 -> 30 sigmoid MLP with per-bit Bernoulli cross-entropy loss.

    Trained with full-batch backprop + momentum. State:
      W1, b1 : hidden weights & biases  (n_hidden, n_grapheme), (n_hidden,)
      W2, b2 : output weights & biases  (n_sememe, n_hidden),    (n_sememe,)
    """

    def __init__(self,
                 n_grapheme: int = 30,
                 n_hidden: int = 20,
                 n_sememe: int = 30,
                 init_scale: float = 0.5,
                 seed: int = 0):
        self.n_grapheme = n_grapheme
        self.n_hidden = n_hidden
        self.n_sememe = n_sememe
        self.rng = np.random.default_rng(seed)
        # Symmetric uniform init in [-init_scale, +init_scale]; for
        # small n this matches the 1986-era recipe better than Xavier.
        self.W1 = init_scale * (2 * self.rng.random((n_hidden, n_grapheme)) - 1)
        self.b1 = init_scale * (2 * self.rng.random(n_hidden) - 1)
        self.W2 = init_scale * (2 * self.rng.random((n_sememe, n_hidden)) - 1)
        self.b2 = init_scale * (2 * self.rng.random(n_sememe) - 1)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """X : (n, n_grapheme). Returns (h, o), both (n, .)."""
        h = sigmoid(X @ self.W1.T + self.b1)
        o = sigmoid(h @ self.W2.T + self.b2)
        return h, o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, o = self.forward(X)
        return o

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size


def backprop_grads(model: GraphemeSememeMLP,
                   X: np.ndarray, Y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Gradients of mean-per-pattern Bernoulli cross-entropy loss.

    With a sigmoid output and BCE, the output-layer error is the simple
    `(o - y)` form (the sigmoid' factor cancels with the BCE derivative).
    """
    n = X.shape[0]
    h, o = model.forward(X)
    delta_o = (o - Y)                                  # (n, n_sememe)
    dW2 = delta_o.T @ h / n                            # (n_sememe, n_hidden)
    db2 = delta_o.mean(axis=0)                         # (n_sememe,)
    delta_h = (delta_o @ model.W2) * h * (1.0 - h)     # (n, n_hidden)
    dW1 = delta_h.T @ X / n                            # (n_hidden, n_grapheme)
    db1 = delta_h.mean(axis=0)                         # (n_hidden,)
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_bce(model: GraphemeSememeMLP, X: np.ndarray, Y: np.ndarray) -> float:
    eps = 1e-12
    o = model.predict(X)
    o = np.clip(o, eps, 1.0 - eps)
    return float(-np.mean(Y * np.log(o) + (1 - Y) * np.log(1 - o)))


def accuracy_bitwise(model: GraphemeSememeMLP,
                     X: np.ndarray, Y: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Fraction of output BITS correct (averaged across patterns and bits)."""
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(pred == Y))


def accuracy_pattern(model: GraphemeSememeMLP,
                     X: np.ndarray, Y: np.ndarray,
                     threshold: float = 0.5) -> float:
    """Fraction of patterns where ALL output bits match."""
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(np.all(pred == Y, axis=1)))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train_step(model: GraphemeSememeMLP,
               X: np.ndarray, Y: np.ndarray,
               velocities: dict[str, np.ndarray],
               lr: float, momentum: float,
               weight_decay: float = 0.0,
               weight_mask: dict[str, np.ndarray] | None = None,
               ) -> dict[str, np.ndarray]:
    """One full-batch gradient step. Mutates model & velocities in place.

    `weight_mask` : optional dict with same keys as `velocities` whose values
    are 0/1 masks. After each update we re-zero the masked-out weights so
    that lesioned weights stay lesioned during relearning. This matches
    the "permanent damage" interpretation in Hinton & Sejnowski 1986.

    `weight_decay` : L2 weight-decay coefficient applied only to W1/W2.
    """
    grads = backprop_grads(model, X, Y)
    if weight_decay != 0.0:
        grads["W1"] = grads["W1"] + weight_decay * model.W1
        grads["W2"] = grads["W2"] + weight_decay * model.W2
    for k, dG in grads.items():
        velocities[k] = momentum * velocities[k] - lr * dG
    model.W1 += velocities["W1"]
    model.b1 += velocities["b1"]
    model.W2 += velocities["W2"]
    model.b2 += velocities["b2"]
    if weight_mask is not None:
        if "W1" in weight_mask:
            model.W1 *= weight_mask["W1"]
        if "W2" in weight_mask:
            model.W2 *= weight_mask["W2"]
    return velocities


def train(model: GraphemeSememeMLP,
          X: np.ndarray, Y: np.ndarray,
          n_cycles: int = 2000,
          lr: float = 0.5,
          momentum: float = 0.9,
          weight_decay: float = 0.0,
          weight_mask: dict[str, np.ndarray] | None = None,
          history: dict | None = None,
          history_eval: dict | None = None,
          history_phase: str = "",
          history_every: int = 1,
          history_offset: int = 0,
          early_stop_acc: float | None = None,
          verbose: bool = False,
          ) -> dict:
    """Train `model` on the patterns (X, Y) with full-batch backprop + momentum.

    Logs into `history` if given. `history_eval` may contain extra
    (name, (X, Y)) pairs to track, e.g. {"trained": (X18, Y18),
    "held_out": (X2, Y2)}, so the same call records accuracies on
    several disjoint subsets per epoch.
    """
    velocities = {k: np.zeros_like(v)
                  for k, v in [("W1", model.W1), ("b1", model.b1),
                                ("W2", model.W2), ("b2", model.b2)]}
    if history is None:
        history = {"phase": [], "epoch": [], "loss": [], "acc_bit": [],
                   "acc_pattern": [], "weight_norm": []}
        for k in (history_eval or {}):
            history[f"acc_pattern_{k}"] = []
            history[f"acc_bit_{k}"] = []

    for epoch in range(n_cycles):
        train_step(model, X, Y, velocities, lr, momentum,
                   weight_decay=weight_decay, weight_mask=weight_mask)

        if (epoch % history_every == 0) or (epoch == n_cycles - 1):
            history["phase"].append(history_phase)
            history["epoch"].append(history_offset + epoch + 1)
            history["loss"].append(loss_bce(model, X, Y))
            history["acc_bit"].append(accuracy_bitwise(model, X, Y))
            history["acc_pattern"].append(accuracy_pattern(model, X, Y))
            history["weight_norm"].append(
                float(np.linalg.norm(model.W1)) +
                float(np.linalg.norm(model.W2)))
            if history_eval is not None:
                for name, (Xe, Ye) in history_eval.items():
                    history[f"acc_pattern_{name}"].append(
                        accuracy_pattern(model, Xe, Ye))
                    history[f"acc_bit_{name}"].append(
                        accuracy_bitwise(model, Xe, Ye))

        if verbose and (epoch % 200 == 0 or epoch == n_cycles - 1):
            print(f"  [{history_phase or 'train'}] "
                  f"epoch {epoch+1:5d}  loss={history['loss'][-1]:.4f}  "
                  f"acc_bit={history['acc_bit'][-1]*100:5.1f}%  "
                  f"acc_pat={history['acc_pattern'][-1]*100:5.1f}%")

        if (early_stop_acc is not None
                and history["acc_pattern"]
                and history["acc_pattern"][-1] >= early_stop_acc):
            break

    return history


# ----------------------------------------------------------------------
# Lesion + relearn (the headline experiment)
# ----------------------------------------------------------------------

def lesion(model: GraphemeSememeMLP,
           fraction: float,
           layers: tuple[str, ...] = ("W1", "W2"),
           seed: int = 0,
           ) -> dict[str, np.ndarray]:
    """Randomly zero `fraction` of the weights in each layer in `layers`.

    Returns the survival mask used (1 = weight kept, 0 = weight zeroed),
    so the caller can keep the same connections lesioned during relearning.
    Biases are NOT lesioned (they are not really "synapses" in the 1986
    interpretation).
    """
    if not 0.0 <= fraction <= 1.0:
        raise ValueError(f"lesion fraction must be in [0, 1], got {fraction}")
    rng = np.random.default_rng(seed)
    mask: dict[str, np.ndarray] = {}
    for layer in layers:
        W = getattr(model, layer)
        keep = (rng.random(W.shape) >= fraction).astype(W.dtype)
        W *= keep
        mask[layer] = keep
    return mask


def relearn_subset(model: GraphemeSememeMLP,
                   X: np.ndarray, Y: np.ndarray,
                   indices: list[int],
                   n_cycles: int,
                   lr: float = 0.5,
                   momentum: float = 0.9,
                   weight_decay: float = 0.0,
                   weight_mask: dict[str, np.ndarray] | None = None,
                   history: dict | None = None,
                   history_eval: dict | None = None,
                   history_offset: int = 0,
                   ) -> dict:
    """Retrain `model` on the patterns indexed by `indices` only.

    The lesion mask (if any) is preserved across updates so the damaged
    connections stay zero — relearning routes around the damage rather
    than repairing it. The 2 held-out patterns are NEVER shown during
    `relearn_subset`; their post-relearn accuracy is the spontaneous-
    recovery headline.
    """
    Xs, Ys = X[indices], Y[indices]
    return train(model, Xs, Ys,
                 n_cycles=n_cycles, lr=lr, momentum=momentum,
                 weight_decay=weight_decay,
                 weight_mask=weight_mask,
                 history=history,
                 history_eval=history_eval,
                 history_phase="relearn",
                 history_offset=history_offset)


# ----------------------------------------------------------------------
# 4-stage protocol
# ----------------------------------------------------------------------

def run_protocol(seed: int = 0,
                 n_words: int = 20,
                 n_held_out: int = 2,
                 lesion_fraction: float = 0.5,
                 n_train_cycles: int = 1500,
                 n_relearn_cycles: int = 50,
                 lr: float = 0.3,
                 momentum: float = 0.5,
                 weight_decay: float = 1e-3,
                 init_scale: float = 0.5,
                 n_semantic_prototypes: int = 4,
                 prototypes_per_word: int = 2,
                 history_every: int = 1,
                 verbose: bool = False,
                 ) -> dict:
    """Full 4-stage protocol; returns a dict with model, history, summary."""
    # ---- stage 0: data ----
    X, Y = generate_mapping(n_words=n_words,
                            n_semantic_prototypes=n_semantic_prototypes,
                            prototypes_per_word=prototypes_per_word,
                            seed=seed)
    held_out = list(range(n_words - n_held_out, n_words))
    trained_idx = [i for i in range(n_words) if i not in held_out]
    X_train, Y_train = X[trained_idx], Y[trained_idx]
    X_held, Y_held = X[held_out], Y[held_out]

    history_eval = {"trained": (X_train, Y_train),
                    "held_out": (X_held, Y_held)}

    # ---- stage 1: train on all 20 ----
    model = GraphemeSememeMLP(seed=seed, init_scale=init_scale)
    if verbose:
        print(f"# stage 1: train on all {n_words} associations "
              f"({n_train_cycles} cycles)")
    history = train(model, X, Y,
                    n_cycles=n_train_cycles, lr=lr, momentum=momentum,
                    weight_decay=weight_decay,
                    history_eval=history_eval,
                    history_phase="train",
                    history_every=history_every,
                    verbose=verbose)

    pre_lesion_acc_all = accuracy_pattern(model, X, Y)
    pre_lesion_acc_held = accuracy_pattern(model, X_held, Y_held)
    pre_lesion_bit_held = accuracy_bitwise(model, X_held, Y_held)

    # ---- stage 2: lesion ----
    if verbose:
        print(f"# stage 2: lesion {lesion_fraction*100:.0f}% of W1+W2")
    mask = lesion(model, fraction=lesion_fraction, seed=seed + 1)

    post_lesion_acc_all = accuracy_pattern(model, X, Y)
    post_lesion_acc_trained = accuracy_pattern(model, X_train, Y_train)
    post_lesion_acc_held = accuracy_pattern(model, X_held, Y_held)
    post_lesion_bit_held = accuracy_bitwise(model, X_held, Y_held)

    # log a single lesion datapoint into the history so plots can show it
    last_epoch = history["epoch"][-1] if history["epoch"] else 0
    history["phase"].append("lesion")
    history["epoch"].append(last_epoch)
    history["loss"].append(loss_bce(model, X, Y))
    history["acc_bit"].append(accuracy_bitwise(model, X, Y))
    history["acc_pattern"].append(post_lesion_acc_all)
    history["weight_norm"].append(
        float(np.linalg.norm(model.W1)) + float(np.linalg.norm(model.W2)))
    history["acc_pattern_trained"].append(post_lesion_acc_trained)
    history["acc_bit_trained"].append(accuracy_bitwise(model, X_train, Y_train))
    history["acc_pattern_held_out"].append(post_lesion_acc_held)
    history["acc_bit_held_out"].append(post_lesion_bit_held)

    # ---- stage 3: relearn on 18 of 20 ----
    if verbose:
        print(f"# stage 3: relearn on {len(trained_idx)} patterns "
              f"({n_relearn_cycles} cycles); {len(held_out)} held out")
    relearn_subset(model, X, Y, indices=trained_idx,
                   n_cycles=n_relearn_cycles,
                   lr=lr, momentum=momentum,
                   weight_decay=weight_decay,
                   weight_mask=mask,
                   history=history,
                   history_eval=history_eval,
                   history_offset=last_epoch)

    # ---- stage 4: report ----
    post_relearn_acc_trained = accuracy_pattern(model, X_train, Y_train)
    post_relearn_acc_held = accuracy_pattern(model, X_held, Y_held)
    post_relearn_bit_trained = accuracy_bitwise(model, X_train, Y_train)
    post_relearn_bit_held = accuracy_bitwise(model, X_held, Y_held)

    summary = {
        "seed": seed,
        "n_words": n_words,
        "n_held_out": n_held_out,
        "lesion_fraction": lesion_fraction,
        "n_train_cycles": n_train_cycles,
        "n_relearn_cycles": n_relearn_cycles,
        "lr": lr, "momentum": momentum,
        "init_scale": init_scale,
        "n_params": model.n_params(),
        "n_semantic_prototypes": n_semantic_prototypes,
        "prototypes_per_word": prototypes_per_word,
        # pattern accuracy (all bits correct)
        "pre_lesion_pattern_all": pre_lesion_acc_all,
        "pre_lesion_pattern_held_out": pre_lesion_acc_held,
        "post_lesion_pattern_all": post_lesion_acc_all,
        "post_lesion_pattern_trained": post_lesion_acc_trained,
        "post_lesion_pattern_held_out": post_lesion_acc_held,
        "post_relearn_pattern_trained": post_relearn_acc_trained,
        "post_relearn_pattern_held_out": post_relearn_acc_held,
        # bit accuracy (per-bit fraction correct)
        "pre_lesion_bit_held_out": pre_lesion_bit_held,
        "post_lesion_bit_held_out": post_lesion_bit_held,
        "post_relearn_bit_trained": post_relearn_bit_trained,
        "post_relearn_bit_held_out": post_relearn_bit_held,
        # spontaneous recovery: how much held-out bit accuracy was
        # restored without ever showing the held-out patterns again
        "spontaneous_recovery_bits": (
            post_relearn_bit_held - post_lesion_bit_held),
    }
    return {
        "model": model,
        "history": history,
        "summary": summary,
        "data": {
            "X": X, "Y": Y,
            "trained_idx": trained_idx,
            "held_out_idx": held_out,
        },
        "mask": mask,
    }


# ----------------------------------------------------------------------
# Multi-seed sweep
# ----------------------------------------------------------------------

def sweep(n_seeds: int = 10,
          lesion_fraction: float = 0.5,
          n_train_cycles: int = 1500,
          n_relearn_cycles: int = 50,
          verbose: bool = False,
          ) -> dict:
    """Repeat the protocol across `n_seeds` and aggregate the headline numbers."""
    rows = []
    for s in range(n_seeds):
        out = run_protocol(seed=s,
                           lesion_fraction=lesion_fraction,
                           n_train_cycles=n_train_cycles,
                           n_relearn_cycles=n_relearn_cycles,
                           history_every=10**6,  # don't grow history
                           verbose=False)
        rows.append(out["summary"])
        if verbose:
            print(f"  seed={s}  pre={rows[-1]['pre_lesion_pattern_all']*100:5.1f}%  "
                  f"post-lesion-held={rows[-1]['post_lesion_bit_held_out']*100:5.1f}%(bit)  "
                  f"post-relearn-held={rows[-1]['post_relearn_bit_held_out']*100:5.1f}%(bit)  "
                  f"recovery={rows[-1]['spontaneous_recovery_bits']*100:+5.1f}pp")

    def stat(key: str) -> tuple[float, float, float, float]:
        vals = np.array([r[key] for r in rows])
        return (float(vals.mean()), float(vals.std()),
                float(vals.min()), float(vals.max()))

    keys = ["pre_lesion_pattern_all", "pre_lesion_bit_held_out",
            "post_lesion_pattern_all", "post_lesion_bit_held_out",
            "post_relearn_pattern_trained", "post_relearn_bit_trained",
            "post_relearn_pattern_held_out", "post_relearn_bit_held_out",
            "spontaneous_recovery_bits"]
    return {"n_seeds": n_seeds,
            "rows": rows,
            "stats": {k: dict(zip(["mean", "std", "min", "max"], stat(k)))
                      for k in keys}}


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--lesion-fraction", type=float, default=0.5,
                   help="fraction of W1+W2 weights to zero after stage 1")
    p.add_argument("--n-train-cycles", type=int, default=1500)
    p.add_argument("--n-relearn-cycles", type=int, default=50)
    p.add_argument("--lr", type=float, default=0.3)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--weight-decay", type=float, default=1e-3)
    p.add_argument("--init-scale", type=float, default=0.5)
    p.add_argument("--n-semantic-prototypes", type=int, default=4,
                   help="size of the shared semantic micro-feature pool")
    p.add_argument("--prototypes-per-word", type=int, default=2,
                   help="number of prototypes ORed together per sememe")
    p.add_argument("--sweep", type=int, default=0,
                   help="if > 0, run this many seeds and aggregate")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args()

    print(f"# python  : {sys.version.split()[0]}")
    print(f"# numpy   : {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    if args.sweep > 0:
        t0 = time.time()
        out = sweep(n_seeds=args.sweep,
                    lesion_fraction=args.lesion_fraction,
                    n_train_cycles=args.n_train_cycles,
                    n_relearn_cycles=args.n_relearn_cycles,
                    verbose=not args.quiet)
        dt = time.time() - t0
        print(f"\nSweep results ({args.sweep} seeds, "
              f"lesion={args.lesion_fraction*100:.0f}%):")
        for k, s in out["stats"].items():
            print(f"  {k:32s}  mean={s['mean']*100:6.2f}%  "
                  f"std={s['std']*100:5.2f}pp  "
                  f"min={s['min']*100:5.1f}%  max={s['max']*100:5.1f}%")
        print(f"  total wallclock: {dt:.1f}s")
        return

    t0 = time.time()
    out = run_protocol(seed=args.seed,
                       lesion_fraction=args.lesion_fraction,
                       n_train_cycles=args.n_train_cycles,
                       n_relearn_cycles=args.n_relearn_cycles,
                       lr=args.lr, momentum=args.momentum,
                       weight_decay=args.weight_decay,
                       init_scale=args.init_scale,
                       n_semantic_prototypes=args.n_semantic_prototypes,
                       prototypes_per_word=args.prototypes_per_word,
                       verbose=not args.quiet)
    dt = time.time() - t0

    s = out["summary"]
    print("\n=== Stage 1: trained on all 20 ===")
    print(f"  pattern acc (all 20)   : {s['pre_lesion_pattern_all']*100:5.1f}%")
    print(f"  pattern acc (held-out 2): {s['pre_lesion_pattern_held_out']*100:5.1f}%")
    print(f"  bit acc     (held-out 2): {s['pre_lesion_bit_held_out']*100:5.2f}%")
    print(f"\n=== Stage 2: lesion {s['lesion_fraction']*100:.0f}% of weights ===")
    print(f"  pattern acc (all 20)   : {s['post_lesion_pattern_all']*100:5.1f}%")
    print(f"  pattern acc (trained 18): {s['post_lesion_pattern_trained']*100:5.1f}%")
    print(f"  pattern acc (held-out 2): {s['post_lesion_pattern_held_out']*100:5.1f}%")
    print(f"  bit acc     (held-out 2): {s['post_lesion_bit_held_out']*100:5.2f}%")
    print(f"\n=== Stage 3-4: relearn on 18, test held-out 2 ===")
    print(f"  pattern acc (trained 18): {s['post_relearn_pattern_trained']*100:5.1f}%")
    print(f"  pattern acc (held-out 2): {s['post_relearn_pattern_held_out']*100:5.1f}%")
    print(f"  bit acc     (trained 18): {s['post_relearn_bit_trained']*100:5.2f}%")
    print(f"  bit acc     (held-out 2): {s['post_relearn_bit_held_out']*100:5.2f}%")
    print(f"\n  spontaneous recovery (bit, held-out): "
          f"{s['post_lesion_bit_held_out']*100:5.2f}% "
          f"-> {s['post_relearn_bit_held_out']*100:5.2f}%  "
          f"(+{s['spontaneous_recovery_bits']*100:+5.2f}pp)")
    print(f"\nWallclock: {dt:.2f}s   params: {s['n_params']}")


if __name__ == "__main__":
    main()
