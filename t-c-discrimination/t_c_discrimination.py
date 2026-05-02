"""
T-C discrimination via a weight-tied (convolutional) network — early-CNN
reproduction of Rumelhart, Hinton & Williams (1986), PDP Vol. 1, Ch. 8.

Problem:
    8 patterns: a "block T" and a "block C", each in 4 rotations (0, 90, 180,
    270 degrees), drawn on a small 2D binary retina. The network must
    output the class label (T = 0, C = 1) regardless of rotation.

Architecture (the unique element):
    A single 3x3 receptive field is *slid* across the retina with **shared
    weights**. K such kernels run in parallel; their feature maps are
    mean-pooled to a K-dim vector and a final linear+sigmoid layer reads off
    T-vs-C. Forward in numpy:

        for each k in 1..K:
            pre_h[k, i, j] = sum_{a, c} W1[k, a, c] * X[i+a, j+c] + b1[k]
            h[k, i, j]     = sigmoid(pre_h[k, i, j])
        pooled[k]          = mean_{i, j} h[k, i, j]
        o                  = sigmoid(W2 @ pooled + b2)

    Weight tying makes this a convolutional layer in everything but name —
    Rumelhart et al. wrote the rule down 3 years before LeCun's first CNN.
    Backprop with a tied kernel just sums the per-position gradients before
    updating the kernel: dW1[k, a, c] = sum_{i, j} dpre_h[k, i, j] * X[i+a, j+c].

The interesting property — emergent feature detectors:
    Even with only 4 hidden kernels, training produces 3x3 weight patterns
    that visibly fall into recognisable categories: bar detectors (one row,
    column, or diagonal dominates), compactness detectors (a 2x2 block
    dominates), and on-centre / off-surround detectors (centre versus
    surround opposition). `taxonomize_filter()` labels each discovered
    kernel by the closest archetype; `visualize_filters()` returns the raw
    kernels.

This file is a numpy reproduction. Run `python3 t_c_discrimination.py` to
train, evaluate, and print the discovered kernels.
"""

from __future__ import annotations
import argparse
import platform
import sys
import time
import numpy as np


# ----------------------------------------------------------------------
# Patterns
# ----------------------------------------------------------------------

# Block T (3x3 bounding box, 5 cells): top bar + stem.
T_3X3 = np.array([
    [1, 1, 1],
    [0, 1, 0],
    [0, 1, 0],
], dtype=np.float64)

# Block C (3x3 bounding box, 5 cells): left bar + top tip + bottom tip.
# Asymmetric — its 4 rotations are all distinct, like T's.
C_3X3 = np.array([
    [1, 1, 0],
    [1, 0, 0],
    [1, 1, 0],
], dtype=np.float64)


def make_dataset(retina_size: int = 6, augment_positions: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Return X (n, R, R), y (n, 1), names (n,).

    Strict mode (`augment_positions=False`, default): exactly 8 patterns —
    block T at 4 rotations + block C at 4 rotations, placed at the centre
    of an R x R binary retina. Matches the spec from issue #24.

    Augmented mode (`augment_positions=True`): each of the 8 base patterns
    is placed at every valid (R - 2) x (R - 2) position on the retina,
    yielding 8 * (R - 2) ** 2 patterns. Encourages translation-invariant
    filters by making the same kernel see each feature at many positions.
    """
    R = retina_size
    if R < 4:
        raise ValueError(f"retina_size must be >= 4 (got {R})")

    bases = []
    for k in range(4):
        bases.append(("T", k, np.rot90(T_3X3, k)))
    for k in range(4):
        bases.append(("C", k, np.rot90(C_3X3, k)))

    patterns: list[np.ndarray] = []
    labels: list[float] = []
    names: list[str] = []

    if augment_positions:
        for cls, rot, pat in bases:
            for i in range(R - 2):
                for j in range(R - 2):
                    ret = np.zeros((R, R), dtype=np.float64)
                    ret[i:i + 3, j:j + 3] = pat
                    patterns.append(ret)
                    labels.append(0.0 if cls == "T" else 1.0)
                    names.append(f"{cls}_rot{rot * 90}_pos{i}{j}")
    else:
        offset = (R - 3) // 2
        for cls, rot, pat in bases:
            ret = np.zeros((R, R), dtype=np.float64)
            ret[offset:offset + 3, offset:offset + 3] = pat
            patterns.append(ret)
            labels.append(0.0 if cls == "T" else 1.0)
            names.append(f"{cls}_rot{rot * 90}")

    X = np.stack(patterns)
    y = np.array(labels, dtype=np.float64).reshape(-1, 1)
    return X, y, names


# ----------------------------------------------------------------------
# Activations
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def dsigmoid_from_y(y: np.ndarray) -> np.ndarray:
    return y * (1.0 - y)


# ----------------------------------------------------------------------
# Weight-tied convolutional model
# ----------------------------------------------------------------------

class WeightTiedConvNet:
    """K weight-tied 3x3 kernels -> sigmoid -> sum-pool -> FC -> sigmoid.

    Parameters
    ----------
    retina_size : R, side length of the square binary retina.
    kernel_size : side length of the receptive field (default 3).
    n_kernels   : number of independent shared-weight detectors (K).
    init_scale  : std of zero-mean Gaussian init for W1 and W2.
    seed        : RNG seed for weight init.

    Attributes
    ----------
    W1 : (K, kernel, kernel)  — shared kernels.
    b1 : (K,)                 — per-kernel scalar bias (also shared).
    W2 : (1, K)               — readout from pooled features.
    b2 : (1,)                 — output bias.
    """

    def __init__(self, retina_size: int = 6, kernel_size: int = 3,
                 n_kernels: int = 4, init_scale: float = 0.5, seed: int = 0):
        if retina_size < kernel_size:
            raise ValueError(
                f"retina_size ({retina_size}) must be >= kernel_size "
                f"({kernel_size})")
        self.R = retina_size
        self.K = kernel_size
        self.M = retina_size - kernel_size + 1   # spatial dim of feature map
        self.n_kernels = n_kernels
        self.rng = np.random.default_rng(seed)
        self.W1 = (init_scale
                   * self.rng.standard_normal((n_kernels, kernel_size,
                                                kernel_size)))
        self.b1 = np.zeros(n_kernels)
        self.W2 = init_scale * self.rng.standard_normal((1, n_kernels))
        self.b2 = np.zeros(1)

    def _patches(self, X: np.ndarray) -> np.ndarray:
        """Sliding 3x3 patches. X (B, R, R) -> patches (B, M, M, K, K)."""
        return np.lib.stride_tricks.sliding_window_view(
            X, (self.K, self.K), axis=(1, 2))

    def forward(self, X: np.ndarray
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray,
                            np.ndarray, np.ndarray]:
        """Return (h, pre_h, pooled, o, pre_o).

        Shapes: h, pre_h: (B, n_kernels, M, M); pooled: (B, n_kernels);
                pre_o, o: (B, 1).
        """
        patches = self._patches(X)
        # einsum: pre_h[b, k, i, j] = sum_{a, c} patches[b, i, j, a, c] *
        #                              W1[k, a, c] + b1[k]
        pre_h = (np.einsum("bijac,kac->bkij", patches, self.W1)
                 + self.b1[None, :, None, None])
        h = sigmoid(pre_h)
        # Mean-pool — keeps pooled in [0, 1] regardless of feature-map size,
        # so the readout sigmoid sees a calibrated input range. (Sum-pool
        # values scale with M^2 and saturate the output sigmoid at init.)
        pooled = h.mean(axis=(2, 3))
        pre_o = pooled @ self.W2.T + self.b2
        o = sigmoid(pre_o)
        return h, pre_h, pooled, o, pre_o

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, _, _, o, _ = self.forward(X)
        return o

    def n_params(self) -> int:
        return (self.W1.size + self.b1.size
                + self.W2.size + self.b2.size)

    def n_unique_params(self) -> int:
        """Same as `n_params`. The contrast number — how many params an
        equivalent fully-connected (untied) conv layer would need — is
        n_kernels * M * M * K * K + n_kernels * M * M + (W2/b2)."""
        return self.n_params()

    def n_params_if_untied(self) -> int:
        """Hypothetical param count if every position had its own kernel.

        Demonstrates the savings from weight-tying. Per-position kernel:
        n_kernels * M * M * K * K weights + n_kernels * M * M biases.
        """
        per_position = self.n_kernels * self.M * self.M * self.K * self.K
        per_position_bias = self.n_kernels * self.M * self.M
        readout = self.W2.size + self.b2.size
        return per_position + per_position_bias + readout


# ----------------------------------------------------------------------
# Backprop
# ----------------------------------------------------------------------

def backprop_grads(model: WeightTiedConvNet, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """Mean-squared-error gradients (loss = 0.5 * mean (o - y)**2)."""
    B = X.shape[0]
    h, _pre_h, pooled, o, _pre_o = model.forward(X)

    # Output layer
    delta_o = (o - y) * dsigmoid_from_y(o)               # (B, 1)
    dW2 = delta_o.T @ pooled / B                          # (1, K)
    db2 = delta_o.mean(axis=0)                            # (1,)

    # Through mean-pool: gradient at every spatial position equals
    # d_pooled / (M * M).
    d_pooled = delta_o @ model.W2                         # (B, K)
    d_h = (d_pooled[:, :, None, None]
           * np.ones((1, 1, model.M, model.M))
           / (model.M * model.M))                          # (B, K, M, M)
    d_pre_h = d_h * dsigmoid_from_y(h)                    # (B, K, M, M)

    # Tied-kernel gradient: sum the per-position gradient before updating.
    db1 = d_pre_h.sum(axis=(0, 2, 3)) / B                 # (K,)

    patches = model._patches(X)                            # (B, M, M, K, K)
    # dW1[k, a, c] = sum_{b, i, j} d_pre_h[b, k, i, j] * patches[b, i, j, a, c]
    dW1 = np.einsum("bkij,bijac->kac", d_pre_h, patches) / B
    return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}


def loss_mse(model: WeightTiedConvNet, X: np.ndarray, y: np.ndarray) -> float:
    o = model.predict(X)
    return 0.5 * float(np.mean((o - y) ** 2))


def accuracy(model: WeightTiedConvNet, X: np.ndarray, y: np.ndarray,
             threshold: float = 0.5) -> float:
    o = model.predict(X)
    pred = (o >= threshold).astype(np.float64)
    return float(np.mean(pred == y))


def converged(model: WeightTiedConvNet, X: np.ndarray, y: np.ndarray,
              tol: float = 0.5) -> bool:
    o = model.predict(X)
    return bool(np.all(np.abs(o - y) < tol))


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(retina_size: int = 6,
          kernel_size: int = 3,
          n_kernels: int = 4,
          lr: float = 0.5,
          momentum: float = 0.9,
          init_scale: float = 0.5,
          max_epochs: int = 5000,
          seed: int = 0,
          augment_positions: bool = False,
          snapshot_callback=None,
          snapshot_every: int = 20,
          early_stop_after: int = 50,
          verbose: bool = True,
          ) -> tuple[WeightTiedConvNet, dict]:
    """Train the weight-tied conv net on T-C with full-batch backprop+momentum.

    Returns (trained_model, history). `history["converged_epoch"]` is the
    first epoch where every output is within 0.5 of its target, or None.
    """
    model = WeightTiedConvNet(retina_size, kernel_size, n_kernels,
                               init_scale, seed)
    X, y, names = make_dataset(retina_size,
                                augment_positions=augment_positions)

    velocities = {k: np.zeros_like(v) for k, v in
                  [("W1", model.W1), ("b1", model.b1),
                   ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "loss": [], "accuracy": [],
               "weight_norm": [], "converged_epoch": None,
               "names": names}

    if verbose:
        print(f"# T-C discrimination on a {retina_size}x{retina_size} retina, "
              f"kernel {kernel_size}x{kernel_size}, "
              f"K = {n_kernels} weight-tied detectors")
        print(f"# patterns: {len(X)}  "
              f"(augmented = {augment_positions})  "
              f"params: {model.n_params()} "
              f"(untied equiv: {model.n_params_if_untied()})")

    for epoch in range(max_epochs):
        grads = backprop_grads(model, X, y)
        for k in velocities:
            velocities[k] = momentum * velocities[k] - lr * grads[k]
        model.W1 += velocities["W1"]
        model.b1 += velocities["b1"]
        model.W2 += velocities["W2"]
        model.b2 += velocities["b2"]

        L = loss_mse(model, X, y)
        acc = accuracy(model, X, y)
        wn = float(np.linalg.norm(np.concatenate(
            [model.W1.ravel(), model.W2.ravel()])))

        history["epoch"].append(epoch + 1)
        history["loss"].append(L)
        history["accuracy"].append(acc)
        history["weight_norm"].append(wn)

        if (history["converged_epoch"] is None
                and converged(model, X, y, tol=0.5)):
            history["converged_epoch"] = epoch + 1
            if verbose:
                print(f"  converged at epoch {epoch + 1}  "
                      f"loss={L:.4f}  acc={acc * 100:.0f}%")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == max_epochs - 1):
            snapshot_callback(epoch, model, history)

        log_step = max(max_epochs // 20, 50)
        if verbose and (epoch % log_step == 0 or epoch == max_epochs - 1):
            print(f"  epoch {epoch + 1:5d}  loss={L:.4f}  "
                  f"acc={acc * 100:5.1f}%  |W|={wn:.3f}")

        if (history["converged_epoch"] is not None
                and epoch + 1 >= history["converged_epoch"] + early_stop_after):
            break

    return model, history


# ----------------------------------------------------------------------
# Filter inspection: emergent detectors
# ----------------------------------------------------------------------

def visualize_filters(model: WeightTiedConvNet) -> np.ndarray:
    """Return the K kernels (n_kernels, kernel, kernel) for inspection.

    Stub-required signature. The actual rendering lives in
    `visualize_t_c_discrimination.py:plot_filters()`, which calls this and
    paints the resulting (K, k, k) tensor as a row of heatmaps annotated by
    `taxonomize_filter()`.
    """
    return model.W1.copy()


def taxonomize_filter(W: np.ndarray) -> str:
    """Classify a single 3x3 kernel as bar / compactness / on-centre /
    off-centre / mixed.

    Heuristics, in priority order:

    1. **on-centre / off-centre**: centre cell has opposite sign from the
       average of the surround, both substantial in magnitude. The classic
       Difference-of-Gaussians pattern (positive centre, negative surround
       = on-centre; opposite = off-centre).
    2. **bar**: a single row, column, or diagonal contains > 55 % of the
       total absolute weight. Detects oriented edges.
    3. **compactness**: a single 2x2 sub-block contains > 55 % of the total
       absolute weight. Detects compact 2x2 blobs (corners of C, tip of T).
    4. **mixed**: none of the above stand out.

    Thresholds were tuned to give crisp labels on our seeds. Kernels close
    to a boundary may be reported as "mixed" — the readout layer can still
    use them productively.
    """
    if W.shape != (3, 3):
        return "unknown"
    centre = float(W[1, 1])
    surround = float((W.sum() - W[1, 1]) / 8.0)
    total_abs = float(np.abs(W).sum())
    if total_abs < 1e-6:
        return "dead"
    max_abs = float(np.abs(W).max())

    # 1. centre-surround opposition
    if (centre * surround < 0
            and abs(centre) > 0.30 * max_abs
            and abs(surround) > 0.10 * max_abs):
        return "on-centre" if centre > 0 else "off-centre"

    # 2. bar detector — one line dominates
    row_sums = np.abs(W).sum(axis=1)         # 3 rows
    col_sums = np.abs(W).sum(axis=0)         # 3 cols
    diag1 = abs(W[0, 0]) + abs(W[1, 1]) + abs(W[2, 2])
    diag2 = abs(W[0, 2]) + abs(W[1, 1]) + abs(W[2, 0])
    line_strengths = list(row_sums) + list(col_sums) + [diag1, diag2]
    if max(line_strengths) > 0.55 * total_abs:
        return "bar"

    # 3. compactness — one 2x2 block dominates
    blocks = []
    for i in range(2):
        for j in range(2):
            blocks.append(np.abs(W[i:i + 2, j:j + 2]).sum())
    if max(blocks) > 0.55 * total_abs:
        return "compactness"

    return "mixed"


def filter_taxonomy(model: WeightTiedConvNet) -> list[str]:
    """Return the per-kernel detector type for the trained model."""
    return [taxonomize_filter(model.W1[k]) for k in range(model.n_kernels)]


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(
        description=__doc__.split("\n\n")[0] if __doc__ else None)
    p.add_argument("--retina-size", type=int, default=6,
                   help="Side length R of the square retina (default 6).")
    p.add_argument("--kernel-size", type=int, default=3,
                   help="Side length of the shared receptive field (default 3).")
    p.add_argument("--n-kernels", type=int, default=4,
                   help="Number of independent weight-tied kernels (default 4).")
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=0.5)
    p.add_argument("--max-epochs", type=int, default=5000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--augment-positions", action="store_true",
                   help="Place each pattern at every valid retina position. "
                        "Yields 8 * (R - 2)^2 patterns.")
    p.add_argument("--sweep", type=int, default=0,
                   help="If > 0, run this many seeds and report convergence "
                        "+ filter-taxonomy stats.")
    args = p.parse_args()

    print(f"# python  : {sys.version.split()[0]}")
    print(f"# numpy   : {np.__version__}")
    print(f"# platform: {platform.platform()}")
    print()

    if args.sweep > 0:
        run_sweep(args)
        return

    t0 = time.time()
    model, history = train(retina_size=args.retina_size,
                            kernel_size=args.kernel_size,
                            n_kernels=args.n_kernels, lr=args.lr,
                            momentum=args.momentum,
                            init_scale=args.init_scale,
                            max_epochs=args.max_epochs, seed=args.seed,
                            augment_positions=args.augment_positions)
    dt = time.time() - t0

    X, y, names = make_dataset(args.retina_size,
                                augment_positions=args.augment_positions)
    print(f"\nFinal accuracy : {accuracy(model, X, y) * 100:.0f}% "
          f"({int(accuracy(model, X, y) * len(y))}/{len(y)})")
    print(f"Final loss     : {loss_mse(model, X, y):.4f}")
    print(f"Converged epoch: {history['converged_epoch']}")
    print(f"Wallclock      : {dt:.3f}s")

    print(f"\nDiscovered {args.n_kernels} weight-tied "
          f"{args.kernel_size}x{args.kernel_size} kernels:")
    filters = visualize_filters(model)
    taxonomy = filter_taxonomy(model)
    for k in range(args.n_kernels):
        print(f"\n  Kernel {k + 1}  [{taxonomy[k]}]:")
        for row in filters[k]:
            print("    " + "  ".join(f"{x:+.3f}" for x in row))

    print("\nPer-pattern outputs:")
    o = model.predict(X).ravel()
    for nm, t, p in zip(names, y.ravel(), o):
        ok = "ok" if (p >= 0.5) == (t >= 0.5) else "FAIL"
        print(f"  {nm:20s}  target={int(t)}  output={p:.3f}  [{ok}]")


def run_sweep(args):
    """Multi-seed sweep — convergence rate + per-seed filter taxonomy."""
    print(f"# Running sweep over {args.sweep} seeds...")
    converged_seeds = 0
    epochs_to_converge: list[int] = []
    taxonomies: list[list[str]] = []
    t0 = time.time()
    for s in range(args.sweep):
        model, hist = train(retina_size=args.retina_size,
                             kernel_size=args.kernel_size,
                             n_kernels=args.n_kernels, lr=args.lr,
                             momentum=args.momentum,
                             init_scale=args.init_scale,
                             max_epochs=args.max_epochs, seed=s,
                             augment_positions=args.augment_positions,
                             verbose=False)
        ce = hist["converged_epoch"]
        if ce is not None:
            converged_seeds += 1
            epochs_to_converge.append(ce)
        tax = filter_taxonomy(model)
        taxonomies.append(tax)
        print(f"  seed {s:2d}  "
              f"converged={'y' if ce else 'n':1s}  "
              f"epoch={ce if ce else '-':>5}  "
              f"filters={tax}")
    dt = time.time() - t0

    from collections import Counter
    flat = [t for tax in taxonomies for t in tax]
    counts = Counter(flat)
    print(f"\nSweep summary  ({args.sweep} seeds, {dt:.1f}s):")
    print(f"  converged    : {converged_seeds}/{args.sweep}")
    if epochs_to_converge:
        print(f"  epochs       : "
              f"median={int(np.median(epochs_to_converge))}  "
              f"min={min(epochs_to_converge)}  "
              f"max={max(epochs_to_converge)}")
    print(f"  filter types : {dict(counts)}")
    total = sum(counts.values())
    if total:
        for tname, c in counts.most_common():
            print(f"    {tname:14s} {c:3d}  ({100 * c / total:5.1f}%)")


if __name__ == "__main__":
    main()
