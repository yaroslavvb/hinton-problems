"""
Synthetic-spectrogram riser/non-riser discrimination
(Plaut & Hinton 1987, "Learning sets of filters using back-propagation",
Computer Speech and Language, vol 2, pp 35-61).

A 6 frequency x 9 time = 54-D synthetic spectrogram with one "track"
(a path of frequencies indexed by time) plus iid Gaussian noise:

    Class 0 ("rising")  -- track is monotonically non-decreasing in freq
    Class 1 ("falling") -- track is monotonically non-increasing in freq

(Plaut & Hinton's original task contrasts upward-sweeping and
downward-sweeping formants -- the "non-rising" class is "falling"
formants. Constant tracks belong to both classes; we leave them in.)

A 54-24-2 sigmoid-hidden / softmax-output MLP trained with full-batch
backprop with momentum approaches the Bayes-optimal accuracy. The
Bayes-optimal classifier, computed in closed form here, marginalises
over all 2002 monotonic-rising tracks and all 2002 monotonic-falling
tracks via a small dynamic program.

Paper headline:
    network    97.8%
    Bayes-opt  98.8%
    gap         1.0 pp
"""

from __future__ import annotations
import argparse
import math
import platform
import sys
import time
import numpy as np


# ----------------------------------------------------------------------
# Constants / defaults
# ----------------------------------------------------------------------

N_FREQ = 6
N_TIME = 9
N_IN = N_FREQ * N_TIME            # 54
N_HIDDEN = 24
N_OUT = 2
DEFAULT_NOISE_STD = 0.6


# ----------------------------------------------------------------------
# Track sampling
# ----------------------------------------------------------------------

def sample_rising_tracks(rng: np.random.Generator, n: int,
                         n_freq: int = N_FREQ, n_time: int = N_TIME
                         ) -> np.ndarray:
    """Return n monotonically non-decreasing tracks of shape (n, n_time).

    Sampling trick: n_time uniform draws from {0..n_freq-1}, sorted in
    place. This gives a uniform distribution over the multiset
    "combinations with replacement" (= all monotone non-decreasing
    sequences); count = C(n_freq + n_time - 1, n_time) = 2002 for the
    default 6/9.
    """
    raw = rng.integers(0, n_freq, size=(n, n_time))
    raw.sort(axis=1)
    return raw


def sample_falling_tracks(rng: np.random.Generator, n: int,
                          n_freq: int = N_FREQ, n_time: int = N_TIME
                          ) -> np.ndarray:
    """Same as `sample_rising_tracks` but reversed in time."""
    raw = rng.integers(0, n_freq, size=(n, n_time))
    raw.sort(axis=1)
    return raw[:, ::-1].copy()


def tracks_to_specs(tracks: np.ndarray, n_freq: int = N_FREQ
                    ) -> np.ndarray:
    """(n, n_time) int -> (n, n_freq, n_time) clean spectrograms in {0, 1}."""
    n, n_time = tracks.shape
    spec = np.zeros((n, n_freq, n_time), dtype=np.float64)
    rows = tracks                                            # (n, n_time)
    cols = np.broadcast_to(np.arange(n_time), tracks.shape)
    samples = np.broadcast_to(np.arange(n)[:, None], tracks.shape)
    spec[samples, rows, cols] = 1.0
    return spec


def generate_dataset(n_samples: int,
                     noise_std: float = DEFAULT_NOISE_STD,
                     n_freq: int = N_FREQ, n_time: int = N_TIME,
                     seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Half rising (class 0), half falling (class 1).

    Returns
    -------
    X : (n_samples, n_freq * n_time) float64  -- noisy flattened spectrogram
    y : (n_samples,) int64                    -- 0 = rising, 1 = falling
    """
    rng = np.random.default_rng(seed)
    n_rise = n_samples // 2
    n_fall = n_samples - n_rise

    rise = sample_rising_tracks(rng, n_rise, n_freq, n_time)
    fall = sample_falling_tracks(rng, n_fall, n_freq, n_time)
    tracks = np.concatenate([rise, fall], axis=0)
    labels = np.concatenate([np.zeros(n_rise, dtype=np.int64),
                              np.ones(n_fall, dtype=np.int64)])

    clean = tracks_to_specs(tracks, n_freq)
    noise = noise_std * rng.standard_normal(clean.shape)
    X = (clean + noise).reshape(len(tracks), n_freq * n_time)

    perm = rng.permutation(len(tracks))
    return X[perm], labels[perm]


# ----------------------------------------------------------------------
# Bayes-optimal classifier (closed form via DP)
# ----------------------------------------------------------------------
#
# For any track f, p(x | f) factorises across (frequency, time) cells:
#
#     log p(x | f) = const(x, sigma)  +  (1/sigma^2) * sum_t x[f(t), t]
#
# Constants common to both classes drop out of the LR. So if
#
#     U[i, t] := x[i, t] / sigma^2
#
# then the class-conditional likelihood (uniform over the |C| tracks
# in class C) satisfies
#
#     log p(x | C) = const + log [ (1/|C|) * sum_{f in C} exp(sum_t U[f(t), t]) ]
#
# The inner sum is computable for monotone-rising tracks by a DP whose
# state is "the frequency of the track at time t":
#
#     dp[0, j] = U[j, 0]
#     dp[t, j] = U[j, t] + logsumexp_{j' <= j} dp[t-1, j']
#     log_Z   = logsumexp_j dp[T-1, j]
#
# For falling tracks, reverse time.

def logsumexp(a: np.ndarray, axis: int) -> np.ndarray:
    m = np.max(a, axis=axis, keepdims=True)
    out = np.squeeze(m, axis=axis) + np.log(
        np.sum(np.exp(a - m), axis=axis))
    return out


def log_cumsumexp_along_freq(log_dp: np.ndarray) -> np.ndarray:
    """Along the freq axis (axis=1 of (n, n_freq))."""
    return np.logaddexp.accumulate(log_dp, axis=1)


def _log_z_monotone(U: np.ndarray, ascending: bool) -> np.ndarray:
    """log sum_{f monotone} exp(sum_t U[f(t), t]).

    Parameters
    ----------
    U : (n, n_freq, n_time)
    ascending : True for non-decreasing tracks, False for non-increasing.
    """
    if not ascending:
        U = U[:, ::-1, :]
    n_time = U.shape[2]
    log_dp = U[:, :, 0].copy()                        # (n, n_freq)
    for t in range(1, n_time):
        log_csum = log_cumsumexp_along_freq(log_dp)
        log_dp = log_csum + U[:, :, t]
    return logsumexp(log_dp, axis=1)                  # (n,)


def bayes_log_posterior(X: np.ndarray, noise_std: float,
                        n_freq: int = N_FREQ, n_time: int = N_TIME
                        ) -> np.ndarray:
    """Return log p(class=0 | x) - log p(class=1 | x) for each row of x.

    Equal class priors. Positive value -> rising more likely, negative
    -> falling more likely. Class sizes are equal, so the per-class
    log-mean term cancels in the LR.
    """
    sigma2 = noise_std ** 2
    X_grid = X.reshape(-1, n_freq, n_time) / sigma2          # = U

    log_z_rise = _log_z_monotone(X_grid, ascending=True)
    log_z_fall = _log_z_monotone(X_grid, ascending=False)
    return log_z_rise - log_z_fall


def bayes_optimal_accuracy(noise_std: float,
                           n_samples: int = 50_000,
                           n_freq: int = N_FREQ, n_time: int = N_TIME,
                           seed: int = 12345) -> float:
    """Empirical Bayes-optimal accuracy at this noise level.

    Generates `n_samples` (x, y) pairs and reports the fraction the
    closed-form classifier gets right. The estimator's std is
    ~ sqrt(p(1-p)/n) -- with n=50000 and p~0.99 that's ~0.04 pp.
    """
    X, y = generate_dataset(n_samples, noise_std, n_freq, n_time,
                            seed=seed)
    log_lr = bayes_log_posterior(X, noise_std, n_freq, n_time)
    pred = (log_lr <= 0).astype(np.int64)            # 0 if rising more likely
    return float(np.mean(pred == y))


# ----------------------------------------------------------------------
# Model: 54-24-2 MLP
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - np.max(z, axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class RiserMLP:
    """Two-layer net: n_in -> n_hidden sigmoids -> n_out softmax."""

    def __init__(self, n_in: int = N_IN, n_hidden: int = N_HIDDEN,
                 n_out: int = N_OUT, init_scale: float = 0.3,
                 seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        rng = np.random.default_rng(seed)
        # Glorot-ish small init -- unitless because inputs are O(1)
        self.W1 = init_scale * rng.standard_normal((n_hidden, n_in)) \
            / math.sqrt(n_in)
        self.b1 = np.zeros(n_hidden, dtype=np.float64)
        self.W2 = init_scale * rng.standard_normal((n_out, n_hidden)) \
            / math.sqrt(n_hidden)
        self.b2 = np.zeros(n_out, dtype=np.float64)

    def forward(self, X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        h = sigmoid(X @ self.W1.T + self.b1)            # (n, H)
        z = h @ self.W2.T + self.b2                     # (n, n_out)
        p = softmax(z)
        return h, p

    def predict(self, X: np.ndarray) -> np.ndarray:
        _, p = self.forward(X)
        return np.argmax(p, axis=1)

    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def snapshot(self) -> dict:
        return {"W1": self.W1.copy(), "b1": self.b1.copy(),
                "W2": self.W2.copy(), "b2": self.b2.copy()}


def cross_entropy(p: np.ndarray, y: np.ndarray) -> float:
    """Mean -log p[i, y[i]]."""
    return float(-np.mean(np.log(p[np.arange(len(y)), y] + 1e-12)))


def accuracy(model: RiserMLP, X: np.ndarray, y: np.ndarray) -> float:
    return float(np.mean(model.predict(X) == y))


def backprop_grads(model: RiserMLP, X: np.ndarray, y: np.ndarray
                   ) -> dict[str, np.ndarray]:
    """d / d-param of mean cross-entropy on softmax output."""
    n = X.shape[0]
    h, p = model.forward(X)
    one_hot = np.zeros_like(p)
    one_hot[np.arange(n), y] = 1.0
    dz = (p - one_hot) / n                             # (n, n_out)
    grads = {
        "W2": dz.T @ h,                                # (n_out, H)
        "b2": dz.sum(axis=0),                          # (n_out,)
    }
    dh = (dz @ model.W2) * h * (1.0 - h)              # (n, H)
    grads["W1"] = dh.T @ X                             # (H, n_in)
    grads["b1"] = dh.sum(axis=0)                       # (H,)
    return grads


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(n_train: int = 2000,
          n_test: int = 4000,
          n_sweeps: int = 200,
          batch_size: int = 100,
          lr: float = 0.5,
          momentum: float = 0.9,
          weight_decay: float = 0.0,
          init_scale: float = 1.0,
          n_hidden: int = N_HIDDEN,
          noise_std: float = DEFAULT_NOISE_STD,
          online: bool = True,
          seed: int = 0,
          snapshot_callback=None,
          snapshot_every: int = 5,
          verbose: bool = True
          ) -> tuple[RiserMLP, dict, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Mini-batch backprop with momentum.

    With `online=True` (the default), the training set is *re-sampled*
    every epoch -- new tracks AND new noise. The clean track set has
    only ~4000 distinct elements so any fixed dataset is quickly memorised
    by a 1370-parameter net at the chosen noise level; online sampling
    gives the network an effectively unlimited stream of (track, noise)
    pairs. This is closer to the regime Plaut & Hinton (1987) evaluated
    in -- they report 97.8% test accuracy, a ~1 pp gap from Bayes.

    With `online=False`, a single fixed train set of size `n_train` is
    used for all epochs. Useful for studying overfitting.
    """
    # Independent seeds for data, test set, weights, and per-epoch resamples
    seed_seq = np.random.SeedSequence(seed)
    sd_data, sd_test, sd_weights, sd_epoch = seed_seq.spawn(4)

    X_test, y_test = generate_dataset(
        n_test, noise_std=noise_std, seed=int(sd_test.generate_state(1)[0]))

    model = RiserMLP(n_hidden=n_hidden, init_scale=init_scale,
                     seed=int(sd_weights.generate_state(1)[0]))

    velocities = {k: np.zeros_like(v) for k, v in [
        ("W1", model.W1), ("b1", model.b1),
        ("W2", model.W2), ("b2", model.b2)]}

    history = {"epoch": [], "train_loss": [], "train_acc": [],
               "test_acc": [], "weight_norm": []}

    if verbose:
        print(f"# {model.n_in}-{n_hidden}-{model.n_out} riser/non-riser  "
              f"params={model.n_params()}  train={n_train}  test={n_test}  "
              f"sigma={noise_std}  lr={lr}  momentum={momentum}  "
              f"batch={batch_size}  online={online}  seed={seed}")

    epoch_seed_seq = sd_epoch
    if not online:
        X_train, y_train = generate_dataset(
            n_train, noise_std=noise_std,
            seed=int(sd_data.generate_state(1)[0]))
    rng_shuffle = np.random.default_rng(sd_data)
    n_batches = max(1, n_train // batch_size)

    for epoch in range(n_sweeps):
        if online:
            # Independent fresh draw each epoch -- new tracks + new noise.
            (epoch_seed_seq,) = epoch_seed_seq.spawn(1)
            X_train, y_train = generate_dataset(
                n_train, noise_std=noise_std,
                seed=int(epoch_seed_seq.generate_state(1)[0]))
        perm = rng_shuffle.permutation(n_train)
        for b in range(n_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            grads = backprop_grads(model, X_train[idx], y_train[idx])
            for k in velocities:
                velocities[k] = momentum * velocities[k] - lr * (
                    grads[k] + weight_decay * getattr(model, k))
            model.W1 += velocities["W1"]
            model.b1 += velocities["b1"]
            model.W2 += velocities["W2"]
            model.b2 += velocities["b2"]

        _, p_train = model.forward(X_train)
        loss = cross_entropy(p_train, y_train)
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        wn = float(np.linalg.norm(model.W1))

        history["epoch"].append(epoch + 1)
        history["train_loss"].append(loss)
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        history["weight_norm"].append(wn)

        if snapshot_callback is not None and (
                epoch % snapshot_every == 0 or epoch == n_sweeps - 1):
            snapshot_callback(epoch, model, history)

        if verbose and (epoch % max(1, n_sweeps // 12) == 0
                        or epoch == n_sweeps - 1):
            print(f"  epoch {epoch+1:4d}  loss={loss:.4f}  "
                  f"train_acc={train_acc*100:5.2f}%  "
                  f"test_acc={test_acc*100:5.2f}%  |W1|={wn:.2f}")

    return model, history, X_train, y_train, X_test, y_test


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--noise-std", type=float, default=DEFAULT_NOISE_STD)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--offline", action="store_true",
                   help="train on a single fixed dataset (default: re-sample each epoch)")
    p.add_argument("--batch-size", type=int, default=100)
    p.add_argument("--lr", type=float, default=0.5)
    p.add_argument("--momentum", type=float, default=0.9)
    p.add_argument("--init-scale", type=float, default=1.0)
    p.add_argument("--n-hidden", type=int, default=N_HIDDEN)
    p.add_argument("--n-train", type=int, default=2000)
    p.add_argument("--n-test", type=int, default=4000)
    p.add_argument("--bayes-samples", type=int, default=50_000,
                   help="samples used to estimate Bayes-optimal accuracy")
    p.add_argument("--no-bayes", action="store_true",
                   help="skip Bayes-optimal estimate (faster)")
    args = p.parse_args()

    _print_environment()

    t_train_start = time.time()
    model, hist, _, _, X_test, y_test = train(
        n_train=args.n_train,
        n_test=args.n_test,
        n_sweeps=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        momentum=args.momentum,
        init_scale=args.init_scale,
        n_hidden=args.n_hidden,
        noise_std=args.noise_std,
        online=not args.offline,
        seed=args.seed,
    )
    t_train = time.time() - t_train_start

    net_acc = accuracy(model, X_test, y_test)

    print(f"\n=== final ===")
    print(f"network test accuracy : {net_acc*100:.2f}%  ({int(net_acc*len(y_test))}/{len(y_test)})")
    print(f"final train loss      : {hist['train_loss'][-1]:.4f}")
    print(f"training wallclock    : {t_train:.2f}s")

    if not args.no_bayes:
        t0 = time.time()
        bayes_acc = bayes_optimal_accuracy(
            args.noise_std,
            n_samples=args.bayes_samples,
            seed=args.seed + 7919,            # disjoint from train/test seeds
        )
        t_bayes = time.time() - t0
        gap = bayes_acc - net_acc
        print(f"Bayes-optimal accuracy: {bayes_acc*100:.2f}%  "
              f"(estimated from {args.bayes_samples} samples, "
              f"{t_bayes:.2f}s)")
        print(f"network - Bayes gap   : {gap*100:+.2f} pp")


if __name__ == "__main__":
    main()
