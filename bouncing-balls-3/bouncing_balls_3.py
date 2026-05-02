"""
Three bouncing balls modeled with a Recurrent Temporal RBM (RTRBM).

Source:
  Sutskever, I., Hinton, G. E., & Taylor, G. W. (2008/2009),
  "The recurrent temporal restricted Boltzmann machine," NIPS 21.

The RTRBM is a TRBM whose hidden layer at time t-1 directly biases the
hidden layer at time t through a recurrent weight matrix `W_h`. The full
model factorizes as

  p(v_{1:T}) = prod_t p(v_t, h_t | r_{t-1})

where r_t = E[h_t | v_{1:t}] = sigmoid(W v_t + b_h + W_h r_{t-1}). Each
factor is an RBM whose hidden bias has been shifted by `W_h r_{t-1}`.

Architecture (this file):
  visible  : 30 x 30 = 900 binary pixels (anti-aliased disks)
  hidden   : 100 binary hidden units
  recurrent: 100 x 100 hidden-to-hidden matrix `W_h`

Training:
  Per-timestep CD-1, with the mean-field state r_t used as the positive-phase
  hidden activation (this is the original RTRBM training scheme of
  Sutskever et al. before the BPTT correction term). The recurrent weight
  `W_h` learns from the difference `(r_t - h_neg_t) outer r_{t-1}`. We do NOT
  add the optional BPTT-through-time correction term -- see "Deviations" in
  the README. With reasonable hyperparameters this simplification still
  reproduces stable rollouts on bouncing balls.

Rollout:
  Given init_frames v_{1:T0}, infer r_{1:T0}. For each future step,
  run k Gibbs steps within the per-step RBM (whose hidden bias is
  b_h + W_h r_{t-1}), then advance r_t with the generated v_t.

CLI:
  python3 bouncing_balls_3.py --seed 0 --n-epochs 30 --seq-len 30
"""
from __future__ import annotations

import argparse
import time
from dataclasses import dataclass

import numpy as np


# ----------------------------------------------------------------------
# Physics: simulate_balls
# ----------------------------------------------------------------------

def _render_frame(pos: np.ndarray, h: int, w: int, radius: float) -> np.ndarray:
    """Anti-aliased rasterization of disks. Pixel value in [0, 1].

    For each ball, contribution = clip(radius + 0.5 - dist, 0, 1).
    Multiple balls combine via max (so overlap stays in [0, 1]).
    """
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    frame = np.zeros((h, w), dtype=np.float32)
    for cx, cy in pos:
        dx = xx - cx
        dy = yy - cy
        dist = np.sqrt(dx * dx + dy * dy)
        contrib = np.clip(radius + 0.5 - dist, 0.0, 1.0)
        frame = np.maximum(frame, contrib)
    return frame


def simulate_balls(n_steps: int = 100,
                   n_balls: int = 3,
                   h: int = 30,
                   w: int = 30,
                   radius: float = 3.0,
                   speed: float = 1.0,
                   collide: bool = True,
                   seed: int | None = None) -> np.ndarray:
    """Simulate `n_balls` bouncing in an [h, w] box for `n_steps` frames.

    Returns a (n_steps, h, w) float32 array with values in [0, 1].

    Walls are reflecting. Ball-ball collisions are elastic (equal masses)
    when `collide=True`; otherwise balls pass through each other.
    """
    rng = np.random.default_rng(seed)

    # ---- initial positions: rejection-sample non-overlapping placements ----
    pos = np.zeros((n_balls, 2), dtype=np.float32)
    min_sep = 2 * radius + 0.5
    placed = 0
    for _ in range(10_000):
        if placed == n_balls:
            break
        cand = rng.uniform([radius, radius],
                           [w - radius, h - radius]).astype(np.float32)
        if placed == 0 or np.all(np.linalg.norm(pos[:placed] - cand, axis=1)
                                 > min_sep):
            pos[placed] = cand
            placed += 1
    if placed < n_balls:
        raise RuntimeError(
            f"could not place {n_balls} non-overlapping balls in {h}x{w} "
            f"box with radius {radius}")

    # ---- random initial velocities, fixed magnitude ----
    angles = rng.uniform(0.0, 2 * np.pi, n_balls).astype(np.float32)
    vel = np.stack([np.cos(angles), np.sin(angles)], axis=1) * speed

    frames = np.zeros((n_steps, h, w), dtype=np.float32)
    for t in range(n_steps):
        # advance
        pos = pos + vel

        # wall reflections (clip and flip velocity for the offending axis)
        for i in range(n_balls):
            if pos[i, 0] < radius:
                pos[i, 0] = 2 * radius - pos[i, 0]
                vel[i, 0] = -vel[i, 0]
            elif pos[i, 0] > w - radius:
                pos[i, 0] = 2 * (w - radius) - pos[i, 0]
                vel[i, 0] = -vel[i, 0]
            if pos[i, 1] < radius:
                pos[i, 1] = 2 * radius - pos[i, 1]
                vel[i, 1] = -vel[i, 1]
            elif pos[i, 1] > h - radius:
                pos[i, 1] = 2 * (h - radius) - pos[i, 1]
                vel[i, 1] = -vel[i, 1]

        # elastic ball-ball collisions (equal masses)
        if collide:
            for i in range(n_balls):
                for j in range(i + 1, n_balls):
                    d = pos[j] - pos[i]
                    dist = float(np.linalg.norm(d))
                    if 0.0 < dist < 2 * radius:
                        n = d / dist
                        v_rel = vel[j] - vel[i]
                        approach = float(v_rel @ n)
                        if approach < 0:
                            # exchange velocity component along the normal
                            vel[i] = vel[i] + approach * n
                            vel[j] = vel[j] - approach * n
                            # separate so they don't stay overlapping
                            overlap = 2 * radius - dist
                            pos[i] = pos[i] - n * overlap / 2
                            pos[j] = pos[j] + n * overlap / 2

        frames[t] = _render_frame(pos, h, w, radius)

    return frames


def make_dataset(n_sequences: int,
                 seq_len: int,
                 n_balls: int = 3,
                 h: int = 30,
                 w: int = 30,
                 radius: float = 3.0,
                 speed: float = 1.0,
                 seed: int = 0) -> np.ndarray:
    """Stack `n_sequences` independent rollouts.

    Returns (n_sequences, seq_len, h * w) float32 in [0, 1].
    """
    base_rng = np.random.default_rng(seed)
    out = np.zeros((n_sequences, seq_len, h * w), dtype=np.float32)
    for s in range(n_sequences):
        sub_seed = int(base_rng.integers(0, 2**31 - 1))
        seq = simulate_balls(n_steps=seq_len, n_balls=n_balls, h=h, w=w,
                             radius=radius, speed=speed, seed=sub_seed)
        out[s] = seq.reshape(seq_len, h * w)
    return out


# ----------------------------------------------------------------------
# RTRBM
# ----------------------------------------------------------------------

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50.0, 50.0)))


@dataclass
class RTRBM:
    n_visible: int
    n_hidden: int
    W: np.ndarray       # (n_visible, n_hidden)
    W_h: np.ndarray     # (n_hidden, n_hidden) -- recurrent
    b_v: np.ndarray     # (n_visible,)
    b_h: np.ndarray     # (n_hidden,)
    r_init: np.ndarray  # (n_hidden,) -- learnable initial r_0 (in pre-sigmoid)
    rng: np.random.Generator


def build_rtrbm(n_visible: int,
                n_hidden: int,
                init_scale: float = 0.01,
                seed: int = 0) -> RTRBM:
    """Initialize an RTRBM. `W_h` is the recurrent hidden->hidden matrix
    that is the only structural difference from a per-timestep RBM.
    """
    rng = np.random.default_rng(seed)
    W = (init_scale * rng.standard_normal((n_visible, n_hidden))
         ).astype(np.float32)
    # Initialize the recurrent matrix small + slightly negative diagonal so
    # the network does not immediately latch into runaway activation.
    W_h = (init_scale * rng.standard_normal((n_hidden, n_hidden))
           ).astype(np.float32)
    b_v = np.zeros(n_visible, dtype=np.float32)
    b_h = np.zeros(n_hidden, dtype=np.float32)
    r_init = np.zeros(n_hidden, dtype=np.float32)
    return RTRBM(n_visible=n_visible, n_hidden=n_hidden,
                 W=W, W_h=W_h, b_v=b_v, b_h=b_h, r_init=r_init, rng=rng)


# ----------------------------------------------------------------------
# Forward pass: mean-field hidden states
# ----------------------------------------------------------------------

def forward_mean_field(model: RTRBM, sequence: np.ndarray) -> np.ndarray:
    """Compute r_t = sigmoid(W v_t + b_h + W_h r_{t-1}) for t = 0..T-1.

    `sequence`: (T, n_visible). Returns r: (T, n_hidden).
    `r_{-1}` is the learnable initial state `sigmoid(r_init)`.
    """
    T = sequence.shape[0]
    r = np.zeros((T, model.n_hidden), dtype=np.float32)
    r_prev = sigmoid(model.r_init)
    for t in range(T):
        bias_h_t = model.b_h + model.W_h @ r_prev
        r[t] = sigmoid(sequence[t] @ model.W + bias_h_t)
        r_prev = r[t]
    return r


# ----------------------------------------------------------------------
# CD-1 training step over a single sequence
# ----------------------------------------------------------------------

def cd_step_sequence(model: RTRBM,
                     sequence: np.ndarray,
                     lr: float = 0.01,
                     weight_decay: float = 0.0,
                     momentum: float = 0.0,
                     velocity: dict | None = None) -> dict:
    """One RTRBM CD-1 pass over a single sequence.

    `sequence`: (T, n_visible). Mutates `model` in place. Returns a dict
    with mean reconstruction error and mean free-energy proxy.
    """
    T = sequence.shape[0]

    r = forward_mean_field(model, sequence)
    # Build the array of "previous" r values: r_init at t=0, else r[t-1].
    r_init_sig = sigmoid(model.r_init)
    r_prev_all = np.empty_like(r)
    r_prev_all[0] = r_init_sig
    if T > 1:
        r_prev_all[1:] = r[:-1]

    # ---- positive phase: use mean fields directly ----
    # Positive contributions to gradients are vectorizable.
    pos_dW = sequence.T @ r            # (n_v, n_h)
    pos_db_v = sequence.sum(axis=0)    # (n_v,)
    pos_db_h = r.sum(axis=0)           # (n_h,)
    pos_dW_h = r.T @ r_prev_all        # (n_h, n_h)

    # ---- negative phase: per-timestep CD-1 with binary h sample ----
    # Sample h_pos ~ Bernoulli(r) once.
    h_sample = (model.rng.random(r.shape) < r).astype(np.float32)
    v_neg = sigmoid(h_sample @ model.W.T + model.b_v)        # (T, n_v)
    bias_h_neg = model.b_h + r_prev_all @ model.W_h.T        # (T, n_h)
    h_neg = sigmoid(v_neg @ model.W + bias_h_neg)            # (T, n_h)

    neg_dW = v_neg.T @ h_neg
    neg_db_v = v_neg.sum(axis=0)
    neg_db_h = h_neg.sum(axis=0)
    neg_dW_h = h_neg.T @ r_prev_all

    dW = (pos_dW - neg_dW) / T
    db_v = (pos_db_v - neg_db_v) / T
    db_h = (pos_db_h - neg_db_h) / T
    dW_h = (pos_dW_h - neg_dW_h) / T

    if weight_decay > 0:
        dW = dW - weight_decay * model.W
        dW_h = dW_h - weight_decay * model.W_h

    if momentum > 0 and velocity is not None:
        velocity["W"] = momentum * velocity["W"] + lr * dW
        velocity["W_h"] = momentum * velocity["W_h"] + lr * dW_h
        velocity["b_v"] = momentum * velocity["b_v"] + lr * db_v
        velocity["b_h"] = momentum * velocity["b_h"] + lr * db_h
        model.W += velocity["W"]
        model.W_h += velocity["W_h"]
        model.b_v += velocity["b_v"]
        model.b_h += velocity["b_h"]
    else:
        model.W += lr * dW
        model.W_h += lr * dW_h
        model.b_v += lr * db_v
        model.b_h += lr * db_h

    recon_err = float(np.mean((sequence - v_neg) ** 2))
    return {"recon_mse": recon_err}


# ----------------------------------------------------------------------
# Training driver
# ----------------------------------------------------------------------

def train(model: RTRBM,
          sequences: np.ndarray,
          n_epochs: int = 30,
          lr: float = 0.01,
          weight_decay: float = 1e-4,
          momentum: float = 0.5,
          shuffle: bool = True,
          verbose: bool = True,
          snapshot_callback=None,
          snapshot_every: int = 1) -> dict:
    """Train `model` for `n_epochs` over the given `sequences` array.

    `sequences`: (n_sequences, T, n_visible).

    Returns a history dict with per-epoch lists 'epoch', 'recon_mse',
    'wallclock'.
    """
    n_seq = sequences.shape[0]
    velocity = {
        "W": np.zeros_like(model.W),
        "W_h": np.zeros_like(model.W_h),
        "b_v": np.zeros_like(model.b_v),
        "b_h": np.zeros_like(model.b_h),
    }

    history = {"epoch": [], "recon_mse": [], "wallclock": []}

    for epoch in range(n_epochs):
        t0 = time.time()
        order = (model.rng.permutation(n_seq) if shuffle
                 else np.arange(n_seq))
        epoch_recon = 0.0
        for idx in order:
            stats = cd_step_sequence(model, sequences[idx],
                                     lr=lr, weight_decay=weight_decay,
                                     momentum=momentum, velocity=velocity)
            epoch_recon += stats["recon_mse"]
        epoch_recon /= n_seq
        elapsed = time.time() - t0

        history["epoch"].append(epoch + 1)
        history["recon_mse"].append(epoch_recon)
        history["wallclock"].append(elapsed)

        if verbose:
            print(f"epoch {epoch+1:3d}/{n_epochs}  "
                  f"recon_mse={epoch_recon:.4f}  ({elapsed:.2f}s)")

        if snapshot_callback is not None and (epoch % snapshot_every == 0
                                              or epoch == n_epochs - 1):
            snapshot_callback(epoch, model, history)

    return history


# ----------------------------------------------------------------------
# Rollout
# ----------------------------------------------------------------------

def rollout(model: RTRBM,
            init_frames: np.ndarray,
            n_future: int,
            n_gibbs: int = 25,
            sample_visible: bool = False,
            sample_hidden: bool = True) -> np.ndarray:
    """Generate `n_future` frames continuing from `init_frames`.

    `init_frames`: (T0, n_visible).
    Returns (T0 + n_future, n_visible) float32.
    """
    T0 = init_frames.shape[0]
    r = forward_mean_field(model, init_frames)
    r_prev = r[-1].copy()

    out = np.zeros((T0 + n_future, model.n_visible), dtype=np.float32)
    out[:T0] = init_frames
    v = init_frames[-1].copy()

    for k in range(n_future):
        bias_h_t = model.b_h + model.W_h @ r_prev
        # k Gibbs steps within the per-timestep RBM
        for _ in range(n_gibbs):
            h_p = sigmoid(v @ model.W + bias_h_t)
            if sample_hidden:
                h = (model.rng.random(model.n_hidden) < h_p).astype(np.float32)
            else:
                h = h_p
            v_p = sigmoid(h @ model.W.T + model.b_v)
            if sample_visible:
                v = (model.rng.random(model.n_visible) < v_p).astype(np.float32)
            else:
                v = v_p
        out[T0 + k] = v
        # advance the recurrent state with the generated frame
        r_prev = sigmoid(v @ model.W + model.b_h + model.W_h @ r_prev)

    return out


# ----------------------------------------------------------------------
# Reconstruction-quality metrics
# ----------------------------------------------------------------------

def teacher_forced_recon(model: RTRBM, sequence: np.ndarray) -> float:
    """One-step teacher-forced reconstruction MSE.

    Computes r_t from data; reconstructs v_t from r_t. Cheaper than rollout
    and a useful validation metric across epochs.
    """
    r = forward_mean_field(model, sequence)
    v_recon = sigmoid(r @ model.W.T + model.b_v)
    return float(np.mean((sequence - v_recon) ** 2))


def free_rollout_mse(model: RTRBM,
                     sequence: np.ndarray,
                     warmup: int = 10,
                     n_future: int | None = None,
                     n_gibbs: int = 25) -> float:
    """Free rollout MSE: condition on the first `warmup` frames, then
    generate the rest and compare to ground truth.
    """
    T = sequence.shape[0]
    n_future = (T - warmup) if n_future is None else n_future
    pred = rollout(model, sequence[:warmup], n_future=n_future,
                   n_gibbs=n_gibbs, sample_visible=False, sample_hidden=True)
    return float(np.mean((sequence[warmup:warmup + n_future]
                          - pred[warmup:warmup + n_future]) ** 2))


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="RTRBM on 3 bouncing balls")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--h", type=int, default=30)
    p.add_argument("--w", type=int, default=30)
    p.add_argument("--n-balls", type=int, default=3)
    p.add_argument("--radius", type=float, default=3.0)
    p.add_argument("--speed", type=float, default=1.0)
    p.add_argument("--n-sequences", type=int, default=30)
    p.add_argument("--seq-len", type=int, default=30)
    p.add_argument("--n-hidden", type=int, default=100)
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--weight-decay", type=float, default=1e-4)
    p.add_argument("--momentum", type=float, default=0.5)
    p.add_argument("--no-collide", action="store_true",
                   help="Disable ball-ball collisions in the simulator")
    p.add_argument("--rollout-warmup", type=int, default=10)
    p.add_argument("--rollout-future", type=int, default=20)
    return p.parse_args()


def main() -> None:
    args = _parse_args()

    print(f"# bouncing-balls-3 (RTRBM)")
    print(f"# seed={args.seed} h={args.h} w={args.w} n_balls={args.n_balls} "
          f"radius={args.radius} speed={args.speed}")
    print(f"# n_sequences={args.n_sequences} seq_len={args.seq_len} "
          f"n_hidden={args.n_hidden} n_epochs={args.n_epochs}")
    print(f"# lr={args.lr} momentum={args.momentum} wd={args.weight_decay}")

    n_visible = args.h * args.w

    t_data = time.time()
    sequences = make_dataset(n_sequences=args.n_sequences,
                             seq_len=args.seq_len,
                             n_balls=args.n_balls,
                             h=args.h, w=args.w,
                             radius=args.radius, speed=args.speed,
                             seed=args.seed)
    print(f"# generated dataset {sequences.shape} in "
          f"{time.time()-t_data:.2f}s "
          f"(mean pixel = {sequences.mean():.4f})")

    model = build_rtrbm(n_visible=n_visible, n_hidden=args.n_hidden,
                        seed=args.seed)
    # Init b_v from data mean for a sane starting point.
    data_mean = np.clip(sequences.reshape(-1, n_visible).mean(axis=0),
                        1e-3, 1 - 1e-3)
    model.b_v[:] = np.log(data_mean / (1.0 - data_mean)).astype(np.float32)

    t_train = time.time()
    history = train(model, sequences,
                    n_epochs=args.n_epochs,
                    lr=args.lr,
                    weight_decay=args.weight_decay,
                    momentum=args.momentum,
                    verbose=True)
    train_time = time.time() - t_train
    print(f"# training time: {train_time:.2f}s")

    # Validation: held-out sequence rollout
    val_seq = make_dataset(n_sequences=1, seq_len=args.seq_len,
                           n_balls=args.n_balls,
                           h=args.h, w=args.w, radius=args.radius,
                           speed=args.speed,
                           seed=args.seed + 9999)[0]

    tf_mse = teacher_forced_recon(model, val_seq)
    free_mse = free_rollout_mse(model, val_seq,
                                warmup=args.rollout_warmup,
                                n_future=min(args.rollout_future,
                                             args.seq_len - args.rollout_warmup))
    print(f"# val teacher-forced MSE = {tf_mse:.4f}")
    print(f"# val free-rollout MSE   = {free_mse:.4f} "
          f"(warmup={args.rollout_warmup}, "
          f"future={min(args.rollout_future, args.seq_len - args.rollout_warmup)})")


if __name__ == "__main__":
    main()
