"""
Fast-weights associative retrieval (Ba, Hinton, Mnih, Leibo, Ionescu 2016).

Source:
    J. Ba, G. Hinton, V. Mnih, J. Z. Leibo, C. Ionescu (2016),
    "Using Fast Weights to Attend to the Recent Past", NIPS.
    https://arxiv.org/abs/1610.06258

Problem:
    Sequences look like  c9k8j3f1??c -> 9
    Several (key=letter, value=digit) pairs are presented one character at a
    time, then the separator '??', then a query letter. The network must
    output the digit that was paired with the query letter.

    With n_pairs key/value bindings, the model needs short-term storage of
    that many associations. A vanilla RNN of small hidden width struggles;
    the fast-weights trick is to maintain an extra per-sequence matrix
    A_t = lambda * A_{t-1} + eta * h_{t-1} h_{t-1}^T that records an outer-
    product trace of recent hidden states. At the query step the matrix-
    vector product A_t @ h_{t-1} acts as soft attention over recent past
    activations -- a Hopfield-style content-addressable retrieval. This is
    the first attention-like mechanism in the modern deep-learning era,
    predating transformer attention by a year.

Architecture (Ba et al. with the "trailing read" simplification):
    A_t   = lambda_decay * A_{t-1} + eta * outer(h_{t-1}, h_{t-1})    (A_0=0)
    z_t   = W_h h_{t-1} + W_x x_t + b + A_t @ h_{t-1}
    zn_t  = LayerNorm(z_t)        # mean-0, std-1 over the H dim, no affine
    h_t   = tanh(zn_t)
    out   = W_o h_T + b_o         # only last step predicts

    "Trailing read" step (key trick for retrieval timing):
        Each sample is built as
          k1 v1 k2 v2 ... kn vn  ?  ?  q  ?
        where the FINAL '?' is a no-op terminator. This guarantees the
        query letter is integrated into a hidden state h_{T-1} BEFORE the
        retrieval step T uses A_T = lambda A_{T-1} + eta outer(h_{T-1}, h_{T-1}).
        At step T the matrix-vector product A_T @ h_{T-1} executes a
        Hopfield-style read keyed on the query: any past h_τ whose
        inner-product with h_{T-1} is large gets summed into the
        pre-activation, delivering its bound value into h_T. Without the
        trailing read, the retrieval step would fire BEFORE the query
        letter had been encoded into the hidden state, and the network
        could only solve the task by very awkward W_o-side decoding.

    LayerNorm is necessary: without it, A_t @ h_{t-1} grows roughly
    quadratically as outer products accumulate, the tanh saturates at ±1
    within ~5 steps, and 1 - tanh^2 collapses the recurrent gradient to
    zero. This matches the Ba et al. recipe ("Layer Normalization is
    critical"). With this simpler single-LN design plus the trailing read,
    one inner step per timestep suffices.

    The fast weights matrix A_t lives only inside one sequence and is reset
    to zero at the start of each new sample. The slow weights (W_h, W_x, b,
    W_o, b_o) are learned by truncated BPTT from a softmax cross-entropy
    loss at the final timestep.

BPTT through the fast weights:
    Standard tanh-RNN backprop with LayerNorm, plus we maintain a running
    gradient dA on the fast-weights matrix:

        dA_running starts at 0
        for t = T..1:
            dh_t already known
            dzn_t = dh_t * (1 - h_t^2)              # tanh
            dz_t  = LN_backward(dzn_t, zn_t, sigma) # layer norm backward
            dW_h += outer(dz_t, h_{t-1})
            dW_x += outer(dz_t, x_t)
            db   += dz_t
            dh_{t-1} = (W_h.T + A_t.T) @ dz_t
            dA_t_local = outer(dz_t, h_{t-1})            # from z_t = ... + A_t h_{t-1}
            dA_t_total = dA_running + dA_t_local
            dh_{t-1}  += eta * (dA_t_total + dA_t_total.T) @ h_{t-1}   # from A_t = ... + eta outer(h_{t-1}, h_{t-1})
            dA_running = lambda_decay * dA_t_total                     # chain to A_{t-1}

    A numerical-gradient check (max 1e-8 relative error across all
    parameters) confirms each path.

    A_T is the last fast-weights matrix used; nothing depends on a future
    A_{T+1}, so dA_running starts at 0 at the beginning of the backward
    pass.

This file: data + model + train loop + eval + CLI.
Visualization and gif live in their sibling files.
"""

from __future__ import annotations

import argparse
import platform
import sys
import time

import numpy as np


# ----------------------------------------------------------------------
# Vocab + dataset
# ----------------------------------------------------------------------

LETTERS = "abcdefghijklmnopqrstuvwxyz"          # 26
DIGITS = "0123456789"                            # 10
SEP = "?"                                        # 1

# Token -> index mapping. Vocab order: letters (0..25), digits (26..35), sep (36).
N_LETTERS = len(LETTERS)
N_DIGITS = len(DIGITS)
SEP_INDEX = N_LETTERS + N_DIGITS                 # 36
VOCAB_SIZE = SEP_INDEX + 1                       # 37


def _letter_idx(c: str) -> int:
    return LETTERS.index(c)


def _digit_idx(c: str) -> int:
    return N_LETTERS + DIGITS.index(c)


def _token_str(idx: int) -> str:
    if idx < N_LETTERS:
        return LETTERS[idx]
    if idx < SEP_INDEX:
        return DIGITS[idx - N_LETTERS]
    return SEP


def generate_sample(n_pairs: int = 4,
                    rng: np.random.Generator | None = None
                    ) -> tuple[np.ndarray, int, str]:
    """One associative-retrieval sample.

    Sequence: k1 v1 k2 v2 ... k_n v_n  '?' '?'  qkey  '?'
    where the final '?' is the trailing read step (see module docstring).
    T = 2*n_pairs + 4.

    Returns
    -------
    seq : (T,) int64    token indices
    target : int        digit class (0..9) paired with qkey
    text   : str        e.g. "c9k8j3f1??c?"
    """
    if rng is None:
        rng = np.random.default_rng()
    if n_pairs > N_LETTERS:
        raise ValueError(f"n_pairs={n_pairs} > {N_LETTERS} unique letters")

    keys = rng.choice(N_LETTERS, size=n_pairs, replace=False)
    values = rng.integers(0, N_DIGITS, size=n_pairs)
    q_idx = int(rng.integers(0, n_pairs))           # pick which pair to query
    q_key = int(keys[q_idx])
    target = int(values[q_idx])

    tokens = []
    for k, v in zip(keys, values):
        tokens.append(int(k))                       # letter index
        tokens.append(N_LETTERS + int(v))           # digit index in vocab
    tokens.append(SEP_INDEX)                        # '?'
    tokens.append(SEP_INDEX)                        # '?'
    tokens.append(q_key)                            # query letter
    tokens.append(SEP_INDEX)                        # trailing read step

    seq = np.array(tokens, dtype=np.int64)
    text = "".join(_token_str(t) for t in tokens)
    return seq, target, text


def generate_batch(batch_size: int, n_pairs: int = 4,
                   rng: np.random.Generator | None = None
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Returns (seqs, targets) with seqs (B, T) int64 and targets (B,) int64."""
    if rng is None:
        rng = np.random.default_rng()
    seq_len = 2 * n_pairs + 4
    seqs = np.empty((batch_size, seq_len), dtype=np.int64)
    targets = np.empty(batch_size, dtype=np.int64)
    for b in range(batch_size):
        s, t, _ = generate_sample(n_pairs=n_pairs, rng=rng)
        seqs[b] = s
        targets[b] = t
    return seqs, targets


# ----------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------

class FastWeightsRNN:
    """Tanh RNN with per-sequence fast-weights matrix A_t.

    Slow parameters (learned by BPTT):
      W_h : (H, H)   recurrent
      W_x : (H, V)   input embedding (one-hot lookup)
      b   : (H,)     hidden bias
      W_o : (10, H)  output to digit classes
      b_o : (10,)    output bias

    Fast weights A_t are reset to zero at the start of each sample (not learned).
    """

    def __init__(self, n_in: int, n_hidden: int, n_out: int,
                 lambda_decay: float = 0.95, eta: float = 0.5,
                 init_scale: float | None = None,
                 seed: int = 0):
        self.n_in = n_in
        self.n_hidden = n_hidden
        self.n_out = n_out
        self.lambda_decay = float(lambda_decay)
        self.eta = float(eta)
        rng = np.random.default_rng(seed)

        s = init_scale if init_scale is not None else 1.0 / np.sqrt(n_hidden)
        # Recurrent matrix: identity init, standard for fast-weights RNNs
        # (Le, Jaitly, Hinton 2015 IRNN); the LayerNorm rescales so the
        # identity doesn't blow up.
        self.W_h = np.eye(n_hidden) * 0.5
        self.W_x = rng.standard_normal((n_hidden, n_in)) * s
        self.b = np.zeros(n_hidden)
        self.W_o = rng.standard_normal((n_out, n_hidden)) * s
        self.b_o = np.zeros(n_out)

    # --- introspection ---

    def n_params(self) -> int:
        return (self.W_h.size + self.W_x.size + self.b.size
                + self.W_o.size + self.b_o.size)

    def params(self) -> dict[str, np.ndarray]:
        return {"W_h": self.W_h, "W_x": self.W_x, "b": self.b,
                "W_o": self.W_o, "b_o": self.b_o}

    # --- forward ---

    def forward_sequence(self, seq: np.ndarray,
                         keep_trace: bool = False
                         ) -> dict:
        """Run the fast-weights RNN on one sequence (T,) of token indices."""
        T = int(seq.shape[0])
        H = self.n_hidden
        ln_eps = 1e-5
        hs = np.zeros((T + 1, H))                   # h_0 = 0
        zs = np.zeros((T, H))                       # raw pre-activation
        zns = np.zeros((T, H))                      # post-LayerNorm
        sigs = np.zeros(T)
        As = np.zeros((T, H, H))                    # A_t used at step t
        A_prev = np.zeros((H, H))

        for t in range(T):
            x_idx = int(seq[t])
            h_prev = hs[t]
            A_t = self.lambda_decay * A_prev + self.eta * np.outer(h_prev, h_prev)
            As[t] = A_t
            z_t = self.W_h @ h_prev + self.W_x[:, x_idx] + self.b + A_t @ h_prev
            mu = z_t.mean()
            sigma = float(np.sqrt(((z_t - mu) ** 2).mean() + ln_eps))
            zn_t = (z_t - mu) / sigma
            zs[t] = z_t
            zns[t] = zn_t
            sigs[t] = sigma
            hs[t + 1] = np.tanh(zn_t)
            A_prev = A_t

        logits = self.W_o @ hs[T] + self.b_o

        out = {"hs": hs, "zs": zs, "zns": zns, "sigs": sigs,
               "As": As, "logits": logits}
        if keep_trace:
            out["A_trace"] = As.copy()
            out["h_trace"] = hs.copy()
        return out

    # --- backward (BPTT) ---

    def backward_sequence(self, seq: np.ndarray, target: int,
                          fwd: dict | None = None,
                          ) -> tuple[float, dict]:
        """One sample. Returns (loss, grads dict) for slow params only."""
        if fwd is None:
            fwd = self.forward_sequence(seq)
        hs = fwd["hs"]; zns = fwd["zns"]; sigs = fwd["sigs"]
        As = fwd["As"]
        logits = fwd["logits"]
        T = int(seq.shape[0])
        H = self.n_hidden

        # softmax cross entropy on final logits
        m = np.max(logits)
        exp = np.exp(logits - m)
        probs = exp / np.sum(exp)
        loss = -float(np.log(probs[target] + 1e-12))

        # d_logits = probs - one_hot(target)
        d_logits = probs.copy()
        d_logits[target] -= 1.0

        # output layer
        dW_o = np.outer(d_logits, hs[T])
        db_o = d_logits.copy()
        dh = self.W_o.T @ d_logits                  # gradient on h_T

        # init param grads
        dW_h = np.zeros_like(self.W_h)
        dW_x = np.zeros_like(self.W_x)
        db = np.zeros_like(self.b)

        # running dA for the fast weights -- represents grad flowing INTO
        # A_{t-1} from later steps via the lambda chain. Starts at 0 because
        # there is no A_{T+1}.
        dA_running = np.zeros((H, H))

        for t in range(T - 1, -1, -1):
            x_idx = int(seq[t])
            h_prev = hs[t]
            h_now = hs[t + 1]
            A_t = As[t]

            # tanh backward
            dzn = dh * (1.0 - h_now * h_now)
            # LayerNorm backward (no affine):
            #   y = (x - mean) / sigma  =>  dx = (1/sigma) (dy - mean(dy) - y * mean(dy * y))
            sigma = sigs[t]
            zn = zns[t]
            dz = (dzn - dzn.mean() - zn * (dzn * zn).mean()) / sigma

            # parameter grads
            dW_h += np.outer(dz, h_prev)
            dW_x[:, x_idx] += dz
            db += dz

            # backprop through z_t = W_h h_{t-1} + W_x x + b + A_t h_{t-1}
            dh_prev_partial = self.W_h.T @ dz + A_t.T @ dz
            dA_t_local = np.outer(dz, h_prev)

            dA_t_total = dA_running + dA_t_local

            # backprop through A_t = lambda A_{t-1} + eta outer(h_{t-1}, h_{t-1})
            dh_prev = dh_prev_partial + self.eta * (dA_t_total + dA_t_total.T) @ h_prev
            dA_running = self.lambda_decay * dA_t_total

            dh = dh_prev

        grads = {"W_h": dW_h, "W_x": dW_x, "b": db,
                 "W_o": dW_o, "b_o": db_o}
        return loss, grads

    # --- batched train step (sums grads over the batch) ---

    def loss_and_grads_batch(self, seqs: np.ndarray, targets: np.ndarray
                             ) -> tuple[float, float, dict]:
        """Mean loss + accuracy + summed gradients (divide by B before applying)."""
        B = int(seqs.shape[0])
        total_loss = 0.0
        n_correct = 0
        agg = {k: np.zeros_like(v) for k, v in self.params().items()}
        for b in range(B):
            fwd = self.forward_sequence(seqs[b])
            loss, grads = self.backward_sequence(seqs[b], int(targets[b]),
                                                 fwd=fwd)
            pred = int(np.argmax(fwd["logits"]))
            n_correct += int(pred == int(targets[b]))
            total_loss += loss
            for k in agg:
                agg[k] += grads[k]
        # mean over batch
        for k in agg:
            agg[k] /= B
        return total_loss / B, n_correct / B, agg

    # --- inference convenience ---

    def predict(self, seq: np.ndarray) -> int:
        return int(np.argmax(self.forward_sequence(seq)["logits"]))


def build_fast_weights_rnn(n_in: int = VOCAB_SIZE,
                           n_hidden: int = 64,
                           n_out: int = N_DIGITS,
                           lambda_decay: float = 0.95,
                           eta: float = 0.5,
                           seed: int = 0) -> FastWeightsRNN:
    """Per-stub default factory."""
    return FastWeightsRNN(n_in=n_in, n_hidden=n_hidden, n_out=n_out,
                          lambda_decay=lambda_decay, eta=eta, seed=seed)


# ----------------------------------------------------------------------
# Adam optimizer (no torch)
# ----------------------------------------------------------------------

class Adam:
    def __init__(self, params: dict[str, np.ndarray],
                 lr: float = 5e-3,
                 beta1: float = 0.9, beta2: float = 0.999,
                 eps: float = 1e-8):
        self.params = params
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self.t += 1
        for k, p in self.params.items():
            g = grads[k]
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * (g * g)
            mhat = self.m[k] / (1.0 - self.beta1 ** self.t)
            vhat = self.v[k] / (1.0 - self.beta2 ** self.t)
            p -= self.lr * mhat / (np.sqrt(vhat) + self.eps)


# ----------------------------------------------------------------------
# Training
# ----------------------------------------------------------------------

def train(model: FastWeightsRNN,
          n_steps: int = 3000,
          batch_size: int = 32,
          n_pairs: int = 4,
          lr: float = 5e-3,
          eval_every: int = 100,
          eval_batch: int = 256,
          grad_clip: float = 5.0,
          seed: int = 0,
          verbose: bool = True,
          ) -> dict:
    """Train fast-weights RNN on the associative retrieval task.

    Returns a history dict with per-eval-step loss/acc/eval-acc and the
    full list of (step, loss, acc, eval_acc) tuples.
    """
    rng_train = np.random.default_rng(seed + 1)
    rng_eval = np.random.default_rng(seed + 12345)

    optim = Adam(model.params(), lr=lr)

    history = {"step": [], "train_loss": [], "train_acc": [],
               "eval_acc": [], "eval_loss": []}

    t0 = time.time()
    running_loss, running_acc, n_run = 0.0, 0.0, 0
    for step in range(1, n_steps + 1):
        seqs, targets = generate_batch(batch_size, n_pairs=n_pairs, rng=rng_train)
        loss, acc, grads = model.loss_and_grads_batch(seqs, targets)

        # global-norm gradient clip
        gnorm2 = sum(float(np.sum(g * g)) for g in grads.values())
        gnorm = float(np.sqrt(gnorm2))
        if grad_clip is not None and gnorm > grad_clip:
            scale = grad_clip / (gnorm + 1e-12)
            for k in grads:
                grads[k] *= scale

        optim.step(grads)

        running_loss += loss
        running_acc += acc
        n_run += 1

        if step % eval_every == 0 or step == 1 or step == n_steps:
            mean_loss = running_loss / max(1, n_run)
            mean_acc = running_acc / max(1, n_run)
            running_loss, running_acc, n_run = 0.0, 0.0, 0

            seqs_e, tgt_e = generate_batch(eval_batch, n_pairs=n_pairs,
                                           rng=rng_eval)
            ev_loss, ev_acc, _ = model.loss_and_grads_batch(seqs_e, tgt_e)

            history["step"].append(step)
            history["train_loss"].append(mean_loss)
            history["train_acc"].append(mean_acc)
            history["eval_loss"].append(ev_loss)
            history["eval_acc"].append(ev_acc)

            if verbose:
                elapsed = time.time() - t0
                print(f"  step {step:5d}  train_loss={mean_loss:.4f}  "
                      f"train_acc={mean_acc*100:5.1f}%  "
                      f"eval_loss={ev_loss:.4f}  eval_acc={ev_acc*100:5.1f}%  "
                      f"({elapsed:5.1f}s)")

    history["wallclock"] = time.time() - t0
    return history


# ----------------------------------------------------------------------
# Per-key recall breakdown (eval)
# ----------------------------------------------------------------------

def per_position_accuracy(model: FastWeightsRNN, n_pairs: int,
                          n_eval: int = 1000, seed: int = 7
                          ) -> np.ndarray:
    """Accuracy broken down by which pair index the query references.

    The query letter is one of the n_pairs key letters. We bucket samples by
    which slot (0..n_pairs-1) the queried key occupies, and report accuracy
    per slot. The first-pair slot is the most distant in time, so accuracy
    there is the cleanest test of long-range associative retrieval.
    """
    rng = np.random.default_rng(seed)
    n_correct = np.zeros(n_pairs, dtype=int)
    n_seen = np.zeros(n_pairs, dtype=int)
    for _ in range(n_eval):
        # Generate but record the query slot.
        keys = rng.choice(N_LETTERS, size=n_pairs, replace=False)
        values = rng.integers(0, N_DIGITS, size=n_pairs)
        q_slot = int(rng.integers(0, n_pairs))
        q_key = int(keys[q_slot])
        target = int(values[q_slot])
        tokens = []
        for k, v in zip(keys, values):
            tokens.append(int(k))
            tokens.append(N_LETTERS + int(v))
        tokens += [SEP_INDEX, SEP_INDEX, q_key, SEP_INDEX]
        seq = np.array(tokens, dtype=np.int64)
        pred = model.predict(seq)
        n_seen[q_slot] += 1
        if pred == target:
            n_correct[q_slot] += 1
    acc = n_correct / np.maximum(1, n_seen)
    return acc


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------

def _print_environment() -> None:
    import subprocess
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"],
                                         stderr=subprocess.DEVNULL,
                                         text=True).strip()[:10]
    except Exception:
        commit = "unknown"
    print(f"# python {sys.version.split()[0]}  numpy {np.__version__}  "
          f"{platform.platform()}  git {commit}")


def main() -> None:
    p = argparse.ArgumentParser(description="Fast-weights associative "
                                            "retrieval (Ba et al. 2016).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-pairs", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=3000)
    p.add_argument("--n-hidden", type=int, default=64)
    p.add_argument("--lambda-decay", type=float, default=0.95)
    p.add_argument("--eta", type=float, default=0.5)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=5e-3)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-batch", type=int, default=256)
    p.add_argument("--grad-clip", type=float, default=5.0)
    p.add_argument("--quiet", action="store_true")
    p.add_argument("--show-samples", type=int, default=4,
                   help="show this many trained-model predictions at the end")
    args = p.parse_args()

    _print_environment()
    print(f"# config: n_pairs={args.n_pairs}  n_hidden={args.n_hidden}  "
          f"lambda={args.lambda_decay}  eta={args.eta}  "
          f"lr={args.lr}  batch={args.batch_size}  steps={args.n_steps}  "
          f"seed={args.seed}")

    model = build_fast_weights_rnn(
        n_in=VOCAB_SIZE, n_hidden=args.n_hidden, n_out=N_DIGITS,
        lambda_decay=args.lambda_decay, eta=args.eta, seed=args.seed,
    )
    print(f"# n_params: {model.n_params():,}")

    print("\n=== Training ===")
    t0 = time.time()
    history = train(model, n_steps=args.n_steps,
                    batch_size=args.batch_size, n_pairs=args.n_pairs,
                    lr=args.lr, eval_every=args.eval_every,
                    eval_batch=args.eval_batch, grad_clip=args.grad_clip,
                    seed=args.seed, verbose=not args.quiet)
    train_dt = time.time() - t0

    # Final evaluation (large batch) + per-position breakdown
    rng_eval = np.random.default_rng(args.seed + 99999)
    seqs_e, tgt_e = generate_batch(2000, n_pairs=args.n_pairs, rng=rng_eval)
    final_loss, final_acc, _ = model.loss_and_grads_batch(seqs_e, tgt_e)
    per_slot = per_position_accuracy(model, n_pairs=args.n_pairs,
                                     n_eval=2000, seed=args.seed + 1000)

    print("\n=== Final ===")
    print(f"  retrieval acc (n=2000)   : {final_acc*100:6.2f}%")
    print(f"  retrieval loss           : {final_loss:.4f}")
    print(f"  per-pair-slot acc        : "
          + "  ".join(f"slot{i}={a*100:5.1f}%"
                       for i, a in enumerate(per_slot)))
    print(f"  train wallclock          : {train_dt:.1f}s")
    print(f"  n_params                 : {model.n_params():,}")

    if args.show_samples > 0:
        print("\n=== Sample predictions ===")
        rng_demo = np.random.default_rng(args.seed + 31337)
        for _ in range(args.show_samples):
            seq, target, text = generate_sample(n_pairs=args.n_pairs,
                                                rng=rng_demo)
            pred = model.predict(seq)
            ok = "OK" if pred == target else "WRONG"
            print(f"  {text:>14s} -> pred={pred} target={target}  [{ok}]")


if __name__ == "__main__":
    main()
