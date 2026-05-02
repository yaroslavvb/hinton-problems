"""
Forward-Forward sequence learning on Aesop's Fables (Hinton 2022, §3.4).

The model is a stack of fully-connected ReLU layers trained with the
Forward-Forward rule -- one layer at a time, no backprop across layers --
to discriminate real 10-character substrings of Aesop's Fables (positives)
from synthetic windows whose final character was wrong (negatives).

Two ways to make the negatives are implemented and compared:

    teacher_forcing
        Take a real 10-char window, keep chars[0..8] (the context) and
        replace char[9] with whatever the *current* model predicts is the
        most likely next character (which is, until convergence, almost
        always wrong). One forward pass per batch.

    self_generated
        Seed the model with the real first 10 chars of each string and roll
        the network forward autoregressively for 90 more characters using
        argmax(goodness) over the 30-symbol alphabet at every step. Sample
        windows that contain at least one generated character as negatives.
        These rollouts can be produced *offline* between training epochs --
        which is the whole point: it shows that the negative ("sleep") phase
        does not need to interleave with the positive ("wake") phase.

Hinton's headline empirical result for this benchmark is that the two
schemes work nearly identically -- supporting the idea that the negative
phase can be decoupled from the positive phase in a biologically-plausible
learning system.

Reference
---------
Hinton (2022), "The Forward-Forward Algorithm: Some Preliminary
Investigations", https://arxiv.org/abs/2212.13345

Constraints
-----------
numpy + matplotlib + imageio/pillow + urllib only. No pytorch / tensorflow.
"""

from __future__ import annotations
import argparse
import os
import time
import urllib.request
from dataclasses import dataclass, field
from typing import Callable, Optional

import numpy as np


# ---------------------------------------------------------------------------
# Corpus
# ---------------------------------------------------------------------------

ALPHABET = list("abcdefghijklmnopqrstuvwxyz ,;.")  # 30 symbols
N_SYMBOLS = len(ALPHABET)
ALPHA_TO_IDX = {c: i for i, c in enumerate(ALPHABET)}

WINDOW = 10        # input window of 10 consecutive characters
STR_LEN = 100      # each training string is 100 characters
N_STRINGS = 248    # how many strings to slice out of the corpus
INPUT_DIM = WINDOW * N_SYMBOLS   # 300

CACHE_DIR = os.path.expanduser("~/.cache/hinton-aesop")
GUTENBERG_URL = "https://www.gutenberg.org/cache/epub/19994/pg19994.txt"
START_MARK = "*** START OF THE PROJECT GUTENBERG EBOOK"
END_MARK = "*** END OF THE PROJECT GUTENBERG EBOOK"


def _download_aesop(cache_dir: str = CACHE_DIR) -> str:
    """Download the Project Gutenberg Aesop's Fables text. Returns the path."""
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, "pg19994.txt")
    if os.path.exists(path) and os.path.getsize(path) > 100_000:
        return path
    print(f"  downloading {GUTENBERG_URL} -> {path}")
    tmp = path + ".part"
    with urllib.request.urlopen(GUTENBERG_URL) as resp, open(tmp, "wb") as f:
        f.write(resp.read())
    os.replace(tmp, path)
    return path


def _strip_gutenberg(text: str) -> str:
    """Strip the Gutenberg header and footer (boilerplate)."""
    start = text.find(START_MARK)
    if start != -1:
        # advance past the rest of that line
        nl = text.find("\n", start)
        if nl != -1:
            text = text[nl + 1:]
    end = text.find(END_MARK)
    if end != -1:
        text = text[:end]
    return text


def _filter_to_alphabet(text: str) -> str:
    """Lowercase and keep only characters in ALPHABET; collapse whitespace."""
    text = text.lower()
    # Replace any whitespace (incl. newlines, tabs) with a single space.
    out_chars = []
    last_was_space = False
    for ch in text:
        if ch.isspace():
            if not last_was_space:
                out_chars.append(" ")
                last_was_space = True
            continue
        if ch in ALPHA_TO_IDX:
            out_chars.append(ch)
            last_was_space = False
        # silently drop anything else (digits, accents, brackets, ...)
    return "".join(out_chars)


def load_aesop_strings(n_strings: int = N_STRINGS, str_len: int = STR_LEN,
                       cache_dir: str = CACHE_DIR) -> tuple[list[str], np.ndarray]:
    """Load 248 fixed-length character strings from Aesop's Fables.

    Returns
    -------
    strings : list of n_strings str, each of length str_len
    indices : (n_strings, str_len) int32 -- alphabet indices
    """
    path = _download_aesop(cache_dir)
    with open(path, "r", encoding="utf-8") as f:
        text = f.read()
    text = _strip_gutenberg(text)
    text = _filter_to_alphabet(text)
    needed = n_strings * str_len
    if len(text) < needed:
        raise RuntimeError(f"After filtering, corpus has {len(text)} chars; "
                           f"need {needed} for {n_strings} x {str_len}")
    sliced = text[:needed]
    strings = [sliced[i * str_len:(i + 1) * str_len] for i in range(n_strings)]
    indices = np.array([[ALPHA_TO_IDX[c] for c in s] for s in strings],
                       dtype=np.int32)
    return strings, indices


# ---------------------------------------------------------------------------
# Encoding
# ---------------------------------------------------------------------------

def encode_windows(window_indices: np.ndarray) -> np.ndarray:
    """One-hot-encode a batch of W-character windows.

    window_indices : (B, W) int  -> (B, W * N_SYMBOLS) float32
    """
    B, W = window_indices.shape
    out = np.zeros((B, W, N_SYMBOLS), dtype=np.float32)
    out[np.arange(B)[:, None], np.arange(W)[None, :], window_indices] = 1.0
    return out.reshape(B, W * N_SYMBOLS)


def all_windows(string_indices: np.ndarray, window: int = WINDOW) -> np.ndarray:
    """Return every length-W substring (sliding by 1) from a (T,) sequence.

    Returns (T - W + 1, W) int.
    """
    T = string_indices.shape[0]
    n_w = T - window + 1
    if n_w <= 0:
        return np.zeros((0, window), dtype=string_indices.dtype)
    out = np.zeros((n_w, window), dtype=string_indices.dtype)
    for w in range(window):
        out[:, w] = string_indices[w:w + n_w]
    return out


# ---------------------------------------------------------------------------
# Forward-Forward layer
# ---------------------------------------------------------------------------

@dataclass
class FFLayer:
    """One ReLU layer trained with the FF rule.

    See ff-label-in-input/ff_label_in_input.py for the gradient derivation;
    this is the same FF mechanic with a different goodness target.
    """
    W: np.ndarray
    b: np.ndarray
    threshold: float
    m_W: np.ndarray = field(default=None)
    v_W: np.ndarray = field(default=None)
    m_b: np.ndarray = field(default=None)
    v_b: np.ndarray = field(default=None)
    step: int = 0

    @classmethod
    def init(cls, in_dim: int, out_dim: int, threshold: float,
             rng: np.random.Generator) -> "FFLayer":
        scale = np.sqrt(2.0 / in_dim)   # Glorot/He
        W = (scale * rng.standard_normal((in_dim, out_dim))).astype(np.float32)
        b = np.zeros(out_dim, dtype=np.float32)
        return cls(W=W, b=b, threshold=threshold,
                   m_W=np.zeros_like(W), v_W=np.zeros_like(W),
                   m_b=np.zeros_like(b), v_b=np.zeros_like(b))

    @staticmethod
    def normalize(h: np.ndarray, eps: float = 1e-8) -> np.ndarray:
        """Hinton's between-layer norm: rescale so mean(h^2) = 1."""
        norm = np.sqrt((h * h).mean(axis=-1, keepdims=True) + eps)
        return h / norm

    def forward(self, x: np.ndarray) -> np.ndarray:
        z = x @ self.W + self.b
        return np.maximum(z, 0.0, out=z)

    def goodness(self, h: np.ndarray) -> np.ndarray:
        return (h * h).mean(axis=-1)


def ff_layer_loss_grad(layer: FFLayer,
                       x_pos: np.ndarray, x_neg: np.ndarray
                       ) -> tuple[float, np.ndarray, np.ndarray, float, float]:
    """Compute FF loss + gradients for one layer.

    Loss (Hinton 2022, eq. 1, averaged over 2*B samples):
        L = log(1 + exp(-(g_pos - theta))) + log(1 + exp(g_neg - theta))
    """
    B, D = x_pos.shape[0], layer.W.shape[1]
    z_pos = x_pos @ layer.W + layer.b
    z_neg = x_neg @ layer.W + layer.b
    h_pos = np.maximum(z_pos, 0.0)
    h_neg = np.maximum(z_neg, 0.0)

    g_pos = (h_pos * h_pos).mean(axis=-1)
    g_neg = (h_neg * h_neg).mean(axis=-1)

    theta = layer.threshold
    a_pos = g_pos - theta
    a_neg = g_neg - theta

    def softplus(x):
        return np.where(x > 0,
                        x + np.log1p(np.exp(-x)),
                        np.log1p(np.exp(x)))

    def sigmoid(x):
        return np.where(x >= 0,
                        1.0 / (1.0 + np.exp(-x)),
                        np.exp(x) / (1.0 + np.exp(x)))

    loss_pos = softplus(-a_pos)
    loss_neg = softplus(a_neg)
    loss = float((loss_pos.sum() + loss_neg.sum()) / (2 * B))

    dg_pos = (sigmoid(a_pos) - 1.0) / (2 * B)
    dg_neg = sigmoid(a_neg) / (2 * B)

    coeff = 2.0 / D
    dh_pos = coeff * h_pos * dg_pos[:, None]
    dh_neg = coeff * h_neg * dg_neg[:, None]

    dz_pos = dh_pos * (z_pos > 0)
    dz_neg = dh_neg * (z_neg > 0)

    grad_W = (x_pos.T @ dz_pos + x_neg.T @ dz_neg).astype(np.float32)
    grad_b = (dz_pos.sum(axis=0) + dz_neg.sum(axis=0)).astype(np.float32)

    return loss, grad_W, grad_b, float(g_pos.mean()), float(g_neg.mean())


def adam_update(layer: FFLayer, gW: np.ndarray, gb: np.ndarray,
                lr: float, beta1: float = 0.9, beta2: float = 0.999,
                eps: float = 1e-8) -> None:
    layer.step += 1
    layer.m_W = beta1 * layer.m_W + (1 - beta1) * gW
    layer.v_W = beta2 * layer.v_W + (1 - beta2) * (gW * gW)
    layer.m_b = beta1 * layer.m_b + (1 - beta1) * gb
    layer.v_b = beta2 * layer.v_b + (1 - beta2) * (gb * gb)
    bc1 = 1 - beta1 ** layer.step
    bc2 = 1 - beta2 ** layer.step
    layer.W -= lr * (layer.m_W / bc1) / (np.sqrt(layer.v_W / bc2) + eps)
    layer.b -= lr * (layer.m_b / bc1) / (np.sqrt(layer.v_b / bc2) + eps)


# ---------------------------------------------------------------------------
# Model
# ---------------------------------------------------------------------------

@dataclass
class FFModel:
    layers: list  # list[FFLayer]
    window: int = WINDOW
    n_symbols: int = N_SYMBOLS

    @classmethod
    def init(cls, layer_sizes: tuple, threshold: float,
             rng: np.random.Generator,
             window: int = WINDOW, n_symbols: int = N_SYMBOLS) -> "FFModel":
        layers = []
        for in_dim, out_dim in zip(layer_sizes[:-1], layer_sizes[1:]):
            layers.append(FFLayer.init(in_dim, out_dim, threshold, rng))
        return cls(layers=layers, window=window, n_symbols=n_symbols)


def build_ff_seq_model(window: int = WINDOW, n_hidden: int = 500,
                       n_layers: int = 3, threshold: float = 2.0,
                       seed: int = 0,
                       layer_sizes: Optional[tuple] = None) -> FFModel:
    """Build an FF sequence model.

    If ``layer_sizes`` is given, it overrides ``window`` / ``n_hidden`` /
    ``n_layers`` -- it must start with the input dim (window*n_symbols).
    Otherwise the model is (input, n_hidden, n_hidden, ..., n_hidden) with
    ``n_layers`` hidden layers.
    """
    rng = np.random.default_rng(seed)
    if layer_sizes is None:
        layer_sizes = (window * N_SYMBOLS,) + (n_hidden,) * n_layers
    return FFModel.init(layer_sizes, threshold, rng,
                        window=window, n_symbols=N_SYMBOLS)


def model_layer_activations(model: FFModel, x: np.ndarray) -> list:
    """Forward through all layers, returning post-ReLU pre-norm activations."""
    acts = []
    h = x
    for layer in model.layers:
        h_relu = layer.forward(h)
        acts.append(h_relu)
        h = FFLayer.normalize(h_relu)
    return acts


def goodness_per_layer(model: FFModel, x: np.ndarray,
                       skip_first: bool = False) -> np.ndarray:
    """(B, n_used) per-sample goodness from each (used) layer."""
    acts = model_layer_activations(model, x)
    if skip_first and len(acts) > 1:
        acts = acts[1:]
    return np.stack([(h * h).mean(axis=-1) for h in acts], axis=1)


def predict_next_char_batch(model: FFModel, contexts: np.ndarray,
                            n_symbols: int = N_SYMBOLS,
                            window: int = WINDOW,
                            chunk: int = 4096) -> np.ndarray:
    """For each (W-1)-char context, pick the next char by argmax(goodness).

    contexts : (B, W-1) int  -- the previous W-1 characters
    returns  : (B,) int      -- predicted next character index
    """
    B = contexts.shape[0]
    W = window
    # Build (B*N, W) candidate windows: context + every possible next char.
    ctx_rep = np.repeat(contexts, n_symbols, axis=0)              # (B*N, W-1)
    cand = np.tile(np.arange(n_symbols), B).astype(np.int32)      # (B*N,)
    full = np.concatenate([ctx_rep, cand[:, None]], axis=1)       # (B*N, W)

    # Chunked goodness eval to keep memory bounded.
    summed = np.zeros(B * n_symbols, dtype=np.float32)
    for s in range(0, B * n_symbols, chunk):
        e = min(s + chunk, B * n_symbols)
        x = encode_windows(full[s:e])
        g = goodness_per_layer(model, x, skip_first=False)
        summed[s:e] = g.sum(axis=1)

    summed = summed.reshape(B, n_symbols)
    return summed.argmax(axis=1).astype(np.int32)


def predict_next_char_distribution(model: FFModel, contexts: np.ndarray
                                   ) -> np.ndarray:
    """Return (B, N) summed-goodness scores for each candidate next char."""
    B = contexts.shape[0]
    ctx_rep = np.repeat(contexts, N_SYMBOLS, axis=0)
    cand = np.tile(np.arange(N_SYMBOLS), B).astype(np.int32)
    full = np.concatenate([ctx_rep, cand[:, None]], axis=1)
    x = encode_windows(full)
    g = goodness_per_layer(model, x, skip_first=False)
    return g.sum(axis=1).reshape(B, N_SYMBOLS)


def sample_next_char(model: FFModel, contexts: np.ndarray,
                     rng: np.random.Generator,
                     temperature: float = 1.0) -> np.ndarray:
    """Sample the next char from softmax(goodness / temperature).

    Argmax (temperature -> 0) tends to collapse the autoregressive rollout
    onto a fixed-point attractor (e.g. always predict space). A non-zero
    temperature keeps the negative distribution broad.
    """
    scores = predict_next_char_distribution(model, contexts)
    if temperature <= 1e-6:
        return scores.argmax(axis=1).astype(np.int32)
    logits = scores / temperature
    logits = logits - logits.max(axis=1, keepdims=True)
    probs = np.exp(logits)
    probs /= probs.sum(axis=1, keepdims=True)
    cum = probs.cumsum(axis=1)
    u = rng.random(contexts.shape[0])[:, None]
    return (cum < u).sum(axis=1).astype(np.int32)


# ---------------------------------------------------------------------------
# Negative generation
# ---------------------------------------------------------------------------

def make_negatives_teacher_forcing(model: FFModel, pos_windows: np.ndarray,
                                   rng: Optional[np.random.Generator] = None
                                   ) -> np.ndarray:
    """Replace the last char of each window with the model's argmax prediction.

    pos_windows : (B, W) int  -- real 10-char substrings
    returns     : (B, W) int  -- contexts unchanged, last char = model.predict
    """
    if rng is None:
        rng = np.random.default_rng()
    contexts = pos_windows[:, :-1]                    # (B, W-1)
    pred = predict_next_char_batch(model, contexts)   # (B,)
    # If the model already predicts the right char, we still keep it so the
    # negative may equal the positive. The FF loss will see g_pos == g_neg
    # for those rows, contributing nothing -- acceptable. Alternatively we
    # could replace with a uniform-random wrong char. We do that here for a
    # tiny bit of extra signal early on.
    same = pred == pos_windows[:, -1]
    if same.any():
        offsets = rng.integers(1, N_SYMBOLS, size=int(same.sum())).astype(np.int32)
        replacement = (pos_windows[same, -1] + offsets) % N_SYMBOLS
        pred = pred.copy()
        pred[same] = replacement
    neg = pos_windows.copy()
    neg[:, -1] = pred
    return neg


def make_negatives_self_generated(model: FFModel, indices: np.ndarray,
                                  window: int = WINDOW,
                                  rng: Optional[np.random.Generator] = None,
                                  temperature: float = 1.0,
                                  ) -> np.ndarray:
    """Fully autoregressive rollout for every string.

    indices : (N, T) int  -- the real strings
    returns : (N, T) int  -- first ``window`` positions are real (the seed);
                             positions ``window..T-1`` are model-generated.

    With ``temperature > 0`` we sample from softmax(goodness / temperature)
    at each step; ``temperature == 0`` is argmax (Hinton's spec but tends to
    collapse onto fixed-point attractors during early training).
    """
    if rng is None:
        rng = np.random.default_rng()
    N, T = indices.shape
    out = indices.copy()
    if T <= window:
        return out
    for t in range(window, T):
        contexts = out[:, t - window + 1:t]
        if temperature <= 1e-6:
            pred = predict_next_char_batch(model, contexts)
        else:
            pred = sample_next_char(model, contexts, rng, temperature)
        out[:, t] = pred
    return out


# ---------------------------------------------------------------------------
# Sampling utilities
# ---------------------------------------------------------------------------

def sample_positive_windows(indices: np.ndarray, batch_size: int,
                            rng: np.random.Generator,
                            window: int = WINDOW) -> np.ndarray:
    """Sample a batch of real W-char windows uniformly at random."""
    N, T = indices.shape
    n_pos = T - window + 1
    string_idx = rng.integers(0, N, size=batch_size)
    pos_idx = rng.integers(0, n_pos, size=batch_size)
    out = np.zeros((batch_size, window), dtype=np.int32)
    for w in range(window):
        out[:, w] = indices[string_idx, pos_idx + w]
    return out


def sample_negative_windows_self(generated_indices: np.ndarray,
                                 batch_size: int,
                                 rng: np.random.Generator,
                                 window: int = WINDOW) -> np.ndarray:
    """Sample windows that contain at least one model-generated character.

    A window starting at position p covers [p, p+W). It contains a generated
    char iff p + W - 1 >= window (i.e. p > 0). For T=100, W=10, valid start
    positions are 1..90 (90 starts per string).
    """
    N, T = generated_indices.shape
    # Valid starts: at least one position >= window in the window.
    # window covers positions [p..p+W-1]. We need p + W - 1 >= window => p >= 1.
    min_start = 1
    max_start = T - window  # inclusive
    string_idx = rng.integers(0, N, size=batch_size)
    pos_idx = rng.integers(min_start, max_start + 1, size=batch_size)
    out = np.zeros((batch_size, window), dtype=np.int32)
    for w in range(window):
        out[:, w] = generated_indices[string_idx, pos_idx + w]
    return out


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

@dataclass
class TrainConfig:
    n_epochs: int = 30
    batch_size: int = 128
    steps_per_epoch: int = 200
    lr: float = 0.003
    threshold: float = 2.0
    layer_sizes: tuple = (INPUT_DIM, 500, 500, 500)
    seed: int = 0
    negatives: str = "teacher_forcing"  # or "self_generated"
    eval_every: int = 1
    rollout_every: int = 1               # for self_generated: regenerate every K epochs
    rollout_temperature: float = 1.0     # softmax temperature for self_generated rollout


def evaluate_per_char_accuracy(model: FFModel, indices: np.ndarray,
                               window: int = WINDOW
                               ) -> tuple[float, np.ndarray]:
    """Per-character next-char accuracy across positions [W..T-1].

    For each string and each position t >= W, predict char[t] from the
    previous W-1 chars and compare to the truth.

    Returns
    -------
    overall_accuracy : float in [0, 1]
    per_position_accuracy : (T - W,) float -- accuracy at each predicted index
    """
    N, T = indices.shape
    if T <= window:
        return 0.0, np.zeros(0, dtype=np.float32)
    # Build all (N * (T-W)) contexts in one go.
    rows, cols = [], []
    for t in range(window, T):
        rows.append(np.arange(N))
        cols.append(np.full(N, t))
    # We collect contexts (N*(T-W), W-1) and targets (N*(T-W),) in order.
    contexts = np.zeros((N * (T - window), window - 1), dtype=np.int32)
    targets = np.zeros(N * (T - window), dtype=np.int32)
    for i, t in enumerate(range(window, T)):
        contexts[i * N:(i + 1) * N] = indices[:, t - window + 1:t]
        targets[i * N:(i + 1) * N] = indices[:, t]
    pred = predict_next_char_batch(model, contexts)
    correct = (pred == targets).astype(np.float32)
    per_pos = correct.reshape(T - window, N).mean(axis=1)
    overall = float(correct.mean())
    return overall, per_pos


def train(model: FFModel,
          indices: np.ndarray,
          cfg: TrainConfig,
          snapshot_callback: Optional[Callable] = None,
          snapshot_every: int = 1,
          verbose: bool = True
          ) -> dict:
    """Train the FF sequence model under the configured negative scheme."""
    rng = np.random.default_rng(cfg.seed)
    N_train, T = indices.shape
    n_layers = len(model.layers)

    history = {
        "epoch": [],
        "loss_per_layer": [[] for _ in range(n_layers)],
        "g_pos_per_layer": [[] for _ in range(n_layers)],
        "g_neg_per_layer": [[] for _ in range(n_layers)],
        "test_acc": [],            # per-char accuracy on `indices`
        "wallclock": [],
        "negatives": cfg.negatives,
    }

    # Pre-generate the first rollout for self_generated.
    if cfg.negatives == "self_generated":
        generated = make_negatives_self_generated(
            model, indices, rng=rng,
            temperature=cfg.rollout_temperature)

    t0 = time.time()
    for epoch in range(cfg.n_epochs):
        epoch_loss = [0.0 for _ in range(n_layers)]
        epoch_gpos = [0.0 for _ in range(n_layers)]
        epoch_gneg = [0.0 for _ in range(n_layers)]

        # Refresh self-generated rollout periodically.
        if cfg.negatives == "self_generated" and epoch > 0 \
                and (epoch % cfg.rollout_every) == 0:
            generated = make_negatives_self_generated(
                model, indices, rng=rng,
                temperature=cfg.rollout_temperature)

        for step in range(cfg.steps_per_epoch):
            pos = sample_positive_windows(indices, cfg.batch_size, rng,
                                          window=model.window)
            if cfg.negatives == "teacher_forcing":
                neg = make_negatives_teacher_forcing(model, pos, rng=rng)
            elif cfg.negatives == "self_generated":
                neg = sample_negative_windows_self(generated, cfg.batch_size,
                                                   rng, window=model.window)
            else:
                raise ValueError(f"unknown negatives mode: {cfg.negatives}")

            x_pos = encode_windows(pos)
            x_neg = encode_windows(neg)

            h_pos, h_neg = x_pos, x_neg
            for L, layer in enumerate(model.layers):
                loss, gW, gb, gp, gn = ff_layer_loss_grad(layer, h_pos, h_neg)
                adam_update(layer, gW, gb, cfg.lr)

                epoch_loss[L] += loss
                epoch_gpos[L] += gp
                epoch_gneg[L] += gn

                z_pos_next = h_pos @ layer.W + layer.b
                z_neg_next = h_neg @ layer.W + layer.b
                h_pos = FFLayer.normalize(np.maximum(z_pos_next, 0.0))
                h_neg = FFLayer.normalize(np.maximum(z_neg_next, 0.0))

        nb = max(cfg.steps_per_epoch, 1)
        for L in range(n_layers):
            history["loss_per_layer"][L].append(epoch_loss[L] / nb)
            history["g_pos_per_layer"][L].append(epoch_gpos[L] / nb)
            history["g_neg_per_layer"][L].append(epoch_gneg[L] / nb)
        history["epoch"].append(epoch + 1)
        history["wallclock"].append(time.time() - t0)

        if (epoch + 1) % cfg.eval_every == 0 or epoch == cfg.n_epochs - 1:
            acc, _ = evaluate_per_char_accuracy(model, indices,
                                                window=model.window)
            history["test_acc"].append(acc)
        else:
            history["test_acc"].append(history["test_acc"][-1]
                                       if history["test_acc"]
                                       else float("nan"))

        if verbose:
            losses = " ".join(f"L{L}={epoch_loss[L]/nb:.3f}"
                              for L in range(n_layers))
            print(f"epoch {epoch + 1:3d}/{cfg.n_epochs}  "
                  f"{losses}  "
                  f"acc={history['test_acc'][-1]*100:.1f}%  "
                  f"({history['wallclock'][-1]:.1f}s)")

        if snapshot_callback is not None and (epoch + 1) % snapshot_every == 0:
            snapshot_callback(epoch, model, history)

    return history


# ---------------------------------------------------------------------------
# Baselines
# ---------------------------------------------------------------------------

def random_fixed_hidden_baseline(indices: np.ndarray,
                                 window: int = WINDOW,
                                 n_hidden: int = 500,
                                 n_layers: int = 3,
                                 seed: int = 0
                                 ) -> tuple[float, np.ndarray]:
    """Per-char accuracy with an *untrained* random FF stack (no learning).

    Confirms that representation learning is essential -- if random hidden
    layers already gave high accuracy, the FF training would not be doing
    real work.
    """
    model = build_ff_seq_model(window=window, n_hidden=n_hidden,
                               n_layers=n_layers, seed=seed)
    return evaluate_per_char_accuracy(model, indices, window=window)


def unigram_baseline(indices: np.ndarray, window: int = WINDOW
                     ) -> tuple[float, np.ndarray]:
    """Always predict the corpus's most common character. Simple sanity-check."""
    counts = np.bincount(indices.reshape(-1), minlength=N_SYMBOLS)
    most_common = int(counts.argmax())
    N, T = indices.shape
    targets = indices[:, window:]                 # (N, T - W)
    correct = (targets == most_common).astype(np.float32)
    per_pos = correct.mean(axis=0)
    return float(correct.mean()), per_pos


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_run(path: str, model: FFModel, history: dict, cfg: TrainConfig
             ) -> None:
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    weights = {f"layer{i}_W": L.W for i, L in enumerate(model.layers)}
    biases = {f"layer{i}_b": L.b for i, L in enumerate(model.layers)}
    np.savez(path,
             layer_sizes=np.array([L.W.shape[0] for L in model.layers]
                                  + [model.layers[-1].W.shape[1]],
                                  dtype=np.int32),
             threshold=np.float32(cfg.threshold),
             seed=np.int32(cfg.seed),
             negatives=np.array(cfg.negatives),
             history_test_acc=np.array(history["test_acc"], dtype=np.float32),
             history_loss_per_layer=np.array(history["loss_per_layer"],
                                             dtype=np.float32),
             history_g_pos=np.array(history["g_pos_per_layer"],
                                    dtype=np.float32),
             history_g_neg=np.array(history["g_neg_per_layer"],
                                    dtype=np.float32),
             history_wallclock=np.array(history["wallclock"], dtype=np.float32),
             **weights, **biases)


def load_saved_model(path: str) -> tuple[FFModel, dict, str]:
    npz = np.load(path, allow_pickle=False)
    layer_sizes = tuple(int(x) for x in npz["layer_sizes"])
    threshold = float(npz["threshold"])
    seed = int(npz["seed"])
    negatives = str(npz["negatives"])
    model = build_ff_seq_model(seed=seed,
                               threshold=threshold,
                               layer_sizes=layer_sizes)
    for i, layer in enumerate(model.layers):
        layer.W = npz[f"layer{i}_W"].astype(np.float32)
        layer.b = npz[f"layer{i}_b"].astype(np.float32)
    history = {
        "epoch": list(range(1, len(npz["history_test_acc"]) + 1)),
        "test_acc": list(npz["history_test_acc"]),
        "loss_per_layer": [list(row) for row in npz["history_loss_per_layer"]],
        "g_pos_per_layer": [list(row) for row in npz["history_g_pos"]],
        "g_neg_per_layer": [list(row) for row in npz["history_g_neg"]],
        "wallclock": list(npz["history_wallclock"]),
        "negatives": negatives,
    }
    return model, history, negatives


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--negatives", type=str, default="teacher_forcing",
                   choices=["teacher_forcing", "self_generated"])
    p.add_argument("--n-epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--steps-per-epoch", type=int, default=200)
    p.add_argument("--lr", type=float, default=0.003)
    p.add_argument("--threshold", type=float, default=2.0)
    p.add_argument("--layer-sizes", type=str, default="300,500,500,500",
                   help="Comma-separated; first must be 300 (window*30).")
    p.add_argument("--eval-every", type=int, default=1)
    p.add_argument("--rollout-every", type=int, default=1)
    p.add_argument("--rollout-temperature", type=float, default=1.0,
                   help="Softmax temperature for self_generated rollout. "
                        "0 = argmax (collapses to attractors), 1 = full sample.")
    p.add_argument("--save", type=str, default=None,
                   help="Save model + history to this .npz path.")
    p.add_argument("--baseline", action="store_true",
                   help="After training, also report random-fixed-hidden + unigram.")
    args = p.parse_args()

    layer_sizes = tuple(int(x) for x in args.layer_sizes.split(","))
    if layer_sizes[0] != INPUT_DIM:
        raise ValueError(f"layer_sizes[0] must be {INPUT_DIM} (window*alphabet)")

    cfg = TrainConfig(n_epochs=args.n_epochs,
                      batch_size=args.batch_size,
                      steps_per_epoch=args.steps_per_epoch,
                      lr=args.lr,
                      threshold=args.threshold,
                      layer_sizes=layer_sizes,
                      seed=args.seed,
                      negatives=args.negatives,
                      eval_every=args.eval_every,
                      rollout_every=args.rollout_every,
                      rollout_temperature=args.rollout_temperature)

    print(f"Loading Aesop from {CACHE_DIR} ...")
    strings, indices = load_aesop_strings()
    print(f"  loaded {indices.shape[0]} strings of {indices.shape[1]} chars "
          f"each (alphabet size {N_SYMBOLS})")

    model = build_ff_seq_model(seed=args.seed,
                               threshold=args.threshold,
                               layer_sizes=layer_sizes)
    n_params = sum(L.W.size + L.b.size for L in model.layers)
    print(f"Model: {len(model.layers)} layers, sizes={layer_sizes}, "
          f"params={n_params:,}, threshold={args.threshold}, "
          f"negatives={args.negatives}")

    print("Training...")
    history = train(model, indices, cfg, verbose=True)

    if args.baseline:
        print("\nBaselines:")
        rand_acc, _ = random_fixed_hidden_baseline(indices,
                                                   window=WINDOW,
                                                   n_hidden=layer_sizes[1],
                                                   n_layers=len(layer_sizes) - 1,
                                                   seed=args.seed)
        uni_acc, _ = unigram_baseline(indices, window=WINDOW)
        print(f"  random fixed hidden : {rand_acc * 100:.2f}%")
        print(f"  unigram (most common): {uni_acc * 100:.2f}%  "
              f"(predicts '{ALPHABET[int(np.bincount(indices.reshape(-1), minlength=N_SYMBOLS).argmax())]}')")
        print(f"  trained FF ({args.negatives}): {history['test_acc'][-1] * 100:.2f}%")

    if args.save:
        save_run(args.save, model, history, cfg)
        print(f"Saved model + history to {args.save}")


if __name__ == "__main__":
    main()
