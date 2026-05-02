"""
Adaptive mixture of local experts on Peterson-Barney vowels.

Replicates the headline experiment of Jacobs, Jordan, Nowlan & Hinton (1991),
"Adaptive mixtures of local experts", Neural Computation 3(1):79-87.

Task: 4-class speaker-independent vowel classification ([i], [I], [a], [Lambda])
from F1 and F2 formant frequencies.  Data: Peterson & Barney (1952), 76 speakers
(33 men, 28 women, 15 children) x 10 vowels x 2 repetitions.

Compares:
  * `MoE`  - mixture of K linear softmax experts with a softmax gate; trained
             end-to-end by maximum-likelihood gradient descent on
             p(y|x) = sum_k g_k(x) * p_k(y|x).
  * `MLP`  - monolithic 1-hidden-layer tanh MLP, parameter-matched, trained by
             standard backprop with cross-entropy.

Numpy + urllib only.  No torch, no sklearn.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import ssl
import subprocess
import sys
import time
import urllib.request
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

CACHE_DIR = Path.home() / ".cache" / "hinton-vowels"
CACHE_FILE = CACHE_DIR / "PetersonBarney.dat"

# Mirror chain.  The Hillenbrand WMU page redirects to a CMS landing page
# (not the raw data) so we hit the phiresky mirror first; users with a clean
# Hillenbrand URL can drop the file into ~/.cache/hinton-vowels/ directly.
DATA_URLS = (
    "https://raw.githubusercontent.com/phiresky/neural-network-demo/"
    "65437c7f70682ab43555a82c84ed12671f199c43/lib/peterson_barney_data",
)

# Phoneme codes used in the Peterson-Barney file for the four vowels in
# Jacobs et al. 1991 (heed / hid / hod / hud).
VOWEL_CODES = ("IY", "IH", "AA", "AH")
VOWEL_LABELS = ("[i]", "[I]", "[a]", "[Lambda]")


def _download(url: str, dest: Path) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context()
    # Some corporate proxies break cert verification; fall back rather than
    # silently disabling verification on the trusted call.
    try:
        with urllib.request.urlopen(url, context=ctx, timeout=20) as r:
            data = r.read()
    except Exception:
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx, timeout=20) as r:
            data = r.read()
    dest.write_bytes(data)


def _parse_pb_dat(text: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Parse the Peterson-Barney text file.

    Each non-comment line is whitespace-separated:
        speaker_type speaker_id phoneme_id phoneme_label F0 F1 F2 F3
    The phoneme_label may be prefixed with '*' to mark listener disagreement;
    we keep all tokens (the original Jacobs setup did not filter these).
    """
    feats: list[list[float]] = []
    labels: list[int] = []
    speakers: list[int] = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 8:
            continue
        try:
            stype = int(parts[0])
            sid = int(parts[1])
        except ValueError:
            continue
        tag = parts[3].lstrip("*")
        if tag not in VOWEL_CODES:
            continue
        try:
            f1 = float(parts[5])
            f2 = float(parts[6])
        except ValueError:
            continue
        feats.append([f1, f2])
        labels.append(VOWEL_CODES.index(tag))
        # Pack (speaker_type, speaker_id) into a single integer so the
        # by-speaker split treats men #1 and women #1 as different speakers.
        speakers.append(stype * 1000 + sid)
    if not feats:
        raise RuntimeError("No matching vowel tokens parsed from data file")
    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(speakers, dtype=np.int64),
    )


def _synthesize_pb(rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fallback if no network: Gaussian-mixture mock with class-typical centres.

    Centres are taken from Peterson & Barney 1952 male means (Hz).  The result
    is *not* the real corpus and should never be used to claim the headline; we
    still get to exercise the code path.  Documented as a deviation in README.
    """
    centres = {
        "IY": (270.0, 2290.0),
        "IH": (390.0, 1990.0),
        "AA": (730.0, 1090.0),
        "AH": (640.0, 1190.0),
    }
    sigma = (60.0, 200.0)
    n_speakers = 76
    feats: list[list[float]] = []
    labels: list[int] = []
    speakers: list[int] = []
    for s in range(n_speakers):
        # Per-speaker offset so the synthetic data still has speaker structure.
        off1 = rng.normal(0.0, 80.0)
        off2 = rng.normal(0.0, 250.0)
        for cls, code in enumerate(VOWEL_CODES):
            mu = centres[code]
            for _rep in range(2):
                f1 = rng.normal(mu[0] + off1, sigma[0])
                f2 = rng.normal(mu[1] + off2, sigma[1])
                feats.append([max(80.0, f1), max(300.0, f2)])
                labels.append(cls)
                speakers.append(1000 + s)
    return (
        np.asarray(feats, dtype=np.float64),
        np.asarray(labels, dtype=np.int64),
        np.asarray(speakers, dtype=np.int64),
    )


def load_peterson_barney(allow_download: bool = True) -> Tuple[
    np.ndarray, np.ndarray, np.ndarray, bool
]:
    """Return (X, y, speakers, is_real).

    X: (N, 2) F1, F2 in Hz.
    y: (N,) int in {0..3} matching VOWEL_CODES order.
    speakers: (N,) speaker id (packed (type, id)).
    is_real: True if loaded from the real Peterson-Barney file, False if
        synthesized.
    """
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    if not CACHE_FILE.exists() and allow_download:
        for url in DATA_URLS:
            try:
                _download(url, CACHE_FILE)
                break
            except Exception as e:
                sys.stderr.write(f"  download failed: {url}: {e}\n")
                continue
    if CACHE_FILE.exists():
        try:
            text = CACHE_FILE.read_text(errors="replace")
            X, y, sp = _parse_pb_dat(text)
            return X, y, sp, True
        except Exception as e:
            sys.stderr.write(f"  parse failed ({e}); falling back to synthetic\n")
    rng = np.random.default_rng(0)
    X, y, sp = _synthesize_pb(rng)
    return X, y, sp, False


# ---------------------------------------------------------------------------
# Train / test split, by speaker
# ---------------------------------------------------------------------------


def split_by_speaker(
    X: np.ndarray, y: np.ndarray, speakers: np.ndarray, train_frac: float, seed: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    uniq = np.unique(speakers)
    rng.shuffle(uniq)
    n_train = int(round(len(uniq) * train_frac))
    train_set = set(uniq[:n_train].tolist())
    train_mask = np.array([s in train_set for s in speakers])
    return (
        X[train_mask],
        y[train_mask],
        X[~train_mask],
        y[~train_mask],
    )


def standardise(
    X_train: np.ndarray, X_test: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    mu = X_train.mean(axis=0)
    sd = X_train.std(axis=0) + 1e-9
    return (X_train - mu) / sd, (X_test - mu) / sd, mu, sd


# ---------------------------------------------------------------------------
# Numerics helpers
# ---------------------------------------------------------------------------


def softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def log_softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = z.max(axis=axis, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=axis, keepdims=True))


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


@dataclass
class MoE:
    """Mixture of K linear softmax experts with a linear softmax gate.

    p(y|x) = sum_k g_k(x) * p_k(y|x)
    g(x)  = softmax(W_g x + b_g)            shape (K,)
    p_k(.|x) = softmax(W_k x + b_k)         shape (n_classes,)
    """

    n_in: int
    n_classes: int
    n_experts: int
    W_e: np.ndarray  # (K, n_classes, n_in)
    b_e: np.ndarray  # (K, n_classes)
    W_g: np.ndarray  # (K, n_in)
    b_g: np.ndarray  # (K,)

    @classmethod
    def init(cls, n_in: int, n_classes: int, n_experts: int, seed: int) -> "MoE":
        rng = np.random.default_rng(seed)
        # Small but non-zero init so experts start with distinguishable
        # decision boundaries.  Symmetry breaking is essential -- if all
        # experts begin identical and the gate is zero, nothing will diverge.
        return cls(
            n_in=n_in,
            n_classes=n_classes,
            n_experts=n_experts,
            W_e=rng.normal(0.0, 0.3, size=(n_experts, n_classes, n_in)),
            b_e=np.zeros((n_experts, n_classes)),
            W_g=rng.normal(0.0, 0.3, size=(n_experts, n_in)),
            b_g=np.zeros((n_experts,)),
        )

    @property
    def n_params(self) -> int:
        return (
            self.W_e.size + self.b_e.size + self.W_g.size + self.b_g.size
        )

    def expert_logits(self, X: np.ndarray) -> np.ndarray:
        # X: (B, n_in) -> z: (B, K, n_classes)
        return np.einsum("bi,kci->bkc", X, self.W_e) + self.b_e[None, :, :]

    def gate_logits(self, X: np.ndarray) -> np.ndarray:
        # X: (B, n_in) -> (B, K)
        return X @ self.W_g.T + self.b_g[None, :]

    def predict(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Return (mixture probs (B, n_classes), gate (B, K), expert probs (B, K, n_classes))."""
        z_e = self.expert_logits(X)
        p_e = softmax(z_e, axis=-1)
        z_g = self.gate_logits(X)
        g = softmax(z_g, axis=-1)
        # mixture: sum over k of g_k * p_k(c)
        p_mix = np.einsum("bk,bkc->bc", g, p_e)
        return p_mix, g, p_e

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, dict, dict]:
        """Negative log-likelihood and gradients (averaged over the batch).

        Derivation
        ----------
        L_n = -log sum_k g_k * p_k[y_n]
        h_{nk} = g_k * p_k[y_n] / sum_j g_j * p_j[y_n]   (posterior over experts)
        dL_n / dz_e_{kc} = h_{nk} * (p_k[c] - 1[c == y_n])
        dL_n / dz_g_{k}  = g_k - h_{nk}
        """
        B = X.shape[0]
        p_mix, g, p_e = self.predict(X)
        # Probability the expert assigns to the true class:  p_e[b, k, y_b]
        idx = np.arange(B)
        p_e_true = p_e[idx, :, y]  # (B, K)
        # Mixture probability of the true class.
        p_true = (g * p_e_true).sum(axis=1)  # (B,)
        eps = 1e-12
        loss = -np.log(p_true + eps).mean()
        # Posterior responsibility of each expert for each example.
        h = (g * p_e_true) / (p_true[:, None] + eps)  # (B, K)
        # Expert gradients: standard cross-entropy delta scaled by h.
        # dL/dz_e[b,k,c] = h[b,k] * (p_e[b,k,c] - 1[c==y_b])
        dz_e = h[:, :, None] * p_e  # (B, K, n_classes)
        dz_e[idx, :, y] -= h
        dW_e = np.einsum("bkc,bi->kci", dz_e, X) / B
        db_e = dz_e.mean(axis=0)
        # Gate gradient: (g - h), classic cross-entropy form with target h.
        dz_g = (g - h) / B  # (B, K)
        dW_g = dz_g.T @ X
        db_g = dz_g.sum(axis=0)
        grads = dict(W_e=dW_e, b_e=db_e, W_g=dW_g, b_g=db_g)
        diag = dict(p_mix=p_mix, g=g, p_e=p_e, h=h)
        return float(loss), grads, diag


@dataclass
class MLP:
    """Monolithic 1-hidden-layer tanh MLP with softmax cross-entropy.

    Used as the parameter-matched baseline ("monolithic backprop").
    """

    n_in: int
    n_hidden: int
    n_classes: int
    W1: np.ndarray  # (n_hidden, n_in)
    b1: np.ndarray  # (n_hidden,)
    W2: np.ndarray  # (n_classes, n_hidden)
    b2: np.ndarray  # (n_classes,)

    @classmethod
    def init(cls, n_in: int, n_hidden: int, n_classes: int, seed: int) -> "MLP":
        rng = np.random.default_rng(seed)
        return cls(
            n_in=n_in,
            n_hidden=n_hidden,
            n_classes=n_classes,
            W1=rng.normal(0.0, 1.0 / np.sqrt(n_in), size=(n_hidden, n_in)),
            b1=np.zeros(n_hidden),
            W2=rng.normal(0.0, 1.0 / np.sqrt(n_hidden), size=(n_classes, n_hidden)),
            b2=np.zeros(n_classes),
        )

    @property
    def n_params(self) -> int:
        return self.W1.size + self.b1.size + self.W2.size + self.b2.size

    def predict(self, X: np.ndarray) -> np.ndarray:
        h = np.tanh(X @ self.W1.T + self.b1)
        z = h @ self.W2.T + self.b2
        return softmax(z, axis=-1)

    def loss_and_grads(self, X: np.ndarray, y: np.ndarray) -> Tuple[float, dict, dict]:
        B = X.shape[0]
        h_pre = X @ self.W1.T + self.b1
        h = np.tanh(h_pre)
        z = h @ self.W2.T + self.b2
        p = softmax(z, axis=-1)
        idx = np.arange(B)
        loss = -np.log(p[idx, y] + 1e-12).mean()
        dz = p.copy()
        dz[idx, y] -= 1.0
        dz /= B
        dW2 = dz.T @ h
        db2 = dz.sum(axis=0)
        dh = dz @ self.W2
        dh_pre = dh * (1.0 - h * h)
        dW1 = dh_pre.T @ X
        db1 = dh_pre.sum(axis=0)
        return float(loss), dict(W1=dW1, b1=db1, W2=dW2, b2=db2), dict(p=p)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def accuracy(model, X: np.ndarray, y: np.ndarray) -> float:
    if isinstance(model, MoE):
        p, _g, _pe = model.predict(X)
    else:
        p = model.predict(X)
    return float((p.argmax(axis=1) == y).mean())


def sgd_step(model, grads: dict, lr: float) -> None:
    for k, g in grads.items():
        cur = getattr(model, k)
        cur -= lr * g


def train(
    model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    n_epochs: int,
    lr: float,
    batch_size: int,
    seed: int,
    record_snapshots: bool = False,
) -> dict:
    """Simple mini-batch SGD with reshuffling each epoch."""
    rng = np.random.default_rng(seed)
    N = X_train.shape[0]
    history = dict(epoch=[], train_loss=[], train_acc=[], test_acc=[])
    snapshots = [] if record_snapshots else None
    for epoch in range(n_epochs):
        perm = rng.permutation(N)
        X_sh = X_train[perm]
        y_sh = y_train[perm]
        epoch_loss = 0.0
        n_batches = 0
        for start in range(0, N, batch_size):
            xb = X_sh[start:start + batch_size]
            yb = y_sh[start:start + batch_size]
            loss, grads, _ = model.loss_and_grads(xb, yb)
            sgd_step(model, grads, lr)
            epoch_loss += loss
            n_batches += 1
        train_acc = accuracy(model, X_train, y_train)
        test_acc = accuracy(model, X_test, y_test)
        history["epoch"].append(epoch)
        history["train_loss"].append(epoch_loss / max(1, n_batches))
        history["train_acc"].append(train_acc)
        history["test_acc"].append(test_acc)
        if snapshots is not None:
            snapshots.append(_snapshot(model))
    return dict(history=history, snapshots=snapshots)


def _snapshot(model) -> dict:
    if isinstance(model, MoE):
        return dict(
            kind="moe",
            W_e=model.W_e.copy(),
            b_e=model.b_e.copy(),
            W_g=model.W_g.copy(),
            b_g=model.b_g.copy(),
        )
    return dict(
        kind="mlp",
        W1=model.W1.copy(),
        b1=model.b1.copy(),
        W2=model.W2.copy(),
        b2=model.b2.copy(),
    )


# ---------------------------------------------------------------------------
# Required by the wave-8 spec
# ---------------------------------------------------------------------------


def build_moe(n_experts: int = 4, n_in: int = 2, n_out: int = 4, seed: int = 0) -> MoE:
    """Spec-required factory.  Wraps `MoE.init` for a stable public name."""
    return MoE.init(n_in=n_in, n_classes=n_out, n_experts=n_experts, seed=seed)


def visualize_partitioning(model: MoE, X: np.ndarray, y: np.ndarray, path: str) -> None:
    """Render the gate's argmax partition over a 2-D F1/F2 grid.

    Standalone wrapper so the visualizer module doesn't have to be imported here.
    """
    # Local import to avoid a hard matplotlib dependency for code paths that
    # only need the model/data.
    from visualize_vowel_mixture_experts import plot_partitioning

    plot_partitioning(model, X, y, path)


# ---------------------------------------------------------------------------
# Environment / reproducibility
# ---------------------------------------------------------------------------


def env_info() -> dict:
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except Exception:
        commit = ""
    return dict(
        python=sys.version.split()[0],
        numpy=np.__version__,
        platform=platform.platform(),
        processor=platform.processor(),
        git_commit=commit,
    )


# ---------------------------------------------------------------------------
# Headline experiment
# ---------------------------------------------------------------------------


def headline(
    seed: int,
    n_experts: int,
    n_epochs: int,
    lr: float,
    batch_size: int,
    train_frac: float,
    save_results: Path | None,
) -> dict:
    X, y, speakers, is_real = load_peterson_barney()
    Xtr_raw, ytr, Xte_raw, yte = split_by_speaker(X, y, speakers, train_frac, seed=seed)
    Xtr, Xte, mu, sd = standardise(Xtr_raw, Xte_raw)

    moe = build_moe(n_experts=n_experts, n_in=2, n_out=4, seed=seed)
    # Match the MoE parameter count for a fair "monolithic backprop" baseline.
    # MoE params:  K * (C*I + C) + (K*I + K)  with I=2, C=4  ->  K=4 -> 60 params.
    # MLP params: 2*H + H + H*4 + 4 = 7H + 4.  H = round((moe_params - 4) / 7).
    n_hidden = max(2, int(round((moe.n_params - 4) / 7)))
    mlp = MLP.init(n_in=2, n_hidden=n_hidden, n_classes=4, seed=seed)

    t0 = time.perf_counter()
    moe_run = train(moe, Xtr, ytr, Xte, yte, n_epochs, lr, batch_size, seed)
    t_moe = time.perf_counter() - t0

    t0 = time.perf_counter()
    mlp_run = train(mlp, Xtr, ytr, Xte, yte, n_epochs, lr, batch_size, seed)
    t_mlp = time.perf_counter() - t0

    def epochs_to_threshold(history: dict, thresh: float) -> int | None:
        for e, a in zip(history["epoch"], history["test_acc"]):
            if a >= thresh:
                return int(e)
        return None

    moe_e90 = epochs_to_threshold(moe_run["history"], 0.90)
    mlp_e90 = epochs_to_threshold(mlp_run["history"], 0.90)

    config = dict(
        seed=seed,
        n_experts=n_experts,
        n_epochs=n_epochs,
        lr=lr,
        batch_size=batch_size,
        train_frac=train_frac,
        n_hidden_baseline=n_hidden,
        n_train=int(Xtr.shape[0]),
        n_test=int(Xte.shape[0]),
        moe_params=int(moe.n_params),
        mlp_params=int(mlp.n_params),
        is_real_data=bool(is_real),
        feat_mean=mu.tolist(),
        feat_std=sd.tolist(),
    )
    summary = dict(
        moe_test_acc=float(moe_run["history"]["test_acc"][-1]),
        mlp_test_acc=float(mlp_run["history"]["test_acc"][-1]),
        moe_train_acc=float(moe_run["history"]["train_acc"][-1]),
        mlp_train_acc=float(mlp_run["history"]["train_acc"][-1]),
        moe_epochs_to_90=moe_e90,
        mlp_epochs_to_90=mlp_e90,
        moe_wallclock_s=t_moe,
        mlp_wallclock_s=t_mlp,
    )
    out = dict(
        config=config,
        env=env_info(),
        summary=summary,
        moe_history=moe_run["history"],
        mlp_history=mlp_run["history"],
    )
    if save_results is not None:
        # Save model snapshots separately as .npz so results.json stays small.
        np.savez(
            save_results.with_suffix(".npz"),
            W_e=moe.W_e, b_e=moe.b_e, W_g=moe.W_g, b_g=moe.b_g,
            W1=mlp.W1, b1=mlp.b1, W2=mlp.W2, b2=mlp.b2,
            X_train=Xtr, y_train=ytr, X_test=Xte, y_test=yte,
            feat_mean=mu, feat_std=sd,
        )
        save_results.write_text(json.dumps(out, indent=2))
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--n-experts", type=int, default=4)
    parser.add_argument("--n-epochs", type=int, default=80)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--train-frac", type=float, default=0.75)
    parser.add_argument(
        "--results",
        type=str,
        default="results.json",
        help="Path to write headline results.  Use '' to skip writing.",
    )
    args = parser.parse_args(argv)
    out_path = Path(args.results) if args.results else None
    res = headline(
        seed=args.seed,
        n_experts=args.n_experts,
        n_epochs=args.n_epochs,
        lr=args.lr,
        batch_size=args.batch_size,
        train_frac=args.train_frac,
        save_results=out_path,
    )
    s = res["summary"]
    c = res["config"]
    print(f"data: {'real Peterson-Barney' if c['is_real_data'] else 'SYNTHETIC fallback'}; "
          f"train={c['n_train']} test={c['n_test']}")
    print(f"params: MoE={c['moe_params']}  MLP(hidden={c['n_hidden_baseline']})={c['mlp_params']}")
    print(f"MoE   final test acc: {s['moe_test_acc']:.3f}   epochs->90% : {s['moe_epochs_to_90']}   "
          f"wallclock {s['moe_wallclock_s']:.2f}s")
    print(f"MLP   final test acc: {s['mlp_test_acc']:.3f}   epochs->90% : {s['mlp_epochs_to_90']}   "
          f"wallclock {s['mlp_wallclock_s']:.2f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
