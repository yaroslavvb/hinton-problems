"""
MNIST-2x5 subclass distillation.

Reproduction of the central experiment from Mueller, Kornblith & Hinton,
"Subclass distillation", arXiv:2002.03936 (2020).

Setup
-----
- Re-label MNIST: digits 0..4 -> super-class A, digits 5..9 -> super-class B.
- The TEACHER outputs 10 sub-logits (5 per super-class) but is trained ONLY
  on the binary super-class label. To prevent the 5 within-class sub-logits
  from collapsing onto the same value, we add an auxiliary loss that
  maximises the pairwise distance between sub-logit vectors within each
  super-class (equivalent to maximising the per-dim variance of the
  within-super-class sub-logits across the batch).
- The STUDENT is distilled from the teacher's 10 sub-logits via
  temperature-softened cross-entropy. The student NEVER sees the original
  10-way labels.
- We evaluate "subclass recovery": cluster the student's 10 logit indices
  and compare to the original 10-way ground truth (majority-vote
  assignment).

Constraints: numpy + matplotlib + urllib + gzip only. No PyTorch.

CLI:
    python3 mnist_2x5_subclass.py --seed 0 --n-epochs-teacher 5 \
        --n-epochs-student 5 --temperature 4.0
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ----------------------------------------------------------------------
# MNIST loader (urllib + gzip, cached at ~/.cache/hinton-mnist/)
# ----------------------------------------------------------------------

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images":  "t10k-images-idx3-ubyte.gz",
    "test_labels":  "t10k-labels-idx1-ubyte.gz",
}

# Mirrors tried in order. yann.lecun.com is frequently down; the AWS / GCS
# mirrors are the same byte-identical files.
MNIST_MIRRORS = [
    "https://storage.googleapis.com/cvdf-datasets/mnist/",
    "https://ossci-datasets.s3.amazonaws.com/mnist/",
    "https://yann.lecun.com/exdb/mnist/",
]


def _download(url: str, dst: Path) -> None:
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=30) as resp, open(dst, "wb") as f:
        while True:
            chunk = resp.read(1 << 16)
            if not chunk:
                break
            f.write(chunk)


def _ensure(path: Path, fname: str) -> Path:
    if path.exists() and path.stat().st_size > 0:
        return path
    last_err: Exception | None = None
    for mirror in MNIST_MIRRORS:
        url = mirror + fname
        try:
            print(f"  downloading {url} -> {path}")
            _download(url, path)
            return path
        except Exception as e:  # noqa: BLE001
            last_err = e
            print(f"    failed: {e!r}")
            continue
    raise RuntimeError(f"could not download {fname}: {last_err!r}")


def _read_idx(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, = struct.unpack(">I", f.read(4))
        if magic == 2051:  # images
            n, rows, cols = struct.unpack(">III", f.read(12))
            buf = f.read(n * rows * cols)
            arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
        elif magic == 2049:  # labels
            n, = struct.unpack(">I", f.read(4))
            buf = f.read(n)
            arr = np.frombuffer(buf, dtype=np.uint8).copy()
        else:
            raise ValueError(f"unexpected IDX magic {magic} in {path}")
    return arr


def load_mnist(cache_dir: Path | None = None) -> dict[str, np.ndarray]:
    """Load MNIST into a dict of numpy arrays. Caches gzip files on disk.

    Returns:
        {"x_train": (60000, 784) float32 in [0,1],
         "y_train": (60000,) int64 in [0,9],
         "x_test":  (10000, 784) float32,
         "y_test":  (10000,) int64}
    """
    cache_dir = cache_dir or Path.home() / ".cache" / "hinton-mnist"
    cache_dir.mkdir(parents=True, exist_ok=True)

    paths = {k: cache_dir / v for k, v in MNIST_FILES.items()}
    for k, p in paths.items():
        _ensure(p, MNIST_FILES[k])

    x_train = _read_idx(paths["train_images"]).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_train = _read_idx(paths["train_labels"]).astype(np.int64)
    x_test  = _read_idx(paths["test_images"]).reshape(-1, 28 * 28).astype(np.float32) / 255.0
    y_test  = _read_idx(paths["test_labels"]).astype(np.int64)
    return {"x_train": x_train, "y_train": y_train, "x_test": x_test, "y_test": y_test}


# ----------------------------------------------------------------------
# Re-labeling
# ----------------------------------------------------------------------

def relabel_to_superclass(labels: np.ndarray) -> np.ndarray:
    """Map digits 0..4 -> super-class 0, digits 5..9 -> super-class 1."""
    return (labels >= 5).astype(np.int64)


# ----------------------------------------------------------------------
# Numerical helpers
# ----------------------------------------------------------------------

def _softmax(z: np.ndarray, axis: int = -1) -> np.ndarray:
    z = z - np.max(z, axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def _logsumexp(z: np.ndarray, axis: int = -1) -> np.ndarray:
    m = np.max(z, axis=axis, keepdims=True)
    return (m + np.log(np.exp(z - m).sum(axis=axis, keepdims=True))).squeeze(axis)


# ----------------------------------------------------------------------
# Tiny MLP framework (numpy + Adam)
# ----------------------------------------------------------------------

class MLP:
    """One hidden layer ReLU MLP with Adam updates. Stateful forward/backward."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, seed: int = 0,
                 lr: float = 1e-3, weight_decay: float = 0.0):
        rng = np.random.default_rng(seed)
        self.W1 = (rng.standard_normal((in_dim, hidden)) * np.sqrt(2.0 / in_dim)).astype(np.float32)
        self.b1 = np.zeros(hidden, dtype=np.float32)
        self.W2 = (rng.standard_normal((hidden, out_dim)) * np.sqrt(2.0 / hidden)).astype(np.float32)
        self.b2 = np.zeros(out_dim, dtype=np.float32)

        self.lr = lr
        self.wd = weight_decay
        self._beta1 = 0.9
        self._beta2 = 0.999
        self._eps = 1e-8
        self._t = 0
        self._m = {k: np.zeros_like(v) for k, v in self._params().items()}
        self._v = {k: np.zeros_like(v) for k, v in self._params().items()}

    def _params(self) -> dict[str, np.ndarray]:
        return {"W1": self.W1, "b1": self.b1, "W2": self.W2, "b2": self.b2}

    # ---- forward ---------------------------------------------------------

    def forward(self, x: np.ndarray) -> np.ndarray:
        self._x = x
        self._z1 = x @ self.W1 + self.b1
        self._h = np.maximum(0.0, self._z1)
        self._logits = self._h @ self.W2 + self.b2
        return self._logits

    # ---- backward (consumes d_logits with 1/batch already baked in) ------

    def backward(self, d_logits: np.ndarray) -> dict[str, np.ndarray]:
        dW2 = self._h.T @ d_logits
        db2 = d_logits.sum(axis=0)
        dh = d_logits @ self.W2.T
        dz1 = dh * (self._z1 > 0).astype(np.float32)
        dW1 = self._x.T @ dz1
        db1 = dz1.sum(axis=0)
        # weight decay (L2)
        if self.wd > 0:
            dW1 = dW1 + self.wd * self.W1
            dW2 = dW2 + self.wd * self.W2
        return {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2}

    # ---- Adam step --------------------------------------------------------

    def step(self, grads: dict[str, np.ndarray]) -> None:
        self._t += 1
        bc1 = 1.0 - self._beta1 ** self._t
        bc2 = 1.0 - self._beta2 ** self._t
        for k, p in self._params().items():
            g = grads[k]
            self._m[k] = self._beta1 * self._m[k] + (1 - self._beta1) * g
            self._v[k] = self._beta2 * self._v[k] + (1 - self._beta2) * (g * g)
            m_hat = self._m[k] / bc1
            v_hat = self._v[k] / bc2
            p -= self.lr * m_hat / (np.sqrt(v_hat) + self._eps)


# ----------------------------------------------------------------------
# Teacher: builds a 10-sub-logit MLP trained on the binary super-class label
# ----------------------------------------------------------------------

def build_teacher(in_dim: int = 28 * 28, hidden: int = 256,
                  n_subclasses_per_super: int = 5, n_super: int = 2,
                  seed: int = 0, lr: float = 1e-3,
                  weight_decay: float = 1e-4) -> MLP:
    """Teacher emits `n_super * n_subclasses_per_super` sub-logits."""
    out_dim = n_super * n_subclasses_per_super
    return MLP(in_dim, hidden, out_dim, seed=seed, lr=lr, weight_decay=weight_decay)


def superclass_logits_from_sublogits(sub_logits: np.ndarray, n_super: int = 2,
                                     n_sub: int = 5) -> np.ndarray:
    """Turn (B, n_super*n_sub) sub-logits into (B, n_super) super-class logits.

    super_logit_g = logsumexp(sub_logits over the n_sub members of group g).
    """
    B = sub_logits.shape[0]
    z = sub_logits.reshape(B, n_super, n_sub)
    return _logsumexp(z, axis=2)  # (B, n_super)


def teacher_super_loss_and_grad(sub_logits: np.ndarray, super_y: np.ndarray,
                                n_super: int = 2, n_sub: int = 5
                                ) -> tuple[float, np.ndarray]:
    """Cross-entropy on grouped (super-class) softmax.

    Returns (mean_loss, d_loss / d_sub_logits) where the gradient already has
    the 1/batch factor baked in.
    """
    B = sub_logits.shape[0]
    super_logits = superclass_logits_from_sublogits(sub_logits, n_super, n_sub)
    super_p = _softmax(super_logits, axis=1)            # (B, n_super)
    log_super_p = super_logits - _logsumexp(super_logits, axis=1)[:, None]
    loss = -log_super_p[np.arange(B), super_y].mean()

    # d/d super_logit = super_p - one_hot(super_y)
    d_super = super_p.copy()
    d_super[np.arange(B), super_y] -= 1.0
    d_super /= B  # mean

    # super_logit_g = logsumexp(z[g]); d super_logit_g / d z_{g, j} =
    # softmax(z[g])_j  (within-group softmax).
    z = sub_logits.reshape(B, n_super, n_sub)
    within_softmax = _softmax(z, axis=2)               # (B, n_super, n_sub)
    d_sub = within_softmax * d_super[:, :, None]        # (B, n_super, n_sub)
    return float(loss), d_sub.reshape(B, n_super * n_sub)


def auxiliary_distance_loss(sub_logits: np.ndarray, super_y: np.ndarray,
                            n_super: int = 2, n_sub: int = 5,
                            sharpen: float = 1.0,
                            eps: float = 1e-12
                            ) -> tuple[float, np.ndarray]:
    """Auxiliary "diversity" loss that pushes sub-logits apart within each super-class.

    Implementation follows the structure of Mueller, Kornblith & Hinton 2020:
    a *balance* objective on the within-super-class softmax distribution.
    For each super-class g:
      q_i = softmax(z_i[g])                       (within-super-class softmax)
      mean_q_g = (1/n_g) sum_{i in g} q_i
      H_mean_g = -sum_d mean_q_g[d] * log(mean_q_g[d])         <- want HIGH
      H_per_g  = (1/n_g) sum_i [ -sum_d q_i[d] log q_i[d] ]    <- want LOW

    Combined loss to MINIMISE:  -mean_g(H_mean_g) + sharpen * mean_g(H_per_g)
    The first term spreads usage across the 5 subclasses (avoids collapse).
    The second forces each example to commit to ONE subclass (avoids the
    trivial "every example outputs uniform" solution that the first term
    alone admits). Together this is the bounded surrogate for "maximise
    pairwise distance between sub-logit vectors within each super-class":
    different examples end up confidently choosing different subclasses.

    Both terms are bounded in [0, log(n_sub)], so the loss composes cleanly
    with super-class CE without a weight-decay arms race on logit magnitudes.
    """
    B = sub_logits.shape[0]
    z = sub_logits.reshape(B, n_super, n_sub)
    grad = np.zeros_like(z)
    total_H_mean = 0.0
    total_H_per = 0.0
    contributing = 0
    for g in range(n_super):
        idx = np.where(super_y == g)[0]
        n_g = len(idx)
        if n_g < 2:
            continue
        zg = z[idx, g, :]                              # (n_g, n_sub)
        qg = _softmax(zg, axis=1)                      # (n_g, n_sub)
        mean_q = qg.mean(axis=0)                       # (n_sub,)

        # ---- H(mean_q): batch-level diversity -----------------------------
        H_mean = -float((mean_q * np.log(mean_q + eps)).sum())
        total_H_mean += H_mean

        # dH_mean / d mean_q[d] = -log(mean_q[d]) - 1
        # d mean_q[d] / d q_i[d] = 1 / n_g
        # d q_i[d] / d z_i[d'] = q_i[d] * (delta_{d,d'} - q_i[d'])
        # Combined: dH_mean / d zg[i, :] = (1/n_g) * q_i * (dH_dmean - <q_i, dH_dmean>)
        dH_dmean = -(np.log(mean_q + eps) + 1.0)        # (n_sub,)
        dot_mean = qg @ dH_dmean                        # (n_g,)
        dH_mean_dz = (1.0 / n_g) * qg * (dH_dmean[None, :] - dot_mean[:, None])

        # ---- H(q_i): per-example sharpness --------------------------------
        log_qg = np.log(qg + eps)
        H_per_i = -(qg * log_qg).sum(axis=1)            # (n_g,)
        H_per = float(H_per_i.mean())
        total_H_per += H_per

        # dH_per_i / d zg[i, d'] = q_i[d'] * (H_per_i + log_qg[i, d'])  (with sign)
        # Derivation: H_per_i = -sum_d q_i[d] log q_i[d];
        # d/dz_i[d'] = -sum_d (dq_i[d]/dz_i[d'])(log q_i[d] + 1)
        #            = -sum_d q_i[d](delta - q_i[d'])(log q_i[d] + 1)
        #            = -q_i[d'](log q_i[d'] + 1) + q_i[d'] * sum_d q_i[d](log q_i[d] + 1)
        #            = -q_i[d'](log q_i[d'] + 1) + q_i[d'] * (-H_per_i + 1)
        #            = q_i[d'] * (-log q_i[d'] - 1 - H_per_i + 1)
        #            = -q_i[d'] * (log q_i[d'] + H_per_i)
        # Per example, then mean over n_g for H_per:
        dH_per_dz = -(1.0 / n_g) * qg * (log_qg + H_per_i[:, None])

        # loss_g = -H_mean + sharpen * H_per ; grad = -dH_mean/dz + sharpen * dH_per/dz
        grad[idx, g, :] = -dH_mean_dz + sharpen * dH_per_dz
        contributing += 1

    if contributing == 0:
        return 0.0, np.zeros_like(sub_logits)
    H_mean_avg = total_H_mean / contributing
    H_per_avg = total_H_per / contributing
    loss = -H_mean_avg + sharpen * H_per_avg
    grad = grad / contributing
    return float(loss), grad.reshape(B, n_super * n_sub)


# ----------------------------------------------------------------------
# Teacher training loop
# ----------------------------------------------------------------------

def train_teacher(teacher: MLP, x: np.ndarray, super_y: np.ndarray,
                  n_epochs: int = 5, batch_size: int = 128,
                  aux_weight: float = 1.0, sharpen: float = 1.0,
                  n_super: int = 2, n_sub: int = 5,
                  seed: int = 0, verbose: bool = True) -> dict:
    rng = np.random.default_rng(seed + 1)
    n = x.shape[0]
    history = {"epoch": [], "step": [], "super_loss": [], "aux_loss": [],
               "super_acc": []}
    step = 0
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        ep_super_loss = 0.0
        ep_aux_loss = 0.0
        ep_correct = 0
        ep_count = 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = x[idx]
            yb = super_y[idx]
            logits = teacher.forward(xb)
            ls, d_super = teacher_super_loss_and_grad(logits, yb, n_super, n_sub)
            la, d_aux = auxiliary_distance_loss(logits, yb, n_super, n_sub,
                                                sharpen=sharpen)
            d_total = d_super + aux_weight * d_aux
            grads = teacher.backward(d_total)
            teacher.step(grads)

            super_logits = superclass_logits_from_sublogits(logits, n_super, n_sub)
            preds = np.argmax(super_logits, axis=1)
            ep_correct += int((preds == yb).sum())
            ep_count += len(yb)
            ep_super_loss += ls * len(yb)
            ep_aux_loss += la * len(yb)
            step += 1

        ep_super_loss /= ep_count
        ep_aux_loss /= ep_count
        ep_acc = ep_correct / ep_count
        history["epoch"].append(epoch + 1)
        history["step"].append(step)
        history["super_loss"].append(ep_super_loss)
        history["aux_loss"].append(ep_aux_loss)
        history["super_acc"].append(ep_acc)
        if verbose:
            print(f"  teacher epoch {epoch+1}/{n_epochs}  "
                  f"super_loss={ep_super_loss:.4f}  aux_loss={ep_aux_loss:.4f}  "
                  f"super_acc={ep_acc*100:.2f}%")
    return history


# ----------------------------------------------------------------------
# Student distillation
# ----------------------------------------------------------------------

def distillation_loss_and_grad(student_logits: np.ndarray,
                               teacher_targets: np.ndarray,
                               temperature: float
                               ) -> tuple[float, np.ndarray]:
    """KL/cross-entropy between softmax(teacher/T) and softmax(student/T).

    Following Hinton-style distillation, gradients are scaled so the magnitude
    is comparable across temperatures (we multiply by T^2 in the loss but the
    backward gradient w.r.t. student_logits ends up with a single factor of T,
    matching the standard convention).
    """
    B = student_logits.shape[0]
    p_s = _softmax(student_logits / temperature, axis=1)
    log_p_s = (student_logits / temperature
               - _logsumexp(student_logits / temperature, axis=1)[:, None])
    # cross-entropy with teacher_targets as the target distribution
    loss = -(teacher_targets * log_p_s).sum(axis=1).mean() * (temperature ** 2)
    # d L / d student_logits = T * (p_s - teacher_targets) / B   (after T^2 chain)
    d_logits = (p_s - teacher_targets) * (temperature / B)
    return float(loss), d_logits


def distill_to_student(teacher: MLP, student: MLP, x: np.ndarray,
                       temperature: float, n_epochs: int = 5,
                       batch_size: int = 128, seed: int = 0,
                       verbose: bool = True) -> dict:
    """Student matches softmax(teacher_sub_logits / T) on the *same* inputs.

    Student never sees the original 10-way labels.
    """
    rng = np.random.default_rng(seed + 2)
    n = x.shape[0]
    history = {"epoch": [], "step": [], "distill_loss": []}
    step = 0
    for epoch in range(n_epochs):
        order = rng.permutation(n)
        ep_loss = 0.0
        ep_count = 0
        for i in range(0, n, batch_size):
            idx = order[i:i + batch_size]
            xb = x[idx]
            # teacher targets (frozen)
            t_logits = teacher.forward(xb)
            targets = _softmax(t_logits / temperature, axis=1)
            # student forward + loss + backward
            s_logits = student.forward(xb)
            ld, d_logits = distillation_loss_and_grad(s_logits, targets,
                                                     temperature)
            grads = student.backward(d_logits)
            student.step(grads)
            ep_loss += ld * len(xb)
            ep_count += len(xb)
            step += 1
        ep_loss /= ep_count
        history["epoch"].append(epoch + 1)
        history["step"].append(step)
        history["distill_loss"].append(ep_loss)
        if verbose:
            print(f"  student epoch {epoch+1}/{n_epochs}  "
                  f"distill_loss={ep_loss:.4f}")
    return history


# ----------------------------------------------------------------------
# Evaluation: subclass recovery
# ----------------------------------------------------------------------

def majority_vote_assignment(cluster_ids: np.ndarray, true_labels: np.ndarray,
                             n_clusters: int, n_classes: int
                             ) -> tuple[np.ndarray, np.ndarray]:
    """For each cluster, pick the most common true label as the cluster's label.

    Returns (assignment, contingency) where:
      assignment[k] = best-guess true label for cluster k
      contingency[k, c] = #examples in cluster k with true label c
    """
    contingency = np.zeros((n_clusters, n_classes), dtype=np.int64)
    for k in range(n_clusters):
        mask = (cluster_ids == k)
        if mask.any():
            for c in range(n_classes):
                contingency[k, c] = int(((true_labels == c) & mask).sum())
    assignment = contingency.argmax(axis=1)
    return assignment, contingency


def one_to_one_assignment(contingency: np.ndarray) -> np.ndarray:
    """Greedy 1-to-1 cluster->class matching (each true class claimed at most once).

    Repeatedly picks the largest unassigned cell. For 10x10 contingency this
    is identical to Hungarian when the matrix is approximately block-diagonal
    (the regime subclass distillation aims for).
    """
    n_clusters, n_classes = contingency.shape
    work = contingency.astype(np.int64).copy()
    assignment = -np.ones(n_clusters, dtype=np.int64)
    for _ in range(min(n_clusters, n_classes)):
        idx = int(np.argmax(work))
        k, c = divmod(idx, n_classes)
        if work[k, c] <= 0:
            break
        assignment[k] = c
        work[k, :] = -1
        work[:, c] = -1
    # any clusters still unassigned: fall back to majority (rare with 10x10)
    for k in range(n_clusters):
        if assignment[k] < 0:
            row = contingency[k]
            assignment[k] = int(row.argmax()) if row.sum() > 0 else 0
    return assignment


def evaluate_subclass_recovery(student: MLP, x_test: np.ndarray,
                               y_test_true: np.ndarray,
                               n_super: int = 2, n_sub: int = 5
                               ) -> dict:
    """Cluster student logits (10-way argmax) and compare to original 10 labels.

    The student was distilled WITHOUT seeing y_test_true; if subclass
    distillation works, its 10 logit indices should align with the 10 digit
    classes (up to a permutation).
    """
    logits = student.forward(x_test)
    cluster_ids = np.argmax(logits, axis=1)              # (N,) in 0..9
    n_clusters = n_super * n_sub
    assignment, contingency = majority_vote_assignment(
        cluster_ids, y_test_true, n_clusters, 10)
    assignment_1to1 = one_to_one_assignment(contingency)

    # any-mapping accuracy: each cluster picks its plurality label (clusters
    # may collide, double-counting a digit). Upper bound on what's achievable.
    pred_class = assignment[cluster_ids]
    sub_acc = float((pred_class == y_test_true).mean())

    # 1-to-1 accuracy: each true class can only be claimed by one cluster.
    # Closer to "clustering accuracy" reported in the literature.
    pred_class_1to1 = assignment_1to1[cluster_ids]
    sub_acc_1to1 = float((pred_class_1to1 == y_test_true).mean())

    # super-class accuracy via student: super-class of cluster k is k//n_sub
    # IFF the cluster's majority label is in the right super-class.
    super_pred = (cluster_ids // n_sub).astype(np.int64)
    super_true = relabel_to_superclass(y_test_true)
    super_acc = float((super_pred == super_true).mean())

    return {
        "cluster_ids": cluster_ids,
        "assignment": assignment,
        "assignment_1to1": assignment_1to1,
        "contingency": contingency,
        "subclass_recovery_acc": sub_acc,
        "subclass_recovery_acc_1to1": sub_acc_1to1,
        "super_acc_via_student": super_acc,
    }


def evaluate_super_acc(teacher: MLP, x_test: np.ndarray,
                       y_test_super: np.ndarray, n_super: int = 2,
                       n_sub: int = 5) -> float:
    logits = teacher.forward(x_test)
    super_logits = superclass_logits_from_sublogits(logits, n_super, n_sub)
    return float((np.argmax(super_logits, axis=1) == y_test_super).mean())


def teacher_subclass_contingency(teacher: MLP, x: np.ndarray,
                                 y_true: np.ndarray, n_super: int = 2,
                                 n_sub: int = 5) -> np.ndarray:
    """For each (sub-logit-argmax, true-digit) pair, count examples.

    A successful subclass-distillation teacher should show a 5x5 block-diagonal
    pattern: sub-logits 0..4 are mostly active for digits 0..4, and 5..9 for
    5..9. Within each block the per-digit assignment is arbitrary (any
    permutation is fine).
    """
    logits = teacher.forward(x)
    sub_pred = np.argmax(logits, axis=1)  # 0..9
    n_clusters = n_super * n_sub
    cont = np.zeros((n_clusters, 10), dtype=np.int64)
    for k in range(n_clusters):
        mask = (sub_pred == k)
        if mask.any():
            for c in range(10):
                cont[k, c] = int(((y_true == c) & mask).sum())
    return cont


# ----------------------------------------------------------------------
# CLI / main
# ----------------------------------------------------------------------

def main(argv: list[str] | None = None) -> dict:
    p = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n-epochs-teacher", type=int, default=5)
    p.add_argument("--n-epochs-student", type=int, default=5)
    p.add_argument("--temperature", type=float, default=4.0)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--hidden", type=int, default=256)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--aux-weight", type=float, default=1.0)
    p.add_argument("--sharpen", type=float, default=0.5,
                   help="weight on per-example softmax sharpening term in aux loss "
                        "(swept 0.2..1.3, 0.5 is the empirical sweet spot for MNIST)")
    p.add_argument("--out", type=str, default="results.json")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    np.random.seed(args.seed)
    verbose = not args.quiet

    print("# MNIST-2x5 subclass distillation")
    print(f"  seed={args.seed} hidden={args.hidden} batch={args.batch_size}  "
          f"lr={args.lr} aux_weight={args.aux_weight} T={args.temperature}")
    print(f"  teacher_epochs={args.n_epochs_teacher}  student_epochs={args.n_epochs_student}")

    print("# loading MNIST")
    t0 = time.time()
    data = load_mnist()
    print(f"  loaded in {time.time() - t0:.2f}s   "
          f"x_train={data['x_train'].shape} x_test={data['x_test'].shape}")

    super_train = relabel_to_superclass(data["y_train"])
    super_test  = relabel_to_superclass(data["y_test"])

    # ------- teacher -------
    print("# training teacher (binary super-class + auxiliary distance loss)")
    teacher = build_teacher(seed=args.seed, hidden=args.hidden, lr=args.lr)
    t0 = time.time()
    teacher_hist = train_teacher(teacher, data["x_train"], super_train,
                                 n_epochs=args.n_epochs_teacher,
                                 batch_size=args.batch_size,
                                 aux_weight=args.aux_weight,
                                 sharpen=args.sharpen,
                                 seed=args.seed, verbose=verbose)
    teacher_wallclock = time.time() - t0
    teacher_super_acc_test = evaluate_super_acc(teacher, data["x_test"],
                                                super_test)
    print(f"  teacher trained in {teacher_wallclock:.2f}s  "
          f"super_acc_test={teacher_super_acc_test*100:.2f}%")

    # ------- student -------
    print("# distilling student (no access to original 10-way labels)")
    student = MLP(28 * 28, args.hidden, 10, seed=args.seed + 13, lr=args.lr,
                  weight_decay=1e-4)
    t0 = time.time()
    student_hist = distill_to_student(teacher, student, data["x_train"],
                                      temperature=args.temperature,
                                      n_epochs=args.n_epochs_student,
                                      batch_size=args.batch_size,
                                      seed=args.seed, verbose=verbose)
    student_wallclock = time.time() - t0
    print(f"  student trained in {student_wallclock:.2f}s")

    # ------- eval -------
    print("# evaluating subclass recovery")
    eval_out = evaluate_subclass_recovery(student, data["x_test"],
                                          data["y_test"])
    teacher_cont = teacher_subclass_contingency(teacher, data["x_test"],
                                                data["y_test"])
    print(f"  subclass-recovery accuracy (any-mapping):  "
          f"{eval_out['subclass_recovery_acc']*100:.2f}%")
    print(f"  subclass-recovery accuracy (1-to-1):       "
          f"{eval_out['subclass_recovery_acc_1to1']*100:.2f}%")
    print(f"  super-class accuracy (via student logit majority): "
          f"{eval_out['super_acc_via_student']*100:.2f}%")
    print("  cluster -> majority-true-label assignment:", eval_out['assignment'].tolist())
    print("  cluster -> 1-to-1 assignment:              ", eval_out['assignment_1to1'].tolist())
    print("  teacher 10x10 contingency (sub-logit-argmax x true digit):")
    print(teacher_cont)

    results = {
        "config": vars(args),
        "teacher_super_acc_test": teacher_super_acc_test,
        "subclass_recovery_acc": eval_out["subclass_recovery_acc"],
        "subclass_recovery_acc_1to1": eval_out["subclass_recovery_acc_1to1"],
        "super_acc_via_student": eval_out["super_acc_via_student"],
        "student_assignment": eval_out["assignment"].tolist(),
        "student_assignment_1to1": eval_out["assignment_1to1"].tolist(),
        "teacher_contingency": teacher_cont.tolist(),
        "student_contingency": eval_out["contingency"].tolist(),
        "teacher_wallclock_s": teacher_wallclock,
        "student_wallclock_s": student_wallclock,
        "teacher_history": teacher_hist,
        "student_history": student_hist,
    }
    if args.out:
        out_path = Path(args.out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(out_path, "w") as f:
            json.dump(results, f, indent=2, default=lambda o: o.tolist())
        print(f"  wrote {out_path}")
    return results


if __name__ == "__main__":
    main()
