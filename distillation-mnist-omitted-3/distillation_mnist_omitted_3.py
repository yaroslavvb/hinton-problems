"""
MNIST knowledge distillation with the digit "3" omitted from the student's
training set.

Source: Hinton, Vinyals & Dean (2015), "Distilling the knowledge in a neural
network", NIPS Deep Learning Workshop. Section 3 reports that a student trained
on a transfer set with no examples of digit "3" can still classify test 3s at
98.6% accuracy after a single-bias correction, by matching the teacher's soft
targets at high temperature.

Pipeline:
  1. Download MNIST (cached at ~/.cache/hinton-mnist/).
  2. Train a teacher (784-1200-1200-10 MLP, ReLU, ~2 px jittered inputs).
  3. Train a student (784-800-800-10 MLP, ReLU, no regularization) by
     distillation at T=20 on a transfer set with all "3"s removed.
  4. Evaluate student on test 3s -- it usually still gets nontrivial accuracy
     because the teacher's soft targets carry "dark knowledge" about 3.
  5. Apply bias correction: increase student logit-bias for class 3 to match
     the expected class frequency on the full training set (~10%).
  6. Re-evaluate; bias-corrected student approaches teacher accuracy on 3s.

Implementation: pure numpy, Adam optimizer.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import platform
import struct
import sys
import time
import urllib.request
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# MNIST loader
# ---------------------------------------------------------------------------

MNIST_CACHE = Path.home() / ".cache" / "hinton-mnist"

# Yann LeCun's original URL is unreliable; ossci-datasets is Facebook's mirror.
MNIST_URLS = [
    "https://ossci-datasets.s3.amazonaws.com/mnist/{name}",
    "https://storage.googleapis.com/cvdf-datasets/mnist/{name}",
    "http://yann.lecun.com/exdb/mnist/{name}",
]

MNIST_FILES = {
    "train_images": "train-images-idx3-ubyte.gz",
    "train_labels": "train-labels-idx1-ubyte.gz",
    "test_images": "t10k-images-idx3-ubyte.gz",
    "test_labels": "t10k-labels-idx1-ubyte.gz",
}


def _download(name: str, dest: Path) -> None:
    last_err: Exception | None = None
    for tmpl in MNIST_URLS:
        url = tmpl.format(name=name)
        try:
            print(f"  fetching {url}")
            urllib.request.urlretrieve(url, dest)
            return
        except Exception as exc:  # network / 404 / etc.
            last_err = exc
            print(f"    failed: {exc}")
    raise RuntimeError(f"could not download {name}: {last_err}")


def _read_images(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n, rows, cols = struct.unpack(">IIII", f.read(16))
        if magic != 2051:
            raise ValueError(f"bad magic {magic} in {path}")
        buf = f.read(n * rows * cols)
    arr = np.frombuffer(buf, dtype=np.uint8).reshape(n, rows, cols)
    return arr.astype(np.float32) / 255.0


def _read_labels(path: Path) -> np.ndarray:
    with gzip.open(path, "rb") as f:
        magic, n = struct.unpack(">II", f.read(8))
        if magic != 2049:
            raise ValueError(f"bad magic {magic} in {path}")
        buf = f.read(n)
    return np.frombuffer(buf, dtype=np.uint8).astype(np.int64)


def load_mnist(cache_dir: Path = MNIST_CACHE) -> dict[str, np.ndarray]:
    """Return dict with train_x, train_y, test_x, test_y (images flattened to 784)."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    paths = {}
    for key, name in MNIST_FILES.items():
        p = cache_dir / name
        if not p.exists() or p.stat().st_size == 0:
            _download(name, p)
        paths[key] = p

    train_x = _read_images(paths["train_images"])
    train_y = _read_labels(paths["train_labels"])
    test_x = _read_images(paths["test_images"])
    test_y = _read_labels(paths["test_labels"])
    return {
        "train_x": train_x.reshape(-1, 28 * 28),
        "train_y": train_y,
        "train_x_2d": train_x,
        "test_x": test_x.reshape(-1, 28 * 28),
        "test_y": test_y,
    }


# ---------------------------------------------------------------------------
# Class filtering
# ---------------------------------------------------------------------------

def filter_class(images: np.ndarray, labels: np.ndarray,
                 omitted_class: int = 3) -> tuple[np.ndarray, np.ndarray]:
    """Drop all examples of ``omitted_class`` from ``(images, labels)``."""
    keep = labels != omitted_class
    return images[keep], labels[keep]


# ---------------------------------------------------------------------------
# Jitter (teacher input augmentation)
# ---------------------------------------------------------------------------

def jitter_batch(images_2d: np.ndarray, max_shift: int,
                 rng: np.random.Generator) -> np.ndarray:
    """Randomly shift each 28x28 image by ``[-max_shift, +max_shift]`` px on each axis.

    Hinton's teacher uses small jittered translations as input augmentation.
    Returns a flattened (B, 784) array.
    """
    if max_shift <= 0:
        return images_2d.reshape(images_2d.shape[0], -1)
    n, h, w = images_2d.shape
    out = np.zeros_like(images_2d)
    dy = rng.integers(-max_shift, max_shift + 1, size=n)
    dx = rng.integers(-max_shift, max_shift + 1, size=n)
    for i in range(n):
        sy, sx = dy[i], dx[i]
        ys_src = slice(max(0, -sy), h - max(0, sy))
        ys_dst = slice(max(0, sy), h - max(0, -sy))
        xs_src = slice(max(0, -sx), w - max(0, sx))
        xs_dst = slice(max(0, sx), w - max(0, -sx))
        out[i, ys_dst, xs_dst] = images_2d[i, ys_src, xs_src]
    return out.reshape(n, -1)


# ---------------------------------------------------------------------------
# MLP with Adam
# ---------------------------------------------------------------------------

def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(x, 0.0)


def softmax(z: np.ndarray) -> np.ndarray:
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def log_softmax(z: np.ndarray) -> np.ndarray:
    m = z.max(axis=1, keepdims=True)
    return z - m - np.log(np.exp(z - m).sum(axis=1, keepdims=True))


class MLP:
    """Generic MLP: layer sizes (e.g. [784, 1200, 1200, 10]), ReLU + linear output.

    Uses He init for ReLU layers and zero biases. Adam optimizer state is
    held on the instance.
    """

    def __init__(self, sizes: list[int], seed: int = 0):
        self.sizes = sizes
        self.rng = np.random.default_rng(seed)
        self.W: list[np.ndarray] = []
        self.b: list[np.ndarray] = []
        for fan_in, fan_out in zip(sizes[:-1], sizes[1:]):
            scale = np.sqrt(2.0 / fan_in)  # He init
            self.W.append(
                (scale * self.rng.standard_normal((fan_in, fan_out))).astype(np.float32)
            )
            self.b.append(np.zeros(fan_out, dtype=np.float32))
        self._init_adam()

    def _init_adam(self):
        self.mW = [np.zeros_like(W) for W in self.W]
        self.vW = [np.zeros_like(W) for W in self.W]
        self.mb = [np.zeros_like(b) for b in self.b]
        self.vb = [np.zeros_like(b) for b in self.b]
        self.t = 0

    # ---- forward (returns logits + intermediate activations for backprop) ----

    def forward(self, x: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        acts = [x]
        h = x
        n_layers = len(self.W)
        for i, (W, b) in enumerate(zip(self.W, self.b)):
            z = h @ W + b
            if i < n_layers - 1:
                h = relu(z)
            else:
                h = z  # logits
            acts.append(h)
        return h, acts

    def logits(self, x: np.ndarray) -> np.ndarray:
        return self.forward(x)[0]

    # ---- backprop given dL/dlogits ----

    def backward(self, acts: list[np.ndarray], d_logits: np.ndarray
                 ) -> tuple[list[np.ndarray], list[np.ndarray]]:
        n_layers = len(self.W)
        grads_W = [None] * n_layers
        grads_b = [None] * n_layers
        delta = d_logits  # gradient at output (logits)
        for i in range(n_layers - 1, -1, -1):
            a_in = acts[i]
            grads_W[i] = a_in.T @ delta
            grads_b[i] = delta.sum(axis=0)
            if i > 0:
                # backprop through ReLU(z_{i-1}) where acts[i] = relu(z_{i-1})
                dh = delta @ self.W[i].T
                delta = dh * (acts[i] > 0)
        return grads_W, grads_b

    # ---- Adam step ----

    def adam_step(self, grads_W, grads_b, lr: float = 1e-3,
                  beta1: float = 0.9, beta2: float = 0.999,
                  eps: float = 1e-8):
        self.t += 1
        bc1 = 1 - beta1 ** self.t
        bc2 = 1 - beta2 ** self.t
        for i in range(len(self.W)):
            self.mW[i] = beta1 * self.mW[i] + (1 - beta1) * grads_W[i]
            self.vW[i] = beta2 * self.vW[i] + (1 - beta2) * grads_W[i] ** 2
            mhat = self.mW[i] / bc1
            vhat = self.vW[i] / bc2
            self.W[i] -= (lr * mhat / (np.sqrt(vhat) + eps)).astype(np.float32)

            self.mb[i] = beta1 * self.mb[i] + (1 - beta1) * grads_b[i]
            self.vb[i] = beta2 * self.vb[i] + (1 - beta2) * grads_b[i] ** 2
            mhat_b = self.mb[i] / bc1
            vhat_b = self.vb[i] / bc2
            self.b[i] -= (lr * mhat_b / (np.sqrt(vhat_b) + eps)).astype(np.float32)


# ---------------------------------------------------------------------------
# Teacher / student factories
# ---------------------------------------------------------------------------

def build_teacher(seed: int = 0) -> MLP:
    """784 -> 1200 -> 1200 -> 10 ReLU MLP. Trained with jittered inputs."""
    return MLP([784, 1200, 1200, 10], seed=seed)


def build_student(seed: int = 0) -> MLP:
    """784 -> 800 -> 800 -> 10 ReLU MLP. No weight decay, no dropout."""
    return MLP([784, 800, 800, 10], seed=seed)


# ---------------------------------------------------------------------------
# Training: hard-label cross-entropy (teacher)
# ---------------------------------------------------------------------------

def train_teacher(teacher: MLP,
                  data: dict,
                  *,
                  n_epochs: int = 12,
                  batch_size: int = 128,
                  lr: float = 1e-3,
                  jitter_px: int = 2,
                  seed: int = 0,
                  verbose: bool = True,
                  history: dict | None = None) -> dict:
    """Train teacher on full MNIST with jittered inputs.

    Cross-entropy on hard labels. Returns history dict with epoch / train_loss /
    test_acc curves.
    """
    rng = np.random.default_rng(seed + 17)
    x_train_2d = data["train_x_2d"]
    y_train = data["train_y"]
    x_test = data["test_x"]
    y_test = data["test_y"]
    n = x_train_2d.shape[0]

    if history is None:
        history = {"epoch": [], "train_loss": [], "test_acc": []}

    for epoch in range(n_epochs):
        t0 = time.time()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for b0 in range(0, n, batch_size):
            idx = perm[b0:b0 + batch_size]
            xb_2d = x_train_2d[idx]
            xb = jitter_batch(xb_2d, jitter_px, rng).astype(np.float32)
            yb = y_train[idx]

            logits, acts = teacher.forward(xb)
            log_p = log_softmax(logits)
            B = xb.shape[0]
            loss = -log_p[np.arange(B), yb].mean()

            # softmax - one_hot, normalized by batch
            p = np.exp(log_p)
            d_logits = p
            d_logits[np.arange(B), yb] -= 1.0
            d_logits /= B

            grads_W, grads_b = teacher.backward(acts, d_logits.astype(np.float32))
            teacher.adam_step(grads_W, grads_b, lr=lr)

            epoch_loss += float(loss)
            n_batches += 1

        test_acc = evaluate_accuracy(teacher, x_test, y_test)
        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(test_acc)

        if verbose:
            print(f"  teacher epoch {epoch + 1:3d}/{n_epochs}  "
                  f"loss={avg_loss:.4f}  test_acc={test_acc * 100:5.2f}%  "
                  f"({time.time() - t0:.1f}s)")

    return history


# ---------------------------------------------------------------------------
# Distillation
# ---------------------------------------------------------------------------

def distill(teacher: MLP,
            student: MLP,
            data: dict,
            *,
            temperature: float = 20.0,
            n_epochs: int = 20,
            batch_size: int = 128,
            lr: float = 1e-3,
            omitted_class: int = 3,
            seed: int = 0,
            verbose: bool = True,
            history: dict | None = None,
            snapshot_callback=None,
            snapshot_every: int = 1,
            test_data: tuple[np.ndarray, np.ndarray] | None = None) -> dict:
    """Train student to match teacher's softened soft targets.

    Transfer set: full MNIST training set with all examples of ``omitted_class``
    removed (the 3-omission setup). Student loss = T^2 * KL(teacher_T || student_T)
    where the T^2 scale recovers the gradient magnitude after softening
    (Hinton et al. 2015 §2).
    """
    rng = np.random.default_rng(seed + 31)
    x_train = data["train_x"]
    y_train = data["train_y"]

    # Filter: drop all "3"s for the transfer set.
    x_transfer, y_transfer = filter_class(x_train, y_train, omitted_class)
    n = x_transfer.shape[0]

    if history is None:
        history = {"epoch": [], "train_loss": [], "test_acc": [],
                   "test_acc_omitted": [], "test_acc_other": []}

    T = float(temperature)

    if test_data is not None:
        x_test, y_test = test_data
    else:
        x_test, y_test = data["test_x"], data["test_y"]

    # Pre-compute teacher logits on the transfer set in one shot? Memory:
    # ~54000 * 10 * 4B = 2.16 MB -- fine. But also recomputing per epoch is
    # cheap, so we just recompute per batch (avoids stale logits if teacher
    # changes; here teacher is frozen but the cost is negligible).

    for epoch in range(n_epochs):
        t0 = time.time()
        perm = rng.permutation(n)
        epoch_loss = 0.0
        n_batches = 0
        for b0 in range(0, n, batch_size):
            idx = perm[b0:b0 + batch_size]
            xb = x_transfer[idx]

            # Teacher soft targets at temperature T
            t_logits = teacher.logits(xb)
            t_soft = softmax(t_logits / T)

            # Student soft predictions at temperature T
            s_logits, acts = student.forward(xb)
            s_log_soft = log_softmax(s_logits / T)
            s_soft = np.exp(s_log_soft)

            # Cross-entropy under softening: -sum(t_soft * s_log_soft).
            # Gradient wrt s_logits: (s_soft - t_soft) / T.
            # Multiply loss by T^2 to keep gradient magnitudes O(1)
            # relative to a hard cross-entropy at T=1.
            B = xb.shape[0]
            loss = -(t_soft * s_log_soft).sum(axis=1).mean() * (T * T)
            d_logits = ((s_soft - t_soft) / T) * (T * T) / B
            d_logits = (d_logits * 1.0).astype(np.float32)

            grads_W, grads_b = student.backward(acts, d_logits)
            student.adam_step(grads_W, grads_b, lr=lr)

            epoch_loss += float(loss)
            n_batches += 1

        # Evaluation: test accuracy overall + on omitted class only +
        # on non-omitted classes.
        test_acc = evaluate_accuracy(student, x_test, y_test)
        mask_om = y_test == omitted_class
        if mask_om.sum() > 0:
            acc_om = evaluate_accuracy(student, x_test[mask_om], y_test[mask_om])
        else:
            acc_om = 0.0
        mask_other = ~mask_om
        acc_other = evaluate_accuracy(student, x_test[mask_other], y_test[mask_other])

        avg_loss = epoch_loss / max(n_batches, 1)
        history["epoch"].append(epoch + 1)
        history["train_loss"].append(avg_loss)
        history["test_acc"].append(test_acc)
        history["test_acc_omitted"].append(acc_om)
        history["test_acc_other"].append(acc_other)

        if verbose:
            print(f"  student epoch {epoch + 1:3d}/{n_epochs}  "
                  f"loss={avg_loss:.3f}  acc={test_acc * 100:5.2f}%  "
                  f"acc[omitted={omitted_class}]={acc_om * 100:5.2f}%  "
                  f"acc[other]={acc_other * 100:5.2f}%  "
                  f"({time.time() - t0:.1f}s)")

        if snapshot_callback is not None and (
            epoch % snapshot_every == 0 or epoch == n_epochs - 1
        ):
            snapshot_callback(epoch, student, history)

    return history


# ---------------------------------------------------------------------------
# Bias correction for the omitted class
# ---------------------------------------------------------------------------

def bias_correct_for_omitted(student: MLP,
                             data: dict,
                             omitted_class: int = 3,
                             max_iter: int = 60) -> float:
    """Increase logit-bias for ``omitted_class`` until the student's average
    softmax mass on that class matches the expected frequency on the full
    training set (~10% on MNIST).

    Returns the bias offset that was applied (added to ``student.b[-1][omitted_class]``).
    """
    train_y = data["train_y"]
    target_freq = float((train_y == omitted_class).mean())

    # Use a held-out chunk of the training images to measure current frequency
    # cheaply -- a 5000-image probe is plenty.
    probe = data["train_x"][:5000]

    def avg_prob_omitted(offset: float) -> float:
        # Apply the offset transiently (without mutating until we commit).
        original = float(student.b[-1][omitted_class])
        student.b[-1][omitted_class] = np.float32(original + offset)
        try:
            logits = student.logits(probe)
            probs = softmax(logits)
            return float(probs[:, omitted_class].mean())
        finally:
            student.b[-1][omitted_class] = np.float32(original)

    # Binary-search the bias offset that makes the mean p(omitted) == target_freq.
    lo, hi = 0.0, 30.0
    # Ensure hi is high enough -- expand if necessary.
    while avg_prob_omitted(hi) < target_freq and hi < 100.0:
        hi *= 2.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        p = avg_prob_omitted(mid)
        if p < target_freq:
            lo = mid
        else:
            hi = mid
        if hi - lo < 1e-4:
            break
    offset = 0.5 * (lo + hi)
    student.b[-1][omitted_class] = np.float32(
        student.b[-1][omitted_class] + offset
    )
    return offset


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_accuracy(model: MLP, x: np.ndarray, y: np.ndarray,
                      batch_size: int = 1000) -> float:
    if x.shape[0] == 0:
        return 0.0
    correct = 0
    for b0 in range(0, x.shape[0], batch_size):
        logits = model.logits(x[b0:b0 + batch_size])
        pred = np.argmax(logits, axis=1)
        correct += int((pred == y[b0:b0 + batch_size]).sum())
    return correct / x.shape[0]


def per_class_accuracy(model: MLP, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    accs = np.zeros(10, dtype=np.float64)
    for c in range(10):
        m = y == c
        if m.sum() == 0:
            accs[c] = float("nan")
        else:
            accs[c] = evaluate_accuracy(model, x[m], y[m])
    return accs


# ---------------------------------------------------------------------------
# Environment metadata (for reproducibility)
# ---------------------------------------------------------------------------

def env_info() -> dict:
    return {
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "platform": platform.platform(),
        "processor": platform.processor() or platform.machine(),
    }


# ---------------------------------------------------------------------------
# Top-level run
# ---------------------------------------------------------------------------

def run(seed: int = 0,
        temperature: float = 20.0,
        n_epochs_teacher: int = 12,
        n_epochs_student: int = 20,
        omitted_class: int = 3,
        verbose: bool = True,
        return_history: bool = False) -> dict:
    """End-to-end reproduction. Returns a results dict."""
    t_total = time.time()

    if verbose:
        print(f"# distillation-mnist-omitted-{omitted_class}  "
              f"(seed={seed}, T={temperature})")
        print(f"#   {env_info()}")

    if verbose:
        print("\n[1/6] Loading MNIST...")
    data = load_mnist()
    if verbose:
        print(f"  train: {data['train_x'].shape}  test: {data['test_x'].shape}")

    if verbose:
        print(f"\n[2/6] Building + training teacher ({n_epochs_teacher} epochs)...")
    teacher = build_teacher(seed=seed)
    teacher_history = train_teacher(
        teacher, data,
        n_epochs=n_epochs_teacher, seed=seed, verbose=verbose,
    )
    teacher_acc = teacher_history["test_acc"][-1]
    teacher_per_class = per_class_accuracy(
        teacher, data["test_x"], data["test_y"]
    )

    if verbose:
        print(f"\n[3/6] Building + distilling student "
              f"(T={temperature}, {n_epochs_student} epochs, no class {omitted_class})...")
    student = build_student(seed=seed + 1)
    student_history = distill(
        teacher, student, data,
        temperature=temperature, n_epochs=n_epochs_student,
        omitted_class=omitted_class, seed=seed, verbose=verbose,
    )

    if verbose:
        print("\n[4/6] Evaluating student before bias correction...")
    pre_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    pre_overall = evaluate_accuracy(student, data["test_x"], data["test_y"])
    pre_omitted = pre_per_class[omitted_class]

    if verbose:
        print(f"  pre-correction: overall={pre_overall * 100:.2f}%  "
              f"omitted-class[{omitted_class}]={pre_omitted * 100:.2f}%")

    if verbose:
        print(f"\n[5/6] Applying bias correction for class {omitted_class}...")
    offset = bias_correct_for_omitted(student, data, omitted_class=omitted_class)
    if verbose:
        print(f"  applied offset = {offset:+.3f} to b_out[{omitted_class}]")

    post_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    post_overall = evaluate_accuracy(student, data["test_x"], data["test_y"])
    post_omitted = post_per_class[omitted_class]

    wallclock = time.time() - t_total
    if verbose:
        print(f"\n[6/6] DONE in {wallclock:.1f}s")
        print(f"  teacher overall: {teacher_acc * 100:.2f}%")
        print(f"  teacher on class {omitted_class}: "
              f"{teacher_per_class[omitted_class] * 100:.2f}%")
        print(f"  student pre-correction overall: {pre_overall * 100:.2f}%")
        print(f"  student pre-correction class {omitted_class}: "
              f"{pre_omitted * 100:.2f}%")
        print(f"  student post-correction overall: {post_overall * 100:.2f}%")
        print(f"  student post-correction class {omitted_class}: "
              f"{post_omitted * 100:.2f}%  "
              f"<-- target >90% (paper: 98.6%)")

    results = {
        "config": {
            "seed": seed,
            "temperature": temperature,
            "n_epochs_teacher": n_epochs_teacher,
            "n_epochs_student": n_epochs_student,
            "omitted_class": omitted_class,
            "teacher_arch": [784, 1200, 1200, 10],
            "student_arch": [784, 800, 800, 10],
            "jitter_px": 2,
            "lr": 1e-3,
            "batch_size": 128,
            "optimizer": "adam",
        },
        "env": env_info(),
        "teacher": {
            "test_acc": teacher_acc,
            "per_class": teacher_per_class.tolist(),
        },
        "student_pre": {
            "test_acc": pre_overall,
            "per_class": pre_per_class.tolist(),
            "acc_omitted": pre_omitted,
        },
        "student_post": {
            "test_acc": post_overall,
            "per_class": post_per_class.tolist(),
            "acc_omitted": post_omitted,
            "bias_offset": offset,
        },
        "wallclock_sec": wallclock,
    }
    if return_history:
        results["teacher_history"] = teacher_history
        results["student_history"] = student_history
    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--n-epochs-teacher", type=int, default=12)
    p.add_argument("--n-epochs-student", type=int, default=20)
    p.add_argument("--omitted-class", type=int, default=3)
    p.add_argument("--save-results", type=str, default=None,
                   help="Optional JSON path to dump results.")
    p.add_argument("--quiet", action="store_true")
    args = p.parse_args(argv)

    results = run(
        seed=args.seed,
        temperature=args.temperature,
        n_epochs_teacher=args.n_epochs_teacher,
        n_epochs_student=args.n_epochs_student,
        omitted_class=args.omitted_class,
        verbose=not args.quiet,
    )
    if args.save_results:
        with open(args.save_results, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  wrote results to {args.save_results}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
