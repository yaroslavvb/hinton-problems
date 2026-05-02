"""
Static visualizations for the MNIST distillation-with-omitted-3 experiment.

Outputs (in ``viz/``):
  teacher_curves.png      - teacher train loss + test accuracy
  student_curves.png      - student distillation loss + per-class-3 accuracy
                            during distillation
  per_class_accuracy.png  - per-class accuracy of teacher vs student
                            (pre-correction vs post-correction); class 3 highlighted
  bias_correction.png     - student class-3 accuracy and student class-3 average
                            softmax mass as a function of the bias offset

Run:
  python3 visualize_distillation_mnist_omitted_3.py [--seed 0] [--outdir viz]
"""

from __future__ import annotations

import argparse
import os

import numpy as np
import matplotlib.pyplot as plt

from distillation_mnist_omitted_3 import (
    bias_correct_for_omitted,
    build_student,
    build_teacher,
    distill,
    evaluate_accuracy,
    load_mnist,
    per_class_accuracy,
    softmax,
    train_teacher,
)


# ---- training curves ------------------------------------------------------

def plot_teacher_curves(history: dict, out_path: str) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=120)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel("train cross-entropy")
    ax.grid(alpha=0.3)
    ax.set_title("Teacher train loss")

    ax = axes[1]
    ax.plot(epochs, np.array(history["test_acc"]) * 100, color="#2ca02c")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(95, 100)
    ax.grid(alpha=0.3)
    ax.set_title("Teacher test accuracy")

    fig.suptitle("Teacher (784-1200-1200-10, ReLU, ±2 px jitter)", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


def plot_student_curves(history: dict, omitted_class: int, out_path: str) -> None:
    epochs = history["epoch"]
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), dpi=120)

    ax = axes[0]
    ax.plot(epochs, history["train_loss"], color="#1f77b4")
    ax.set_xlabel("epoch")
    ax.set_ylabel(r"$T^2 \cdot$ soft-target CE")
    ax.grid(alpha=0.3)
    ax.set_title("Student distillation loss (T=20)")

    ax = axes[1]
    ax.plot(epochs, np.array(history["test_acc_other"]) * 100,
            color="#2ca02c", label="other classes")
    ax.plot(epochs, np.array(history["test_acc_omitted"]) * 100,
            color="#d62728", label=f"omitted class {omitted_class}",
            linewidth=2.0)
    ax.plot(epochs, np.array(history["test_acc"]) * 100,
            color="#444", linestyle=":", label="overall")
    ax.set_xlabel("epoch")
    ax.set_ylabel("test accuracy (%)")
    ax.set_ylim(0, 100)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=9)
    ax.set_title(
        f"Student accuracy during distillation (no class {omitted_class} in transfer set)"
    )

    fig.suptitle(
        "Student (784-800-800-10) trained by distillation only (no hard labels)",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- per-class summary ----------------------------------------------------

def plot_per_class_accuracy(teacher_per_class: np.ndarray,
                            student_pre: np.ndarray,
                            student_post: np.ndarray,
                            omitted_class: int,
                            out_path: str) -> None:
    classes = np.arange(10)
    width = 0.27

    fig, ax = plt.subplots(figsize=(11, 4.5), dpi=120)
    ax.bar(classes - width, teacher_per_class * 100, width=width,
           label="teacher", color="#7e7e7e")
    ax.bar(classes, student_pre * 100, width=width,
           label="student (no 3, pre-correction)", color="#1f77b4")
    ax.bar(classes + width, student_post * 100, width=width,
           label="student (no 3, post-correction)", color="#2ca02c")

    # Highlight the omitted class
    for offset, vals, color in [
        (-width, teacher_per_class, "#7e7e7e"),
        (0.0, student_pre, "#1f77b4"),
        (width, student_post, "#2ca02c"),
    ]:
        ax.text(omitted_class + offset, vals[omitted_class] * 100 + 0.5,
                f"{vals[omitted_class] * 100:.1f}",
                ha="center", va="bottom", fontsize=8, color=color, weight="bold")

    ax.axvspan(omitted_class - 0.5, omitted_class + 0.5,
               color="#fff4b3", alpha=0.6, zorder=0)
    ax.set_xticks(classes)
    ax.set_xlabel("digit class")
    ax.set_ylabel("accuracy (%)")
    ax.set_ylim(70, 101)
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3, axis="y")
    ax.set_title(
        f"Per-class test accuracy. Yellow column = omitted class ({omitted_class})."
    )
    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- bias-correction sweep ------------------------------------------------

def plot_bias_correction_sweep(student,
                               data: dict,
                               omitted_class: int,
                               applied_offset: float,
                               out_path: str) -> None:
    """Sweep the bias offset for the omitted class and plot how the omitted-class
    accuracy and softmax mass change. Restores the original bias afterwards.
    """
    x_test = data["test_x"]
    y_test = data["test_y"]
    mask = y_test == omitted_class
    probe = data["train_x"][:5000]

    # The student passed in already has the bias offset applied. Compute the
    # original bias to sweep around 0 -- "zero" of the sweep is the
    # pre-correction state.
    orig_bias = float(student.b[-1][omitted_class]) - applied_offset
    offsets = np.linspace(0.0, max(applied_offset * 1.7, 5.0), 35)

    accs_omitted = []
    accs_overall = []
    mean_p_omitted = []
    for off in offsets:
        student.b[-1][omitted_class] = np.float32(orig_bias + off)
        accs_omitted.append(evaluate_accuracy(student, x_test[mask], y_test[mask]))
        accs_overall.append(evaluate_accuracy(student, x_test, y_test))
        probs = softmax(student.logits(probe))
        mean_p_omitted.append(float(probs[:, omitted_class].mean()))

    # Restore
    student.b[-1][omitted_class] = np.float32(orig_bias + applied_offset)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), dpi=120)

    ax = axes[0]
    ax.plot(offsets, np.array(accs_omitted) * 100, color="#d62728",
            label=f"acc on class {omitted_class}", linewidth=2.0)
    ax.plot(offsets, np.array(accs_overall) * 100, color="#444",
            label="overall", linestyle=":")
    ax.axvline(applied_offset, color="#2ca02c", linestyle="--",
               label=f"chosen offset = {applied_offset:.2f}")
    ax.set_xlabel(f"bias offset added to b_out[{omitted_class}]")
    ax.set_ylabel("test accuracy (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title(f"Effect of bias offset on test accuracy")

    target_freq = float((data["train_y"] == omitted_class).mean())
    ax = axes[1]
    ax.plot(offsets, np.array(mean_p_omitted) * 100, color="#1f77b4")
    ax.axhline(target_freq * 100, color="#888", linestyle=":",
               label=f"target freq {target_freq * 100:.1f}%")
    ax.axvline(applied_offset, color="#2ca02c", linestyle="--",
               label=f"chosen offset = {applied_offset:.2f}")
    ax.set_xlabel(f"bias offset added to b_out[{omitted_class}]")
    ax.set_ylabel(f"mean p(class {omitted_class}) on probe (%)")
    ax.legend(loc="lower right", fontsize=9)
    ax.grid(alpha=0.3)
    ax.set_title("Bias-correction criterion (match class frequency)")

    fig.suptitle(
        f"Bias correction sweep -- digit {omitted_class}",
        fontsize=11,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(out_path)
    plt.close(fig)
    print(f"  wrote {out_path}")


# ---- main -----------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--n-epochs-teacher", type=int, default=12)
    p.add_argument("--n-epochs-student", type=int, default=20)
    p.add_argument("--omitted-class", type=int, default=3)
    p.add_argument("--outdir", type=str, default="viz")
    args = p.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    print(f"# visualize: seed={args.seed}, T={args.temperature}, "
          f"omitted_class={args.omitted_class}")

    print("[1/6] Loading MNIST...")
    data = load_mnist()

    print(f"[2/6] Training teacher ({args.n_epochs_teacher} epochs)...")
    teacher = build_teacher(seed=args.seed)
    teacher_history = train_teacher(
        teacher, data,
        n_epochs=args.n_epochs_teacher, seed=args.seed, verbose=False,
    )
    teacher_per_class = per_class_accuracy(teacher, data["test_x"], data["test_y"])
    print(f"  teacher final test acc: {teacher_history['test_acc'][-1] * 100:.2f}%")

    print(f"[3/6] Distilling student ({args.n_epochs_student} epochs, T={args.temperature})...")
    student = build_student(seed=args.seed + 1)
    student_history = distill(
        teacher, student, data,
        temperature=args.temperature,
        n_epochs=args.n_epochs_student,
        omitted_class=args.omitted_class,
        seed=args.seed,
        verbose=False,
    )

    pre_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    print(f"  student final acc on class {args.omitted_class}: "
          f"{pre_per_class[args.omitted_class] * 100:.2f}%")

    print(f"[4/6] Bias correction for class {args.omitted_class}...")
    offset = bias_correct_for_omitted(student, data, omitted_class=args.omitted_class)
    post_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    print(f"  applied offset = {offset:+.3f}")
    print(f"  student post-correction acc on class {args.omitted_class}: "
          f"{post_per_class[args.omitted_class] * 100:.2f}%")

    print("[5/6] Plotting curves...")
    plot_teacher_curves(teacher_history,
                        os.path.join(args.outdir, "teacher_curves.png"))
    plot_student_curves(student_history, args.omitted_class,
                        os.path.join(args.outdir, "student_curves.png"))
    plot_per_class_accuracy(
        teacher_per_class, pre_per_class, post_per_class, args.omitted_class,
        os.path.join(args.outdir, "per_class_accuracy.png"),
    )

    print("[6/6] Sweeping bias offset for diagnostic plot...")
    plot_bias_correction_sweep(
        student, data, args.omitted_class, offset,
        os.path.join(args.outdir, "bias_correction.png"),
    )


if __name__ == "__main__":
    main()
