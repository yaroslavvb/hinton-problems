"""
Render an animated GIF showing the student learn the omitted-3 digit through
distillation.

Layout per frame:
  Top:    bar chart of per-class accuracy at the current epoch, with the
          omitted class (3) highlighted.
  Middle: a row of test-3 example images annotated with the student's top
          prediction (and softmax mass on class 3).
  Bottom: training curves -- overall accuracy, "other-class" accuracy,
          omitted-class accuracy.

A final "bias-correction" frame is appended showing the post-correction
accuracies side-by-side with the pre-correction ones.

Usage:
  python3 make_distillation_mnist_omitted_3_gif.py [--seed 0]
"""

from __future__ import annotations

import argparse
import os
from io import BytesIO

import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

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


PALETTE_OTHER = "#1f77b4"
PALETTE_OMITTED = "#d62728"
PALETTE_BIAS = "#2ca02c"


def render_frame(epoch: int,
                 history: dict,
                 per_class: np.ndarray,
                 sample_imgs_2d: np.ndarray,
                 sample_logits: np.ndarray,
                 sample_labels: np.ndarray,
                 omitted_class: int,
                 max_epochs: int,
                 title_suffix: str = "") -> Image.Image:
    fig = plt.figure(figsize=(11, 7), dpi=100)
    gs = fig.add_gridspec(3, 1, height_ratios=[1.4, 0.9, 1.1], hspace=0.55)

    # ---- top: per-class accuracy bar chart ----
    ax_bar = fig.add_subplot(gs[0])
    classes = np.arange(10)
    colors = [PALETTE_OMITTED if c == omitted_class else PALETTE_OTHER
              for c in classes]
    ax_bar.bar(classes, per_class * 100, color=colors)
    for c in classes:
        ax_bar.text(c, per_class[c] * 100 + 0.5,
                    f"{per_class[c] * 100:.0f}",
                    ha="center", va="bottom", fontsize=8,
                    color=PALETTE_OMITTED if c == omitted_class else "#444")
    ax_bar.axvspan(omitted_class - 0.5, omitted_class + 0.5,
                   color="#fff4b3", alpha=0.5, zorder=0)
    ax_bar.set_xticks(classes)
    ax_bar.set_ylabel("test acc (%)")
    ax_bar.set_ylim(0, 105)
    ax_bar.grid(alpha=0.3, axis="y")
    ax_bar.set_title(
        f"Per-class test accuracy "
        f"(yellow = omitted class {omitted_class})"
    )

    # ---- middle: sample 3-images with student predictions ----
    n_samples = sample_imgs_2d.shape[0]
    sample_probs = softmax(sample_logits)
    pred = np.argmax(sample_logits, axis=1)
    p_omitted = sample_probs[:, omitted_class]
    inner = gs[1].subgridspec(1, n_samples, wspace=0.15)
    for i in range(n_samples):
        ax_i = fig.add_subplot(inner[0, i])
        ax_i.imshow(sample_imgs_2d[i], cmap="gray_r")
        ok = pred[i] == sample_labels[i]
        edge_color = "#2ca02c" if ok else "#d62728"
        for s in ax_i.spines.values():
            s.set_color(edge_color)
            s.set_linewidth(2.0)
        ax_i.set_xticks([])
        ax_i.set_yticks([])
        ax_i.set_title(
            f"pred={pred[i]}\np(3)={p_omitted[i]:.2f}",
            fontsize=8,
            color=edge_color,
        )

    # ---- bottom: training curves up to current epoch ----
    ax_c = fig.add_subplot(gs[2])
    e = np.array(history["epoch"])
    ax_c.plot(e, np.array(history["test_acc"]) * 100,
              color="#444", linestyle=":", label="overall")
    ax_c.plot(e, np.array(history["test_acc_other"]) * 100,
              color=PALETTE_OTHER, label="other classes")
    ax_c.plot(e, np.array(history["test_acc_omitted"]) * 100,
              color=PALETTE_OMITTED, label=f"class {omitted_class} (omitted)",
              linewidth=2.0)
    ax_c.axvline(epoch + 1, color="black", linewidth=1.0, alpha=0.3)
    ax_c.set_xlim(1, max_epochs)
    ax_c.set_ylim(0, 102)
    ax_c.set_xlabel("epoch")
    ax_c.set_ylabel("acc (%)")
    ax_c.legend(loc="lower right", fontsize=9)
    ax_c.grid(alpha=0.3)
    ax_c.set_title("Distillation progress")

    fig.suptitle(
        f"Knowledge distillation: student trained on transfer set without "
        f"class {omitted_class}{title_suffix}",
        fontsize=12, y=0.98,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))

    buf = BytesIO()
    fig.savefig(buf, format="png", dpi=100, bbox_inches="tight")
    plt.close(fig)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--temperature", type=float, default=20.0)
    p.add_argument("--n-epochs-teacher", type=int, default=12)
    p.add_argument("--n-epochs-student", type=int, default=20)
    p.add_argument("--omitted-class", type=int, default=3)
    p.add_argument("--snapshot-every", type=int, default=1)
    p.add_argument("--fps", type=int, default=3)
    p.add_argument("--out", type=str, default="distillation_mnist_omitted_3.gif")
    p.add_argument("--hold-final", type=int, default=4)
    args = p.parse_args()

    print("[1/4] Loading MNIST...")
    data = load_mnist()

    print(f"[2/4] Training teacher ({args.n_epochs_teacher} epochs)...")
    teacher = build_teacher(seed=args.seed)
    train_teacher(teacher, data,
                  n_epochs=args.n_epochs_teacher, seed=args.seed, verbose=False)

    # Pick 8 random test 3s for the sample row, fixed for the whole animation.
    rng_disp = np.random.default_rng(args.seed + 99)
    test_y = data["test_y"]
    threes = np.where(test_y == args.omitted_class)[0]
    sample_idx = rng_disp.choice(threes, size=8, replace=False)
    sample_imgs_2d = data["test_x"][sample_idx].reshape(-1, 28, 28)
    sample_imgs_flat = data["test_x"][sample_idx]
    sample_labels = test_y[sample_idx]

    student = build_student(seed=args.seed + 1)

    frames: list[Image.Image] = []

    def cb(epoch: int, student_now, history):
        if epoch % args.snapshot_every != 0 \
                and epoch != args.n_epochs_student - 1:
            return
        per_class = per_class_accuracy(student_now, data["test_x"], data["test_y"])
        sample_logits = student_now.logits(sample_imgs_flat)
        frame = render_frame(
            epoch, history, per_class, sample_imgs_2d,
            sample_logits, sample_labels, args.omitted_class,
            max_epochs=args.n_epochs_student,
        )
        frames.append(frame)
        print(f"  frame {len(frames):3d}  epoch {epoch + 1}  "
              f"acc[3]={history['test_acc_omitted'][-1] * 100:.1f}%")

    print(f"[3/4] Distilling student "
          f"(T={args.temperature}, {args.n_epochs_student} epochs)...")
    distill(
        teacher, student, data,
        temperature=args.temperature,
        n_epochs=args.n_epochs_student,
        omitted_class=args.omitted_class,
        seed=args.seed,
        verbose=False,
        snapshot_callback=cb,
        snapshot_every=args.snapshot_every,
    )

    # Hold pre-correction frame
    if args.hold_final > 0 and frames:
        frames.extend([frames[-1]] * args.hold_final)

    # Bias-correction frame: same layout, with post-correction per-class.
    print(f"[4/4] Applying bias correction + final frame...")
    pre_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    pre_logits = student.logits(sample_imgs_flat)
    pre_history = {
        "epoch": list(range(1, args.n_epochs_student + 1)),
        "test_acc": list(np.linspace(0, pre_per_class.mean(),
                                     args.n_epochs_student)),
        "test_acc_other": list(np.linspace(0, np.delete(pre_per_class,
                                                        args.omitted_class).mean(),
                                           args.n_epochs_student)),
        "test_acc_omitted": list(np.linspace(0, pre_per_class[args.omitted_class],
                                             args.n_epochs_student)),
    }
    # We don't actually need pre_history for the post-correction frame: the
    # student_curves come from the distillation `history` we already animated.
    # For simplicity we re-run distill's history through the snapshot list:
    # but we already have it captured via the closure above (frames list).

    offset = bias_correct_for_omitted(student, data, omitted_class=args.omitted_class)
    post_per_class = per_class_accuracy(student, data["test_x"], data["test_y"])
    post_logits = student.logits(sample_imgs_flat)

    # Reuse the last training-history dict from `frames`-side if available; we
    # synthesize a tiny one-point history for the final frame so the bottom
    # plot looks consistent with the rest of the animation.
    history_tail = {
        "epoch": list(range(1, args.n_epochs_student + 1)),
        "test_acc": [],
        "test_acc_other": [],
        "test_acc_omitted": [],
    }
    # We'll just plot a single horizontal line for post-correction by passing
    # repeated values.
    final_overall = evaluate_accuracy(student, data["test_x"], data["test_y"])
    other_mask = data["test_y"] != args.omitted_class
    final_other = evaluate_accuracy(
        student, data["test_x"][other_mask], data["test_y"][other_mask]
    )
    final_omitted = post_per_class[args.omitted_class]
    history_tail["test_acc"] = [final_overall] * args.n_epochs_student
    history_tail["test_acc_other"] = [final_other] * args.n_epochs_student
    history_tail["test_acc_omitted"] = [final_omitted] * args.n_epochs_student

    final_frame = render_frame(
        args.n_epochs_student - 1, history_tail, post_per_class,
        sample_imgs_2d, post_logits, sample_labels, args.omitted_class,
        max_epochs=args.n_epochs_student,
        title_suffix=f"  --  AFTER BIAS CORRECTION (offset={offset:+.2f})",
    )
    frames.append(final_frame)
    if args.hold_final > 0:
        frames.extend([final_frame] * args.hold_final)

    duration_ms = max(1000 // max(args.fps, 1), 30)
    out_path = args.out
    frames[0].save(
        out_path, save_all=True, append_images=frames[1:],
        duration=duration_ms, loop=0, optimize=True,
    )
    size_kb = os.path.getsize(out_path) / 1024
    print(f"\nWrote {out_path}  ({len(frames)} frames, {size_kb:.0f} KB)")
    print(f"  pre-correction acc on {args.omitted_class}: "
          f"{pre_per_class[args.omitted_class] * 100:.2f}%")
    print(f"  post-correction acc on {args.omitted_class}: "
          f"{post_per_class[args.omitted_class] * 100:.2f}%  "
          f"(bias offset = {offset:+.3f})")


if __name__ == "__main__":
    main()
