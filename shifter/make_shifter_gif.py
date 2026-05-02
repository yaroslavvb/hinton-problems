"""
Generate `shifter.gif` — an animated illustration of the shifter task.

Each "scene" cycles through a single random V1 pattern paired with each of
the three shifts. For each pair we (a) show V1 alone, (b) fade in V2 with
arrows pointing from each V1 cell to its corresponding V2 cell under the
shift, and (c) hold the resulting state with a colored class label.

Lifted from `cybertronai/sutro-problems/wip-boltzmann-shifter/make_shifter_gif.py`
with minor cleanup. Output is well under the 3 MB limit at the default
settings.
"""

from __future__ import annotations
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, PillowWriter
import numpy as np


SHIFTS = [-1, 0, 1]
LABELS = {-1: "shift left  ( -1 )",
          0:  "no shift  ( 0 )",
          1:  "shift right  ( +1 )"}
COLORS = {-1: "#a83232", 0: "#404040", 1: "#3a78a8"}


def bits(n: int, N: int) -> np.ndarray:
    return np.array([(n >> i) & 1 for i in range(N)], dtype=int)


def build_frame_plan(patterns: list[int], N: int,
                      hold_frames: int = 6,
                      slide_frames: int = 3) -> list[dict]:
    frames = []
    for p in patterns:
        v1 = bits(p, N)
        for s in SHIFTS:
            v2 = np.roll(v1, s)
            for _ in range(hold_frames):
                frames.append({"v1": v1, "v2": None, "shift": s,
                               "phase": "v1_only", "t": 0.0})
            for k in range(1, slide_frames + 1):
                t = k / slide_frames
                frames.append({"v1": v1, "v2": v2, "shift": s,
                               "phase": "slide", "t": t})
            for _ in range(hold_frames):
                frames.append({"v1": v1, "v2": v2, "shift": s,
                               "phase": "hold", "t": 1.0})
    return frames


def cell_color(bit: int, alpha: float = 1.0):
    if bit == 1:
        return (0.13, 0.13, 0.13, alpha)
    return (0.97, 0.97, 0.97, alpha)


def draw_row(ax, y: float, vec, *, alpha: float = 1.0, label: str = ""):
    if vec is None:
        return
    N = len(vec)
    cell_w = 1.0
    for i, b in enumerate(vec):
        rect = mpatches.FancyBboxPatch(
            (i + 0.05, y + 0.05), cell_w - 0.1, cell_w - 0.1,
            boxstyle="round,pad=0.02,rounding_size=0.08",
            linewidth=1.2,
            edgecolor=(0.2, 0.2, 0.2, alpha),
            facecolor=cell_color(int(b), alpha=alpha),
        )
        ax.add_patch(rect)
        ax.text(i + 0.5, y + 0.5, str(int(b)),
                ha="center", va="center",
                color=(0.95 if b == 1 else 0.2,) * 3 + (alpha,),
                fontsize=14, fontfamily="monospace", fontweight="bold")
    if label:
        ax.text(-0.5, y + 0.5, label, ha="right", va="center",
                fontsize=13, fontfamily="monospace",
                color=(0.2, 0.2, 0.2, alpha))


def draw_arrows(ax, y_top: float, y_bot: float, N: int, shift: int,
                alpha: float):
    if shift == 0:
        return
    color = COLORS[shift]
    rgb = tuple(int(color[1 + 2 * k:3 + 2 * k], 16) / 255 for k in range(3))
    for i in range(N):
        j = (i + shift) % N
        wraps = abs(j - i) > 1
        rad = 0.4 if wraps else 0.0
        ax.annotate("",
                    xy=(j + 0.5, y_bot + 0.85),
                    xytext=(i + 0.5, y_top + 0.15),
                    arrowprops=dict(
                        arrowstyle="->", lw=1.3,
                        color=(*rgb, alpha * (0.6 if wraps else 1.0)),
                        connectionstyle=f"arc3,rad={rad}",
                    ))


def render_frame(ax, frame: dict, N: int):
    ax.clear()
    ax.set_xlim(-3.5, N + 0.5)
    ax.set_ylim(-1.5, 5.5)
    ax.set_aspect("equal")
    ax.axis("off")

    y_top, y_bot = 2.5, 0.5
    s = frame["shift"]
    t = frame["t"]
    phase = frame["phase"]
    v1 = frame["v1"]
    v2 = frame["v2"]

    ax.text(N / 2, 4.6, "Boltzmann shifter task",
            ha="center", va="center", fontsize=14, fontweight="bold",
            color=(0.15, 0.15, 0.15))
    ax.text(N / 2, -0.9, LABELS[s], ha="center", va="center",
            fontsize=15, fontfamily="monospace", fontweight="bold",
            color=COLORS[s])

    draw_row(ax, y_top, v1, alpha=1.0, label="V1")
    if phase == "v1_only":
        return
    alpha = t if phase == "slide" else 1.0
    draw_row(ax, y_bot, v2, alpha=alpha, label="V2")
    draw_arrows(ax, y_top, y_bot, N, s, alpha=alpha)


def make_gif(out_path: Path, N: int = 8, fps: int = 12, seed: int = 7):
    rng = np.random.default_rng(seed)
    patterns = [
        0b00010110,
        0b01101001,
        0b11000110,
        0b00111100,
    ]
    if N != 8:
        patterns = [int(rng.integers(1, 2**N - 1)) for _ in range(4)]

    frame_plan = build_frame_plan(patterns, N,
                                   hold_frames=6, slide_frames=3)

    fig, ax = plt.subplots(figsize=(7.5, 4.4), dpi=110)
    fig.patch.set_facecolor("white")

    def update(i):
        render_frame(ax, frame_plan[i], N)
        return []

    anim = FuncAnimation(fig, update, frames=len(frame_plan),
                         interval=1000 / fps, blit=False)
    writer = PillowWriter(fps=fps)
    anim.save(str(out_path), writer=writer)
    plt.close(fig)
    print(f"wrote {out_path}  ({len(frame_plan)} frames @ {fps} fps)")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--N", type=int, default=8)
    p.add_argument("--fps", type=int, default=12)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--out", default="shifter.gif")
    args = p.parse_args()
    make_gif(Path(args.out), N=args.N, fps=args.fps, seed=args.seed)
