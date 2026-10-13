"""Design diagrams for the Wan VAE streaming decode (warm decoder cache).

Draws, with matplotlib only:

  design_before_after.png  per-window decode before (re-decoded context, empty
                           cache) and after (new latents only, cache carried)
  design_cache_lifecycle.png  when a stream's decoder cache is created, used
                           and freed, and the stateless fallback
  design_multi_pod.png     one generator and two decoder copies: sticky
                           windows and what a lost copy does

Each figure is written as PNG and SVG.

    python draw_design.py [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

# Okabe-Ito
GREY = "#BBBBBB"
GREEN = "#009E73"
BLUE = "#0072B2"
SKY = "#56B4E9"
ORANGE = "#E69F00"
RED = "#D55E00"
INK = "#222222"
MUTED = "#666666"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "font.size": 13,
    "svg.fonttype": "none",
})


def _canvas(w_in: float, h_in: float):
    """A figure whose data units are tenths of an inch, so boxes and text
    keep their proportions."""
    fig = plt.figure(figsize=(w_in, h_in), facecolor="white")
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, w_in * 10)
    ax.set_ylim(0, h_in * 10)
    ax.axis("off")
    return fig, ax


def _box(ax, x, y, w, h, text, fc, ec=None, tc="white", fs=13, ls="-", bold=False, lw=1.6):
    ax.add_patch(FancyBboxPatch(
        (x, y), w, h, boxstyle="round,pad=0,rounding_size=1.2",
        fc=fc, ec=ec or fc, lw=lw, ls=ls,
    ))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs,
            fontweight="bold" if bold else "normal", linespacing=1.25)


def _arrow(ax, a, b, color=INK, ls="-", rad=0.0, lw=2.0):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=16, color=color, lw=lw, ls=ls,
        connectionstyle=f"arc3,rad={rad}", shrinkA=2, shrinkB=2,
    ))


def _save(fig, out: Path, name: str) -> None:
    for ext in ("png", "svg"):
        fig.savefig(out / f"{name}.{ext}", dpi=110, facecolor="white")
    plt.close(fig)


def before_after(out: Path) -> None:
    fig, ax = _canvas(10, 4.2)
    cell, gap = 1.7, 0.25
    step = cell + gap
    win_w = 11 * step - gap
    x0, win_gap = 14.5, 2.4
    starts = [x0 + k * (win_w + win_gap) for k in range(3)]
    top, bot, h = 26.0, 9.5, 3.6

    for k, xs in enumerate(starts):
        ax.text(xs + win_w / 2, 35.5, ("window k", "window k+1", "window k+2")[k],
                ha="center", fontsize=13, color=MUTED)

    # Before: 8 re-decoded context latents + 3 new, from an empty cache.
    ax.text(1, top + h / 2 + 1.2, "Before", fontsize=15, fontweight="bold", va="center", color=INK)
    ax.text(1, top + h / 2 - 2.0, "empty cache", fontsize=13, va="center", color=MUTED)
    for xs in starts:
        for i in range(11):
            ax.add_patch(Rectangle((xs + i * step, top), cell, h, fc=GREY if i < 8 else GREEN, ec="white", lw=0.8))
        ax.text(xs + 8 * step / 2 - gap / 2, top + h / 2, "8 context", ha="center", va="center", fontsize=12, color=INK)
        ax.text(xs + 8 * step + 3 * step / 2 - gap / 2, top + h / 2, "new", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        ax.text(xs + 8 * step / 2 - gap / 2, top - 2.6, "output dropped", ha="center", fontsize=12,
                color=MUTED, style="italic")

    # After: 3 new latents continuing the stream's cache.
    ax.text(1, bot + h / 2 + 1.2, "After", fontsize=15, fontweight="bold", va="center", color=INK)
    ax.text(1, bot + h / 2 - 2.0, "cache kept", fontsize=13, va="center", color=MUTED)
    cache_w = 6.2
    for k, xs in enumerate(starts):
        gx = xs + 8 * step
        for i in range(3):
            ax.add_patch(Rectangle((gx + i * step, bot), cell, h, fc=GREEN, ec="white", lw=0.8))
        ax.text(gx + 3 * step / 2 - gap / 2, bot + h / 2, "new", ha="center", va="center",
                fontsize=12, color="white", fontweight="bold")
        cx = gx - 0.5 - cache_w
        _box(ax, cx, bot, cache_w, h, "cache", BLUE, fs=12, bold=True)
        src = (starts[k - 1] + 11 * step - gap, bot + h / 2) if k else (x0 - 1.5, bot + h / 2)
        _arrow(ax, src, (cx, bot + h / 2), color=BLUE, lw=2.2)
        if k:
            ax.text((src[0] + cx) / 2, bot + h + 1.0, "kept", ha="center", fontsize=12, color=BLUE)

    # Cost per window.
    mx = 86.0
    ax.text(mx, top + h / 2 + 1.4, "335 ms", fontsize=16, fontweight="bold", va="center", color=INK)
    ax.text(mx, top + h / 2 - 1.8, "3.4× FLOPs", fontsize=12, va="center", color=MUTED)
    ax.text(mx, bot + h / 2 + 1.4, "56 ms", fontsize=16, fontweight="bold", va="center", color=GREEN)
    ax.text(mx, bot + h / 2 - 1.8, "6.0× faster", fontsize=12, va="center", color=MUTED)

    ax.text(50, 2.2, "Wan 2.1 VAE, B200, bf16, per 3 new latents (12 frames at 480×832).  "
            "Cache: 1.76 GiB per live stream.",
            ha="center", fontsize=12, color=MUTED)
    _save(fig, out, "design_before_after")


def cache_lifecycle(out: Path) -> None:
    fig, ax = _canvas(10, 4.2)
    y, h = 17.0, 8.0
    _box(ax, 1.0, y, 16.0, h, "Windowed\nrequest starts", GREY, tc=INK)
    _box(ax, 23.0, y, 20.0, h, "Cache created\nfirst window decoded", SKY, tc=INK)
    _box(ax, 49.0, y, 20.0, h, "Warm windows\nnew latents only", BLUE, bold=True)
    _box(ax, 75.0, y - 3.0, 24.0, h + 6.0,
         "Cache freed\nlast window decoded,\nabort or error,\nshutdown", GREEN, fs=12)
    _arrow(ax, (17.0, y + h / 2), (23.0, y + h / 2))
    _arrow(ax, (43.0, y + h / 2), (49.0, y + h / 2))
    _arrow(ax, (69.0, y + h / 2), (75.0, y + h / 2))
    # Each window updates the cache in place.
    _arrow(ax, (55.0, y + h), (63.0, y + h), color=BLUE, rad=-1.2)
    ax.text(65.0, y + h + 3.4, "each window", ha="left", fontsize=12, color=BLUE)

    # Supported fallback: a resumed session has no cache yet.
    _box(ax, 13.0, 32.0, 30.0, 7.0, "Resumed session: no cache\n→ re-decode 8 stored latents",
         "white", ec=ORANGE, tc=INK, fs=12, lw=2.2)
    _arrow(ax, (33.0, 32.0), (33.0, y + h), color=ORANGE, lw=2.2)
    ax.text(34.0, 28.6, "stateless, one window", fontsize=12, color=ORANGE)

    # Not handled: a window arriving on a copy without the stream's state.
    _box(ax, 37.0, 1.5, 44.0, 9.0,
         "Window lands on another copy: not handled yet\n(cannot happen today: routing is sticky,\n"
         "a lost copy fails the request)",
         "white", ec=RED, tc=RED, fs=12, ls="--", lw=2.2)
    _arrow(ax, (59.0, y), (59.0, 10.5), color=RED, ls="--", lw=2.0)
    ax.text(1.0, 4.0, "Sessions store 8 latents,\nnever the cache.", fontsize=12, color=MUTED, linespacing=1.3)
    _save(fig, out, "design_cache_lifecycle")


def multi_pod(out: Path) -> None:
    fig, ax = _canvas(10, 4.2)
    _box(ax, 1.0, 14.0, 20.0, 14.0, "Generator pod\nDiT\nwindow latents", GREY, tc=INK, fs=13)
    _box(ax, 46.0, 25.0, 24.0, 12.0, "Decoder copy 1\ncache A", BLUE, bold=True)
    _box(ax, 46.0, 4.0, 24.0, 12.0, "Decoder copy 2\ncache B", "white", ec=RED, tc=RED, ls="--", lw=2.2)
    ax.text(58.0, 18.0, "pod lost", ha="center", fontsize=13, color=RED, fontweight="bold")

    _arrow(ax, (21.0, 24.0), (46.0, 31.0), color=BLUE, lw=2.4)
    ax.text(24.0, 31.0, "stream A: every\nwindow, same copy", fontsize=12, color=BLUE, linespacing=1.2)
    _arrow(ax, (21.0, 18.0), (46.0, 10.0), color=RED, lw=2.4)
    ax.text(30.0, 10.0, "stream B", fontsize=12, color=RED)

    _box(ax, 76.0, 25.0, 23.0, 12.0, "Whole-stream frames\n6× faster decode", GREEN, fs=12)
    _arrow(ax, (70.0, 31.0), (76.0, 31.0), color=GREEN)
    _box(ax, 76.0, 4.0, 23.0, 12.0, "Request fails (503)\nclient retries\nor resumes", "white",
         ec=RED, tc=RED, fs=12, lw=2.2)
    _arrow(ax, (70.0, 10.0), (76.0, 10.0), color=RED)
    ax.text(50.0, 40.0, "Decoder copy picked once per request, kept for the whole rollout",
            ha="center", fontsize=12, color=MUTED)
    _save(fig, out, "design_multi_pod")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    before_after(args.out)
    cache_lifecycle(args.out)
    multi_pod(args.out)


if __name__ == "__main__":
    main()
