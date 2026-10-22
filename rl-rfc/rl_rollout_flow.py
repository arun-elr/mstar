"""One RL training step with M* as the rollout engine and RLinf as the trainer.

Renders rl-rollout-flow.svg and rl-rollout-flow.png (2x) next to this file:
    python rl_rollout_flow.py
"""

from pathlib import Path as FsPath

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from matplotlib.path import Path

plt.rcParams["font.family"] = "DejaVu Sans"
plt.rcParams["svg.fonttype"] = "path"

W, H = 120, 89
OUT = FsPath(__file__).resolve().parent
DASH = (0, (4, 2.5))

INK = "#0F172A"
MUTED = "#475569"
FLOW = "#475569"
BLUE = dict(bg="#EFF6FF", edge="#3B82F6", head="#1D4ED8", solid="#2563EB", ink="#1E3A8A", chip="#DBEAFE")
PURPLE = dict(bg="#F5F3FF", edge="#7C3AED", head="#6D28D9", ink="#4C1D95", chip="#EDE9FE")
ORANGE = dict(bg="#FFF7ED", edge="#F97316", head="#C2410C", box="#FFEDD5", line="#EA580C", ink="#7C2D12")
GREEN = dict(bg="#F0FDF4", edge="#22C55E", head="#15803D", box="#DCFCE7", line="#16A34A", ink="#14532D")
PINK = "#DB2777"

fig = plt.figure(figsize=(W / 10, H / 10), facecolor="white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")


def box(x0, y0, x1, y1, fc, ec, lw=1.6, ls="-", r=1.2, z=2):
    ax.add_patch(
        FancyBboxPatch(
            (x0, y0), x1 - x0, y1 - y0,
            boxstyle=f"round,pad=0,rounding_size={r}",
            fc=fc, ec=ec, lw=lw, ls=ls, zorder=z,
        )
    )


def text(x, y, s, size=11.5, color=INK, weight="normal", ha="center", va="center",
         family=None, z=6, bg=None):
    kw = dict(fontsize=size, color=color, weight=weight, ha=ha, va=va, zorder=z)
    if family:
        kw["family"] = family
    if bg:
        kw["bbox"] = dict(fc=bg, ec="none", pad=1.6, boxstyle="round,pad=0.25")
    ax.text(x, y, s, **kw)


def badge(x, y, s, fc, color, size=10):
    text(x, y, s, size=size, color=color, weight="bold", z=8)
    ax.texts[-1].set_bbox(dict(fc=fc, ec=color, lw=1.0, boxstyle="round,pad=0.3,rounding_size=0.8"))


def arrow(pts, color=FLOW, lw=2.0, ls="-", head=True, z=5):
    path = Path(pts, [Path.MOVETO] + [Path.LINETO] * (len(pts) - 1))
    style = "-|>,head_length=0.7,head_width=0.38" if head else "-"
    ax.add_patch(
        FancyArrowPatch(
            path=path, arrowstyle=style, mutation_scale=15, color=color, lw=lw,
            linestyle=ls, shrinkA=0, shrinkB=0, zorder=z, joinstyle="miter",
        )
    )


# Title
text(3, 85.6, "One RL training step: M* runs rollouts, RLinf trains", 18, INK, "bold", ha="left")
text(3, 82.3, "Target policy: Cosmos3 Nano. Cosmos3 Edge for cheap development runs.",
     12.5, MUTED, ha="left")

# Legend
box(79, 68.6, 117, 80.6, "white", "#CBD5E1", lw=1.2, r=1.0)
legend = [
    ("existing M*", dict(fc=BLUE["solid"], ec=BLUE["head"], ls="-")),
    ("opt-in M* pieces", dict(fc="white", ec=BLUE["solid"], ls=DASH)),
    ("M* control plane (opt-in)", dict(fc=PURPLE["chip"], ec=PURPLE["edge"], ls=DASH)),
    ("trainer (RLinf)", dict(fc=ORANGE["box"], ec=ORANGE["line"], ls="-")),
    ("simulator", dict(fc=GREEN["box"], ec=GREEN["line"], ls="-")),
]
for i, (label, sw) in enumerate(legend):
    y = 78.5 - i * 2.25
    box(81, y - 0.8, 85.2, y + 0.8, lw=1.4, r=0.5, z=3, **sw)
    text(86.4, y, label, 11, INK, ha="left")

# Simulator
box(3, 63, 72, 77.5, GREEN["bg"], GREEN["edge"], r=1.8, z=1)
text(5.5, 74.6, "Simulator", 14, GREEN["head"], "bold", ha="left")
text(5.5, 71.9, "LIBERO, run by", 11.5, GREEN["ink"], ha="left")
text(5.5, 69.8, "RLinf env workers", 11.5, GREEN["ink"], ha="left")
for dx, dy in ((1.6, 1.4), (0.8, 0.7)):
    box(28 + dx, 65.5 + dy, 69 + dx, 74.5 + dy, GREEN["box"], GREEN["line"], lw=1.2, z=2)
box(28, 65.5, 69, 74.5, GREEN["box"], GREEN["line"], lw=1.6, z=3)
text(48.5, 71.6, "N parallel robot task envs", 13, GREEN["ink"], "bold")
text(48.5, 68.4, "camera images + robot state", 12, GREEN["ink"])

# M* rollout engine
box(3, 2.5, 72, 60, BLUE["bg"], BLUE["edge"], r=1.8, z=1)
text(5.5, 57.2, "M* rollout engine", 14, BLUE["head"], "bold", ha="left")
text(5.5, 54.6, "Cosmos3 Nano policy", 11.5, BLUE["ink"], ha="left")

box(27, 47, 51, 56, "white", BLUE["solid"], ls=DASH)
badge(48.4, 56, "opt-in", BLUE["chip"], BLUE["head"])
text(39, 53, "/generate", 13.5, BLUE["ink"], "bold", family="DejaVu Sans Mono")
text(39, 49.7, "rollout mode", 12, BLUE["ink"])

box(5.5, 27, 51, 43.5, BLUE["solid"], BLUE["head"])
text(16.2, 38.6, "Cosmos3 policy", 13.5, "white", "bold")
text(16.2, 35.4, "joint video +", 11.5, "white")
text(16.2, 33.2, "action denoising", 11.5, "white")
box(27.5, 29, 49, 41.5, "white", BLUE["ink"], ls=DASH, z=3)
badge(46.4, 41.5, "opt-in", BLUE["chip"], BLUE["head"])
text(38.25, 37.6, "SDE denoise step", 12.5, BLUE["ink"], "bold")
text(38.25, 34.4, "adds noise", 11.5, BLUE["ink"])
text(38.25, 32.2, "for exploration", 11.5, BLUE["ink"])

box(54, 27, 70.5, 56, "white", BLUE["solid"], ls=DASH)
badge(67.9, 56, "opt-in", BLUE["chip"], BLUE["head"])
text(62.25, 53.2, "Outputs", 13, BLUE["ink"], "bold")
chips = [
    (47.2, 50.8, ["action chunk"], None),
    (39.6, 45.4, ["denoising chain", "+ timesteps"], None),
    (34.0, 37.6, ["action log-probs"], None),
    (28.4, 32.0, ["weight_version"], "DejaVu Sans Mono"),
]
for y0, y1, lines, fam in chips:
    box(55.2, y0, 69.3, y1, BLUE["chip"], BLUE["chip"], lw=0.8, r=0.7, z=3)
    mid = (y0 + y1) / 2
    for j, line in enumerate(lines):
        off = (len(lines) - 1) / 2 - j
        text(62.25, mid + off * 2.2, line, 11, BLUE["ink"], family=fam)

box(5.5, 4.5, 70.5, 19.5, PURPLE["bg"], PURPLE["edge"], ls=DASH, r=1.4)
text(7.8, 16.6, "Admin control plane", 13, PURPLE["head"], "bold", ha="left")
badge(33.6, 16.6, "opt-in", PURPLE["chip"], PURPLE["head"])
steps = [
    (8, 17, ["pause"]),
    (20, 37, ["reload", "every rank idle"]),
    (40, 54.5, ["version bump", "N to N+1"]),
    (57.5, 67.5, ["resume"]),
]
for i, (x0, x1, lines) in enumerate(steps):
    box(x0, 6.5, x1, 12.5, "white", PURPLE["edge"], lw=1.4, r=1.0, z=3)
    for j, line in enumerate(lines):
        off = (len(lines) - 1) / 2 - j
        text((x0 + x1) / 2, 9.5 + off * 2.2, line, 12 if j == 0 else 10.5,
             PURPLE["ink"], "bold" if j == 0 else "normal")
    if i:
        arrow([(steps[i - 1][1] + 0.2, 9.5), (x0 - 0.2, 9.5)], PURPLE["edge"], lw=1.6)

# RLinf trainer
box(82, 2.5, 117, 60, ORANGE["bg"], ORANGE["edge"], r=1.8, z=1)
text(84.5, 57.2, "RLinf trainer", 14, ORANGE["head"], "bold", ha="left")

box(85, 47.5, 114, 53, ORANGE["box"], ORANGE["line"])
text(99.5, 50.25, "rollout buffer", 12.5, ORANGE["ink"], "bold")

box(89, 39.5, 110, 44, ORANGE["box"], ORANGE["line"])
text(99.5, 41.75, "advantage", 12.5, ORANGE["ink"], "bold")

box(84.5, 22.5, 101.5, 35.8, ORANGE["box"], ORANGE["line"])
text(93, 33.3, "GRPO", 12.5, ORANGE["ink"], "bold")
badge(99.2, 35.8, "v1", "white", ORANGE["line"])
for j, line in enumerate(["group-relative", "over K rollouts", "of the same task", "no critic"]):
    text(93, 30.7 - j * 2.15, line, 10.5, ORANGE["ink"])

box(103.5, 22.5, 115, 35.8, "white", "#FDBA74", ls=DASH)
text(109.25, 33.3, "PPO", 12.5, "#9A3412", "bold")
badge(112.4, 35.8, "later", "white", "#C2410C")
text(109.25, 29.2, "value head", 10.5, "#9A3412")
text(109.25, 27.0, "+ GAE", 10.5, "#9A3412")

box(85, 11.8, 114, 19.6, ORANGE["box"], ORANGE["line"])
text(99.5, 17.7, "clipped policy loss", 12.5, ORANGE["ink"], "bold")
text(99.5, 15.3, "ratio = new / old log-prob", 11, ORANGE["ink"])
text(99.5, 13.3, "recomputed from the chain", 11, ORANGE["ink"])

box(85, 3.8, 114, 10.0, ORANGE["box"], ORANGE["line"])
text(99.5, 8.0, "update weights", 12.5, ORANGE["ink"], "bold")
text(99.5, 5.6, "FSDP actor", 11, ORANGE["ink"])

# Arrows inside groups
arrow([(39, 47), (39, 43.6)])
arrow([(51, 35.25), (53.9, 35.25)])
arrow([(99.5, 47.5), (99.5, 44.1)], ORANGE["line"])
arrow([(95, 39.5), (93, 35.9)], ORANGE["line"])
arrow([(103.5, 39.5), (106.5, 35.9)], "#FDBA74", ls=(0, (3, 2)))
arrow([(93, 22.5), (95.5, 19.7)], ORANGE["line"])
arrow([(109.25, 22.5), (106, 19.7)], "#FDBA74", ls=(0, (3, 2)))
arrow([(99.5, 11.8), (99.5, 10.1)], ORANGE["line"])

# Arrows between groups
arrow([(39, 65.5), (39, 56.1)], GREEN["line"], lw=2.4)
text(37.8, 61.3, "observation", 11, GREEN["head"], "bold", ha="right", bg="white")

arrow([(62.25, 56), (62.25, 65.4)], BLUE["solid"], lw=2.4)
text(63.6, 61.3, "action chunk", 11, BLUE["head"], "bold", ha="left", bg="white")

arrow([(69, 66.8), (106, 66.8), (106, 53.1)], GREEN["line"], lw=2.4)
text(87.5, 64.6, "reward (task success)", 11, GREEN["head"], "bold", bg="white")

arrow([(70.5, 50.25), (84.9, 50.25)], BLUE["solid"], lw=2.4)
text(77, 54.7, "chain,\nlog-probs,\nversion", 10.5, BLUE["head"], "bold", bg="white")

arrow([(85, 6.8), (70.6, 6.8)], ORANGE["line"], lw=2.4)
text(77.7, 10.4, "checkpoint\non disk", 10.5, ORANGE["head"], "bold", bg="white")

arrow([(14, 19.5), (14, 26.9)], PURPLE["edge"], lw=2.4)
text(15.4, 23.2, "new weights (version N+1)", 11, PURPLE["head"], "bold", ha="left")

# Parity check between M* log-probs and the trainer's recompute
dot = (0, (1.2, 1.6))
arrow([(69.3, 35.8), (77.7, 35.8), (77.7, 31.2)], PINK, lw=2.0, ls=dot, head=False)
arrow([(77.7, 24.6), (77.7, 15.3), (85, 15.3)], PINK, lw=2.0, ls=dot, head=False)
for px, py in ((69.3, 35.8), (85, 15.3)):
    ax.plot(px, py, "o", ms=6, color=PINK, zorder=7)
text(77.7, 27.9, "parity\ncheck", 11, PINK, "bold", z=8)
ax.texts[-1].set_bbox(dict(fc="#FDF2F8", ec=PINK, lw=1.4, boxstyle="round,pad=0.45,rounding_size=0.9"))

fig.savefig(OUT / "rl-rollout-flow.svg", facecolor="white")
fig.savefig(OUT / "rl-rollout-flow.png", dpi=200, facecolor="white")
