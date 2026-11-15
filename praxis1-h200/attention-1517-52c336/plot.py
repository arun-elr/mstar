#!/usr/bin/env python3
"""Plot the pinned synthetic H200 attention resource measurements."""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
data = json.loads((HERE / "provenance.json").read_text())
assert data["source_revision"] == "52c336af9f19cee9c3a2cfebfe2c45078a6ec7af"
assert data["gpu"] == "NVIDIA H200"
assert data["samples"] == 20 and data["warmups"] == 5

names = [
    ("encoder_cond", "Encoder cond"),
    ("encoder_blank", "Encoder blank"),
    ("decoder", "Decoder"),
    ("expert_cross", "Expert cross"),
    ("expert_self", "Expert self"),
]
eager = [data["values"][name]["eager_wrapper"] for name, _ in names]
replay = [data["values"][name]["captured_replay"] for name, _ in names]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, ax = plt.subplots(figsize=(9.5, 5.4), dpi=180)
fig.patch.set_facecolor("white")
y = np.arange(len(names))[::-1]
ax.scatter(eager, y + 0.14, s=58, color="#136f8a", label="Eager wrapper", zorder=3)
ax.scatter(replay, y - 0.14, s=58, color="#ba562d", label="Captured replay", zorder=3)
for x, yi in zip(eager, y + 0.14, strict=True):
    ax.annotate(f"{x:.3f}", (x, yi), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=9)
for x, yi in zip(replay, y - 0.14, strict=True):
    ax.annotate(f"{x:.3f}", (x, yi), xytext=(5, 0),
                textcoords="offset points", va="center", fontsize=9)
ax.set_xscale("log")
ax.set_xlim(0.009, 4.1)
ax.set_xticks([0.01, 0.03, 0.1, 0.3, 1, 3])
ax.set_xticklabels(["0.01", "0.03", "0.1", "0.3", "1", "3"])
ax.set_yticks(y, [label for _, label in names])
ax.set_ylim(-0.65, 4.65)
ax.grid(axis="x", color="#dde2e6", linewidth=0.8)
ax.set_axisbelow(True)
ax.spines[["top", "right", "left"]].set_visible(False)
ax.tick_params(axis="y", length=0)
ax.set_xlabel("Warm GPU p50 (ms, log scale)")
ax.legend(frameon=False, loc="upper right", ncol=2)
fig.suptitle("Synthetic attention resource timing on one H200",
             x=0.1, ha="left", fontsize=14, fontweight="bold")
fig.text(0.1, 0.91, "PR #1517 resource head 52c336af9f; BF16 QKV; 20 samples after 5 warmups",
         fontsize=9, color="#4d5961")
fig.text(0.1, 0.07,
         "Wrapper and replay do different work; expert cross replay pads Q from 30 to 3,584.",
         fontsize=9, color="#4d5961")
fig.text(0.1, 0.04, "Synthetic resource timing, not checkpoint or serving latency.",
         fontsize=9, color="#4d5961")
fig.subplots_adjust(left=0.21, right=0.96, top=0.85, bottom=0.20)

fig.savefig(HERE / "pr1517-attention-wrapper.png",
            facecolor="white", bbox_inches="tight")
plt.close(fig)
