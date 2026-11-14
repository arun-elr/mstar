"""Render the matched Praxis-1 MoE combine ABBA chart from summary.json."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parent
data = json.loads((ROOT / "summary.json").read_text())
cases = [("cold", "Cold\n21 rows"), ("warm_1", "Warm call 1\n24 rows"), ("warm_2", "Warm call 2\n24 rows")]
base_arms = ("base1", "base2")
cand_arms = ("cand1", "cand2")
colors = ("#334f71", "#e48747")

fig, axes = plt.subplots(1, 3, figsize=(11.6, 4.7), sharey=True)
fig.patch.set_facecolor("white")
for ax, (case, label) in zip(axes, cases, strict=True):
    means = [
        data["headline"][case]["base_mean_arm_p50_ms"],
        data["headline"][case]["candidate_mean_arm_p50_ms"],
    ]
    ax.bar((0, 1), means, color=colors, width=0.58, zorder=2)
    for idx, arms in enumerate((base_arms, cand_arms)):
        values = [data["arms"][arm]["calls"][case]["p50_ms"] for arm in arms]
        ax.plot([idx, idx], [min(values), max(values)], color="#1b2632", lw=1.6, zorder=3)
        ax.scatter([idx, idx], values, color="white", edgecolor="#1b2632", s=27, zorder=4)
    for idx, value in enumerate(means):
        ax.text(idx, value + 13, f"{value:.1f}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks((0, 1), ("Eager", "Fused"))
    ax.set_title(f"{label}\n−{data['headline'][case]['reduction_percent']:.2f}%", fontsize=11, pad=14)
    ax.set_ylim(0, 625)
    ax.set_xlim(-0.55, 1.55)
    ax.grid(axis="y", alpha=0.18, zorder=0)
    ax.spines[["top", "right"]].set_visible(False)
    ax.tick_params(axis="x", length=0)
axes[0].set_ylabel("Client round trip p50 (ms)")
fig.suptitle("Praxis-1 DROID: self-contained fused MoE combine on H200", fontsize=14, y=0.98)
fig.text(
    0.5,
    0.04,
    "ABBA: 2 fresh servers per mode; 4 measured sessions per server. Bars = mean arm p50; dots = arm p50.\n"
    "Downstream 5902054c0f candidate copy, Python frontend, eager -O0, no graph capture.",
    ha="center",
    va="center",
    fontsize=8.5,
    color="#455260",
)
fig.subplots_adjust(left=0.075, right=0.985, top=0.74, bottom=0.21, wspace=0.26)
fig.savefig(ROOT / "moe-combine-h200-abba.png", dpi=200, facecolor="white")
plt.close(fig)
