"""Plot the four fresh-server H200 MoE combine diagnostic arms."""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


root = Path(__file__).parent
arms = {
    name: json.loads((root / f"mstar-590-moe-ab-{name}.json").read_text())
    for name in ("off1", "ref1", "off2", "ref2")
}
assert len({a["script_sha256"] for a in arms.values()}) == 1
assert len({tuple(a["observation_sha256"]) for a in arms.values()}) == 1
assert all(a["sessions"] == 5 and a["warmup_sessions"] == 1 for a in arms.values())

phase = ["Cold (21 rows)", "Warm 1 (24 rows)", "Warm 2 (24 rows)"]
off = np.array([[arms[f"off{i}"]["summary"][str(j)]["p50_ms"] for j in range(3)] for i in (1, 2)])
ref = np.array([[arms[f"ref{i}"]["summary"][str(j)]["p50_ms"] for j in range(3)] for i in (1, 2)])
fig, ax = plt.subplots(figsize=(9.2, 4.9), layout="constrained")
x = np.arange(3)
width = 0.30
for values, dx, color, label in (
    (off, -width / 2, "#457b9d", "M* default combine"),
    (ref, width / 2, "#2a9d8f", "Pinned reference fused combine"),
):
    means = values.mean(axis=0)
    bars = ax.bar(x + dx, means, width, color=color, label=label)
    for arm in values:
        ax.scatter(x + dx, arm, color="white", edgecolor="#203040", linewidth=0.8, s=22, zorder=4)
    ax.bar_label(bars, labels=[f"{v:.0f}" for v in means], padding=3, fontsize=9)

for j in range(3):
    gain = 100 * (1 - ref[:, j].mean() / off[:, j].mean())
    ax.text(j, max(off[:, j].mean(), ref[:, j].mean()) + 37,
            f"{gain:.1f}% lower", ha="center", fontsize=9, color="#225c52")
ax.set_xticks(x, phase)
ax.set_ylabel("Median request latency per arm (ms)")
ax.set_ylim(0, 640)
ax.set_title("Praxis DROID on H200: MoE combine diagnostic")
ax.legend(frameon=False, loc="upper left", fontsize=9)
ax.text(
    0, -0.23,
    "Two fresh-server runs per mode; 4 measured sessions after 1 warmup in each arm. "
    "Dots show arm medians.\n"
    "Exact same observations, seeds, commands and M* 5902054c0f source. "
    "Python frontend, -O0, no graph capture; two Euler steps.",
    transform=ax.transAxes, fontsize=8, va="top",
)
fig.savefig(root / "moe-combine-ab-h200-5902054c0f.png", dpi=180, bbox_inches="tight")
fig.savefig(root / "moe-combine-ab-h200-5902054c0f.svg", bbox_inches="tight")
