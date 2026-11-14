"""Plot the pinned, matched-backend Praxis DROID tower diagnostic."""

import json
import statistics
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
REFERENCE = "reference-triton"
MSTAR = "mstar-triton"
REF_COLOR = "#2369a8"
MSTAR_COLOR = "#d06328"


def tower_median(data, *, call, euler_step, tower, branch):
    matches = [
        row
        for row in data["tower_medians"]
        if row["call"] == call
        and row["euler_step"] == euler_step
        and row["tower"] == tower
        and row["branch"] == branch
    ]
    assert len(matches) == 1 and matches[0]["n"] == 4, matches
    return matches[0]["median_cuda_ms"]


def event_bars(ax, title, labels, values, colors, *, limit=90):
    bars = ax.bar(range(len(labels)), values, color=colors, width=0.68)
    ax.set_title(title, loc="left", fontsize=12, fontweight="semibold")
    ax.set_xticks(range(len(labels)), labels, fontsize=9)
    ax.set_ylim(0, limit)
    ax.set_ylabel("CUDA event (ms)")
    ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.grid(axis="y", color="#e6e9ed")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)


def main():
    clients = json.loads((HERE / "client-latencies.json").read_text())
    ref = json.loads((HERE / f"{REFERENCE}-tower-summary.json").read_text())
    mstar = json.loads((HERE / f"{MSTAR}-tower-summary.json").read_text())
    assert ref["source_pin"] == "68f4d7a254b4bf41951d2e97f44b07f6cac2bf69"
    assert mstar["source_pin"] == "5902054c0f9fa057a80ae5a7e15e67f19d4bcde2"
    assert ref["requests_measured"] == mstar["requests_measured"] == 12
    assert clients[REFERENCE]["warmup_sessions"] == clients[MSTAR]["warmup_sessions"] == 1

    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
    fig = plt.figure(figsize=(15.8, 9.3), facecolor="white")
    grid = fig.add_gridspec(
        2, 3, height_ratios=[1.0, 1.05], left=0.06, right=0.98,
        top=0.88, bottom=0.17, hspace=0.37, wspace=0.25,
    )
    ax = fig.add_subplot(grid[0, :])
    ax.set_title("A. Sequential client round trip", loc="left", fontsize=12, fontweight="semibold")
    x = np.arange(3)
    width = 0.32
    for offset, arm, color, label in [
        (-width / 2, REFERENCE, REF_COLOR, "Reference, Triton"),
        (width / 2, MSTAR, MSTAR_COLOR, "M*, Triton"),
    ]:
        medians = [
            statistics.median(clients[arm]["measured_round_trip_ms_by_call"][str(call)])
            for call in range(3)
        ]
        assert all(
            len(clients[arm]["measured_round_trip_ms_by_call"][str(call)]) == 4
            for call in range(3)
        )
        bars = ax.bar(x + offset, medians, width, label=label, color=color)
        ax.bar_label(bars, fmt="%.1f", padding=3, fontsize=9)
    ax.set_xticks(x, ["Cold: 21 rows", "Warm 1: 24 rows", "Warm 2: 24 rows"])
    ax.set_ylabel("Median client latency (ms)")
    ax.set_ylim(0, 620)
    ax.legend(frameon=False, loc="upper left", ncol=2)
    ax.grid(axis="y", color="#e6e9ed")
    ax.set_axisbelow(True)
    ax.spines[["top", "right"]].set_visible(False)

    def r(step, tower, branch):
        return tower_median(ref, call=1, euler_step=step, tower=tower, branch=branch)

    def m(step, tower, branch):
        return tower_median(mstar, call=1, euler_step=step, tower=tower, branch=branch)

    event_bars(
        fig.add_subplot(grid[1, 0]),
        "B. Warm 1, E0 encoder",
        ["Ref pair\ncond + uncond", "M* solo\ncond", "M* solo\nblank"],
        [r(0, "encoder", "paired"), m(0, "encoder", "cond"), m(0, "encoder", "blank")],
        [REF_COLOR, MSTAR_COLOR, MSTAR_COLOR],
    )
    event_bars(
        fig.add_subplot(grid[1, 1]),
        "C. Warm 1, E0 decoder",
        ["Ref\ncond", "Ref\nuncond", "M*\ncond", "M*\nblank"],
        [
            r(0, "decoder", "cond"),
            r(0, "decoder", "uncond"),
            m(0, "decoder", "cond"),
            m(0, "decoder", "blank"),
        ],
        [REF_COLOR, REF_COLOR, MSTAR_COLOR, MSTAR_COLOR],
    )
    event_bars(
        fig.add_subplot(grid[1, 2]),
        "D. Warm 1, E1 encoder",
        ["Ref\ncond", "M*\ncond"],
        [r(1, "encoder", "cond"), m(1, "encoder", "cond")],
        [REF_COLOR, MSTAR_COLOR],
    )

    fig.suptitle(
        "Praxis DROID timing on one H200, matched Triton backend",
        y=0.96, fontsize=17, fontweight="semibold",
    )
    fig.text(
        0.5,
        0.035,
        "Four measured sessions after one warmup; two Euler steps; synthetic observations. "
        "Tower events can overlap and must not be summed into request latency.",
        ha="center",
        fontsize=9,
        color="#424b55",
    )
    fig.savefig(HERE / "tower-diagnostic.png", dpi=180, facecolor="white")
    svg = HERE / "tower-diagnostic.svg"
    fig.savefig(svg, facecolor="white")
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")


if __name__ == "__main__":
    main()
