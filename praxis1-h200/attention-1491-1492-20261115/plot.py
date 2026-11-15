#!/usr/bin/env python3
"""Plot the recorded synthetic H200 attention wrapper measurements."""

import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
CASES = [
    ("encoder_cond", "Encoder, conditional"),
    ("encoder_blank", "Encoder, blank"),
    ("decoder", "Decoder"),
]
COLORS = {"eager": "#136f8a", "replay": "#ba562d"}


def plot(pr: int, provenance: dict) -> None:
    raw_file = HERE / f"pr{pr}-raw.json"
    assert hashlib.sha256(raw_file.read_bytes()).hexdigest() == provenance["raw_sha256"]
    report = json.loads(raw_file.read_text())
    assert report["environment"]["revision"] == provenance["source_revision"]
    assert report["environment"]["gpu"] == "NVIDIA H200"
    assert report["environment"]["samples"] == 20
    assert report["environment"]["warmups"] == 5

    measured = {case["name"]: case for case in report["cases"]}
    fig, ax = plt.subplots(figsize=(8.8, 4.4), dpi=180)
    fig.patch.set_facecolor("white")
    y = np.arange(len(CASES))[::-1]
    for offset, key, label in [(0.14, "eager", "Eager wrapper"), (-0.14, "replay", "Captured replay")]:
        values = []
        for name, _ in CASES:
            case = measured[name]
            record = (
                case["eager"]["warm"]["wrapper"]
                if key == "eager"
                else case["captured"]["warm"]["replay"]
            )
            assert len(record["raw_gpu_ms"]) == 20
            assert abs(float(np.median(record["raw_gpu_ms"])) - record["gpu_ms"]["p50"]) < 1e-7
            values.append(record["gpu_ms"]["p50"])
        ax.scatter(values, y + offset, s=65, color=COLORS[key], label=label, zorder=3)
        for value, ypos in zip(values, y + offset, strict=True):
            ax.annotate(f"{value:.3f} ms", (value, ypos), xytext=(8, 0),
                        textcoords="offset points", va="center", fontsize=9)

    ax.set_xlim(0, 2.8)
    ax.set_ylim(-0.55, 2.55)
    ax.set_yticks(y, [label for _, label in CASES])
    ax.set_xlabel("Warm GPU p50 (ms)")
    ax.grid(axis="x", color="#dde2e6", linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(axis="y", length=0)
    ax.legend(frameon=False, loc="upper right", ncol=2)
    fig.suptitle(f"PR #{pr}: synthetic attention resource timing", x=0.12,
                 ha="left", fontsize=14, fontweight="bold")
    fig.text(0.12, 0.88,
             f"One H200, BF16 QKV, 20 samples after 5 warmups, measured at {provenance['source_revision'][:10]}",
             fontsize=9, color="#4d5961")
    fig.text(0.12, 0.04,
             "Wrapper timing only. Eager and captured paths do different work; neither is serving latency.",
             fontsize=9, color="#4d5961")
    fig.subplots_adjust(left=0.24, right=0.95, top=0.78, bottom=0.19)
    fig.savefig(HERE / f"pr{pr}-attention-wrapper.png", facecolor="white", bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    provenance = json.loads((HERE / "provenance.json").read_text())
    for pr in (1491, 1492):
        plot(pr, provenance[str(pr)])


if __name__ == "__main__":
    main()
