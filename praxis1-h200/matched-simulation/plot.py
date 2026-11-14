#!/usr/bin/env python3
"""Plot the pinned, paired H200 Praxis simulation diagnostic."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


OUT = Path(__file__).resolve().parent
ROOT = OUT / "raw"
REF = "68f4d7a254b4bf41951d2e97f44b07f6cac2bf69"
MSTAR = "5902054c0f9fa057a80ae5a7e15e67f19d4bcde2"
TASKS = (
    "lift_cube", "cube_on_plate", "block_in_bowl", "block_out_of_bowl",
    "move_block", "stack_block", "press_button",
)
BLUE = "#1764a0"
ORANGE = "#cc5b2d"
GRAY = "#667085"


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def trace(arm: str):
    path = ROOT / f"{arm}-lift-original-scene0" / f"{arm}-scene0-sample0.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    ticks = [row for row in rows if row["kind"] == "tick"]
    calls = [row for row in rows if row["kind"] == "call"]
    summary = next(row["result"] for row in rows if row["kind"] == "summary")
    assert len(calls) == 7
    assert len(ticks) == summary["ticks"]
    assert [row["tick"] for row in ticks] == list(range(1, len(ticks) + 1))
    assert all(row["required_hold_ticks"] == 4 for row in ticks)
    return path, ticks, calls, summary


def episodes(arm: str):
    paths = sorted((ROOT / f"{arm}-original" / "suite").glob("*/episode_*.json"))
    rows = {(row["task"], row["seed"]): row for p in paths if (row := json.loads(p.read_text()))}
    assert len(paths) == len(rows) == 7
    assert set(rows) == {(task, i) for i, task in enumerate(TASKS)}
    return paths, rows


def setup():
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.facecolor": "white",
        "savefig.facecolor": "white",
        "svg.fonttype": "none",
        "svg.hashsalt": "praxis-droid-matched-sim-5902054c0f",
    })


def save(fig, stem: str):
    svg = OUT / f"{stem}.svg"
    fig.savefig(svg, bbox_inches="tight", metadata={"Date": None})
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    fig.savefig(OUT / f"{stem}.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_lift(ref_ticks, mstar_ticks, ref_summary, mstar_summary):
    first_height_gap = next(
        a["tick"] for a, b in zip(ref_ticks, mstar_ticks)
        if abs(a["lift_m"] - b["lift_m"]) > 1e-4
    )
    mstar_hold = [row["tick"] for row in mstar_ticks if row["above_threshold"]]
    assert first_height_gap == 71 and mstar_hold == [163, 164, 165]
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    for ticks, label, color in (
        (ref_ticks, "Reference", BLUE),
        (mstar_ticks, "M* Rust", ORANGE),
    ):
        ax.plot(
            [row["tick"] for row in ticks],
            [100 * row["lift_m"] for row in ticks],
            color=color, linewidth=2.25, label=label,
        )
        hold = [row for row in ticks if row["above_threshold"]]
        ax.scatter([row["tick"] for row in hold], [100 * row["lift_m"] for row in hold],
                   color=color, s=10, zorder=4)
    ax.axhline(5, color=GRAY, linestyle="--", linewidth=1.2)
    ax.text(2, 5.15, "5 cm lift threshold; 4 consecutive ticks required", color=GRAY, fontsize=9)
    ax.axvline(first_height_gap, color="#b7bcc6", linestyle=":", linewidth=1)
    ax.text(73, 9.8, f"first >0.1 mm height difference\ntick {first_height_gap}", color=GRAY, fontsize=8.5)
    ax.text(105, 11.6, f"reference success tick {ref_summary['success_tick']}", color=BLUE, fontsize=9)
    ax.text(6, 2.1, f"M* reaches threshold at ticks {mstar_hold[0]}–{mstar_hold[-1]}\n(3 of 4 required)", color=ORANGE, fontsize=9)
    ax.set_xlim(0, 167)
    ax.set_ylim(-0.2, 13.3)
    ax.set_xlabel("Simulation tick (15 Hz)")
    ax.set_ylabel("Cube lift above rest position (cm)")
    ax.set_title("Praxis DROID lift_cube, scene 0, sampler seed 0", loc="left", weight="bold", pad=12)
    ax.grid(axis="y", color="#e6e9ee", linewidth=0.8)
    ax.legend(loc="upper left", frameon=False, ncol=2)
    fig.text(0.11, 0.005,
             "One matched episode. Reference: effective Triton attention; M*: Flex Triton, ragged kernel choice unverified.",
             fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "lift_cube_scene0")


def plot_outcomes(reference, mstar):
    fig, ax = plt.subplots(figsize=(9.5, 4.5))
    labels = [f"{task.replace('_', ' ')}  ·  scene {i}" for i, task in enumerate(TASKS)]
    for i, task in enumerate(TASKS):
        y = 6 - i
        for x, rows, color in ((0, reference, BLUE), (1, mstar, ORANGE)):
            ok = rows[(task, i)]["success"]
            ax.scatter([x], [y], s=160 if ok else 190, marker="o" if ok else "x",
                       color=color, linewidth=2.7 if not ok else 0, zorder=3)
            if ok:
                ax.text(x, y, "✓", ha="center", va="center", color="white", fontsize=11,
                        weight="bold", zorder=4)
    ax.set_yticks(range(7), labels[::-1])
    ax.set_xticks([0, 1], ["Reference  7/7", "M* Rust  6/7"])
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(-0.6, 6.6)
    ax.grid(axis="y", color="#eceef2", linewidth=0.7)
    ax.tick_params(axis="both", length=0, pad=10)
    ax.set_title("Praxis DROID seven-task guard, sampler seed 0", loc="left", weight="bold", pad=12)
    fig.text(0.16, 0.015,
             "One scene per task (scene 0–6), paired across arms. Descriptive diagnostic, not a success-rate estimate.",
             fontsize=8, color=GRAY)
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, "seven_task_guard")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    setup()
    ref_path, ref_ticks, ref_calls, ref_summary = trace("reference")
    ms_path, ms_ticks, ms_calls, ms_summary = trace("mstar")
    assert ref_summary["success"] and not ms_summary["success"]
    assert ref_summary["success_tick"] == 135
    assert all(ref_calls[0]["observation"][k] == ms_calls[0]["observation"][k]
               for k in ref_calls[0]["observation"])
    ref_paths, ref_rows = episodes("reference")
    ms_paths, ms_rows = episodes("mstar")
    assert sum(r["success"] for r in ref_rows.values()) == 7
    assert sum(r["success"] for r in ms_rows.values()) == 6
    assert ms_rows[("lift_cube", 0)]["success"] is False
    backend_gates = (ROOT / "backend-gates.txt").read_text()
    assert "reference trace effective Triton modules: 20" in backend_gates
    assert "M* effective flex backend: TRITON" in backend_gates
    for key in ref_rows:
        a, b = ref_rows[key], ms_rows[key]
        assert a["prompt"] == b["prompt"] and a["sampler_seed"] == b["sampler_seed"] == 0
        assert b["sample_seed_base"] == 1000 * key[1]
        assert b["request_seed_sent"] and b["restarts"] == b["bad_view_ticks"] == 0
    plot_lift(ref_ticks, ms_ticks, ref_summary, ms_summary)
    plot_outcomes(ref_rows, ms_rows)
    input_paths = [
        ref_path, ms_path,
        ref_path.with_suffix(".npz"), ms_path.with_suffix(".npz"),
        *ref_paths, *ms_paths,
        ROOT / "reference-original/summary.json",
        ROOT / "mstar-original/summary.json",
        ROOT / "reference-served.yaml", ROOT / "mstar-served-original.yaml",
        ROOT / "reference-working-tree.actual.diff",
        ROOT / "reference-working-tree.actual.status",
        ROOT / "backend-gates.txt", ROOT / "source-config.sha256",
        ROOT / "model-artifacts.sha256", ROOT / "runner-source.sha256",
        ROOT / "operator-source.sha256", ROOT / "analysis.json",
    ]
    lines = [
        "Matched Praxis DROID H200 diagnostic, 2026-11-14 UTC",
        f"Reference commit: {REF}", f"M* commit: {MSTAR}",
        "Reference encoder/decoder: effective Triton, verified from module inspection.",
        "M* expert Flex: TRITON, verified from runtime import; ragged backend not observed.",
        "Original seven units: (lift_cube,0), (cube_on_plate,1), (block_in_bowl,2),",
        "(block_out_of_bowl,3), (move_block,4), (stack_block,5), (press_button,6).",
        "Sampler seed 0; M* session seed base = 1000 * scene seed.",
        "Lift trace: reference succeeds at tick 135 and runs through tick 150; M* fails at tick 165.",
        "Caveat: one episode per task; these are descriptive outcomes, not an estimated pass rate.",
        "Caveat: different attention kernels and Torch/CUDA builds prevent strict numeric parity attribution.",
        "Input SHA256:",
        *(f"{p.relative_to(ROOT)}  {digest(p)}" for p in input_paths),
    ]
    (OUT / "PROVENANCE.txt").write_text("\n".join(lines) + "\n")
    for name in ("lift_cube_scene0", "seven_task_guard"):
        print(OUT / f"{name}.png")
        print(OUT / f"{name}.svg")


if __name__ == "__main__":
    main()
