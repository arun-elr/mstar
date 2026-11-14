"""Plot counted cold request latency from two benchmark_remote_session runs."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

RECIPE = (
    "num_steps",
    "t_shift",
    "cfg_scale",
    "cfg_video_scale",
    "max_context_chunks",
    "skip_video_decoder",
)


def checked_pair(reference: dict, mstar: dict) -> tuple[dict, dict]:
    """Require one H200, one cold recipe, and auditable server revisions."""
    if reference.get("mode") != "reference" or mstar.get("mode") not in (
        "mstar",
        "mstar-openpi",
    ):
        raise ValueError("pass the reference JSON first and the M* JSON second")
    if reference.get("observation_sha256") != mstar.get("observation_sha256"):
        raise ValueError("the two runs used different observations")
    hardware = reference.get("gpu")
    if (
        not isinstance(hardware, str)
        or "H200" not in hardware
        or mstar.get("gpu") != hardware
    ):
        raise ValueError("the two runs must record the same H200 GPU")
    ref_meta, mstar_meta = reference["server_meta"], mstar["server_meta"]
    if ref_meta.get("context_mode") != "stateless":
        raise ValueError("the reference must use context_mode=stateless")
    for name, meta in (("reference", ref_meta), ("M*", mstar_meta)):
        if not isinstance(meta.get("code_version"), str) or not meta["code_version"]:
            raise ValueError(f"{name} lacks a served code revision")
        if type(meta.get("skip_video_decoder")) is not bool:
            raise ValueError(f"{name} must declare skip_video_decoder as a bool")
    for key in RECIPE:
        if ref_meta.get(key) != mstar_meta.get(key):
            raise ValueError(f"the server recipes differ at {key}")
    ref_checkpoint = str(ref_meta["checkpoint"]).rstrip("/").split("/")[-1]
    mstar_checkpoint = str(mstar_meta["checkpoint"]).rstrip("/").split("/")[-1]
    if ref_checkpoint != mstar_checkpoint:
        raise ValueError("the server checkpoint names differ")
    return reference["summary"], mstar["summary"]


def plot(reference: dict, mstar: dict, out: Path) -> None:
    import matplotlib.pyplot as plt

    ref, port = checked_pair(reference, mstar)
    summaries = (ref, port)
    names = ("Reference", "M* Positronic" if mstar["mode"] == "mstar" else "M* openpi")
    x = (0, 1)
    fig, ax = plt.subplots(figsize=(10, 6))
    medians = [item["p50_ms"] for item in summaries]
    tails = [item["p95_ms"] for item in summaries]
    median_bars = ax.bar(
        [v - 0.18 for v in x],
        medians,
        width=0.34,
        color="#2b6cb0",
        label="Median (p50)",
    )
    tail_bars = ax.bar(
        [v + 0.18 for v in x],
        tails,
        width=0.34,
        color="#c05621",
        label="95th percentile (p95)",
    )
    for bars in (median_bars, tail_bars):
        for bar in bars:
            ax.annotate(
                f"{bar.get_height():.1f} ms",
                (bar.get_x() + bar.get_width() / 2, bar.get_height()),
                xytext=(0, 4),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=9,
            )
    ax.set_ylim(0, max(tails + medians) * 1.17)
    ax.set_xticks(
        x,
        [
            f"{name}\nn={item['n']} counted calls"
            for name, item in zip(names, summaries, strict=True)
        ],
    )
    ax.set_ylabel("Round trip latency per cold request (ms)")
    protocol = "Positronic" if mstar["mode"] == "mstar" else "openpi"
    ax.set_title(f"Praxis DROID cold debug: reference vs M* {protocol}")
    ax.legend(loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    ax.set_axisbelow(True)

    recipe = reference["server_meta"]
    gpu = reference["gpu"].splitlines()[0]
    lines = [
        f"Hardware: {gpu}",
        f"Served revisions: reference {recipe['code_version']} | M* {mstar['server_meta']['code_version']}",
        (
            f"Recipe: {recipe['num_steps']} Euler steps, t_shift={recipe['t_shift']}, "
            f"cfg_scale={recipe['cfg_scale']}, cfg_video_scale={recipe['cfg_video_scale']}, "
            f"max_context_chunks={recipe['max_context_chunks']}, "
            f"skip_video_decoder={recipe['skip_video_decoder']}"
        ),
        "Sequential fixed raw observations; one warmup session excluded. No closed-loop or concurrent-load claim.",
    ]
    if mstar["mode"] == "mstar":
        lines.append(
            "M* Positronic sampling seed is uncontrolled; output parity is not established."
        )
    else:
        lines.append(
            "M* openpi sends explicit seeds in its cold debugging protocol; output parity is not established."
        )
    for i, line in enumerate(lines):
        fig.text(0.02, 0.23 - i * 0.04, line, ha="left", fontsize=8)
    fig.tight_layout(rect=(0, 0.25, 1, 1))
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("reference", type=Path)
    parser.add_argument("mstar", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    plot(
        json.loads(args.reference.read_text()),
        json.loads(args.mstar.read_text()),
        args.out,
    )


if __name__ == "__main__":
    main()
