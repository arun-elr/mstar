#!/usr/bin/env python3
"""Plot paired #735 metrics diagnostics from benchmark.runner results.json.

Manifest CSV columns: block,arm,path. Rows are in actual startup order.
Use four paired off/full blocks for a preliminary ABBA check, or complete
off/api/api_data/full blocks for the stage diagnostic. Label the frontend
explicitly because Python and Rust serving results are not interchangeable.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ARMS = ("off", "api", "api_data", "full")
COLORS = {"off": "#334155", "api": "#0f766e", "api_data": "#2563eb", "full": "#b45309"}


def write_csv(path: Path, rows: list[dict], fields: list[str]) -> None:
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_runs(manifest: Path) -> list[dict]:
    rows: list[dict] = []
    seen: set[tuple[str, str]] = set()
    seen_paths: set[Path] = set()
    with manifest.open(newline="") as f:
        source = list(csv.DictReader(f))
    if not source:
        raise ValueError("manifest is empty")
    for order, item in enumerate(source, 1):
        block, arm = item["block"].strip(), item["arm"].strip()
        if arm not in ARMS:
            raise ValueError(f"row {order}: unknown arm {arm!r}")
        if (block, arm) in seen:
            raise ValueError(f"block {block}: duplicate {arm} run")
        seen.add((block, arm))
        result_path = Path(item["path"])
        if not result_path.is_absolute():
            result_path = manifest.parent / result_path
        result_path = result_path.resolve()
        if result_path in seen_paths:
            raise ValueError(f"row {order}: repeated result file {result_path}")
        seen_paths.add(result_path)
        data = json.loads(result_path.read_text())
        if data.get("failed") != 0 or data.get("completed") != data.get("num_requests"):
            raise ValueError(f"{result_path}: incomplete requests")
        if data.get("num_requests") != 2600 or data.get("num_warmup") != 4:
            raise ValueError(f"{result_path}: expected 2600 measured requests and 4 in-run warmups")
        if len(data.get("per_request", [])) != data["completed"]:
            raise ValueError(f"{result_path}: per-request records do not match completed count")
        agg = data["aggregate"]
        tokens = int(agg["total_text_tokens"])
        if tokens != int(data["completed"]) * 256:
            raise ValueError(f"{result_path}: expected 256 output tokens per request, got {tokens}")
        throughput = float(agg["text_token_throughput"])
        if not math.isfinite(throughput) or throughput <= 0:
            raise ValueError(f"{result_path}: invalid throughput {throughput}")
        wall_time = float(data["wall_time_s"])
        if not math.isfinite(wall_time) or wall_time <= 0 or not math.isclose(
            throughput, tokens / wall_time, rel_tol=1e-3
        ):
            raise ValueError(f"{result_path}: throughput does not match tokens / wall time")
        text_ttft = agg.get("ttft_s", {}).get("text") or {}
        text_itl = agg.get("itl_s", {}).get("text") or {}
        rows.append({
            "start": order, "block": block, "arm": arm, "path": item["path"],
            "throughput_tok_s": throughput,
            "completed": int(data["completed"]), "total_text_tokens": tokens,
            "wall_time_s": wall_time,
            "ttft_p50_ms": 1000 * float(text_ttft.get("p50", float("nan"))),
            "ttft_p95_ms": 1000 * float(text_ttft.get("p95", float("nan"))),
            "itl_p50_ms": 1000 * float(text_itl.get("p50", float("nan"))),
            "jct_p95_ms": float(data.get("jct_p95_ms", float("nan"))),
            "model": data.get("model"), "request_type": data.get("request_type"),
            "profiling_type": data.get("profiling_type"),
            "max_concurrency": agg.get("max_concurrency"),
            "total_input_tokens": agg.get("total_input_tokens"),
        })
    expected = (rows[0]["model"], rows[0]["request_type"], rows[0]["profiling_type"],
                rows[0]["max_concurrency"], rows[0]["completed"], rows[0]["total_input_tokens"])
    for row in rows[1:]:
        actual = (row["model"], row["request_type"], row["profiling_type"],
                  row["max_concurrency"], row["completed"], row["total_input_tokens"])
        if actual != expected:
            raise ValueError(f"{row['path']}: workload differs from first run: {actual} != {expected}")
    if expected[1] != "text_to_text" or expected[2] != "closed_loop" or expected[3] != 64:
        raise ValueError(f"unexpected workload: {expected}")
    for block in {row["block"] for row in rows}:
        if (block, "off") not in seen:
            raise ValueError(f"block {block} has no off run")
    return rows


def detect_mode(runs: list[dict], requested: str) -> tuple[str, bool]:
    by_block: dict[str, set[str]] = defaultdict(set)
    for row in runs:
        by_block[row["block"]].add(row["arm"])
    arm_sets = list(by_block.values())
    if all(arms == {"off", "full"} for arms in arm_sets):
        actual = "preliminary"
    elif all(arms == set(ARMS) for arms in arm_sets):
        actual = "full"
    else:
        raise ValueError("every block must be an off/full pair or a complete four-arm set")
    if requested != "auto" and requested != actual:
        raise ValueError(f"manifest is {actual}, but --mode {requested} was requested")
    if actual == "preliminary" and len(by_block) != 4:
        raise ValueError("the preliminary analysis requires exactly four off/full pairs")
    order = [row["arm"] for row in runs]
    abba = order == ["off", "full", "full", "off"] * 2
    return actual, abba


def comparisons(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    by_block: dict[str, dict[str, dict]] = defaultdict(dict)
    for row in runs:
        by_block[row["block"]][row["arm"]] = row
    paired, increments = [], []
    for block, arms in by_block.items():
        off = arms["off"]["throughput_tok_s"]
        for arm in ARMS[1:]:
            if arm in arms:
                measured = arms[arm]["throughput_tok_s"]
                paired.append({"block": block, "arm": arm, "off_tok_s": off,
                               "arm_tok_s": measured, "delta_pct": 100 * (measured / off - 1)})
        for before, after in (("off", "api"), ("api", "api_data"), ("api_data", "full")):
            if before in arms and after in arms:
                base = arms[before]["throughput_tok_s"]
                measured = arms[after]["throughput_tok_s"]
                increments.append({"block": block, "before": before, "after": after,
                                   "before_tok_s": base, "after_tok_s": measured,
                                   "delta_pct": 100 * (measured / base - 1)})
    return paired, increments


def chart(runs: list[dict], paired: list[dict], increments: list[dict], output: Path,
          mode: str, frontend: str, abba: bool) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    panels = 4 if mode == "preliminary" else 5
    fig, axes = plt.subplots(panels, 1, figsize=(11, 15 if panels == 4 else 18),
                             constrained_layout=True)
    mode_label = "Preliminary off/full pairs" if mode == "preliminary" else "Four-arm stage diagnostic"
    fig.suptitle(f"M* #735 | {frontend.title()} frontend | {mode_label}", fontsize=14)
    ax = axes[0]
    for arm in (("off", "full") if mode == "preliminary" else ARMS):
        group = [row for row in runs if row["arm"] == arm]
        ax.scatter([r["start"] for r in group], [r["throughput_tok_s"] for r in group],
                   s=65, color=COLORS[arm], label=f"{arm} (n={len(group)})", zorder=3)
    ax.set(title="Measured throughput by fresh server start", xlabel="Start order",
           ylabel="Text output tokens / second")
    ax.set_xticks([r["start"] for r in runs])
    ax.grid(alpha=.2)
    ax.legend(ncol=4, loc="best", fontsize=9)

    def deltas_axis(ax, groups: list[tuple[str, list[float], str]], title: str) -> None:
        ax.axhline(0, color="#64748b", linewidth=1)
        for x, (label, values, color) in enumerate(groups):
            if not values:
                continue
            mean = sum(values) / len(values)
            ax.scatter([x + (i - (len(values) - 1) / 2) * .08 for i in range(len(values))],
                       values, color=color, s=25, alpha=.7, zorder=3)
            ax.errorbar([x], [mean], yerr=[[mean - min(values)], [max(values) - mean]],
                        fmt="D", color=color, capsize=5, markersize=7, zorder=4)
            ax.annotate(f"n={len(values)}", (x, max(values)), xytext=(0, 8),
                        textcoords="offset points", ha="center", fontsize=8)
        ax.set_xticks(range(len(groups)), [g[0] for g in groups])
        ax.set_ylabel("Throughput change (%)")
        ax.set_title(title)
        ax.grid(axis="y", alpha=.2)

    compared_arms = ("full",) if mode == "preliminary" else ARMS[1:]
    paired_title = "Within-pair metrics effect"
    if mode == "preliminary":
        paired_title += " (ABBA order)" if abba else " (not ABBA order)"
    deltas_axis(axes[1], [(arm + " vs off", [p["delta_pct"] for p in paired if p["arm"] == arm], COLORS[arm])
                          for arm in compared_arms], paired_title)
    if mode == "full":
        deltas_axis(axes[2], [(f"{before} → {after}",
                               [p["delta_pct"] for p in increments if p["before"] == before and p["after"] == after],
                               COLORS[after]) for before, after in (("off", "api"), ("api", "api_data"),
                                                                    ("api_data", "full"))],
                    "Incremental metrics stages within each block")
    ttft_ax, itl_ax = axes[-2:]
    arms = ("off", "full") if mode == "preliminary" else ARMS
    for arm in arms:
        group = [row for row in runs if row["arm"] == arm]
        ttft_ax.scatter([r["start"] for r in group], [r["ttft_p50_ms"] for r in group],
                        marker="o", color=COLORS[arm], label=f"{arm} p50", s=42)
        ttft_ax.scatter([r["start"] for r in group], [r["ttft_p95_ms"] for r in group],
                        marker="^", color=COLORS[arm], label=f"{arm} p95", s=42)
        itl_ax.scatter([r["start"] for r in group], [r["itl_p50_ms"] for r in group],
                       color=COLORS[arm], label=arm, s=42)
    ttft_ax.set(title="Time to first text token by fresh server start",
                ylabel="TTFT (ms)")
    ttft_ax.legend(ncol=4, fontsize=8)
    itl_ax.set(title="Median inter-token latency by fresh server start",
               xlabel="Start order", ylabel="ITL p50 (ms)")
    itl_ax.legend(ncol=4, fontsize=8)
    for ax in (ttft_ax, itl_ax):
        ax.set_xticks([r["start"] for r in runs])
        ax.grid(alpha=.2)
    fig.supxlabel("Diamonds: mean across starts. Bars: observed min to max, not confidence intervals."
                  " Dots: individual paired blocks. Each server starts fresh.", fontsize=9)
    fig.savefig(output / "metrics_diagnostic.png", dpi=180)
    svg_path = output / "metrics_diagnostic.svg"
    fig.savefig(svg_path)
    svg_path.write_text("\n".join(line.rstrip() for line in svg_path.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--mode", choices=("auto", "preliminary", "full"), default="auto")
    parser.add_argument("--frontend", choices=("python", "rust"), required=True)
    parser.add_argument("--commit", help="M* source commit used for every arm")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    runs = read_runs(args.manifest)
    mode, abba = detect_mode(runs, args.mode)
    paired, increments = comparisons(runs)
    for row in runs:
        row["frontend"] = args.frontend
        row["source_commit"] = args.commit or "unrecorded"
    write_csv(args.out / "runs.csv", runs, list(runs[0]))
    write_csv(args.out / "paired.csv", paired,
              ["block", "arm", "off_tok_s", "arm_tok_s", "delta_pct"])
    write_csv(args.out / "increments.csv", increments,
              ["block", "before", "after", "before_tok_s", "after_tok_s", "delta_pct"])
    metadata = {
        "mode": mode, "frontend": args.frontend, "source_commit": args.commit,
        "abba_order": abba if mode == "preliminary" else None,
        "starts": len(runs), "blocks": len({r["block"] for r in runs}),
        "request_type": runs[0]["request_type"], "profiling_type": runs[0]["profiling_type"],
        "max_concurrency": runs[0]["max_concurrency"], "measured_requests_per_start": runs[0]["completed"],
        "output_tokens_per_request": 256,
        "limitations": "Prompt file, server flags, scrape cadence, and hardware are not encoded in results.json; verify run logs separately.",
    }
    (args.out / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    chart(runs, paired, increments, args.out, mode, args.frontend, abba)
    print(f"Plotted {len(runs)} {args.frontend} starts ({mode}) in {args.out}")


if __name__ == "__main__":
    main()
