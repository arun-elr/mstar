#!/usr/bin/env python3
"""Local-only PR versus experimental readiness prototype Chatterbox charts.

Requires mstar-pr1567-cap-ablation-charts.py beside this file for raw-request
validation. Manifest: pair,order,arm,burst,sweep_dir, in PR/proto/proto/PR order.
The prototype is unpublished and these outputs are not a production claim.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

helper_path = Path(__file__).with_name("mstar-pr1567-cap-ablation-charts.py")
spec = importlib.util.spec_from_file_location("cap_ablation_chart", helper_path)
if spec is None or spec.loader is None:
    raise RuntimeError(f"cannot load raw-request validator: {helper_path}")
base = importlib.util.module_from_spec(spec)
spec.loader.exec_module(base)

ARMS = ("pr", "proto")
COLORS = {"pr": "#334155", "proto": "#0f766e"}
METRICS = (
    ("burst_short_ttfa_p50_s", "Burst short TTFA p50", "s"),
    ("burst_short_ttfa_p95_s", "Burst short TTFA p95", "s"),
    ("c8_ttfa_p50_s", "c8 TTFA p50", "s"),
    ("c8_ttfa_p95_s", "c8 TTFA p95", "s"),
    ("c16_ttfa_p50_s", "c16 TTFA p50", "s"),
    ("c16_ttfa_p95_s", "c16 TTFA p95", "s"),
    ("c8_audio_s_per_s", "c8 audio throughput", "audio-s/s"),
    ("c16_audio_s_per_s", "c16 audio throughput", "audio-s/s"),
    ("burst_audio_s_total", "Burst generated audio", "audio s"),
    ("c8_audio_s_total", "c8 generated audio", "audio s"),
    ("c16_audio_s_total", "c16 generated audio", "audio s"),
)


def normalize_svg(path: Path) -> None:
    content = path.read_text()
    path.write_text("\n".join(line.rstrip(" \t") for line in content.splitlines()) + "\n")


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 4 or not {"pair", "order", "arm", "burst", "sweep_dir"} <= rows[0].keys():
        raise ValueError("manifest needs four rows with pair,order,arm,burst,sweep_dir")
    if [(int(r["pair"]), int(r["order"]), r["arm"]) for r in rows] != [
        (1, 1, "pr"), (1, 2, "proto"), (2, 3, "proto"), (2, 4, "pr")
    ]:
        raise ValueError("expected PR/proto/proto/PR fresh-start rotation")
    return [{"pair": int(r["pair"]), "order": int(r["order"]), "arm": r["arm"],
             "burst": base.resolve(path.parent, r["burst"]),
             "sweep_dir": base.resolve(path.parent, r["sweep_dir"])} for r in rows]


def read_run(entry: dict) -> dict:
    burst, burst_audio, burst_signature = base.read_burst(entry)
    row = {k: entry[k] for k in ("pair", "order", "arm")}
    row.update(burst)
    audio = {"burst": burst_audio}
    signatures = {"burst": burst_signature}
    for c in (8, 16):
        metrics, current_audio, signature = base.read_sweep(entry, c)
        row.update(metrics)
        audio[f"c{c}"] = current_audio
        signatures[f"c{c}"] = signature
    row["_audio"] = audio
    row["_signatures"] = signatures
    return row


def pair_metrics(runs: list[dict]) -> list[dict]:
    indexed = {(r["pair"], r["arm"]): r for r in runs}
    return [
        {"pair": pair, "metric": metric,
         "pr": indexed[pair, "pr"][metric], "proto": indexed[pair, "proto"][metric],
         "proto_vs_pr_pct": 100 * (indexed[pair, "proto"][metric] /
                                  indexed[pair, "pr"][metric] - 1)}
        for pair in (1, 2) for metric, _, _ in METRICS
    ]


def compare_audio(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    indexed = {(r["pair"], r["arm"]): r for r in runs}
    comparisons = [
        ("pair 1, PR vs prototype", indexed[1, "pr"], indexed[1, "proto"]),
        ("pair 2, PR vs prototype", indexed[2, "pr"], indexed[2, "proto"]),
        ("PR across starts", indexed[1, "pr"], indexed[2, "pr"]),
        ("prototype across starts", indexed[1, "proto"], indexed[2, "proto"]),
    ]
    request_rows = []
    summaries = []
    for label, a, b in comparisons:
        for cell in ("burst", "c8", "c16"):
            left, right = a["_audio"][cell], b["_audio"][cell]
            if set(left) != set(right):
                raise ValueError(f"{label} {cell}: request identities differ")
            group = []
            for request_id in sorted(left):
                first = base.positive(left[request_id], "first audio duration")
                second = base.positive(right[request_id], "second audio duration")
                row = {"comparison": label, "cell": cell, "request_id": request_id,
                       "first_arm": a["arm"], "first_order": a["order"],
                       "second_arm": b["arm"], "second_order": b["order"],
                       "first_audio_s": first, "second_audio_s": second,
                       "delta_audio_s": second - first, "equal": first == second}
                group.append(row)
                request_rows.append(row)
            first_total = sum(r["first_audio_s"] for r in group)
            second_total = sum(r["second_audio_s"] for r in group)
            summaries.append({
                "comparison": label, "cell": cell, "requests": len(group),
                "duration_mismatches": sum(not r["equal"] for r in group),
                "first_audio_s_total": first_total,
                "second_audio_s_total": second_total,
                "total_change_pct": 100 * (second_total / first_total - 1),
                "max_abs_request_delta_s": max(abs(r["delta_audio_s"]) for r in group),
            })
    return request_rows, summaries


def plot_metrics(runs: list[dict], out: Path, title: str, mismatch: int) -> None:
    indexed = {(r["pair"], r["arm"]): r for r in runs}
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, (metric, label, unit) in zip(axes.flat, METRICS[:8]):
        for pair in (1, 2):
            a, b = indexed[pair, "pr"], indexed[pair, "proto"]
            ax.plot([pair - .12, pair + .12], [a[metric], b[metric]], color="#94a3b8")
            ax.scatter(pair - .12, a[metric], color=COLORS["pr"],
                       label="PR" if pair == 1 else None)
            ax.scatter(pair + .12, b[metric], color=COLORS["proto"],
                       label="prototype" if pair == 1 else None)
        ax.set(title=label, xlabel="matched pair", ylabel=unit, xticks=[1, 2])
        ax.grid(axis="y", alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title)
    fig.supxlabel(f"Experimental prototype. {mismatch}/306 paired output durations differ; "
                  "latency and throughput are descriptive.", fontsize=9,
                  color="#9a3412" if mismatch else "#475569")
    for ext in ("png", "svg"):
        fig.savefig(out / f"chatterbox_readiness_ab.{ext}", dpi=170, bbox_inches="tight")
        if ext == "svg":
            normalize_svg(out / f"chatterbox_readiness_ab.{ext}")
    plt.close(fig)


def plot_durations(runs: list[dict], out: Path, title: str) -> None:
    indexed = {(r["pair"], r["arm"]): r for r in runs}
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, cell in zip(axes, ("burst", "c8", "c16")):
        metric = f"{cell}_audio_s_total"
        for pair in (1, 2):
            a, b = indexed[pair, "pr"], indexed[pair, "proto"]
            ax.plot([pair - .12, pair + .12], [a[metric], b[metric]], color="#94a3b8")
            ax.scatter(pair - .12, a[metric], color=COLORS["pr"],
                       label="PR" if pair == 1 else None)
            ax.scatter(pair + .12, b[metric], color=COLORS["proto"],
                       label="prototype" if pair == 1 else None)
        ax.set(title=f"{cell} generated audio", xlabel="matched pair",
               ylabel="audio seconds", xticks=[1, 2])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle(title + "\nOutput-duration differences can affect throughput and queueing latency.")
    for ext in ("png", "svg"):
        fig.savefig(out / f"chatterbox_readiness_audio_duration.{ext}", dpi=170, bbox_inches="tight")
        if ext == "svg":
            normalize_svg(out / f"chatterbox_readiness_audio_duration.{ext}")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pr-head", default="1959fb2eac")
    parser.add_argument("--proto-head", default="unpublished")
    args = parser.parse_args()
    runs = [read_run(entry) for entry in read_manifest(args.manifest)]
    signature = runs[0]["_signatures"]
    if any(run["_signatures"] != signature for run in runs[1:]):
        raise ValueError("request identities or CFG mix differ across fresh starts")
    comparisons = pair_metrics(runs)
    request_audio, audio_summary = compare_audio(runs)
    published = [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs]
    args.out.mkdir(parents=True, exist_ok=True)
    base.write_csv(args.out / "per_start.csv", published)
    base.write_csv(args.out / "paired.csv", comparisons)
    base.write_csv(args.out / "request_audio_comparison.csv", request_audio)
    base.write_csv(args.out / "audio_comparability.csv", audio_summary)
    pair_mismatches = sum(not r["equal"] for r in request_audio if r["comparison"].startswith("pair"))
    title = (f"Experimental #1567 readiness gate | PR {args.pr_head[:10]} vs "
             f"prototype {args.proto_head[:10]}")
    plot_metrics(runs, args.out, title, pair_mismatches)
    plot_durations(runs, args.out, title)
    lines = ["**Experimental prototype. No production or review-readiness claim.**", "",
             "| Metric | PR mean | Prototype mean | Mean paired change | Pairwise range |",
             "|---|---:|---:|---:|---:|"]
    for metric, label, unit in METRICS:
        cells = [r for r in comparisons if r["metric"] == metric]
        mean_pr = sum(r["pr"] for r in cells) / 2
        mean_proto = sum(r["proto"] for r in cells) / 2
        changes = [r["proto_vs_pr_pct"] for r in cells]
        lines.append(f"| {label} ({unit}) | {mean_pr:.3f} | {mean_proto:.3f} | "
                     f"{sum(changes)/2:+.2f}% | {min(changes):+.2f}% to {max(changes):+.2f}% |")
    lines += ["", "| Comparison | Cell | Different request durations | Total audio change | Largest request difference |",
              "|---|---|---:|---:|---:|"]
    for row in audio_summary:
        lines.append(f"| {row['comparison']} | {row['cell']} | "
                     f"{row['duration_mismatches']}/{row['requests']} | "
                     f"{row['total_change_pct']:+.2f}% | {row['max_abs_request_delta_s']:.3f} s |")
    lines.append(f"\n{pair_mismatches}/306 paired PR/prototype request durations differ. "
                 "Latency and throughput changes are descriptive when output lengths differ. "
                 "Two pairs show observed variation, not a confidence interval.")
    table = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
