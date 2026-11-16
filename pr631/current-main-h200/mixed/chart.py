#!/usr/bin/env python3
"""Validate and plot the matched H200 CosyVoice3 #631 speech runs.

Input is the manifest written by mstar-pr631-h200-run.sh. The script checks
the individual request records, then writes per-start and paired CSVs plus a
PNG/SVG figure. No result is accepted from an incomplete or failed run.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("ttfa_p50_ms", "ttfa_p95_ms", "audio_s_per_wall_s")
FIELDS = (
    "order", "pair", "arm", "concurrency", "path", "num_sentences", "warmup",
    "requests", "errors", "ttfa_p50_ms", "ttfa_p95_ms", "audio_s_total",
    "wall_s", "audio_s_per_wall_s", "sample_rate", "torch_gpu",
)


def _write_csv(path: Path, rows: list[dict], fields: tuple[str, ...]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _percentile(values: list[float], p: float) -> float:
    ordered = sorted(values)
    return ordered[round(p * (len(ordered) - 1))]


def _positive(value: object, name: str, path: Path) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: {name} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{path}: {name} must be positive and finite")
    return number


def _read_report(path: Path, arm: str, expected_c: int, expected_n: int, expected_warmup: int) -> dict:
    report = json.loads(path.read_text())
    if report.get("engine") != "mstar" or report.get("model") != "cosyvoice3":
        raise ValueError(f"{path}: wrong engine or model")
    if report.get("label") != f"{arm}-c{expected_c}":
        raise ValueError(f"{path}: label does not match the manifest arm")
    if report.get("concurrency") != expected_c or report.get("repeats") != 1:
        raise ValueError(f"{path}: wrong concurrency or repeat count")
    if report.get("num_sentences") != expected_n or report.get("warmup") != expected_warmup:
        raise ValueError(f"{path}: wrong sentence or warmup count")
    if report.get("errors_total") != 0 or len(report.get("repeats_detail", [])) != 1:
        raise ValueError(f"{path}: failed or missing repeat")
    if "H200" not in json.dumps(report.get("gpu", {})):
        raise ValueError(f"{path}: report does not identify an H200")
    detail = report["repeats_detail"][0]
    requests = detail.get("results", [])
    if detail.get("requests") != expected_n or detail.get("errors") != 0 or len(requests) != expected_n:
        raise ValueError(f"{path}: request count or error mismatch")
    if sorted(item.get("sentence_id") for item in requests) != list(range(1, expected_n + 1)):
        raise ValueError(f"{path}: incomplete sentence ids")
    if any(item.get("error") is not None for item in requests):
        raise ValueError(f"{path}: request error")
    times = [_positive(item.get("ttfa_s"), "request TTFA", path) for item in requests]
    audio = [_positive(item.get("audio_s"), "request audio duration", path) for item in requests]
    wall = _positive(detail.get("wall_s"), "wall time", path)
    summary = report["median_over_repeats"]
    for name in METRICS:
        _positive(summary.get(name), name, path)
    checks = {
        "ttfa_p50_ms": 1000 * _percentile(times, .50),
        "ttfa_p95_ms": 1000 * _percentile(times, .95),
        "audio_s_total": sum(audio),
        "audio_s_per_wall_s": sum(audio) / wall,
    }
    for name, want in checks.items():
        actual = float(detail[name])
        if not math.isclose(actual, want, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(f"{path}: {name} does not match raw requests")
        if name in summary and not math.isclose(float(summary[name]), want, rel_tol=1e-6, abs_tol=1e-4):
            raise ValueError(f"{path}: median {name} does not match only repeat")
    return {
        "num_sentences": expected_n,
        "warmup": expected_warmup,
        "requests": expected_n,
        "errors": 0,
        **{name: float(summary[name]) for name in METRICS},
        "audio_s_total": float(detail["audio_s_total"]),
        "wall_s": wall,
        "sample_rate": report.get("sample_rate"),
        "torch_gpu": json.dumps(report.get("gpu", {}), sort_keys=True),
        "_matching": (
            report.get("sentences_file"), report.get("sample_rate"),
            json.dumps(report.get("request_fields"), sort_keys=True),
        ),
    }


def read_runs(manifest: Path, expected_c1: int, expected_c8: int) -> list[dict]:
    with manifest.open(newline="") as file:
        entries = list(csv.DictReader(file))
    if len(entries) != 6:
        raise ValueError(f"expected six starts, found {len(entries)}")
    if [int(row["order"]) for row in entries] != list(range(1, 7)):
        raise ValueError("starts are not in recorded order 1..6")
    if [row["arm"] for row in entries] != ["main", "pr", "pr", "main", "main", "pr"]:
        raise ValueError("unexpected arm order; check the run script")
    expected_pairs = [1, 1, 2, 2, 3, 3]
    if [int(row["pair"]) for row in entries] != expected_pairs:
        raise ValueError("unexpected pairing")
    output = []
    matches: dict[int, tuple] = {}
    for entry in entries:
        for c, n, warmup in ((1, expected_c1, 8), (8, expected_c8, 16)):
            path = (manifest.parent / entry[f"c{c}"]).resolve()
            report = _read_report(path, entry["arm"], c, n, warmup)
            if c in matches and report["_matching"] != matches[c]:
                raise ValueError(f"{path}: inputs differ from another c{c} run")
            matches[c] = report.pop("_matching")
            output.append({
                "order": int(entry["order"]), "pair": int(entry["pair"]),
                "arm": entry["arm"], "concurrency": c,
                "path": str(path.relative_to(manifest.parent.resolve())),
                **report,
            })
    if len({(r["pair"], r["arm"], r["concurrency"]) for r in output}) != 12:
        raise ValueError("duplicate paired arm")
    return output


def pair_runs(runs: list[dict]) -> list[dict]:
    indexed = {(r["pair"], r["arm"], r["concurrency"]): r for r in runs}
    paired = []
    for c in (1, 8):
        for pair in (1, 2, 3):
            main, pr = (indexed[(pair, arm, c)] for arm in ("main", "pr"))
            row = {"concurrency": c, "pair": pair, "main_order": main["order"], "pr_order": pr["order"]}
            for name in (*METRICS, "audio_s_total", "wall_s"):
                row[f"main_{name}"] = main[name]
                row[f"pr_{name}"] = pr[name]
                row[f"delta_{name}_pct"] = 100 * (pr[name] / main[name] - 1)
            paired.append(row)
    return paired


def summary_table(paired: list[dict]) -> str:
    lines = [
        "| Concurrency | Metric | Main mean | PR mean | Mean paired change | Range of paired changes |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for c in (1, 8):
        rows = [r for r in paired if r["concurrency"] == c]
        for metric, label, unit in (
            ("ttfa_p50_ms", "TTFA p50", "ms"),
            ("ttfa_p95_ms", "TTFA p95", "ms"),
            ("audio_s_per_wall_s", "Audio throughput", "audio-s/s"),
            ("audio_s_total", "Generated audio duration", "audio s"),
            ("wall_s", "Measured wall time", "s"),
        ):
            main = sum(r[f"main_{metric}"] for r in rows) / len(rows)
            pr = sum(r[f"pr_{metric}"] for r in rows) / len(rows)
            changes = [r[f"delta_{metric}_pct"] for r in rows]
            lines.append(
                f"| {c} | {label} | {main:.2f} {unit} | {pr:.2f} {unit} | "
                f"{sum(changes)/len(changes):+.2f}% | {min(changes):+.2f}% to {max(changes):+.2f}% |"
            )
    lines.extend((
        "",
        "Three matched fresh-start pairs per concurrency. Percentages are the mean of pairwise "
        "PR/main changes, not a confidence interval. Throughput is generated audio duration "
        "divided by wall time, so a duration change can affect it. See paired.csv for each pair.",
    ))
    return "\n".join(lines) + "\n"


def plot(runs: list[dict], paired: list[dict], output: Path, title: str) -> None:
    colors = {"main": "#334155", "pr": "#0f766e"}
    fig, axes = plt.subplots(2, 4, figsize=(20, 8), constrained_layout=True)
    for i, c in enumerate((1, 8)):
        by_key = {(r["pair"], r["arm"]): r for r in runs if r["concurrency"] == c}
        subset = [r for r in paired if r["concurrency"] == c]
        ax = axes[i, 0]
        for name, marker, alpha in (("ttfa_p50_ms", "o", 1), ("ttfa_p95_ms", "s", .55)):
            for pair in (1, 2, 3):
                main, pr = by_key[(pair, "main")], by_key[(pair, "pr")]
                ax.plot([pair - .13, pair + .13], [main[name], pr[name]], color="#94a3b8", alpha=alpha)
                for x, arm, row in ((pair - .13, "main", main), (pair + .13, "pr", pr)):
                    ax.scatter(x, row[name], color=colors[arm], marker=marker, alpha=alpha,
                               label=f"{arm} {name.replace('ttfa_', '').replace('_ms', '')}" if pair == 1 else None)
        ax.set(title=f"c{c}: first audio", ylabel="TTFA (ms)", xticks=[1, 2, 3], xlabel="matched pair")
        ax.legend(fontsize=8, ncol=2)
        ax.grid(axis="y", alpha=.2)

        ax = axes[i, 1]
        for pair in (1, 2, 3):
            main, pr = by_key[(pair, "main")], by_key[(pair, "pr")]
            ax.plot([pair - .13, pair + .13], [main["audio_s_per_wall_s"], pr["audio_s_per_wall_s"]],
                    color="#94a3b8")
            for x, arm, row in ((pair - .13, "main", main), (pair + .13, "pr", pr)):
                ax.scatter(x, row["audio_s_per_wall_s"], color=colors[arm],
                           label=arm if pair == 1 else None)
        ax.set(title=f"c{c}: audio throughput", ylabel="audio seconds / wall second",
               xticks=[1, 2, 3], xlabel="matched pair")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)

        ax = axes[i, 3]
        for pair in (1, 2, 3):
            main, pr = by_key[(pair, "main")], by_key[(pair, "pr")]
            ax.plot([pair - .13, pair + .13], [main["audio_s_total"], pr["audio_s_total"]],
                    color="#94a3b8")
            for x, arm, row in ((pair - .13, "main", main), (pair + .13, "pr", pr)):
                ax.scatter(x, row["audio_s_total"], color=colors[arm],
                           label=arm if pair == 1 else None)
        ax.set(title=f"c{c}: generated audio duration", ylabel="total audio seconds",
               xticks=[1, 2, 3], xlabel="matched pair")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)

        ax = axes[i, 2]
        for j, (name, color, label) in enumerate((
            ("delta_ttfa_p50_ms_pct", "#b45309", "TTFA p50"),
            ("delta_ttfa_p95_ms_pct", "#d97706", "TTFA p95"),
            ("delta_audio_s_per_wall_s_pct", "#0f766e", "audio throughput"),
        )):
            ax.bar([row["pair"] + (j - 1) * .23 for row in subset],
                   [row[name] for row in subset], width=.21, color=color,
                   label=label)
        ax.axhline(0, color="#334155", linewidth=.8)
        ax.set(title=f"c{c}: PR relative to main", ylabel="paired change (%)",
               xticks=[1, 2, 3], xlabel="matched pair")
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(title, fontsize=14)
    fig.supxlabel("Throughput = generated audio seconds / wall seconds. Duration changes may affect throughput; see paired.csv.",
                  fontsize=9, color="#475569")
    fig.savefig(output / "pr631_h200_current_main.png", dpi=180, bbox_inches="tight")
    fig.savefig(output / "pr631_h200_current_main.svg", bbox_inches="tight")
    svg = output / "pr631_h200_current_main.svg"
    svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-c1", type=int, default=27)
    parser.add_argument("--expected-c8", type=int, default=79)
    parser.add_argument("--title", default="CosyVoice3 on one H200: current main vs #631")
    args = parser.parse_args()
    runs = read_runs(args.manifest, args.expected_c1, args.expected_c8)
    paired = pair_runs(runs)
    args.out.mkdir(parents=True, exist_ok=True)
    _write_csv(args.out / "runs.csv", runs, FIELDS)
    _write_csv(args.out / "paired.csv", paired, tuple(paired[0]))
    table = summary_table(paired)
    (args.out / "summary.md").write_text(table)
    plot(runs, paired, args.out, args.title)
    print(table, end="")


if __name__ == "__main__":
    main()
