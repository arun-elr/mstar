#!/usr/bin/env python3
"""Validate and chart Chatterbox peek-off cap 16/32 ABBA runs.

Cap manifest: order,cap,rep,burst,sweep_dir. The optional original #1567
manifest supplies two earlier cap-8 PR runs as an unpaired reference.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONCURRENCIES = (8, 16)
COUNTS = {8: 48, 16: 96}
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
COLORS = {8: "#64748b", 16: "#0f766e", 32: "#b45309"}


def positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(positive(actual, label), expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{label}: reported {actual}, recomputed {expected}")


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[round(q * (len(values) - 1))]


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else base / path


def read_cap_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 4 or not {"order", "cap", "rep", "burst", "sweep_dir"} <= rows[0].keys():
        raise ValueError("cap manifest needs four rows and order,cap,rep,burst,sweep_dir")
    if [(int(r["order"]), int(r["cap"]), int(r["rep"])) for r in rows] != [
        (1, 16, 1), (2, 32, 1), (3, 32, 2), (4, 16, 2)
    ]:
        raise ValueError("unexpected cap rotation; expected 16/32/32/16")
    return [{"order": int(r["order"]), "cap": int(r["cap"]), "rep": int(r["rep"]),
             "burst": resolve(path.parent, r["burst"]),
             "sweep_dir": resolve(path.parent, r["sweep_dir"]), "origin": "rotation"}
            for r in rows]


def read_baseline_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    selected = [r for r in rows if r.get("arm") == "pr"]
    if len(selected) != 2 or sorted(int(r["pair"]) for r in selected) != [1, 2]:
        raise ValueError("baseline manifest needs two cap-8 PR starts in pairs 1 and 2")
    return [{"order": int(r["order"]), "cap": 8, "rep": int(r["pair"]),
             "burst": resolve(path.parent, r["burst"]),
             "sweep_dir": resolve(path.parent, r["sweep_dir"]), "origin": "earlier #1567"}
            for r in selected]


def read_burst(entry: dict) -> tuple[dict, dict, tuple]:
    path = entry["burst"]
    report = json.loads(path.read_text())
    s = report["summary"]
    rows = report["rows"]
    if report.get("arm") != entry.get("arm", "pr") or report.get("start") != entry["order"]:
        raise ValueError(f"{path}: wrong arm or start")
    if report.get("protocol") != "pcm_s16le_24000" or report.get("delay_ms") != 50:
        raise ValueError(f"{path}: wrong PCM protocol or burst delay")
    if s.get("ok") != 9 or s.get("total") != 9 or not s.get("long_active_at_burst"):
        raise ValueError(f"{path}: failed or nonoverlapping burst")
    if len(rows) != 9 or [r["label"] for r in rows] != ["long", *(f"short-{i}" for i in range(8))]:
        raise ValueError(f"{path}: wrong burst request set")
    if [r["seed"] for r in rows] != list(range(48600, 48609)):
        raise ValueError(f"{path}: wrong burst seeds")
    if [r["cfg_weight"] for r in rows] != [0.0, *(0.0 if i % 2 == 0 else None for i in range(8))]:
        raise ValueError(f"{path}: wrong burst CFG mix")
    if any(r["status"] != 200 or r["error"] is not None or r["audio_bytes"] <= 0 for r in rows):
        raise ValueError(f"{path}: failed or empty PCM stream")
    long_finish = rows[0]["finish_at_s"]
    if any(r["submit_at_s"] >= long_finish for r in rows[1:]):
        raise ValueError(f"{path}: short request did not overlap long stream")
    close(s["long_finish_after_short_submit_s"], long_finish - rows[1]["submit_at_s"],
          f"{path}: long/short overlap")
    for r in rows:
        close(r["audio_s"], r["audio_bytes"] / 48000, f"{path}: PCM duration")
        close(r["ttfa_s"], r["first_byte_at_s"] - r["submit_at_s"], f"{path}: TTFA")
        close(r["e2e_s"], r["finish_at_s"] - r["submit_at_s"], f"{path}: E2E")
    shorts = rows[1:]
    for q, level in (("p50", .5), ("p95", .95)):
        close(s[f"short_ttfa_{q}_s"], percentile([r["ttfa_s"] for r in shorts], level),
              f"{path}: short TTFA {q}")
        close(s[f"short_e2e_{q}_s"], percentile([r["e2e_s"] for r in shorts], level),
              f"{path}: short E2E {q}")
    close(s["long_e2e_s"], rows[0]["e2e_s"], f"{path}: long E2E")
    close(s["wall_s"], max(r["finish_at_s"] for r in rows), f"{path}: wall")
    total = sum(r["audio_s"] for r in rows)
    close(s["audio_s_per_s"], total / s["wall_s"], f"{path}: throughput")
    metrics = {"burst_short_ttfa_p50_s": s["short_ttfa_p50_s"],
               "burst_short_ttfa_p95_s": s["short_ttfa_p95_s"],
               "burst_audio_s_total": total, "burst_audio_s_per_s": s["audio_s_per_s"],
               "burst_short_e2e_p50_s": s["short_e2e_p50_s"],
               "burst_short_e2e_p95_s": s["short_e2e_p95_s"],
               "burst_long_e2e_s": s["long_e2e_s"]}
    audio = {r["label"]: r["audio_s"] for r in rows}
    signature = tuple((r["label"], r["seed"], r["prompt"], r["cfg_weight"]) for r in rows)
    return metrics, audio, signature


def read_sweep(entry: dict, c: int) -> tuple[dict, dict, tuple]:
    path = entry["sweep_dir"] / f"mixed_c{c}.json"
    report = json.loads(path.read_text())
    s = report["summary"]
    rows = report["rows"]
    n = COUNTS[c]
    if (s.get("mode"), s.get("c"), s.get("n"), s.get("ok")) != ("mixed", c, n, n):
        raise ValueError(f"{path}: wrong sweep setup or failures")
    if len(rows) != n or sorted(r["i"] for r in rows) != list(range(n)):
        raise ValueError(f"{path}: incomplete or duplicate requests")
    if any(r["status"] != 200 or r["audio_s"] <= 0 or r["ttfa_s"] is None for r in rows):
        raise ValueError(f"{path}: failed or empty stream")
    if any(r["cfg_weight"] != (0.0 if r["i"] % 2 == 0 else None) for r in rows):
        raise ValueError(f"{path}: wrong CFG mix")
    wall = max(r["end"] for r in rows) - min(r["start"] for r in rows)
    total = sum(positive(r["audio_s"], "request audio") for r in rows)
    close(s["wall_s"], wall, f"{path}: wall")
    close(s["audio_total_s"], total, f"{path}: audio total")
    close(s["audio_s_per_s"], total / wall, f"{path}: throughput")
    metrics = {f"c{c}_audio_s_total": total, f"c{c}_audio_s_per_s": s["audio_s_per_s"],
               f"c{c}_wall_s": wall}
    for q, level in (("p50", .5), ("p95", .95)):
        value = percentile([positive(r["ttfa_s"], "request TTFA") for r in rows], level)
        close(s["ttfa_s"][q], value, f"{path}: TTFA {q}")
        metrics[f"c{c}_ttfa_{q}_s"] = value
    for lane in ("cfg0", "default"):
        selected = [r for r in rows if (r["cfg_weight"] == 0.0) == (lane == "cfg0")]
        if len(selected) != n // 2 or s[lane]["n"] != n // 2:
            raise ValueError(f"{path}: wrong {lane} count")
        for q, level in (("p50", .5), ("p95", .95)):
            close(s[lane]["ttfa_s"][q], percentile([r["ttfa_s"] for r in selected], level),
                  f"{path}: {lane} TTFA {q}")
    audio = {r["i"]: r["audio_s"] for r in rows}
    signature = tuple(sorted((r["i"], r["cfg_weight"]) for r in rows))
    return metrics, audio, signature


def read_run(entry: dict) -> dict:
    burst, burst_audio, burst_signature = read_burst(entry)
    row = {k: entry[k] for k in ("order", "cap", "rep", "origin")}
    row.update(burst)
    audio = {"burst": burst_audio}
    signatures = {"burst": burst_signature}
    for c in CONCURRENCIES:
        metrics, current_audio, signature = read_sweep(entry, c)
        row.update(metrics)
        audio[f"c{c}"] = current_audio
        signatures[f"c{c}"] = signature
    row["_audio"] = audio
    row["_signatures"] = signatures
    return row


def compare_audio(runs: list[dict]) -> tuple[list[dict], list[dict]]:
    indexed = {(r["cap"], r["rep"]): r for r in runs if r["cap"] in (16, 32)}
    requests = []
    summary = []
    comparisons = [
        ("between caps, rep 1", indexed[16, 1], indexed[32, 1]),
        ("between caps, rep 2", indexed[16, 2], indexed[32, 2]),
        ("within cap 16", indexed[16, 1], indexed[16, 2]),
        ("within cap 32", indexed[32, 1], indexed[32, 2]),
    ]
    for label, a, b in comparisons:
        for cell in ("burst", "c8", "c16"):
            left, right = a["_audio"][cell], b["_audio"][cell]
            if set(left) != set(right):
                raise ValueError(f"{label} {cell}: request identities differ")
            rows = []
            for request_id in sorted(left):
                first = positive(left[request_id], "first audio duration")
                second = positive(right[request_id], "second audio duration")
                row = {"comparison": label, "cell": cell, "request_id": request_id,
                       "first_cap": a["cap"], "first_order": a["order"],
                       "second_cap": b["cap"], "second_order": b["order"],
                       "first_audio_s": first, "second_audio_s": second,
                       "delta_audio_s": second - first, "equal": first == second}
                rows.append(row)
                requests.append(row)
            total_a = sum(r["first_audio_s"] for r in rows)
            total_b = sum(r["second_audio_s"] for r in rows)
            summary.append({"comparison": label, "cell": cell, "requests": len(rows),
                            "duration_mismatches": sum(not r["equal"] for r in rows),
                            "first_audio_s_total": total_a, "second_audio_s_total": total_b,
                            "total_change_pct": 100 * (total_b / total_a - 1),
                            "max_abs_request_delta_s": max(abs(r["delta_audio_s"]) for r in rows)})
    return requests, summary


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired_metrics(runs: list[dict]) -> list[dict]:
    indexed = {(r["cap"], r["rep"]): r for r in runs if r["cap"] in (16, 32)}
    return [
        {"rep": rep, "metric": metric, "cap16": indexed[16, rep][metric],
         "cap32": indexed[32, rep][metric],
         "cap32_vs_16_pct": 100 * (indexed[32, rep][metric] / indexed[16, rep][metric] - 1)}
        for rep in (1, 2) for metric, _, _ in METRICS
    ]


def plot_metrics(runs: list[dict], out: Path, title: str) -> None:
    selected = METRICS[:8]
    fig, axes = plt.subplots(2, 4, figsize=(18, 8), constrained_layout=True)
    for ax, (metric, label, unit) in zip(axes.flat, selected):
        for cap in (8, 16, 32):
            values = [r[metric] for r in runs if r["cap"] == cap]
            if not values:
                continue
            ax.scatter([cap - .5, cap + .5][:len(values)], values,
                       color=COLORS[cap], alpha=.8, label=f"cap {cap}" if cap == 8 else None)
            ax.plot([cap - .7, cap + .7], [sum(values) / len(values)] * 2,
                    color=COLORS[cap], linewidth=2)
        ax.set(title=label, xlabel="Max consecutive spec steps", ylabel=unit, xticks=[8, 16, 32])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    fig.supxlabel("Cap 8 is an earlier, unpaired #1567 run. Dots are fresh starts; horizontal marks are means.",
                  fontsize=9, color="#475569")
    for ext in ("png", "svg"):
        fig.savefig(out / f"chatterbox_cap_ablation.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def plot_durations(runs: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.5), constrained_layout=True)
    for ax, cell in zip(axes, ("burst", "c8", "c16")):
        metric = f"{cell}_audio_s_total"
        for cap in (8, 16, 32):
            values = [r[metric] for r in runs if r["cap"] == cap]
            if not values:
                continue
            ax.scatter([cap - .5, cap + .5][:len(values)], values, color=COLORS[cap], alpha=.8)
            ax.plot([cap - .7, cap + .7], [sum(values) / len(values)] * 2,
                    color=COLORS[cap], linewidth=2)
        ax.set(title=f"{cell} generated duration", xlabel="Max consecutive spec steps",
               ylabel="audio seconds", xticks=[8, 16, 32])
        ax.grid(axis="y", alpha=.2)
    fig.suptitle(title + "\nOutput-duration differences can affect throughput and queueing latency.")
    for ext in ("png", "svg"):
        fig.savefig(out / f"chatterbox_cap_audio_duration.{ext}", dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--baseline-manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--pr-head", required=True)
    args = parser.parse_args()
    runs = [read_run(entry) for entry in read_cap_manifest(args.manifest)]
    if args.baseline_manifest:
        runs += [read_run(entry) for entry in read_baseline_manifest(args.baseline_manifest)]
    expected_signatures = runs[0]["_signatures"]
    for run in runs[1:]:
        if run["_signatures"] != expected_signatures:
            raise ValueError("request identities or CFG mix differ across starts")
    requests, duration_rows = compare_audio(runs)
    published = [{k: v for k, v in run.items() if not k.startswith("_")} for run in runs]
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "per_start.csv", published)
    write_csv(args.out / "paired.csv", paired_metrics(runs))
    write_csv(args.out / "request_audio_comparison.csv", requests)
    write_csv(args.out / "audio_comparability.csv", duration_rows)
    title = f"#1567 cap ablation on one H200 | PR {args.pr_head[:10]}"
    plot_metrics(runs, args.out, title)
    plot_durations(runs, args.out, title)
    for svg in args.out.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    lines = [
        "**Cap 8 is historical context, not part of the cap 16/32 rotation.**",
        "",
        "| Metric | cap 8 earlier mean | cap 16 mean | cap 32 mean | cap 32 vs 16 paired mean |",
        "|---|---:|---:|---:|---:|",
    ]
    by_cap = {(r["cap"], r["rep"]): r for r in runs}
    for metric, label, unit in METRICS:
        means = {cap: sum(r[metric] for r in runs if r["cap"] == cap) /
                 sum(r["cap"] == cap for r in runs) for cap in (16, 32)}
        baseline = [r[metric] for r in runs if r["cap"] == 8]
        paired_changes = [100 * (by_cap[32, rep][metric] / by_cap[16, rep][metric] - 1)
                          for rep in (1, 2)]
        baseline_text = f"{sum(baseline)/len(baseline):.3f}" if baseline else "n/a"
        lines.append(f"| {label} ({unit}) | {baseline_text} | {means[16]:.3f} | {means[32]:.3f} | "
                     f"{sum(paired_changes)/2:+.2f}% |")
    lines += ["", "| Comparison | Cell | Different request durations | Total audio change | Largest request difference |",
              "|---|---|---:|---:|---:|"]
    for row in duration_rows:
        lines.append(f"| {row['comparison']} | {row['cell']} | "
                     f"{row['duration_mismatches']}/{row['requests']} | "
                     f"{row['total_change_pct']:+.2f}% | {row['max_abs_request_delta_s']:.3f} s |")
    mismatches = sum(not row["equal"] for row in requests)
    lines.append(f"\nDuration check: {mismatches}/{len(requests)} matched request comparisons differ. "
                 "Throughput and latency changes are descriptive when output durations differ. "
                 "Two starts per cap do not establish a confidence interval.")
    table = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
