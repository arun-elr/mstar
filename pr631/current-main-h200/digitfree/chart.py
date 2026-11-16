#!/usr/bin/env python3
"""Validate and chart four fresh-start ABBA CosyVoice3 c8 digit-free runs.

Manifest columns: pair,order,arm,c8. Paths are relative to the manifest.
The parser reports per-request PCM length differences within matched main/PR
pairs and between starts of the same arm. Differences prevent a claim that
the latency or throughput change isolates frontend overhead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

METRICS = ("ttfa_p50_ms", "ttfa_p95_ms", "audio_s_per_wall_s", "audio_s_total", "wall_s")
COLORS = {"main": "#334155", "pr": "#0f766e"}


def positive(value: object, label: str) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(number) or number <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return number


def check_close(actual: object, expected: float, label: str) -> None:
    if not math.isclose(positive(actual, label), expected, rel_tol=1e-6, abs_tol=1e-4):
        raise ValueError(f"{label}: reported {actual}, recomputed {expected}")


def percentile(values: list[float], q: float) -> float:
    return sorted(values)[round(q * (len(values) - 1))]


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_manifest(path: Path) -> list[dict]:
    with path.open(newline="") as file:
        rows = list(csv.DictReader(file))
    if len(rows) != 4:
        raise ValueError(f"expected four ABBA starts, found {len(rows)}")
    if not {"pair", "order", "arm", "c8"} <= rows[0].keys():
        raise ValueError("manifest needs pair,order,arm,c8")
    if [int(r["pair"]) for r in rows] != [1, 1, 2, 2]:
        raise ValueError("manifest pair order is not 1,1,2,2")
    if [int(r["order"]) for r in rows] != [1, 2, 3, 4]:
        raise ValueError("manifest start order is not 1..4")
    if [r["arm"] for r in rows] != ["main", "pr", "pr", "main"]:
        raise ValueError("manifest arm order is not ABBA")
    return rows


def read_report(manifest: Path, entry: dict) -> tuple[dict, dict[int, dict], tuple]:
    raw_path = Path(entry["c8"])
    path = raw_path if raw_path.is_absolute() else manifest.parent / raw_path
    report = json.loads(path.read_text())
    arm = entry["arm"]
    if (report.get("engine"), report.get("model"), report.get("label")) != (
        "mstar", "cosyvoice3", f"{arm}-c8-plain"
    ):
        raise ValueError(f"{path}: wrong engine, model or label")
    if (report.get("concurrency"), report.get("repeats"), report.get("warmup"),
            report.get("num_sentences")) != (8, 1, 16, 46):
        raise ValueError(f"{path}: wrong c8/46/warmup/repeat setting")
    fields = report.get("request_fields", {})
    if fields.get("seed") != 7 or fields.get("voice") != "zero_shot":
        raise ValueError(f"{path}: wrong seed or voice")
    if "H200" not in json.dumps(report.get("gpu", {})):
        raise ValueError(f"{path}: report does not identify an H200")
    if report.get("errors_total") != 0 or len(report.get("repeats_detail", [])) != 1:
        raise ValueError(f"{path}: failed or missing repeat")
    detail = report["repeats_detail"][0]
    requests = detail.get("results", [])
    if detail.get("requests") != 46 or detail.get("errors") != 0 or len(requests) != 46:
        raise ValueError(f"{path}: incomplete measured requests")
    by_id = {r["sentence_id"]: r for r in requests}
    if len(by_id) != 46 or sorted(by_id) != list(range(1, 47)):
        raise ValueError(f"{path}: incomplete or duplicate sentence ids")
    if any(r.get("error") is not None for r in requests):
        raise ValueError(f"{path}: failed request")
    times = [positive(r["ttfa_s"], "request TTFA") for r in requests]
    durations = [positive(r["audio_s"], "request audio duration") for r in requests]
    wall = positive(detail["wall_s"], "wall time")
    expected = {
        "ttfa_p50_ms": 1000 * percentile(times, .5),
        "ttfa_p95_ms": 1000 * percentile(times, .95),
        "audio_s_total": sum(durations),
        "audio_s_per_wall_s": sum(durations) / wall,
    }
    summary = report["median_over_repeats"]
    for name, value in expected.items():
        check_close(detail[name], value, f"{path}: {name}")
        if name in summary:
            check_close(summary[name], value, f"{path}: median {name}")
    row = {
        "order": int(entry["order"]), "pair": int(entry["pair"]), "arm": arm,
        "path": str(path), "requests": 46, "errors": 0,
        **{k: (wall if k == "wall_s" else expected[k]) for k in METRICS},
    }
    signature = (report.get("sentences_file"), report.get("sample_rate"),
                 json.dumps(fields, sort_keys=True))
    return row, by_id, signature


def compare_pair_audio(by_run: dict[tuple[int, str], dict[int, dict]]) -> list[dict]:
    compared = []
    for pair in (1, 2):
        main = by_run[pair, "main"]
        pr = by_run[pair, "pr"]
        for sentence_id in range(1, 47):
            a, b = main[sentence_id], pr[sentence_id]
            same = (a["pcm_bytes"] == b["pcm_bytes"] and a["audio_s"] == b["audio_s"])
            compared.append({
                "pair": pair, "sentence_id": sentence_id,
                "main_audio_s": a["audio_s"], "pr_audio_s": b["audio_s"],
                "main_pcm_bytes": a["pcm_bytes"], "pr_pcm_bytes": b["pcm_bytes"],
                "delta_audio_s": b["audio_s"] - a["audio_s"],
                "delta_audio_pct": 100 * (b["audio_s"] / a["audio_s"] - 1),
                "equal": same,
            })
    return compared


def compare_within_arm(by_run: dict[tuple[int, str], dict[int, dict]],
                       rows: list[dict]) -> list[dict]:
    order = {(r["pair"], r["arm"]): r["order"] for r in rows}
    compared = []
    for arm in ("main", "pr"):
        first, second = by_run[1, arm], by_run[2, arm]
        for sentence_id in range(1, 47):
            a, b = first[sentence_id], second[sentence_id]
            compared.append({
                "arm": arm, "sentence_id": sentence_id,
                "first_order": order[1, arm], "second_order": order[2, arm],
                "first_audio_s": a["audio_s"], "second_audio_s": b["audio_s"],
                "first_pcm_bytes": a["pcm_bytes"], "second_pcm_bytes": b["pcm_bytes"],
                "delta_audio_s": b["audio_s"] - a["audio_s"],
                "delta_audio_pct": 100 * (b["audio_s"] / a["audio_s"] - 1),
                "equal": a["audio_s"] == b["audio_s"] and a["pcm_bytes"] == b["pcm_bytes"],
            })
    return compared


def pair_rows(rows: list[dict]) -> list[dict]:
    indexed = {(r["pair"], r["arm"]): r for r in rows}
    paired = []
    for pair in (1, 2):
        main, pr = indexed[pair, "main"], indexed[pair, "pr"]
        row = {"pair": pair, "main_order": main["order"], "pr_order": pr["order"]}
        for metric in METRICS:
            row[f"main_{metric}"] = main[metric]
            row[f"pr_{metric}"] = pr[metric]
            row[f"delta_{metric}_pct"] = 100 * (pr[metric] / main[metric] - 1)
        paired.append(row)
    return paired


def summary_table(paired: list[dict], pair_audio: list[dict], within_audio: list[dict],
                  runs: list[dict], prompt_hash: str) -> str:
    pair_mismatches = sum(not r["equal"] for r in pair_audio)
    within_mismatches = sum(not r["equal"] for r in within_audio)
    lines = []
    if pair_mismatches:
        lines += [
            f"**Cannot isolate frontend overhead:** {pair_mismatches}/{len(pair_audio)} matched "
            "main/PR requests generated different PCM lengths despite matched prompts and seed. "
            "TTFA and throughput below are descriptive; output-length and scheduling effects may contribute.",
            "",
        ]
    else:
        lines += [f"All {len(pair_audio)} matched main/PR request PCM lengths agree. "
                  "This controls output length within pairs, but other run effects remain.", ""]
    lines += [
        "| Metric | Main mean | PR mean | Mean paired change | Pairwise range |",
        "|---|---:|---:|---:|---:|",
    ]
    labels = {
        "ttfa_p50_ms": ("TTFA p50", "ms"), "ttfa_p95_ms": ("TTFA p95", "ms"),
        "audio_s_per_wall_s": ("Audio throughput", "audio-s/s"),
        "audio_s_total": ("Generated audio duration", "audio s"),
        "wall_s": ("Measured wall time", "s"),
    }
    for metric in METRICS:
        label, unit = labels[metric]
        main = sum(r[f"main_{metric}"] for r in paired) / 2
        pr = sum(r[f"pr_{metric}"] for r in paired) / 2
        changes = [r[f"delta_{metric}_pct"] for r in paired]
        lines.append(f"| {label} | {main:.2f} {unit} | {pr:.2f} {unit} | "
                     f"{sum(changes)/2:+.2f}% | {min(changes):+.2f}% to {max(changes):+.2f}% |")
    lines += ["", "| Matched pair | Different request lengths | Main audio s | PR audio s | "
              "Largest request difference |",
              "|---:|---:|---:|---:|---:|"]
    for item in paired:
        group = [r for r in pair_audio if r["pair"] == item["pair"]]
        lines.append(f"| {item['pair']} | {sum(not r['equal'] for r in group)}/{len(group)} | "
                     f"{item['main_audio_s_total']:.2f} | {item['pr_audio_s_total']:.2f} | "
                     f"{max(abs(r['delta_audio_s']) for r in group):.2f} s |")
    lines += ["", "| Same-arm starts | Different request lengths | First audio s | Second audio s | "
              "Largest request difference |",
              "|---|---:|---:|---:|---:|"]
    for arm in ("main", "pr"):
        group = [r for r in within_audio if r["arm"] == arm]
        first = next(r for r in runs if r["pair"] == 1 and r["arm"] == arm)
        second = next(r for r in runs if r["pair"] == 2 and r["arm"] == arm)
        lines.append(f"| {arm} order {first['order']} vs {second['order']} | "
                     f"{sum(not r['equal'] for r in group)}/{len(group)} | "
                     f"{first['audio_s_total']:.2f} | {second['audio_s_total']:.2f} | "
                     f"{max(abs(r['delta_audio_s']) for r in group):.2f} s |")
    lines += ["", f"Within-arm variability: {within_mismatches}/{len(within_audio)} request "
              "durations differ across fresh starts. The two paired changes above are observed runs, "
              "not a confidence interval.",
              f"Digit-free 46-prompt file SHA256: `{prompt_hash}`."]
    return "\n".join(lines) + "\n"


def plot(rows: list[dict], paired: list[dict], out: Path, title: str,
         pair_mismatches: int) -> None:
    indexed = {(r["pair"], r["arm"]): r for r in rows}
    fig, axes = plt.subplots(1, 5, figsize=(24, 4.6), constrained_layout=True)
    panels = (
        ("ttfa_p50_ms", "TTFA p50 (ms)"),
        ("ttfa_p95_ms", "TTFA p95 (ms)"),
        ("audio_s_per_wall_s", "Audio seconds / wall second"),
        ("audio_s_total", "Generated audio seconds"),
    )
    for ax, (metric, label) in zip(axes, panels):
        for pair in (1, 2):
            a, b = indexed[pair, "main"], indexed[pair, "pr"]
            ax.plot([pair - .12, pair + .12], [a[metric], b[metric]], color="#94a3b8")
            ax.scatter(pair - .12, a[metric], color=COLORS["main"], label="main" if pair == 1 else None)
            ax.scatter(pair + .12, b[metric], color=COLORS["pr"], label="PR" if pair == 1 else None)
        ax.set(title=label, xlabel="matched pair", xticks=[1, 2])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    ax = axes[4]
    for j, (metric, color, label) in enumerate((
        ("ttfa_p50_ms", "#b45309", "TTFA p50"),
        ("ttfa_p95_ms", "#d97706", "TTFA p95"),
        ("audio_s_per_wall_s", "#0f766e", "throughput"),
        ("audio_s_total", "#64748b", "audio duration"),
    )):
        ax.bar([r["pair"] + (j - 1.5) * .18 for r in paired],
               [r[f"delta_{metric}_pct"] for r in paired], width=.17,
               color=color, label=label)
    ax.axhline(0, color="#334155", linewidth=.8)
    ax.set(title="PR relative to main", xlabel="matched pair", ylabel="paired change (%)", xticks=[1, 2])
    ax.legend(frameon=False, fontsize=8)
    ax.grid(axis="y", alpha=.2)
    fig.suptitle(title)
    note = (f"{pair_mismatches}/92 matched request PCM lengths differ; frontend overhead is not isolated. "
            "See per_request_audio.csv." if pair_mismatches else
            "All 92 paired request PCM lengths agree. For paired percentage changes, see paired.csv.")
    fig.supxlabel(note, fontsize=9, color="#9a3412" if pair_mismatches else "#475569")
    for suffix in ("png", "svg"):
        figure = out / f"pr631_digitfree_c8.{suffix}"
        fig.savefig(figure, dpi=180, bbox_inches="tight")
        if suffix == "svg":
            figure.write_text("\n".join(line.rstrip() for line in figure.read_text().splitlines()) + "\n")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--prompts", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--main-head", required=True)
    parser.add_argument("--pr-head", required=True)
    args = parser.parse_args()
    prompt_bytes = args.prompts.read_bytes()
    prompts = [line.strip() for line in prompt_bytes.decode("utf-8").splitlines() if line.strip()]
    if len(prompts) != 46 or any(ch.isdecimal() for prompt in prompts for ch in prompt):
        raise ValueError("prompt file must have exactly 46 nonblank digit-free lines")
    signature = None
    rows = []
    by_run = {}
    for entry in read_manifest(args.manifest):
        row, requests, current_signature = read_report(args.manifest, entry)
        if signature is not None and current_signature != signature:
            raise ValueError("benchmark inputs differ between starts")
        signature = current_signature
        rows.append(row)
        by_run[row["pair"], row["arm"]] = requests
    compared = compare_pair_audio(by_run)
    within = compare_within_arm(by_run, rows)
    paired = pair_rows(rows)
    args.out.mkdir(parents=True, exist_ok=True)
    write_csv(args.out / "runs.csv", rows)
    write_csv(args.out / "paired.csv", paired)
    write_csv(args.out / "per_request_audio.csv", compared)
    write_csv(args.out / "within_arm_audio.csv", within)
    table = summary_table(paired, compared, within, rows,
                          hashlib.sha256(prompt_bytes).hexdigest())
    (args.out / "summary.md").write_text(table)
    mismatches = sum(not row["equal"] for row in compared)
    title = (f"#631 digit-free c8 on H200 | main {args.main_head[:10]} vs PR "
             f"{args.pr_head[:10]} | {mismatches}/92 PCM lengths differ")
    plot(rows, paired, args.out, title, mismatches)
    print(table, end="")


if __name__ == "__main__":
    main()
