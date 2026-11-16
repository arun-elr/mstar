"""Plot matched Chatterbox sweep and burst runs for #1567.

Manifest CSV: pair,order,arm,burst,sweep_dir. Each pair has one main and one pr
start. Paths may be relative to the manifest. Sweep JSON files are the exact
#636 client outputs (mixed_c4.json, mixed_c8.json, mixed_c16.json). Burst JSON
is from mstar-chatterbox-burst.py. Both are raw per-request reports.
"""

import argparse
import csv
import json
import math
import statistics
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

CONCURRENCIES = (4, 8, 16)
EXPECTED_N = {4: 48, 8: 48, 16: 96}
COLORS = {"main": "#1565C0", "pr": "#E65100"}
METRICS = (
    "burst_short_ttfa_p50_s",
    "burst_short_ttfa_p95_s",
    "burst_short_e2e_p50_s",
    "burst_short_e2e_p95_s",
    "burst_long_e2e_s",
    "burst_audio_s_per_s",
    *(f"mixed_c{c}_ttfa_p50_s" for c in CONCURRENCIES),
    *(f"mixed_c{c}_ttfa_p95_s" for c in CONCURRENCIES),
    *(f"mixed_c{c}_{lane}_ttfa_{q}_s" for c in CONCURRENCIES
      for lane in ("cfg0", "default") for q in ("p50", "p95")),
    *(f"mixed_c{c}_audio_s_per_s" for c in CONCURRENCIES),
)


def path(base: Path, value: str) -> Path:
    p = Path(value)
    return p if p.is_absolute() else base / p


def positive(value: object, label: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} is not numeric") from exc
    if not math.isfinite(result) or result <= 0:
        raise ValueError(f"{label} must be positive and finite")
    return result


def near(actual: object, expected: float, label: str) -> None:
    if not math.isclose(positive(actual, label), expected, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(f"{label}: reported {actual}, recomputed {expected}")


def pct(values: list[float], q: float) -> float:
    values = sorted(values)
    return values[round(q * (len(values) - 1))]


def read_manifest(file: Path) -> list[dict]:
    with file.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) not in (4, 6):
        raise ValueError(f"expected 4 or 6 ABBA starts, found {len(rows)}")
    required = {"pair", "order", "arm", "burst", "sweep_dir"}
    if not required <= rows[0].keys():
        raise ValueError(f"manifest needs columns {sorted(required)}")
    if [int(r["order"]) for r in rows] != list(range(1, len(rows) + 1)):
        raise ValueError("order must be recorded as 1..N")
    if [r["arm"] for r in rows] != ["main", "pr", "pr", "main", "main", "pr"][:len(rows)]:
        raise ValueError("unexpected ABBA arm order")
    if [int(r["pair"]) for r in rows] != [1, 1, 2, 2, 3, 3][:len(rows)]:
        raise ValueError("unexpected ABBA pair numbering")
    for row in rows:
        if row["arm"] not in COLORS:
            raise ValueError(f"unknown arm {row['arm']!r}")
        row["pair"] = int(row["pair"])
        row["order"] = int(row["order"])
        row["burst"] = path(file.parent, row["burst"])
        row["sweep_dir"] = path(file.parent, row["sweep_dir"])
    pairs = defaultdict(list)
    for row in rows:
        pairs[row["pair"]].append(row["arm"])
    if any(sorted(arms) != ["main", "pr"] for arms in pairs.values()):
        raise ValueError("each pair must contain exactly one main and one pr start")
    return sorted(rows, key=lambda r: r["order"])


def extract(row: dict) -> dict:
    burst = json.loads(row["burst"].read_text())
    summary = burst["summary"]
    if burst.get("protocol") != "pcm_s16le_24000" or burst.get("delay_ms") != 50:
        raise ValueError(f"unexpected burst protocol or delay: {row['burst']}")
    if burst.get("start") != row["order"]:
        raise ValueError(f"burst order disagrees with manifest: {row['burst']}")
    if summary["ok"] != 9 or summary["total"] != 9 or not summary["long_active_at_burst"]:
        raise ValueError(f"invalid or nonoverlapping burst: {row['burst']}")
    if burst.get("arm") != row["arm"]:
        raise ValueError(f"burst arm disagrees with manifest: {row['burst']}")
    burst_rows = burst["rows"]
    if len(burst_rows) != 9 or any(
        r["status"] != 200 or r["audio_bytes"] <= 0 or r["error"] is not None
        for r in burst_rows
    ):
        raise ValueError(f"missing/failed PCM stream: {row['burst']}")
    if [r["label"] for r in burst_rows] != ["long", *(f"short-{i}" for i in range(8))]:
        raise ValueError(f"wrong burst labels: {row['burst']}")
    if [r["seed"] for r in burst_rows] != list(range(48600, 48609)):
        raise ValueError(f"wrong burst seeds: {row['burst']}")
    shorts = burst_rows[1:]
    long_finish = burst_rows[0]["finish_at_s"]
    if any(short["submit_at_s"] >= long_finish for short in shorts):
        raise ValueError(f"burst shorts did not all overlap the long stream: {row['burst']}")
    near(summary["long_finish_after_short_submit_s"],
         long_finish - shorts[0]["submit_at_s"], "long finish after short submit")
    if [r["cfg_weight"] for r in burst_rows] != [0.0, *(0.0 if i % 2 == 0 else None for i in range(8))]:
        raise ValueError(f"wrong burst CFG mix: {row['burst']}")
    for r in burst_rows:
        near(r["audio_s"], r["audio_bytes"] / 48000, "burst audio duration")
        near(r["ttfa_s"], r["first_byte_at_s"] - r["submit_at_s"], "burst first-audio latency")
        near(r["e2e_s"], r["finish_at_s"] - r["submit_at_s"], "burst end-to-end latency")
    near(summary["wall_s"], max(r["finish_at_s"] for r in burst_rows), "burst wall time")
    near(summary["audio_s_per_s"], sum(r["audio_s"] for r in burst_rows) / summary["wall_s"],
         "burst audio throughput")
    near(summary["short_ttfa_p50_s"], pct([r["ttfa_s"] for r in shorts], .5), "burst short TTFA p50")
    near(summary["short_ttfa_p95_s"], pct([r["ttfa_s"] for r in shorts], .95), "burst short TTFA p95")
    near(summary["short_e2e_p50_s"], pct([r["e2e_s"] for r in shorts], .5), "burst short E2E p50")
    near(summary["short_e2e_p95_s"], pct([r["e2e_s"] for r in shorts], .95), "burst short E2E p95")
    near(summary["long_e2e_s"], burst_rows[0]["e2e_s"], "burst long E2E")
    out = {
        "pair": row["pair"],
        "order": row["order"],
        "arm": row["arm"],
        "burst_path": str(row["burst"]),
        "sweep_dir": str(row["sweep_dir"]),
        "burst_short_ttfa_p50_s": summary["short_ttfa_p50_s"],
        "burst_short_ttfa_p95_s": summary["short_ttfa_p95_s"],
        "burst_short_e2e_p50_s": summary["short_e2e_p50_s"],
        "burst_short_e2e_p95_s": summary["short_e2e_p95_s"],
        "burst_long_e2e_s": summary["long_e2e_s"],
        "burst_audio_s_per_s": summary["audio_s_per_s"],
        "burst_audio_s_total": sum(r["audio_s"] for r in burst_rows),
        "_burst_signature": [(r["label"], r["seed"], r["prompt"], r["cfg_weight"]) for r in burst_rows],
        "_audio_burst": {r["label"]: r["audio_s"] for r in burst_rows},
    }
    for c in CONCURRENCIES:
        file = row["sweep_dir"] / f"mixed_c{c}.json"
        data = json.loads(file.read_text())
        s = data["summary"]
        if s["mode"] != "mixed" or s["c"] != c or s["n"] != EXPECTED_N[c] or s["ok"] != s["n"]:
            raise ValueError(f"invalid mixed sweep cell: {file}")
        requests = data["rows"]
        if len(requests) != s["n"] or any(
            r["status"] != 200 or r["audio_s"] <= 0 or r["ttfa_s"] is None
            for r in requests
        ):
            raise ValueError(f"missing/failed sweep request: {file}")
        if sorted(r["i"] for r in requests) != list(range(EXPECTED_N[c])):
            raise ValueError(f"wrong sweep request ids: {file}")
        if any(r["cfg_weight"] != (0.0 if r["i"] % 2 == 0 else None) for r in requests):
            raise ValueError(f"wrong sweep CFG mix: {file}")
        near(s["wall_s"], max(r["end"] for r in requests) - min(r["start"] for r in requests),
             "sweep wall time")
        near(s["audio_total_s"], sum(r["audio_s"] for r in requests), "sweep audio duration")
        near(s["audio_s_per_s"], s["audio_total_s"] / s["wall_s"], "sweep audio throughput")
        for q in ("p50", "p95"):
            near(s["ttfa_s"][q], pct([positive(r["ttfa_s"], "sweep TTFA") for r in requests],
                                     .5 if q == "p50" else .95), f"sweep TTFA {q}")
        for lane in ("cfg0", "default"):
            lane_rows = [r for r in requests if (r["cfg_weight"] == 0.0) == (lane == "cfg0")]
            lane_summary = s[lane]
            if lane_summary["n"] != EXPECTED_N[c] // 2 or len(lane_rows) != EXPECTED_N[c] // 2:
                raise ValueError(f"wrong {lane} request count: {file}")
            for q, level in (("p50", .5), ("p95", .95)):
                value = pct([positive(r["ttfa_s"], f"{lane} TTFA") for r in lane_rows], level)
                near(lane_summary["ttfa_s"][q], value, f"{file}: {lane} TTFA {q}")
                out[f"mixed_c{c}_{lane}_ttfa_{q}_s"] = value
        out[f"mixed_c{c}_ttfa_p50_s"] = s["ttfa_s"]["p50"]
        out[f"mixed_c{c}_ttfa_p95_s"] = s["ttfa_s"]["p95"]
        out[f"mixed_c{c}_audio_s_per_s"] = s["audio_s_per_s"]
        out[f"mixed_c{c}_audio_s_total"] = s["audio_total_s"]
        out[f"mixed_c{c}_wall_s"] = s["wall_s"]
        out[f"_sweep_signature_c{c}"] = sorted((r["i"], r["cfg_weight"]) for r in requests)
        out[f"_audio_c{c}"] = {r["i"]: r["audio_s"] for r in requests}
    if any(out[m] is None for m in METRICS):
        raise ValueError(f"missing benchmark metric for order {row['order']}")
    return out


def write_csv(file: Path, rows: list[dict]) -> None:
    with file.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def paired(rows: list[dict]) -> list[dict]:
    by_pair = defaultdict(dict)
    for row in rows:
        by_pair[row["pair"]][row["arm"]] = row
    out = []
    for pair, arms in sorted(by_pair.items()):
        for metric in METRICS:
            main = arms["main"][metric]
            pr = arms["pr"][metric]
            out.append({
                "pair": pair, "metric": metric,
                "main": main, "pr": pr,
                "pr_minus_main": pr - main,
                "pr_vs_main_pct": (pr / main - 1) * 100 if main else None,
            })
    return out


def metric_label(metric: str) -> str:
    if metric == "burst_long_e2e_s":
        return "Burst long last-byte (s)"
    if metric.startswith("burst_"):
        return (metric.replace("burst_", "Burst ").replace("short_", "short ")
                .replace("long_", "long ").replace("ttfa_", "TTFA ")
                .replace("e2e_", "last-byte ").replace("audio_s_per_s", "audio-s/s")
                .replace("_s", " (s)"))
    parts = metric.split("_")
    c = parts[1]
    rest = " ".join(parts[2:]).replace("ttfa", "TTFA").replace("audio s per s", "audio-s/s")
    if rest.endswith(" s"):
        rest = rest[:-2] + " (s)"
    return f"{c} {rest}"


def compare_audio(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    by_pair = {(r["pair"], r["arm"]): r for r in rows}
    requests = []
    cells = []
    for pair in sorted({r["pair"] for r in rows}):
        main, pr = by_pair[pair, "main"], by_pair[pair, "pr"]
        for cell in ("burst", *(f"c{c}" for c in CONCURRENCIES)):
            a, b = main[f"_audio_{cell}"], pr[f"_audio_{cell}"]
            if set(a) != set(b):
                raise ValueError(f"pair {pair} {cell}: request identities differ")
            for request_id in sorted(a):
                main_s = positive(a[request_id], "main audio duration")
                pr_s = positive(b[request_id], "PR audio duration")
                requests.append({
                    "pair": pair, "cell": cell, "request_id": request_id,
                    "main_audio_s": main_s, "pr_audio_s": pr_s,
                    "delta_s": pr_s - main_s,
                    "delta_pct": 100 * (pr_s / main_s - 1),
                    "equal": main_s == pr_s,
                })
            group = [r for r in requests if r["pair"] == pair and r["cell"] == cell]
            main_total = sum(r["main_audio_s"] for r in group)
            pr_total = sum(r["pr_audio_s"] for r in group)
            cells.append({
                "pair": pair, "cell": cell, "requests": len(group),
                "duration_mismatches": sum(not r["equal"] for r in group),
                "main_audio_s_total": main_total,
                "pr_audio_s_total": pr_total,
                "delta_audio_s_total_pct": 100 * (pr_total / main_total - 1),
                "max_abs_request_delta_s": max(abs(r["delta_s"]) for r in group),
            })
    return requests, cells


def observed(ax, rows: list[dict], metric: str, offset: float, color: str, arm: str) -> None:
    xs = []
    medians = []
    for c in CONCURRENCIES:
        vals = [r[f"mixed_c{c}_{metric}"] for r in rows if r["arm"] == arm]
        x = c + offset
        xs.append(x)
        medians.append(statistics.median(vals))
        ax.vlines(x, min(vals), max(vals), color=color, alpha=.5, linewidth=2)
        ax.scatter([x] * len(vals), vals, color=color, alpha=.55, s=25)
    ax.plot(xs, medians, color=color, marker="o", linewidth=2, label=f"{arm} median")


def sweep_plot(rows: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.7), constrained_layout=True)
    for arm, offset in (("main", -.25), ("pr", .25)):
        observed(axes[0], rows, "ttfa_p50_s", offset, COLORS[arm], arm)
        observed(axes[1], rows, "ttfa_p95_s", offset, COLORS[arm], arm)
        observed(axes[2], rows, "audio_s_per_s", offset, COLORS[arm], arm)
    axes[0].set(ylabel="First-audio p50 (s), lower is better", xlabel="Concurrent streams")
    axes[1].set(ylabel="First-audio p95 (s), lower is better", xlabel="Concurrent streams")
    axes[2].set(ylabel="Audio seconds / wall second, higher is better", xlabel="Concurrent streams")
    for ax in axes:
        ax.set_xticks(CONCURRENCIES)
        ax.grid(alpha=.2)
        ax.legend(frameon=False)
    fig.suptitle(title + "\nMixed CFG closed loop. Dots are fresh starts; bars are observed min/max, not confidence intervals.")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"chatterbox_mixed_sweep.{suffix}", dpi=170)
    plt.close(fig)


def lanes_plot(rows: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7), constrained_layout=True)
    for ax, (lane, q) in zip(axes, ((lane, q) for lane in ("cfg0", "default")
                                     for q in ("p50", "p95"))):
        for arm, offset in (("main", -.25), ("pr", .25)):
            observed(ax, rows, f"{lane}_ttfa_{q}_s", offset, COLORS[arm], arm)
        ax.set(title=f"{lane} first audio {q}", xlabel="Concurrent streams", ylabel="seconds")
        ax.set_xticks(CONCURRENCIES)
        ax.grid(alpha=.2)
        ax.legend(frameon=False, fontsize=8)
    fig.suptitle(title + "\nPer CFG lane. Dots are fresh starts; bars are observed min/max, not confidence intervals.")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"chatterbox_cfg_lanes.{suffix}", dpi=170)
    plt.close(fig)


def duration_plot(cells: list[dict], out: Path, title: str) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.7), constrained_layout=True)
    for ax, cell in zip(axes, ("burst", "c4", "c8", "c16")):
        selected = [r for r in cells if r["cell"] == cell]
        for r in selected:
            pair = r["pair"]
            ax.plot([pair - .12, pair + .12],
                    [r["main_audio_s_total"], r["pr_audio_s_total"]], color="#9E9E9E")
            ax.scatter(pair - .12, r["main_audio_s_total"], color=COLORS["main"],
                       label="main" if pair == selected[0]["pair"] else None)
            ax.scatter(pair + .12, r["pr_audio_s_total"], color=COLORS["pr"],
                       label="PR" if pair == selected[0]["pair"] else None)
        ax.set(title=f"{cell}: generated audio", xlabel="matched pair", ylabel="audio seconds")
        ax.set_xticks([r["pair"] for r in selected])
        ax.grid(axis="y", alpha=.2)
    axes[0].legend(frameon=False)
    fig.suptitle(title + "\nOutput duration can change throughput even when scheduling is unchanged.")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"chatterbox_audio_duration.{suffix}", dpi=170)
    plt.close(fig)


def burst_plot(rows: list[dict], out: Path, title: str) -> None:
    figures = (
        ("burst_short_ttfa_p50_s", "Short first-audio p50 (s)"),
        ("burst_short_ttfa_p95_s", "Short first-audio p95 (s)"),
        ("burst_short_e2e_p50_s", "Short last-byte p50 (s)"),
        ("burst_short_e2e_p95_s", "Short last-byte p95 (s)"),
        ("burst_long_e2e_s", "Long last-byte (s)"),
        ("burst_audio_s_per_s", "Audio seconds / wall second"),
    )
    fig, axes = plt.subplots(2, 3, figsize=(16, 8), constrained_layout=True)
    by_pair = defaultdict(list)
    for row in rows:
        by_pair[row["pair"]].append(row)
    for ax, (metric, label) in zip(axes.flat, figures):
        for _, pair_rows in sorted(by_pair.items()):
            pair_rows.sort(key=lambda r: r["order"])
            ax.plot([r["order"] for r in pair_rows], [r[metric] for r in pair_rows],
                    color="#9E9E9E", alpha=.55, linewidth=1)
        for arm in ("main", "pr"):
            selected = [r for r in rows if r["arm"] == arm]
            ax.scatter([r["order"] for r in selected], [r[metric] for r in selected],
                       label=arm, color=COLORS[arm], s=44, zorder=3)
        ax.set(xlabel="Fresh-start order", ylabel=label)
        ax.set_xticks([r["order"] for r in rows])
        ax.grid(alpha=.2)
    axes[0, 0].legend(frameon=False)
    fig.suptitle(title + "\nNew fixed long/short prompts, same contention shape as #636; not the unpublished #636 prompts.")
    for suffix in ("png", "svg"):
        fig.savefig(out / f"chatterbox_mixed_burst.{suffix}", dpi=170)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--main-head", required=True)
    parser.add_argument("--pr-head", required=True)
    parser.add_argument("--setting", default="peek off, cap 8")
    args = parser.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    rows = [extract(r) for r in read_manifest(args.manifest)]
    signatures = {}
    for row in rows:
        for key in ("_burst_signature", *(f"_sweep_signature_c{c}" for c in CONCURRENCIES)):
            value = row.pop(key)
            if key in signatures and value != signatures[key]:
                raise ValueError(f"{key} traffic differs across fresh starts")
            signatures[key] = value
    request_audio, cell_audio = compare_audio(rows)
    for row in rows:
        for key in ("_audio_burst", *(f"_audio_c{c}" for c in CONCURRENCIES)):
            row.pop(key)
    pair_rows = paired(rows)
    write_csv(args.out / "per_start.csv", rows)
    write_csv(args.out / "paired.csv", pair_rows)
    write_csv(args.out / "request_audio_comparison.csv", request_audio)
    write_csv(args.out / "audio_comparability.csv", cell_audio)
    mismatches = sum(not row["equal"] for row in request_audio)
    duration_note = "audio durations match" if mismatches == 0 else f"{mismatches} audio durations differ"
    title = (f"#1567 H200, {args.setting} | main {args.main_head[:10]} vs PR "
             f"{args.pr_head[:10]} | {duration_note}")
    sweep_plot(rows, args.out, title)
    lanes_plot(rows, args.out, title)
    burst_plot(rows, args.out, title)
    duration_plot(cell_audio, args.out, title)
    for svg in args.out.glob("*.svg"):
        svg.write_text("\n".join(line.rstrip() for line in svg.read_text().splitlines()) + "\n")
    lines = [
        f"Validated {len(rows)} fresh starts in {len(set(r['pair'] for r in rows))} matched pairs.",
        "| Metric | Main mean | PR mean | Mean paired change | Pairwise range |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric in METRICS:
        cells = [r for r in pair_rows if r["metric"] == metric]
        mean_main = statistics.mean(r["main"] for r in cells)
        mean_pr = statistics.mean(r["pr"] for r in cells)
        changes = [r["pr_vs_main_pct"] for r in cells]
        lines.append(f"| {metric_label(metric)} | {mean_main:.3f} | {mean_pr:.3f} | "
                     f"{statistics.mean(changes):+.2f}% | {min(changes):+.2f}% to {max(changes):+.2f}% |")
    lines.extend((
        "",
        "| Pair | Cell | Request duration mismatches | Main audio s | PR audio s | Total duration change |",
        "|---:|---|---:|---:|---:|---:|",
    ))
    for cell in cell_audio:
        lines.append(f"| {cell['pair']} | {cell['cell']} | {cell['duration_mismatches']}/{cell['requests']} | "
                     f"{cell['main_audio_s_total']:.3f} | {cell['pr_audio_s_total']:.3f} | "
                     f"{cell['delta_audio_s_total_pct']:+.2f}% |")
    lines.append("\nPairwise ranges show observed runs, not confidence intervals.")
    if mismatches:
        lines.append(f"**Duration mismatch:** {mismatches}/{len(request_audio)} matched requests differ. "
                     "Throughput and latency comparisons cannot be read as isolated scheduler effects; "
                     "use the per-request CSV and duration chart to inspect the change.")
    else:
        lines.append(f"All {len(request_audio)} matched request audio durations agree exactly. "
                     "Output length is controlled for this run; other run effects remain.")
    table = "\n".join(lines) + "\n"
    (args.out / "summary.md").write_text(table)
    print(table, end="")


if __name__ == "__main__":
    main()
