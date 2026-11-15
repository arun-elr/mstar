#!/usr/bin/env python3
"""Validate the six pinned Qwen runs and draw the combined-stack comparison."""

import hashlib
import json
import math
import statistics
import subprocess
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
ORDER = ("main", "e5", "e5", "main", "main", "e5")
BASE = "f9073827da6111ba76c085b2a715e53cd0ef51c2"
STACK = "532e656636eb4eaa64d9fcd61e98d722a38d4449"
INK = "#172638"
MUTED = "#536575"
GRID = "#dce4e8"
TEAL = "#0b8f82"
NAVY = "#22354c"


def load(name):
    return json.loads((SOURCE / name).read_text())


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def txt(x, y, value, size=15, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def line(x1, y1, x2, y2, color=GRID, width=1):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}" stroke-width="{width}"/>'


def dot(x, y, color):
    return f'<circle cx="{x}" cy="{y}" r="7" fill="{color}" stroke="white" stroke-width="2"/>'


def validate():
    summary = load("analysis-summary.json")
    assert summary["source_pins"] == {"main_merge_base": BASE, "e5": STACK}
    assert summary["order"] == list(ORDER)
    assert "ABBA six fresh-process runs complete" in (SOURCE / "complete.txt").read_text()
    assert "NVIDIA H200" in (SOURCE / "gpu.txt").read_text()
    assert all("source matched" in (SOURCE / f"{arm}-source-match.log").read_text() for arm in ("main", "e5"))
    assert all(line.endswith("OK") for line in (SOURCE / "qwen-files-verified.log").read_text().splitlines())
    assert (SOURCE / "qwen-bench.sha256").read_text().split()[0] == sha(SOURCE / "qwen-bench.yaml")
    captures = load("capture-summary.json")["runs"]
    assert len(captures) == 6
    assert [run["arm"] for run in captures] == list(ORDER)
    assert all(run["capture_mib"] and min(run["capture_mib"]) > 0 for run in captures)
    order_log = (SOURCE / "order.log").read_text()
    runs = []
    for i, arm in enumerate(ORDER, 1):
        assert f"run={i} arm={arm} start=" in order_log and f"run={i} arm={arm} end=" in order_log
        run = load(f"run-{i}-{arm}.json")
        assert run["label"] == arm and run["model"] == "Qwen3.5-2B" and run["max_tokens"] == 32
        assert {row["concurrency"] for row in run["rows"]} == {1, 4, 16}
        for row in run["rows"]:
            assert row["requests"] == len(row["samples"]) == 50
            assert all(sample["tokens"] == 32 for sample in row["samples"])
            assert math.isclose(row["throughput_tok_s"], 1600 / row["wall_s"], rel_tol=1e-9)
            assert math.isclose(row["total_ms"]["p50"], statistics.median(s["total_ms"] for s in row["samples"]), rel_tol=1e-9)
        runs.append(run)
    for c in (1, 4, 16):
        values = summary["per_concurrency"][str(c)]
        for arm in ("main", "e5"):
            rows = [next(row for row in run["rows"] if row["concurrency"] == c) for run in runs if run["label"] == arm]
            for summary_key, value in (("throughput_tok_s_median", [r["throughput_tok_s"] for r in rows]),
                                       ("total_p50_ms_median", [r["total_ms"]["p50"] for r in rows])):
                assert math.isclose(values[arm][summary_key], statistics.median(value), rel_tol=1e-9)
            for prompt in range(4):
                hashes = {s["output_sha256"] for row in rows for s in row["samples"] if s["prompt_index"] == prompt}
                assert len(hashes) == summary["hashes"][str(c)][arm][str(prompt)]
                assert summary["hashes_cross_arm"][str(c)][str(prompt)]["same_set"]
        for ratio_key, metric in (("e5_over_main_total_p50", "total_p50_ms_median"),
                                  ("e5_over_main_throughput", "throughput_tok_s_median")):
            ratio = values["e5"][metric] / values["main"][metric]
            assert math.isclose(ratio, values["ratios"][ratio_key], rel_tol=1e-9)
    assert all(summary["hashes"][str(c)][a]["0"] == 2 for c in (4, 16) for a in ("main", "e5"))
    return summary


def chart(summary):
    parts = [txt(56, 55, "Qwen3.5-2B combined-stack A/B", 28, weight=650),
             txt(56, 86, "One H200; six fresh servers in base / stack / stack / base / base / stack order", 15, MUTED),
             txt(56, 118, "3 runs per arm, 8 warmups + 50 measured requests at each concurrency", 14, MUTED),
             txt(56, 165, "Streaming throughput", 21, weight=600),
             txt(56, 188, "Median of three process rates; positive change means more output chunks per second", 14, MUTED)]
    x0, scale = 750, 55  # 55 px per percentage point
    for top, bottom in ((210, 368), (440, 598)):
        for tick in (-4, -2, 0, 2, 4):
            x = x0 + tick * scale
            parts += [line(x, top, x, bottom, NAVY if tick == 0 else GRID, 2 if tick == 0 else 1),
                      txt(x, bottom + 23, f"{tick:+d}%" if tick else "0", 12, MUTED, anchor="middle")]
    for c, y in ((1, 247), (4, 299), (16, 351)):
        row = summary["per_concurrency"][str(c)]
        a = row["main"]["throughput_tok_s_median"]
        b = row["e5"]["throughput_tok_s_median"]
        delta = 100 * (b / a - 1)
        x = x0 + delta * scale
        parts += [txt(56, y + 5, f"C{c}: base {a:,.0f}, stack {b:,.0f} chunks/s", 15),
                  line(x0, y, x, y, TEAL, 5), dot(x, y, TEAL),
                  txt(1057, y + 5, f"{delta:+.2f}%", 15, TEAL, 600, anchor="end")]
    parts += [txt(56, 425, "Total request latency", 21, weight=600),
              txt(56, 448, "Median of three process p50 values; positive change means slower", 14, MUTED)]
    for c, y in ((1, 485), (4, 537), (16, 589)):
        row = summary["per_concurrency"][str(c)]
        a = row["main"]["total_p50_ms_median"]
        b = row["e5"]["total_p50_ms_median"]
        delta = 100 * (b / a - 1)
        x = x0 + delta * scale
        parts += [txt(56, y + 5, f"C{c}: base {a:.1f}, stack {b:.1f} ms", 15),
                  line(x0, y, x, y, TEAL, 5), dot(x, y, TEAL),
                  txt(1057, y + 5, f"{delta:+.2f}%", 15, TEAL, 600, anchor="end")]
    parts += [txt(56, 660, "Base f9073827 is the pinned merge base; stack 532e6566 includes all episode, attention and Praxis commits.", 12, MUTED),
              txt(56, 681, "Four short prompts repeat with prefix caching. Prompt 0 has two output hashes in both arms at C4/C16.", 12, MUTED),
              txt(56, 702, "Hash sets match across arms, not per-request outputs. This is a Qwen check, not Praxis serving latency.", 12, MUTED),
              txt(56, 723, "Three process runs per arm; observed ranges and source pins are in the raw data and provenance.", 12, MUTED)]
    svg = ('<svg xmlns="http://www.w3.org/2000/svg" width="1140" height="752" viewBox="0 0 1140 752">'
           '<rect width="100%" height="100%" fill="white"/>'
           '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif}</style>' + "".join(parts) + "</svg>\n")
    path = HERE / "qwen-stack-abba.svg"
    path.write_text(svg)
    subprocess.run(["rsvg-convert", "-o", str(HERE / "qwen-stack-abba.png"), str(path)], check=True)


def provenance(summary):
    files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    data = {
        "base_commit": BASE,
        "stack_commit": STACK,
        "gpu": (SOURCE / "gpu.txt").read_text().strip(),
        "order": list(ORDER),
        "workload": "Qwen3.5-2B, four short repeated prompts with prefix caching, 32 streamed output chunks/request; 8 warmups and 50 measured requests at concurrency 1, 4 and 16 for each fresh process",
        "settings": "same model/config bytes, source-matched Rust extension and server, -O0 and nonzero LLM graph capture on each arm",
        "summary_statistic": "median of three per-process throughput rates and p50 total request latencies",
        "scope": "pinned merge-base vs full combined stack; no per-PR attribution and no Praxis model serving measurement",
        "output_limit": "prompt 0 has two output hashes within each arm at C4 and C16; sets match across arms, not per-request outputs",
        "run_limit": "three fresh processes per arm on one H200 and repeated prompts; observed changes do not generalize to mixed traffic",
        "interrupted_first_attempt": "excluded; this asset contains only the six-run retry on one boot",
        "per_concurrency": summary["per_concurrency"],
        "hash_counts": summary["hashes"],
        "hash_sets_cross_arm": summary["hashes_cross_arm"],
        "source_sha256": {str(path.relative_to(HERE)): sha(path) for path in files},
    }
    (HERE / "provenance.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


if __name__ == "__main__":
    summary = validate()
    chart(summary)
    provenance(summary)
