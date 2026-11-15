#!/usr/bin/env python3
"""Validate the frozen paired run and draw its three descriptive charts."""

import hashlib
import json
import statistics
import subprocess
from html import escape
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOURCE = HERE / "source"
REF = "#24364b"
MSTAR = "#0b8f82"
INK = "#172638"
MUTED = "#536575"
GRID = "#dce4e8"


def read(name):
    return json.loads((SOURCE / name).read_text())


def episodes(name):
    return [json.loads(line) for line in (SOURCE / name).read_text().splitlines()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def txt(x, y, value, size=15, color=INK, weight=400, anchor="start"):
    return (f'<text x="{x}" y="{y}" fill="{color}" font-size="{size}" '
            f'font-weight="{weight}" text-anchor="{anchor}">{escape(str(value))}</text>')


def rect(x, y, width, height, color, radius=0):
    return (f'<rect x="{x}" y="{y}" width="{width}" height="{height}" '
            f'rx="{radius}" fill="{color}"/>')


def line(x1, y1, x2, y2, color=GRID):
    return f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="{color}"/>'


def save(name, width, height, parts):
    svg = (f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
           f'viewBox="0 0 {width} {height}"><rect width="100%" height="100%" fill="white"/>'
           '<style>text{font-family:Inter,DejaVu Sans,Arial,sans-serif}</style>' + "".join(parts) + "</svg>\n")
    path = HERE / f"{name}.svg"
    path.write_text(svg)
    subprocess.run(["rsvg-convert", "-o", str(HERE / f"{name}.png"), str(path)], check=True)


def validate():
    report = read("paired-56-report.json")
    commands = read("paired-commands.json")
    m = read("mstar-matched.json")
    r = read("reference-matched.json")
    me = episodes("mstar-episodes.jsonl")
    re = episodes("reference-episodes.jsonl")
    key = lambda a: (a["task"], a["seed"], a["sampler_seed"])
    assert report["units"] == len(me) == len(re) == 56
    assert len({key(a) for a in me}) == len({key(a) for a in re}) == 56
    assert {key(a) for a in me} == {key(a) for a in re}
    assert (sum(a["success"] for a in me), sum(a["success"] for a in re)) == (54, 55)
    assert (report["total_mstar_successes"], report["total_reference_successes"]) == (54, 55)
    assert {a["server_meta"]["code_version"] for a in me} == {"f574af84969a3e5191b5e781cd30e12bd029165f"}
    assert m["reference_client_commit"] == r["reference_client_commit"] == "68f4d7a254b4bf41951d2e97f44b07f6cac2bf69"
    assert m["script_sha256"] == r["script_sha256"]
    assert m["observation_sha256"] == r["observation_sha256"]
    assert commands["same_observation_bytes"] and commands["compared_command_values"] == 4485
    assert m["sessions"] == r["sessions"] == 5 and m["warmup_sessions"] == r["warmup_sessions"] == 1
    for arm in (m, r):
        for call in range(3):
            samples = [a["round_trip_ms"] for a in arm["calls"] if a["session"] >= 1 and a["call"] == call]
            assert len(samples) == arm["summary"][str(call)]["n"] == 4
            assert abs(statistics.median(samples) - arm["summary"][str(call)]["p50_ms"]) < 1e-5
    assert "NVIDIA H200" in (SOURCE / "gpu.txt").read_text()
    assert "f574 Rust extension and server source matched" in (SOURCE / "source-match.log").read_text()
    return report, commands, m, r, me, re


def latency(m, r):
    p = [txt(58, 55, "Praxis DROID session latency", 28, weight=650),
         txt(58, 86, "One H200, same observation bytes, sequential server arms", 15, MUTED),
         rect(58, 108, 14, 14, REF, 2), txt(82, 121, "Reference", 14, MUTED),
         rect(206, 108, 14, 14, MSTAR, 2), txt(230, 121, "M*", 14, MUTED),
         txt(935, 121, "p50 round trip", 14, MUTED, anchor="end")]
    x0, scale = 300, 530 / 600
    for tick in range(0, 601, 100):
        x = x0 + scale * tick
        p += [line(x, 156, x, 475), txt(x, 496, tick, 12, MUTED, anchor="middle")]
    for k, y, label in ((0, 194, "Cold, 21 rows"), (1, 302, "Warm 1, 24 rows"), (2, 410, "Warm 2, 24 rows")):
        a = r["summary"][str(k)]["p50_ms"]
        b = m["summary"][str(k)]["p50_ms"]
        p += [txt(58, y + 10, label, 15, weight=550), rect(x0, y - 17, a * scale, 23, REF, 4),
              rect(x0, y + 15, b * scale, 23, MSTAR, 4),
              txt(x0 + a * scale + 9, y, f"{a:.1f} ms", 13, REF, 600),
              txt(x0 + b * scale + 9, y + 32, f"{b:.1f} ms  ({b / a:.2f}x)", 13, MSTAR, 600)]
    p += [txt(565, 528, "milliseconds per call", 14, MUTED, anchor="middle"),
          txt(58, 558, "4 measured sessions after 1 warmup; three calls each. M* f574af8; reference 68f4d7a with pinned dirty harness.", 12, MUTED),
          txt(58, 579, "Bring-up settings: M* -O0; reference compile=None, action CUDA graphs off. This is not production latency.", 12, MUTED)]
    save("droid-session-latency", 1000, 610, p)


def outcomes(report):
    tasks = list(report["per_task"])
    p = [txt(58, 55, "Praxis DROID simulation guard", 28, weight=650),
         txt(58, 86, "Fresh sequential arms on one H200, 7 tasks x 8 scene seeds, sampler seed 0", 15, MUTED),
         rect(58, 108, 14, 14, REF, 2), txt(82, 121, "Reference 55/56", 14, MUTED),
         rect(258, 108, 14, 14, MSTAR, 2), txt(282, 121, "M* 54/56", 14, MUTED),
         txt(945, 121, "successful scenes", 14, MUTED, anchor="end")]
    x0, scale = 300, 535 / 8
    for tick in (0, 2, 4, 6, 8):
        x = x0 + tick * scale
        p += [line(x, 155, x, 681), txt(x, 703, tick, 12, MUTED, anchor="middle")]
    for i, task in enumerate(tasks):
        y = 185 + 72 * i
        row = report["per_task"][task]
        assert row["episodes"] == 8
        a, b = row["reference_successes"], row["mstar_successes"]
        p += [txt(58, y + 12, task.replace("_", " "), 15, weight=550),
              rect(x0, y - 17, a * scale, 21, REF, 4),
              rect(x0, y + 9, b * scale, 21, MSTAR, 4),
              txt(x0 + a * scale + 8, y, f"{a}/8", 13, REF, 600),
              txt(x0 + b * scale + 8, y + 26, f"{b}/8", 13, MSTAR, 600)]
    p += [txt(58, 739, "M* misses lift_cube/6 and block_out_of_bowl/1; reference misses stack_block/0.", 12, MUTED),
          txt(58, 760, "Scene seeds match. The two arms have different VAE implementations; internal parity was not tested.", 12, MUTED)]
    save("droid-simulation-56", 1000, 785, p)


def differences(commands):
    p = [txt(58, 55, "Composed actuator differences", 28, weight=650),
         txt(58, 86, "Identical synthetic observation bytes; 4,485 values across 5 sessions including warmup", 15, MUTED),
         rect(58, 109, 14, 14, REF, 2), txt(82, 122, "95th percentile absolute", 14, MUTED),
         rect(317, 109, 14, 14, MSTAR, 2), txt(341, 122, "maximum absolute", 14, MUTED)]
    x0, scale = 300, 530 / .1
    for tick in (0, .02, .04, .06, .08, .10):
        x = x0 + scale * tick
        p += [line(x, 160, x, 475), txt(x, 498, f"{tick:.2f}", 12, MUTED, anchor="middle")]
    for k, y, label in ((0, 199, "Cold, 21 rows"), (1, 307, "Warm 1, 24 rows"), (2, 415, "Warm 2, 24 rows")):
        row = commands["by_call"][str(k)]
        a, b = row["p95_abs"], row["max_abs"]
        p += [txt(58, y + 7, label, 15, weight=550), rect(x0, y - 17, a * scale, 22, REF, 4),
              rect(x0, y + 15, b * scale, 22, MSTAR, 4),
              txt(x0 + a * scale + 8, y, f"{a:.4f}", 13, REF, 600),
              txt(x0 + b * scale + 8, y + 32, f"{b:.4f}", 13, MSTAR, 600)]
    p += [txt(58, 547, "Overall max 0.0911; p95 0.0542. Reference timestamp metadata was excluded.", 12, MUTED),
          txt(58, 568, "Differences are descriptive. No action-equivalence threshold was set; this is not numerical parity.", 12, MUTED)]
    save("droid-command-differences", 1000, 595, p)


def provenance(report, commands, m, r):
    files = sorted(path for path in SOURCE.rglob("*") if path.is_file())
    data = {
        "mstar_source_commit": "f574af84969a3e5191b5e781cd30e12bd029165f",
        "reference_source_commit": m["reference_client_commit"],
        "reference_checkout": "tracked simulator patch and untracked client, exact status and diff saved in source/",
        "gpu": (SOURCE / "gpu.txt").read_text().strip(),
        "arms": "M* then fresh reference, sequential on one VM",
        "mstar_rust_extension_and_server_source_matched": True,
        "mstar_serve_gate": "enabled only in a disposable source copy; production _SERVABLE remains false",
        "weights": "DROID and Qwen checkpoint files hash-verified; reference .pth VAE and M* Diffusers VAE separately verified, not asserted equivalent",
        "simulation": {
            "scenes": report["units"],
            "mstar_successes": report["total_mstar_successes"],
            "reference_successes": report["total_reference_successes"],
            "discordant": report["discordant"],
        },
        "matched_sessions": {
            "sessions": m["sessions"],
            "warmup_sessions": m["warmup_sessions"],
            "same_observation_bytes": commands["same_observation_bytes"],
            "script_sha256": m["script_sha256"],
            "mstar_p50_ms": {k: v["p50_ms"] for k, v in m["summary"].items()},
            "reference_p50_ms": {k: v["p50_ms"] for k, v in r["summary"].items()},
            "command_values": commands["compared_command_values"],
            "command_max_abs": commands["max_abs"],
            "command_p95_abs": commands["p95_abs"],
        },
        "limits": ["Bring-up settings, not optimized production latency", "different VAE implementations",
                   "no internal numerical parity claim", "5 sessions in one run", "no exact current-head 532e validation"],
        "source_sha256": {str(path.relative_to(HERE)): sha(path) for path in files},
    }
    (HERE / "provenance.json").write_text(json.dumps(data, indent=2, sort_keys=True) + "\n")


def main():
    report, commands, m, r, _, _ = validate()
    latency(m, r)
    outcomes(report)
    differences(commands)
    provenance(report, commands, m, r)


if __name__ == "__main__":
    main()
