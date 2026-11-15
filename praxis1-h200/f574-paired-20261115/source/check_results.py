#!/usr/bin/env python3
"""Validate exact paired DROID guard units and summarize per-task outcomes."""
from __future__ import annotations

import json
import pathlib
import statistics
import sys

TASKS = (
    "lift_cube", "cube_on_plate", "block_in_bowl", "block_out_of_bowl",
    "move_block", "stack_block", "press_button",
)
SCENES = range(8)


def load(root: pathlib.Path) -> dict[tuple[str, int], dict]:
    out = {}
    for path in sorted(root.glob("suite/*/episode_*.json")):
        row = json.loads(path.read_text())
        key = row["task"], int(row["seed"])
        if key in out:
            raise SystemExit(f"duplicate unit {key} in {root}")
        out[key] = row
    return out


def percentile(values: list[float], p: float) -> float:
    sorted_values = sorted(values)
    if not sorted_values:
        raise SystemExit("no request latency samples")
    at = p * (len(sorted_values) - 1)
    lo = int(at)
    hi = min(lo + 1, len(sorted_values) - 1)
    return sorted_values[lo] + (sorted_values[hi] - sorted_values[lo]) * (at - lo)


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    ref = load(root / "reference")
    mstar = load(root / "mstar")
    expected = {(task, scene) for scene in SCENES for task in TASKS}
    if set(ref) != expected or set(mstar) != expected:
        raise SystemExit(f"expected exact 56 units; reference={len(ref)}, M*={len(mstar)}, missing={expected - set(ref) | expected - set(mstar)}")
    report = {"units": 56, "per_task": {}, "discordant": []}
    for task in TASKS:
        a, b = [], []
        for scene in SCENES:
            key = task, scene
            x, y = ref[key], mstar[key]
            if x["prompt"] != y["prompt"] or not x["prompt"]:
                raise SystemExit(f"prompt mismatch or empty for {key}")
            if x["sampler_seed"] != 0 or y["sampler_seed"] != 0:
                raise SystemExit(f"sampler seed mismatch for {key}")
            if y.get("sample_seed_base") != 1000 * scene or y.get("request_seed_sent") is not True:
                raise SystemExit(f"M* seed-base/query mismatch for {key}")
            meta = y.get("server_meta") or {}
            if any((meta.get("attention_backend") != "triton",
                    meta.get("seed_source") != "session_param",
                    meta.get("code_version") != "f574af84969a3e5191b5e781cd30e12bd029165f",
                    meta.get("checkpoint") != "runwayml/praxis-1-droid",
                    meta.get("max_context_chunks") != 1)):
                raise SystemExit(f"M* ready metadata mismatch for {key}: {meta}")
            for label, row in (("reference", x), ("M*", y)):
                if row.get("restarts") != 0 or row.get("bad_view_ticks") != 0:
                    raise SystemExit(f"{label} restart or bad view for {key}")
                chunks = row.get("call_rows") or []
                calls = row.get("call_s") or []
                if (not chunks or chunks[0] != 21 or any(n != 24 for n in chunks[1:])
                        or len(calls) != len(chunks) or not all(v >= 0 for v in calls)):
                    raise SystemExit(f"{label} row/latency mismatch for {key}")
            a.extend(x["call_s"])
            b.extend(y["call_s"])
            if bool(x["success"]) != bool(y["success"]):
                report["discordant"].append({"task": task, "scene_seed": scene,
                                             "reference_success": bool(x["success"]),
                                             "mstar_success": bool(y["success"])})
        report["per_task"][task] = {
            "episodes": 8,
            "reference_successes": sum(bool(ref[(task, scene)]["success"]) for scene in SCENES),
            "mstar_successes": sum(bool(mstar[(task, scene)]["success"]) for scene in SCENES),
            "reference_calls": len(a), "mstar_calls": len(b),
            "reference_call_p50_ms": 1000 * percentile(a, .5),
            "reference_call_p95_ms": 1000 * percentile(a, .95),
            "mstar_call_p50_ms": 1000 * percentile(b, .5),
            "mstar_call_p95_ms": 1000 * percentile(b, .95),
        }
    report["total_reference_successes"] = sum(v["reference_successes"] for v in report["per_task"].values())
    report["total_mstar_successes"] = sum(v["mstar_successes"] for v in report["per_task"].values())
    (root / "paired-56-report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
