#!/usr/bin/env python3
"""Fail closed on task/scene/seed/metadata mismatches in a matched run."""
from __future__ import annotations

import json
import pathlib
import sys

TASKS = (
    "lift_cube", "cube_on_plate", "block_in_bowl", "block_out_of_bowl",
    "move_block", "stack_block", "press_button",
)


def records(root: pathlib.Path) -> dict[tuple[str, int], dict]:
    paths = sorted(root.glob("suite/*/episode_*.json"))
    out = {}
    for path in paths:
        row = json.loads(path.read_text())
        key = row["task"], int(row["seed"])
        if key in out:
            raise SystemExit(f"duplicate episode: {key}")
        out[key] = row
    return out


def main() -> None:
    root = pathlib.Path(sys.argv[1])
    mode = sys.argv[2]
    expected = {(task, i if mode == "original" else int(mode)) for i, task in enumerate(TASKS)}
    reference = records(root / f"reference-{mode}")
    mstar = records(root / f"mstar-{mode}")
    if set(reference) != expected or set(mstar) != expected:
        raise SystemExit(f"unit mismatch: expected={expected}, reference={set(reference)}, M*={set(mstar)}")
    for key in sorted(expected):
        a, b = reference[key], mstar[key]
        if a["prompt"] != b["prompt"] or not a["prompt"]:
            raise SystemExit(f"task prompt mismatch or empty for {key}: {a['prompt']!r}, {b['prompt']!r}")
        if a["sampler_seed"] != 0 or b["sampler_seed"] != 0:
            raise SystemExit(f"sampler seed mismatch for {key}")
        if b.get("sample_seed_base") != 1000 * key[1] or b.get("request_seed_sent") is not True:
            raise SystemExit(f"M* sample seed mismatch for {key}: {b.get('sample_seed_base')}")
        meta = b.get("server_meta") or {}
        if (
            meta.get("attention_backend") != "triton"
            or meta.get("seed_source") != "session_param"
            or meta.get("code_version") != "5902054c0f"
            or meta.get("checkpoint") != "runwayml/praxis-1-droid"
            or meta.get("max_context_chunks") != 1
        ):
            raise SystemExit(f"M* backend/seed/source/recipe mismatch for {key}: {meta}")
        if b.get("restarts") != 0 or b.get("bad_view_ticks") != 0:
            raise SystemExit(f"M* restart/bad view for {key}")
        for label, row in (("reference", a), ("M*", b)):
            rows = row.get("call_rows") or []
            if not rows or rows[0] != 21 or any(n != 24 for n in rows[1:]):
                raise SystemExit(f"{label} row layout mismatch for {key}: {rows}")
    print(json.dumps({
        "mode": mode, "episodes": len(expected),
        "reference_successes": sum(v["success"] for v in reference.values()),
        "mstar_successes": sum(v["success"] for v in mstar.values()),
        "units": [{"task": key[0], "scene_seed": key[1],
                   "reference_success": reference[key]["success"],
                   "mstar_success": mstar[key]["success"]} for key in sorted(expected)],
    }, indent=2))


if __name__ == "__main__":
    main()
