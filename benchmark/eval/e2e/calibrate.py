"""Cross-process noise floors from A/A (base == head) ``compare`` runs.

The passes inside one server session underestimate the spread between two server processes (kernel selection and
batching differ per process). Every A/A run records, per row, metric and GPU type, how far two processes of the
same code drifted; the report then requires a delta to clear ``report.FLOOR_FACTOR`` times the largest drift before
it calls the row faster or a regression. The same runs record how far each row's correctness number moved (WER
points, identical-text share, audio length, PSNR ...), and the report widens a manifest limit that this noise alone
would trip (``report.correct_limit``).

``calibration.json`` next to this file is the shared baseline; ``--calibration PATH`` keeps a private one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from benchmark.eval.e2e import report
from benchmark.eval.e2e.manifest import Model

DEFAULT = Path(__file__).with_name("calibration.json")


def load(path: str | Path = DEFAULT) -> dict:
    path = Path(path)
    return json.loads(path.read_text()) if path.exists() else {}


def floor_pct(cal: dict, key: str) -> float | None:
    e = cal.get(key)
    if not e or not e.get("deltas"):
        return None
    return report.FLOOR_FACTOR * max(abs(d) for d in e["deltas"])


def add_run(run_dir: Path, models: list[Model], path: str | Path = DEFAULT) -> int:
    """Fold one A/A run into the calibration; returns the number of entries recorded (0 when already folded in).

    Perf drifts (``_runs``) and correctness swings (``_runs_correct``) are tracked apart, so a run folded in before
    correctness swings were recorded can still add those."""
    path = Path(path)
    run = json.loads((run_dir / "run.json").read_text())
    if run["base"]["sha"] != run["head"]["sha"]:
        raise ValueError(f"{run_dir.name} is not an A/A run (base and head are different commits)")
    gpu = run.get("gpu", "unknown")
    cal = load(path)
    seen, seen_c = set(cal.get("_runs", [])), set(cal.get("_runs_correct", []))
    if run["id"] in seen and run["id"] in seen_c:
        return 0
    n = 0
    if run["id"] not in seen:
        for r in report.build_rows(run_dir, models, calibration={}):
            for p in [r.latency, r.throughput, *r.extra]:
                if p and p.delta_pct is not None:
                    key = report.calibration_key(gpu, r.model, r.case, p.metric, p.conc)
                    cal.setdefault(key, {"deltas": []})["deltas"].append(round(p.delta_pct, 3))
                    n += 1
    if run["id"] not in seen_c:
        for m in models:
            for name in (m.name, m.name + report.RERUN_TAG):
                f = run_dir / "results" / name / "compare.json"
                compare = json.loads(f.read_text()) if f.exists() else {}
                for c in m.cases:
                    swings = [s for s in (report.aa_swing(c.check["kind"], x) for x in compare.get(c.id, []))
                              if s is not None and math.isfinite(s)]
                    if swings:
                        key = report.correct_key(gpu, m.name, c.id)
                        entry = cal.setdefault(key, {"kind": c.check["kind"], "swings": []})
                        entry["swings"] += [round(s, 4) for s in swings]
                        n += 1
    cal["_runs"] = sorted(seen | {run["id"]})
    cal["_runs_correct"] = sorted(seen_c | {run["id"]})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cal, indent=1, sort_keys=True) + "\n")
    return n
