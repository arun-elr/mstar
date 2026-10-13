"""Summarize the e2e runs: timings per phase, the server's per-node split,
peak memory, and before-vs-after frame equivalence from the raw dumps.

  python analyze.py ~/e2e-cosmos3/runs out_dir
"""
import csv
import datetime as dt
import glob
import hashlib
import json
import math
import os
import re
import statistics
import sys
from pathlib import Path

import numpy as np

RUNS, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
VARIANTS = [v for v in ("before", "after0", "after1") if (RUNS / v).is_dir()]
med = statistics.median


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def phases(v):
    out = {}
    for line in open(RUNS / v / "phases.jsonl"):
        r = json.loads(line)
        out.setdefault(r["phase"], {})[r["event"]] = r
    return out


ROW = re.compile(r"^\s{5}(\S+)\s+n=(\d+)\s+([\d.]+)\s+\(\s*([\d.]+)\)")
NODE = re.compile(r"^\s{3}(\S+)\s*$")
TOTAL = re.compile(r"^\s{3}total\s+([\d.]+) ms")


def profiles(v, ph):
    p = phases(v)[ph]
    raw = open(RUNS / v / "stats.txt", "rb").read()[p["start"]["stats_bytes"]: p["end"]["stats_bytes"]].decode()
    profs = []
    for block in raw.split(" Request profile: ")[1:]:
        rows, node, total = {}, None, None
        in_graph = False
        for line in block.splitlines():
            if line.startswith(" Graph timings"):
                in_graph = True
                continue
            if in_graph and line.startswith("---") or line.startswith("==="):
                in_graph = False
            m = TOTAL.match(line)
            if m and total is None:
                total = float(m.group(1))
            if not in_graph:
                continue
            m = NODE.match(line)
            if m:
                node = m.group(1)
                continue
            m = ROW.match(line)
            if m and node:
                rows[f"{node}/{m.group(1)}"] = {"n": int(m.group(2)), "total_ms": float(m.group(3)),
                                                 "avg_ms": float(m.group(4))}
        profs.append({"rid": block.split("\n", 1)[0].strip(), "timeline_total_ms": total, "graph": rows})
    return profs


def smi_peak(v, t0, t1):
    best = 0
    for row in csv.reader(open(RUNS / v / "smi.csv")):
        try:
            ts = dt.datetime.strptime(row[0].strip(), "%Y/%m/%d %H:%M:%S.%f").timestamp()
        except ValueError:
            continue
        if t0 <= ts <= t1:
            best = max(best, int(row[1]))
    return best


def mem(v, ph):
    p = RUNS / v / f"mem_{ph}.json"
    recs = json.load(open(p)) if p.exists() else []
    return {r["pid"]: {k: r[k] for k in ("max_allocated_gib", "max_reserved_gib", "allocated_gib", "resets")}
            | {"argv": r["argv"][:120]} for r in recs}


def phase_summary(v, ph):
    c = json.load(open(RUNS / v / f"client_{ph}.json"))
    conc = c["args"]["concurrency"]
    timed = [r for r in c["rounds"] if r["timed"]]
    reqs = [q for r in timed for q in r["requests"]]
    errs = [q["error"] for r in c["rounds"] for q in r["requests"] if q["error"]]
    ok = [q for q in reqs if not q["error"]]
    res = {"variant": v, "phase": ph, "mode": c["args"]["mode"], "concurrency": conc, "rounds": len(timed),
           "requests_ok": len(ok), "errors": errs}
    if ok:
        gaps = [b - a for q in ok for a, b in zip(q["arrivals"], q["arrivals"][1:])]
        res |= {
            "windows": med([len(q["arrivals"]) for q in ok]),
            "frames": med([sum(q["frames"]) for q in ok]),
            "ttff_s": med([q["arrivals"][0] for q in ok]),
            "window_gap_med_s": med(gaps), "window_gap_p90_s": pct(gaps, 0.9),
            "total_s": med([q["arrivals"][-1] for q in ok]),
            "fps_per_stream": med([sum(q["frames"]) / q["arrivals"][-1] for q in ok]),
            "round_wall_s": med([r["wall_s"] for r in timed]),
            "fps_aggregate": med([sum(sum(q["frames"]) for q in r["requests"] if not q["error"]) / r["wall_s"]
                                  for r in timed]),
            "ttff_all_s": [round(q["arrivals"][0], 3) for q in ok],
            "total_all_s": [round(q["arrivals"][-1], 3) for q in ok],
        }
    # server-side split: the timed requests are the last finished ones
    profs = profiles(v, ph)[-len(reqs):] if reqs else []
    if profs:
        keys = sorted({k for p in profs for k in p["graph"]})
        res["graph"] = {k: {"n": med([p["graph"][k]["n"] for p in profs if k in p["graph"]]),
                            "total_ms": med([p["graph"][k]["total_ms"] for p in profs if k in p["graph"]]),
                            "avg_ms": med([p["graph"][k]["avg_ms"] for p in profs if k in p["graph"]])}
                        for k in keys}
        dec = [k for k in keys if k.startswith("vae_decoder_ar/")]
        dit = [k for k in keys if k.startswith("dit/video_gen")]
        if dit and ok:
            res["dit_ms_per_window"] = res["graph"][dit[0]]["total_ms"] / res["windows"]
        if dec and ok:
            dk = dec[0]
            res["decoder_node"] = dk
            # the node runs once more than there are windows (a last no-op step)
            res["decoder_ms_per_window"] = res["graph"][dk]["total_ms"] / res["windows"]
            res["decoder_ms_per_request"] = res["graph"][dk]["total_ms"]
            res["decoder_share_of_request_wall"] = res["graph"][dk]["total_ms"] / 1e3 / res["total_s"]
            if conc > 1:
                res["decoder_share_of_round_wall"] = (res["graph"][dk]["total_ms"] * conc / 1e3
                                                      / res["round_wall_s"])
    p = phases(v)[ph]
    m = mem(v, ph)
    res["mem_by_pid"] = m
    res["torch_max_allocated_gib"] = max((x["max_allocated_gib"] for x in m.values()), default=None)
    res["torch_max_reserved_gib"] = max((x["max_reserved_gib"] for x in m.values()), default=None)
    gpu = [x for x in m.values() if x["max_allocated_gib"] > 0]
    if gpu:
        w = max(gpu, key=lambda x: x["max_allocated_gib"])
        res["torch_idle_allocated_gib"] = w["allocated_gib"]
        res["torch_peak_above_idle_gib"] = w["max_allocated_gib"] - w["allocated_gib"]
    res["smi_peak_mib"] = smi_peak(v, p["start"]["t"], p["end"]["t"] + 1.0)
    return res


def load_dump(v, mode):
    files = sorted(glob.glob(str(RUNS / v / "dump" / mode / "*.npy")))
    chunks = [np.load(f) for f in files]
    return chunks


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def seams(x, bounds):
    """Consecutive-frame mean |diff| at window boundaries vs inside windows."""
    f = x.astype(np.float32)
    d = np.abs(f[:, 1:] - f[:, :-1]).mean(axis=(0, 2, 3))  # d[t] = |frame t+1 - frame t|
    at = [b - 1 for b in bounds]
    inside = [t for t in range(len(d)) if t not in at]
    ratios = []
    for t in at:
        nb = [d[u] for u in range(max(0, t - 4), min(len(d), t + 5)) if u not in at]
        ratios.append(float(d[t] / np.median(nb)))
    return {"boundary_mad": [round(float(d[t]), 3) for t in at], "inside_mad_median": float(np.median(d[inside])),
            "boundary_over_neighbors": [round(r, 3) for r in ratios], "max_ratio": max(ratios) if ratios else None}


def equivalence(mode):
    ref = load_dump("before", mode)
    if not ref:
        return None
    rx = np.concatenate(ref, axis=1)
    bounds = list(np.cumsum([c.shape[1] for c in ref])[:-1])
    out = {"mode": mode, "chunk_frames": [c.shape[1] for c in ref], "shape_CTHW": list(rx.shape),
           "md5": {"before": hashlib.md5(rx.tobytes()).hexdigest()}, "seams": {"before": seams(rx, bounds)},
           "vs_before": {}}
    for v in VARIANTS[1:]:
        ch = load_dump(v, mode)
        if not ch:
            continue
        x = np.concatenate(ch, axis=1)
        out["md5"][v] = hashlib.md5(x.tobytes()).hexdigest()
        out["seams"][v] = seams(x, list(np.cumsum([c.shape[1] for c in ch])[:-1]))
        if x.shape != rx.shape:
            out["vs_before"][v] = {"shape_mismatch": [list(x.shape), list(rx.shape)]}
            continue
        diff = np.abs(x.astype(np.int16) - rx.astype(np.int16))
        per = [psnr(x[:, t], rx[:, t]) for t in range(x.shape[1])]
        finite = [p for p in per if math.isfinite(p)]
        worst = int(np.argmin([p if math.isfinite(p) else 1e9 for p in per]))
        out["vs_before"][v] = {
            "identical": bool((diff == 0).all()), "differing_values": int((diff > 0).sum()),
            "differing_fraction": float((diff > 0).mean()), "max_abs_diff": int(diff.max()),
            "values_diff_gt1": int((diff > 1).sum()),
            "frames_identical": sum(1 for p in per if not math.isfinite(p)), "frames": len(per),
            "psnr_whole_db": psnr(x, rx),
            "psnr_frame_min_db": min(finite) if finite else float("inf"), "worst_frame": worst,
            "psnr_frame_median_db": float(np.median(finite)) if finite else float("inf"),
            "psnr_at_boundary_frames_db": [round(per[b], 2) if math.isfinite(per[b]) else "inf" for b in bounds],
            "per_window_max_abs_diff": [int(diff[:, a:b].max()) for a, b in
                                        zip([0, *bounds], [*bounds, x.shape[1]])],
            "per_frame_psnr_db": [round(p, 2) if math.isfinite(p) else "inf" for p in per],
        }
    return out


def mp4_md5s():
    out = {}
    for v in VARIANTS:
        for d in sorted((RUNS / v / "mp4").glob("*")):
            groups = {}
            for f in sorted(d.glob("*.mp4")):
                key = f.name.rsplit("_", 1)[0]
                groups.setdefault(key, hashlib.md5())
                groups[key].update(f.read_bytes())
            out[f"{v}/{d.name}"] = {k: h.hexdigest() for k, h in groups.items()}
    return out


rows = [phase_summary(v, ph) for v in VARIANTS for ph in ("kv_c1", "kv_c4", "ch_c1", "kv_c8")
        if (RUNS / v / f"client_{ph}.json").exists()]
startup = {v: {"startup_s": (lambda s: s["ready"] - s["launch"])(json.load(open(RUNS / v / "startup.json"))),
               "mem_after_startup": mem(v, "startup")} for v in VARIANTS if (RUNS / v / "startup.json").exists()}
eq = {m: equivalence(m) for m in ("kv", "chained")}
res = {"phases": rows, "startup": startup, "equivalence": eq, "mp4_md5": mp4_md5s()}
(OUT / "results.json").write_text(json.dumps(res, indent=1, default=str))
cols = ["variant", "phase", "mode", "concurrency", "rounds", "requests_ok", "windows", "frames", "ttff_s",
        "window_gap_med_s", "window_gap_p90_s", "total_s", "fps_per_stream", "fps_aggregate", "round_wall_s",
        "dit_ms_per_window", "decoder_ms_per_window", "decoder_ms_per_request", "decoder_share_of_request_wall",
        "decoder_share_of_round_wall", "torch_max_allocated_gib", "torch_idle_allocated_gib",
        "torch_peak_above_idle_gib", "torch_max_reserved_gib", "smi_peak_mib"]
with open(OUT / "results.csv", "w", newline="") as f:
    w = csv.DictWriter(f, cols, extrasaction="ignore")
    w.writeheader()
    for r in rows:
        w.writerow({k: (round(v, 4) if isinstance(v, float) else v) for k, v in r.items()})
for r in rows:
    print({k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items()
           if k not in ("mem_by_pid", "graph", "ttff_all_s", "total_all_s")})
    for k, g in (r.get("graph") or {}).items():
        print(f"    {k:<40} n={g['n']} total={g['total_ms']:.1f} ms avg={g['avg_ms']:.1f} ms")
for m, e in eq.items():
    if e:
        print(m, json.dumps({k: v for k, v in e.items() if k != "vs_before"}, default=str)[:1500])
        for v, d in e["vs_before"].items():
            print("  ", v, {k: x for k, x in d.items() if k != "per_frame_psnr_db"})
