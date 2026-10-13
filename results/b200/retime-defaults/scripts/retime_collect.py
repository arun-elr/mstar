"""Collect the re-timing runs (retime.sh / retime2.sh outputs) into retime.csv."""
import csv
import glob
import json
import os
import statistics
import sys

out_dir, dst = sys.argv[1], sys.argv[2]
rows = []
CHUNK = {"mstar_step0": "M* before (stateless re-decode)", "mstar_warm": "PR defaults (=1, cudnn.benchmark off)",
         "mstar_warm_bench": "opt-in cudnn.benchmark (=1)", "mstar_warm_lpc0": "opt-in =0 (cudnn.benchmark off)"}
for vae in ("wan22", "wan21"):
    for var, label in CHUNK.items():
        runs = [json.load(open(p)) for p in sorted(glob.glob(f"{out_dir}/chunk/run*/{vae}/{var}.json"))]
        if not runs:
            continue
        ms = [r["median_ms"] for r in runs]
        mean = statistics.mean(ms)
        rows.append(dict(vae=vae, bench="chunk (3 latents, 12 frames)", config=label, variant=var,
                         runs=len(ms), run_ms=" / ".join(f"{m:.2f}" for m in ms), mean_ms=round(mean, 2),
                         spread_pct=round(100 * (max(ms) - min(ms)) / mean, 2),
                         peak_allocated_gib=max(r["peak_allocated_gib"] for r in runs),
                         psnr_u8_vs_fp32=runs[0].get("psnr_u8_db_vs_fp32")))
    for tag, label in (("main", "M* before (main 99f5108b)"), ("default", "PR defaults (=1, cudnn.benchmark off)"),
                       ("lpc0", "opt-in =0 (cudnn.benchmark off)")):
        runs = [json.load(open(p)) for p in sorted(glob.glob(f"{out_dir}/nodes/{vae}_{tag}_run*.json"))]
        if not runs:
            continue
        ms = [r["median_ms_per_12_frames"] for r in runs]
        mean = statistics.mean(ms)
        rows.append(dict(vae=vae, bench="served window (8 latents, 6 new)", config=label, variant=tag,
                         runs=len(ms), run_ms=" / ".join(f"{m:.2f}" for m in ms), mean_ms=round(mean, 2),
                         spread_pct=round(100 * (max(ms) - min(ms)) / mean, 2),
                         peak_allocated_gib=max(r["peak_allocated_gib"] for r in runs), psnr_u8_vs_fp32=""))
with open(dst, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)
for r in rows:
    print(r)
