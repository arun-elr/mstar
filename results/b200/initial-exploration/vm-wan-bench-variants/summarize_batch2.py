"""Batch-2 results dir -> markdown tables + chart-friendly CSV.

  python summarize_batch2.py results/b200-batch2 [--base results]

Reads bench JSONs (variance/run*/, px/, tf32/, wan22/), kernel profiles
(profile*/<variant>.kernels.json) and FLOP counts (profile*/flops_*.json). Writes
``summary_batch2.md`` and ``profile_roofline.csv`` into the results dir.
"""

import argparse
import csv
import json
import statistics
from pathlib import Path

from profile_kernels import CATEGORIES

PEAK_TFLOPS = 2250.0  # B200 dense bf16
PEAK_TBS = 8.0        # B200 HBM


def load(d: Path):
    out = {}
    for p in sorted(d.glob("*.json")):
        if p.name.endswith((".latencies.json", ".kernels.json")) or p.name.startswith("flops"):
            continue
        r = json.loads(p.read_text())
        out[r["variant"]] = r
    return out


def fmt(x, nd=1):
    return "" if x is None else f"{x:.{nd}f}"


def variance(run: Path, lines):
    runs = sorted((run / "variance").glob("run*"))
    if not runs:
        return {}
    per = [load(r) for r in runs]
    variants = list(per[0])
    lines += ["## Run-to-run variance (median ms per chunk; fresh process + fresh compile each)", "",
              "| variant | " + " | ".join(r.name for r in runs) + " | mean of medians | spread (max-min) | p90 (mean) |",
              "|---|" + "---:|" * (len(runs) + 3)]
    means = {}
    for v in variants:
        meds = [p[v]["median_ms"] for p in per if v in p]
        p90 = [p[v]["p90_ms"] for p in per if v in p]
        means[v] = statistics.fmean(meds)
        lines.append(f"| {v} | " + " | ".join(f"{m:.2f}" for m in meds)
                     + f" | {means[v]:.2f} | {max(meds) - min(meds):.2f} ({(max(meds) - min(meds)) / means[v] * 100:.1f}%) | {statistics.fmean(p90):.2f} |")
    lines.append("")
    return means


def table(title, rows, lines):
    lines += [f"## {title}", "",
              "| variant | median ms | p90 ms | peak GiB | PSNR float dB (peak 2) | PSNR float dB (peak 1) | PSNR u8 dB | non-CL conv inputs |",
              "|---|---:|---:|---:|---:|---:|---:|---|"]
    for v, r in rows.items():
        lines.append(f"| {v} | {r['median_ms']:.2f} | {r['p90_ms']:.2f} | {fmt(r.get('peak_allocated_gib'))} | "
                     f"{fmt(r.get('psnr_float_db_vs_fp32'), 2)} | {fmt(r.get('psnr_float_db_vs_fp32_peak1'), 2)} | "
                     f"{fmt(r.get('psnr_u8_db_vs_fp32'), 2)} | {r.get('conv_inputs_not_channels_last', '')} |")
    lines.append("")


def profiles(pdir: Path, timings: dict, lines, csv_rows, tag):
    flops_p = next(pdir.glob("flops_*.json"), None)
    if not flops_p:
        return
    fl = json.loads(flops_p.read_text())
    useful = fl["chunk_flops"]
    intensity = useful / fl["chunk_min_conv_bytes"]
    roof = min(PEAK_TFLOPS, intensity * PEAK_TBS)
    lines += [f"## Where the time goes ({tag})", "",
              f"Useful work per 12-frame chunk: {useful / 1e12:.2f} TFLOP (conv {fl['chunk_conv_flops'] / 1e12:.2f}); "
              f"minimum conv traffic {fl['chunk_min_conv_bytes'] / 1e9:.2f} GB → intensity {intensity:.0f} FLOP/B; "
              f"roof min({PEAK_TFLOPS:.0f}, {intensity:.0f}×{PEAK_TBS:.0f}) = {roof:.0f} TFLOP/s "
              f"(compute-bound). M* ctx-8 streaming executes {fl['mstar_ar_ctx8_executed_flops'] / 1e12:.1f} TFLOP per chunk.", "",
              "| variant | kernels/chunk | GPU ms/chunk | " + " | ".join(CATEGORIES) + " | wall ms (bench) | achieved TFLOP/s (useful) | % of roof |",
              "|---|---:|---:|" + "---:|" * len(CATEGORIES) + "---:|---:|---:|"]
    for kp in sorted(pdir.glob("*.kernels.json")):
        k = json.loads(kp.read_text())
        v = k["variant"]
        wall = timings.get(v)
        ms = wall if wall else k["gpu_ms_per_chunk"]
        ach = useful / (ms / 1e3) / 1e12
        cats = k["category_ms_per_chunk"]
        lines.append(f"| {v} | {k['kernels_per_chunk']:.0f} | {k['gpu_ms_per_chunk']:.1f} | "
                     + " | ".join(f"{cats.get(c, 0):.1f}" for c in CATEGORIES)
                     + f" | {fmt(wall)} | {ach:.0f} | {ach / roof * 100:.0f}% |")
        executed = fl.get("mstar_ar_ctx8_executed_flops") if v.startswith("mstar_ar_ctx8") else useful
        csv_rows.append({"vae": tag, "variant": v, "kernels_per_chunk": k["kernels_per_chunk"],
                         "gpu_ms_per_chunk": k["gpu_ms_per_chunk"], "wall_ms_per_chunk": wall,
                         **{f"ms_{c.replace(' / ', '_').replace(' ', '_')}": cats.get(c, 0) for c in CATEGORIES},
                         "useful_tflop": useful / 1e12, "executed_tflop": executed / 1e12,
                         "intensity_flop_per_byte": intensity, "achieved_tflops_useful": ach,
                         "achieved_tflops_executed": executed / (ms / 1e3) / 1e12, "roof_tflops": roof})
    lines.append("")
    for kp in sorted(pdir.glob("*.kernels.json")):
        k = json.loads(kp.read_text())
        lines += [f"<details><summary>Top-15 kernels: {k['variant']}</summary>", "",
                  "| kernel | category | calls/chunk | ms/chunk |", "|---|---|---:|---:|"]
        for t in k["top15"]:
            lines.append(f"| `{t['name'][:110]}` | {t['category']} | {t['calls_per_chunk']} | {t['ms_per_chunk']:.2f} |")
        lines += ["", "</details>", ""]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    args = ap.parse_args()
    run = Path(args.run)
    lines = [f"# Batch 2 results: {run.name}", ""]
    means = variance(run, lines)
    for sub, title in (("tf32", "TF32 check (fp32 reference)"), ("px", "Beyond the reference ladder (Wan 2.1)"),
                       ("wan22", "Best prototype on the Wan2.2 VAE")):
        rows = load(run / sub) if (run / sub).is_dir() else {}
        if rows:
            table(title, rows, lines)
    timings = dict(means)
    for sub in ("px",):
        for v, r in (load(run / sub) if (run / sub).is_dir() else {}).items():
            timings.setdefault(v, r["median_ms"])
    csv_rows = []
    profiles(run / "profile", timings, lines, csv_rows, "wan21")
    t22 = {v: r["median_ms"] for v, r in (load(run / "wan22") if (run / "wan22").is_dir() else {}).items()}
    profiles(run / "profile_wan22", t22, lines, csv_rows, "wan22")
    (run / "summary_batch2.md").write_text("\n".join(lines) + "\n")
    if csv_rows:
        with open(run / "profile_roofline.csv", "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(csv_rows[0]))
            w.writeheader()
            w.writerows(csv_rows)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
