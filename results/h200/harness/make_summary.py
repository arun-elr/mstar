"""H200 run dir -> runs.csv, variance.csv, profile.csv, comparison.csv (B200 vs H200)
in the B200 results' schemas. Prints markdown tables for SUMMARY.md."""

import csv
import json
import math
import statistics
import sys
from pathlib import Path

RUNS = Path(sys.argv[1])
OUT = Path(sys.argv[2])
B200 = Path(sys.argv[3])  # PR's results/b200
B200_RUNS = Path(sys.argv[4])  # PR's runs/b200 (flops jsons)
sys.path.insert(0, str(Path(__file__).parent / "src/benchmark/wan_vae_decoder"))
from kernel_categories import CATEGORIES  # noqa: E402

PEAK_TFLOPS = 989.0  # H200 SXM dense bf16
PEAK_TBS = 4.8  # H200 HBM3e
PEAK1 = 20 * math.log10(2)


def r2(x):
    return None if x is None else round(x, 2)


def load(d):
    out = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        if p.name.endswith((".latencies.json", ".kernels.json")):
            continue
        r = json.loads(p.read_text())
        out[r["variant"]] = r
    return out


def write_csv(path, rows):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)", file=sys.stderr)


def run_row(vae, group, r):
    p2 = r.get("psnr_float_db_vs_fp32")
    p2 = p2 if p2 is None or math.isfinite(p2) else None
    return {
        "vae": vae, "group": group, "variant": r["variant"], "samples": r["n_samples"],
        "median_ms": r["median_ms"], "p90_ms": r["p90_ms"], "decoder_fps": r["decoder_fps"],
        "peak_allocated_gib": r.get("peak_allocated_gib"),
        "psnr_float_db_peak2": p2, "psnr_float_db_peak1": r2(p2 - PEAK1) if p2 is not None else None,
        "psnr_u8_db": r.get("psnr_u8_db_vs_fp32"),
        "conv_inputs_not_channels_last": r.get("conv_inputs_not_channels_last", ""),
        "decode_latents_per_chunk": r.get("steady_decode_latents", ""),
        "first_timed_call_ms": r.get("first_timed_call_ms"), "wall_s": r.get("wall_s"),
        "gpu": r.get("env", {}).get("gpu", ""),
    }


def three_runs(d, vae, session):
    rows = []
    per = [load(p) for p in sorted(d.glob("run*"))]
    for v in per[0] if per else []:
        rs = [p[v] for p in per if v in p]
        meds = [r["median_ms"] for r in rs]
        mean = statistics.fmean(meds)
        p2 = rs[0].get("psnr_float_db_vs_fp32")
        rows.append({
            "session": session, "vae": vae, "variant": v, **{f"run{i + 1}_ms": m for i, m in enumerate(meds)},
            "mean_ms": r2(mean), "spread_ms": r2(max(meds) - min(meds)),
            "spread_pct": r2((max(meds) - min(meds)) / mean * 100),
            "p90_mean_ms": r2(statistics.fmean(r["p90_ms"] for r in rs)),
            "peak_allocated_gib": rs[0].get("peak_allocated_gib"),
            "psnr_float_db_peak1": r2(p2 - PEAK1) if p2 is not None else None,
            "psnr_u8_db": rs[0].get("psnr_u8_db_vs_fp32"),
        })
    return rows


runs = []
for vae, sub in (("wan21", "intree"), ("wan22", "intree_wan22")):
    for rd in sorted((RUNS / sub).glob("run*")):
        runs += [run_row(vae, f"{sub}/{rd.name}", r) for r in load(rd).values()]
for vae, sub in (("wan21", "ladder"), ("wan22", "ladder_wan22"), ("wan21", "refcheck_wan21"), ("wan22", "refcheck_wan22"), ("wan21", "drift")):
    runs += [run_row(vae, sub, r) for r in load(RUNS / sub).values()]
write_csv(OUT / "runs.csv", runs)

var = three_runs(RUNS / "intree", "wan21", "in-tree") + three_runs(RUNS / "intree_wan22", "wan22", "in-tree")
write_csv(OUT / "variance.csv", var)
mean = {(r["vae"], r["variant"]): r["mean_ms"] for r in var}
single = {(r["vae"], r["variant"]): r["median_ms"] for r in runs if r["group"] in ("ladder", "ladder_wan22")}


def h200_ms(vae, v):
    if (vae, v) in mean:
        return mean[(vae, v)], "3-run mean"
    if (vae, v) in single:
        return single[(vae, v)], "single run"
    return None, ""


# ---------------------------------------------------------------- profile.csv
prof = []
for vae, pdir, fdir in (("wan21", RUNS / "profile", B200_RUNS / "ablation/profile"),
                        ("wan22", RUNS / "profile_wan22", B200_RUNS / "ablation/profile_wan22")):
    fl = json.loads(next(fdir.glob("flops_*.json")).read_text())
    useful = fl["chunk_flops"] / 1e12
    intensity = fl["chunk_flops"] / fl["chunk_min_conv_bytes"]
    roof = min(PEAK_TFLOPS, intensity * PEAK_TBS)
    for kp in sorted(pdir.glob("*.kernels.json")):
        k = json.loads(kp.read_text())
        v = k["variant"]
        ms, src = h200_ms(vae, v)
        executed = fl["mstar_ar_ctx8_executed_flops"] / 1e12 if v.startswith("mstar_ar_ctx8") else useful
        cats = k["category_ms_per_chunk"]
        ach = useful / (ms / 1e3) if ms else None
        prof.append({
            "vae": vae, "variant": v, "kernels_per_chunk": k["kernels_per_chunk"],
            "gpu_ms_per_chunk": k["gpu_ms_per_chunk"], "wall_ms_per_chunk": r2(ms), "wall_source": src,
            **{"ms_" + c.replace(" / ", "_").replace(" ", "_"): cats.get(c, 0.0) for c in CATEGORIES},
            "useful_tflop": round(useful, 2), "executed_tflop": round(executed, 2),
            "conv_tflop": round(fl["chunk_conv_flops"] / 1e12, 2),
            "intensity_flop_per_byte": round(intensity), "roof_tflops": roof,
            "achieved_tflops": round(ach) if ach else None,
            "pct_of_roof": round(ach / roof * 100, 1) if ach else None,
            "conv_kernel_tflops": round(fl["chunk_conv_flops"] / 1e12 * executed / useful / (cats["convolution"] / 1e3)),
            "conv_kernel_pct_of_peak": round(fl["chunk_conv_flops"] / 1e12 * executed / useful
                                             / (cats["convolution"] / 1e3) / PEAK_TFLOPS * 100, 1),
            "top_conv_kernel": next((t["name"] for t in k["top15"] if t["category"] == "convolution"), ""),
        })
write_csv(OUT / "profile.csv", prof)

# ---------------------------------------------------------------- B200 vs H200
b_var = {(r["vae"], r["variant"]): float(r["mean_ms"]) for r in csv.DictReader(open(B200 / "variance.csv"))}
b_in = {}
for r in csv.DictReader(open(B200 / "intree.csv")):
    key = (r["vae"], r["variant"])
    if r["session"] == "in-tree":
        b_in[key] = float(r["mean_ms"])
    elif r["variant"] == "mstar_warm_nobench":
        b_in[key] = float(r["mean_ms"])
b_runs = {(r["vae"], r["variant"]): float(r["median_ms"]) for r in csv.DictReader(open(B200 / "runs.csv"))
          if r["group"] == "ablation/px"}

LADDER = [
    ("wan21", "mstar_ar_ctx8", "M* before (re-decode 8 context latents, compiled)"),
    ("wan21", "diffusers_bf16", "stock diffusers, eager"),
    ("wan21", "diffusers_bf16_compile", "1. keep conv cache (+ compile)"),
    ("wan21", "channels_last_compile", "2. + channels-last"),
    ("wan21", "channels_last_one_call_compile", "3. + one decoder call per chunk"),
    ("wan21", "px_nopad", "4. + H/W padding inside the conv"),
    ("wan21", "px_nopad_bench", "5. + cudnn.benchmark (prototype best)"),
    ("wan21", "mstar_warm", "in-tree node, this change's defaults"),
    ("wan21", "mstar_warm_eager", "in-tree node, eager"),
    ("wan21", "mstar_warm_nobench", "in-tree node, cudnn.benchmark off"),
    ("wan22", "mstar_ar_ctx8", "M* before (re-decode 8 context latents, compiled)"),
    ("wan22", "px_nopad_bench", "prototype best"),
    ("wan22", "mstar_warm", "in-tree node, this change's defaults"),
    ("wan22", "mstar_warm_eager", "in-tree node, eager"),
    ("wan22", "mstar_warm_nobench", "in-tree node, cudnn.benchmark off"),
]


def b200_ms(vae, v):
    if (vae, v) in b_in:
        return b_in[(vae, v)], "in-tree session 3-run mean" if v != "mstar_warm_nobench" else "control 3-run mean"
    if (vae, v) in b_var:
        return b_var[(vae, v)], "variance 3-run mean"
    if (vae, v) in b_runs:
        return b_runs[(vae, v)], "single run (ablation/px)"
    return None, ""


cmp_rows = []
for vae, v, label in LADDER:
    h, hs = h200_ms(vae, v)
    b, bs = b200_ms(vae, v)
    h0, b0 = h200_ms(vae, "mstar_ar_ctx8")[0], b200_ms(vae, "mstar_ar_ctx8")[0]
    cmp_rows.append({
        "vae": vae, "variant": v, "change": label, "h200_ms": h, "h200_source": hs,
        "h200_speedup_vs_mstar_before": r2(h0 / h) if h else None,
        "b200_ms": b, "b200_source": bs,
        "b200_speedup_vs_mstar_before": r2(b0 / b) if b else None,
        "h200_over_b200": r2(h / b) if h and b else None,
        "h200_fps": round(12 / (h / 1e3), 1) if h else None,
    })
write_csv(OUT / "comparison_b200.csv", cmp_rows)

print("\n## comparison")
print("| VAE | variant | change | H200 ms | H200 × | B200 ms | B200 × | H200 / B200 |")
print("|---|---|---|---:|---:|---:|---:|---:|")
for r in cmp_rows:
    star = "" if r["h200_source"] == "3-run mean" else "\\*"
    bstar = "" if "3-run" in r["b200_source"] else "\\*"
    b = f"{r['b200_ms']:.2f}{bstar} | {r['b200_speedup_vs_mstar_before']:.1f}× | {r['h200_over_b200']:.2f}" \
        if r["b200_ms"] else "— | — | —"
    print(f"| {'Wan 2.1' if r['vae'] == 'wan21' else 'Wan 2.2'} | `{r['variant']}` | {r['change']} | "
          f"{r['h200_ms']:.2f}{star} | {r['h200_speedup_vs_mstar_before']:.1f}× | {b} |")

print("\n## variance")
print("| VAE | variant | run 1 | run 2 | run 3 | mean | spread | p90 mean | peak GiB | PSNR u8 dB |")
print("|---|---|---:|---:|---:|---:|---:|---:|---:|---:|")
for r in var:
    print(f"| {r['vae']} | `{r['variant']}` | {r['run1_ms']} | {r['run2_ms']} | {r['run3_ms']} | {r['mean_ms']} | "
          f"{r['spread_ms']} ({r['spread_pct']}%) | {r['p90_mean_ms']} | {r['peak_allocated_gib']} | {r['psnr_u8_db']} |")

print("\n## single runs")
for r in runs:
    if not r["group"].startswith("intree"):
        print(r)

print("\n## profile")
for r in prof:
    print(r)

wall = sum(r["wall_s"] or 0 for r in runs)
print(f"\nsum of bench wall_s: {wall:.0f} s")
