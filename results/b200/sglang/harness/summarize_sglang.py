"""SGLang run dir -> CSVs in the shape of M*'s summarize.py output.

  python summarize_sglang.py RUNS_DIR --flops-dir DIR --out DIR

Reads ``RUNS_DIR/{wan21,wan22}/run*/`` (3 fresh-process runs), ``screen_*/`` (one
run per candidate setting) and ``profile_*/`` (kernel profiles), plus M*'s FLOP
counts (``flops_wan21.json`` / ``flops_wan22.json`` from profile_kernels.py
--flops; the decode math is the same, so are the useful FLOPs). Writes
runs.csv, variance.csv, profile.csv.
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from kernel_categories_sglang import CATEGORIES

PEAK_TFLOPS = 2250.0  # B200 dense bf16
PEAK_TBS = 8.0
PEAK1 = 20 * math.log10(2)


def load(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        if p.name.endswith((".latencies.json", ".kernels.json")):
            continue
        r = json.loads(p.read_text())
        out[r["variant"]] = r
    return out


def r2(x):
    return None if x is None else round(x, 2)


def write_csv(path: Path, rows: list[dict]):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def settings_str(r):
    s = r.get("sglang_settings", {})
    parts = [s.get("path", ""), f"quality={s.get('quality')}", f"decode={s.get('decode_dtype')}"]
    if s.get("enable_torch_compile"):
        parts.append(f"compile({s.get('vae_torch_compile_mode')})")
    return " ".join(parts)


def run_row(vae, group, r):
    return {
        "vae": vae, "group": group, "variant": r["variant"], "settings": settings_str(r),
        "samples": r["n_samples"], "median_ms": r["median_ms"], "p90_ms": r["p90_ms"],
        "decoder_fps": r["decoder_fps"], "peak_allocated_gib": r.get("peak_allocated_gib"),
        "psnr_float_db_peak1": r.get("psnr_float_db_vs_fp32_peak1"),
        "psnr_u8_db": r.get("psnr_u8_db_vs_fp32"),
        "conv_inputs_not_channels_last": r.get("conv_inputs_not_channels_last", ""),
        "first_timed_call_ms": r.get("first_timed_call_ms"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("runs")
    ap.add_argument("--flops-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    runs, out = Path(args.runs), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    rows = []
    for vae in ("wan21", "wan22"):
        for r in load(runs / f"screen_{vae}").values():
            rows.append(run_row(vae, "screen", r))
        for rd in sorted((runs / vae).glob("run*")):
            for r in load(rd).values():
                rows.append(run_row(vae, f"{vae}/{rd.name}", r))
    write_csv(out / "runs.csv", rows)

    var, means = [], {}
    for vae in ("wan21", "wan22"):
        per = [load(p) for p in sorted((runs / vae).glob("run*"))]
        for v in per[0] if per else []:
            rs = [p[v] for p in per if v in p]
            meds = [r["median_ms"] for r in rs]
            mean = statistics.fmean(meds)
            means[(vae, v)] = mean
            var.append({
                "vae": vae, "variant": v, "settings": settings_str(rs[0]),
                **{f"run{i + 1}_ms": m for i, m in enumerate(meds)},
                "mean_ms": r2(mean), "spread_ms": r2(max(meds) - min(meds)),
                "spread_pct": r2((max(meds) - min(meds)) / mean * 100),
                "p90_mean_ms": r2(statistics.fmean(r["p90_ms"] for r in rs)),
                "peak_allocated_gib": rs[0].get("peak_allocated_gib"),
                "psnr_float_db_peak1": rs[0].get("psnr_float_db_vs_fp32_peak1"),
                "psnr_u8_db": rs[0].get("psnr_u8_db_vs_fp32"),
            })
    write_csv(out / "variance.csv", var)

    prof = []
    for vae in ("wan21", "wan22"):
        fl = json.loads((Path(args.flops_dir) / f"flops_{vae}.json").read_text())
        useful = fl["chunk_flops"] / 1e12
        intensity = fl["chunk_flops"] / fl["chunk_min_conv_bytes"]
        roof = min(PEAK_TFLOPS, intensity * PEAK_TBS)
        screen = load(runs / f"screen_{vae}")
        for kp in sorted((runs / f"profile_{vae}").glob("*.kernels.json")):
            k = json.loads(kp.read_text())
            v = k["variant"]
            if (vae, v) in means:
                ms, src = means[(vae, v)], "3-run mean"
            elif v in screen:
                ms, src = screen[v]["median_ms"], "screen"
            else:
                ms, src = None, ""
            cats = k["category_ms_per_chunk"]
            achieved = useful / (ms / 1e3) if ms else None
            prof.append({
                "vae": vae, "variant": v, "kernels_per_chunk": k["kernels_per_chunk"],
                "gpu_ms_per_chunk": k["gpu_ms_per_chunk"], "wall_ms_per_chunk": r2(ms), "wall_source": src,
                **{"ms_" + c.replace(" / ", "_").replace(" ", "_"): cats.get(c, 0.0) for c in CATEGORIES},
                "useful_tflop": round(useful, 2), "conv_tflop": round(fl["chunk_conv_flops"] / 1e12, 2),
                "intensity_flop_per_byte": round(intensity), "roof_tflops": roof,
                "achieved_tflops": round(achieved) if achieved else None,
                "pct_of_roof": round(achieved / roof * 100, 1) if achieved else None,
                "conv_kernel_tflops": round(fl["chunk_conv_flops"] / 1e12 / (cats["convolution"] / 1e3)),
            })
    write_csv(out / "profile.csv", prof)


if __name__ == "__main__":
    main()
