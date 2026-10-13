"""Raw run dir -> small summary CSVs (the files committed under results/).

  python summarize.py runs/<run> --out results/<name>

Reads ``run_all.sh`` output (``<run>/wan21/``, ``<run>/wan22/``) and, if present,
``run_ablation.sh`` output (``<run>/ablation/{variance/run*,intree/run*,intree_wan22/run*,
px,tf32,wan22,nodes,profile,profile_wan22}``).
Writes:
  runs.csv      one row per single bench run (median / p90 / peak memory / PSNR)
  variance.csv  per-variant medians of the independent runs, mean and spread
  intree.csv    the same for the in-tree session (in-tree node and anchors, both VAEs)
                and the cudnn.benchmark-off control of the in-tree node
  nodes.csv     M*'s decoder node on served windows (bench_decode_nodes.py)
  ladder_wan21.csv, ladder_wan22.csv  the change ladder through M*'s node (run_ablation.sh ladder):
                3-run timing, kernel profile, FLOPs, peak memory, PSNR and frame-stream md5 per step
  profile.csv   kernel-time categories per chunk, FLOPs and achieved TFLOP/s
"""

import argparse
import csv
import json
import math
import statistics
from pathlib import Path

from kernel_categories import CATEGORIES

PEAK_TFLOPS = 2250.0  # B200 dense bf16
PEAK_TBS = 8.0  # B200 HBM
PEAK1 = 20 * math.log10(2)
FRAMES_PER_CHUNK = 12  # PSNR on [-1, 1] with peak 1 instead of the range 2


def load(d: Path) -> dict:
    out = {}
    for p in sorted(d.glob("*.json")) if d.is_dir() else []:
        if p.name.endswith((".latencies.json", ".kernels.json")) or p.name.startswith("flops"):
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


def run_row(vae: str, group: str, r: dict) -> dict:
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
        "gpu": r.get("env", {}).get("gpu", ""),
    }


def three_runs(d: Path, vae: str) -> list[dict]:
    """Per-variant medians of the independent runs under ``d/run*``, mean and spread."""
    rows = []
    per = [load(p) for p in sorted(d.glob("run*"))]
    for v in per[0] if per else []:
        rs = [p[v] for p in per if v in p]
        meds = [r["median_ms"] for r in rs]
        mean = statistics.fmean(meds)
        p2 = rs[0].get("psnr_float_db_vs_fp32")
        rows.append({
            "vae": vae, "variant": v, **{f"run{i + 1}_ms": m for i, m in enumerate(meds)},
            "mean_ms": r2(mean), "spread_ms": r2(max(meds) - min(meds)),
            "spread_pct": r2((max(meds) - min(meds)) / mean * 100),
            "p90_mean_ms": r2(statistics.fmean(r["p90_ms"] for r in rs)),
            "peak_allocated_gib": rs[0].get("peak_allocated_gib"),
            "psnr_float_db_peak1": r2(p2 - PEAK1) if p2 is not None else None,
            "psnr_u8_db": rs[0].get("psnr_u8_db_vs_fp32"),
        })
    return rows


def load_nodes(d: Path) -> list[dict]:
    return [json.loads(p.read_text()) for p in sorted(d.glob("*.json"))] if d.is_dir() else []


def node_row(r: dict) -> dict:
    """One ``bench_decode_nodes.py`` run (served windows through M*'s decoder node)."""
    return {
        "tag": r["tag"], "node": r["node"], "vae": r["vae"], "size": r["size"],
        "window_latents": r.get("window_latents"), "overlap": r.get("overlap"),
        "latents_per_call": r["overrides"].get("windowed_decode_latents_per_call", "default"),
        "median_ms_per_window": r2(r.get("median_ms_per_window")),
        "p90_ms_per_window": r2(r.get("p90_ms_per_window")),
        "median_ms_per_12_frames": r2(r.get("median_ms_per_12_frames")),
        "peak_allocated_gib": r2(r.get("peak_allocated_gib")),
        "cache_mib_per_stream": r2(r.get("cache_mib_per_stream")),
        "psnr_u8_vs_fp32": r2(r.get("psnr_u8_vs_fp32")),
        "gpu": r.get("env", {}).get("gpu", ""),
    }


LADDER = [  # (step, change) in order; each step adds its change to the one before
    (0, "before: stateless context re-decode (8 latents, empty cache)"),
    (1, "+ warm conv cache across chunks"),
    (2, "+ channels-last weights and activations, conv-internal padding"),
    (3, "+ one decoder call per chunk"),
    (5, "+ cudnn.benchmark"),
]


def read_rows(path: Path) -> list[dict]:
    """CSV rows; repeated header lines (appended part files) are skipped."""
    if not path.exists():
        return []
    with open(path) as f:
        rows = list(csv.DictReader(f))
    return [r for r in rows if r.get("vae") != "vae"]


def ladder_rows(abl: Path, vae: str) -> list[dict]:
    sfx = "" if vae == "wan21" else "_wan22"
    per = {r["variant"]: r for r in three_runs(abl / f"ladder{sfx}", vae)}
    if not per:
        return []
    fl_dir = abl / ("profile" if vae == "wan21" else "profile_wan22")
    fl = json.loads(next(fl_dir.glob("flops_*.json")).read_text())
    useful = fl["chunk_flops"] / 1e12
    frames = {r["variant"]: r for r in read_rows(abl / f"ladder_frames_{vae}.csv")}
    rows = []
    for mode, suffix in (("compiled", ""), ("eager", "_eager")):
        prev_ms = base_ms = None
        for step, change in LADDER:
            v = f"mstar_step{step}{suffix}"
            r = per.get(v)
            if r is None:
                continue
            ms = r["mean_ms"]
            base_ms = base_ms or ms
            kp = abl / f"profile_ladder{sfx}" / f"{v}.kernels.json"
            k = json.loads(kp.read_text()) if kp.exists() else {}
            fr = frames.get(v, {})
            executed = fl["mstar_ar_ctx8_executed_flops"] / 1e12 if step == 0 else useful
            rows.append({
                "vae": vae, "step": step, "change": change, "mode": mode, "variant": v,
                **{f"run{i}_ms": r.get(f"run{i}_ms") for i in (1, 2, 3)},
                "mean_ms": ms, "spread_ms": r["spread_ms"], "spread_pct": r["spread_pct"],
                "speedup_vs_previous": r2(prev_ms / ms) if prev_ms else "",
                "speedup_vs_step0": r2(base_ms / ms),
                "decoder_fps": round(FRAMES_PER_CHUNK / (ms / 1e3), 1),
                "kernels_per_chunk": k.get("kernels_per_chunk", ""),
                "gpu_ms_per_chunk": k.get("gpu_ms_per_chunk", ""),
                "executed_tflop": round(executed, 2),
                "achieved_tflops": round(useful / (ms / 1e3)),
                "peak_allocated_gib": r["peak_allocated_gib"],
                "psnr_u8_vs_fp32": r["psnr_u8_db"],
                "md5_u8_stream": fr.get("md5_u8_stream", ""),
                "identical_to_previous": fr.get("identical_to_previous", ""),
                "pct_values_diff_vs_previous": fr.get("pct_values_diff_vs_previous", ""),
                "pct_pixels_diff_vs_previous": fr.get("pct_pixels_diff_vs_previous", ""),
                "identical_to_stock_node_input": fr.get("identical_to_ref", ""),
                "pct_values_diff_vs_stock_node_input": fr.get("pct_values_diff_vs_ref", ""),
            })
            prev_ms = ms
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    run, out = Path(args.run), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    abl = run / "ablation"

    runs = []
    for vae in ("wan21", "wan22"):
        runs += [run_row(vae, vae, r) for r in load(run / vae).values()]
    for group, vae in (("px", "wan21"), ("tf32", "wan21"), ("wan22", "wan22")):
        runs += [run_row(vae, f"ablation/{group}", r) for r in load(abl / group).values()]
    write_csv(out / "runs.csv", runs)

    var_rows = three_runs(abl / "variance", "wan21")
    means = {r["variant"]: statistics.fmean(v for k, v in r.items() if k.startswith("run")) for r in var_rows}
    write_csv(out / "variance.csv", var_rows)
    write_csv(out / "intree.csv", [
        {"session": session, **r}
        for session, d, vae in (("in-tree", "intree", "wan21"), ("in-tree", "intree_wan22", "wan22"),
                                ("cudnn.benchmark control", "control", "wan21"))
        for r in three_runs(abl / d, vae)
    ])
    write_csv(out / "nodes.csv", [node_row(r) for r in load_nodes(abl / "nodes")])
    for vae in ("wan21", "wan22"):
        write_csv(out / f"ladder_{vae}.csv", ladder_rows(abl, vae))

    def wall(vae, variant):
        """Wall ms per chunk: 3-run mean, else an ablation run, else the run_all suite."""
        if vae == "wan21" and variant in means:
            return means[variant], "variance mean"
        for row in runs:
            if row["vae"] == vae and row["variant"] == variant and row["group"].startswith("ablation"):
                return row["median_ms"], row["group"]
        for row in runs:
            if row["vae"] == vae and row["variant"] == variant:
                return row["median_ms"], row["group"]
        return None, ""

    prof_rows = []
    for vae, pdir in (("wan21", abl / "profile"), ("wan22", abl / "profile_wan22")):
        flops_p = next(pdir.glob("flops_*.json"), None) if pdir.is_dir() else None
        if not flops_p:
            continue
        fl = json.loads(flops_p.read_text())
        useful = fl["chunk_flops"] / 1e12
        intensity = fl["chunk_flops"] / fl["chunk_min_conv_bytes"]
        roof = min(PEAK_TFLOPS, intensity * PEAK_TBS)
        for kp in sorted(pdir.glob("*.kernels.json")):
            k = json.loads(kp.read_text())
            v = k["variant"]
            ms, src = wall(vae, v)
            executed = fl["mstar_ar_ctx8_executed_flops"] / 1e12 if v.startswith("mstar_ar_ctx8") else useful
            cats = k["category_ms_per_chunk"]
            achieved = useful / (ms / 1e3) if ms else None
            prof_rows.append({
                "vae": vae, "variant": v, "kernels_per_chunk": k["kernels_per_chunk"],
                "gpu_ms_per_chunk": k["gpu_ms_per_chunk"], "wall_ms_per_chunk": r2(ms), "wall_source": src,
                **{"ms_" + c.replace(" / ", "_").replace(" ", "_"): cats.get(c, 0.0) for c in CATEGORIES},
                "useful_tflop": round(useful, 2), "executed_tflop": round(executed, 2),
                "conv_tflop": round(fl["chunk_conv_flops"] / 1e12, 2),
                "intensity_flop_per_byte": round(intensity), "roof_tflops": roof,
                "achieved_tflops": round(achieved) if achieved else None,
                "pct_of_roof": round(achieved / roof * 100, 1) if achieved else None,
                # FLOPs the conv kernels execute (the re-decode runs more than the useful work).
                "conv_kernel_tflops": round(
                    fl["chunk_conv_flops"] / 1e12 * executed / useful / (cats["convolution"] / 1e3)
                ),
            })
    write_csv(out / "profile.csv", prof_rows)


if __name__ == "__main__":
    main()
