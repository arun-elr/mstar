"""Merge metrics_<prompt>.csv into metrics.csv and print the markdown tables."""

import csv
import json
import sys
from pathlib import Path

import torch

out = Path(sys.argv[1])
rows = []
for p in ("fox", "waves"):
    with open(out / f"metrics_{p}.csv") as f:
        rows += list(csv.DictReader(f))
with open(out / "metrics.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)

NAMES = {
    "stock_bf16": "stock diffusers `vae.decode` (reference)",
    "stock_fp32": "diffusers pipeline decode, fp32 VAE (context)",
    "s0_mstar_before": "M* before (stateless re-decode)",
    "s1_warm_cache": "+ warm cache",
    "s2_channels_last": "+ channels-last (incl. native padding)",
    "s3_one_call": "+ one decoder call per chunk",
    "s4_cudnn_benchmark": "+ cudnn.benchmark (opt-in)",
    "c1_warm_cache_eager": "control: warm cache only, eager",
}


def f(x, nd=2):
    if x in ("", None):
        return "–"
    if x in ("inf", "Infinity"):
        return "∞"
    v = float(x)
    return f"{v:.{nd}f}"


for p in ("fox", "waves"):
    print(f"\n### {p}\n")
    print("| step | md5 (uint8 frames) | = stock | px ≥1 lvl | px ≥2 lvl | max | PSNR mean / worst (dB) | SSIM worst "
          "| ms/chunk (median) | total ms |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    base = None
    for r in rows:
        if r["prompt"] != p:
            continue
        if r["step"] == "s0_mstar_before":
            base = float(r["steady_chunk_ms_median"])
        print(f"| {NAMES[r['step']]} | `{r['md5_u8'][:12]}` | {'yes' if r['identical_to_stock'] == 'True' else 'no'} "
              f"| {f(r['pct_pixels_diff_ge1'])} % | {f(r['pct_pixels_diff_ge2'], 3)} % | {r['max_abs_diff']} "
              f"| {f(r['psnr_frame_mean_db'], 1)} / {f(r['psnr_worst_db'], 1)} | {f(r['ssim_worst'], 4)} "
              f"| {f(r['steady_chunk_ms_median'], 1)} | {f(r['decode_total_ms'], 0)} |")
    print()
    for r in rows:
        if r["prompt"] == p and r["steady_chunk_ms_median"] and base:
            print(f"  {r['step']}: {base / float(r['steady_chunk_ms_median']):.2f}x vs M* before per chunk, "
                  f"peak {r['peak_alloc_gib']} GiB, pass1 {r['pass1_total_ms']} ms, "
                  f"first window {r['first_window_ms']}, "
                  f"identical frames {r['identical_frames']}, both passes same {r['same_md5_both_passes']}, "
                  f"prev same {r['identical_to_prev_step']}, clip psnr {r['psnr_clip_db']}")

for p in ("fox", "waves"):
    m = torch.load(Path(sys.argv[2]) / f"{p}_latents.pt")["meta"]
    print(json.dumps({k: v for k, v in m.items() if k != "negative_prompt"}))
