"""quality_compare.csv and memory_compare.csv, plus the markdown tables for SUMMARY.md.

M* rows (stock diffusers bf16, M* before, M* this PR) come from the PR's quality.csv
unchanged; engine rows from quality_compare_raw.csv (qc_analyze.py). ms per chunk
and peak GiB come from the existing timing runs (3-run means; peaks are identical
in all 3 runs); the memory split comes from runs/mem_probe.jsonl (mem_probe.py).
"""
import csv
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PR = Path("${MSTAR_PR}/benchmark/wan_vae_decoder/results/b200")
NAMES = {"wan21": "Wan 2.1", "wan22": "Wan 2.2"}

# key: (ms per chunk, timing peak_allocated_gib, latents per decoder call, timing source)
TIMING = {
    ("wan21", "stock"): (191.98, 4.44, "1", "PR variance.csv / runs.csv"),
    ("wan22", "stock"): (184.57, 5.53, "1", "PR runs.csv"),
    ("wan21", "mstar_before"): (335.25, 6.33, "11 (8 context + 3 new, stateless)", "PR variance.csv"),
    ("wan22", "mstar_before"): (368.85, 6.44, "11 (8 context + 3 new, stateless)", "PR intree.csv"),
    ("wan21", "mstar_warm"): (56.30, 19.03, "3 (=0: one call per chunk)", "PR intree.csv"),
    ("wan22", "mstar_warm"): (73.76, 15.95, "3 (=0: one call per chunk)", "PR intree.csv"),
    ("wan21", "vo_default"): (97.32, 4.53, "1", "handoff vllm-omni runs/main"),
    ("wan22", "vo_default"): (105.92, 5.42, "1", "handoff vllm-omni runs/main"),
    ("wan21", "vo_best"): (64.08, 3.82, "1", "handoff vllm-omni runs/main"),
    ("wan22", "vo_best"): (77.85, 4.66, "1", "handoff vllm-omni runs/main"),
    ("wan21", "sg_default"): (107.64, 6.81, "1", "handoff sglang runs/wan21"),
    ("wan22", "sg_default"): (170.97, 9.33, "1", "handoff sglang runs/wan22"),
    ("wan22", "sg_bf16"): (115.72, 7.76, "1", "handoff sglang runs/wan22"),
    ("wan21", "sg_best"): (92.50, 3.80, "1", "handoff sglang runs/wan21"),
    ("wan22", "sg_best"): (94.36, 4.73, "1", "handoff sglang runs/wan22"),
}
MSTAR = {"diffusers_bf16": ("diffusers", "stock bf16, eager", "stock"),
         "mstar_ar_ctx8": ("M*", "before (re-decode 8 context latents)", "mstar_before"),
         "mstar_warm": ("M*", "this PR (`mstar_warm`, defaults)", "mstar_warm")}
ENGINE = {("vLLM-Omni", "default"): ("default (`vae_fast_path=lossless`)", "vo_default"),
          ("vLLM-Omni", "best"): ("best (`--vae-fast-path channels_last`)", "vo_best"),
          ("SGLang", "default"): ("default (`quality=lossless`, bf16)", "sg_default"),
          ("SGLang", "default (as served, fp32 decode)"): ("default as served (fp32 decode)", "sg_default"),
          ("SGLang", "bf16"): ("default + `--vae-decode-precision bf16`", "sg_bf16"),
          ("SGLang", "best"): ("best (`--quality extra-high`, bf16)", "sg_best"),
          ("SGLang", "stock diffusers bf16 via SGLang md5_frames.py (control)"):
              ("reference: stock bf16 decode, uint8 conversion in fp32 like SGLang", None)}
PROBE = {"vllm_omni_default": "vo_default", "vllm_omni_best": "vo_best", "sglang_default": "sg_default",
         "sglang_default_bf16": "sg_bf16", "sglang_best": "sg_best", "mstar_warm": "mstar_warm",
         "mstar_warm_lpc1": "mstar_lpc1"}
COLS = ["vae", "engine", "setting", "ms_per_chunk", "peak_allocated_gib", "latents_per_decoder_call",
        "md5_u8_stream", "identical_to_stock", "pct_u8_diff_ge1_vs_stock", "pct_u8_diff_ge2_vs_stock",
        "max_abs_u8_vs_stock", "pct_u8_diff_vs_stock_same_uint8_conversion",
        "psnr_u8_vs_fp32", "worst_psnr_u8_vs_fp32", "worst_psnr_vs_fp32_frame", "worst_psnr_vs_fp32_at",
        "ssim_vs_fp32", "worst_ssim_vs_fp32", "worst_ssim_vs_fp32_frame", "worst_ssim_vs_fp32_at",
        "psnr_u8_vs_stock", "psnr_trend_db_per_chunk", "seam_db", "uint8_conversion", "quality_source",
        "timing_source"]


def frame_of(at: str) -> str:
    if at == "first latent":
        return "0"
    c, k = at.removeprefix("chunk ").split(", frame ")
    return str(1 + int(c) * 12 + int(k.split("/")[0]) - 1)


def short_at(at: str) -> str:
    if at == "first latent":
        return "f0"
    c, k = at.removeprefix("chunk ").split(", frame ")
    return f"c{c} f{k.split('/')[0]}" + ("⟂" if "boundary" in at else "")


def quality_rows():
    rows = []
    raw = list(csv.DictReader(open(HERE / "quality_compare_raw.csv")))
    pr = list(csv.DictReader(open(PR / "quality.csv")))
    for vae in ("wan21", "wan22"):
        for v in MSTAR:
            r = next(x for x in pr if x["vae"] == vae and x["variant"] == v)
            eng, setting, key = MSTAR[v]
            ms, peak, lpc, tsrc = TIMING[(vae, key)]
            rows.append({**{k: r.get(k, "") for k in COLS}, "vae": vae, "engine": eng, "setting": setting,
                         "ms_per_chunk": ms, "peak_allocated_gib": peak, "latents_per_decoder_call": lpc,
                         "worst_psnr_vs_fp32_frame": frame_of(r["worst_psnr_vs_fp32_at"]),
                         "worst_ssim_vs_fp32_frame": frame_of(r["worst_ssim_vs_fp32_at"]),
                         "pct_u8_diff_vs_stock_same_uint8_conversion": r["pct_u8_diff_ge1_vs_stock"],
                         "uint8_conversion": "M* node (uint8 out)" if v != "diffusers_bf16"
                         else "M* to_uint8 on the bf16 output",
                         "quality_source": "PR results/b200/quality.csv (unchanged)", "timing_source": tsrc})
        for (eng, setting), (label, key) in ENGINE.items():
            r = next((x for x in raw if x["vae"] == vae and x["engine"] == eng and x["setting"] == setting), None)
            if r is None:
                continue
            ms, peak, lpc, tsrc = TIMING[(vae, key)] if key else ("", "", "", "")
            sg = eng == "SGLang"
            same = (round(100 - float(r["pct_u8_equal_own_stack_stock"]), 3) if sg and key
                    else 0.0 if sg else r["pct_u8_diff_ge1_vs_stock"])
            rows.append({**{k: r.get(k, "") for k in COLS}, "engine": eng, "setting": label,
                         "ms_per_chunk": ms, "peak_allocated_gib": peak, "latents_per_decoder_call": lpc,
                         "pct_u8_diff_vs_stock_same_uint8_conversion": same,
                         "uint8_conversion": "SGLang: (x.float()/2+0.5)*255 in fp32" if sg
                         else "M* to_uint8 on the bf16 output",
                         "quality_source": "this comparison (quality_compare_raw.csv)", "timing_source": tsrc})
    return rows


def memory_rows():
    probes = [json.loads(l) for l in open(HERE / "runs" / "mem_probe.jsonl") if l.strip()]
    labels = {"vo_default": ("vLLM-Omni", "default"), "vo_best": ("vLLM-Omni", "best"),
              "sg_default": ("SGLang", "default"), "sg_bf16": ("SGLang", "bf16"), "sg_best": ("SGLang", "best"),
              "mstar_warm": ("M*", "this PR, `windowed_decode_latents_per_call=0` (default)"),
              "mstar_lpc1": ("M*", "this PR, `windowed_decode_latents_per_call=1`")}
    rows = []
    for vae in ("wan21", "wan22"):
        for key, (eng, setting) in labels.items():
            p = next((x for x in probes if x["vae"] == vae and PROBE[x["variant"]] == key), None)
            if p is None:
                continue
            t = TIMING.get((vae, key))
            if eng == "SGLang" and key == "sg_default" and vae == "wan22":
                setting = "default as served (fp32 decode)"
            rows.append({
                "vae": vae, "engine": eng, "setting": setting,
                "latents_per_decoder_call": p["latents_per_decoder_call"] if key != "mstar_warm" else 3,
                "peak_allocated_gib_timing": t[1] if t else "",
                "peak_allocated_gib_probe": p["peak_allocated_gib"],
                "weights_resident_gib": p["weights_resident_steady_gib"],
                "encoder_weights_gib": p["encoder_weights_resident_gib"],
                "latents_on_gpu_gib": p["latents_on_gpu_gib"],
                "state_between_chunks_gib": p["state_between_chunks_gib"],
                "held_tensors": p["held_tensors"], "held_logical_gib": p["held_logical_gib"],
                "held_storage_gib": p["held_storage_gib"],
                "transient_peak_one_chunk_gib": p["transient_peak_gib"],
                "torch": p["torch"],
            })
    return rows


def write(path, rows, cols):
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=cols, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote {path} ({len(rows)} rows)")


def main():
    q = quality_rows()
    write(HERE / "quality_compare.csv", q, COLS)
    m = memory_rows()
    write(HERE / "memory_compare.csv", m, list(m[0]))
    print()
    print("| VAE | engine | setting | ms per chunk | peak GiB | PSNR vs fp32 mean / worst (dB) | SSIM vs fp32 worst | "
          "% values ≠ stock (≥1 / ≥2) | md5 = stock? | seam dB |")
    print("|---|---|---|---:|---:|---|---:|---|---|---:|")
    for r in q:
        pct = f"{float(r['pct_u8_diff_ge1_vs_stock']):.2f}% / {float(r['pct_u8_diff_ge2_vs_stock']):.3f}%"
        if r["engine"] == "SGLang":
            pct += f" ({float(r['pct_u8_diff_vs_stock_same_uint8_conversion']):.2f}%)"
        ms = f"{float(r['ms_per_chunk']):.2f}" if r["ms_per_chunk"] != "" else "—"
        pk = f"{float(r['peak_allocated_gib']):.2f}" if r["peak_allocated_gib"] != "" else "—"
        md5 = "yes" if r["identical_to_stock"] == "True" else "no"
        print(f"| {NAMES[r['vae']]} | {r['engine']} | {r['setting']} | {ms} | {pk} | "
              f"{float(r['psnr_u8_vs_fp32']):.2f} / {float(r['worst_psnr_u8_vs_fp32']):.2f} "
              f"({short_at(r['worst_psnr_vs_fp32_at'])}) | {float(r['worst_ssim_vs_fp32']):.4f} | {pct} | {md5} | "
              f"{float(r['seam_db']):+.2f} |")
    print()
    print("| VAE | engine | setting | latents per decoder call | peak GiB (timing / probe) | weights resident (encoder) | "
          "latents on GPU | state held between chunks: allocated (logical cache) | transient peak of one chunk |")
    print("|---|---|---|---:|---:|---|---:|---|---:|")
    for r in m:
        enc = f" ({r['encoder_weights_gib']:.2f})" if r["encoder_weights_gib"] else " (none)"
        tp = f"{r['peak_allocated_gib_timing']:.2f}" if r["peak_allocated_gib_timing"] != "" else "—"
        print(f"| {NAMES[r['vae']]} | {r['engine']} | {r['setting']} | {r['latents_per_decoder_call']} | "
              f"{tp} / {r['peak_allocated_gib_probe']:.2f} | {r['weights_resident_gib']:.2f}{enc} | "
              f"{r['latents_on_gpu_gib']:.2f} | {r['held_storage_gib']:.2f} ({r['held_logical_gib']:.2f}, "
              f"{r['held_tensors']} tensors) | {r['transient_peak_one_chunk_gib']:.2f} |")


if __name__ == "__main__":
    main()
