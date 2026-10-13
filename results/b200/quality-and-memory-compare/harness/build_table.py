"""quality_compare.csv (one row per engine / setting, the M* rows taken from the PR's
quality.csv unchanged) and the markdown table for SUMMARY.md."""
import csv
from pathlib import Path

HERE = Path(__file__).resolve().parent.parent
PR = Path("${MSTAR_PR}/benchmark/wan_vae_decoder/results/b200")

# ms per 3-latent chunk, 3-run means, from the existing timing (PR comparison.csv / handoff SUMMARY.md).
MS = {
    ("wan21", "M* before"): 335.25, ("wan22", "M* before"): 368.85,
    ("wan21", "stock diffusers bf16"): 191.98, ("wan22", "stock diffusers bf16"): 184.57,
    ("wan21", "M* this PR"): 56.30, ("wan22", "M* this PR"): 73.76,
    ("wan21", "vLLM-Omni default"): 97.32, ("wan22", "vLLM-Omni default"): 105.92,
    ("wan21", "vLLM-Omni best"): 64.08, ("wan22", "vLLM-Omni best"): 77.85,
    ("wan21", "SGLang default"): 107.64, ("wan22", "SGLang default"): 170.97,
    ("wan22", "SGLang bf16"): 115.72,
    ("wan21", "SGLang best"): 92.50, ("wan22", "SGLang best"): 94.36,
}
MSTAR = {"mstar_ar_ctx8": ("M*", "before (re-decode 8 context latents)", "M* before"),
         "diffusers_bf16": ("diffusers", "stock bf16, eager", "stock diffusers bf16"),
         "mstar_warm": ("M*", "this PR (mstar_warm)", "M* this PR")}
ENGINE = {("vLLM-Omni", "default"): ("vLLM-Omni default", "`vae_fast_path=lossless` (default)"),
          ("vLLM-Omni", "best"): ("vLLM-Omni best", "`--vae-fast-path channels_last`"),
          ("SGLang", "default"): ("SGLang default", "default (`quality=lossless`, bf16)"),
          ("SGLang", "default (as served, fp32 decode)"): ("SGLang default", "default as served (fp32 decode)"),
          ("SGLang", "bf16"): ("SGLang bf16", "default + `--vae-decode-precision bf16`"),
          ("SGLang", "best"): ("SGLang best", "best (`--quality extra-high`, bf16)")}
COLS = ["vae", "engine", "setting", "ms_per_chunk", "md5_u8_stream", "identical_to_stock",
        "pct_u8_diff_ge1_vs_stock", "pct_u8_diff_ge2_vs_stock", "max_abs_u8_vs_stock",
        "pct_u8_equal_stock_same_uint8_conversion",
        "psnr_u8_vs_fp32", "worst_psnr_u8_vs_fp32", "worst_psnr_vs_fp32_frame", "worst_psnr_vs_fp32_at",
        "ssim_vs_fp32", "worst_ssim_vs_fp32", "worst_ssim_vs_fp32_frame", "worst_ssim_vs_fp32_at",
        "psnr_u8_vs_stock", "psnr_trend_db_per_chunk", "seam_db", "uint8_conversion", "source"]


def frame_of(at: str) -> str:
    if at == "first latent":
        return "0"
    c, k = at.removeprefix("chunk ").split(", frame ")
    return str(1 + int(c) * 12 + int(k.split("/")[0]) - 1)


def main():
    rows = []
    raw = list(csv.DictReader(open(HERE / "quality_compare_raw.csv")))
    for vae in ("wan21", "wan22"):
        for r in csv.DictReader(open(PR / "quality.csv")):
            if r["vae"] != vae or r["variant"] not in MSTAR:
                continue
            eng, setting, key = MSTAR[r["variant"]]
            rows.append({**{k: r.get(k, "") for k in COLS}, "vae": vae, "engine": eng, "setting": setting,
                         "ms_per_chunk": MS[(vae, key)],
                         "worst_psnr_vs_fp32_frame": frame_of(r["worst_psnr_vs_fp32_at"]),
                         "worst_ssim_vs_fp32_frame": frame_of(r["worst_ssim_vs_fp32_at"]),
                         "pct_u8_equal_stock_same_uint8_conversion": round(100 - float(r["pct_u8_diff_ge1_vs_stock"]), 3),
                         "uint8_conversion": "M* to_uint8 on the bf16 output" if r["variant"] != "mstar_warm"
                         else "M* node (uint8 out)",
                         "source": "PR results/b200/quality.csv (unchanged)"})
        for r in raw:
            if r["vae"] != vae or (r["engine"], r["setting"]) not in ENGINE:
                continue
            key, label = ENGINE[(r["engine"], r["setting"])]
            sg = r["engine"] == "SGLang"
            same = (r["pct_u8_equal_own_stack_stock"] if sg else round(100 - float(r["pct_u8_diff_ge1_vs_stock"]), 3))
            rows.append({**{k: r.get(k, "") for k in COLS}, "engine": r["engine"], "setting": label,
                         "ms_per_chunk": MS[(vae, key)], "pct_u8_equal_stock_same_uint8_conversion": same,
                         "uint8_conversion": "SGLang: (x.float()/2+0.5)*255 in fp32" if sg
                         else "M* to_uint8 on the bf16 output",
                         "source": "this comparison (quality_compare_raw.csv)"})
    with open(HERE / "quality_compare.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    print(f"wrote quality_compare.csv ({len(rows)} rows)\n")
    names = {"wan21": "Wan 2.1", "wan22": "Wan 2.2"}
    print("| VAE | engine | setting | ms per chunk | PSNR vs fp32 mean / worst (dB) | SSIM worst | "
          "% values ≠ stock (≥1 / ≥2) | md5 = stock? | seam dB |")
    print("|---|---|---|---:|---|---:|---|---|---:|")
    for r in rows:
        pct = f"{float(r['pct_u8_diff_ge1_vs_stock']):.2f}% / {float(r['pct_u8_diff_ge2_vs_stock']):.3f}%"
        if r["engine"] == "SGLang":
            pct += f" ({100 - float(r['pct_u8_equal_stock_same_uint8_conversion']):.2f}%)"
        bnd = "⟂" if "boundary" in r["worst_psnr_vs_fp32_at"] else ""
        md5 = "yes" if r["identical_to_stock"] == "True" else "no"
        print(f"| {names[r['vae']]} | {r['engine']} | {r['setting']} | {float(r['ms_per_chunk']):.2f} | "
              f"{float(r['psnr_u8_vs_fp32']):.2f} / {float(r['worst_psnr_u8_vs_fp32']):.2f} "
              f"(f{r['worst_psnr_vs_fp32_frame']}{bnd}) | {float(r['worst_ssim_vs_fp32']):.4f} | {pct} | {md5} | "
              f"{float(r['seam_db']):+.2f} |")


if __name__ == "__main__":
    main()
