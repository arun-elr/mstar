"""Engine frame streams vs M*'s stock diffusers bf16 and fp32 reference, with
quality.py's own metric functions and definitions (analyze()).

  PYTHONDONTWRITEBYTECODE=1 python qc_analyze.py --vae wan21 --dir frames/wan21 --out quality_compare_raw.csv \
      --stock diffusers_bf16 --ref ref_fp32 --streams engine:setting:file.npy ...

Per stream: md5 of the uint8 stream (quality.md5) and whether it equals stock;
share of uint8 values that differ from stock by >= 1 and >= 2 levels and the max;
per-frame PSNR (uint8, peak 255) vs fp32, mean and worst frame (quality.frame_place);
SSIM vs fp32 (quality.ssim, Gaussian 11x11), mean and worst; PSNR vs stock;
seam dB exactly as analyze(): mean over chunks of (first steady frame PSNR vs fp32
minus the chunk's other frames). Also bytes identical to an optional second
"own-stack stock" stream per engine.
"""

import argparse
import csv
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, os.environ.get("MSTAR_BENCH", "${MSTAR_PR}/benchmark/wan_vae_decoder"))
import quality as Q  # noqa: E402
from common import FRAMES_PER_CHUNK  # noqa: E402


def load(d: Path, name: str):
    p = d / name if name.endswith(".npy") else d / f"{name}.u8.npy"
    return np.load(p, mmap_mode="r")


def analyze_stream(x, st, rf, win, bs):
    n_frames = st.shape[0]
    assert x.shape == st.shape == rf.shape, (x.shape, st.shape, rf.shape)
    psnr_ref, psnr_stock, ssim_ref = [], [], []
    diff1 = diff2 = total = 0
    max_u8 = 0
    for s in range(0, n_frames, bs):
        sl = slice(s, min(s + bs, n_frames))
        xt = torch.from_numpy(np.ascontiguousarray(x[sl])).permute(0, 3, 1, 2).float()
        stt = torch.from_numpy(np.ascontiguousarray(st[sl])).permute(0, 3, 1, 2).float()
        rft = torch.from_numpy(np.ascontiguousarray(rf[sl])).permute(0, 3, 1, 2).float()
        ad = (xt - stt).abs()
        diff1 += int((ad >= 1).sum())
        diff2 += int((ad >= 2).sum())
        total += ad.numel()
        max_u8 = max(max_u8, int(ad.max()))
        psnr_ref.append(Q.psnr_frames(xt, rft))
        psnr_stock.append(Q.psnr_frames(xt, stt))
        ssim_ref.append(Q.ssim(xt, rft, win))
    a = {"psnr_ref": torch.cat(psnr_ref).double(), "psnr_stock": torch.cat(psnr_stock).double(),
         "ssim_ref": torch.cat(ssim_ref).double()}
    steady = a["psnr_ref"][1:].view(-1, FRAMES_PER_CHUNK)
    chunk_mean = steady.mean(dim=1).numpy()
    slope = float(np.polyfit(np.arange(len(chunk_mean)), chunk_mean, 1)[0])
    seam = float((steady[:, 0] - steady[:, 1:].mean(dim=1)).mean())
    wpr, wpr_i = Q.worst(a["psnr_ref"])
    wsr, wsr_i = Q.worst(a["ssim_ref"])
    return a, {
        "pct_u8_diff_ge1_vs_stock": round(100 * diff1 / total, 3),
        "pct_u8_diff_ge2_vs_stock": round(100 * diff2 / total, 4),
        "max_abs_u8_vs_stock": max_u8,
        "psnr_u8_vs_fp32": round(Q.mean_or_inf(a["psnr_ref"]), 2),
        "worst_psnr_u8_vs_fp32": round(wpr, 2), "worst_psnr_vs_fp32_frame": wpr_i,
        "worst_psnr_vs_fp32_at": Q.frame_place(wpr_i),
        "ssim_vs_fp32": round(float(a["ssim_ref"].mean()), 5),
        "worst_ssim_vs_fp32": round(wsr, 5), "worst_ssim_vs_fp32_frame": wsr_i,
        "worst_ssim_vs_fp32_at": Q.frame_place(wsr_i),
        "psnr_u8_vs_stock": round(Q.mean_or_inf(a["psnr_stock"]), 2),
        "psnr_trend_db_per_chunk": round(slope, 4),
        "seam_db": round(seam, 3),
        "frames": n_frames,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--stock", default="diffusers_bf16")
    ap.add_argument("--ref", default="ref_fp32")
    ap.add_argument("--streams", nargs="+", required=True,
                    help="engine:setting:file[:own_stock_file] (files relative to --dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames-out", default="")
    ap.add_argument("--batch", type=int, default=12)
    args = ap.parse_args()
    torch.set_grad_enabled(False)
    d = Path(args.dir)
    st, rf = load(d, args.stock), load(d, args.ref)
    stock_md5 = Q.md5(st)
    win = Q.gaussian_window("cpu")
    rows, frame_rows = [], []
    for spec in args.streams:
        engine, setting, f, *own = spec.split(":")
        x = load(d, f)
        h = Q.md5(x)
        a, m = analyze_stream(x, st, rf, win, args.batch)
        row = {"vae": args.vae, "engine": engine, "setting": setting, "file": f, "md5_u8_stream": h,
               "stock_md5": stock_md5, "identical_to_stock": h == stock_md5, **m}
        if own:
            o = load(d, own[0])
            vals, _ = Q.count_diff(x, o)
            row["own_stack_stock_file"] = own[0]
            row["own_stack_stock_md5"] = Q.md5(o)
            row["pct_u8_equal_own_stack_stock"] = round(100 * (1 - vals / x.size), 3)
        rows.append(row)
        print(row, flush=True)
        frame_rows += [{"vae": args.vae, "engine": engine, "setting": setting, "frame": i,
                        "psnr_u8_vs_fp32": round(float(a["psnr_ref"][i]), 3),
                        "psnr_u8_vs_stock": round(float(a["psnr_stock"][i]), 3),
                        "ssim_vs_fp32": round(float(a["ssim_ref"][i]), 6)} for i in range(x.shape[0])]
    fields = list(dict.fromkeys(k for r in rows for k in r))
    write(Path(args.out), rows, fields, args.vae)
    if args.frames_out:
        write(Path(args.frames_out), frame_rows, list(frame_rows[0]), args.vae)


def write(path: Path, rows, fields, vae):
    """Replace this VAE's rows, keep the other VAE's (like quality.append_csv, but tolerant of new columns)."""
    keep = []
    if path.exists():
        with open(path) as fh:
            keep = [r for r in csv.DictReader(fh) if r["vae"] != vae]
    allf = list(dict.fromkeys([*fields, *(k for r in keep for k in r)]))
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=allf, lineterminator="\n")
        w.writeheader()
        w.writerows(keep + rows)
    print(f"wrote {path} ({len(keep) + len(rows)} rows)")


if __name__ == "__main__":
    main()
