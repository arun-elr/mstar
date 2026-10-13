"""Quick output check on the whole stream (1,021 frames), no LPIPS: md5 of each
uint8 stream (and whether it equals the B200 stream's md5), PSNR vs the fp32
reference decoded on this GPU (mean / worst), values that differ from stock
diffusers bf16, seam, and second-process determinism. Same definitions as
quality.py analyze."""

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np

F = 12
VARIANTS = ["ref_fp32", "diffusers_bf16", "mstar_ar_ctx8", "mstar_warm", "mstar_warm_nobench"]


def md5(a):
    h = hashlib.md5()
    for i in range(0, a.shape[0], 64):
        h.update(np.ascontiguousarray(a[i:i + 64]).tobytes())
    return h.hexdigest()


def psnr_frames(a, b):
    out = []
    for i in range(0, a.shape[0], 32):
        d = a[i:i + 32].astype(np.float32) - b[i:i + 32].astype(np.float32)
        mse = (d * d).reshape(d.shape[0], -1).mean(axis=1)
        with np.errstate(divide="ignore"):
            out.append(np.where(mse == 0, np.inf, 10 * np.log10(255.0**2 / np.maximum(mse, 1e-12))))
    return np.concatenate(out)


def place(i):
    if i == 0:
        return "first latent"
    c, k = divmod(i - 1, F)
    return f"c{c} f{k + 1}" + ("⟂" if k in (0, F - 1) else "")


def diff_stats(a, b):
    d1 = d2 = tot = mx = 0
    for i in range(0, a.shape[0], 32):
        d = np.abs(a[i:i + 32].astype(np.int16) - b[i:i + 32].astype(np.int16))
        d1 += int((d >= 1).sum()); d2 += int((d >= 2).sum()); tot += d.size; mx = max(mx, int(d.max()))
    return d1 / tot * 100, d2 / tot * 100, mx


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vae", required=True)
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--b200", help="B200 quality.csv (md5 per variant)")
    args = ap.parse_args()
    d = Path(args.dir)
    b200 = {}
    if args.b200 and Path(args.b200).exists():
        b200 = {r["variant"]: r for r in csv.DictReader(open(args.b200)) if r["vae"] == args.vae}
    u8 = {v: np.load(d / f"{v}.u8.npy", mmap_mode="r") for v in VARIANTS if (d / f"{v}.u8.npy").exists()}
    ref, stock = u8["ref_fp32"], u8["diffusers_bf16"]
    rows = []
    for v, a in u8.items():
        h = md5(a)
        pr = psnr_frames(a, ref)
        fin = pr[np.isfinite(pr)]
        ps = psnr_frames(a, stock)
        steady = pr[1:].reshape(-1, F)
        seam = float((steady[:, 0] - steady[:, 1:].mean(axis=1)).mean()) if np.isfinite(steady).all() else ""
        g1, g2, mx = diff_stats(a, stock)
        run2 = d / f"{v}.run2.u8.npy"
        det = ""
        if run2.exists():
            b = np.load(run2, mmap_mode="r")
            det = "bit-identical (uint8)" if np.array_equal(a, b) else f"max uint8 diff {diff_stats(a, b)[2]}"
        br = b200.get(v, {})
        rows.append({
            "vae": args.vae, "variant": v, "frames": a.shape[0], "md5_u8_stream": h,
            "b200_md5_u8_stream": br.get("md5_u8_stream", ""),
            "same_md5_as_b200": (h == br["md5_u8_stream"]) if br else "",
            "identical_to_stock": h == md5(stock),
            "psnr_u8_vs_fp32": round(float(fin.mean()), 2) if len(fin) else "inf",
            "worst_psnr_u8_vs_fp32": round(float(pr.min()), 2),
            "worst_psnr_vs_fp32_at": place(int(np.argmin(pr))),
            "b200_psnr_u8_vs_fp32": br.get("psnr_u8_vs_fp32", ""),
            "b200_worst_psnr_u8_vs_fp32": br.get("worst_psnr_u8_vs_fp32", ""),
            "psnr_u8_vs_stock": round(float(ps[np.isfinite(ps)].mean()), 2) if np.isfinite(ps).any() else "inf",
            "pct_u8_diff_ge1_vs_stock": round(g1, 3), "pct_u8_diff_ge2_vs_stock": round(g2, 4),
            "max_abs_u8_vs_stock": mx,
            "b200_pct_u8_diff_ge1_vs_stock": br.get("pct_u8_diff_ge1_vs_stock", ""),
            "b200_max_abs_u8_vs_stock": br.get("max_abs_u8_vs_stock", ""),
            "seam_db": round(seam, 3) if seam != "" else "",
            "b200_seam_db": br.get("seam_db", ""),
            "deterministic_run2": det,
            "b200_deterministic_run2": br.get("deterministic_run2", ""),
        })
        print(json.dumps(rows[-1]))
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    p = out / "quality_check.csv"
    keep = [r for r in csv.DictReader(open(p)) if r["vae"] != args.vae] if p.exists() else []
    with open(p, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        w.writeheader(); w.writerows(keep + rows)
    print(f"wrote {p}")


if __name__ == "__main__":
    main()
