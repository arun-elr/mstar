"""Metrics vs stock and the PR media for one prompt's decoded ladder.

  python analyze.py --run runs/fox --out out --prompt-id fox

Reads ``<step>.u8.npy`` / ``<step>.json`` from ``--run``; writes
``metrics_<prompt>.csv`` and the media into ``--out``.

Metrics are on the raw uint8 frames (not the mp4s), against ``stock_bf16``
(stock diffusers ``vae.decode`` on the same bf16 VAE input):
  pixel = one (frame, y, x) position; it differs by >= k when any of its RGB
  values does. PSNR per frame over all three channels, peak 255; the clip
  PSNR pools the squared error of every frame. SSIM: Gaussian 11x11,
  sigma 1.5, per RGB channel on [0, 255], averaged over the frame.
"""

import argparse
import csv
import hashlib
import json
import math
import subprocess
from pathlib import Path

import imageio_ffmpeg
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import colormaps
from PIL import Image, ImageDraw, ImageFont

FPS = 24
STEPS = [
    ("stock_bf16", "stock diffusers vae.decode (bf16, reference)"),
    ("stock_fp32", "diffusers pipeline decode (fp32 VAE, model card; context)"),
    ("s0_mstar_before", "M* before: stateless re-decode of 8 context latents"),
    ("s1_warm_cache", "+ warm conv cache"),
    ("s2_channels_last", "+ channels-last (with native conv padding)"),
    ("s3_one_call", "+ one decoder call per chunk"),
    ("s4_cudnn_benchmark", "+ cudnn.benchmark (= this PR's defaults)"),
    ("c1_warm_cache_eager", "control: warm conv cache only, eager (COSMOS3_COMPILE_VAE=0)"),
]
CONTROLS = {"stock_fp32", "c1_warm_cache_eager"}
REF, BEFORE, PR = "stock_bf16", "s0_mstar_before", "s4_cudnn_benchmark"
FFMPEG = imageio_ffmpeg.get_ffmpeg_exe()


def gaussian_window(device, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim_frames(a, b, win):
    """a, b: float [N, 3, H, W] on [0, 255] -> per-frame SSIM [N]."""
    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    mu_a = F.conv2d(a, win, groups=3)
    mu_b = F.conv2d(b, win, groups=3)
    s_aa = F.conv2d(a * a, win, groups=3) - mu_a**2
    s_bb = F.conv2d(b * b, win, groups=3) - mu_b**2
    s_ab = F.conv2d(a * b, win, groups=3) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean(dim=(1, 2, 3))


def compare(x, ref, win, batch=8):
    """x, ref: uint8 numpy [F, H, W, 3]."""
    n = x.shape[0]
    ge1 = ge2 = 0
    maxd = 0
    sse, ssim = [], []
    for i in range(0, n, batch):
        a = torch.from_numpy(x[i:i + batch]).cuda()
        b = torch.from_numpy(ref[i:i + batch]).cuda()
        d = (a.int() - b.int()).abs()
        dpx = d.amax(dim=-1)
        ge1 += int((dpx >= 1).sum())
        ge2 += int((dpx >= 2).sum())
        maxd = max(maxd, int(d.max()))
        sse.append((d.double() ** 2).sum(dim=(1, 2, 3)))
        af = a.permute(0, 3, 1, 2).float()
        bf = b.permute(0, 3, 1, 2).float()
        ssim.append(ssim_frames(af, bf, win))
    sse = torch.cat(sse).cpu().numpy()
    ssim = torch.cat(ssim).cpu().numpy()
    per = x[0].size
    mse = sse / per
    with np.errstate(divide="ignore"):
        psnr = np.where(mse > 0, 10 * np.log10(255.0**2 / np.maximum(mse, 1e-30)), np.inf)
    px = n * x.shape[1] * x.shape[2]
    finite = psnr[np.isfinite(psnr)]
    clip_mse = sse.sum() / (per * n)
    return {
        "pct_pixels_diff_ge1": 100 * ge1 / px, "pct_pixels_diff_ge2": 100 * ge2 / px, "max_abs_diff": maxd,
        "identical_frames": int((mse == 0).sum()),
        "psnr_clip_db": (10 * math.log10(255.0**2 / clip_mse)) if clip_mse > 0 else math.inf,
        "psnr_frame_mean_db": float(finite.mean()) if finite.size else math.inf,
        "psnr_worst_db": float(psnr.min()), "psnr_worst_frame": int(psnr.argmin()),
        "ssim_worst": float(ssim.min()), "ssim_worst_frame": int(ssim.argmin()),
    }


def ffmpeg_encode(frames_iter, w, h, out, fps=FPS, extra=()):
    cmd = [FFMPEG, "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{w}x{h}",
           "-r", str(fps), "-i", "-", *extra, str(out)]
    p = subprocess.Popen(cmd, stdin=subprocess.PIPE)
    for f in frames_iter:
        p.stdin.write(np.ascontiguousarray(f).tobytes())
    p.stdin.close()
    assert p.wait() == 0, cmd


H264 = ("-c:v", "libx264", "-preset", "slow", "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart")


def font(size):
    try:
        return ImageFont.truetype("DejaVuSans.ttf", size)
    except OSError:
        return ImageFont.load_default(size=size)


def labeled(img: np.ndarray, label: str, size: int) -> np.ndarray:
    bar = max(14, int(size * 1.6))
    canvas = Image.new("RGB", (img.shape[1], img.shape[0] + bar), (0, 0, 0))
    canvas.paste(Image.fromarray(img), (0, bar))
    ImageDraw.Draw(canvas).text((6, (bar - size) // 2), label, fill=(255, 255, 255), font=font(size))
    return np.asarray(canvas)


def resize(img: np.ndarray, w: int) -> np.ndarray:
    h = round(img.shape[0] * w / img.shape[1] / 2) * 2
    return np.asarray(Image.fromarray(img).resize((w, h), Image.LANCZOS))


LUT = (colormaps["inferno"](np.arange(256))[:, :3] * 255).astype(np.uint8)


def heatmap(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(axis=-1)
    return LUT[np.clip(d * 10, 0, 255).astype(np.uint8)]


PANELS = [("stock diffusers", REF), ("M* before", BEFORE), ("this PR", PR)]
HEAT = "|this PR - stock| x10"


def sbs_frame(fr, i, panel_w, label_size, grid=False):
    tiles = [labeled(resize(fr[v][i], panel_w), name, label_size) for name, v in PANELS]
    tiles.append(labeled(resize(heatmap(fr[PR][i], fr[REF][i]), panel_w), HEAT, label_size))
    if grid:
        return np.concatenate([np.concatenate(tiles[:2], 1), np.concatenate(tiles[2:], 1)], 0)
    return np.concatenate(tiles, 1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt-id", required=True)
    ap.add_argument("--gif-width", type=int, default=480)
    ap.add_argument("--gif-fps", type=int, default=8)
    ap.add_argument("--sheet-panel-width", type=int, default=320)
    ap.add_argument("--sbs-panel-width", type=int, default=480)
    ap.add_argument("--gif-colors", type=int, default=256)
    ap.add_argument("--crf", type=int, default=20)
    ap.add_argument("--step-start", type=int, default=37)
    ap.add_argument("--step-frames", type=int, default=24, help="0: whole clip")
    ap.add_argument("--media-only", action="store_true")
    args = ap.parse_args()
    run, out, pid = Path(args.run), Path(args.out), args.prompt_id
    out.mkdir(parents=True, exist_ok=True)

    fr = {v: np.load(run / f"{v}.u8.npy", mmap_mode="r") for v, _ in STEPS}
    meta = {v: json.loads((run / f"{v}.json").read_text()) for v, _ in STEPS}
    n, h, w, _ = fr[REF].shape
    if args.media_only:
        write_media(fr, out, pid, n, h, w, args)
        return
    win = gaussian_window("cuda")

    rows = []
    prev = None
    for v, label in STEPS:
        m = meta[v]
        md5 = hashlib.md5(np.ascontiguousarray(fr[v]).tobytes()).hexdigest()
        assert md5 == m["md5"], v
        row = {"prompt": pid, "step": v, "label": label, "md5_u8": md5,
               "identical_to_stock": md5 == meta[REF]["md5"],
               "identical_to_prev_step": (md5 == meta[prev]["md5"]) if prev and v not in CONTROLS else "",
               "same_md5_both_passes": md5 == m["md5_pass1"]}
        row |= compare(fr[v], fr[REF], win) if v != REF else {
            "pct_pixels_diff_ge1": 0.0, "pct_pixels_diff_ge2": 0.0, "max_abs_diff": 0, "identical_frames": n,
            "psnr_clip_db": math.inf, "psnr_frame_mean_db": math.inf, "psnr_worst_db": math.inf,
            "psnr_worst_frame": "", "ssim_worst": 1.0, "ssim_worst_frame": ""}
        row |= {"decode_total_ms": m["total_ms"], "steady_chunk_ms_median": m["steady_chunk_ms_median"],
                "steady_windows": m.get("steady_windows", ""),
                "first_window_ms": m["window_ms"][0] if len(m["window_ms"]) > 1 else "",
                "pass1_total_ms": m["pass1_total_ms"], "peak_alloc_gib": m["peak_alloc_gib"],
                "frames": n, "height": h, "width": w}
        for k, val in row.items():
            if isinstance(val, float) and math.isfinite(val):
                row[k] = round(val, 4)
        rows.append(row)
        print(json.dumps(row))
        prev = v if v not in CONTROLS else prev
    with open(out / f"metrics_{pid}.csv", "w", newline="") as f:
        wr = csv.DictWriter(f, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)

    write_media(fr, out, pid, n, h, w, args)


# The ladder's mp4s (the fp32 pipeline decode and the eager control get none).
MP4_STEPS = ["stock_bf16", "s0_mstar_before", "s1_warm_cache", "s2_channels_last", "s3_one_call",
             "s4_cudnn_benchmark"]


def write_media(fr, out, pid, n, h, w, args):
    # One mp4 per step, original resolution, frames [step_start, step_start + step_frames).
    seg = range(args.step_start, min(n, args.step_start + (args.step_frames or n)))
    crf = ("-crf", str(args.crf))
    h264 = [*H264[:4], *crf, *H264[6:]]
    for i, v in enumerate(MP4_STEPS):
        ffmpeg_encode((fr[v][t] for t in seg), w, h, out / f"{pid}_{i}_{v}.mp4", extra=h264)

    # Side-by-side: stock | M* before | this PR | x10 abs-diff heatmap, whole clip, panels downscaled.
    first = sbs_frame(fr, 0, args.sbs_panel_width, 18)
    ffmpeg_encode((sbs_frame(fr, t, args.sbs_panel_width, 18) for t in range(n)), first.shape[1], first.shape[0],
                  out / f"{pid}_side_by_side.mp4", extra=h264)

    # GIF preview: the same four panels as a 2x2 grid, args.gif_width wide.
    pw = args.gif_width // 2
    idx = sorted(set(np.round(np.arange(0, n, FPS / args.gif_fps)).astype(int).clip(0, n - 1)))
    g0 = sbs_frame(fr, 0, pw, 10, grid=True)
    ffmpeg_encode((sbs_frame(fr, t, pw, 10, grid=True) for t in idx), g0.shape[1], g0.shape[0],
                  out / f"{pid}_side_by_side.gif", fps=args.gif_fps,
                  extra=("-filter_complex", f"[0:v]split[a][b];[a]palettegen=max_colors={args.gif_colors}:"
                         "stats_mode=diff[p];[b][p]paletteuse=dither=bayer:bayer_scale=4:diff_mode=rectangle",
                         "-loop", "0"))

    # Contact sheet: first / middle / last frame, plus frame 49 (the first frame of chunk 4, where
    # M* before's stateless re-decode is furthest from stock), for stock, M* before and this PR,
    # then both x10 abs-diff heatmaps against stock.
    pw = args.sheet_panel_width
    cols = sorted({0, 49, n // 2, n - 1})
    rows_img = []
    for name, v in PANELS:
        rows_img.append(np.concatenate(
            [labeled(resize(np.asarray(fr[v][t]), pw), f"{name}, frame {t}", 13) for t in cols], 1))
    for name, v in (("|M* before - stock| x10", BEFORE), (HEAT, PR)):
        rows_img.append(np.concatenate(
            [labeled(resize(heatmap(fr[v][t], fr[REF][t]), pw), f"{name}, frame {t}", 13) for t in cols], 1))
    Image.fromarray(np.concatenate(rows_img, 0)).save(out / f"{pid}_contact_sheet.png", optimize=True)


if __name__ == "__main__":
    main()
