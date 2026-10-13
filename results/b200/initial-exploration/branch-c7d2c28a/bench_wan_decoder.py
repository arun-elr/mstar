"""Wan VAE streaming-decode microbenchmark, one variant per process.

Every streaming variant replays the same latent stream: one first latent frame
(untimed), then ``--warmup`` chunks (untimed; absorbs torch.compile), then
``--chunks`` steady chunks of 3 latent frames timed with CUDA events around
decode + 8-bit conversion, synchronized per chunk. The stream is replayed
``--repeats`` times from a cold cache; samples from all repeats are pooled.

Variants (see PLAN.md for what each one maps to):
  ref_fp32            diffusers, eager, fp32, TF32 off, per-frame calls: exact reference
  diffusers_bf16      stock diffusers, eager, bf16, per-frame
  diffusers_bf16_compile  + torch.compile of the decoder (M*'s compile knob, streaming)
  mstar_ar_ctx{K}[_eager]  M*'s Cosmos3VAEDecoderSubmodule (Wan VAE, bf16 by the cuDNN gate,
                      torch.compile by default) driven the way Cosmos3VAEDecoderARSubmodule
                      drives it: re-decode K context latents + the 3 new ones from a cold
                      cache, keep the last 12 frames. K=8 is M*'s default.
  mstar_wan22_oneshot_{F}f[_tiled]  M*'s Wan22VaeDecoderSubmodule.forward on a whole
                      F-frame clip (its only mode); reported per 12 output frames.
  rx_cl               ladder step 1: channels-last weights + activations, eager, per-frame
  rx_cl_chunk         + one decoder call per chunk (ladder step 2), eager
  rx_cl_chunk_compile + torch.compile (the reference ladder's configuration)
  rx_full[_fp16]      + nearest upsample in the activation dtype (part of ladder step 3)
  rx_cl_compile       ablation: channels-last + compile, per-frame calls
  rx_chunk_compile    ablation: one call per chunk + compile, channels-first
  ref_fp32_tf32       control: ref_fp32 with TF32 allowed (shows the flag changes the convs)
  px_<flags>          beyond the reference ladder, on top of rx_full; flags joined by "_":
                        base   rx_full as is          cg     manual CUDA-graph replay per chunk
                        nopad  conv-internal H/W pad, channels-last enforced at every conv
                        bench  cudnn.benchmark        maxat  compile mode max-autotune-no-cudagraphs
                        ro     compile mode reduce-overhead (torch's CUDA graphs)   fp16

  python bench_wan_decoder.py --variant diffusers_bf16 --latents latents_wan21.pt --out res/
"""

import argparse
import json
import math
import os
import platform
import re
import statistics
import subprocess
import sys
import time
from pathlib import Path

import torch

from common import (
    FRAMES_PER_CHUNK, WAN21_REPO, WAN22_REPO, GraphedStreamingDecoder, StreamingDecoder,
    bf16_nearest_upsample, latent_stats, load_vae, mstar_path, patch_causal_conv_native_pad, to_uint8,
)

QUALITY_CHUNKS = 10  # PSNR over the first 10 steady chunks
REPO = WAN21_REPO    # --vae wan22 switches every variant to the Wan2.2 VAE M* serves


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def psnr(a: torch.Tensor, b: torch.Tensor, peak: float) -> float:
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def env_info():
    info = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
    }
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
    except Exception:
        pass
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        info["gpu"] = p.name
        info["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
        try:
            q = subprocess.run(
                ["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                 "--format=csv,noheader"], capture_output=True, text=True, check=True)
            info["nvidia_smi"] = q.stdout.strip()
            apps = subprocess.run(
                ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
                capture_output=True, text=True, check=True)
            info["gpu_apps_at_start"] = apps.stdout.strip().splitlines()
        except Exception:
            pass
    try:
        info["mstar_commit"] = subprocess.run(
            ["git", "-C", mstar_path(), "rev-parse", "--short", "HEAD"],
            capture_output=True, text=True, check=True).stdout.strip()
    except Exception:
        pass
    return info


class Timer:
    def __init__(self, device):
        self.cuda = device.startswith("cuda")

    def __enter__(self):
        if self.cuda:
            torch.cuda.synchronize()
            self.s = torch.cuda.Event(enable_timing=True)
            self.e = torch.cuda.Event(enable_timing=True)
            self.s.record()
        else:
            self.t0 = time.perf_counter()
        return self

    def __exit__(self, *exc):
        if self.cuda:
            self.e.record()
            self.e.synchronize()
            self.ms = self.s.elapsed_time(self.e)
        else:
            self.ms = (time.perf_counter() - self.t0) * 1e3


def reset_peak(device):
    if device.startswith("cuda"):
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()


def peak(device):
    if not device.startswith("cuda"):
        return {}
    return {
        "peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
        "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2),
    }


# --------------------------------------------------------------------------- variants

def build_streaming(variant, device):
    """(StreamingDecoder, dtype) for a streaming variant, or None if not one."""
    if variant in ("ref_fp32", "ref_fp32_tf32"):
        tf32 = variant == "ref_fp32_tf32"
        torch.backends.cuda.matmul.allow_tf32 = tf32
        torch.backends.cudnn.allow_tf32 = tf32
        vae = load_vae(REPO, torch.float32, device)
        sd = StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=False)
        return sd, torch.float32
    m = re.fullmatch(r"diffusers_(bf16|fp16)(_compile)?", variant)
    if m:
        dtype = torch.bfloat16 if m[1] == "bf16" else torch.float16
        vae = load_vae(REPO, dtype, device)
        sd = StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=bool(m[2]))
        return sd, dtype
    m = re.fullmatch(r"rx_(cl|cl_chunk|cl_chunk_compile|full|full_fp16|cl_compile|chunk_compile)", variant)
    if m:
        kind = m[1]
        dtype = torch.float16 if kind == "full_fp16" else torch.bfloat16
        if kind.startswith("full"):
            bf16_nearest_upsample()
        vae = load_vae(REPO, dtype, device)
        sd = StreamingDecoder(
            vae, per_frame=kind in ("cl", "cl_compile"), channels_last=kind != "chunk_compile",
            compile_decoder=kind not in ("cl", "cl_chunk"),
        )
        return sd, dtype
    m = re.fullmatch(r"px_([a-z0-9_]+)", variant)
    if m:
        flags = set(m[1].split("_"))
        unknown = flags - {"base", "cg", "nopad", "bench", "maxat", "ro", "fp16"}
        assert not unknown, f"unknown px flags {unknown}"
        dtype = torch.float16 if "fp16" in flags else torch.bfloat16
        bf16_nearest_upsample()
        if "nopad" in flags:
            patch_causal_conv_native_pad()
        if "bench" in flags:
            torch.backends.cudnn.benchmark = True
        mode = ("max-autotune-no-cudagraphs" if "maxat" in flags
                else "reduce-overhead" if "ro" in flags else None)
        vae = load_vae(REPO, dtype, device)
        sd = StreamingDecoder(vae, per_frame=False, channels_last=True, compile_decoder=True,
                              compile_mode=mode)
        if "cg" in flags:
            sd = GraphedStreamingDecoder(sd)
        return sd, dtype
    return None


def precision_flags():
    """TF32 / precision state as the run actually saw it."""
    out = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    for name, get in (("cudnn_conv_fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision),
                      ("cuda_matmul_fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision)):
        try:
            out[name] = get()
        except Exception as exc:  # API absent or mixed-API error
            out[name] = f"n/a ({type(exc).__name__})"
    return out


def run_streaming(args, raw, ref):
    built = build_streaming(args.variant, args.device)
    if built is None:
        sys.exit(f"unknown variant {args.variant}")
    sd, dtype = built
    z = raw.to(args.device, dtype)
    n_need = 1 + 3 * (args.warmup + args.chunks)
    assert z.shape[2] >= n_need, f"latent stream has {z.shape[2]} frames, need {n_need}"
    lat, quality_f, quality_u8 = [], [], []
    out_ref = [] if args.save_ref else None
    mem = {}
    t_compile = None
    with torch.inference_mode():
        for rep in range(args.repeats):
            sd.reset()
            sd.first(z[:, :, :1])
            for c in range(args.warmup + args.chunks):
                zc = z[:, :, 1 + 3 * c: 4 + 3 * c]
                if rep == 0 and c == args.warmup:
                    reset_peak(args.device)
                with Timer(args.device) as t:
                    out = sd.chunk(zc)
                    u8 = to_uint8(out)
                if rep == 0 and c == 0:
                    t_compile = t.ms
                if c < args.warmup:
                    continue
                lat.append(t.ms)
                k = c - args.warmup
                if rep == 0 and k < QUALITY_CHUNKS:
                    if out_ref is not None:
                        out_ref.append(out.float().cpu())
                    if ref is not None:
                        quality_f.append(psnr(out.float().cpu(), ref["float"][k], 2.0))
                        quality_u8.append(psnr(u8.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
            if rep == 0:
                mem = peak(args.device)
    if out_ref is not None:
        torch.save({"float": out_ref}, args.save_ref)
    if args.layout_check:
        mem["conv_inputs_not_channels_last"] = layout_check(sd, z)
    return lat, quality_f, quality_u8, mem, t_compile


def layout_check(sd, z):
    """Count conv kernel calls (F.conv3d / F.conv2d, i.e. after any padding/concat)
    whose input is not channels-last, over one steady chunk run eagerly."""
    import torch.nn.functional as F

    bad = {"n": 0, "total": 0}
    orig = {"conv3d": F.conv3d, "conv2d": F.conv2d}

    def wrap(name, fmt):
        def conv(x, *a, **k):
            bad["total"] += 1
            if not x.is_contiguous(memory_format=fmt):
                bad["n"] += 1
            return orig[name](x, *a, **k)
        return conv

    if isinstance(sd, GraphedStreamingDecoder):
        sd = sd.sd
    step = sd.step
    sd.step = sd.decoder
    F.conv3d = wrap("conv3d", torch.channels_last_3d)
    F.conv2d = wrap("conv2d", torch.channels_last)
    try:
        sd.reset()
        with torch.inference_mode():
            sd.first(z[:, :, :1])
            sd.chunk(z[:, :, 1:4])
            bad["n"] = bad["total"] = 0
            sd.chunk(z[:, :, 4:7])
    finally:
        sd.step = step
        F.conv3d, F.conv2d = orig["conv3d"], orig["conv2d"]
    return f"{bad['n']}/{bad['total']}"


def run_mstar_ar(args, raw, ref, ctx, compiled):
    """M*'s streaming Wan decode: Cosmos3VAEDecoderARSubmodule's per-window logic,
    using the real Cosmos3VAEDecoderSubmodule._decode_pixels (normalization, dtype
    gate, compiled decode, first-call warm, uint8)."""
    sys.path.insert(0, mstar_path())
    os.environ["COSMOS3_COMPILE_VAE"] = "1" if compiled else "0"
    from types import SimpleNamespace
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

    vae = load_vae(REPO, torch.float32, args.device)
    sub = Cosmos3VAEDecoderSubmodule(vae=vae, config=SimpleNamespace())
    mean, std = latent_stats(vae, "cpu")
    norm = ((raw - mean) / std).to(args.device)  # what the DiT hands the decoder node
    lat, quality_u8 = [], []
    mem = {}
    t_compile = None
    keep = FRAMES_PER_CHUNK
    with torch.inference_mode():
        for rep in range(args.repeats):
            sub._decode_pixels(norm[:, :, :1])  # the stream's first window, untimed
            tail = norm[:, :, :1]
            for c in range(args.warmup + args.chunks):
                new = norm[:, :, 1 + 3 * c: 4 + 3 * c]
                if rep == 0 and c == args.warmup:
                    reset_peak(args.device)
                with Timer(args.device) as t:
                    stream_tail = torch.cat([tail, new.to(tail.dtype)], dim=2)
                    pixels = sub._decode_pixels(stream_tail)[:, :, -keep:]
                if rep == 0 and c == 0:
                    t_compile = t.ms
                tail = stream_tail[:, :, -ctx:]
                if c < args.warmup:
                    continue
                lat.append(t.ms)
                k = c - args.warmup
                if rep == 0 and k < QUALITY_CHUNKS and ref is not None:
                    quality_u8.append(psnr(pixels.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
            if rep == 0:
                mem = peak(args.device)
    mem["decode_dtype"] = str(sub._decode_dtype())
    mem["steady_decode_latents"] = ctx + 3
    return lat, [], quality_u8, mem, t_compile


def run_mstar_wan22_oneshot(args, raw, frames, tiled):
    """M*'s Wan22VaeDecoderSubmodule.forward (eager, cuDNN-gated dtype, VRAM-gated
    tiling) on a whole clip — the only way that node decodes."""
    sys.path.insert(0, mstar_path())
    os.environ["WAN22_VAE_DECODE_TILING"] = "tiled" if tiled else "auto"
    from mstar.model.wan22.config import Wan22Config
    from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule

    assert (frames - 1) % 4 == 0
    n_lat = (frames - 1) // 4 + 1
    vae = load_vae(REPO, torch.float32, args.device)
    cfg = Wan22Config(vae_z_dim=vae.config.z_dim, vae_scale_factor_spatial=vae.config.scale_factor_spatial)
    sub = Wan22VaeDecoderSubmodule(vae, cfg)
    mean, std = latent_stats(vae, "cpu")
    norm = ((raw[:, :, :n_lat] - mean) / std).to(args.device)
    lat = []
    with torch.inference_mode():
        sub.forward("video_gen", engine_inputs=None, latents=norm)  # warm cuDNN autotune
        reset_peak(args.device)
        for _ in range(args.oneshot_repeats):
            with Timer(args.device) as t:
                video = sub.forward("video_gen", engine_inputs=None, latents=norm)["video_output"][0]
            lat.append(t.ms)
    assert video.shape[2] == frames, video.shape
    mem = peak(args.device)
    mem["decode_dtype"] = str(sub._decode_dtype())
    mem["clip_frames"] = frames
    mem["clip_ms_median"] = statistics.median(lat)
    per12 = [x * FRAMES_PER_CHUNK / frames for x in lat]
    return per12, [], [], mem, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ref", help="ref_fp32 outputs (.pt) for PSNR")
    ap.add_argument("--save-ref", help="(ref_fp32 only) write first steady chunks' float outputs here")
    ap.add_argument("--out", required=True, help="results dir")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=75)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--oneshot-repeats", type=int, default=5)
    ap.add_argument("--layout-check", action="store_true")
    ap.add_argument("--profile", action="store_true", help="also write a torch.profiler kernel table")
    args = ap.parse_args()

    global REPO
    REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    torch._dynamo.config.cache_size_limit = 64
    if not args.device.startswith("cuda"):  # CPU smoke runs: the Wan22 node reads free VRAM
        torch.cuda.mem_get_info = lambda *a, **k: (1 << 40, 1 << 40)
    raw = torch.load(args.latents)["raw"]
    ref = torch.load(args.ref) if args.ref and Path(args.ref).exists() else None
    v = args.variant
    t0 = time.time()
    if m := re.fullmatch(r"mstar_ar_ctx(\d+)(_eager)?", v):
        res = run_mstar_ar(args, raw, ref, int(m[1]), compiled=not m[2])
    elif m := re.fullmatch(r"mstar_wan22_oneshot_(\d+)f(_tiled)?", v):
        res = run_mstar_wan22_oneshot(args, raw, int(m[1]), tiled=bool(m[2]))
    else:
        res = run_streaming(args, raw, ref)
    lat, qf, qu8, extra, t_first = res

    if args.profile and args.device.startswith("cuda"):
        extra["profile"] = str(profile_variant(args, raw))

    med = statistics.median(lat)
    row = {
        "variant": v,
        "n_samples": len(lat),
        "median_ms": round(med, 2),
        "p90_ms": round(pct(lat, 0.9), 2),
        "p10_ms": round(pct(lat, 0.1), 2),
        "mean_ms": round(statistics.fmean(lat), 2),
        "decoder_fps": round(FRAMES_PER_CHUNK / (med / 1e3), 1),
        "psnr_float_db_vs_fp32": round(statistics.fmean(qf), 2) if qf else None,
        "psnr_float_db_vs_fp32_peak1": round(statistics.fmean(qf) - 20 * math.log10(2), 2) if qf else None,
        "psnr_u8_db_vs_fp32": round(statistics.fmean([q for q in qu8 if math.isfinite(q)] or [float("inf")]), 2) if qu8 else None,
        "first_timed_call_ms": round(t_first, 1) if t_first else None,
        "wall_s": round(time.time() - t0, 1),
        **extra,
        "precision_flags": precision_flags(),
        "env": env_info(),
        "latents": str(args.latents),
        "vae": REPO,
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{v}.json").write_text(json.dumps(row, indent=2))
    (out / f"{v}.latencies.json").write_text(json.dumps(lat))
    print(json.dumps({k: row[k] for k in row if k not in ("env",)}))


def profile_variant(args, raw):
    """Kernel-time table over 6 steady chunks of a fresh streaming decoder (streaming
    variants only), grouped by kernel name."""
    from torch.profiler import ProfilerActivity, profile

    built = build_streaming(args.variant, args.device)
    if built is None:
        return "n/a"
    sd, dtype = built
    z = raw.to(args.device, dtype)
    with torch.inference_mode():
        sd.first(z[:, :, :1])
        for c in range(args.warmup):
            sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c])
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA]) as prof:
            for c in range(args.warmup, args.warmup + 6):
                to_uint8(sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]))
            torch.cuda.synchronize()
    path = Path(args.out) / f"{args.variant}.profile.txt"
    path.write_text(prof.key_averages().table(sort_by="cuda_time_total", row_limit=60))
    return path


if __name__ == "__main__":
    main()
