"""Wan VAE decode through M*'s own decoder nodes, one configuration per process.

``--node cosmos3_ar`` drives ``Cosmos3VAEDecoderARSubmodule.forward`` (the
streaming rollout decoder) over a latent stream cut into windows, exactly as the
window_decoder partition feeds it: each window re-pins ``--overlap`` latents of
the one before and emits only its new frames. Steady windows are timed one by
one (CUDA-synchronized wall time around ``forward``, which includes the uint8
conversion), after ``--warmup`` windows that absorb torch.compile and cuDNN
autotuning. ``--streams N`` interleaves N concurrent rollouts on the node.

``--node wan22`` times ``Wan22VaeDecoderSubmodule.forward`` on whole clips
(that node's only mode), reported per clip and per 12 output frames.

Configurations are config overrides (``--set key=value``, any field of the
node's config) on top of the checkout's defaults, so the same command measures
upstream main (point ``MSTAR`` at a main checkout, no overrides) and this
branch's switches one at a time. Presets (``--preset``) name the ladder:

  redecode        warm cache off, default layout: main before this change
  warm            + per-stream warm decoder cache
  warm_cl         + channels-last
  warm_cl_chunk   + a window's latents in one decoder call
  warm_cl_chunk_bench  + cudnn.benchmark
  default         no overrides (this checkout: warm + cl + one-call + bench)

Quality: ``--psnr-windows K`` decodes the stream's first K windows in fp32
(TF32 off, eager, default layout, one ``vae.decode`` of the whole prefix) and
reports the uint8 PSNR of the served frames against it.

  python bench_decode_nodes.py --node cosmos3_ar --vae cosmos3-edge --preset warm_cl --out res/
  MSTAR=~/mstar-main python bench_decode_nodes.py --node cosmos3_ar --preset default --tag main --out res/
  python bench_decode_nodes.py --node cosmos3_ar --vae wan21 --size 832x480 --window 5 --overlap 2 ...
  python bench_decode_nodes.py --node wan22 --size 1280x704 --frames 81 --set compile_vae_decode=true
"""

from __future__ import annotations

import argparse
import json
import math
import os
import statistics
import subprocess
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import torch

MSTAR = Path(os.environ.get("MSTAR", Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(MSTAR))

VAE_REPOS = {
    "cosmos3-edge": "nvidia/Cosmos3-Edge",
    "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers",
    "wan21": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers",
}
PRESETS = {
    "default": {},
    "redecode": {"windowed_decode_warm_cache": False, "vae_decode_channels_last": False},
    "warm": {"windowed_decode_warm_cache": True, "vae_decode_channels_last": False},
    "warm_cl": {"windowed_decode_warm_cache": True, "vae_decode_channels_last": True},
    "warm_cl_chunk": {"windowed_decode_warm_cache": True, "vae_decode_channels_last": True,
                      "windowed_decode_latents_per_call": 0},
    "warm_cl_chunk_bench": {"windowed_decode_warm_cache": True, "vae_decode_channels_last": True,
                            "windowed_decode_latents_per_call": 0, "vae_decode_cudnn_benchmark": True},
}


def parse_value(v: str):
    low = v.lower()
    if low in ("true", "false"):
        return low == "true"
    for cast in (int, float):
        try:
            return cast(v)
        except ValueError:
            pass
    return v


def load_vae(name: str, dtype=torch.float32):
    from diffusers import AutoencoderKLWan
    from huggingface_hub import snapshot_download

    path = Path(name) if Path(name).is_dir() else Path(
        snapshot_download(VAE_REPOS[name], allow_patterns=["vae/*"])) / "vae"
    vae = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=dtype)
    vae.encoder = vae.quant_conv = None
    return vae.cuda().eval()


def apply_overrides(cfg, overrides: dict) -> list[str]:
    """Set the overrides the checkout's config has; returns the ones it lacks
    (a preset's switch that this checkout predates), which are reported."""
    missing = [k for k in overrides if not hasattr(cfg, k)]
    for k, v in overrides.items():
        if k not in missing:
            setattr(cfg, k, v)
    if missing:
        print(f"warning: {type(cfg).__name__} in {MSTAR} has no {missing}; ignored", file=sys.stderr)
    return missing


def psnr_u8(a: torch.Tensor, b: torch.Tensor) -> float:
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def timed(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def latent_stream(args, z_dim, h, w, units):
    if args.latents:
        raw = torch.load(args.latents)["raw"][:, :, :units]
        assert raw.shape[2] == units and raw.shape[3:] == (h, w), f"latents {tuple(raw.shape)} too short or wrong size"
        return raw.float()
    gen = torch.Generator().manual_seed(args.seed)
    return torch.randn(1, z_dim, units, h, w, generator=gen)


def run_cosmos3_ar(args, overrides):
    from mstar.model.cosmos3 import constants as C
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule

    os.environ["COSMOS3_COMPILE_VAE"] = "0" if args.eager else "1"
    vae = load_vae(args.vae)
    cfg = Cosmos3Config()
    ignored = apply_overrides(cfg, overrides)
    sub = Cosmos3VAEDecoderARSubmodule(vae, cfg)
    width, height = (int(x) for x in args.size.split("x"))
    sf = vae.config.scale_factor_spatial
    stride = args.window - args.overlap
    n_windows = 1 + args.warmup + args.windows
    units = args.window + stride * (n_windows - 1)
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    raw = latent_stream(args, vae.config.z_dim, height // sf, width // sf, units)
    # What the DiT hands the decoder: normalized latents, bf16.
    latents = ((raw - mean) / std).to("cuda", torch.bfloat16)
    frames = 1 + (units - 1) * 4
    md = {"num_windows": n_windows, "overlap_latent_units": args.overlap, "num_frames": frames, "stream_video": True}
    rids = [f"s{i}" for i in range(args.streams)]
    infos = {r: SimpleNamespace(request_ids=[r], per_request_info={r: SimpleNamespace(step_metadata=md)}) for r in rids}
    lat, first_ms, cache_mib, outs = [], None, None, []
    with torch.inference_mode():
        for k in range(n_windows):
            win = latents[:, :, stride * k: stride * k + args.window]
            if k == 1 + args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            for r in rids:
                out, ms = timed(lambda r=r, win=win: sub.forward(C.VIDEO_DECODE_AR_WALK, infos[r], win))
                if r == rids[0]:
                    outs.append(out["video_output"][0] if k < args.psnr_windows else None)
                if k == 0 and r == rids[0]:
                    first_ms = ms
                elif k > args.warmup:
                    lat.append(ms)
            if k == n_windows - 2:
                cache = sub.request_state(rids[0]).get("ar_feat_cache")
                cache_mib = sum(t.numel() * t.element_size() for t in cache if torch.is_tensor(t)) / 2**20 \
                    if cache else 0.0
    peak = torch.cuda.max_memory_allocated() / 2**30
    new_frames = stride * 4
    res = {
        "window_latents": args.window, "overlap": args.overlap, "new_frames_per_window": new_frames,
        "streams": args.streams, "windows_timed": args.windows,
        "median_ms_per_window": statistics.median(lat), "p90_ms_per_window": pct(lat, 0.9),
        "median_ms_per_12_frames": statistics.median(lat) * 12 / new_frames,
        "decoded_fps": new_frames * 1e3 / statistics.median(lat),
        "first_window_ms": first_ms, "peak_allocated_gib": peak, "cache_mib_per_stream": cache_mib,
        "decode_dtype": str(sub._decode_dtype()), "compiled": not args.eager, "ignored_overrides": ignored,
    }
    if args.psnr_windows:
        del sub
        # The decoder's own input, de-normalized in fp32: the reference sees the
        # same bf16 latents, so the PSNR prices the decode alone.
        z = latents.float().cpu() * std + mean
        res["psnr_u8_vs_fp32"] = psnr_vs_fp32(args, z, outs[: args.psnr_windows], stride)
    return res


def psnr_vs_fp32(args, z, outs, stride):
    """uint8 PSNR of the served frames (first windows) against one fp32 decode of
    the whole prefix, TF32 off."""
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
    vae = load_vae(args.vae, torch.float32)
    units = args.window + stride * (len(outs) - 1)
    with torch.inference_mode():
        ref = vae.decode(z[:, :, :units].cuda()).sample
    ref = (ref / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)
    served = torch.cat(outs, dim=2)
    return psnr_u8(served.float(), ref.float())


def run_wan22(args, overrides):
    from mstar.model.wan22.config import Wan22Config
    from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule

    vae = load_vae(args.vae if args.vae != "cosmos3-edge" else "wan22")
    cfg = Wan22Config(vae_z_dim=vae.config.z_dim, vae_scale_factor_spatial=vae.config.scale_factor_spatial)
    ignored = apply_overrides(cfg, overrides)
    sub = Wan22VaeDecoderSubmodule(vae, cfg)
    width, height = (int(x) for x in args.size.split("x"))
    sf = vae.config.scale_factor_spatial
    units = (args.frames - 1) // 4 + 1
    mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
    std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
    latents = ((latent_stream(args, vae.config.z_dim, height // sf, width // sf, units) - mean) / std).cuda()
    lat = []
    with torch.inference_mode():
        _, first_ms = timed(lambda: sub.forward("video_gen", None, latents=latents))
        for _ in range(args.warmup):
            sub.forward("video_gen", None, latents=latents)
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.windows):
            out, ms = timed(lambda: sub.forward("video_gen", None, latents=latents))
            lat.append(ms)
    assert out["video_output"][0].shape[2] == args.frames
    return {
        "frames": args.frames, "clips_timed": args.windows, "median_ms_per_clip": statistics.median(lat),
        "p90_ms_per_clip": pct(lat, 0.9), "median_ms_per_12_frames": statistics.median(lat) * 12 / args.frames,
        "first_clip_ms": first_ms, "peak_allocated_gib": torch.cuda.max_memory_allocated() / 2**30,
        "decode_dtype": str(sub._decode_dtype()), "tiling": sub._resolve_tiling_mode(), "ignored_overrides": ignored,
    }


def env_info():
    info = {"torch": torch.__version__, "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name(0), "mstar": str(MSTAR)}
    try:
        import diffusers
        info["diffusers"] = diffusers.__version__
        info["mstar_commit"] = subprocess.run(["git", "-C", str(MSTAR), "rev-parse", "--short", "HEAD"],
                                              capture_output=True, text=True, check=True).stdout.strip()
        info["gpu_apps_at_start"] = subprocess.run(
            ["nvidia-smi", "--query-compute-apps=pid,used_memory", "--format=csv,noheader"],
            capture_output=True, text=True, check=True).stdout.strip().splitlines()
    except Exception:
        pass
    return info


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--node", choices=["cosmos3_ar", "wan22"], default="cosmos3_ar")
    ap.add_argument("--vae", default="cosmos3-edge", help="cosmos3-edge | wan22 | wan21 | a local vae/ dir")
    ap.add_argument("--preset", choices=sorted(PRESETS), default="default")
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="config override (repeatable)")
    ap.add_argument("--eager", action="store_true", help="COSMOS3_COMPILE_VAE=0 (cosmos3_ar)")
    ap.add_argument("--size", default="832x480", help="WIDTHxHEIGHT in pixels")
    ap.add_argument("--window", type=int, default=8, help="latents per window (served default 8 = 29 frames)")
    ap.add_argument("--overlap", type=int, default=2, help="latents a window re-pins (served default 2)")
    ap.add_argument("--frames", type=int, default=81, help="wan22: clip length")
    ap.add_argument("--warmup", type=int, default=5, help="untimed windows (clips) after the first")
    ap.add_argument("--windows", type=int, default=30, help="timed windows (clips)")
    ap.add_argument("--streams", type=int, default=1, help="cosmos3_ar: concurrent rollouts, interleaved")
    ap.add_argument("--psnr-windows", type=int, default=0)
    ap.add_argument("--latents", help="optional .pt with raw VAE latents under 'raw' (else random)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tag", default="", help="name of this run in the output (default: preset + overrides)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    overrides = dict(PRESETS[args.preset])
    for kv in args.set:
        k, _, v = kv.partition("=")
        overrides[k] = parse_value(v)
    tag = args.tag or "_".join([args.node, args.preset, *(f"{k}={v}" for k, v in sorted(overrides.items())
                                                          if k not in PRESETS[args.preset])]) + (
        "_eager" if args.eager else "")
    from mstar.engine import apply_torch_config

    apply_torch_config()  # the serving process's dynamo / precision settings
    t0 = time.time()
    run = run_cosmos3_ar if args.node == "cosmos3_ar" else run_wan22
    res = run(args, overrides)
    row = {"tag": tag, "node": args.node, "vae": args.vae, "size": args.size, "overrides": overrides,
           **{k: round(v, 3) if isinstance(v, float) else v for k, v in res.items()},
           "wall_s": round(time.time() - t0, 1), "env": env_info()}
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{tag}.json").write_text(json.dumps(row, indent=2))
    print(json.dumps({k: v for k, v in row.items() if k != "env"}), flush=True)


if __name__ == "__main__":
    main()
