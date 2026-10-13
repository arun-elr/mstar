"""Where the decode time goes (by kernel category), one variant per process.

  python profile_kernels.py --variant rx_full --latents L.pt --out DIR      # kernel profile
  python profile_kernels.py --flops --latents L.pt --out DIR                # FLOP / byte count

Kernel profile: warm the stream (``--warmup`` chunks; absorbs compile and CUDA-graph
capture), then torch.profiler over ``--chunks`` steady chunks. Writes
``<variant>.kernels.json``: kernels per chunk, GPU time per category, top-15
kernels. Categories are assigned from kernel names (heuristic; see ``category``).

FLOP / byte count (eager, unpatched decoder, one steady 3-latent chunk):
FlopCounterMode FLOPs, plus the minimum conv traffic a roofline uses (each
conv's padded input, output and weights read/written once). Also the FLOPs M*'s
streaming decode actually executes per chunk (cold re-decode of ctx + 3 latents).
"""

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

import torch

import bench_wan_decoder as B
from common import (
    WAN21_REPO, WAN22_REPO, StreamingDecoder, latent_stats, load_vae, mstar_path, to_uint8,
)

CATEGORIES = [
    "convolution", "layout conversion", "norm / SiLU", "concat / pad", "upsample",
    "attention", "copies", "other elementwise", "other",
]


def category(name: str) -> str:
    n = name
    if re.search(r"nchwToNhwc|nhwcToNchw|nchw2nhwc|nhwc2nchw|convertTensor|transpose_readWrite", n):
        return "layout conversion"
    if n.startswith("triton_"):
        kind = n.split("_")[1]
        if re.search(r"_unsafe_index|upsample", n):
            return "upsample"
        if kind in ("per", "red"):
            return "norm / SiLU"  # Inductor reductions here are the RMS norms (+ fused SiLU/pad/cat)
        if re.search(r"constant_pad|_cat_|cat_|_cat$", n):
            return "concat / pad"
        return "other elementwise"
    if re.search(r"fmha|flash|attention|softmax", n, re.I):
        return "attention"
    if re.search(r"conv|implicit_gemm|xmma|fprop|cutlass|sm\d+_.*tensorop|cudnn", n, re.I):
        return "convolution"
    if re.search(r"upsample", n):
        return "upsample"
    if re.search(r"silu|NormTwo|norm_kernel|normalize|clamp_min|div_true|rsqrt|reduce_kernel", n):
        return "norm / SiLU"
    if re.search(r"CatArrayBatchedCopy|constant_pad|ConstantPad|FillFunctor|fill_kernel", n):
        return "concat / pad"
    if re.search(r"copy|Memcpy|Memset", n, re.I):
        return "copies"
    if re.search(r"elementwise", n):
        return "other elementwise"
    return "other"


def make_runner(variant, raw, device):
    """(start, chunk) closures replaying the latent stream for one variant."""
    m = re.fullmatch(r"mstar_ar_ctx(\d+)(_eager)?", variant)
    if m:
        import os
        from types import SimpleNamespace

        sys.path.insert(0, mstar_path())
        os.environ["COSMOS3_COMPILE_VAE"] = "0" if m[2] else "1"
        from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

        ctx = int(m[1])
        vae = load_vae(B.REPO, torch.float32, device)
        sub = Cosmos3VAEDecoderSubmodule(vae=vae, config=SimpleNamespace())
        mean, std = latent_stats(vae, "cpu")
        norm = ((raw - mean) / std).to(device)
        st = {}

        def start():
            sub._decode_pixels(norm[:, :, :1])
            st["tail"] = norm[:, :, :1]

        def chunk(c):
            stream_tail = torch.cat([st["tail"], norm[:, :, 1 + 3 * c: 4 + 3 * c]], dim=2)
            pixels = sub._decode_pixels(stream_tail)[:, :, -12:]
            st["tail"] = stream_tail[:, :, -ctx:]
            return pixels

        return start, chunk

    built = B.build_streaming(variant, device)
    assert built is not None, f"unknown variant {variant}"
    sd, dtype = built
    z = raw.to(device, dtype)

    def start():
        sd.reset()
        sd.first(z[:, :, :1])

    def chunk(c):
        return to_uint8(sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]))

    return start, chunk


def profile_variant(args, raw):
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity, profile

    start, chunk = make_runner(args.variant, raw, args.device)
    with torch.inference_mode():
        start()
        for c in range(args.warmup):
            chunk(c)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
            for c in range(args.warmup, args.warmup + args.chunks):
                chunk(c)
            torch.cuda.synchronize()
    per_name = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type != DeviceType.CUDA:
            continue
        rec = per_name[e.name]
        rec[0] += 1
        rec[1] += e.device_time_total / 1e3  # ms
    n = args.chunks
    cats = {c: 0.0 for c in CATEGORIES}
    for name, (cnt, ms) in per_name.items():
        cats[category(name)] += ms / n
    top = sorted(per_name.items(), key=lambda kv: -kv[1][1])[:15]
    result = {
        "variant": args.variant,
        "vae": B.REPO,
        "profiled_chunks": n,
        "kernels_per_chunk": round(sum(c for c, _ in per_name.values()) / n, 1),
        "distinct_kernels": len(per_name),
        "gpu_ms_per_chunk": round(sum(ms for _, ms in per_name.values()) / n, 2),
        "category_ms_per_chunk": {k: round(v, 2) for k, v in cats.items()},
        "top15": [
            {"name": name[:160], "category": category(name), "calls_per_chunk": round(cnt / n, 2),
             "ms_per_chunk": round(ms / n, 3)}
            for name, (cnt, ms) in top
        ],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.variant}.kernels.json").write_text(json.dumps(result, indent=2))
    if args.trace:
        prof.export_chrome_trace(str(out / f"{args.variant}.trace.json"))
    print(json.dumps({k: result[k] for k in ("variant", "kernels_per_chunk", "gpu_ms_per_chunk",
                                             "category_ms_per_chunk")}))


def count_flops(args, raw):
    import torch.nn.functional as F
    from torch.utils.flop_counter import FlopCounterMode

    vae = load_vae(B.REPO, torch.bfloat16, args.device)
    z = raw.to(args.device, torch.bfloat16)
    traffic = {"bytes": 0, "conv_calls": 0}
    orig = {"conv3d": F.conv3d, "conv2d": F.conv2d}

    def wrap(name):
        def conv(x, w, *a, **k):
            out = orig[name](x, w, *a, **k)
            traffic["bytes"] += (x.numel() + out.numel() + w.numel()) * x.element_size()
            traffic["conv_calls"] += 1
            return out
        return conv

    def flops_of(fn):
        with FlopCounterMode(display=False) as fc:
            fn()
        counts = fc.get_flop_counts().get("Global", {})
        conv = sum(v for k, v in counts.items() if "convolution" in str(k))
        return fc.get_total_flops(), conv

    with torch.inference_mode():
        sd = StreamingDecoder(vae, per_frame=False, channels_last=True, compile_decoder=False)
        sd.first(z[:, :, :1])
        for c in range(3):
            sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c])
        zc = z[:, :, 10:13]
        F.conv3d, F.conv2d = wrap("conv3d"), wrap("conv2d")
        try:
            sd_copy_feat = list(sd.feat)
            total, conv = flops_of(lambda: sd.chunk(zc, list(sd_copy_feat)))
            traffic["bytes"] = 0
            traffic["conv_calls"] = 0
            sd.chunk(zc, list(sd_copy_feat))
        finally:
            F.conv3d, F.conv2d = orig["conv3d"], orig["conv2d"]

        res = {"vae": B.REPO, "chunk_flops": total, "chunk_conv_flops": conv,
               "chunk_min_conv_bytes": traffic["bytes"], "conv_calls_per_chunk": traffic["conv_calls"]}
        for ctx in (8, 4, 2, 1):
            n_lat = ctx + 3
            cold = StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=False)
            seg = z[:, :, 20:20 + n_lat]
            f_first, _ = flops_of(lambda: cold.first(seg[:, :, :1]))
            f_rest, _ = flops_of(lambda: cold.chunk(seg[:, :, 1:]))
            res[f"mstar_ar_ctx{ctx}_executed_flops"] = f_first + f_rest
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"flops_{'wan21' if B.REPO == WAN21_REPO else 'wan22'}.json").write_text(json.dumps(res, indent=2))
    print(json.dumps(res))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant")
    ap.add_argument("--flops", action="store_true")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=6)
    ap.add_argument("--trace", action="store_true", help="also export a chrome trace (large)")
    args = ap.parse_args()
    B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    torch._dynamo.config.cache_size_limit = 64
    raw = torch.load(args.latents)["raw"]
    if args.flops:
        count_flops(args, raw)
    else:
        profile_variant(args, raw)


if __name__ == "__main__":
    main()
