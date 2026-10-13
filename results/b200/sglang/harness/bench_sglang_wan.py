"""SGLang (sglang.multimodal_gen) Wan VAE streaming decode under M*'s benchmark
methodology (benchmark/wan_vae_decoder/bench_wan_decoder.py). One variant per process.

Code path (``--path stage``, default): the VAE is loaded by SGLang's own
``VAELoader`` under the pipeline config SGLang registers for the model
(``WanT2V480PConfig`` / ``Wan2_2_TI2V_5B_Config``), so its load-time switches
apply (channels-last-3D Conv3d weights, quality-gated CUDA fast-path wrappers).
It is then cast to the decode dtype the decode stage's residency gives it
(``resolve_decode_precision``). Every chunk goes through
``DecodingStage.decode`` (scale/shift of the latents, autocast, SGLang's
``torch.compile`` wrapper when ``--compile``, conversion to [0, 1]) under
``use_vae_fast_path(vae, quality_allows_kernel_fusions(quality))`` as
``DecodingStage.forward`` does, with the VAE decode function pointed at
``AutoencoderKLWan.causal_decode``: SGLang's streaming decode that keeps the
causal conv cache warm across calls (one latent frame per decoder call).

``--path realtime``: ``CausalVaeDecodingStage.decode_causal`` as is (SGLang's
realtime streaming stage, used by the LingBot-World causal pipeline): VAE in
``vae_precision``, no compile, no quality gate.

Stream and timing are M*'s: the first latent frame untimed, ``--warmup`` chunks
untimed, ``--chunks`` timed chunks of 3 latent frames, CUDA events around
decode + 8-bit conversion, synchronized per chunk, ``--repeats`` replays from a
cold cache pooled. PSNR over the first 10 steady chunks against M*'s fp32
reference (diffusers fp32, eager, TF32 off).

  python bench_sglang_wan.py --variant sglang_default --vae wan21 \
      --latents inputs/wan21/latents.pt --ref inputs/wan21/ref_fp32_outputs.pt --out res/
"""

import argparse
import hashlib
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from collections import defaultdict
from pathlib import Path

import torch

REPOS = {"wan21": "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "wan22": "Wan-AI/Wan2.2-TI2V-5B-Diffusers"}
REVISIONS = {"wan21": "0fad780a534b6463e45facd96134c9f345acfa5b",
             "wan22": "b8fff7315c768468a5333511427288870b2e9635"}
SGLANG_COMMIT = "e00930c5489053f26d86b179cee0d087f846acbb"  # tag v0.5.21
FRAMES_PER_CHUNK = 12
QUALITY_CHUNKS = 10


def pct(xs, q):
    xs = sorted(xs)
    k = (len(xs) - 1) * q
    lo, hi = math.floor(k), math.ceil(k)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


def psnr(a, b, peak):
    mse = torch.mean((a.double() - b.double()) ** 2).item()
    return float("inf") if mse == 0 else 10 * math.log10(peak * peak / mse)


def to_uint8(video):
    """[-1, 1] float -> uint8, M*'s formula (for the reference)."""
    return (video / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)


class Timer:
    def __enter__(self):
        torch.cuda.synchronize()
        self.s = torch.cuda.Event(enable_timing=True)
        self.e = torch.cuda.Event(enable_timing=True)
        self.s.record()
        return self

    def __exit__(self, *exc):
        self.e.record()
        self.e.synchronize()
        self.ms = self.s.elapsed_time(self.e)


def precision_flags():
    out = {
        "cuda_matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
        "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
    }
    for name, get in (("cudnn_conv_fp32_precision", lambda: torch.backends.cudnn.conv.fp32_precision),
                      ("cuda_matmul_fp32_precision", lambda: torch.backends.cuda.matmul.fp32_precision)):
        try:
            out[name] = get()
        except Exception as exc:
            out[name] = f"n/a ({type(exc).__name__})"
    return out


def env_info():
    import sglang
    import triton

    info = {
        "torch": torch.__version__, "cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "python": platform.python_version(),
        "triton": triton.__version__, "sglang": sglang.__version__,
        "sglang_commit": SGLANG_COMMIT,
        "dynamo_recompile_limit": torch._dynamo.config.recompile_limit,
    }
    p = torch.cuda.get_device_properties(0)
    info["gpu"] = p.name
    info["gpu_mem_gib"] = round(p.total_memory / 2**30, 1)
    try:
        q = subprocess.run(["nvidia-smi", "--query-gpu=driver_version,clocks.max.sm,power.limit",
                            "--format=csv,noheader"], capture_output=True, text=True, check=True)
        info["nvidia_smi"] = q.stdout.strip()
        apps = subprocess.run(["nvidia-smi", "--query-compute-apps=pid,used_memory",
                               "--format=csv,noheader"], capture_output=True, text=True, check=True)
        info["gpu_apps_at_start"] = apps.stdout.strip().splitlines()
    except Exception:
        pass
    root = Path(sglang.__file__).parent
    info["source_md5"] = {
        rel: hashlib.md5((root / rel).read_bytes()).hexdigest()
        for rel in ("multimodal_gen/runtime/models/vaes/wanvae.py",
                    "multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py",
                    "multimodal_gen/runtime/pipelines_core/stages/decoding.py",
                    "multimodal_gen/runtime/pipelines_core/stages/realtime/vae.py",
                    "kernels/ops/diffusion/layout/wan_causal_cache_triton.py")
    }
    return info


def build(args):
    """(start(z1), chunk(zc) -> [0, 1] float video, vae, sa, settings)."""
    if args.compile_mode:
        os.environ["SGLANG_VAE_TORCH_COMPILE_MODE"] = args.compile_mode
    from huggingface_hub import snapshot_download

    from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows_kernel_fusions
    from sglang.multimodal_gen.runtime.distributed import get_local_torch_device
    from sglang.multimodal_gen.runtime.loader.component_loaders.vae_loader import VAELoader
    from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
    from sglang.multimodal_gen.runtime.server_args import ServerArgs, set_global_server_args
    from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision
    from sglang.multimodal_gen.test.single_test_file.component_accuracy.utils import (
        initialize_parallel_runtime,
    )
    import sglang.multimodal_gen.runtime.layers.lora.linear  # noqa: F401  serving import; sets dynamo recompile_limit=64

    repo = REPOS[args.vae]
    sa = ServerArgs.from_kwargs(model_path=repo, num_gpus=1, enable_torch_compile=args.compile,
                                log_level="warning")
    pc = sa.pipeline_config
    if args.decode_precision:
        pc.vae_decode_precision = args.decode_precision  # --vae-decode-precision
    set_global_server_args(sa)
    initialize_parallel_runtime(sa)
    snap = snapshot_download(repo, revision=REVISIONS[args.vae], allow_patterns=["vae/*", "model_index.json"],
                             local_files_only=True)
    vae = VAELoader().load_customized(os.path.join(snap, "vae"), sa, "vae")
    device = get_local_torch_device()
    settings = {
        "path": args.path, "pipeline_config": type(pc).__name__, "quality": args.quality,
        "enable_torch_compile": sa.enable_torch_compile,
        "vae_torch_compile_mode": os.environ.get("SGLANG_VAE_TORCH_COMPILE_MODE", "default"),
        "vae_precision": pc.vae_precision, "vae_decode_precision": pc.vae_decode_precision,
        "disable_autocast": sa.disable_autocast, "vae_tiling": pc.vae_tiling,
        "use_feature_cache": pc.vae_config.use_feature_cache,
        "channels_last_3d_conv3d_weights": sum(
            1 for m in vae.modules() if isinstance(m, torch.nn.Conv3d)
            and m.weight.is_contiguous(memory_format=torch.channels_last_3d)),
        "conv3d_modules": sum(1 for m in vae.modules() if isinstance(m, torch.nn.Conv3d)),
        "fast_path_installed": any(type(m).__name__ == "FusedWanRMSNormSiLU" for m in vae.modules()),
    }

    if args.path == "realtime":
        from sglang.multimodal_gen.runtime.pipelines_core.stages.realtime.vae import CausalVaeDecodingStage

        stage = CausalVaeDecodingStage(vae)
        settings["decode_dtype"] = pc.vae_precision
        settings["autocast"] = pc.vae_precision != "fp32" and not sa.disable_autocast

        def start(z1):
            stage._get_causal_decode_reset_fn()()
            return stage.decode_causal(z1, sa, first_chunk=True)

        def chunk(zc):
            return stage.decode_causal(zc, sa, first_chunk=False)

        return start, chunk, stage.vae, sa, settings

    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
    from sglang.multimodal_gen.runtime.utils.precision import autocast_enabled

    vae_dtype = resolve_decode_precision(sa, "vae", quality=args.quality)
    vae = vae.to(device=device, dtype=vae_dtype).eval()  # decode-stage residency (ComponentUse.target_dtype)
    vae.decode = vae.causal_decode  # stream: warm conv cache across DecodingStage.decode calls
    stage = DecodingStage(vae)
    fusions = quality_allows_kernel_fusions(args.quality)
    settings["decode_dtype"] = str(vae_dtype).removeprefix("torch.")
    settings["autocast"] = autocast_enabled(vae_dtype, sa.disable_autocast)
    settings["quality_gate_kernel_fusions"] = fusions

    def call(z):
        with use_vae_fast_path(vae, fusions):
            return stage.decode(z, sa, vae_dtype=vae_dtype)

    def start(z1):
        vae.reset_causal_decode_state()
        return call(z1)

    return start, call, vae, sa, settings


def normalized_latents(raw, vae, sa, device):
    """What the DiT hands the decode stage: (raw - mean) / std, fp32 on the GPU.
    Checks that SGLang's scale_and_shift maps it back to the raw latents."""
    cfg = sa.pipeline_config.vae_config.arch_config
    z = cfg.z_dim
    mean = torch.tensor(cfg.latents_mean, dtype=torch.float32).view(1, z, 1, 1, 1)
    std = torch.tensor(cfg.latents_std, dtype=torch.float32).view(1, z, 1, 1, 1)
    norm = ((raw - mean) / std).to(device)
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import scale_and_shift_latents

    back = scale_and_shift_latents(norm[:, :, :4], sa, vae)
    err = (back.cpu() - raw[:, :, :4]).abs().max().item()
    assert err < 1e-3, f"scale_and_shift does not invert the normalization (max err {err})"
    return norm


def run(args, raw, ref):
    start, chunk, vae, sa, settings = build(args)
    device = "cuda"
    z = normalized_latents(raw, vae, sa, device)
    n_need = 1 + 3 * (args.warmup + args.chunks)
    assert z.shape[2] >= n_need, f"latent stream has {z.shape[2]} frames, need {n_need}"
    lat, qf, qu8 = [], [], []
    mem, t_first = {}, None
    for rep in range(args.repeats):
        start(z[:, :, :1])
        for c in range(args.warmup + args.chunks):
            zc = z[:, :, 1 + 3 * c: 4 + 3 * c]
            if rep == 0 and c == args.warmup:
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
            with Timer() as t:
                img = chunk(zc)
                u8 = img.mul(255).to(torch.uint8)  # == M*'s to_uint8 of the [-1, 1] output
            if rep == 0 and c == 0:
                t_first = t.ms
                assert img.shape[2] == FRAMES_PER_CHUNK, img.shape
            if c < args.warmup:
                continue
            lat.append(t.ms)
            k = c - args.warmup
            if rep == 0 and k < QUALITY_CHUNKS and ref is not None:
                out = img.float().cpu() * 2 - 1
                qf.append(psnr(out, ref["float"][k], 2.0))
                qu8.append(psnr(u8.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
        if rep == 0:
            mem = {"peak_allocated_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
                   "peak_reserved_gib": round(torch.cuda.max_memory_reserved() / 2**30, 2)}
    if args.layout_check:
        mem["conv_inputs_not_channels_last"] = layout_check(args, start, chunk, sa, z)
    mem["output_dtype"] = str(img.dtype)
    return lat, qf, qu8, mem, t_first, settings


def layout_check(args, start, chunk, sa, z):
    """Conv calls (F.conv3d / F.conv2d) whose input is not channels-last, over one
    steady chunk run eagerly (n/a for compiled variants: the compiled graph runs
    without SGLang's custom kernels)."""
    if args.compile:
        return "n/a (compiled)"
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

    F.conv3d = wrap("conv3d", torch.channels_last_3d)
    F.conv2d = wrap("conv2d", torch.channels_last)
    try:
        start(z[:, :, :1])
        chunk(z[:, :, 1:4])
        bad["n"] = bad["total"] = 0
        chunk(z[:, :, 4:7])
    finally:
        F.conv3d, F.conv2d = orig["conv3d"], orig["conv2d"]
    return f"{bad['n']}/{bad['total']}"


def profile(args, raw):
    """M*'s profile_kernels.py: warm the stream, torch.profiler over
    ``--profile-chunks`` steady chunks, GPU time per kernel name and category."""
    from torch.autograd import DeviceType
    from torch.profiler import ProfilerActivity
    from torch.profiler import profile as tprofile

    sys.path.insert(0, str(Path(__file__).parent))
    from kernel_categories_sglang import CATEGORIES, category, mstar_category

    start, chunk, vae, sa, settings = build(args)
    z = normalized_latents(raw, vae, sa, "cuda")
    start(z[:, :, :1])
    for c in range(args.warmup):
        chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]).mul(255).to(torch.uint8)
    torch.cuda.synchronize()
    with tprofile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
        for c in range(args.warmup, args.warmup + args.profile_chunks):
            chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]).mul(255).to(torch.uint8)
        torch.cuda.synchronize()
    per_name = defaultdict(lambda: [0, 0.0])
    for e in prof.events():
        if e.device_type != DeviceType.CUDA:
            continue
        rec = per_name[e.name]
        rec[0] += 1
        rec[1] += e.device_time_total / 1e3
    n = args.profile_chunks
    cats = {c: 0.0 for c in CATEGORIES}
    cats_mstar = {c: 0.0 for c in CATEGORIES}
    for name, (_, ms) in per_name.items():
        cats[category(name)] += ms / n
        cats_mstar[mstar_category(name)] += ms / n
    top = sorted(per_name.items(), key=lambda kv: -kv[1][1])[:15]
    sglang_kernels = sorted(
        ({"name": name[:160], "category": category(name), "mstar_category": mstar_category(name),
          "calls_per_chunk": round(cnt / n, 2), "ms_per_chunk": round(ms / n, 3)}
         for name, (cnt, ms) in per_name.items() if category(name) != mstar_category(name)),
        key=lambda r: -r["ms_per_chunk"])
    result = {
        "variant": args.variant,
        "vae": REPOS[args.vae],
        "profiled_chunks": n,
        "kernels_per_chunk": round(sum(c for c, _ in per_name.values()) / n, 1),
        "distinct_kernels": len(per_name),
        "gpu_ms_per_chunk": round(sum(ms for _, ms in per_name.values()) / n, 2),
        "category_ms_per_chunk": {k: round(v, 2) for k, v in cats.items()},
        "category_ms_per_chunk_mstar_rules": {k: round(v, 2) for k, v in cats_mstar.items()},
        "recategorized_kernels": sglang_kernels,
        "top15": [
            {"name": name[:160], "category": category(name), "calls_per_chunk": round(cnt / n, 2),
             "ms_per_chunk": round(ms / n, 3)}
            for name, (cnt, ms) in top
        ],
        "sglang_settings": settings,
        "env": env_info(),
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.variant}.kernels.json").write_text(json.dumps(result, indent=2))
    if args.trace:
        prof.export_chrome_trace(str(out / f"{args.variant}.trace.json"))
    print(json.dumps({k: result[k] for k in ("variant", "kernels_per_chunk", "gpu_ms_per_chunk",
                                             "category_ms_per_chunk")}))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True, help="result label")
    ap.add_argument("--vae", choices=list(REPOS), default="wan21")
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ref")
    ap.add_argument("--out", required=True)
    ap.add_argument("--path", choices=["stage", "realtime"], default="stage")
    ap.add_argument("--quality", choices=["lossless", "extra-high", "high"], default="lossless")
    ap.add_argument("--compile", action="store_true", help="--enable-torch-compile")
    ap.add_argument("--compile-mode", help="SGLANG_VAE_TORCH_COMPILE_MODE")
    ap.add_argument("--decode-precision", choices=["fp32", "bf16", "fp16"], help="--vae-decode-precision")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--chunks", type=int, default=75)
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--layout-check", action="store_true")
    ap.add_argument("--profile", action="store_true", help="kernel profile only (profile_kernels.py schema)")
    ap.add_argument("--profile-chunks", type=int, default=6)
    ap.add_argument("--trace", action="store_true")
    args = ap.parse_args()

    raw = torch.load(args.latents)["raw"]
    if args.profile:
        profile(args, raw)
        return
    ref = torch.load(args.ref) if args.ref and Path(args.ref).exists() else None
    t0 = time.time()
    lat, qf, qu8, extra, t_first, settings = run(args, raw, ref)
    counters = torch._dynamo.utils.counters
    if settings["enable_torch_compile"]:
        extra["dynamo_unique_graph_breaks"] = len(counters["graph_break"])
        extra["dynamo_graph_break_reasons"] = sorted(counters["graph_break"])[:20]
        extra["dynamo_frames_compiled"] = counters["stats"].get("unique_graphs", None)
    med = statistics.median(lat)
    finite_u8 = [q for q in qu8 if math.isfinite(q)] or [float("inf")]
    row = {
        "variant": args.variant,
        "n_samples": len(lat),
        "median_ms": round(med, 2),
        "p90_ms": round(pct(lat, 0.9), 2),
        "p10_ms": round(pct(lat, 0.1), 2),
        "mean_ms": round(statistics.fmean(lat), 2),
        "decoder_fps": round(FRAMES_PER_CHUNK / (med / 1e3), 1),
        "psnr_float_db_vs_fp32": round(statistics.fmean(qf), 2) if qf else None,
        "psnr_float_db_vs_fp32_peak1": round(statistics.fmean(qf) - 20 * math.log10(2), 2) if qf else None,
        "psnr_u8_db_vs_fp32": round(statistics.fmean(finite_u8), 2) if qu8 else None,
        "first_timed_call_ms": round(t_first, 1) if t_first else None,
        "wall_s": round(time.time() - t0, 1),
        **extra,
        "steady_decode_latents": 3,
        "warm_cache": True,
        "sglang_settings": settings,
        "precision_flags": precision_flags(),
        "env": env_info(),
        "latents": str(args.latents),
        "vae": REPOS[args.vae],
        "vae_revision": REVISIONS[args.vae],
    }
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{args.variant}.json").write_text(json.dumps(row, indent=2))
    (out / f"{args.variant}.latencies.json").write_text(json.dumps(lat))
    print(json.dumps({k: row[k] for k in row if k not in ("env",)}))


if __name__ == "__main__":
    main()
