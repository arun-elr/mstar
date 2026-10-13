"""vLLM-Omni's Wan VAE decode under the M* streaming-decode benchmark.

Reuses ``bench_wan_decoder.py`` (timing loop, PSNR, JSON schema) and
``profile_kernels.py`` (kernel table) unchanged; only the decoder object differs.
A ``vllm_omni_*`` variant builds the VAE the way vLLM-Omni's diffusion stage
does (``DistributedAutoencoderKLWan.from_pretrained`` in bf16, the registry's
slicing/tiling/parallel setup, then the Wan VAE fast path install) and streams
the latents through ``WanStreamingDecoder``, the session-owned warm-cache
decoder LingBot-World serves AR blocks with (one latent frame per decoder call,
cache carried across calls, ``_execution_context`` autocast entered).

Variants (``--vae wan21|wan22``):
  vllm_omni_default        vae_fast_path=lossless (the engine default), no tiling
  vllm_omni_off            vae_fast_path=off (reference diffusers forwards)
  vllm_omni_channels_last  vae_fast_path=channels_last
  vllm_omni_lingbot_{exact,fused}  LingBot-World's lingbot_vae_decode_fast_path
                           (installed before the registry fast path, which then skips)
  vllm_omni_best           the fastest vLLM-Omni configuration (BEST below)
  any of the above + _cudnnbench   torch.backends.cudnn.benchmark=True (a torch flag,
                           not a vLLM-Omni switch)
  vllm_omni_oneshot_{F}f[_tiled]   whole-clip ``vae.decode`` (the Wan2.2 pipeline's decode)
                           of the first F frames, default fast path; per 12 output frames
  anything else            delegated to bench_wan_decoder.py (M* variants as controls)

  python bench_vllm_omni.py --variant vllm_omni_default --vae wan21 \
      --latents /vo/inputs/wan21/latents.pt --ref /vo/inputs/wan21/ref_fp32_outputs.pt --out out/

Runs inside the vLLM-Omni image with benchmark/wan_vae_decoder on PYTHONPATH (run1.sh);
VO_VAE_ROOT holds the two VAE folders as <root>/wan21 and <root>/wan22.
  python bench_vllm_omni.py --profile-kernels --variant vllm_omni_default --vae wan21 --latents ... --out out/
"""

import os
import re
import socket
import statistics
import sys
from pathlib import Path

import bench_wan_decoder as B
import common
import torch
from common import FRAMES_PER_CHUNK, WAN21_REPO

VAE_ROOT = Path(os.environ.get("VO_VAE_ROOT", "/vo/vae"))
BEST = "channels_last"
CONFIGS = {
    "default": {"vae_fast_path": "lossless", "lingbot_vae_decode_fast_path": None},
    "off": {"vae_fast_path": "off", "lingbot_vae_decode_fast_path": None},
    "channels_last": {"vae_fast_path": "channels_last", "lingbot_vae_decode_fast_path": None},
    "lingbot_exact": {"vae_fast_path": "lossless", "lingbot_vae_decode_fast_path": "exact"},
    "lingbot_fused": {"vae_fast_path": "lossless", "lingbot_vae_decode_fast_path": "fused"},
}
RUN_INFO: dict = {}


def vae_key() -> str:
    return "wan21" if B.REPO == WAN21_REPO else "wan22"


def init_world():
    """World of size 1: ``DistributedAutoencoderKLWan`` needs vLLM-Omni's world group."""
    import torch.distributed as dist
    from vllm_omni.diffusion.distributed.parallel_state import init_distributed_environment

    if dist.is_initialized():
        return
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    init_distributed_environment(world_size=1, rank=0, local_rank=0,
                                 distributed_init_method=f"tcp://127.0.0.1:{port}")


def build_vae(cfg: dict, use_tiling: bool, device: str):
    """The VAE as vLLM-Omni's diffusion stage hands it to the pipeline."""
    from vllm_omni.diffusion.distributed.autoencoders.autoencoder_kl_wan import DistributedAutoencoderKLWan
    from vllm_omni.diffusion.distributed.autoencoders.wan_decoder_fast_path import install_wan_decoder_fast_path
    from vllm_omni.diffusion.distributed.autoencoders.wan_vae_fastpath import install_wan_vae_fastpath

    init_world()
    vae = DistributedAutoencoderKLWan.from_pretrained(str(VAE_ROOT / vae_key()), torch_dtype=torch.bfloat16)
    vae = vae.to(device).eval()
    # registry.initialize_model: memory settings and VAE parallel size from OmniDiffusionConfig defaults
    vae.use_slicing = False
    vae.use_tiling = use_tiling
    vae.set_parallel_size(1, mode="tile")
    info = {"vae_use_tiling": use_tiling, **cfg}
    if cfg["lingbot_vae_decode_fast_path"]:
        # LingBot-World pipeline __init__, before the registry's fast path install
        info["lingbot_fast_path_counts"] = {
            k: v for k, v in install_wan_decoder_fast_path(
                vae, conv_dtype=None, level=cfg["lingbot_vae_decode_fast_path"]).items()}
    if cfg["vae_fast_path"] != "off":
        report = install_wan_vae_fastpath(vae, level=cfg["vae_fast_path"])
        info["vae_fast_path_installed"] = report.installed
        info["vae_fast_path_reason"] = report.reason
        info["vae_fast_path_fused_silu"] = list(report.fused_silu_dtypes)
    else:
        info["vae_fast_path_installed"] = False
    info["autocast_dtype"] = str(next(vae.parameters()).dtype)
    RUN_INFO.update(info)
    return vae


class VoStreamingDecoder:
    """``WanStreamingDecoder`` behind bench_wan_decoder's reset / first / chunk interface."""

    def __init__(self, vae):
        from vllm_omni.experimental.ar_diffusion.streaming_decode import WanStreamingDecoder

        self.vae = vae
        self.dec = WanStreamingDecoder(vae)
        self.state = None
        self.reset()

    def reset(self):
        if self.state is not None:
            self.dec.release(self.state)
        self.state = self.dec.new_decode_state("bench")

    def first(self, z1):
        return self.dec.decode_chunk(z1, self.state)

    def chunk(self, z):
        return self.dec.decode_chunk(z, self.state)


def parse_variant(variant: str):
    m = re.fullmatch(r"vllm_omni_([a-z_]+?)(_cudnnbench)?", variant)
    if not m or variant.startswith("vllm_omni_oneshot"):
        return None
    name = BEST if m[1] == "best" else m[1]
    assert name in CONFIGS, f"unknown vLLM-Omni config {m[1]}"
    return name, bool(m[2])


_orig_build = B.build_streaming


def build_streaming(variant, device):
    parsed = parse_variant(variant)
    if parsed is None:
        return _orig_build(variant, device)
    name, cudnn_bench = parsed
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    if cudnn_bench:
        torch.backends.cudnn.benchmark = True
    vae = build_vae(CONFIGS[name], use_tiling=False, device=device)
    RUN_INFO.update({"vllm_omni_config": name, "decode_path": "WanStreamingDecoder.decode_chunk",
                     "latents_per_decoder_call": 1})
    return VoStreamingDecoder(vae), torch.bfloat16


def run_oneshot(args, raw, ref, frames, tiled):
    """Whole-clip ``vae.decode`` (what Wan22Pipeline runs after denoising), default fast path."""
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    assert (frames - 1) % 4 == 0
    n_lat = (frames - 1) // 4 + 1
    vae = build_vae(CONFIGS["default"], use_tiling=tiled, device=args.device)
    RUN_INFO.update({"vllm_omni_config": "default", "decode_path": "OmniAutoencoderKLWan.decode (whole clip)"})
    z = raw[:, :, :n_lat].to(args.device, torch.bfloat16)
    lat = []
    with torch.inference_mode():
        vae.decode(z, return_dict=False)  # warm Triton JIT, fast-path verdicts, cuDNN
        B.reset_peak(args.device)
        for _ in range(args.oneshot_repeats):
            with B.Timer(args.device) as t:
                video = vae.decode(z, return_dict=False)[0]
                B.to_uint8(video)
            lat.append(t.ms)
    assert video.shape[2] == frames, video.shape
    mem = B.peak(args.device)
    mem.update({"clip_frames": frames, "clip_ms_median": statistics.median(lat)})
    qf, qu8 = [], []
    if ref is not None:
        for k in range(B.QUALITY_CHUNKS):
            s = 1 + FRAMES_PER_CHUNK * (args.warmup + k)
            if s + FRAMES_PER_CHUNK > frames:
                break
            out = video[:, :, s:s + FRAMES_PER_CHUNK].float().cpu()
            qf.append(B.psnr(out, ref["float"][k], 2.0))
            qu8.append(B.psnr(B.to_uint8(out).float(), B.to_uint8(ref["float"][k]).float(), 255.0))
        mem["psnr_chunks"] = len(qf)
    per12 = [x * FRAMES_PER_CHUNK / frames for x in lat]
    return per12, qf, qu8, mem, None


_orig_run_streaming = B.run_streaming


def run_streaming(args, raw, ref):
    if m := re.fullmatch(r"vllm_omni_oneshot_(\d+)f(_tiled)?", args.variant):
        res = run_oneshot(args, raw, ref, int(m[1]), tiled=bool(m[2]))
    else:
        res = _orig_run_streaming(args, raw, ref)
    res[3].update(RUN_INFO)
    return res


_orig_env = B.env_info


def env_info():
    info = _orig_env()
    info.pop("mstar_commit", None)
    try:
        import triton
        import vllm
        import vllm_omni

        info["vllm"] = vllm.__version__
        info["vllm_omni"] = getattr(vllm_omni, "__version__", None)
        info["triton"] = triton.__version__
    except Exception as exc:
        info["vllm_omni"] = f"n/a ({exc})"
    info["vllm_omni_commit"] = os.environ.get("VLLM_OMNI_COMMIT")
    info["docker_image"] = os.environ.get("DOCKER_IMAGE")
    info["docker_image_digest"] = os.environ.get("DOCKER_IMAGE_DIGEST")
    return info


def hf_snapshot(repo: str) -> Path:
    """M* control variants load the same VAE files from the relayed snapshot."""
    return VAE_ROOT.parent / "snap" / ("wan21" if repo == WAN21_REPO else "wan22")


B.build_streaming = build_streaming
B.run_streaming = run_streaming
B.env_info = env_info
common.hf_snapshot = hf_snapshot

def refined_category(name: str) -> str:
    """kernel_categories.category, plus vLLM-Omni's own Triton kernels (named ``_<op>_kernel``,
    not Inductor's ``triton_*``), which it leaves in "other"."""
    from kernel_categories import category

    cat = category(name)
    if cat != "other":
        return cat
    if re.match(r"_(rms_norm|wan_rmsnorm)", name):
        return "norm / SiLU"
    if re.match(r"_cat_", name):
        return "concat / pad"
    if re.match(r"_(add_bias_residual|dup_up3d_add)", name):
        return "other elementwise"
    return cat


def profile_all_kernels():
    """profile_kernels.py as is, plus every kernel name with both categorizations."""
    import collections
    import json

    import profile_kernels
    from kernel_categories import CATEGORIES, category

    tables = []

    def capturing_defaultdict(factory):
        d = collections.defaultdict(factory)
        tables.append(d)
        return d

    profile_kernels.defaultdict = capturing_defaultdict
    profile_kernels.main()
    args = sys.argv
    out = Path(args[args.index("--out") + 1])
    variant = args[args.index("--variant") + 1]
    chunks = int(args[args.index("--chunks") + 1]) if "--chunks" in args else 6
    per_name = tables[-1]
    refined = {c: 0.0 for c in CATEGORIES}
    for name, (_, ms) in per_name.items():
        refined[refined_category(name)] += ms / chunks
    path = out / f"{variant}.kernels.json"
    res = json.loads(path.read_text())
    res["category_ms_per_chunk_refined"] = {k: round(v, 2) for k, v in refined.items()}
    res["refined_rule"] = ("kernel_categories.category; kernels it puts in 'other' re-labelled by name: "
                           "_rms_norm*/_wan_rmsnorm* -> norm / SiLU, _cat_* -> concat / pad, "
                           "_add_bias_residual*/_dup_up3d_add* -> other elementwise")
    res["vllm_omni_run_info"] = RUN_INFO
    res["all_kernels"] = [
        {"name": name[:200], "category": category(name), "refined_category": refined_category(name),
         "calls_per_chunk": round(cnt / chunks, 2), "ms_per_chunk": round(ms / chunks, 4)}
        for name, (cnt, ms) in sorted(per_name.items(), key=lambda kv: -kv[1][1])
    ]
    path.write_text(json.dumps(res, indent=2))
    print(json.dumps({"variant": variant, "refined": res["category_ms_per_chunk_refined"]}))


if __name__ == "__main__":
    if "--profile-kernels" in sys.argv:
        sys.argv.remove("--profile-kernels")
        profile_all_kernels()
    else:
        B.main()
