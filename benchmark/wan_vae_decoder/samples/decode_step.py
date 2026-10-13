"""Decode one prompt's saved latents with one step of PR #425's change ladder
(one step per process), through M*'s in-tree streaming decoder node
``Cosmos3VAEDecoderARSubmodule.forward``, or with stock diffusers ``vae.decode``.

  python decode_step.py --step s4_cudnn_benchmark --latents runs/fox_latents.pt --out runs/fox

The node gets the stream as one windowed request: the clip-start latent, then
3-latent windows (overlap 0, stream_video), the DiT's normalized latents as
input, exactly as the engine hands them over. The stream runs twice in the
process: pass 1 compiles / warms (and runs cuDNN's algorithm search where
enabled), pass 2 is timed and its frames are written: ``<step>.u8.npy``
(uint8 ``[F, H, W, 3]``) and ``<step>.json`` (per-window ms, md5 of both passes).
"""

import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

MSTAR = os.environ.get("MSTAR_PATH", str(Path(__file__).resolve().parents[3]))  # default: this checkout
sys.path.insert(0, MSTAR)
REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
CHUNK = 3

OFF = dict(windowed_decode_warm_cache=False, vae_decode_channels_last=False,
           windowed_decode_latents_per_call=1, vae_decode_cudnn_benchmark=False)
# Cumulative: each step turns on one more switch. Native conv padding has no
# switch of its own: ChannelsLastCausalConv3d always pads inside the conv, so it
# lands with vae_decode_channels_last.
NODE_STEPS = {
    "s0_mstar_before": OFF,
    "s1_warm_cache": OFF | dict(windowed_decode_warm_cache=True),
    "s2_channels_last": OFF | dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True),
    "s3_one_call": OFF | dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True,
                              windowed_decode_latents_per_call=0),
    "s4_cudnn_benchmark": dict(windowed_decode_warm_cache=True, vae_decode_channels_last=True,
                               windowed_decode_latents_per_call=0, vae_decode_cudnn_benchmark=True),
    # Control, not a ladder step: the warm cache alone with COSMOS3_COMPILE_VAE=0.
    "c1_warm_cache_eager": OFF | dict(windowed_decode_warm_cache=True),
}
EAGER = {"c1_warm_cache_eager"}
# Windows 4.. (0-based): past the clip-start window, the first 3-latent window
# (which the compiled warm path runs eagerly) and the stateless re-decode's
# growing context (1 + 3k latents until it reaches 8 + 3).
STEADY_FROM = 4
STOCK_STEPS = ("stock_bf16", "stock_fp32")


def load_vae(dtype):
    from diffusers import AutoencoderKLWan
    from huggingface_hub import snapshot_download

    path = snapshot_download(REPO, revision=REVISION, local_files_only=True)
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=dtype)
    vae.encoder = None
    vae.quant_conv = None
    return vae.to("cuda").eval()


def make_node(overrides, compiled):
    os.environ["COSMOS3_COMPILE_VAE"] = "1" if compiled else "0"
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule

    cfg = Cosmos3Config()
    for k, v in overrides.items():
        assert hasattr(cfg, k), k
        setattr(cfg, k, v)
    # The VAE is loaded fp32; the node casts it to its decode dtype (bf16 on cuDNN >= 9.16).
    return Cosmos3VAEDecoderARSubmodule(vae=load_vae(torch.float32), config=cfg), cfg


def sync_ms(fn):
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    out = fn()
    torch.cuda.synchronize()
    return out, (time.perf_counter() - t0) * 1e3


def run_node(step, lat):
    from mstar.model.cosmos3 import constants as C

    sub, cfg = make_node(NODE_STEPS[step], compiled=step not in EAGER)
    n = lat.shape[2]
    assert (n - 1) % CHUNK == 0, n
    windows = [1] + [CHUNK] * ((n - 1) // CHUNK)
    md = {"num_windows": len(windows), "overlap_latent_units": 0, "num_frames": 1 + (n - 1) * 4,
          "stream_video": True}

    def stream(rid):
        infos = SimpleNamespace(request_ids=[rid], per_request_info={rid: SimpleNamespace(step_metadata=md)})
        outs, ms, a = [], [], 0
        for w in windows:
            out, t = sync_ms(
                lambda a=a, w=w: sub.forward(C.VIDEO_DECODE_AR_WALK, infos, lat[:, :, a:a + w])["video_output"][0])
            outs.append(out)
            ms.append(t)
            a += w
        sub.cleanup_request(rid)
        return torch.cat(outs, dim=2), ms

    with torch.inference_mode():
        px1, ms1 = stream("warmup")
        u8_1 = px1[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        del px1
        torch.cuda.reset_peak_memory_stats()
        px2, ms2 = stream("timed")
    effective = {k: getattr(cfg, k) for k in OFF} | {
        "windowed_decode_context_latents": cfg.windowed_decode_context_latents,
        "compiled": sub._decode_compiled, "decode_dtype": str(sub._decode_dtype())}
    return px2, ms1, ms2, u8_1, effective


def run_stock(step, lat):
    """Stock diffusers ``vae.decode`` of the whole clip, eager. ``stock_bf16`` is
    fed the node's own bf16 VAE input (the same ``z`` every node step decodes),
    so it isolates the decoder; ``stock_fp32`` is the model card's pipeline
    decode (fp32 VAE, WanPipeline's de-normalization)."""
    if step == "stock_bf16":
        sub, _ = make_node(OFF, compiled=False)
        z = sub._vae_input(lat).contiguous()
        vae = sub.vae
        to_u8 = sub._to_uint8
    else:
        vae = load_vae(torch.float32)
        zd = vae.config.z_dim
        mean = torch.tensor(vae.config.latents_mean).view(1, zd, 1, 1, 1).to(lat.device, torch.float32)
        inv_std = 1.0 / torch.tensor(vae.config.latents_std).view(1, zd, 1, 1, 1).to(lat.device, torch.float32)
        z = lat.float() / inv_std + mean
        to_u8 = lambda v: (v / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)  # noqa: E731
    with torch.inference_mode():
        px1, t1 = sync_ms(lambda: to_u8(vae.decode(z).sample))
        u8_1 = px1[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        del px1
        torch.cuda.reset_peak_memory_stats()
        px2, t2 = sync_ms(lambda: to_u8(vae.decode(z).sample))
    return px2, [t1], [t2], u8_1, {"decode_dtype": str(next(vae.parameters()).dtype), "compiled": False,
                                   "call": "vae.decode(whole clip)"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--step", required=True, choices=[*STOCK_STEPS, *NODE_STEPS])
    ap.add_argument("--latents", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    blob = torch.load(args.latents)
    lat = blob["latents"].to("cuda")

    run = run_stock if args.step in STOCK_STEPS else run_node
    px, ms1, ms2, u8_1, effective = run(args.step, lat)
    peak = torch.cuda.max_memory_allocated() / 2**30
    u8 = px[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
    np.save(out / f"{args.step}.u8.npy", u8)
    md5 = hashlib.md5(u8.tobytes()).hexdigest()
    steady = ms2[STEADY_FROM:] or ms2[1:]
    rec = {
        "step": args.step, "prompt_id": blob["meta"]["prompt_id"], "frames": u8.shape[0],
        "height": u8.shape[1], "width": u8.shape[2], "md5": md5,
        "md5_pass1": hashlib.md5(u8_1.tobytes()).hexdigest(),
        "pass1_total_ms": round(sum(ms1), 1), "total_ms": round(sum(ms2), 1),
        "window_ms": [round(t, 2) for t in ms2],
        "steady_chunk_ms_median": round(float(np.median(steady)), 2) if len(ms2) > 1 else None,
        "steady_windows": f"{STEADY_FROM}..{len(ms2) - 1}" if len(ms2) > STEADY_FROM else "1..",
        "peak_alloc_gib": round(peak, 2), "config": effective,
        "torch": torch.__version__, "cudnn": torch.backends.cudnn.version(),
        "cudnn_benchmark_after": torch.backends.cudnn.benchmark,
    }
    (out / f"{args.step}.json").write_text(json.dumps(rec, indent=2))
    print(json.dumps({k: rec[k] for k in ("step", "md5", "md5_pass1", "total_ms", "steady_chunk_ms_median",
                                          "peak_alloc_gib")}))


if __name__ == "__main__":
    main()
