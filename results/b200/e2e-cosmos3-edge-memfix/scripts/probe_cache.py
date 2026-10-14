"""Decoder-only probe of what a live warm stream holds between windows, on the
PR's Cosmos3VAEDecoderARSubmodule (set up as bench_decode_nodes.py does):
N streams decode their first two 8-latent kv windows interleaved, then reports
torch.cuda.memory_allocated per live stream, and per stream the feature cache's
tensor bytes (numel * element size) against the bytes of the storages those
tensors keep alive (a slice of a larger buffer keeps all of it).

  MSTAR=~/e2e-cosmos3/mstar-pr python probe_cache.py --set windowed_decode_latents_per_call=0 [--eager]
"""
import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

MSTAR = Path(os.environ["MSTAR"])
sys.path.insert(0, str(MSTAR))
ap = argparse.ArgumentParser()
ap.add_argument("--set", action="append", default=[])
ap.add_argument("--eager", action="store_true")
ap.add_argument("--streams", type=int, default=4)
args = ap.parse_args()
os.environ["COSMOS3_COMPILE_VAE"] = "0" if args.eager else "1"

from diffusers import AutoencoderKLWan  # noqa: E402
from huggingface_hub import snapshot_download  # noqa: E402

from mstar.engine import apply_torch_config  # noqa: E402
from mstar.model.cosmos3 import constants as C  # noqa: E402
from mstar.model.cosmos3.config import Cosmos3Config  # noqa: E402
from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule  # noqa: E402

apply_torch_config()
path = Path(snapshot_download("nvidia/Cosmos3-Edge", revision="344d602b128d1bbdacb43b08d0a3626f46343e29",
                              allow_patterns=["vae/*"])) / "vae"
vae = AutoencoderKLWan.from_pretrained(str(path), torch_dtype=torch.float32)
vae.encoder = vae.quant_conv = None
vae = vae.cuda().eval()
cfg = Cosmos3Config()
for kv in args.set:
    k, _, v = kv.partition("=")
    setattr(cfg, k, int(v) if v.lstrip("-").isdigit() else (v.lower() == "true" if v.lower() in ("true", "false") else v))
sub = Cosmos3VAEDecoderARSubmodule(vae, cfg)
W, H, sf, win = 832, 480, vae.config.scale_factor_spatial, 8
g = torch.Generator().manual_seed(0)
lat = torch.randn(1, vae.config.z_dim, 3 * win, H // sf, W // sf, generator=g).to("cuda", torch.bfloat16)
md = {"num_windows": 3, "overlap_latent_units": 0, "num_frames": 1 + (3 * win - 1) * 4, "stream_video": True}
rids = [f"s{i}" for i in range(args.streams)]
infos = {r: SimpleNamespace(request_ids=[r], per_request_info={r: SimpleNamespace(step_metadata=md)}) for r in rids}


def held(r):
    cache = sub.request_state(r).get("ar_feat_cache") or []
    ts = [t for t in cache if torch.is_tensor(t)]
    storages = {t.untyped_storage().data_ptr(): t.untyped_storage().nbytes() for t in ts}
    return sum(t.numel() * t.element_size() for t in ts) / 2**30, sum(storages.values()) / 2**30


torch.cuda.synchronize()
base = torch.cuda.memory_allocated()
res = {"overrides": args.set, "compiled": not args.eager, "base_allocated_gib": base / 2**30, "rounds": []}
with torch.inference_mode():
    for k in range(2):  # window 0 (clip start, builds the cache), window 1 (steady, warm)
        for r in rids:
            sub.forward(C.VIDEO_DECODE_AR_WALK, infos[r], lat[:, :, k * win:(k + 1) * win])
        torch.cuda.synchronize()
        torch.cuda.empty_cache()
        per = [held(r) for r in rids]
        res["rounds"].append({
            "after_window": k, "live_streams": len(rids),
            "allocated_gib": torch.cuda.memory_allocated() / 2**30,
            "held_gib_per_stream": (torch.cuda.memory_allocated() - base) / len(rids) / 2**30,
            "cache_tensor_gib_per_stream": per[0][0], "cache_storage_gib_per_stream": per[0][1],
        })
    for r in rids:
        sub.cleanup_request(r)
    torch.cuda.synchronize()
    res["allocated_after_cleanup_gib"] = torch.cuda.memory_allocated() / 2**30
    res["peak_allocated_gib"] = torch.cuda.max_memory_allocated() / 2**30
print(json.dumps(res))
