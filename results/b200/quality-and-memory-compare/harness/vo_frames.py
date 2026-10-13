"""Whole-stream uint8 frames of one vLLM-Omni variant, for quality.py-style analysis.

Same decoder object and latent schedule as the timing (bench_vllm_omni.py's
build_streaming, then bench_wan_decoder.run_streaming's stream: latent 0 alone,
then every 3-latent chunk), all 85 chunks once, through M*'s to_uint8. Writes
<out>/<variant>.u8.npy ([1021, H, W, 3]) and <variant>.meta.json (md5, and the
uint8 PSNR over the 10 chunks after the 10 warm-up chunks against
ref_fp32_outputs.pt, which reproduces the timing JSON's psnr_u8 when the
stream matches).

  PYTHONPATH=/bench python vo_frames.py --variant vllm_omni_default --vae wan21 \
      --latents /vo/inputs/wan21/latents.pt --ref /vo/inputs/wan21/ref_fp32_outputs.pt --out /qc/wan21
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.argv, _argv = sys.argv[:1], sys.argv
import bench_vllm_omni  # noqa: E402,F401  (patches build_streaming / env_info)
import bench_wan_decoder as B  # noqa: E402
from common import FRAMES_PER_CHUNK, WAN21_REPO, WAN22_REPO, to_uint8  # noqa: E402

sys.argv = _argv


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", required=True)
    ap.add_argument("--vae", choices=["wan21", "wan22"], required=True)
    ap.add_argument("--latents", required=True)
    ap.add_argument("--ref", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--warmup", type=int, default=10)
    args = ap.parse_args()
    B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    raw = torch.load(args.latents)["raw"]
    ref = torch.load(args.ref)
    n_chunks = (raw.shape[2] - 1) // 3
    n_frames = 1 + n_chunks * FRAMES_PER_CHUNK
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    sd, dtype = B.build_streaming(args.variant, "cuda")
    z = raw.to("cuda", dtype)
    u8 = None
    n = 0
    psnr_u8 = []
    with torch.inference_mode():
        sd.reset()
        outs = [sd.first(z[:, :, :1])]
        for c in range(n_chunks + 1):
            if c > 0:
                outs = [sd.chunk(z[:, :, 1 + 3 * (c - 1): 4 + 3 * (c - 1)])]
            video = outs[0]
            f = to_uint8(video)
            k = c - 1 - args.warmup
            if 0 <= k < len(ref["float"]):
                psnr_u8.append(B.psnr(f.float().cpu(), to_uint8(ref["float"][k]).float(), 255.0))
            arr = f[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
            if u8 is None:
                u8 = np.lib.format.open_memmap(out / f"{args.variant}.u8.npy", "w+", np.uint8,
                                               (n_frames, arr.shape[1], arr.shape[2], 3))
            u8[n: n + arr.shape[0]] = arr
            n += arr.shape[0]
    assert n == n_frames, (n, n_frames)
    u8.flush()
    h = hashlib.md5()
    for i in range(0, n_frames, 64):
        h.update(np.ascontiguousarray(u8[i: i + 64]).tobytes())
    meta = {"variant": args.variant, "vae": B.REPO, "frames": n_frames, "chunks": n_chunks,
            "md5_u8_stream": h.hexdigest(), "output_dtype": str(video.dtype),
            "psnr_u8_vs_ref_fp32_outputs_10_chunks_after_warmup": round(float(np.mean(psnr_u8)), 4),
            "latents": args.latents, "run_info": bench_vllm_omni.RUN_INFO,
            "precision_flags": B.precision_flags() if hasattr(B, "precision_flags") else None,
            "env": B.env_info()}
    (out / f"{args.variant}.meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(json.dumps({k: meta[k] for k in ("variant", "vae", "frames", "md5_u8_stream", "output_dtype",
                                           "psnr_u8_vs_ref_fp32_outputs_10_chunks_after_warmup")}))


if __name__ == "__main__":
    main()
