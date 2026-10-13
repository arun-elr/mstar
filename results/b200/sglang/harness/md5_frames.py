"""uint8 frame stream of one decoder over the whole latent stream (first latent
+ every 3-latent chunk = 1 + 85 * 12 = 1021 frames), md5 over the raw THWC
bytes (SGLang's ``_sample_to_uint8_frames`` layout and formula:
``(x * 255).clamp(0, 255).to(uint8)`` on the [0, 1] video, which is M*'s
``to_uint8`` on the [-1, 1] video). Saves the stream as .npy for compare mode.

  python md5_frames.py stream   --vae wan21 --latents L --out DIR --label sglang_default [--quality ...]
  python md5_frames.py whole    --vae wan21 --latents L --out DIR --label sglang_default_whole [--quality ...]
  python md5_frames.py diffusers --vae wan21 --latents L --out DIR --label diffusers_bf16
  python md5_frames.py compare  --out DIR --vae wan21 a b [c ...]

stream:    bench_sglang_wan.build(): DecodingStage.decode -> AutoencoderKLWan.causal_decode
           per chunk (the timed path).
whole:     DecodingStage.decode with SGLang's stock AutoencoderKLWan.decode on all 256
           latents in one call (the offline path PR #34125 measured).
diffusers: M*'s StreamingDecoder (stock diffusers AutoencoderKLWan, eager, bf16,
           cache kept across chunks, one latent per call) = M*'s diffusers_bf16 variant.
"""

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).parent


def thwc_u8_from01(video01):
    return (video01[0] * 255).clamp(0, 255).to(torch.uint8).permute(1, 2, 3, 0).contiguous()


def sglang_stream(args, raw):
    sys.argv = [sys.argv[0]]
    import bench_sglang_wan as B

    a = argparse.Namespace(vae=args.vae, path=args.path, quality=args.quality, compile=args.compile,
                           compile_mode=None, decode_precision=args.decode_precision)
    start, chunk, vae, sa, settings = B.build(a)
    z = B.normalized_latents(raw, vae, sa, "cuda")
    frames = [thwc_u8_from01(start(z[:, :, :1])).cpu()]
    for c in range((z.shape[2] - 1) // 3):
        frames.append(thwc_u8_from01(chunk(z[:, :, 1 + 3 * c: 4 + 3 * c])).cpu())
    return torch.cat(frames), settings


def sglang_whole(args, raw):
    import bench_sglang_wan as B
    from sglang.multimodal_gen.configs.sample.sampling_params import quality_allows_kernel_fusions
    from sglang.multimodal_gen.runtime.models.vaes.fast_path_gate import use_vae_fast_path
    from sglang.multimodal_gen.runtime.pipelines_core.stages.decoding import DecodingStage
    from sglang.multimodal_gen.runtime.utils.precision import resolve_decode_precision

    a = argparse.Namespace(vae=args.vae, path="stage", quality=args.quality, compile=False,
                           compile_mode=None, decode_precision=args.decode_precision)
    _, _, vae, sa, settings = B.build(a)
    del vae.decode  # back to the class's AutoencoderKLWan.decode (whole video, cache cleared)
    stage = DecodingStage(vae)
    z = B.normalized_latents(raw, vae, sa, "cuda")
    dtype = resolve_decode_precision(sa, "vae", quality=args.quality)
    with use_vae_fast_path(vae, quality_allows_kernel_fusions(args.quality)):
        video = stage.decode(z, sa, vae_dtype=dtype)
    settings["path"] = "DecodingStage.decode -> AutoencoderKLWan.decode (whole clip)"
    return thwc_u8_from01(video).cpu(), settings


def diffusers_stream(args, raw):
    import diffusers
    from mstar_common import WAN21_REPO, WAN22_REPO, StreamingDecoder, load_vae

    vae = load_vae(WAN21_REPO if args.vae == "wan21" else WAN22_REPO, torch.bfloat16, "cuda")
    sd = StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=False)
    z = raw.to("cuda", torch.bfloat16)
    with torch.inference_mode():
        frames = [thwc_u8_from01(sd.first(z[:, :, :1]).float() / 2 + 0.5).cpu()]
        for c in range((z.shape[2] - 1) // 3):
            out = sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c])
            frames.append(thwc_u8_from01((out.float() / 2 + 0.5).clamp(0, 1)).cpu())
    return torch.cat(frames), {"path": "diffusers AutoencoderKLWan, eager bf16, per-frame, warm cache",
                               "diffusers": diffusers.__version__, "diffusers_file": diffusers.__file__}


def compare(args):
    out = Path(args.out)
    streams = {l: np.load(out / f"{args.vae}_{l}.npy", mmap_mode="r") for l in args.labels}
    meta = {l: json.loads((out / f"{args.vae}_{l}.json").read_text()) for l in args.labels}
    base = args.labels[0]
    rows = []
    for l in args.labels:
        row = {"label": l, "md5": meta[l]["md5"], "frames": meta[l]["frames"]}
        if l != base:
            a, b = streams[base], streams[l]
            same, mse, maxd = 0, 0.0, 0
            for i in range(0, a.shape[0], 32):
                x = np.asarray(a[i:i + 32], dtype=np.int16)
                y = np.asarray(b[i:i + 32], dtype=np.int16)
                d = x - y
                same += int((d == 0).sum())
                mse += float((d.astype(np.float64) ** 2).sum())
                maxd = max(maxd, int(np.abs(d).max()))
            n = a.size
            mse /= n
            row.update({f"vs_{base}_identical_bytes_pct": round(same / n * 100, 4),
                        f"vs_{base}_max_abs_u8": maxd,
                        f"vs_{base}_psnr_u8_db": round(10 * math.log10(255 ** 2 / mse), 2) if mse else float("inf"),
                        f"vs_{base}_md5_equal": meta[l]["md5"] == meta[base]["md5"]})
        rows.append(row)
    (out / f"{args.vae}_compare_{'_vs_'.join(args.labels)}.json").write_text(json.dumps(rows, indent=2))
    print(json.dumps(rows, indent=2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["stream", "whole", "diffusers", "compare"])
    ap.add_argument("labels", nargs="*")
    ap.add_argument("--vae", default="wan21")
    ap.add_argument("--latents")
    ap.add_argument("--out", required=True)
    ap.add_argument("--label")
    ap.add_argument("--path", default="stage")
    ap.add_argument("--quality", default="lossless")
    ap.add_argument("--compile", action="store_true")
    ap.add_argument("--decode-precision")
    args = ap.parse_intermixed_args()
    if args.mode == "compare":
        return compare(args)
    raw = torch.load(args.latents)["raw"]
    fn = {"stream": sglang_stream, "whole": sglang_whole, "diffusers": diffusers_stream}[args.mode]
    frames, settings = fn(args, raw)
    arr = frames.numpy()
    md5 = hashlib.md5(arr.tobytes()).hexdigest()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / f"{args.vae}_{args.label}.npy", arr)
    meta = {"label": args.label, "vae": args.vae, "md5": md5, "frames": arr.shape[0], "shape_thwc": list(arr.shape),
            "settings": settings, "torch": torch.__version__}
    (out / f"{args.vae}_{args.label}.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("label", "md5", "frames", "shape_thwc")}))


if __name__ == "__main__":
    main()
