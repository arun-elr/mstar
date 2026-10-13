"""CPU-only correctness checks for the benchmark harness (never touches a GPU).

Real Wan 2.1 VAE weights, tiny spatial size, fp32:
  1. warm-cache streaming (per-frame) reproduces AutoencoderKLWan.decode of the whole stream
  2. one decoder call per chunk == per-frame calls      (ladder step 2 is exact)
  3. channels-last == channels-first                    (ladder step 1 is exact)
  4. nearest upsample in bf16 == fp32 round trip        (exact: nearest copies values)
  5. M*'s Cosmos3 AR decode (ctx 8 / 1) vs exact streaming, and M*'s Wan22 node runs
"""

import os

os.environ["CUDA_VISIBLE_DEVICES"] = ""

import sys
from types import SimpleNamespace

import torch

from common import WAN21_REPO, StreamingDecoder, bf16_nearest_upsample, latent_stats, load_vae, mstar_path

torch.manual_seed(0)
H_LAT, W_LAT, CHUNKS = 6, 8, 4


def stream(vae, z, **kw):
    sd = StreamingDecoder(vae, compile_decoder=False, **kw)
    outs = [sd.first(z[:, :, :1])]
    for c in range(CHUNKS):
        outs.append(sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]))
    return torch.cat(outs, 2)


def maxdiff(a, b):
    return (a.float() - b.float()).abs().max().item()


def main():
    torch.set_num_threads(16)
    vae = load_vae(WAN21_REPO, torch.float32, "cpu")
    mean, std = latent_stats(vae, "cpu")
    z = torch.randn(1, 16, 1 + 3 * CHUNKS, H_LAT, W_LAT) * std + mean
    with torch.inference_mode():
        full = vae.decode(z).sample
        per_frame = stream(vae, z, per_frame=True, channels_last=False)
        chunked = stream(vae, z, per_frame=False, channels_last=False)
        print(f"output {tuple(full.shape)}")
        print(f"1. streaming per-frame vs vae.decode   max|diff| = {maxdiff(per_frame, full):.3e}")
        print(f"2. one call per chunk vs per-frame     max|diff| = {maxdiff(chunked, per_frame):.3e}")

        vae_cl = load_vae(WAN21_REPO, torch.float32, "cpu")
        cl = stream(vae_cl, z, per_frame=False, channels_last=True)
        print(f"3. channels-last chunked vs per-frame  max|diff| = {maxdiff(cl, per_frame):.3e}")

        vae_bf = load_vae(WAN21_REPO, torch.bfloat16, "cpu")
        a = stream(vae_bf, z.bfloat16(), per_frame=False, channels_last=False)
        bf16_nearest_upsample()
        b = stream(vae_bf, z.bfloat16(), per_frame=False, channels_last=False)
        print(f"4. bf16 nearest upsample vs fp32 trip  max|diff| = {maxdiff(a, b):.3e}")

        sys.path.insert(0, mstar_path())
        os.environ["COSMOS3_COMPILE_VAE"] = "0"
        os.environ["COSMOS3_VAE_DECODE_FP32"] = "1"
        from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

        sub = Cosmos3VAEDecoderSubmodule(vae=vae, config=SimpleNamespace())
        norm = (z - mean) / std
        exact_u8 = ((per_frame / 2 + 0.5).clamp(0, 1) * 255).to(torch.uint8)
        for ctx in (8, 1):
            tail, outs = norm[:, :, :1], [sub._decode_pixels(norm[:, :, :1])]
            for c in range(CHUNKS):
                st = torch.cat([tail, norm[:, :, 1 + 3 * c: 4 + 3 * c]], 2)
                outs.append(sub._decode_pixels(st)[:, :, -12:])
                tail = st[:, :, -ctx:]
            ar = torch.cat(outs, 2)
            per_chunk = [maxdiff(ar[:, :, 1 + 12 * c: 13 + 12 * c], exact_u8[:, :, 1 + 12 * c: 13 + 12 * c])
                         for c in range(CHUNKS)]
            print(f"5. M* Cosmos3 AR ctx={ctx} vs exact stream, max uint8 diff per chunk = {per_chunk}")

        from mstar.model.wan22.config import Wan22Config
        from mstar.model.wan22.submodules import Wan22VaeDecoderSubmodule

        torch.cuda.mem_get_info = lambda *a, **k: (1 << 40, 1 << 40)
        os.environ["WAN22_VAE_DECODE_FP32"] = "1"
        cfg = Wan22Config(vae_z_dim=16, vae_scale_factor_spatial=8)
        w22 = Wan22VaeDecoderSubmodule(load_vae(WAN21_REPO, torch.float32, "cpu"), cfg)
        v = w22.forward("video_gen", engine_inputs=None, latents=norm)["video_output"][0]
        print(f"5b. M* Wan22 node on Wan2.1 VAE: {tuple(v.shape)} {v.dtype}, "
              f"max uint8 diff vs exact = {maxdiff(v, exact_u8):.0f}")


if __name__ == "__main__":
    main()
