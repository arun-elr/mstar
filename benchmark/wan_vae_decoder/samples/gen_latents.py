"""Text-to-video latents with Wan 2.2 TI2V-5B through diffusers' WanPipeline
(model-card settings), saved before the VAE decode.

  python gen_latents.py --prompt-id fox --out runs/fox_latents.pt
"""

import argparse
import json
import time

import torch
from diffusers import AutoencoderKLWan, WanPipeline
from huggingface_hub import snapshot_download

REPO = "Wan-AI/Wan2.2-TI2V-5B-Diffusers"
REVISION = "b8fff7315c768468a5333511427288870b2e9635"
PROMPTS = {
    "fox": "A red fox running through fresh snow in a pine forest, morning light, slow tracking shot",
    "waves": "Waves rolling onto a rocky beach at sunset, seagulls in the distance",
}
# The model card's negative prompt, verbatim.
NEGATIVE = ("色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，"
            "JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，"
            "手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走")
SETTINGS = dict(height=704, width=1280, num_frames=121, guidance_scale=5.0, num_inference_steps=50)
SEED = 42


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prompt-id", required=True, choices=sorted(PROMPTS))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    path = snapshot_download(REPO, revision=REVISION, local_files_only=True)
    t0 = time.perf_counter()
    vae = AutoencoderKLWan.from_pretrained(path, subfolder="vae", torch_dtype=torch.float32)
    pipe = WanPipeline.from_pretrained(path, vae=vae, torch_dtype=torch.bfloat16).to("cuda")
    load_s = time.perf_counter() - t0

    gen = torch.Generator(device="cuda").manual_seed(SEED)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    latents = pipe(prompt=PROMPTS[args.prompt_id], negative_prompt=NEGATIVE, generator=gen,
                   output_type="latent", **SETTINGS).frames
    torch.cuda.synchronize()
    gen_s = time.perf_counter() - t0

    meta = {"repo": REPO, "revision": REVISION, "prompt_id": args.prompt_id, "prompt": PROMPTS[args.prompt_id],
            "negative_prompt": NEGATIVE, "seed": SEED, "generator": "torch.Generator('cuda').manual_seed",
            **SETTINGS, "transformer_dtype": "bfloat16", "pipeline": "diffusers.WanPipeline, output_type='latent'",
            "latent_shape": list(latents.shape), "latent_dtype": str(latents.dtype),
            "load_s": round(load_s, 2), "pipeline_s_to_latents": round(gen_s, 2),
            "torch": str(torch.__version__), "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "gpu": torch.cuda.get_device_name()}
    import diffusers
    import transformers
    meta |= {"diffusers": diffusers.__version__, "transformers": transformers.__version__}
    # The pipeline's own output: normalized latents, what the DiT hands a decoder node.
    torch.save({"latents": latents.float().cpu(), "meta": meta}, args.out)
    print(json.dumps({k: v for k, v in meta.items() if k != "negative_prompt"}))


if __name__ == "__main__":
    main()
