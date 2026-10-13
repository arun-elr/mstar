# Prompt-to-video samples for PR #425: prompts, settings, commands

## Model and generation

- Model: `Wan-AI/Wan2.2-TI2V-5B-Diffusers`, revision `b8fff7315c768468a5333511427288870b2e9635`
  (text-to-video mode: no image input).
- Path: diffusers `WanPipeline` with `output_type="latent"`, not M*'s `wan22` model. The `wan22` node
  graph hands its final latents straight to its own whole-clip `vae_decoder` node, and there is no
  switch to dump them. The pipeline's latent output is the DiT's normalized latents, the same thing
  M*'s decoder nodes take as input.
- Settings, all from the model card's diffusers T2V example:
  - resolution 1280×704 (W×H);
  - 121 frames, 24 fps: 5.04 s, 31 latent frames `[1, 48, 31, 44, 80]`;
  - 50 steps (`UniPCMultistepScheduler`, flow_shift 5.0, as shipped);
  - guidance scale 5.0;
  - transformer and text encoder in bf16, VAE loaded in fp32.

  The model card uses 121 frames, not 81. 121 frames is also 1 clip-start latent plus exactly ten
  3-latent chunks, so the streaming decode has no ragged tail.
- Seed: 42 for both prompts, `torch.Generator(device="cuda").manual_seed(42)`.
- Negative prompt: the model card's, verbatim:

  > 色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走

| id | prompt | latents md5 (fp32) |
|---|---|---|
| `fox` | A red fox running through fresh snow in a pine forest, morning light, slow tracking shot | `1a2bfc9a73400ffa599794777b7562b2` |
| `waves` | Waves rolling onto a rocky beach at sunset, seagulls in the distance | `261f483b6bbaf7a9b157f9539fdfc790` |

The latents are not included here; the commands below regenerate them.

## Software and hardware

- GPU: 1× NVIDIA B200, driver 580.173.02.
- Plain venv (uv-managed CPython 3.12.15, not the vLLM-Omni container):
  - torch 2.12.1+cu130 (CUDA 13.0, cuDNN 9.20.0), Triton 3.7.1;
  - diffusers 0.39.0, transformers 5.19.0.
- M* code: PR #425 at `c56b572bf8186a5d93e290f0f0864a4646ec106b`, copied with `git archive HEAD`. The
  worktree was not modified.

## Decode

Each ladder step decodes the same latents through M*'s in-tree streaming decoder node,
`Cosmos3VAEDecoderARSubmodule.forward` (`mstar/model/cosmos3/submodules.py`, which calls
`mstar/model/components/wan_vae.py`). The node is fed one windowed request:

- the clip-start latent, then ten 3-latent windows;
- overlap 0, `stream_video=True`;
- the bf16 decode the node picks on cuDNN ≥ 9.16;
- `COSMOS3_COMPILE_VAE` at its default (on).

M*'s `wan22` node decodes whole clips with `vae.decode`, so the streaming path here is the Cosmos3
node, which uses the same Wan 2.2 VAE.

Each step runs in a fresh process and streams the clip twice. Pass 1 compiles and warms up, and runs
cuDNN's algorithm search where it is enabled. Pass 2 is timed (CUDA-synchronized around each window)
and its frames are kept. Both passes produce the same md5 for every step.

| step | `windowed_decode_warm_cache` | `vae_decode_channels_last` | `windowed_decode_latents_per_call` | `vae_decode_cudnn_benchmark` |
|---|---|---|---|---|
| M* before | False (re-decodes 8 context latents per window) | False | 1 (unused) | False |
| + warm cache | True | False | 1 | False |
| + channels-last | True | True | 1 | False |
| + one call per chunk | True | True | 0 | False |
| + native padding | no separate switch: `ChannelsLastCausalConv3d` always pads inside the conv, so this lands with channels-last | | | |
| + cudnn.benchmark (`=0`, opt-in) | True | True | 0 | True |
| control: warm cache, eager | True (`COSMOS3_COMPILE_VAE=0`) | False | 1 | False |

There are two references:

- **Stock** (the reference for all metrics): diffusers `AutoencoderKLWan.decode` on the whole clip,
  eager, bf16 VAE. Its input is the bf16 `z` the node's own `_vae_input` computes, so the decoder is
  the only thing that differs.
- **Pipeline fp32** (context only): the model card's decode, with an fp32 VAE and `WanPipeline`'s
  fp32 de-normalization.

The uint8 conversion is `(x / 2 + 0.5).clamp(0, 1) * 255 → uint8` everywhere, the formula M*'s nodes
use.

## Commands

The scripts are in [`benchmark/wan_vae_decoder/samples/`](../../../samples/). In a venv with torch
2.12.1, diffusers 0.39.0 and M*'s base dependencies, and the model snapshot in the HF cache
(`hf download Wan-AI/Wan2.2-TI2V-5B-Diffusers --revision b8fff7315c768468a5333511427288870b2e9635`):

```bash
cd benchmark/wan_vae_decoder/samples
WORK=./work PY=python bash run_all.sh   # latents (gen_latents.py), every step (decode_step.py),
                                        # media and metrics (analyze.py), metrics.csv (tables.py)
```

`analyze.py`'s defaults are the media settings used here:

- per-step mp4s: frames 37–60 at CRF 20 (`--step-start 37 --step-frames 24 --crf 20`);
- side-by-side panels 480 px wide (`--sbs-panel-width 480`);
- GIF 480 px wide at 8 fps (`--gif-width 480 --gif-fps 8`);
- contact-sheet panels 320 px wide (`--sheet-panel-width 320`).

The metrics are the same whatever the media settings.

The equivalent generation call:

```python
pipe = WanPipeline.from_pretrained(path, vae=AutoencoderKLWan.from_pretrained(path, subfolder="vae",
                                   torch_dtype=torch.float32), torch_dtype=torch.bfloat16).to("cuda")
latents = pipe(prompt=PROMPT, negative_prompt=NEGATIVE, height=704, width=1280, num_frames=121,
               guidance_scale=5.0, num_inference_steps=50, output_type="latent",
               generator=torch.Generator(device="cuda").manual_seed(42)).frames
```

## License

Wan 2.2 TI2V-5B is released under the Apache 2.0 license (`license: apache-2.0` on the model card).
The card's License Agreement says the authors "claim no rights over the your generated contents,
granting you the freedom to use them while ensuring that your usage complies with the provisions of
this license". It also excludes harmful or unlawful content, which these neutral nature clips are
not. So these sample videos can be shared in the PR.
