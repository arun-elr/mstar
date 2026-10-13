# vLLM-Omni Wan VAE streaming decode, same method as M*

vLLM-Omni's own decoder, driven through this benchmark's stream protocol,
timing and PSNR code. `bench_vllm_omni.py` imports `bench_wan_decoder.py`,
`common.py` and `profile_kernels.py` from `benchmark/wan_vae_decoder/`
unchanged (the directory is mounted into the container and put on
`PYTHONPATH`); only the decoder object differs.

| | |
|---|---|
| vLLM-Omni | `0.31.0rc2.dev98+gf69b1f2b1`, commit `f69b1f2b19cece03f6736c7da093ea25de0e24fd`; vLLM 0.31.0 |
| Image | `vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd`, digest `sha256:d9d81b77c0c5055425f388ddde756ec4206c53e141d1ca6d1cbee21b2ba24e9f` |
| Stack in the image | torch 2.13.0+cu130, cuDNN 9.20.0, diffusers 0.40.0, Triton 3.7.1, Python 3.12.3 |
| Code path | `vllm_omni.experimental.ar_diffusion.streaming_decode.WanStreamingDecoder.decode_chunk`, the session-owned warm-cache decoder LingBot-World serves AR blocks with: `post_quant_conv` and the decoder one latent frame per call, cache carried across calls, under the VAE's `_execution_context()` (bf16 autocast) |
| VAE | `DistributedAutoencoderKLWan.from_pretrained(..., torch_dtype=bfloat16)` in a vLLM-Omni world of size 1, no slicing or tiling, then `install_wan_vae_fastpath(vae, level=...)`, as `vllm_omni.diffusion.registry.initialize_model` does |

## Settings

| row | variant | setting |
|---|---|---|
| default | `vllm_omni_default` | `vae_fast_path=lossless` (the engine default), eager |
| best | `vllm_omni_best` | `--vae-fast-path channels_last`, eager |
| best + cudnn.benchmark | `vllm_omni_best_cudnnbench` | `channels_last` plus `torch.backends.cudnn.benchmark=True` (a torch flag; vLLM-Omni does not set it) |

`vae_fast_path=off` and LingBot-World's `lingbot_vae_decode_fast_path`
(`exact`, `fused`) are screened once each; none is faster than `channels_last`.
vLLM-Omni has no compile option for its VAE: no Wan pipeline compiles it,
and its Triton fast-path kernels switch themselves off under compile tracing.
Its streaming decoder has no mode that decodes a chunk in one decoder call.

## Reproduce

```bash
docker pull vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd
# VO_DATA/inputs/<vae>/{latents.pt,ref_fp32_outputs.pt}: run_all.sh's latents and fp32 references
# VO_DATA/vae/<vae>/: the VAE folders (config.json + safetensors) of
#   Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a and Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff731
VO_DATA=/path/to/data GPU=0 bash main.sh        # 3 runs x 3 variants x 2 VAEs, screen, controls, profiles
python summarize_vo.py "$VO_DATA/runs" --out "$VO_DATA/summary"   # variance.csv, runs.csv, profile.csv
```

`run1.sh` runs one variant in a fresh container (cold Triton cache) and
refuses a GPU with other compute processes. One row by hand:
`VO_DATA=... bash run1.sh main/wan21/run1 wan21 vllm_omni_best`.

Achieved TFLOP/s uses M*'s useful FLOPs per chunk (`../flops/`, from
`profile_kernels.py --flops`).
