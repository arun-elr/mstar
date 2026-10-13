# vLLM-Omni Wan VAE streaming decode on one B200 (M* benchmark method)

Same latents and fp32 reference tensors as M*'s B200 runs (sha256 below), same chunking, timing,
PSNR convention and JSON schema as `bench_wan_decoder.py`. ms per 3-latent chunk (12 frames at
480×832), median of 150 steady samples per run, 3 fresh-process runs per variant.

## Headline

| VAE | variant | vLLM-Omni configuration | run 1 / 2 / 3 (ms) | mean ms | spread | PSNR peak 1 (dB) | PSNR uint8 (dB) |
|---|---|---|---|---:|---:|---:|---:|
| Wan 2.1 | `vllm_omni_default` | `vae_fast_path=lossless` (engine default), eager | 97.22 / 97.44 / 97.31 | **97.32** | 0.22 | 53.73 | 54.91 |
| Wan 2.1 | `vllm_omni_best` | `--vae-fast-path channels_last`, eager | 63.99 / 64.03 / 64.22 | **64.08** | 0.23 | 53.88 | 54.88 |
| Wan 2.2 | `vllm_omni_default` | `vae_fast_path=lossless`, eager | 106.01 / 105.85 / 105.90 | **105.92** | 0.16 | 56.17 | 55.27 |
| Wan 2.2 | `vllm_omni_best` | `--vae-fast-path channels_last`, eager | 77.80 / 77.86 / 77.88 | **77.85** | 0.08 | 56.42 | 55.16 |

For `comparison.csv`: the row "vLLM-Omni, Wan VAE fast path, channels-last" (reported elsewhere
66.2) is `vllm_omni_best`, measured 64.08 ms (3-run mean). vLLM-Omni's out-of-the-box decode is
`vllm_omni_default`, 97.32 ms.

Context from M*'s own runs (M*'s environment): stock diffusers eager 191.98 / 184.57 ms, M* this
change `px_nopad_bench` 55.74 / 72.99 ms (Wan 2.1 / Wan 2.2).

## What was measured

- **Code path.** `vllm_omni.experimental.ar_diffusion.streaming_decode.WanStreamingDecoder.decode_chunk`:
  the session-owned warm-cache decoder LingBot-World serves AR blocks with
  (`LingBotWorldPipeline._streaming_decode_chunk`). It enters the VAE's `_execution_context()`
  (bf16 autocast), runs `post_quant_conv` and `decoder` **one latent frame per decoder call** with
  the causal-conv cache carried across calls, then unpatchify (Wan 2.2) and clamp. This is
  vLLM-Omni's only streaming decode. It has no mode that decodes a chunk in one decoder call.
- **VAE construction** follows `vllm_omni.diffusion.registry.initialize_model`:
  `DistributedAutoencoderKLWan.from_pretrained(..., torch_dtype=bfloat16)` in a vLLM-Omni world of
  size 1 (`init_distributed_environment(world_size=1)`), `use_slicing=False`, `use_tiling=False`
  (OmniDiffusionConfig default), `set_parallel_size(1, "tile")`, then
  `install_wan_vae_fastpath(vae, level=vae_fast_path)`, which is what `_apply_wan_vae_fastpath_if_enabled`
  does. The report confirmed `installed=True, fused_silu=[bfloat16, float16]` for both levels.
  No `vllm serve` process: like M*'s numbers, this drives the decoder directly, so there is no scheduler or IPC overhead.
- **Chunking.** Latent 0 is decoded alone first (untimed; the session's opening frame decodes to one pixel frame).
  Then come chunks of latents `[1+3c, 4+3c)`, 12 frames each: 10 warm-up chunks (untimed; absorbs
  Triton JIT, the fast path's one-time bitwise verdict checks and cuDNN autotuning) and 75 timed chunks. The stream is
  replayed twice from a fresh session state and the 150 samples are pooled. Timed with CUDA events around
  `decode_chunk` + M*'s `to_uint8`, synchronized per chunk. This is `bench_wan_decoder.run_streaming`
  unmodified, with the decoder object swapped.
- **Runs.** One variant per process, each in a fresh container (cold Triton cache), orders
  ABC / CBA / BCA. The GPU was checked empty before every run. SM clock held at 1965 MHz under load (`runs/main_smi.csv`).
- **Precision.** bf16 weights and autocast. TF32 off (`cuda.matmul.allow_tf32 = cudnn.allow_tf32 = False`,
  recorded per JSON; irrelevant for bf16 convs). `cudnn.benchmark` off (vLLM-Omni does not set it).
- **PSNR.** On the float output in [-1, 1] over the first 10 steady chunks against M*'s fp32
  reference (diffusers fp32, eager, TF32 off). Peak 1 (`psnr_float_db_vs_fp32_peak1`), plus uint8. The default path is
  bit-identical to stock diffusers bf16 (53.73 dB, the same as M*'s `diffusers_bf16`), as vLLM-Omni documents for `lossless`.

### How `vllm_omni_best` was chosen

These are all of vLLM-Omni's Wan VAE switches, each run once on Wan 2.1 (`runs/screen/`, `runs/main/`):

| configuration | ms | note |
|---|---:|---|
| `vae_fast_path=off` | 191.92 | reference diffusers forwards (= M*'s stock diffusers in this container: 191.88) |
| `vae_fast_path=lossless` (default) | 97.32 | 3-run mean |
| `vae_fast_path=channels_last` | **64.08** | 3-run mean → `vllm_omni_best` |
| LingBot `lingbot_vae_decode_fast_path=exact` | 191.87 | blocks the registry fast path (its upsample rebinding trips the installer's custom-forward check); with bf16 weights on one GPU it only rewrites 3 upsamples |
| LingBot `lingbot_vae_decode_fast_path=fused` | 110.07 | same blocking; channels-last + fused RMSNorm+SiLU, no fused cat/pad |
| `channels_last` + `torch.backends.cudnn.benchmark=True` | 62.60 (Wan 2.2: 77.72) | 3-run mean; a torch global flag, not a vLLM-Omni switch, so it is reported separately (`vllm_omni_best_cudnnbench`) |

`torch.compile` is not an option for vLLM-Omni's VAE. No Wan pipeline compiles it, there is no switch, and the fast
path's Triton kernels turn themselves off under compile tracing.

### Whole-clip decode (supplementary)

`OmniAutoencoderKLWan.decode` on the first 241 frames (61 latents), which is what `Wan22Pipeline` runs after
denoising. The default fast path goes through `decode_frames`, one latent per decoder call. Each figure is the median of 5 decodes after one warm-up, scaled to
ms per 12 frames as `clip_ms × 12 / 241`. PSNR is taken on frames `121 + 12k ... 132 + 12k` (k = 0..9), the same frames as
the streaming reference.

| VAE | tiling | clip ms | ms per 12 frames | PSNR peak 1 |
|---|---|---:|---:|---:|
| Wan 2.1 | off (default) | 1947.6 | 96.97 | 53.73 |
| Wan 2.2 | off | 2129.9 | 106.05 | 56.17 |
| Wan 2.2 | on (`deploy/wan2_2_ti2v.yaml` default) | 6589.9 | 328.13 | 47.45 (tile blending) |

## Kernel profile (torch.profiler, 6 steady chunks after 10 warm-up, `profile_kernels.py` + `kernel_categories.py`)

`kernel_categories.py` names Inductor's `triton_*` kernels. vLLM-Omni's own Triton kernels (`_rms_norm_cl_kernel`,
`_cat_time_planes_kernel`, ...) fall into "other" under it. The *refined* columns re-label only those, by name
(`_rms_norm*` → norm / SiLU, `_cat_*` → concat / pad, `_add_bias_residual*` / `_dup_up3d_add*` → other elementwise).
Both are in `profile.csv`, and every kernel name is in `runs/profile/*/<variant>.kernels.json` (`all_kernels`).
Achieved TFLOP/s = M*'s useful FLOPs per chunk (`flops_wan21.json` 40.65 TFLOP, `flops_wan22.json` 88.18 TFLOP) / 3-run wall ms.

| VAE | variant | kernels / chunk | GPU ms | conv | layout conv. | norm+SiLU* | concat/pad* | other elementwise* | achieved TFLOP/s | conv kernel TFLOP/s |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan 2.1 | `vllm_omni_default` | 999 | 96.06 | 39.48 | 22.87 | 11.58 | 5.75 | 13.09 | 418 | 1025 |
| Wan 2.1 | `vllm_omni_best` | 444 | 61.38 | 41.37 | 0.00 | 6.37 | 5.63 | 4.97 | 634 | 978 |
| Wan 2.1 | `vllm_omni_best_cudnnbench` | 423 | 60.82 | 40.41 | 0.00 | 6.55 | 5.64 | 5.10 | 649 | 1001 |
| Wan 2.2 | `vllm_omni_default` | 1078 | 103.79 | 56.50 | 19.77 | 10.00 | 4.59 | 7.39 | 833 | 1560 |
| Wan 2.2 | `vllm_omni_best` | 463 | 74.53 | 60.74 | 0.00 | 4.14 | 4.24 | 3.60 | 1133 | 1451 |
| Wan 2.2 | `vllm_omni_best_cudnnbench` | 499 | 74.31 | 59.56 | 0.00 | 4.22 | 4.52 | 3.66 | 1135 | 1480 |

\* refined categories. Attention is ≈2.0 ms (Wan 2.1) and ≈0.9 ms (Wan 2.2), upsample 0.4–1.3 ms, copies 0.4–2.9 ms.
For reference, M*'s `px_nopad_bench` (M*'s environment) has 139 / 154 kernels per chunk, conv 37.97 / 58.35 ms, norm+SiLU 9.18 / 5.19 ms,
and 729 / 1208 TFLOP/s achieved. The default lossless level keeps activations channels-first, so cuDNN transposes around every
conv (22.9 ms per chunk on Wan 2.1). `channels_last` removes that, but it still issues 3 decoder calls per chunk (444 kernels).

## Fairness caveats

1. **Software stack differs from M*'s runs.** This used the vLLM-Omni image: torch 2.13.0+cu130, diffusers 0.40.0, Triton 3.7.1.
   M*'s runs used torch 2.12.1+cu132 and diffusers 0.39.0. cuDNN (9.20.0), driver (580.173.02) and GPU model are the same, but this
   is a different B200 VM. Controls run inside this container (`runs/controls/`):
   - Eager stock diffusers carries over: 191.88 vs 191.98 ms (Wan 2.1) and 184.30 vs 184.57 ms (Wan 2.2).
   - M*'s compiled prototype `px_nopad_bench` is faster here: 53.45 ms vs 55.74 ms on Wan 2.1, PSNR 54.68 vs 53.91 dB.
     On the same stack, the M*-vs-vLLM-Omni gap is therefore ≈10.6 ms, not the ≈8.3 ms you get from
     mixing environments.
   - On Wan 2.2 the M* prototype gives wrong output in this container (PSNR 13.87 vs 62.58 dB at peak 2 in M*'s environment;
     the `patch_causal_conv_native_pad` monkeypatch is likely incompatible with diffusers 0.40). That control's 72.04 ms is
     invalid and is kept only for the record.
2. **One decoder call per latent.** vLLM-Omni's streaming decoder cannot decode a chunk in one call, and it does not compile. Both are
   properties of vLLM-Omni being measured, not harness choices.
3. **Wan 2.2 streaming is not a served vLLM-Omni path.** Only LingBot-World (Wan 2.1 VAE) streams. The Wan 2.2 TI2V pipeline decodes
   whole clips with tiling on by default (328 ms per 12 frames above). The Wan 2.2 streaming numbers use the same
   VAE-agnostic `WanStreamingDecoder` with the registry fast path.
4. **Timed region excludes the host copy.** LingBot's `_uint8_frames` also permutes and rounds, then copies uint8 frames to the host.
   Like M*, only the GPU decode plus `to_uint8` is timed.
5. **Peak memory** includes the VAE encoder weights (vLLM-Omni keeps them loaded; M* drops them).

## Versions

- vLLM-Omni `0.31.0rc2.dev98+gf69b1f2b1`, commit `f69b1f2b19cece03f6736c7da093ea25de0e24fd` (matches `~/oss/vllm-omni` HEAD); vLLM 0.31.0
- Docker image `vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd`,
  digest `vllm/vllm-omni@sha256:d9d81b77c0c5055425f388ddde756ec4206c53e141d1ca6d1cbee21b2ba24e9f`
- torch 2.13.0+cu130 (CUDA 13.0), cuDNN 9.20.0 (`92000`), diffusers 0.40.0, Triton 3.7.1, Python 3.12.3
- GPU: 1× NVIDIA B200 (Nebius `gpu-b200-sxm`, `1gpu-20vcpu-224gb`), driver 580.173.02, power limit 1000 W

## Inputs (sha256, identical to `wan-bench-b200:~/wan-bench/results/b200-wan2{1,2}/wan2{1,2}/`)

```
40b622fcb24d36be0bc821254cdde77418ac92ea8938c53e3753f1db9f8ed7d4  wan21/latents.pt
f31b53acbe8e77fa4c41a450670d3ea111d5dcf064b4d00b6225c56da099338b  wan21/ref_fp32_outputs.pt
7e2d6ddbeb8846c48213b5c7886f07fde5ab3a4d7936b49f92996fe3748ac74a  wan22/latents.pt
659d8ec50f0c7502ef6e4dde69f61103a880daaadb786d850fd4d982af651b93  wan22/ref_fp32_outputs.pt
d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793  Wan2.1-T2V-1.3B-Diffusers@0fad780a vae/diffusion_pytorch_model.safetensors
62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b  Wan2.2-TI2V-5B-Diffusers@b8fff731 vae/diffusion_pytorch_model.safetensors
```

## Files

- `variance.csv`, `runs.csv`, `profile.csv`: the column layout of M*'s `summarize.py` (profile adds the `refined_ms_*` columns)
- `runs/main/<vae>/run{1,2,3}/`: headline JSONs (`bench_wan_decoder.py` schema plus `vllm_omni_*` / `vae_fast_path*` fields
  and vLLM-Omni / image versions in `env`), with `.latencies.json`
- `runs/screen/`, `runs/controls/`, `runs/oneshot/`, `runs/profile/`; `runs/main.log`; `runs/main_smi.csv`
- `harness/`: `bench_vllm_omni.py` (imports M*'s `bench_wan_decoder.py`, `common.py`, `profile_kernels.py`,
  `kernel_categories.py` unchanged), `run1.sh` (one fresh container per run), `main.sh` (the batch), `summarize_vo.py`
