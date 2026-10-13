# SGLang Wan VAE streaming decode, same method as M*

SGLang's own Wan VAE (`sglang.multimodal_gen`), loaded by SGLang's loader
under the pipeline config SGLang registers for each model, driven through
this benchmark's stream protocol: first latent alone, 3-latent chunks, 10
warm-up and 75 timed chunks replayed twice, CUDA events around decode + uint8
conversion, median per chunk, PSNR against the same fp32 reference.

| | |
|---|---|
| SGLang | `sglang[diffusion]==0.5.21` from PyPI, tag commit `e00930c5489053f26d86b179cee0d087f846acbb`; includes [sgl-project/sglang#34125](https://github.com/sgl-project/sglang/pull/34125) (merged as `6424fec3`) |
| Stack | torch 2.13.0+cu130 (SGLang's pin), cuDNN 9.20, Triton 3.7.1, Python 3.12 |
| Code path | `DecodingStage.decode` (latent scale/shift, autocast, SGLang's compile wrapper with `--compile`, clamp, [0, 1]) under `use_vae_fast_path(vae, quality_allows_kernel_fusions(quality))`, with the decode function pointed at `AutoencoderKLWan.causal_decode`: SGLang's streaming decode, cache kept across calls, one latent frame per decoder call |
| VAE | `VAELoader.load_customized` with `WanT2V480PConfig` (Wan 2.1) / `Wan2_2_TI2V_5B_Config` (Wan 2.2), `num_gpus=1`: channels-last-3D Conv3d weights and the quality-gated CUDA fast-path wrappers; then cast to the decode dtype (`resolve_decode_precision`) |

## Settings

| row | variant | flags |
|---|---|---|
| default, Wan 2.1 | `sglang_default` | serving default: `quality=lossless`, bf16, eager |
| default, Wan 2.2 | `sglang_default` | serving default: `Wan2_2_TI2V_5B_Config` decodes in fp32 (TF32 convs) |
| default bf16, Wan 2.2 | `sglang_default_bf16` | `--decode-precision bf16` (`--vae-decode-precision bf16`) |
| best, Wan 2.1 | `sglang_best` | `--quality extra-high` (same VAE path as `high`) |
| best, Wan 2.2 | `sglang_best` | `--quality extra-high --decode-precision bf16` |
| compile | `sglang_compile`, `sglang_compile_bf16` | `--compile` (`--enable-torch-compile`) |
| realtime stage | `sglang_realtime` | `--path realtime`: `CausalVaeDecodingStage.decode_causal` as is (fp32) |

SGLang exposes no `cudnn.benchmark` and no CUDA graphs for the VAE on CUDA
(compile `reduce-overhead` crashes on the cache outliving the replay).

## Reproduce

```bash
SGL_VENV=./sgl-venv bash setup.sh               # venv, VAEs at the pinned revisions, diffusers 0.39 side install
# SGL_INPUTS/<vae>/{latents.pt,ref_fp32_outputs.pt}: run_all.sh's latents and fp32 references
SGL_INPUTS=/path/to/inputs GPU=0 bash reproduce.sh   # screen, 3 runs, profiles, frame md5s
python summarize_sglang.py results --flops-dir ../flops --out results/summary
```

`run_sglang.sh` is the fresh-process runner (fresh Inductor/Triton caches,
refuses a busy GPU, logs an `nvidia-smi` trace). `md5_frames.py` writes the
uint8 frame stream of the whole stream (1,021 frames) per decoder and its md5:
SGLang chunked (`stream`), SGLang whole-clip `DecodingStage` with the stock
`AutoencoderKLWan.decode` (`whole`, the path #34125 measured), and stock
diffusers bf16 through M*'s `common.py` (`diffusers`, with diffusers 0.39 on
`PYTHONPATH`).

`bench_sglang_wan.py` carries its own copies of the benchmark's timing loop,
`pct`, `psnr`, `to_uint8` and `precision_flags` rather than importing
`bench_wan_decoder.py`: it runs in SGLang's venv (torch 2.13, diffusers 0.37,
no M* package), and the committed file is the one the numbers were measured
with. `pct`, `psnr` and `to_uint8` are M*'s formulas verbatim; `Timer` is M*'s
CUDA-event timer without the CPU branch, and `precision_flags` records the
same flags.
`kernel_categories_sglang.py` is M*'s `kernel_categories.py` plus rules for
SGLang's hand-written Triton kernels (`_cat_pad_cl3d_kernel`,
`_dup_up3d_add_kernel`, ...), which M*'s rules would put in "other".
