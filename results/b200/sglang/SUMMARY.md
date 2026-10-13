# SGLang (sglang.multimodal_gen) Wan VAE streaming decode on one B200, M* methodology

SGLang v0.5.21 (commit `e00930c5`, includes PR #34125), one NVIDIA B200, same
latents, fp32 references, VAE weights and stream protocol as M*'s
`benchmark/wan_vae_decoder` (3 latents per chunk, 10 warm-up + 75 timed chunks,
2 replays pooled, median ms per chunk, 3 fresh-process runs).

| VAE | variant | SGLang settings | ms / chunk (3-run mean) | runs | PSNR float (peak 1) | PSNR u8 |
|---|---|---|---|---|---|---|
| Wan 2.1 | `sglang_default` | serving default: `quality=lossless`, bf16 autocast, channels-last-3D, eager | **107.64** | 107.63 / 107.58 / 107.71 | 53.88 dB | 55.27 dB |
| Wan 2.1 | `sglang_best` | `--quality extra-high` (same VAE path as `high`), eager | **92.50** | 92.50 / 92.49 / 92.52 | 53.73 dB | 55.23 dB |
| Wan 2.1 | `sglang_compile` (ref) | `--enable-torch-compile` | 102.16 | 102.03 / 102.08 / 102.37 | 53.91 dB | 55.29 dB |
| Wan 2.1 | `sglang_realtime` (ref) | `CausalVaeDecodingStage` as is (fp32, TF32 convs) | 145.04 | 145.00 / 145.14 / 144.97 | 80.00 dB | 68.25 dB |
| Wan 2.2 | `sglang_default` | serving default: `Wan2_2_TI2V_5B_Config` decodes **fp32** (TF32 convs) | **170.97** | 170.76 / 171.10 / 171.05 | 82.04 dB | 69.40 dB |
| Wan 2.2 | `sglang_default_bf16` | default + `--vae-decode-precision bf16` | 115.72 | 115.72 / 115.74 / 115.70 | 56.42 dB | 56.79 dB |
| Wan 2.2 | `sglang_best` | `--vae-decode-precision bf16 --quality extra-high`, eager | **94.36** | 94.45 / 94.29 / 94.33 | 56.16 dB | 56.85 dB |
| Wan 2.2 | `sglang_compile_bf16` (ref) | bf16 + `--enable-torch-compile` | 110.96 | 111.81 / 110.42 / 110.65 | 56.43 dB | 56.79 dB |

Spread is at most 0.34 ms (0.33%) for every row except `sglang_compile_bf16`
(1.39 ms, 1.25%). For reference, M* on the same latents (`results/b200/`) gets
55.74 ms for Wan 2.1 (`px_nopad_bench`, 3-run mean) and 72.99 ms for Wan 2.2
(single run), 191.98 ms for stock diffusers bf16, and 95.90 ms for diffusers +
`torch.compile`. The PR's "reported elsewhere" SGLang figure (101.9 ms,
"bf16, channels-last, torch.compile") is reproduced by `sglang_compile`
(101.87 ms in the screen, 102.16 ms as the 3-run mean), but that is neither
SGLang's default nor its fastest setting.

Suggested `comparison.csv` rows (same-method numbers go in `measured_ms`):

```
SGLang v0.5.21,"default serving decode: quality=lossless, bf16, channels-last-3D, eager",107.64,"measured, 3-run mean",,this benchmark
SGLang v0.5.21,"best documented: quality=extra-high, bf16, channels-last-3D, eager",92.50,"measured, 3-run mean",,this benchmark
SGLang v0.5.21,"bf16, channels-last, torch.compile (--enable-torch-compile)",102.16,"measured, 3-run mean",101.9,this benchmark
```

## Version check: PR #34125

[sgl-project/sglang#34125](https://github.com/sgl-project/sglang/pull/34125)
(bit-exact data-movement removal for the Wan causal VAE decoder) merged
2026-08-09 as `6424fec3`. The v0.5.21 tag commit `e00930c5` (2026-09-30) is
2763 commits ahead of `6424fec3` and 0 behind (GitHub compare), so v0.5.21
contains it. The PR's new module was later moved by #35114 (2026-08-18,
also in v0.5.21) to `kernels/ops/diffusion/layout/wan_causal_cache_triton.py`.
The installed package has `cat_pad_channels_last_3d`, `dup_up3d_add` and
`_run_cached_causal_conv` in `wanvae.py`, and the profiles show
`_cat_pad_cl3d_kernel` (96 calls per chunk) and, on Wan 2.2,
`_dup_up3d_add_kernel` (9 calls per chunk) running. No install from `main` was
needed; v0.5.21 is the primary (and only) version measured.

Not measured: `main` has later Wan VAE commits after v0.5.21 (#41830 moves
first-decode work off the request path, #42866 and #43064 are refactors).
Whether they change steady-state chunk time is unverified.

## Code path and chunking

- **VAE load**: SGLang's `VAELoader.load_customized` with the `ServerArgs` /
  pipeline config SGLang registers for the hub ID (`WanT2V480PConfig` for
  Wan 2.1, `Wan2_2_TI2V_5B_Config` for Wan 2.2, `num_gpus=1`). That applies
  SGLang's load-time switches: all 35 / 62 Conv3d weights in channels-last-3D
  (`_should_use_channels_last_3d`), and the quality-gated CUDA fast-path
  wrappers (`current_platform.optimize_vae` -> `maybe_optimize_wan_vae`).
  The VAE is then cast to the decode dtype (`resolve_decode_precision`), which
  is what the decode stage's component residency (`ComponentUse.target_dtype`)
  does in serving.
- **Per chunk**: `DecodingStage.decode(latents, server_args, vae_dtype=...)`
  under `use_vae_fast_path(vae, quality_allows_kernel_fusions(quality))`, as in
  `DecodingStage.forward`: scale/shift of the latents, `autocast_context`,
  SGLang's compile wrapper (`_get_vae_decode_fn`) when
  `--enable-torch-compile`, clamp, and conversion to [0, 1]. The VAE decode
  function is pointed at `AutoencoderKLWan.causal_decode`, SGLang's
  streaming API that keeps the causal conv feature cache warm across calls
  and runs the decoder one latent frame per call. Before each replay:
  `reset_causal_decode_state()`, then the first latent alone.
- **Input**: M*'s raw latents normalized as `(raw - mean) / std` (fp32 on GPU),
  what the DiT hands the decode stage. The harness asserts SGLang's
  `scale_and_shift_latents` inverts that.
- **Timed**: CUDA events around `DecodingStage.decode` plus
  `img.mul(255).to(uint8)`, synchronized per chunk. That is SGLang's own frame
  conversion (`(x * 255).clamp(0, 255).to(uint8)` in
  `entrypoints/utils.py`) and byte-identical to M*'s `to_uint8`.
- **Same path as PR #34125**: the PR timed `DecodingStage` with the stock
  whole-clip `AutoencoderKLWan.decode` (lossless and `--quality high`).
  Decoding all 256 latents in one call this way yields **md5-identical** uint8
  frames to our chunked stream, for default and best on both VAEs (table
  below). The chunked numbers measure the same kernels and the same numerics.
  Flags follow SGLang's benchmark guide
  (`.claude/skills/sglang-diffusion-benchmark-profile/benchmark-and-profile.md`):
  eager is the baseline, `--quality {extra-high,high}` are the documented fast
  tiers, and `torch.compile` is a labeled control. BCG (breakable CUDA graph)
  covers the DiT only.
- **`sglang_realtime`**: `CausalVaeDecodingStage.decode_causal` unchanged
  (the realtime stage the LingBot-World causal pipeline uses). It casts the
  VAE to `vae_precision`, which is fp32 for every Wan config, and uses no
  compile and no quality gate.

## Screened settings (one fresh-process run each)

| VAE | setting | ms / chunk | note |
|---|---|---|---|
| Wan 2.1 | lossless (default) | 107.64 | |
| Wan 2.1 | `quality=extra-high` | 92.46 | fastest; picked as `sglang_best` |
| Wan 2.1 | `quality=high` | 92.50 | identical VAE path to extra-high (no `vae_decode_precision_high` for Wan T2V) |
| Wan 2.1 | `--enable-torch-compile` | 101.87 | 148 graphs: Dynamo breaks on every `ContextVar.get/set` in `wanvae.py` |
| Wan 2.1 | compile + extra-high | 102.12 | gated kernels switch off under compile |
| Wan 2.1 | compile, `SGLANG_VAE_TORCH_COMPILE_MODE=max-autotune-no-cudagraphs` | 101.13 | 57 s first call |
| Wan 2.1 | compile, mode `reduce-overhead` | crash | "accessing tensor output of CUDAGraphs that has been overwritten" (the feature cache outlives the replay) |
| Wan 2.1 | realtime stage | 145.14 | fp32 |
| Wan 2.2 | lossless (default, fp32) | 170.89 | |
| Wan 2.2 | `quality=extra-high` (fp32) | 157.63 | |
| Wan 2.2 | `--vae-decode-precision bf16` | 115.67 | |
| Wan 2.2 | bf16 + `quality=extra-high` | 94.29 | fastest; picked as `sglang_best` |
| Wan 2.2 | bf16 + compile | 110.37 | |
| Wan 2.2 | bf16 + compile max-autotune-no-cudagraphs | 110.43 | |
| Wan 2.2 | realtime stage | 171.02 | fp32, same as default |

Conv inputs that are not channels-last (eager, one steady chunk): 6 of 115
(Wan 2.1) and 6 of 118 (Wan 2.2) at lossless; 0 at `extra-high`.

## Kernel profile (torch.profiler, 6 steady chunks after 10 warm-up)

ms of GPU time per chunk by category (`kernel_categories_sglang.py`). Achieved
TFLOP/s = M*'s useful FLOPs per chunk (40.65 TFLOP for Wan 2.1, 88.18 TFLOP for
Wan 2.2, from M*'s `flops_*.json`) divided by the 3-run wall time.

| VAE | variant | kernels / chunk | GPU ms | conv | layout conv. | norm / SiLU | concat / pad | upsample | attention | copies | other elementwise | achieved TFLOP/s | conv kernel TFLOP/s |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Wan 2.1 | `sglang_default` | 891 | 106.09 | 39.11 | 1.11 | 20.49 | 13.84 | 2.04 | 1.82 | 8.91 | 18.74 | 378 | 1035 |
| Wan 2.1 | `sglang_best` | 567 | 91.94 | 39.07 | 0.03 | 21.99 | 12.47 | 1.00 | 1.82 | 0.63 | 14.91 | 439 | 1036 |
| Wan 2.1 | `sglang_compile` | 843 | 95.23 | 39.36 | 0.03 | 7.96 | 13.27 | 8.67 | 1.87 | 2.36 | 21.69 | 398 | 1028 |
| Wan 2.1 | `sglang_realtime` | 829 | 143.67 | 75.20 | 1.40 | 20.73 | 14.51 | 2.46 | 5.08 | 1.50 | 22.79 | 280 | 538 |
| Wan 2.2 | `sglang_default` (fp32) | 863 | 167.91 | 110.32 | 1.53 | 12.81 | 12.54 | 1.79 | 2.21 | 2.68 | 24.02 | 516 | 799 |
| Wan 2.2 | `sglang_default_bf16` | 937 | 112.75 | 56.20 | 1.29 | 12.53 | 11.75 | 1.47 | 0.83 | 8.15 | 20.53 | 762 | 1569 |
| Wan 2.2 | `sglang_best` | 604 | 92.41 | 56.68 | 0.07 | 7.50 | 10.70 | 0.74 | 0.86 | 0.75 | 15.11 | 935 | 1555 |
| Wan 2.2 | `sglang_compile_bf16` | 874 | 102.65 | 56.38 | 0.03 | 4.77 | 10.68 | 6.17 | 0.83 | 1.86 | 21.90 | 795 | 1564 |

For comparison, M*'s `px_nopad_bench` profile: Wan 2.1 has 139 kernels per
chunk, conv 37.97 ms, norm / SiLU 9.18, concat / pad 1.08, other elementwise
1.19, 729 TFLOP/s achieved. Wan 2.2: conv 58.35 ms, 1208 TFLOP/s.

- SGLang's bf16 conv time already matches M*'s: the same CUTLASS sm100 NDHWC
  implicit-GEMM kernels at about 1030 to 1570 conv TFLOP/s. The whole gap is
  outside the convolutions. SGLang `sglang_best` spends about 53 ms per Wan 2.1
  chunk outside the convs, against about 17 ms for M*:
  - The fused RMSNorm+SiLU (`_wan_rmsnorm_silu_kernel`, 87 calls) takes 22.0 ms.
  - The causal cat/pad kernel (96 calls) takes 12.5 ms, where M* pads inside
    the conv.
  - Separate eager residual adds (118 calls) take 12.7 ms.
  - The decoder runs once per latent frame (3 calls per chunk) and is never
    fused across ops.
- The two fp32 rows (realtime stage, Wan 2.2 default) run TF32 CUTLASS convs
  at roughly half the bf16 conv throughput.

**Category mapping**: M*'s `kernel_categories.py` keys Triton kernels off
Inductor's `triton_` prefix, so SGLang's hand-written kernels would land in
"other". `kernel_categories_sglang.py` is M*'s rule set plus these rules, which
are checked first:

| SGLang kernel | M* rules | mapped to |
|---|---|---|
| `_cat_pad_cl3d_kernel` (causal-conv input build plus compact cache; 96 calls per chunk, 10.7 to 14.5 ms) | other | concat / pad |
| `_fused_cat_pad_5d_kernel` (fallback fused cat/pad; not seen in these runs) | other | concat / pad |
| `_dup_up3d_add_kernel` (Wan 2.2 `main + DupUp3D(shortcut)`; 9 calls, 2.0 to 4.0 ms) | other | other elementwise |
| `_nearest_upsample_nhwc_kernel` | upsample | upsample (unchanged) |
| `_wan_rmsnorm_silu_kernel`, `*norm_silu_post*` | norm / SiLU | norm / SiLU (unchanged) |

Each `*.kernels.json` also carries `category_ms_per_chunk_mstar_rules` (M*'s
unmodified rules) and the list of re-categorized kernels.

## uint8 frame-stream md5 (all 1021 frames: first latent + 85 chunks x 12)

The md5 is over raw THWC uint8 bytes. `diffusers_bf16` is M*'s
`diffusers_bf16` variant (stock diffusers 0.39.0 `AutoencoderKLWan`, eager bf16,
warm cache, one latent per call), on the same latents.

| VAE | stream | md5 | vs diffusers bf16: identical bytes | max abs diff | PSNR u8 |
|---|---|---|---|---|---|
| Wan 2.1 | diffusers bf16 | `ac03880592696422f09abbbc9c3d1899` | | | |
| Wan 2.1 | `sglang_default` (chunked) | `d5f5e738699ad986d00e3db17f158ceb` | 92.39% | 11 | 59.31 dB |
| Wan 2.1 | `sglang_default`, whole-clip `DecodingStage` | `d5f5e738699ad986d00e3db17f158ceb` | (identical to chunked) | | |
| Wan 2.1 | `sglang_best` (chunked) | `3f713e0d38bb2f2548a7032faf927404` | 94.20% | 8 | 60.49 dB |
| Wan 2.1 | `sglang_best`, whole-clip | `3f713e0d38bb2f2548a7032faf927404` | (identical to chunked) | | |
| Wan 2.1 | `sglang_compile` | `6bed787a5d50be7337db4e26ce8202ab` | 92.34% | 11 | 59.28 dB |
| Wan 2.1 | `sglang_realtime` (fp32) | `eae4d4fc3782b27dc97d58ba77aba296` | 80.38% | 14 | 55.18 dB |
| Wan 2.2 | diffusers bf16 | `8844f3b715430e73fb25c0f89f13f4c8` | | | |
| Wan 2.2 | `sglang_default` (fp32, chunked) | `a7fe5de0f8c258d99194da52f4b86771` | 86.01% | 42 | 56.63 dB |
| Wan 2.2 | `sglang_default`, whole-clip | `a7fe5de0f8c258d99194da52f4b86771` | (identical to chunked) | | |
| Wan 2.2 | `sglang_default_bf16` | `e6a627674ebf2063bc6dfff08e4ce033` | 93.74% | 24 | 60.14 dB |
| Wan 2.2 | `sglang_best` (chunked) | `b0b0eb98e9f88ad0fe6e3dfc51071152` | 95.58% | 27 | 61.66 dB |
| Wan 2.2 | `sglang_best`, whole-clip | `b0b0eb98e9f88ad0fe6e3dfc51071152` | (identical to chunked) | | |

SGLang bf16 is not byte-identical to stock diffusers bf16, which is expected.
Under bf16 autocast, SGLang computes the RMS norm reduction and its
post-ops in fp32, while diffusers stays in bf16. Against the fp32 reference,
both land at the same PSNR (53.88 vs 53.73 dB float PSNR for Wan 2.1).
SGLang's "lossless" means bit-exact to SGLang's own aten path, not to diffusers.

## Fairness caveats

1. **Wan 2.2 default is fp32.** SGLang's registered `Wan2_2_TI2V_5B_Config`
   sets `vae_decode_precision="fp32"`, so the served default decodes in fp32.
   Its convs run in TF32 because SGLang leaves torch's
   `cudnn.allow_tf32=True` default. M*'s rows are bf16; compare M* against
   `sglang_default_bf16` / `sglang_best`, and treat the fp32 row as "what
   SGLang serves by default".
2. **SGLang's own realtime stage is fp32.** `CausalVaeDecodingStage` reads
   `vae_precision` (fp32 for all Wan configs), not `vae_decode_precision`. The
   headline rows use the offline `DecodingStage`'s precision handling (bf16
   for Wan 2.1) with the same streaming `causal_decode`. `sglang_realtime`
   reports the fp32 stage unchanged.
3. **Per-latent decoder calls** are inherent to SGLang's API (`causal_decode`
   and `decode` both loop one latent frame per decoder call). M*'s best runs
   one call per 3-latent chunk.
4. **No cuDNN benchmark, no CUDA graphs.** SGLang exposes neither for the VAE
   on CUDA: `cudnn.benchmark` is only set on ROCm, compile `reduce-overhead`
   crashes, and BCG is DiT-only. M*'s best uses `cudnn.benchmark`.
5. `quality=extra-high` is not bit-exact to lossless (53.73 vs 53.88 dB float
   PSNR for Wan 2.1). SGLang validates it against a PSNR threshold.
6. **Timing scope**: the timed window includes SGLang's latent scale/shift and
   its fp32 [0, 1] conversion (part of `DecodingStage.decode`) plus the uint8
   cast. M* times decode plus uint8 cast. Both are small elementwise passes.
7. **Harness, not a server**: the VAE is loaded through SGLang's real loader
   and configs, and the decode-dtype residency cast is reproduced with
   `vae.to(dtype)`. No HTTP server or `DiffGenerator` runs. Offline
   resolution of the registered config required a `refs/main` file in the HF
   cache pointing at the pinned commits (same revisions as M*).
8. **Different stack and VM**: SGLang pins torch 2.13.0+cu130, while M* ran
   torch 2.12.1+cu132. cuDNN is 9.20 (92000) in both, the driver is
   580.173.02 in both, and both are B200 at 1000 W, on a separate preemptible
   VM. TF32 flags are torch defaults in every row
   (`cuda.matmul.allow_tf32=False`, `cudnn.allow_tf32=True`), exactly as in
   M*'s bf16 runs. The fp32 reference was computed by M* with TF32 off.
9. Compile caches and Triton caches are fresh in every process. Compile
   warm-up (26 to 28 s first call; 57 to 65 s with max-autotune) is excluded,
   as in M*.

## Versions

- sglang 0.5.21 (PyPI, `sglang[diffusion]==0.5.21`), tag commit
  `e00930c5489053f26d86b179cee0d087f846acbb`; triton 3.7.1; diffusers 0.37.0
  (SGLang's pin; not used in SGLang's Wan VAE path); diffusers 0.39.0 (side
  install, for the stock-diffusers md5 only)
- torch 2.13.0+cu130, CUDA 13.0, cuDNN 92000 (9.20), Python 3.12.3,
  driver 580.173.02
- source md5s of `wanvae.py`, `wan_vae_cuda_opt.py`, `decoding.py`,
  `realtime/vae.py` and `wan_causal_cache_triton.py` are in every result JSON
  (`env.source_md5`); `env/sgl_freeze.txt` has the full environment
- VAE weights: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a`,
  `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff731` (safetensors sha256 identical to
  the M* VM's copies)
- Inputs: the M* B200 run's `latents.pt` and `ref_fp32_outputs.pt` (from
  wan-bench-b200 `~/wan-bench/results/b200-wan21|22/`), sha256 in
  `env/inputs_sha256.txt`, verified at source, locally and on the SGLang VM

## Files

- `runs/{wan21,wan22}/run{1,2,3}/<variant>.json` (+ `.latencies.json`): the
  3-run sets, in `bench_wan_decoder.py`'s schema, plus `sglang_settings`,
  `vae_revision` and (compiled rows) Dynamo graph-break info
- `runs/screen_{wan21,wan22}/`: one run per candidate setting, with layout check
- `runs/profile_{wan21,wan22}/<variant>.kernels.json`: `profile_kernels.py` schema
- `runs/frame_md5/`: per-stream md5 metadata and pairwise comparisons
- `results/{variance,runs,profile,frame_md5}.csv`: summaries
  (`summarize_sglang.py`)
- `harness/`: `bench_sglang_wan.py` (benchmark and profiler),
  `run_sglang.sh` (fresh-process runner: refuses a busy GPU, logs an
  nvidia-smi trace), `kernel_categories_sglang.py`, `summarize_sglang.py`,
  `md5_frames.py`
- `flops/`: M*'s FLOP counts used for TFLOP/s
- `env/`: VM setup script and log, pip freeze, nvidia-smi trace, run logs

VM `wan-bench-b200-sglang` was left running and idle. GPU time was about 50
minutes (08:48 to 09:44 UTC including idle gaps).
