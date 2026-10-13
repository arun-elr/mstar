# Wan VAE decoder benchmark: an external reference vs M* — plan (not yet run on GPU)

Source: an external public write-up on Wan VAE decoder optimization (not archived here).
Figure numbers below were read off its figures.

## 1. The reference's setup and numbers

- **Hardware:** one NVIDIA B200 ("WAN 2.1 VAE · ONE B200" hero; roofline "on one B200, using
  NVIDIA's published dense-bf16 peak (2.25 PFLOP/s) and HBM bandwidth (8 TB/s)"). They first moved
  the decoder off context parallelism onto a single GPU ("shaved off 16 ms, leaving us at 133.8 ms").
- **Model / component:** the Wan 2.1 VAE decoder inside LingBot-World. Per chunk:
  "a block of latents z ∈ ℝ16×3×60×104: 3 latent frames of 16 channels at 60×104. The VAE decoder
  then takes this and expands it to 12 RGB frames at 480×832 pixels." Batch 1, streaming with a
  causal conv cache ("each convolution keeps the last 2 frames of input from the previous call").
  About 40 TFLOP per 12-frame chunk.
- **dtype:** bf16 (main), plus an fp16 variant. PSNR is measured against an fp32 decode with TF32 off.
- **Software:** PyTorch + torch.compile (Inductor) + cuDNN. The post gives no versions.
- **Measurement:** "a small bench that replays the decoder's real inputs: latents recorded from a
  26-chunk run … times the decode plus the conversion to 8-bit frames over 75 steady chunks";
  Figure 10 reports the "median of 75 steady chunks with p90 whiskers". They report ms per 12-frame
  chunk. Whole-model fps: "takes a single B200 from about 36 to about 46 frames per second"
  (35.9 → 42.1 fps after step 1). The decoder was 43% of chunk GPU time at the start (Figure 1).
- **Optimization ladder (ms per chunk, B200):** as ported 133.8 → channels-last weights + activations
  85.9 → one decoder call per chunk instead of per latent frame 78.8 (+~8 GB memory) →
  60.2 after removing the fp32 nearest upsample (9.3 ms), padding/concat traffic, and
  conv inefficiency. Final: 185 kernels, versus 1,024 as ported. Convolution time stays at 38.3 ms;
  layout conversion drops from 25.3 ms to 0, and norm/SiLU/concat/pad from 17.2 ms to 11 ms
  (Figure 11). A rejected "2D-tap" conv variant is mentioned. They did not use CUDA graphs.
- **Other engines on the same B200 (Figure 10, ms per chunk):** stock diffusers (eager) 199.5;
  vLLM-Omni LingBot exact 196.8; SGLang fp32 192.6; SGLang bf16 168.2; SGLang fp32 compiled 163.2;
  SGLang bf16 channels-last 127.7; vLLM-Omni LingBot fused 113.7; vLLM-Omni Wan fast path lossless
  110.0; SGLang bf16 channels-last compiled 101.9; vLLM-Omni Wan fast path channels_last 66.2;
  ours 60.2 (fp16 ≈ 61–62 from the Pareto chart).
- **Quality (Table 1, PSNR vs fp32, dB):** stock diffusers bf16 52.3; ours bf16 52.3; ours fp16 71.3;
  SGLang fp32 74.3.

## 2. M*'s Wan decoder paths

M* has **no Wan 2.1 VAE model**. Every M* model that uses a Wan VAE uses the Wan2.2 VAE (48 latent
channels, 16× spatial):

| M* config | node | how it decodes | compile / graph | dtype |
|---|---|---|---|---|
| `configs/wan22.yaml` (Wan2.2-TI2V-5B) | `Wan22VaeDecoderSubmodule` (`mstar/model/wan22/submodules.py`) | diffusers `AutoencoderKLWan.decode` on the whole clip once per request (one latent frame per decoder call internally); VRAM-gated tiled/untiled (`vae_decode_tiling`, `WAN22_VAE_DECODE_TILING`); uint8 at the node boundary | `disable_torch_compile = True`, no CUDA graph | bf16 when cuDNN ≥ 9.16, else fp32 (`WAN22_VAE_DECODE_FP32`) |
| `cosmos3_*` (Wan2.2 VAE, `z_dim 48`) | `Cosmos3VAEDecoderSubmodule` | `vae.decode` whole clip, `torch.compile(vae.decode, dynamic=False)` on by default (`COSMOS3_COMPILE_VAE`), discards the first compiled bf16 call per shape | Inductor; no CUDA graph | same cuDNN gate (`COSMOS3_VAE_DECODE_FP32`) |
| `cosmos3_*` windowed AR | `Cosmos3VAEDecoderARSubmodule` | **streaming**: each window re-decodes `windowed_decode_context_latents` (default **8**) old latents plus the new ones from a **cleared cache**, then trims; no persistent conv cache across chunks | as above | as above |
| `waypoint*` | TAEHV (`mstar/model/waypoint/components/taehv.py`) | tiny distilled decoder, not the Wan VAE | — | — |

None of M*'s paths applies channels-last, one call per chunk, or a bf16 upsample. All use stock
diffusers 0.39 `AutoencoderKLWan`, so that class *is* M*'s Wan decoder implementation.

**Is an apples-to-apples run possible?** Yes, for the decoder component. Diffusers'
`AutoencoderKLWan` is the same class for Wan 2.1 and Wan 2.2. M*'s two decoder nodes accept a Wan 2.1
VAE unchanged; `bench_wan_decoder.py` instantiates the real M* classes with the Wan 2.1 VAE and
The reference's latent shape. No M* source changes are needed. An end-to-end comparison is not possible:
M* has no LingBot-World or other Wan 2.1 world model. M*'s Wan2.2-TI2V-5B e2e numbers
(`benchmark/wan22/`) are a different model and VAE, so they are context at most.

Correctness of the harness was checked on CPU only (`check_equivalence.py`, real Wan 2.1 weights, fp32):
streaming with a warm cache reproduces `vae.decode` exactly (max |diff| 0); one call per chunk vs
per-frame differs by 1.6e-6; channels-last by 3.0e-6; bf16 nearest upsample is exact; the Wan2.2 VAE is
also exact per chunk (2e-6). M*'s AR decode with ctx=8 matches exact streaming to 1 uint8 level, but
**ctx=1 is wrong by up to 86 levels**, so M*'s only speed knob for streaming trades quality away.

## 3. The queued GPU job

Everything lives in `~/oss/experiment/scratch/wan-bench/`:

- `common.py`: VAE loading from the HF cache (offline), the warm-cache `StreamingDecoder`
  (per-frame or per-chunk, channels-last, compile), and the bf16-upsample patch.
- `prepare_latents.py`: encodes `assets/draw.mp4` (natural 4K hand-drawing clip, cropped to 480×832,
  ping-ponged to 1,021 frames) into 256 Wan 2.1 latent frames, i.e. 1 + 85 chunks; `--random` is a fallback.
- `bench_wan_decoder.py`: one variant per process; CUDA events per chunk with a sync; 10 warmup
  chunks plus 75 timed steady chunks, ×2 repeats (150 samples); median/p10/p90/mean, decoder fps =
  12 / median; peak allocated/reserved memory; PSNR (float on [−1, 1] with peak-to-peak 2, and uint8)
  over the first 10 steady chunks versus the fp32/TF32-off reference; optional profiler table and
  channels-last layout check.
- `run_all.sh`: the whole job. Refuses to start if the GPU has other compute processes (never
  kills; `ALLOW_SHARED=1` overrides). Logs an nvidia-smi trace and re-runs the baseline at the
  end as a drift check.
- `summarize.py`: renders a markdown table next to the reference's B200 numbers.
- `check_equivalence.py`: CPU-only harness checks (already passed).

Command (one job, when the queue hands over the GPU):

```bash
cd ~/oss/experiment/scratch/wan-bench
GPU=0 bash run_all.sh              # Wan 2.1 VAE suite → results/<timestamp>/wan21/summary.md
GPU=0 WAN22=1 bash run_all.sh      # optional: also the Wan2.2 VAE M* actually serves
```

Variants and what they stand for:

| variant | meaning |
|---|---|
| `ref_fp32` | diffusers eager fp32, TF32 off: the exact reference (and an "fp32 as shipped"-style time) |
| `diffusers_bf16` | stock diffusers eager bf16, warm cache, per-frame: same-H200 version of the reference's 199.5 ms bar |
| `diffusers_bf16_compile` | the above + torch.compile: M*'s compile knob applied to true streaming |
| `mstar_ar_ctx8_eager` / `mstar_ar_ctx8` | **M* default** streaming decode (Cosmos3 AR node logic, real `_decode_pixels`), eager / compiled (default on) |
| `mstar_ar_ctx4/2/1` | **M* tuned, config-only** (`windowed_decode_context_latents`); quality drops |
| `mstar_wan22_oneshot_{81,241}f[_tiled]` | M*'s Wan22 node, whole-clip decode; reported as ms per 12 output frames |
| `rx_cl`, `rx_cl_chunk`, `rx_cl_chunk_compile`, `rx_full`, `rx_full_fp16` | the reference's steps applied to the same diffusers module (prototype of an M* change, not current M*) |

**Weights** (all local, `HF_HUB_OFFLINE=1`): Wan 2.1 VAE `Wan-AI/Wan2.1-T2V-1.3B-Diffusers/vae`,
485 MB, downloaded into the HF cache on 2026-10-12 before the GPU freeze. Wan2.2 VAE: already cached
as part of `Wan-AI/Wan2.2-TI2V-5B-Diffusers`. LingBot-World weights are not needed: the decoder is
the Wan 2.1 VAE architecture, and timing does not depend on weight values.

**Expected cost (estimates, H200):**
- Wall time: Wan 2.1 suite ~60–100 min. Most of that is cold Inductor compiles; the M* compiled AR
  decode compiles 3–4 unrolled shapes of up to 11 decoder calls. `WAN22=1` roughly doubles it.
- Peak GPU memory: expected < 25 GiB per variant (fp32 ref and one-call-per-chunk variants are the
  largest); budget 40 GiB. The job needs exclusive use of the GPU for clean timing, not for memory.
- Disk: ~1.5 GB per suite (fp32 reference outputs 575 MB, latents ~100 MB, inductor cache).

**Rough expectation, to sanity-check results:** the H200 has ~0.44× the B200's dense-bf16 compute
(989 vs 2,250 TFLOP/s) and ~0.6× its HBM bandwidth (4.8 vs 8 TB/s). Identical software should
therefore run ~1.7–2.3× slower per chunk, putting stock diffusers bf16 around 330–450 ms and a
Reference-equivalent decode around 100–140 ms on the H200. Compare *ratios* (speedup over the
same-GPU stock-diffusers baseline) across GPUs, not absolute ms.

## 4. True same-GPU comparison

This requires one NVIDIA B200 (the reference cites the 180 GB / 1000 W SXM part; e.g. an HGX/DGX B200
node, using one GPU of it) with a Blackwell-capable stack (CUDA ≥ 12.8, cuDNN 9.x with sm_100
conv3d kernels; our torch 2.12.1+cu132 / cuDNN 9.20 qualifies). Nothing is provisioned here.
The same `run_all.sh` runs unchanged there.

## 5. Candidate M* improvements (upstream ideas)

1. **Persistent conv cache for streaming decode** (`Cosmos3VAEDecoderARSubmodule`): keep the
   decoder's `feat_cache` per request/session instead of re-decoding 8 context latents from a cold
   cache. That is ~3.7× less decode work per 3-latent chunk, exact rather than approximate. The
   biggest win, and it doesn't depend on the reference's tricks.
2. **One decoder call per chunk** after the first latent frame (exact; shown above on CPU).
3. **Channels-last 3D** weights and activations for the Wan VAE; warm torch.compile with
   channels-last dummies (the reference's recompile trap).
4. **bf16/fp16 nearest upsample**: drop `WanUpsample`'s fp32 round trip (exact).
5. Compile the decoder step (not the whole `decode` loop, which unrolls per latent frame) for the
   wan22 node too, which currently disables compile entirely. Consider an fp16 decode option
   (external reference: same speed, 71 dB vs 52 dB PSNR).
6. Longer term: fused pad + cache-concat (as SGLang's Triton kernel does), and custom conv tiling.
