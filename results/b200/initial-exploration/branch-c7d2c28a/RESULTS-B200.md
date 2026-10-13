# Wan VAE decoder on one B200: an external reference vs M* (2026-10-13)

Same GPU type as the external reference: one NVIDIA B200.
Wan 2.1 VAE, batch 1, streaming chunk `[1, 16, 3, 60, 104]` → 12 frames at 480×832, bf16. Timing
covers decode plus 8-bit conversion. Each run reports the median/p90 of 150 steady chunks
(75 × 2 cold-cache replays, 10 warmup chunks each). Latents come from encoding a real clip.
`PLAN.md` describes the method; `results-b200/` holds the raw data (batch 1:
`b200-wan21/`, `b200-wan22/`; batch 2: `b200-batch2/`, with its full tables in
`b200-batch2/summary_batch2.md` and a chart CSV in `b200-batch2/profile_roofline.csv`).

**Headline:** on the same GPU type, stock diffusers reproduces the reference's baseline (192.0 vs 199.5 ms).
The reference's steps applied to the diffusers module M* uses reach 63.2 ms. Two further exact changes
reach **55.7 ms per chunk, 7% faster than the reference's 60.2 ms**: conv-internal padding with
channels-last enforced at every conv, and `cudnn.benchmark`. Spread across 3 fresh-compile runs
is 0.4 ms. M*'s current streaming decode is 335.3 ms, so the full set of changes is **6.0× faster**
than M* today.

## Setup

| | B200 VM | local H200 box |
|---|---|---|
| GPU | NVIDIA B200 180 GB, 1000 W, Nebius `gpu-b200-sxm` / `1gpu-20vcpu-224gb`, us-central1, spot | NVIDIA H200 |
| driver | 580.173.02 (image `ubuntu24.04-cuda13.0`) | 580.173.02 |
| torch / CUDA / cuDNN / diffusers / triton | 2.12.1+cu132 / 13.2 / 9.20.0 / 0.39.0 / 3.7.1 | same |
| M* model code | upstream main `99f5108b` (decoder files identical to the branch base) | same files |
| Python env | exact `pip freeze` of the local `.venv` minus `flash-attn` and the local `mstar-rust` wheel; M* has no lockfile (`uv.lock` is gitignored) | — |

## 1. Run-to-run variance (Wan 2.1)

There were 3 independent runs, each variant a fresh process with fresh Inductor/Triton caches
(fresh compile), in the orders ABC / CBA / BCA. The table gives the median ms per chunk per run.

| variant | run 1 | run 2 | run 3 | mean of medians | spread |
|---|---:|---:|---:|---:|---:|
| stock diffusers eager bf16 | 191.96 | 192.03 | 191.96 | **191.98** | 0.07 |
| + torch.compile (warm cache) | 95.85 | 95.93 | 95.92 | **95.90** | 0.08 |
| M\* default streaming (ctx 8, compiled) | 335.02 | 335.47 | 335.27 | **335.25** | 0.45 |
| M\* default streaming, eager | 656.84 | 656.85 | 656.85 | **656.85** | 0.01 |
| + channels-last (compiled, per frame) | 68.30 | 68.39 | 68.36 | **68.35** | 0.09 |
| + one call per chunk (`rx_cl_chunk_compile`) | 63.29 | 63.17 | 63.33 | **63.26** | 0.16 |
| same + bf16 upsample (`rx_full`) | 63.19 | 63.17 | 63.21 | **63.19** | 0.04 |
| `rx_full` in fp16 | 64.91 | 64.89 | 64.93 | **64.91** | 0.04 |
| best: + native pad + cudnn.benchmark (`px_nopad_bench`) | 55.83 | 55.91 | 55.49 | **55.74** | 0.42 |

Every spread is ≤ 0.8% of the mean, and the p90 is within 1–2 ms of the median. The batch-1 single
runs agree with these means to within 1 ms (≤ 0.5%).

## 2. TF32 and PSNR conventions

The fp32 reference runs with TF32 **off**. The recorded flags are `torch.backends.cuda.matmul.allow_tf32=False`
and `torch.backends.cudnn.allow_tf32=False` (new API: `cuda.matmul.fp32_precision='ieee'`;
`cudnn.conv.fp32_precision='none'`, which defers to the legacy flag, False; with TF32 allowed it
reads `'tf32'`). A re-run reproduces the saved reference
bit-exactly (PSNR ∞). As a control, the same decode with TF32 allowed runs 10× faster
(234 vs 2,407 ms per chunk) and differs from it at 86.0 dB. So the flag does change the convs, and the
reference used for all PSNRs is the exact one.

PSNR vs that reference, on [−1, 1], first 10 steady chunks. "Peak 2" uses the [−1, 1] range;
"peak 1" is 6.02 dB lower and appears to be the reference's convention.

| decoder | ours, peak 2 | ours, peak 1 | ext. ref (Table 1) |
|---|---:|---:|---:|
| stock diffusers bf16 | 59.75 | **53.73** | 52.3 |
| best bf16 prototype (`px_nopad_bench`) | 59.93 | **53.91** | 52.3 (theirs) |
| fp16 prototype | 80.05 | **74.03** | 71.3 (theirs) |
| fp32 with TF32 allowed | 86.02 | **80.00** | 74.3 (SGLang fp32, TF32 convs) |

The ordering and gaps match the reference's; the ~1.5–2.7 dB absolute offset comes from different content
(LingBot-World latents vs our encoded clip). All bf16 variants, including every beat-60 prototype,
land at the same PSNR (59.93 dB, peak 2), i.e. bf16 rounding level.

## 3. Where the time goes (the reference Figures 1 and 11)

The profile uses torch.profiler over 6 steady chunks after warmup (`profile_kernels.py`). Categories are
assigned from kernel names (`kernel_categories.py`). In eager mode, PyTorch's pad/cat/clone and the
norm's fp32 casts show up as copy kernels. One chunk is **40.65 TFLOP** (conv 40.47; the reference:
"about 40 TFLOP"). The minimum conv traffic is 26.3 GB, giving an intensity of ~1,550 FLOP/B, far
above the B200 ridge (2,250 TFLOP/s ÷ 8 TB/s = 281 FLOP/B). The decode is compute-bound and its
roof is 2,250 TFLOP/s.

| ms per chunk | stock diffusers eager | M\* default (ctx 8, compiled) | best prototype | ext. ref as ported | ext. ref final |
|---|---:|---:|---:|---:|---:|
| **kernels per chunk** | **2,046** | **3,432** | **139** | **1,024** | **185** |
| convolution | 39.6 | 136.7 | 38.0 | 38.3 | 38.3 |
| layout conversion | 24.1 | 77.3 | 0 | 25.3 | 0 |
| norm / SiLU (+ fused pad/cat) | 8.4 | 29.9 | 9.2 | 17.2 (incl. concat, pad) | 11 (incl. concat, pad) |
| concat / pad | 4.0 | 40.1 | 1.1 | (above) | (above) |
| upsample | 3.9 | 28.9 | 3.6 | fp32, separate bar | — |
| attention | 1.9 | 7.1 | 1.3 | small | small |
| copies (eager casts, pad, clone) | 62.5 | 3.8 | 0.7 | small | small |
| other elementwise | 48.3 | 7.1 | 1.2 | 38.6 | small |
| GPU time / wall time | 192.8 / 192.0 | 331.1 / 335.3 | 55.0 / 55.7 | 133.8 | 60.2 |
| achieved TFLOP/s (useful 40.65 TFLOP / wall) | 212 (9% of roof) | 121 (5%); 417 counting the 139.9 TFLOP it executes | **729 (32%)** | 304 | 675 |

The M\* streaming decode executes 139.9 TFLOP per chunk, because it re-decodes 8 context latents
with an empty cache. That is 3.4× the useful work, plus 3,432 kernels and 77 ms of layout
conversions. In the best prototype, convolution is 69% of the GPU time. The conv kernels alone reach
40.47 TFLOP / 38.0 ms ≈ 1,065 TFLOP/s, 47% of the dense-bf16 peak. With `cudnn.benchmark` the large
3×3×3 convs switch to CUTLASS `conv3d_fprop_weight_stationary` kernels.
Top-15 kernels for every profiled variant: `summary_batch2.md`. Chart data: `profile_roofline.csv`.

## 4. Beating 60.2 ms (prototypes in the bench script only; no M* changes)

Every row is compiled, channels-last, one call per chunk, bf16 upsample (= `rx_full`) plus the listed change.

| change on top of `rx_full` | median ms | p90 | kernels / chunk | peak GiB | PSNR dB (peak 2 / 1) |
|---|---:|---:|---:|---:|---:|
| none (`px_base`) | 63.44 | 64.71 | 170 | 20.0 | 59.93 / 53.91 |
| (a) manual CUDA graph per chunk (`px_cg`) | 63.38 | 64.20 | 203* | 12.2 | 59.93 / 53.91 |
| (a) `torch.compile(mode="reduce-overhead")` (`px_ro`) | 65.17 | 66.17 | 223 | 3.2 | 59.93 / 53.91 |
| (b) native pad + channels-last at every conv (`px_nopad`) | 59.11 | 60.48 | **139** | 19.0 | 59.93 / 53.91 |
| (c) `cudnn.benchmark` (`px_bench`) | 62.43 | 63.41 | 170 | 20.0 | 59.93 / 53.91 |
| (c) `mode="max-autotune-no-cudagraphs"` (`px_maxat`) | 62.49 | 63.52 | 164 | 20.0 | 59.99 / 53.97 |
| **(b) + (c) benchmark** (`px_nopad_bench`) | **55.75** (3-run mean 55.74) | 56.77 | **139** | 19.0 | 59.93 / 53.91 |
| (a) + (b) | 59.75 | 60.68 | 172* | 12.2 | 59.93 / 53.91 |
| (a) + (b) + (c) | 55.83 | 57.02 | 172* | 12.2 | 59.93 / 53.91 |
| (a) + (b) + (c), fp16 | 58.21 | 59.48 | 172* | 12.2 | 80.05 / 74.03 |

\* includes 33 small in-graph copies that write the cache back into its static buffers.

- **(a) CUDA graphs: no gain.** GPU time already matches wall time within ~1 ms at ~140–170
  kernels, so launch overhead is not the bottleneck. `reduce-overhead` needs the cache cloned
  out of the graph pool every chunk; otherwise its graph trees overwrite the cache the next chunk reads.
  Graphs do lower peak memory (private pool, 12 vs 19–20 GiB).
- **(b) The real win.** Each causal conv concatenates only the temporal context (cache or zero
  frames). H/W zero padding moves into the conv (`padding=(0, 1, 1)`), and every causal-conv input
  and every post-upsample 2D-conv input is made channels-last. That removes the padded copy per conv:
  −31 kernels, concat/pad down from 5.1 to 1.1 ms, and −4.3 ms overall. The layout check counts 0 of
  39 conv calls with a channels-first input (it was 3 of 39). The fp32 math is identical: max |diff|
  2.5e-6 on CPU.
- **(c) `cudnn.benchmark` matters once inputs are clean:** −1.0 ms on its own, −3.4 ms on top of (b).
  Inductor max-autotune benchmarks Triton conv templates but cuDNN wins for nearly every conv; it
  gives −1 ms for a ~65 s compile.
- **Not tried:** a custom Triton cat+pad kernel or ring buffer. After (b), concat/pad is 1.1 ms and
  the cache concat is fused into the norm/SiLU producer by Inductor, so the headroom left there is small.

## 5. Wan2.2 VAE (what M* serves; 48 ch, 16×; latent `[1, 48, 3, 30, 52]`, 88.2 TFLOP per chunk)

| decoder | median ms | p90 | kernels / chunk | PSNR dB (peak 2) |
|---|---:|---:|---:|---:|
| stock diffusers eager bf16 (batch 1) | 184.6 | 184.8 | 2,173 | 62.19 |
| M\* default streaming, ctx 8 compiled (batch 1) | 370.1 | 372.2 | 3,536 | — (u8 51.1) |
| `px_base` (= reference-style) | 75.89 | 77.29 | 186 | 62.58 |
| **`px_nopad_bench`** | **72.99** | 75.31 | 154 | 62.58 |

On Wan2.2 the best configuration is 5.1× faster than M*'s default and reaches 1,208 TFLOP/s
(54% of roof): fewer, larger channels give the convs higher utilization.

## Proposed M* changes (expected gain, Wan 2.1 on B200, measured above)

| # | change | where | gain | exact? |
|---|---|---|---|---|
| 1 | keep the decoder's causal `feat_cache` per stream/session instead of re-decoding 8 context latents | `Cosmos3VAEDecoderARSubmodule` (and any streaming Wan decode) | 335.3 → 95.9 ms (3.5×); 139.9 → 40.7 TFLOP per chunk | yes (tighter than ctx 8, which is ±1 u8 level) |
| 2 | channels-last 3D weights + latents; warm compile with channels-last dummies | shared Wan VAE helper | 95.9 → 68.4 ms | yes |
| 3 | decode a chunk's latent frames in one decoder call after the first frame | same | 68.4 → 63.2 ms (+11 GiB peak) | yes |
| 4 | causal conv with conv-internal H/W padding, channels-last enforced at conv and upsample inputs | patch or subclass `WanCausalConv3d` / `WanUpsample` | 63.4 → 59.1 ms, −31 kernels | yes |
| 5 | `cudnn.benchmark` around the decoder (fixed shapes) | decoder node | 59.1 → 55.7 ms | yes (algorithm choice only) |
| 6 | compile the decoder step in the Wan22 node (it currently disables compile) | `Wan22VaeDecoderSubmodule` | ~2× on its whole-clip decode (stock streaming 192 → 96 ms) | yes |
| 7 | optional fp16 decode | both decoder nodes | +1.5–2.5 ms, +20 dB vs fp32 | n/a |

Not worth it on this workload: CUDA graphs, `reduce-overhead`, max-autotune, and the bf16
upsample patch (Inductor already fuses it). Changes 1–5 together take M*'s streaming decode from
335 to 56 ms per chunk (6.0×, decoder-only 36 → 215 fps); on the Wan2.2 VAE, 370 → 73 ms (5.1×).

## Batch 1 (first pass) results, Wan 2.1, for reference

| decoder | median | p90 | peak GiB | PSNR float dB (peak 2) | PSNR u8 dB |
|---|---:|---:|---:|---:|---:|
| M\* tuned ctx 4 / 2 / 1 (config only) | 206.6 / 142.1 / 110.1 | | 6.3 | | 49.0 / 38.5 / 28.4 |
| M\* Wan22 node, whole 81-frame clip (per 12 frames) | 192.2 | 192.3 | 4.6 | | |
| M\* Wan22 node, 241-frame clip, forced tiled | 429.5 | 430.0 | 3.0 | | |
| eager channels-last / + one call per chunk | 162.1 / 149.2 | | 4.0 / 8.0 | 59.75 | 54.9 |
| one call per chunk + compile, channels-first | 93.2 | 94.7 | 11.8 | 59.93 | 55.0 |
| fp32 reference (TF32 off) | 2,407 | | 7.7 | — | — |

The reference's B200 numbers (Figure 10 / text, ms per chunk): stock diffusers 199.5; as ported 133.8;
+channels-last 85.9; +one call per chunk 78.8; SGLang bf16 channels-last compiled 101.9; vLLM-Omni
Wan fast path channels_last 66.2; final 60.2 (fp16 ≈ 61.5).

## Environment notes

- First pass: the VM's system Python lacked `python3.12-dev`, and Triton needs `Python.h`, so
  compiled variants failed until it was installed; they were rerun with identical arguments.
- Batch 2: `px_cg` / `px_ro` first failed on two harness issues. Permanent `None` cache slots
  (1×1 shortcut convs) broke the steady-state check, and graph trees overwrote the cache. Both were
  fixed in `common.py` and rerun (`batch2.sh pxfix`).

## VM and cost

`wan-bench-b200` (`<instance-id>`, boot disk `computedisk-u00kk11z31ptaywrs5`,
200 GiB) in us-central1, spot, public IP <vm-ip>, created 04:45 UTC. GPU work: batch 1 ran
04:53–05:43 and batch 2 ran 05:52–07:01 UTC (~2 h of the VM's ~2.5 h up). It is left **running**. The
Nebius calculator lists $8.50/h (+$0.02/h disk) for this preset and does not model spot pricing,
so ≤ ~$21 through 07:13 UTC.

    nebius compute instance stop   --id <instance-id>
    nebius compute instance delete --id <instance-id>
    nebius compute disk delete     --id computedisk-u00kk11z31ptaywrs5   # if it outlives the instance
