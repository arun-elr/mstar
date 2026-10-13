# Wan VAE streaming decode on one NVIDIA H200

The same benchmark, inputs and weights as the B200 results, re-run on one local
NVIDIA H200 (141 GB HBM3e, 700 W, driver 580.173.02). Method: the PR's
`benchmark/wan_vae_decoder` at `c56b572b`, one variant per fresh process with
fresh Inductor/Triton caches. Each process decodes an untimed first latent and
10 untimed warm-up chunks, then 75 timed chunks, and the stream is replayed twice (150 samples).
3-run sessions use the orders ABC / CBA / BCA, as in `run_ablation.sh intree`.

**Headline.** The in-tree node (`mstar_warm`) is **5.6×** faster than M* before
on Wan 2.1 (577.81 → 103.27 ms per chunk) and **4.4×** faster on Wan 2.2
(619.09 → 139.97 ms). On the B200 it was 6.0× and 5.0×. The optimized decoders
take 1.83–1.93× longer per chunk on H200 than on B200. Output lands in the same
exactness tiers as on the B200.

| | |
|---|---|
| GPU | NVIDIA H200 (sm_90, 132 SMs), 700 W, driver 580.173.02, persistence on, ECC on; the only GPU on this host, idle before every process |
| Software | torch 2.12.1+cu132, CUDA 13.2, cuDNN 9.20.0, Triton 3.7.1, diffusers 0.39.0. Identical to the B200 env except Python 3.12.15 vs 3.12.3 |
| Code | this branch at `c56b572b`; M* model code imported from it for every variant, including M* before |
| Inputs | the B200 runs' latents and fp32 references; sha256 matches the B200 inputs for all 4 files |
| Weights | `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff731`; same revisions and VAE sha256 as the B200 VM |
| Peak used | **989 TFLOP/s dense bf16**, 4.8 TB/s HBM3e (ridge ≈ 206 FLOP/B; one chunk is 1,548 FLOP/B on Wan 2.1 and 3,774 on Wan 2.2, so both are compute-bound and the roof is 989) |

Files: `runs.csv` (every single run), `variance.csv` (3-run sessions),
`profile.csv` (kernel breakdown, achieved TFLOP/s), `comparison_b200.csv`
(the table below), `quality_check.csv` (whole-stream md5 / PSNR),
`coverage.csv` (M*'s node over irregular windows). The runs are `run_ablation.sh intree` /
`quality` with `GPU=0`; nothing H200-specific was needed.

## 1. H200 vs B200, ms per 3-latent chunk (12 frames at 480×832)

× = speedup vs M* before on the same GPU. H200: 3-run mean, or single run (\*).
B200: the PR's `intree.csv` session (M* before 335.83 / 368.85 ms; the
`mstar_warm_nobench` row is the 3-run control session), otherwise `variance.csv`
3-run means; \* is a single run. RESULTS.md's ladder uses the B200 variance
session's M* before (335.25 ms), so its × differs by ≤ 0.2%.

| VAE | variant | change | H200 ms | H200 × | B200 ms | B200 × | H200 / B200 |
|---|---|---|---:|---:|---:|---:|---:|
| Wan 2.1 | `mstar_ar_ctx8` | M* before (re-decode 8 context latents, compiled) | 577.81 | 1.0× | 335.83 | 1.0× | 1.72 |
| Wan 2.1 | `diffusers_bf16` | stock diffusers, eager | 270.41\* | 2.1× | 191.98 | 1.7× | 1.41 |
| Wan 2.1 | `diffusers_bf16_compile` | 1. keep conv cache (+ compile) | 167.43\* | 3.5× | 95.90 | 3.5× | 1.75 |
| Wan 2.1 | `channels_last_compile` | 2. + channels-last | 135.80\* | 4.3× | 68.35 | 4.9× | 1.99 |
| Wan 2.1 | `channels_last_one_call_compile` | 3. + one decoder call per chunk | 131.36\* | 4.4× | 63.26 | 5.3× | 2.08 |
| Wan 2.1 | `px_nopad` | 4. + H/W padding inside the conv | 125.40\* | 4.6× | 59.11\* | 5.7× | 2.12 |
| Wan 2.1 | `px_nopad_bench` | 5. + `cudnn.benchmark` (prototype best) | 102.58 | 5.6× | 55.85 | 6.0× | 1.84 |
| Wan 2.1 | `mstar_warm` | in-tree node, this change's defaults | **103.27** | **5.6×** | 56.30 | 6.0× | 1.83 |
| Wan 2.1 | `mstar_warm_eager` | in-tree node, eager | 191.07 | 3.0× | 129.03 | 2.6× | 1.48 |
| Wan 2.1 | `mstar_warm_nobench` | in-tree node, `cudnn.benchmark` off | 125.83\* | 4.6× | 58.80 | 5.7× | 2.14 |
| Wan 2.2 | `mstar_ar_ctx8` | M* before (re-decode 8 context latents, compiled) | 619.09 | 1.0× | 368.85 | 1.0× | 1.68 |
| Wan 2.2 | `px_nopad_bench` | prototype best | 139.68 | 4.4× | 72.29 | 5.1× | 1.93 |
| Wan 2.2 | `mstar_warm` | in-tree node, this change's defaults | **139.97** | **4.4×** | 73.76 | 5.0× | 1.90 |
| Wan 2.2 | `mstar_warm_eager` | in-tree node, eager | 222.46 | 2.8× | 142.72 | 2.6× | 1.56 |
| Wan 2.2 | `mstar_warm_nobench` | in-tree node, `cudnn.benchmark` off | 140.17\* | 4.4× | — | — | — |

Decoder fps on H200 (12 / median): in-tree 116 (Wan 2.1) and 86 (Wan 2.2);
M* before 21 and 19. Peak allocated memory is the same as on B200 to 0.02 GiB
(19.03 / 15.93 GiB in-tree, 6.33 / 6.44 GiB M* before).

**Run-to-run variance** (H200, `variance.csv`), median ms per chunk:

| VAE | variant | run 1 | run 2 | run 3 | mean | spread | p90 mean |
|---|---|---:|---:|---:|---:|---:|---:|
| Wan 2.1 | `mstar_ar_ctx8` | 577.46 | 577.75 | 578.22 | 577.81 | 0.76 (0.13%) | 580.39 |
| Wan 2.1 | `mstar_warm` | 103.38 | 103.45 | 102.99 | 103.27 | 0.46 (0.45%) | 105.49 |
| Wan 2.1 | `mstar_warm_eager` | 191.18 | 191.10 | 190.93 | 191.07 | 0.25 (0.13%) | 192.54 |
| Wan 2.1 | `px_nopad_bench` | 102.41 | 102.54 | 102.80 | 102.58 | 0.39 (0.38%) | 106.32 |
| Wan 2.2 | `mstar_ar_ctx8` | 618.81 | 619.08 | 619.37 | 619.09 | 0.56 (0.09%) | 621.53 |
| Wan 2.2 | `mstar_warm` | 140.91 | 139.64 | 139.36 | 139.97 | 1.55 (1.11%) | 141.86 |
| Wan 2.2 | `mstar_warm_eager` | 221.67 | 224.29 | 221.41 | 222.46 | 2.88 (1.29%) | 225.04 |
| Wan 2.2 | `px_nopad_bench` | 139.81 | 139.49 | 139.73 | 139.68 | 0.32 (0.23%) | 142.22 |

Drift check: stock `diffusers_bf16` was re-run at the end of the session and
measured 269.81 ms, against 270.41 ms at the start (−0.2%).

## 2. Per-change ladder, Wan 2.1, H200

Each row adds one change to the row above. Δ is against the previous row.
B200 columns are the RESULTS.md ladder.

| # | change | variant | H200 ms | Δ ms (Δ %) | H200 × | B200 ms | B200 Δ ms (Δ %) | B200 × |
|---|---|---|---:|---:|---:|---:|---:|---:|
| | M* before | `mstar_ar_ctx8` | 577.81 | | 1.0× | 335.25 | | 1.0× |
| 1 | keep the conv cache across chunks (+ compile) | `diffusers_bf16_compile` | 167.43\* | −410.38 (−71.0%) | 3.5× | 95.90 | −239.35 (−71.4%) | 3.5× |
| 2 | channels-last weights and activations | `channels_last_compile` | 135.80\* | −31.63 (−18.9%) | 4.3× | 68.35 | −27.55 (−28.7%) | 4.9× |
| 3 | one decoder call per chunk | `channels_last_one_call_compile` | 131.36\* | −4.44 (−3.3%) | 4.4× | 63.26 | −5.09 (−7.4%) | 5.3× |
| 4 | H/W padding inside the conv, channels-last at every conv | `px_nopad` | 125.40\* | −5.96 (−4.5%) | 4.6× | 59.11\* | −4.15 (−6.6%) | 5.7× |
| 5 | `cudnn.benchmark` | `px_nopad_bench` | **102.58** | **−22.82 (−18.2%)** | **5.6×** | 55.74 | −3.37 (−5.7%) | 6.0× |
| | all five, in-tree node | `mstar_warm` | **103.27** | +0.69 vs step 5 | **5.6×** | 56.30 | +0.45 vs step 5 | 6.0× |
| | in-tree node, `cudnn.benchmark` off | `mstar_warm_nobench` | 125.83\* | +22.56 vs `mstar_warm` | 4.6× | 58.80 | +2.23 | 5.7× |

`px_nopad`'s layout check: 0 of 39 conv inputs are not channels-last per steady
chunk, the same as on B200.

## 3. Where the time goes (kernel profiles, 6 steady chunks)

`profile_kernels.py`, one fresh process per row. Achieved TFLOP/s is useful
work (40.65 TFLOP per chunk on Wan 2.1, 88.18 on Wan 2.2) divided by the
3-run-mean wall time. The FLOP counts come from the B200 run's
`FlopCounterMode` count, which doesn't depend on the GPU.

| VAE | variant | GPU | kernels / chunk | conv ms | layout ms | norm/SiLU ms | concat/pad ms | GPU / wall ms | achieved TFLOP/s (% of peak) | conv kernels TFLOP/s |
|---|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Wan 2.1 | M* before | H200 | 3,520 | 334.7 | 91.2 | 39.7 | 51.1 | 574.8 / 577.8 | 70 (7.1%) | 416 |
| | | B200 | 3,432 | 136.7 | 77.3 | 29.9 | 40.1 | 331.1 / 335.3 | 121 (5.4%) | 1,019 |
| Wan 2.1 | `mstar_warm` | H200 | 175 | 77.7 | 0 | 10.8 | 1.5 | 98.1 / 103.3 | **394 (39.8%)** | 521 |
| | | B200 | 149 | 37.9 | 0 | 9.3 | 1.1 | 54.9 / 56.3 | 722 (32.1%) | 1,067 |
| Wan 2.1 | `mstar_warm_nobench` | H200 | 178 | 100.5 | 0 | 10.8 | 1.5 | 120.9 / 125.8\* | 323 (32.7%) | 403 |
| Wan 2.1 | `px_nopad_bench` | H200 | 165 | 80.2 | 0 | 11.2 | 1.5 | 101.9 / 102.6 | 396 (40.1%) | 505 |
| Wan 2.2 | M* before | H200 | 3,621 | 403.4 | 81.8 | 32.6 | 40.3 | 611.1 / 619.1 | 142 (14.4%) | 751 |
| | | B200 | 3,536 | 196.3 | 66.7 | 26.7 | 33.0 | 362.2 / 370.1 | 238 (10.6%) | 1,543 |
| Wan 2.2 | `mstar_warm` | H200 | 189 | 122.2 | 0 | 6.4 | 1.1 | 136.8 / 140.0 | **630 (63.7%)** | 721 |
| | | B200 | 164 | 58.5 | 0 | 5.2 | 0.9 | 70.3 / 73.8 | 1,195 (53.1%) | 1,508 |
| Wan 2.2 | `mstar_warm_nobench` | H200 | 188 | 122.8 | 0 | 6.3 | 1.1 | 137.9 / 140.2\* | 629 (63.6%) | 718 |
| Wan 2.2 | `px_nopad_bench` | H200 | 181 | 121.1 | 0 | 6.3 | 1.1 | 136.8 / 139.7 | 631 (63.8%) | 728 |

B200 percentages are of 2,250 TFLOP/s. M* before executes 3.4× the useful FLOPs
(139.85 / 303.04 TFLOP per chunk), so its conv-kernel rate is computed on
executed FLOPs, as in the PR's `profile.csv`.

## 4. Output check (exactness tiers)

Whole stream (1 + 85 chunks, 1,021 frames), one fresh process per decode,
compared with stock diffusers bf16 and an fp32 reference (TF32 off), both
decoded on the H200 (`quality_check.csv`, `coverage.csv`). The check covers
md5, PSNR and diff counts only; `lpips` isn't in the M* env and wasn't
installed, so there is no SSIM or LPIPS.

| VAE | decoder | values ≠ stock (≥ 1 / ≥ 2 levels, max) H200 · B200 | PSNR u8 vs fp32 mean / worst H200 · B200 | seam dB H200 · B200 | 2nd process H200 · B200 |
|---|---|---|---|---|---|
| Wan 2.1 | M* before | 30.00% / 2.40%, 80 · 30.00% / 2.40%, 83 | 51.76 / 46.68 (c72 f1) · 51.76 / 46.68 (c72 f1) | −0.59 · −0.59 | |
| Wan 2.1 | `mstar_warm` | 15.65% / 0.12%, 22 · 15.65% / 0.12%, 19 | 54.40 / 53.97 · 54.40 / 53.98 | +0.03 · +0.03 | max 9 levels · max 1 level |
| Wan 2.1 | `mstar_warm_nobench` | 15.65% / 0.12%, 22 · 15.65% / 0.12%, 17 | 54.40 / 53.97 · 54.40 / 53.98 | +0.03 · +0.03 | identical · identical |
| Wan 2.1 | stock bf16 | 0 · 0 | 54.84 / 54.34 · 54.84 / 54.36 | +0.02 · +0.02 | |
| Wan 2.2 | M* before | 30.38% / 4.04%, 207 · 30.38% / 4.04%, 207 | 50.11 / 34.83 (c73 f1) · 50.11 / 34.83 (c73 f1) | −2.30 · −2.30 | |
| Wan 2.2 | `mstar_warm` | 13.11% / 0.077%, 37 · 13.11% / 0.076%, 38 | 55.28 / 54.56 · 55.27 / 54.48 | +0.08 · +0.08 | max 13 levels · max 4 levels |
| Wan 2.2 | `mstar_warm_nobench` | 13.11% / 0.077%, 35 · 13.11% / 0.077%, 34 | 55.28 / 54.60 · 55.27 / 54.54 | +0.08 · +0.08 | identical · identical |
| Wan 2.2 | stock bf16 | 0 · 0 | 55.12 / 54.32 · 55.12 / 54.15 | +0.08 · +0.08 | |

- **Bit-exact tier holds.** The warm cache alone (eager, default layout, one
  latent per call, no `cudnn.benchmark`) gives byte-identical frames to one
  stock decode through M*'s node, for all 4 window patterns on both VAEs
  (1+3+3+3+3+2, 1+2+5+4+3, 8+6+1, and the clip-start latent alone). With this
  change's defaults, the node differs from stock by at most 3–5 uint8 levels
  (58.3–63.3 dB vs stock), as on the B200.
- **Within-bf16-rounding tier holds.** `mstar_warm` matches the B200 to 0.01 dB
  on mean PSNR vs fp32, to 0.01% on the share of differing values, and on the
  seam. The worst frame moves by ≤ 0.08 dB. There is no drift and no seam.
- **Not-equivalent tier holds.** M* before reproduces the B200 numbers, including
  the 34.83 dB worst frame on a Wan 2.2 chunk boundary.
- **md5.** No stream's md5 matches its B200 counterpart, stock bf16 included.
  That's expected: different conv kernels round differently. Aggregate metrics
  match.
- **fp32 reference across GPUs.** The H200 fp32 decode of the first 10 steady
  chunks is bit-identical to the B200's reference for Wan 2.1 (inf dB). For
  Wan 2.2 it differs at fp32 rounding level (134.1 dB float with peak 1,
  95.1 dB uint8).
- **Timing-run PSNR.** The timing runs compute PSNR against the B200 fp32
  reference tensors, and every one matches the B200 runs to the printed 0.01 dB
  (uint8: `mstar_warm` 54.42 / 55.40, M* before 52.09 / 51.08,
  `px_nopad_bench` 54.98 / 55.27).

## 5. What behaves differently on H200

- **`cudnn.benchmark` matters far more on H200 for Wan 2.1, and not at all for
  Wan 2.2.**
  - Wan 2.1: it saves 22.8 ms per chunk in the prototype (18.2%) and 22.6 ms in
    the in-tree node (17.9%), against 3.4 ms and 2.2 ms on B200. It is the
    second-largest step on H200, after keeping the cache.
  - Why: without it, cuDNN 9.20's heuristics pick an Ampere-generation kernel
    (`sm80_xmma_fprop_implicit_gemm ... tilesize256x32x32`) for 7 conv calls per
    chunk, which take 55.8 ms. With it, those convs run on `sm90_xmma` Hopper
    kernels and conv time drops from 100.5 to 77.7 ms, matching the wall-time
    gap.
  - M* before (no benchmark) spends 181 ms per chunk in that same `sm80` kernel
    (77 calls). That is part of why M* before is 1.72× slower on H200 than on
    B200.
  - Wan 2.2: the heuristics already pick `sm90` kernels, so it changes nothing
    (139.97 vs 140.17 ms; conv 122.2 vs 122.8 ms).
  - Implication: on Hopper, keeping `vae_decode_cudnn_benchmark=True` as the
    default is worth ~18% for Wan 2.1. Turning it off for determinism costs much
    more on H200 than on B200.
- **Compile gains are smaller.**
  - Stock eager → compiled: 1.62× on H200 vs 2.00× on B200.
  - In-tree eager → compiled: 1.85× vs 2.29× (Wan 2.1), 1.59× vs 1.93× (Wan 2.2).
  - The eager paths lose only 1.4–1.56× going from B200 to H200, but the compiled,
    compute-bound paths lose 1.83–1.93×. The B200's lead is largest where convs
    dominate: its dense bf16 peak is 2.27× the H200's, against 1.67× for HBM
    bandwidth.
- **Speedup vs M* before is lower** (5.6× vs 6.0× on Wan 2.1, 4.4× vs 5.0× on
  Wan 2.2). M* before's layout conversions and elementwise traffic, and its
  `sm80` conv picks, slow it down less, relative to the optimized path, on H200
  than on B200.
- **Channels-last** saves more milliseconds on H200 (31.6 vs 27.6 ms) but a
  smaller share (18.9% vs 28.7%). One call per chunk and conv-internal padding
  each give 3–5% on H200, against 6–7% on B200.
- **Closer to peak.** `mstar_warm` reaches 40% of the H200's 989 TFLOP/s on
  Wan 2.1 and 64% on Wan 2.2, against 32% and 53% of 2,250 on B200. The conv
  kernels alone run at 53% and 73% of the H200 peak.
- **More kernels per chunk** (in-tree 175 vs 149 on Wan 2.1, 189 vs 164 on
  Wan 2.2) because cuDNN selects different kernels. Host gaps stay small: GPU
  kernel time is 95–99% of wall time.
- **In-tree vs prototype**: +0.69 ms (0.7%) on Wan 2.1 and +0.29 ms (0.2%) on
  Wan 2.2, against +0.45 and +1.47 ms on B200.
- **Less deterministic with `cudnn.benchmark` on.** A second fresh process
  differs by up to 9 (Wan 2.1) / 13 (Wan 2.2) uint8 levels, against 1 / 4 on
  B200; the aggregate metrics are identical. With it off, a second process is
  byte-identical on both GPUs.
- **fp32 reference decode is faster** on H200: 1,481 vs 2,407 ms per chunk on
  Wan 2.1, and 3,206 vs 3,248 ms on Wan 2.2 (TF32 off).

## 6. GPU time

About **83 min of H200 time**, all on GPU 0, one process at a time in tmux.
The main job ran 16:44:57–18:04:40 UTC (79.7 min):

| phase | time |
|---|---|
| Wan 2.1 3-run session | 18.0 min |
| Wan 2.1 single-run ladder | 7.4 min |
| Wan 2.2 3-run session | 19.8 min |
| fp32 reference check | 1.9 min |
| 6 profiles | 6.2 min |
| whole-stream output decodes, coverage and checks | 25.5 min |
| drift run | 0.9 min |

Three short follow-ups added 3.3 min: the Wan 2.1 coverage re-run, the
`mstar_warm_nobench` profiles, and the Wan 2.2 `mstar_warm_nobench` timing. The
benchmark processes' own `wall_s` sums to ~46 min of the total; the rest is
model load, compile, profiling and the quality decodes.

## 7. Caveats and harness notes for the PR

- **GPU hygiene.** Every GPU process (about 60) checked for an idle GPU first;
  no wait was ever needed. Each run JSON's `gpu_apps_at_start` lists
  only the run itself. The main job's 1 s `nvidia-smi` trace was overwritten
  when the follow-up job relaunched the sampler; `runs/gpu_trace_followup.csv`
  only covers the follow-up.
- **Single runs** (\*): the ladder's steps 1–4, `diffusers_bf16`, and both
  `mstar_warm_nobench` rows. On B200, the 3-run spread of the ones it repeated
  was ≤ 0.25%; on H200 the 3-run sessions spread ≤ 1.3%.
- **`quality.py coverage`** at `c56b572b` failed at the very end if `--out`
  didn't exist (`append_csv` did not create it); fixed since. Coverage was
  re-run after creating the directory.
- **`profile_kernels.py`** at `c56b572b` only accepts `mstar_warm[_eager]`.
  `mstar_warm_nobench` was profiled through a helper that calls the same
  `mstar_warm_node(..., cudnn_benchmark=False)`; `profile_kernels.py` now
  covers it through `node_overrides`.
- **Not run on H200:** `mstar_ar_ctx8_eager`, `px_base`, `px_fp16`, the other
  `px_*` ablations, served-window `bench_decode_nodes.py`, and the
  SSIM / LPIPS / flicker / recon parts of `quality.py analyze`.
