# Wan VAE decode (mstar-project/mstar#425): benchmark data index

Every number in the PR body comes from the files indexed here. Two places on this branch hold them:
- `results/` (this folder): raw runs, logs, harnesses and per-dataset summaries.
- [`benchmark/wan_vae_decoder/results/`](../benchmark/wan_vae_decoder/results/): the summary CSVs, `RESULTS.md`, charts and samples
  built from those runs, which the PR body links.

Each folder's README lists what was left out (weights, latents, raw frames, compile caches), with its location and sha256.

Shared inputs for every decoder table. The latents come from a 6.5 s 4K clip cropped to 832×480, ping-ponged to 1,021 frames, and
encoded with each VAE. Weights: `Wan-AI/Wan2.1-T2V-1.3B-Diffusers@0fad780a`, `Wan-AI/Wan2.2-TI2V-5B-Diffusers@b8fff731`. sha256
of the latents and fp32 references: [`b200/in-tree/README.md`](b200/in-tree/README.md#not-committed-left-on-the-vm).

Paths in committed logs and scripts are placeholders: `${VM_HOME}` (VM user home), `${MSTAR}` / `${MSTAR_PR}` (main / PR
checkout), `${SCRATCH}`, `${WAN_BENCH}` (local work dir), `${TMPDIR}`, `${VM_SSH}`, `${SSH_KEY}`, `<vm-ip>`, `<project-id>`, `<subnet-id>`.

All runs are from 2026-10-13 (UTC). "Commit" is the M* code the run imported (`env.mstar_commit` in the JSONs, where recorded).
`99f5108b` is upstream main, used for "M* before" and for the prototype harness.

## B200 (main)

One NVIDIA B200 (178 GiB), driver 580.173.02, on Nebius VMs `wan-bench-b200`, `wan-bench-b200-vllm` and `wan-bench-b200-sglang`.

| dataset | what was measured | GPU / VM | software stack | code commit | runs | folder | PR table |
|---|---|---|---|---|---|---|---|
| headline ladder | fp32 ref, stock diffusers, M* before (ctx 1–8), prototype steps; ms/chunk, PSNR | B200 `wan-bench-b200` | torch 2.12.1+cu132, cuDNN 9.20.0, diffusers 0.39.0 (uv venv) | `99f5108b` + prototype harness | 34 variants × 1 | [`b200/in-tree/runs/wan21`, `wan22`](b200/in-tree/runs/) | Summary ladder chart; Reproduce; prototype-harness bullet |
| per-change ladder | `mstar_step0..5` compiled + eager, in-tree node | B200 `wan-bench-b200` | same | `4b8b4789` | 3 fresh processes × 10 × 2 VAEs = 60 | [`b200/in-tree/runs/ablation/ladder*`](b200/in-tree/runs/ablation/) | Summary decoder table; Per-change gains |
| ladder kernel profiles | kernels/chunk, TFLOP/s per step | B200 `wan-bench-b200` | same | `4b8b4789` | 20 profiled variants | `b200/in-tree/runs/ablation/profile_ladder*` | Per-change gains (kernels, TFLOP/s) |
| prototype variance | diffusers eager / compile, M* before, `px_nopad_bench` etc. | B200 `wan-bench-b200` | same | `99f5108b` + prototype | 3 × 9 = 27 | `b200/in-tree/runs/ablation/variance` | Comparison (diffusers rows); prototype-harness bullet |
| in-tree sessions | `mstar_warm` (± `cudnn.benchmark`, eager) vs `px_nopad_bench` | B200 `wan-bench-b200` | same | `4b8b4789` | 3 × 4 × 2 VAEs = 24 | `b200/in-tree/runs/ablation/intree*` | H200 cross-check (B200 column); Output equivalence (determinism) |
| `latents_per_call` sweep | `windowed_decode_latents_per_call` = 0 / 1 / 2 / 3 | B200 `wan-bench-b200` | same | `60fbc5cd` | 2 × 4 × 2 VAEs = 16 | `b200/in-tree/runs/ablation/lpc*` | Summary (`=1`); Per-change default bullet; Memory (`latents_per_call` table); Comparison (M* rows) |
| decoder node, served window | node alone, 8-latent window (6 new), `lpc0..3` | B200 `wan-bench-b200` | same | `60fbc5cd`, `4b8b4789` | 20 | `b200/in-tree/runs/ablation/nodes`, `b200/in-tree/summary/nodes.csv` | Memory (served-window rows) |
| kernel profiles | category breakdown (stock, M* before, in-tree), raw traces | B200 `wan-bench-b200` | same | `4b8b4789` | 20 kernel tables + 4 traces | `b200/in-tree/runs/ablation/profile*`, `profile_traces.tar.zst` | Where the time goes |
| output equivalence | uint8 vs stock and fp32: PSNR, SSIM, LPIPS, seam, 2nd process | B200 `wan-bench-b200` | same | `4b8b4789` | 28 full-stream decodes | `b200/in-tree/runs/ablation/quality*` | Output equivalence; Per-change PSNR / output columns |
| encode + decode vs source | reconstruction vs the source clip | B200 `wan-bench-b200` | same | `4b8b4789` | 10 | `b200/in-tree/runs/ablation/quality_recon*` | Output equivalence (encode + decode bullet) |
| prototype extras | `px_*` splits, TF32 check of the fp32 reference, controls | B200 `wan-bench-b200` | same | `99f5108b`, `4b8b4789` | 20 | `b200/in-tree/runs/ablation/{px,tf32,control,wan22}` | Per-change prototype bullet; fp32 reference (TF32 off) |
| vLLM-Omni | `WanStreamingDecoder` default / best / + `cudnn.benchmark`, on M*'s latents | B200 `wan-bench-b200-vllm` | container `vllm/vllm-omni:nightly-f69b1f2b` (torch 2.13.0+cu130, cuDNN 9.20.0, diffusers 0.40.0) | vLLM-Omni `f69b1f2b` | 3 × 3 × 2 VAEs = 18, + 12 screen/control/one-shot, 6 profiles | [`b200/vllm-omni`](b200/vllm-omni/) | Comparison with vLLM-Omni and SGLang |
| SGLang | `DecodingStage` / `causal_decode`: default, best, compile, realtime | B200 `wan-bench-b200-sglang` | SGLang v0.5.21 venv (torch 2.13.0+cu130, cuDNN 9.20.0, diffusers 0.39.0) | SGLang `e00930c5` | 3 × 4 × 2 VAEs = 24, + 14 screen, 8 profiles, 19 md5 checks | [`b200/sglang`](b200/sglang/) | Comparison with vLLM-Omni and SGLang |
| quality and memory compare | each engine's full stream vs stock / fp32; held cache memory | B200 (all three VMs) | the engines' stacks above; M* torch 2.12.1+cu132 | M* `4b8b4789`, vLLM-Omni `f69b1f2b`, SGLang `e00930c5` | 19 streams + memory probes | [`b200/quality-and-memory-compare`](b200/quality-and-memory-compare/) | Comparison (quality columns, † note); Memory (per-stream cache vs other engines) |
| Cosmos3-Edge served e2e | streaming rollouts at 1 / 4 / 8 streams: TTFF, gaps, fps, decoder share, memory, output vs fp32 | B200 `wan-bench-b200-vllm` | torch 2.12.1+cu130, cuDNN 9.20, diffusers 0.39.0, flashinfer 0.7.1 | before `99f5108b`, after `60fbc5cd` (`=0`, `=1`) | 3 variants × (3 kv + 2 chained, 2 rounds × 4 streams, 1 round × 8 streams) | [`b200/e2e-cosmos3-edge`](b200/e2e-cosmos3-edge/) | Summary (served); End-to-end; Memory (decoder-only 4-stream table, 8-stream OOM) |
| re-time at the PR defaults | in-tree node at `=1` with `cudnn.benchmark` off, M* before, opt-in `=0` and `cudnn.benchmark`; chunk and served window | B200 `wan-bench-b200` | torch 2.12.1+cu132, cuDNN 9.20.0, diffusers 0.39.0 | `663b7e9b`, `913295e4` (compact cache); main `99f5108b` | 2 fresh processes × 22 configurations | [`b200/retime-defaults`](b200/retime-defaults/), `benchmark/wan_vae_decoder/results/b200/retime.csv` | Summary decoder table; Comparison (M* defaults row); Memory (peak rows) |
| Cosmos3-Edge served e2e re-run | one stream, kv and chained, before vs the final defaults | B200 `wan-bench-b200` | torch 2.12.1+cu132, cuDNN 9.20, diffusers 0.39.0, flashinfer 0.7.1 | before `99f5108b`, default `913295e4` | 2 variants × (3 kv + 2 chained) | [`b200/e2e-cosmos3-edge-rerun`](b200/e2e-cosmos3-edge-rerun/) | Summary (re-run bullet); End-to-end |
| torch 2.13 Wan 2.2 bug | Inductor comprehensive-padding miscompile: repro, minified graph, fix verification | B200 `wan-bench-b200-vllm` | vLLM-Omni container (torch 2.13.0+cu130) and torch 2.12.1+cu132 | `4b8b4789`, checked at `c56b572b` | 19 short runs | [`b200/torch213-wan22-compile-bug`](b200/torch213-wan22-compile-bug/) | Output equivalence (torch 2.13 bullet) |
| samples | Wan2.2-TI2V-5B clips (fox, waves) at 1280×704 through each step | B200 `wan-bench-b200-vllm` | torch 2.12.1+cu130, cuDNN 9.20 | `c56b572b` | 8 steps × 2 prompts × 1 | [`b200/samples`](b200/samples/) + `benchmark/wan_vae_decoder/results/b200/samples/` | Samples; Memory (1280×704 row) |
| memory fix (B200) | clone the compiled cache slices: memory before / after | B200 `wan-bench-b200-sglang` | — | — | **pending**: no `SUMMARY.md` or `memory_after.csv` in the handoff yet | [`b200/memory-fix`](b200/memory-fix/) | Memory ("a fix is in progress"); Open question 4 |
| initial exploration | first session: prototype `rx_*` / `px_*` variants, batch 2, variance, profiles, TF32 | B200 `wan-bench-b200` | torch 2.12.1+cu132, cuDNN 9.20.0, diffusers 0.39.0 | `99f5108b` + prototype (private `experiment/wan-vae-decoder` @ `c7d2c28a`) | 80 run JSONs + 17 profiles | [`b200/initial-exploration`](b200/initial-exploration/) | Things that didn't help |
| design | diagrams and write-up | — | — | — | — | [`design`](design/) + `benchmark/wan_vae_decoder/results/b200/design/` | Design |

## H200 (cross-check)

One NVIDIA H200 (141 GB, sm_90), driver 580.173.02, local host.

| dataset | what was measured | GPU / host | software stack | code commit | runs | folder | PR table |
|---|---|---|---|---|---|---|---|
| H200 cross-check | in-tree node, M* before, ladder steps, quality tiers, `cudnn.benchmark` | H200 (local) | torch 2.12.1+cu132, CUDA 13.2, cuDNN 9.20.0, diffusers 0.39.0 (venv) | `c56b572b` | 3 × 4 × 2 VAEs = 24, + 10 single runs, 8 profiles, 14 quality decodes | [`h200`](h200/) | H200 cross-check; `cudnn.benchmark` bullet in Per-change gains |
| memory census | tensor census of the warm cache, candidate clone fix (Wan 2.1, `=1`) | H200 (local) | same | `daa44149` (+ prototype fix) | 5 probe runs | [`h200/memory-census-notes.md`](h200/memory-census-notes.md) | Memory (held storage under compile); Open question 4 |
| harness smoke | first smoke runs of the prototype harness on tiny inputs | H200 (local) | torch 2.12.1+cu132, cuDNN 9.20.0 | `df0f2b13` | 9 | [`h200/harness-smoke`](h200/harness-smoke/) | — (development only) |
