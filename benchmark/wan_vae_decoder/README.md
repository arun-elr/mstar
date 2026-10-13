# Wan VAE streaming-decode benchmark

Measures the Wan VAE decoder the way a streaming video rollout uses it: the
latent stream arrives in chunks and each chunk is decoded as soon as it lands.

This is the results branch for the Wan VAE decode change (mstar-project/mstar#425):
the full benchmark tooling, the B200 and H200 results ([`results/b200/RESULTS.md`](results/b200/RESULTS.md),
[`results/h200/RESULTS.md`](results/h200/RESULTS.md)), charts, samples
([`results/b200/samples/`](results/b200/samples/)) and the harnesses that ran vLLM-Omni and SGLang
through the same stream protocol ([`external/`](external/)). The PR branch carries only the headline
ladder (`bench_wan_decoder.py`, `run_all.sh`).

## What is measured

| | |
|---|---|
| VAE | Wan 2.1 (`Wan-AI/Wan2.1-T2V-1.3B-Diffusers`, 16 latent channels, 8× spatial); Wan 2.2 (`Wan-AI/Wan2.2-TI2V-5B-Diffusers`, 48 channels, 16× spatial) |
| Chunk | 3 latent frames → 12 RGB frames at 480×832, batch 1 (`[1, 16, 3, 60, 104]` for Wan 2.1, `[1, 48, 3, 30, 52]` for Wan 2.2) |
| Timed | decoder + conversion to uint8 frames, CUDA events around each chunk, synchronized per chunk |
| dtype | bf16 unless the variant says otherwise |
| Stream | first latent untimed, 10 warm-up chunks untimed (absorbs `torch.compile` and cuDNN autotuning), 75 timed chunks; the stream is replayed twice from an empty cache and the 150 samples pooled |
| Reported | median ms per chunk (p10 / p90 / mean in the raw JSON); decoder fps = 12 / median |
| Latents | a 6.5 s 4K natural video clip (hands drawing on a tablet), cropped to 832×480 and ping-ponged to 1,021 frames, encoded with the VAE's own encoder (`latent_dist.mode()`): 256 latent frames = 1 + 85 chunks. Timing does not depend on latent values; PSNR does, so real content is used. `prepare_latents.py --video PATH` encodes any clip the same way; `--random` is the fallback. The clip and latents are not committed. |
| Processes | one variant per process. `run_ablation.sh` gives every variant fresh Inductor/Triton caches (fresh compile) and repeats the variance set 3 times in the orders ABC / CBA / BCA |
| GPU hygiene | `run_all.sh` refuses to start if the GPU has other compute processes, logs an `nvidia-smi` trace and re-runs the stock baseline at the end as a drift check |

### Accuracy conventions

- Reference: diffusers fp32, eager, per-frame calls, **TF32 off**
  (`torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False`;
  the flags each run saw are recorded in its JSON). `ref_fp32_tf32` is the
  control with TF32 allowed.
- PSNR over the first 10 steady chunks, on the decoder's float output in
  [-1, 1]. Tables use **peak 1** (10·log10(1/MSE)); the raw JSON also has
  peak 2 (the [-1, 1] range, 6.02 dB higher) and a uint8 PSNR (peak 255,
  against the reference quantized the same way).
- The M* streaming variants only return uint8 frames, so they have uint8 PSNR only.
  They take the normalized latents the DiT hands the decoder node and
  de-normalize them in bf16 (as M* serves); the other variants are fed the
  raw latents.

### Output equivalence

`quality.py` decodes the whole stream (1 + 85 chunks, 1,021 frames) once per
variant, each in a fresh process, and compares every frame with stock
diffusers bf16 (same dtype) and with the fp32 reference: max / mean absolute
error, the share of uint8 values that differ by ≥ 1 and ≥ 2 levels, per-frame
uint8 PSNR (chunk boundaries every 12 frames), SSIM (11×11 Gaussian, σ 1.5),
LPIPS (AlexNet), and temporal flicker (mean |Δ_t(variant) − Δ_t(stock)| of the
frame-to-frame differences, uint8 levels). Drift is the per-chunk PSNR trend;
the seam is the PSNR of each chunk's first frame minus its other frames.
`cudnn.benchmark` variants and the in-tree node without it are decoded in a
second fresh process and compared bit for bit; each stream's md5 is compared
with stock and with the previous ladder step. The M* node rows are also
compared with `diffusers_bf16_node_input`, stock fed the node's own bf16
de-normalized input.

## Variants

Each row of the change ladder maps to a variant (`bench_wan_decoder.py` docstring
has the full list):

| variant | what it runs | M* switch |
|---|---|---|
| `mstar_ar_ctx8` | M*'s streaming decode before this change: `Cosmos3VAEDecoderSubmodule._decode_pixels` on 8 context latents + 3 new ones, empty cache every chunk, compiled | `windowed_decode_warm_cache=False` |
| `diffusers_bf16` | stock diffusers, eager, cache kept across chunks, one latent per decoder call | — |
| `diffusers_bf16_compile` | + `torch.compile` of the decoder (change 1: warm cache) | `windowed_decode_warm_cache` |
| `channels_last_compile` | + channels-last weights and activations (change 2) | `vae_decode_channels_last` |
| `channels_last_one_call_compile` | + one decoder call per chunk (change 3) | `windowed_decode_latents_per_call=0` |
| `px_nopad` | + H/W padding inside the conv, channels-last at every conv (change 4) | `ChannelsLastCausalConv3d` (part of `vae_decode_channels_last`) |
| `px_nopad_bench` | + `cudnn.benchmark` (change 5) | `vae_decode_cudnn_benchmark` |
| `mstar_warm` | the in-tree `Cosmos3VAEDecoderARSubmodule` at the checkout's config defaults (`=1`, `cudnn.benchmark` off since 663b7e9b; the archived runs before that used `=0` + `cudnn.benchmark`) | all of the above |
| `mstar_warm_nobench` | the same with `cudnn.benchmark` off (control) | `vae_decode_cudnn_benchmark=False` |
| `mstar_step{0,1,2,3,5}[_eager]` | the in-tree ladder: M*'s node with the switches turned on one at a time (`STEPS` in `bench_wan_decoder.py`); step 4 (conv-internal padding) has no switch of its own and comes with step 2 | the four switches above |

`px_<flags>` variants are the prototype harness: compiled, channels-last, one
call per chunk, plus the flags (`nopad`, `bench`, `cg` manual CUDA graph, `ro`
reduce-overhead, `maxat` max-autotune, `fp16`). The prototype also runs
nearest-exact upsampling in bf16 instead of diffusers' fp32 round trip; that
alone changes nothing measurable (`px_base` 63.19 ms vs
`channels_last_one_call_compile` 63.26 ms), so the in-tree code keeps diffusers'
upsample.

## Reproduce

Needs a CUDA GPU, the M* environment, and the two VAEs in the local Hugging Face
cache (`hf download Wan-AI/Wan2.1-T2V-1.3B-Diffusers --include "vae/*"`, same for
Wan 2.2). All scripts run from this directory.

```bash
# CPU-only correctness checks (real Wan 2.1 weights, tiny spatial size)
python check_equivalence.py

# Full suite: latents, fp32 reference, every variant (raw output in runs/<run>/)
GPU=0 VIDEO=/path/to/clip.mp4 WAN22=1 RUN=b200 bash run_all.sh

# Ablations, 3-run variance and kernel profiles, reusing that run's latents
SRC=runs/b200 bash run_ablation.sh px
SRC=runs/b200 bash run_ablation.sh variance
SRC=runs/b200 bash run_ablation.sh wan22
SRC=runs/b200 bash run_ablation.sh intree    # in-tree node vs prototype, both VAEs; served windows
SRC=runs/b200 bash run_ablation.sh quality   # output equivalence (needs the lpips package)

# One variant by hand
python bench_wan_decoder.py --variant mstar_warm --latents runs/b200/wan21/latents.pt \
  --ref runs/b200/wan21/ref_fp32_outputs.pt --out runs/b200/wan21

# Summary CSVs and charts (charts need only matplotlib)
python summarize.py runs/b200 --out results/b200
python plot_results.py --results results/b200
```

`quality.py decode` / `analyze` / `coverage` (driven by `run_ablation.sh
quality`) write `quality.csv`, `quality_frames.csv` (per-frame PSNR, for
`charts/per_frame_psnr.png`), `coverage.csv` (M*'s node over irregular
windows) and, with `VIDEO=<source clip>`, `recon.csv` (encode + decode PSNR
against the clip). `profile_kernels.py` writes the kernel-time breakdown (categories from kernel
names, `kernel_categories.py`) and, with `--flops`, the FLOP and minimum-traffic
count per chunk. `bench_decode_nodes.py` drives M*'s decoder nodes directly with
served window sizes (8-latent windows, 2 overlapping) and config overrides, and
`e2e_stream.sh` serves `configs/cosmos3_edge.yaml` with each decode configuration
for an end-to-end streaming rollout.

## Reproducing every table

One GPU, one job at a time, from this directory. `SRC=runs/b200` is the
`run_all.sh` output whose latents and fp32 references every later phase reuses;
`summarize.py runs/b200 --out results/b200` writes the CSVs and
`plot_results.py` the charts.

| table | command | CSV |
|---|---|---|
| per-change gains through M*'s node (both VAEs) | `SRC=runs/b200 bash run_ablation.sh ladder` | `ladder_wan21.csv`, `ladder_wan22.csv` |
| prototype ladder, run-to-run variance | `bash run_all.sh`, then `run_ablation.sh variance` | `variance.csv`, `runs.csv` |
| in-tree vs prototype, served windows, `cudnn.benchmark` control | `run_ablation.sh intree` | `intree.csv`, `nodes.csv` |
| kernel breakdown, Wan 2.2 rows, things that did not help | `run_ablation.sh px`, `run_ablation.sh wan22` | `profile.csv`, `runs.csv` |
| output equivalence, window coverage, reconstruction | `VIDEO=<clip> run_ablation.sh quality` | `quality.csv`, `quality_frames.csv`, `coverage.csv`, `recon.csv` |

The vLLM-Omni and SGLang rows, the cross-engine `comparison.csv` and the samples
come from the scripts under [`external/`](external/) and [`samples/`](samples/). Another GPU needs nothing new:
`GPU=<index> RUN=<name> bash run_all.sh`, then the `run_ablation.sh` phases with
`SRC=runs/<name>`.

## Adding results

Committed results are small CSV/markdown summaries plus charts; raw JSON,
latents, reference tensors, traces and compile caches stay out of git (`runs/`
is ignored). To add a measurement: run it into `runs/<name>/`, regenerate the
CSVs with `summarize.py` and the charts with `plot_results.py`, and update the
tables in `results/<gpu>/RESULTS.md`.
