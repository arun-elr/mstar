# Output quality and peak memory: vLLM-Omni and SGLang vs M* (Wan VAE streaming decode, B200)

Every engine row was compared with M*'s stock diffusers bf16 stream and fp32 reference, over the
full 1,021-frame stream (first latent plus 85 chunks of 12 frames) from the same draw.mp4
latents the timing used. The metrics are `quality.py`'s own functions (`md5`, `psnr_frames`, `ssim`,
`worst`, `frame_place`, and the seam/drift formulas from `analyze()`), imported from the PR
worktree unchanged. M* rows are copied from the PR's `results/b200/quality.csv` as-is.
No videos were made.

## Table (ready to paste)

| VAE | engine | setting | ms per chunk | peak GiB | PSNR vs fp32 mean / worst (dB) | SSIM vs fp32 worst | % values ≠ stock (≥1 / ≥2) | md5 = stock? | seam dB |
|---|---|---|---:|---:|---|---:|---|---|---:|
| Wan 2.1 | diffusers | stock bf16, eager | 191.98 | 4.44 | 54.84 / 54.36 (c26 f4) | 0.9979 | 0.00% / 0.000% | yes | +0.02 |
| Wan 2.1 | M* | before (re-decode 8 context latents) | 335.25 | 6.33 | 51.76 / 46.68 (c72 f1⟂) | 0.9923 | 30.00% / 2.401% | no | -0.59 |
| Wan 2.1 | M* | this PR (`mstar_warm`, defaults) | 56.30 | 19.03 | 54.40 / 53.98 (c73 f12⟂) | 0.9978 | 15.65% / 0.125% | no | +0.03 |
| Wan 2.1 | vLLM-Omni | default (`vae_fast_path=lossless`) | 97.32 | 4.53 | 54.84 / 54.36 (c26 f4) | 0.9979 | 0.00% / 0.000% | yes | +0.02 |
| Wan 2.1 | vLLM-Omni | best (`--vae-fast-path channels_last`) | 64.08 | 3.82 | 54.82 / 54.31 (c59 f9) | 0.9979 | 7.61% / 0.003% | no | +0.02 |
| Wan 2.1 | SGLang | default (`quality=lossless`, bf16) | 107.64 | 6.81 | 55.21 / 54.40 (c82 f12⟂) | 0.9980 | 14.60% / 0.002% (7.61%) | no | +0.07 |
| Wan 2.1 | SGLang | best (`--quality extra-high`, bf16) | 92.50 | 3.80 | 55.16 / 54.33 (c82 f12⟂) | 0.9980 | 13.38% / 0.001% (5.80%) | no | +0.07 |
| Wan 2.1 | SGLang | reference: stock bf16 decode, uint8 conversion in fp32 like SGLang | — | — | 55.16 / 54.34 (c82 f12⟂) | 0.9980 | 9.33% / 0.000% (0.00%) | no | +0.08 |
| Wan 2.2 | diffusers | stock bf16, eager | 184.57 | 5.53 | 55.12 / 54.15 (c73 f8) | 0.9980 | 0.00% / 0.000% | yes | +0.08 |
| Wan 2.2 | M* | before (re-decode 8 context latents) | 368.85 | 6.44 | 50.11 / 34.83 (c73 f1⟂) | 0.9724 | 30.38% / 4.038% | no | -2.30 |
| Wan 2.2 | M* | this PR (`mstar_warm`, defaults) | 73.76 | 15.95 | 55.27 / 54.48 (c31 f8) | 0.9981 | 13.11% / 0.076% | no | +0.08 |
| Wan 2.2 | vLLM-Omni | default (`vae_fast_path=lossless`) | 105.92 | 5.42 | 55.12 / 54.15 (c73 f8) | 0.9980 | 0.00% / 0.000% | yes | +0.08 |
| Wan 2.2 | vLLM-Omni | best (`--vae-fast-path channels_last`) | 77.85 | 4.66 | 55.02 / 54.34 (c64 f5) | 0.9980 | 6.26% / 0.008% | no | +0.07 |
| Wan 2.2 | SGLang | default as served (fp32 decode) | 170.97 | 9.33 | 69.27 / 68.28 (c72 f3) | 0.9999 | 19.86% / 0.037% (13.98%) | no | +0.13 |
| Wan 2.2 | SGLang | default + `--vae-decode-precision bf16` | 115.72 | 7.76 | 56.58 / 55.56 (c64 f5) | 0.9984 | 11.74% / 0.008% (6.26%) | no | +0.13 |
| Wan 2.2 | SGLang | best (`--quality extra-high`, bf16) | 94.36 | 4.73 | 56.65 / 55.38 (c73 f7) | 0.9984 | 10.82% / 0.003% (4.42%) | no | +0.15 |
| Wan 2.2 | SGLang | reference: stock bf16 decode, uint8 conversion in fp32 like SGLang | — | — | 56.65 / 55.32 (c73 f8) | 0.9984 | 7.80% / 0.000% (0.00%) | no | +0.15 |

How to read the columns:

- **ms per chunk** and **peak GiB** come from the existing timing runs. ms is the 3-run mean, and the peak is `torch.cuda.max_memory_allocated` over the timed steady chunks, identical in all 3 runs. Nothing was re-timed.
- **PSNR** is per frame on uint8 against M*'s fp32 reference (TF32 off). The worst frame is shown as cN fK, meaning chunk N, frame K of 12. ⟂ marks the first or last frame of a chunk.
- **% values ≠ stock** is measured against M*'s stock stream (`f2dc2237…` for Wan 2.1, `38576c14…` for Wan 2.2), share of uint8 values off by ≥1 and ≥2 levels. Max diff and frame indices are in the CSV. For SGLang, the number in parentheses is against the same stock decode converted to uint8 the way SGLang converts. See the first finding below for why that is the fair comparison for SGLang.
- **Seam** is computed as in `quality.py`: each chunk's first frame PSNR minus its other frames, averaged over chunks.
- **Drift:** the PSNR trend is within ±0.001 dB per chunk on every engine row (CSV column `psnr_trend_db_per_chunk`).

## Conclusion

Every bf16 engine configuration lands at bf16-rounding quality. Against fp32, all of them are within 0.5 dB of stock diffusers bf16 once the uint8 conversion is matched. Worst-frame SSIM is ≥ 0.9977, at most 0.13% of values differ from stock by two or more levels, and there are no seams (|seam| ≤ 0.15 dB) and no drift. So M*'s speedup (56.3 / 73.8 ms vs vLLM-Omni best at 64.1 / 77.9 and SGLang best at 92.5 / 94.4) comes at the same output quality.

- **vLLM-Omni default is byte-identical to stock** on both VAEs. Its md5 equals M*'s stock md5, which confirms vLLM-Omni's "lossless" claim against M*'s own reference.
- **SGLang's higher PSNR comes from its uint8 conversion, not its decode.** SGLang converts to uint8 from an fp32 [0, 1] tensor. M*'s stock and node streams round `x / 2 + 0.5` in bf16 first. Converting M*'s stock bf16 output SGLang's way gives exactly the "diffusers bf16" stream SGLang hashed on its VM (`ac038805…` / `8844f3b7…`, md5 equal), and that stream scores the same PSNR as SGLang's rows (55.16 vs 55.16–55.21 dB; 56.65 vs 56.58–56.65 dB). The ≥1-level difference counts have the same cause: SGLang best differs from M*'s stock in 13.4% of values, but from the conversion-matched stock in only 5.8%.
- **Worst frames on chunk boundaries are content, not seams.** SGLang's worst frame (c82 f12) is also the conversion-matched stock's worst frame. M* this PR's Wan 2.1 worst frame (c73 f12, 53.98 dB) ties with c26 f4 (53.98 dB in the `cudnn.benchmark`-off control). Every seam value is ≤ 0.15 dB.
- **M* this PR on Wan 2.1 is 0.44 dB below stock** (54.40 vs 54.84), the largest bf16 gap in the table. The PR's RESULTS.md attributes it to the node's bf16 latent de-normalization, which is unchanged by this PR. On Wan 2.2 the node is 0.15 dB above stock.
- **M* before is the only approximation.** Its boundary frames are 0.6 / 2.3 dB worse, and its worst frame drops to 34.8 dB on Wan 2.2.

## Peak memory

Below is the breakdown from one short probe run per engine and setting (`harness/mem_probe.py`). It uses the same decoder objects and stream as the timing, with 10 warm-up chunks and then 5 measured chunks. The probe peaks reproduce the timing peaks within 0.07 GiB.

Columns:

- **weights resident** is the parameters on the GPU during steady decode; the encoder share is in parentheses.
- **state held between chunks** is the memory still allocated after a chunk's output is freed. The parentheses give the logical size of the tensors behind it (elements × bytes) and how many tensors there are.
- **transient** is the peak of one chunk above that held state.

| VAE | engine | setting | latents per decoder call | peak GiB (timing / probe) | weights resident (encoder) | latents on GPU | state held between chunks: allocated (logical, tensors) | transient peak of one chunk |
|---|---|---|---:|---:|---|---:|---|---:|
| Wan 2.1 | vLLM-Omni | default | 1 | 4.53 / 4.49 | 0.24 (0.10) | 0.05 | 1.76 (1.76, 32 tensors) | 2.45 |
| Wan 2.1 | vLLM-Omni | best | 1 | 3.82 / 3.78 | 0.24 (0.10) | 0.05 | 1.76 (1.76, 32 tensors) | 1.73 |
| Wan 2.1 | SGLang | default | 1 | 6.81 / 6.74 | 0.14 (none) | 0.10 | 3.47 (3.47, 32 tensors) | 3.03 |
| Wan 2.1 | SGLang | best | 1 | 3.80 / 3.73 | 0.14 (none) | 0.10 | 1.76 (1.76, 32 tensors) | 1.74 |
| Wan 2.1 | M* | this PR, `windowed_decode_latents_per_call=0` (default) | 3 | 19.03 / 19.02 | 0.14 (none) | 0.10 | 10.20 (2.04, 332 tensors) | 8.58 |
| Wan 2.1 | M* | this PR, `windowed_decode_latents_per_call=1` | 1 | — / 8.23 | 0.14 (none) | 0.10 | 4.53 (2.04, 332 tensors) | 3.46 |
| Wan 2.2 | vLLM-Omni | default | 1 | 5.42 / 5.38 | 1.31 (0.28) | 0.04 | 1.33 (1.33, 32 tensors) | 2.69 |
| Wan 2.2 | vLLM-Omni | best | 1 | 4.66 / 4.62 | 1.31 (0.28) | 0.04 | 1.33 (1.33, 32 tensors) | 1.93 |
| Wan 2.2 | SGLang | default as served (fp32 decode) | 1 | 9.33 / 9.26 | 2.62 (0.56) | 0.07 | 2.65 (2.65, 32 tensors) | 3.89 |
| Wan 2.2 | SGLang | bf16 | 1 | 7.76 / 7.69 | 1.31 (0.28) | 0.07 | 2.62 (2.62, 32 tensors) | 3.66 |
| Wan 2.2 | SGLang | best | 1 | 4.73 / 4.66 | 1.31 (0.28) | 0.07 | 1.33 (1.33, 32 tensors) | 1.93 |
| Wan 2.2 | M* | this PR, `windowed_decode_latents_per_call=0` (default) | 3 | 15.95 / 15.92 | 1.03 (none) | 0.07 | 7.85 (1.61, 332 tensors) | 6.96 |
| Wan 2.2 | M* | this PR, `windowed_decode_latents_per_call=1` | 1 | — / 7.39 | 1.03 (none) | 0.07 | 3.47 (1.61, 332 tensors) | 2.81 |

**Is M*'s `=0` peak higher, and does `=1` bring it in line?** Yes, `=0` is higher: 19.0 / 15.9 GiB is 4–5× vLLM-Omni best (3.8 / 4.7) and SGLang best (3.8 / 4.7). `=1` cuts it to 8.2 / 7.4 GiB, which is still about 2× the competitors. The main reason is the held state. M*'s 332 cached tensors hold only 2.0 / 1.6 GiB of data, but they are views that keep whole activation buffers alive: 4.5 GiB at `=1` and 10.2 GiB at `=0`. vLLM-Omni and SGLang best hold a compact 1.76 / 1.33 GiB cache (the same size as the logical cache in M*'s `nodes.csv`, 1,801 / 1,359 MiB).

Notes on the memory numbers:

- **Weights are not what separates the engines.** vLLM-Omni keeps the encoder resident (0.10 / 0.28 GiB). SGLang keeps it for Wan 2.2 only. M* drops it, and its node casts the fp32-loaded decoder to bf16 in place on the first decode, so steady-state weights are bf16.
- **The M* `=1` row is a probe measurement, not a timing run.** Its config is `mstar_warm` with `windowed_decode_latents_per_call=1`. Its 8.23 / 7.39 GiB peak matches the PR ladder's step 2 (compiled, one latent per call: 8.24 / 7.39 GiB at 61.98 / 77.51 ms) and `nodes.csv`'s served-window `=1` row (8.22 / 7.38 GiB).
- **Untested estimate: cloning the cache slices** instead of keeping views would drop the held state to about 2.0 GiB. On paper that would put `=1` near 5.7 / 5.5 GiB. Compiled transients (3.5 / 2.8 GiB) are still higher than eager vLLM-Omni / SGLang best (1.7 / 1.9 GiB).
- **SGLang's `lossless` and plain-bf16 paths** hold twice the cache of its `extra-high` path (3.47 vs 1.76 GiB on Wan 2.1).

## How the data was obtained

- **Inputs.** `latents.pt` and `ref_fp32_outputs.pt` match the timing's sha256 on all three VMs (`40b622fc…` / `f31b53ac…` for Wan 2.1, `7e2d6ddb…` / `659d8ec5…` for Wan 2.2).
- **M* stock and fp32 full streams (re-decoded).** The PR's quality frames on `wan-bench-b200` had been deleted after analysis. I re-decoded `diffusers_bf16` and `ref_fp32` on that VM (idle GPU) with `quality.py decode`. Its `quality.py`, `bench_wan_decoder.py` and `common.py` are byte-identical to the PR worktree, and it runs the same venv (torch 2.12.1, diffusers 0.39.0) with fresh compile caches. Two checks: stock md5 equals the PR's on both VAEs, and stock-vs-fp32 per-frame PSNR matches the PR's `quality_frames.csv` within 0.001 dB on all 1,021 frames of both VAEs, which validates the fp32 reference.
- **vLLM-Omni.** No full streams had been saved, so I re-decoded on `wan-bench-b200-vllm` (GPU idle and checked before each run), one fresh container per decode. The image digest is the timing's (`sha256:d9d81b77…`). `harness/vo_frames.py` builds the decoder with the handoff harness's `build_streaming`, streams the same latent schedule, and converts with M*'s `to_uint8`. Each stream reproduces its timing JSON's uint8 PSNR: 54.91 / 54.88 / 55.27 / 55.16 dB. Stock diffusers bf16 in that container also equals M*'s stock md5 (control).
- **SGLang.** The full streams saved by the handoff's `md5_frames.py` were still on `wan-bench-b200-sglang` (`~/bench/results/md5/`), and their md5s match the handoff. No GPU decoding was needed for quality.
- **Analysis.** All metrics were computed on CPU locally, in the `mstar` venv (torch 2.12.1).
- **Memory probes.** One fresh process per setting, sequential, in tmux, after checking `nvidia-smi` and `tmux ls`. vLLM-Omni ran in its container, SGLang in its venv, and M* on `wan-bench-b200`. No GPU was shared.

## VM state

- **SGLang VM.** `wan-bench-b200-sglang` was **STOPPED** when I started; the Code2Wav run had finished and stopped it. Its IP is unchanged (<vm-ip>). I started it to read the saved streams and run the memory probe, and **left it running** as instructed. Stop it if nothing else needs it.
- **Other VMs.** `wan-bench-b200` is running and idle. On `wan-bench-b200-vllm`, the end-to-end Cosmos3 agent started its `e2e-cosmos3` tmux session at 19:30 UTC, and its first GPU process started at 19:35:10. My last probe there had already finished at 19:31:42, so the GPU was never shared.
- **Scratch files.** My frame files are left in `~/qc/` on `wan-bench-b200` (about 20 GB) and `wan-bench-b200-vllm` (about 7 GB), and locally in `${WAN_BENCH}/qc-frames/` (about 30 GB). Delete them if not needed.
- **Repo.** Nothing in `wt-wan-decoder` was edited, committed or pushed.

## Files

- `quality_compare.csv`: one row per engine and setting with all metrics, plus ms, peak GiB, latents per call, uint8 conversion and sources.
- `memory_compare.csv`: the peak-memory breakdown.
- `quality_compare_raw.csv` (every analyzed stream, including controls) and `quality_compare_frames.csv` (per-frame PSNR and SSIM).
- `runs/`: `mem_probe.jsonl`, per-stream metadata JSONs (md5, environment) and analysis logs.
- `harness/`:
  - `qc_analyze.py`: metrics via `quality.py`.
  - `vo_frames.py` / `vo_frames.sh`: vLLM-Omni full-stream decode.
  - `stock_fp32conv.py`: the uint8-conversion check.
  - `mem_probe.py` + `mem_{vo,sg,mstar}.sh`: memory probes.
  - `make_outputs.py`: builds both CSVs and these tables.
  - `build_table.py`: written by another agent in this directory during my run. I did not touch it. It predates the peak-memory column, and running it overwrites `quality_compare.csv` without that column; `make_outputs.py` supersedes it.
