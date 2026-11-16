# Wan 2.2 dense FP8 check on H200, 2026-11-16

Source: mstar PR #1540 head `1546535ace4eb7c4f1b57bf89dc3df4d94b2e0d3`. Official checkpoint: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` at Hub commit `b8fff7315c768468a5333511427288870b2e9635`. GPU: NVIDIA H200. Runtime: torch 2.12.1+cu132, Diffusers 0.43.0. This directory contains the raw outputs and scripts behind the charts.

## Result

The BF16 and dense FP8 DiT forward medians were about 203 ms and 103 ms over three seeds (six warmed samples per arm and seed). This is a DiT forward measurement, not an end-to-end serving speedup. Real-checkpoint native BF16 equivalence passed four tests; the published checkpoint loaded with all 300 selected linears quantized.

Served 832x480, 33-frame, 20-step videos have large visual differences. Mean frame PSNR and SSIM against BF16 were 16.01 dB / 0.575 (seed 41), 13.61 / 0.555 (seed 42), and 14.17 / 0.586 (seed 43). BF16 repeated seed 41 decoded identically; FP8 repeated seed 41 reached 38.70 dB / 0.968, so repeat variability is much smaller than the cross-arm difference. Human visual review is still needed.

A test-only selector patch kept the first and last DiT blocks in BF16, leaving 280 selected linears in FP8. One served seed 41 still measured 14.91 dB / 0.518 against BF16, below the full FP8 result. The patch was not committed to the PR and is saved in `raw/pr1540-ablation.patch`.

- [Frame quality chart](wan22-quality-per-frame.png) and [SVG](wan22-quality-per-frame.svg)
- [DiT forward latency chart](wan22-dit-forward-latency.png) and [SVG](wan22-dit-forward-latency.svg)
- [Per-frame metrics, file hashes, and source/checkpoint pins](raw/pr1540-quality.json)
- [Forward timings and tensor errors](raw/pr1540-forward.log)
- [BF16 and FP8 clips](videos/), [server profiles and branch logs](raw/)

The served timings were not gathered in a matched A/B protocol: the first BF16 request included a one-time JIT build. No served speedup or throughput claim follows from this run. LTX-2.5, TP=32, and a complete video quality sweep were not run.

`raw/pr1540-quality.json` uses the original VM file paths. The corresponding files are named `videos/{bf16,fp8,ablate}-<basename>` here. The scripts under `scripts/` use those original paths and are retained for reproduction on the VM layout.
