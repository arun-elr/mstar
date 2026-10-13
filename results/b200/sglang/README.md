# SGLang Wan VAE streaming decode on one B200

Copied from the SGLang handoff. Start with [`SUMMARY.md`](SUMMARY.md). It ran on VM `wan-bench-b200-sglang`
(1× B200): SGLang v0.5.21 (`e00930c5`, which includes sgl-project/sglang#34125), torch 2.13.0+cu130,
cuDNN 9.20.0, diffusers 0.39.0 (`env/sgl_freeze.txt`). Date: 2026-10-13.

- `results/`: `runs.csv`, `variance.csv`, `profile.csv`, `frame_md5.csv`.
- `runs/wan2{1,2}/run{1,2,3}/`: 3 fresh-process runs × 4 variants per VAE. `runs/screen_*`: the settings screen. `runs/profile_*`: kernel tables. `runs/frame_md5/`: chunked vs whole-clip md5s.
- `flops/`: FLOP counts per chunk. `harness/`: the scripts as run.
- `env/`: VM setup and run logs (`vm_setup.*`, `bench_run.log`, `queue2.*`, `md5*`), `gpu_trace.csv`, `inputs_sha256.txt`. `env/vm-create/`: VM provisioning, with IDs, key and address replaced by placeholders.

Not committed: the inputs at `${WAN_BENCH}/sglang/inputs/` (the same latents and fp32 references as
`../in-tree/README.md` lists; `env/inputs_sha256.txt` matches them). The local SGLang source excerpts
in `${WAN_BENCH}/sglang/src/` are third-party upstream code, read for the analysis.
