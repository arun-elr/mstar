# H200 cross-check

Copied from the H200 handoff. Start with [`SUMMARY.md`](SUMMARY.md). It is the same decoder benchmark, inputs and weights as
the B200, on one local H200: torch 2.12.1+cu132, CUDA 13.2, cuDNN 9.20.0, diffusers 0.39.0. PR code at
`c56b572b`. Date: 2026-10-13. The tables were already on this branch in
[`benchmark/wan_vae_decoder/results/h200/`](../../benchmark/wan_vae_decoder/results/h200/); this folder adds
the raw runs and harness.

- `runs.csv`, `variance.csv`, `profile.csv`, `comparison_b200.csv`, `coverage.csv`, `quality_check.csv`, `env.txt`.
- `runs/`: `intree{,_wan22}/run{1,2,3}/` (3 runs × 4 variants), `ladder*/`, `drift/`, `refcheck_*`, `profile*/`, `quality_meta/`,
  `quality_results/`, `job.log`, `gpu_trace_followup.csv` and the job start/end epoch stamps.
- `harness/`: the job scripts as run.
- `memory-census-notes.md`: the memory-fix H200 tensor census and candidate-fix prototype (from the memory-fix handoff).
- `harness-smoke/`: the harness's first smoke runs, on this H200 host before the B200 VM existed (tiny inputs; `mstar_commit` `df0f2b13`).

Not committed: the code snapshot the runs were imported from, `${WAN_BENCH}/mstar_c56b572b.tar.gz`
(sha256 `8ccb1c3d389e8dc31c49c8ae8d194980aee6a1ef666dff41b346c17f1fb93206`). Also not committed: the smoke inputs
in `${WAN_BENCH}/smoke/`: `lat.pt` `606a1561…0ee1`, `lat22.pt` `ba9b5713…dbbc`, `lat_video.pt` `ce4ccfd0…d1a6`, `ref.pt` `945773d1…2948`.
The latents and references are the B200 files (see `../b200/in-tree/README.md`).
