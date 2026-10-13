# In-tree decoder runs on one B200 (ladder, variance, profile, quality, `latents_per_call` sweep)

Every raw run behind the per-change tables, the output-equivalence tables, the kernel profiles
and the `windowed_decode_latents_per_call` sweep. Run on VM `wan-bench-b200` (1× B200, torch
2.12.1+cu132, CUDA 13.2, cuDNN 9.20.0, diffusers 0.39.0, Python 3.12.3) on 2026-10-13. The
summary CSVs built from these runs are already on this branch in
[`benchmark/wan_vae_decoder/results/b200/`](../../../benchmark/wan_vae_decoder/results/b200/)
(`RESULTS.md`, `runs.csv`, `variance.csv`, `profile.csv`, `quality.csv`, `lpc.csv`, ...).

## Layout

| path | what | runs | `env.mstar_commit` |
|---|---|---|---|
| `runs/wan21/`, `runs/wan22/` | headline ladder (`run_all.sh`): fp32 ref, stock diffusers, M* before (`mstar_ar_ctx1..8`), prototype `channels_last*`, `one_call*`, `px_*` | 1 per variant (18 + 16) | `99f5108b` |
| `runs/ablation/ladder{,_wan22}/run{1,2,3}/` | per-change ladder `mstar_step0..5` (+ `_eager`), `run_ablation.sh ladder` | 3 × 10 per VAE | `4b8b4789` |
| `runs/ablation/variance/run{1,2,3}/` | prototype-harness variance (diffusers, compile, M* before, `px_nopad_bench`, ...) | 3 × 9 | `99f5108b` |
| `runs/ablation/intree{,_wan22}/run{1,2,3}/` | in-tree node `mstar_warm` / `_nobench` / `_eager` vs `px_nopad_bench` | 3 × 4 per VAE | `4b8b4789` |
| `runs/ablation/lpc{,_wan22}/run{1,2}/` | `windowed_decode_latents_per_call` = 0, 1, 2, 3 | 2 × 4 per VAE | `60fbc5cd` |
| `runs/ablation/nodes/` | decoder node alone, served window (8 latents, 6 new), `lpc0..3` | 20 | `60fbc5cd` / `4b8b4789` |
| `summary/nodes.csv` | `nodes.csv` incl. the `lpc` rows (newer than the copy on this branch, which lacks them) | | |
| `runs/ablation/profile{,_wan22}/`, `profile_ladder{,_wan22}/`, `profile_intree{,_wan22}/` | `torch.profiler` kernel tables (`*.kernels.json`) | 1 per variant | |
| `runs/ablation/profile_traces.tar.zst` | the 4 raw Chrome traces of `profile_intree*` (8.5 MB raw); `profile_traces.sha256` lists the files inside | | |
| `runs/ablation/quality/`, `quality_results/` | output-equivalence metas and results (`quality.csv`, `quality_frames.csv`, `coverage.csv`) | 1 decode per variant, 2nd process for `*.run2` | |
| `runs/ablation/quality_recon/`, `quality_recon_results/` | encode + decode vs the source clip | | |
| `runs/ablation/{ladder,lpc}_frames*` | node-input metas and per-frame CSVs for the ladder/lpc output column | | |
| `runs/ablation/px/`, `tf32/`, `control/`, `wan22/` | prototype splits (`px_*`), TF32 check on the fp32 reference, controls | 1 each (control: 3 × 2) | |
| `runs/ablation/ablation.log` | full `run_ablation.sh` log | | |
| `vm-home/` | the VM's queue scripts and logs (`queue_*.sh/.log`), env setup (`vm_setup.sh`, `vm_env.sh`, `vm_freeze.txt`), staged harness copies (`stage*/`) | | |

Sources: `benchmark/wan_vae_decoder/runs/b200/` of the PR worktree, plus the files that existed only on
the VM in `~/mstar-pr/benchmark/wan_vae_decoder/runs/b200/ablation/` (traces, quality metas, frame
CSVs, `ablation.log`). Where both had a file they were byte-identical.

## Not committed (left on the VM)

| what | VM location | size | sha256 |
|---|---|---|---|
| Wan 2.1 latents | `~/wan-bench/results/b200-wan21/wan21/latents.pt` | 102,237,929 B | `40b622fcb24d36be0bc821254cdde77418ac92ea8938c53e3753f1db9f8ed7d4` |
| Wan 2.1 fp32 reference frames | `~/wan-bench/results/b200-wan21/wan21/ref_fp32_outputs.pt` | 575,082,581 B | `f31b53acbe8e77fa4c41a450670d3ea111d5dcf064b4d00b6225c56da099338b` |
| Wan 2.2 latents | `~/wan-bench/results/b200-wan22/wan22/latents.pt` | 76,678,889 B | `7e2d6ddbeb8846c48213b5c7886f07fde5ab3a4d7936b49f92996fe3748ac74a` |
| Wan 2.2 fp32 reference frames | `~/wan-bench/results/b200-wan22/wan22/ref_fp32_outputs.pt` | 575,082,581 B | `659d8ec50f0c7502ef6e4dde69f61103a880daaadb786d850fd4d982af651b93` |
| Inductor / Triton compile caches | `~/mstar-pr/benchmark/wan_vae_decoder/runs/b200/ablation/cache/` | 12 GB, ~327k files | not hashed |

`~/mstar-pr/benchmark/wan_vae_decoder/runs/b200/wan21|wan22/` on the VM are symlinks to the four `.pt` files above.

Paths in logs and JSONs are rewritten: `${VM_HOME}` is the VM user's home, `${MSTAR}` / `${MSTAR_PR}` local
checkouts, `${WAN_BENCH}` the local work dir, `${TMPDIR}` the temp dir, `<vm-ip>` the VM address.
