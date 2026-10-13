# Initial exploration (prototype harness, batch 1 and batch 2) on one B200

The first B200 session, before the in-tree work: stock diffusers, M* before at context 1–8, the prototype
`rx_*` / `px_*` variants (channels-last, one call per chunk, conv-internal padding, CUDA graphs, autotune,
fp16), variance, kernel profiles and the TF32 check. Run on VM `wan-bench-b200` (torch 2.12.1+cu132,
cuDNN 9.20.0, diffusers 0.39.0) on 2026-10-13. The JSONs record `mstar_commit` `99f5108b`, which was
upstream main, imported for "M* before".

- `branch-c7d2c28a/`: `benchmark/wan_vae_decoder/` from private branch `experiment/wan-vae-decoder` @ `c7d2c28a`
  (exported with `git archive`): the scripts, `PLAN.md`, `RESULTS-B200.md`, and `results-b200/`
  (`b200-wan21`, `b200-wan22` and `b200-batch2` runs, logs, GPU traces and `vm-setup/`).
- `scratch-extra/profile_breakdown.py`: the one script in the local scratch copy that isn't on the branch. The
  scratch copy's `results-b200/` is byte-identical to the branch's.
- `vm-wan-bench-variants/`: the VM's `~/wan-bench` versions of `profile_kernels.py` and `summarize_batch2.py`,
  which differ from the branch's. The VM's `~/wan-bench/results/` result files are otherwise identical to the branch's.

These runs compared against an external public write-up on Wan VAE decoding. Its figures are quoted
in a few summaries as "external reference" / "Ext. ref". The write-up itself is not archived here.

Not committed:

| what | location | sha256 |
|---|---|---|
| input clip `assets/draw.mp4` (7,894,483 B; the 4K hand-drawing clip the latents are encoded from) | local scratch `assets/`, VM `~/wan-bench/assets/` | `ee2ad048d075fc5fc61e8a2a4e75dbb41e093662f6236bbc6259f1f606d05eda` |
| one decoded frame, `draw_frame.png` | `${WAN_BENCH}/` | `12e21a0e5a2fa2366267cc820cf53deb79838af4c32a0b9a3ee3bd7ca63c4d25` |
| latents and fp32 references | VM `~/wan-bench/results/b200-wan2{1,2}/wan2{1,2}/*.pt` | see `../in-tree/README.md` |
| compile caches | VM `~/wan-bench/results/b200-batch2/cache/` (9.0 GB), `b200-wan2{1,2}/inductor-cache/` (1.1 GB, 725 MB) | not hashed |
