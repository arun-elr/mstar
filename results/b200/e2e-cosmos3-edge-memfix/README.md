# Cosmos3-Edge served end to end with the compact decoder cache (B200)

Copied from the e2e re-run handoff; start with [`SUMMARY.md`](SUMMARY.md). Same VM, venv, checkpoint, request
and scripts as [`../e2e-cosmos3-edge`](../e2e-cosmos3-edge/) (`wan-bench-b200-vllm`, torch 2.12.1+cu130,
flashinfer 0.7.1, cuDNN 9.20), 2026-10-13. Variants: `after1fix` = PR `913295e4` with no overrides
(`=1`, `cudnn.benchmark` off, compact cache), `after0fix` = the same with `windowed_decode_latents_per_call: 0`;
`before` / `after0` / `after1` are the earlier session's. `results_c8_recheck_oldtree.*` re-runs the old PR
tree `60fbc5cd` at 8 streams, to attribute the DiT `AssertionError` seen at 8 streams (it fails the same way).

Not committed: frames, mp4s and latents, on the VM under `~/e2e-cosmos3/runs-fix*/` and `runs-c8-*/`.
