# Cosmos3-Edge served end to end on one B200

Copied from the e2e handoff. Start with [`SUMMARY.md`](SUMMARY.md). It ran on VM `wan-bench-b200-vllm` (1× B200):
torch 2.12.1+cu130, diffusers 0.39.0, flashinfer-python 0.7.1, cuDNN 9.20. Variants: `before` = upstream
main `99f5108b`, `after0` / `after1` = PR `60fbc5cd` at `windowed_decode_latents_per_call=0` / `1`.
Checkpoint: `nvidia/Cosmos3-Edge@344d602b`. Date: 2026-10-13.

- `results.csv` / `results.json`: 1 stream (3 kv + 2 chained timed rollouts) and 4 streams (2 rounds).
  `results_c8.*`: 8 streams (1 round). `reference.json`: the fp32 whole-stream reference metrics.
- `logs/runs/`, `logs/runs-c8/`, `logs/runs-latents/`: per-variant server and client logs, `stats.txt`,
  memory snapshots, `smi.csv` GPU traces. `logs/probe/`: the decoder-only cache probe.
- `logs/attempts/`: two failed environment attempts (the fa3 sm90-only wheel; a missing ninja), kept for the record.
- `scripts/`: the scripts as run.

Not committed: the code snapshots each variant ran from (`git archive` tarballs in `${WAN_BENCH}/e2e-cosmos3-stage/`)

| file | sha256 |
|---|---|
| `mstar-main-99f5108b.tar.gz` | `1645fb92a1a90c24c61015d45e7d6be5dd60559556a05133b67e05019a9eceab` |
| `mstar-pr-60fbc5cd.tar.gz` | `955f9c7359f73d7f9db7cef0c6e7cd9b567a0ac0f9d257c0b2c4bd0def9d7425` |

The staged log tarballs there (`e2e-logs.tgz`, `e2e-c8.tgz`) hold the same files as `logs/`. The checkpoint
and the captured latents stay in the VM's `~/e2e-cosmos3/`.
