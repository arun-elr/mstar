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
- `scripts/`: the scripts as run, since parameterized by `E2E_DIR` and with a `default` variant (no
  overrides: the PR checkout's own defaults) and `PHASES=c1` (one stream, kv and chained only).

## Reproduce the served numbers

On one GPU with the M* environment (flashinfer, as `pip install -e ".[cosmos3]"`):

```bash
E2E_DIR=~/e2e-cosmos3; mkdir -p $E2E_DIR && cd $E2E_DIR
ln -s /path/to/mstar/.venv venv                      # the M* environment
ln -s ~/.cache/huggingface hf                        # HF_HOME with nvidia/Cosmos3-Edge@344d602b
git -C /path/to/mstar worktree add $E2E_DIR/mstar-main 99f5108b && echo 99f5108b > mstar-main/REV
git -C /path/to/mstar worktree add $E2E_DIR/mstar-pr perf/wan-vae-decode && git -C mstar-pr rev-parse HEAD > mstar-pr/REV
cp -r /path/to/this/folder/scripts scripts
E2E_DIR=$E2E_DIR bash scripts/run_all.sh before default        # all phases: 1 and 4 streams, kv and chained
E2E_DIR=$E2E_DIR PHASES=c8 bash scripts/run_variant.sh default # 8 streams
python scripts/analyze.py $E2E_DIR/runs out/                   # results.csv
```

`run_variant.sh` overlays a variant's decoder switches into `configs/cosmos3_edge.yaml` (`before`: none,
on main; `after0` / `after1`: the switches of these runs; `default`: none, on the PR), serves it, and runs
`stream_client.py` against it. `hook/sitecustomize.py` records peak memory per process.

Not committed: the code snapshots each variant ran from (`git archive` tarballs in `${WAN_BENCH}/e2e-cosmos3-stage/`)

| file | sha256 |
|---|---|
| `mstar-main-99f5108b.tar.gz` | `1645fb92a1a90c24c61015d45e7d6be5dd60559556a05133b67e05019a9eceab` |
| `mstar-pr-60fbc5cd.tar.gz` | `955f9c7359f73d7f9db7cef0c6e7cd9b567a0ac0f9d257c0b2c4bd0def9d7425` |

The staged log tarballs there (`e2e-logs.tgz`, `e2e-c8.tgz`) hold the same files as `logs/`. The checkpoint
and the captured latents stay in the VM's `~/e2e-cosmos3/`.
