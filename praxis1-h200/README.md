# Praxis DROID H200 checks at `5902054c0f`

These figures come from one NVIDIA H200 (driver 580.173.02) with the published DROID checkpoint. M* ran at `5902054c0f` with `-O0` and no graph capture. The reference checkout was `68f4d7a254b4bf41951d2e97f44b07f6cac2bf69` with `compile=None`. Both used two Euler steps, the same synthetic 46-frame trajectory, and matching observation hashes. The warm benchmark script SHA-256 was `65370861d3545910e63275e5b2d4bb63fdb875c958157e83a6c1efdacf75a06c`.

| Call | Reference p50 | M* Python p50 | M* Rust p50 |
|---|---:|---:|---:|
| Cold, 21 rows | 116.75 ms | 268.04 ms | 269.96 ms |
| Warm 1, 24 rows | 253.74 ms | 535.26 ms | 541.45 ms |
| Warm 2, 24 rows | 255.05 ms | 542.96 ms | 547.87 ms |

Each column has four measured sessions after one warmup. [The latency chart](latency-h200-5902054c0f.png) uses the summaries in `reference-matched-warm-590-script.json`, `mstar-590-warm-python.json`, and `mstar-590-warm-rust.json`. The same run produced 21, 24, then 24 command rows. Separate cold openpi reports are also included for Python and Rust; they are not the plotted cold session measurements.

[The stage chart](stages-h200-5902054c0f.png) uses `mstar-590-stage-bench.jsonl`. It is a diagnostic run that synchronizes CUDA around each submodule, so its totals should not be compared directly with the request latencies above. Backbone and action expert each include two Euler calls. `plot.py` regenerates both PNGs from the included summary reports and stage rows with NumPy and Matplotlib.

The single seven-task simulation guard in `mstar-590-sim-summary.json` completed 6/7 episodes; `lift_cube` missed. One episode per task is not a parity estimate. The serving gate remained off in the committed source; these checks used a disposable serving copy.

## Reproduce the measured requests

The exact benchmark client used for the three-call session measurements is [`mstar-b4-warm-benchmark.py`](mstar-b4-warm-benchmark.py), SHA-256 `65370861d3545910e63275e5b2d4bb63fdb875c958157e83a6c1efdacf75a06c`. It constructs the 46-frame DROID input, applies the same prompt (`Pick up the red cube.`), and checks 21/24/24 command rows. These were the client invocations, after starting each server separately; only the output path is shortened here:

```sh
# Reference server, port 8133; the same invocation was used for both ablations.
python mstar-b4-warm-benchmark.py --mode reference \
  --url ws://127.0.0.1:8133/api/v1/session \
  --sessions 5 --warmup-sessions 1 --out reference-matched-warm-590-script.json

# M* Python and Rust server runs used the same flags at their respective ports.
python mstar-b4-warm-benchmark.py --mode mstar \
  --url ws://127.0.0.1:${SERVER_PORT}/api/v1/session \
  --expect-code-version 5902054c0f --sessions 5 --warmup-sessions 1 \
  --out mstar-590-warm-python.json
```

The reference server used `configs/serve_droid.yaml` with `context_mode=observed`, `max_context_chunks=1`, `exec_horizon=24`, `control_mode=droid_impedance`, `skip_video_decoder=on`, and Triton encoder/decoder attention. The checkpoint, VAE, and text-encoder paths were local to the GPU host. M* used `configs/praxis1_droid.yaml`, `-O0`, no graph capture, two Euler steps, and the same DROID weights. The JSON reports retain input hashes, run counts, and the recipe metadata. The Python and Rust output files above name separate runs, not two outputs from one server.

The CPU selection rerun in the `5902054c0f` worktree used:

```sh
source /home/aruneli/oss/mstar/.venv/bin/activate
PYTHONPATH=$PWD:/home/aruneli/oss/praxis-1-oss OMP_NUM_THREADS=2 \
  pytest -q test/modular/test_praxis1_*.py test/modular/test_positronic_input.py \
  -k 'not test_droid_session_config_is_sized_and_gated'
```

It passed 527 tests, skipped 2 optional stock-client tests, and deselected the named deployment test because the shared Rust extension was stale. This selection does not run the Rust frontend tests.

## Reference ablations

[`reference-nocache-590-script.json`](reference-nocache-590-script.json), [`reference-unpaired-590-script.json`](reference-unpaired-590-script.json), and [`reference-nofusednorm-590-script.json`](reference-nofusednorm-590-script.json) retain the summary values from three diagnostic reference runs. Each used the same benchmark client, inputs, session count, server flags, and H200 as the matched baseline. Disabling the context cache changed warm p50 latency from 253.74/255.05 ms to 263.68/267.04 ms; disabling paired guidance changed it to 267.40/269.81 ms. Disabling fused norm changed it to 275.90/277.14 ms (8.7% slower than baseline on each warm call), with all 15 command hashes and numeric command rows identical to baseline. The [fused-norm patch](reference-fused-norm-off.patch) shows that one-variable change; its file paths were sanitized, and `source-manifest.json` records the original patch hash. The ablations used disposable reference source copies. Exact source patches for the cache and guidance ablations are not included yet, so their summaries document measured results but do not fully reproduce those source modifications.

The JSON files keep the original summaries and run identifiers but omit per-call numeric action arrays, local checkpoint paths, GPU UUID, and process IDs. `source-manifest.json` gives SHA-256 hashes of the full source reports and charts. The reference checkout and benchmark client were dirty during the matched run, so their commit IDs alone cannot reproduce it bit for bit. These numbers describe the eager bring-up setup, not tuned serving throughput or GPU capture parity.
