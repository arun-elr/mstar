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

The JSON files keep the original summaries and run identifiers but omit per-call numeric action arrays, local checkpoint paths, GPU UUID, and process IDs. `source-manifest.json` gives SHA-256 hashes of the full source reports and charts. The reference checkout and benchmark client were dirty during the matched run, so their commit IDs alone cannot reproduce it bit for bit. These numbers describe the eager bring-up setup, not tuned serving throughput or GPU capture parity.
