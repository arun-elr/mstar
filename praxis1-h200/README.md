# Praxis DROID H200 checks at `5902054c0f`

These figures come from one NVIDIA H200 (driver 580.173.02) with the published DROID checkpoint. M* ran at `5902054c0f` with `-O0` and no graph capture. The reference checkout was `68f4d7a254b4bf41951d2e97f44b07f6cac2bf69` with `compile=None`. Both used two Euler steps, the same synthetic 46-frame trajectory, and matching observation hashes. The warm benchmark script source SHA-256 was `65370861d3545910e63275e5b2d4bb63fdb875c958157e83a6c1efdacf75a06c`. This published copy changes only illustrative paths in its opening docstring; `source-manifest.json` records both hashes.

| Call | Reference p50 | M* Python p50 | M* Rust p50 |
|---|---:|---:|---:|
| Cold, 21 rows | 116.75 ms | 268.04 ms | 269.96 ms |
| Warm 1, 24 rows | 253.74 ms | 535.26 ms | 541.45 ms |
| Warm 2, 24 rows | 255.05 ms | 542.96 ms | 547.87 ms |

Each column has four measured sessions after one warmup. [The latency chart](latency-h200-5902054c0f.png) uses the summaries in `reference-matched-warm-590-script.json`, `mstar-590-warm-python.json`, and `mstar-590-warm-rust.json`. The same run produced 21, 24, then 24 command rows. Separate cold openpi reports are also included for Python and Rust; they are not the plotted cold session measurements.

[The stage chart](stages-h200-5902054c0f.png) uses `mstar-590-stage-bench.jsonl`. It is a diagnostic run that synchronizes CUDA around each submodule, so its totals should not be compared directly with the request latencies above. Backbone and action expert each include two Euler calls. `plot.py` regenerates both PNGs from the included summary reports and stage rows with NumPy and Matplotlib.

The single seven-task simulation guard in `mstar-590-sim-summary.json` completed 6/7 episodes; `lift_cube` missed. One episode per task is not a parity estimate. The serving gate remained off in the committed source; these checks used a disposable serving copy.

## Reproduce the measured requests

The benchmark client used for the three-call session measurements is [`mstar-b4-warm-benchmark.py`](mstar-b4-warm-benchmark.py), with only its example paths changed for publication. It constructs the 46-frame DROID input, applies the same prompt (`Pick up the red cube.`), and checks 21/24/24 command rows. These were the client invocations, after starting each server separately; only the output path is shortened here:

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

The CPU selection rerun in the `5902054c0f` worktree used the following command, with `MSTAR_VENV` and `PRAXIS_REF` set to the local virtual environment and reference checkout:

```sh
source "$MSTAR_VENV/bin/activate"
PYTHONPATH="$PWD:$PRAXIS_REF" OMP_NUM_THREADS=2 \
  pytest -q test/modular/test_praxis1_*.py test/modular/test_positronic_input.py \
  -k 'not test_droid_session_config_is_sized_and_gated'
```

It passed 527 tests, skipped 2 optional stock-client tests, and deselected the named deployment test because the shared Rust extension was stale. This selection does not run the Rust frontend tests.

## Reference ablations

[`reference-nocache-590-script.json`](reference-nocache-590-script.json), [`reference-unpaired-590-script.json`](reference-unpaired-590-script.json), and [`reference-nofusednorm-590-script.json`](reference-nofusednorm-590-script.json) retain the summary values from three diagnostic reference runs. Each used the same benchmark client, inputs, session count, server flags, and H200 as the matched baseline. Disabling the context cache changed warm p50 latency from 253.74/255.05 ms to 263.68/267.04 ms; disabling paired guidance changed it to 267.40/269.81 ms. Disabling fused norm changed it to 275.90/277.14 ms (8.7% slower than baseline on each warm call), with all 15 command hashes and numeric command rows identical to baseline. The [fused-norm patch](reference-fused-norm-off.patch) shows that one-variable change; its file paths were sanitized, and `source-manifest.json` records the original patch hash. The ablations used disposable reference source copies. Exact source patches for the cache and guidance ablations are not included yet, so their summaries document measured results but do not fully reproduce those source modifications.

## Attention resource measurements

The [attention chart](attention-resource-h200-5902054c0f.png) and [compact report](mstar-590-attention-resource-bench.json) measure five current M* attention wrappers on one H200, with `MSTAR_FLEX_BACKEND=TRITON`. Inputs were synthetic BF16 Q/K/V shaped for DROID; the conditional and blank text lengths (6 and 1 tokens) were measured from the Qwen3.5-2B processor for the benchmark prompt. The [benchmark script](mstar-praxis-attention-h200-bench.py) used 20 warmups and 100 measured samples per case. The measured run used an [archive variant](mstar-praxis-attention-h200-bench-archive.py) that substitutes the verified source revision for a Git query because the GPU source archive had no `.git`; its measurement logic is unchanged. Both source hashes are in `source-manifest.json`. The [plot script](plot_attention_resource.py) regenerates the chart from the compact report.

| Warm resource | Eager wrapper p50 | Direct same-mask p50 | Captured replay p50 |
|---|---:|---:|---:|
| Encoder conditional | 0.505 ms | 0.501 ms | 0.415 ms |
| Encoder blank | 0.517 ms | 0.512 ms | 0.420 ms |
| Decoder | 2.345 ms | 2.338 ms | 2.093 ms |
| Expert cross | 0.136 ms | 0.124 ms | 0.678 ms |
| Expert self | 0.130 ms | 0.120 ms | 0.013 ms |

All values are median CUDA-event timings. The direct path uses the wrapper's already-built `BlockMask`. Captured replay uses a synthetic fixed warm-size bucket and excludes plan and host-to-device transfer work. Its padded expert-cross case is slower than eager, so these numbers do not establish a production performance gain. Eager wrapper and direct same-mask outputs were identical; captured output differed by at most 0.000977 in BF16 and all padding rows were zero. This is a resource-level parity check, not model or served-session parity. The compact report omits raw samples and local environment paths but keeps their source SHA-256.

To run the resource benchmark from an M* checkout at `5902054c0f` with CUDA and the reference processor available, set `MSTAR_CHECKOUT` to that checkout:

```sh
MSTAR_FLEX_BACKEND=TRITON python mstar-praxis-attention-h200-bench.py \
  --repo "$MSTAR_CHECKOUT" --text-tokens 6 --blank-tokens 1 \
  --text-source 'Qwen3.5-2B AutoProcessor on Pick up the red cube. / blank' \
  --samples 100 --warmups 20 --output attention-resource-report.json
```

## Reference checkout provenance

The matched reference run used base commit `68f4d7a254b4bf41951d2e97f44b07f6cac2bf69` plus the [tracked checkout diff](reference-working-tree.diff) and four [untracked simulation/benchmark source files](reference-untracked/). The tracked diff changes the simulator evaluator and its documentation; it does not change `praxis` model files. The untracked files are provided to make that dirty checkout inspectable. Virtual environments and unrelated untracked files were excluded. `source-manifest.json` records the original SHA-256 of each source artifact.

The first H200 serving/ablation session used 0.54 GPU-hours; the separate attention-resource session used 0.23 GPU-hours, for 0.77 GPU-hours total. These are GPU usage durations, not a cost estimate.

The JSON files keep the original summaries and run identifiers but omit per-call numeric action arrays, local checkpoint paths, GPU UUID, and process IDs. `source-manifest.json` gives SHA-256 hashes of the full source reports and charts. The reference checkout and benchmark client were dirty during the matched run, so their commit IDs alone cannot reproduce it bit for bit. The served numbers describe the eager bring-up setup, not tuned throughput or full-request graph capture parity.
