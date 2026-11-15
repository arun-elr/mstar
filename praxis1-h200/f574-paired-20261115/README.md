# Praxis DROID paired guard, 15 November 2026

This is a fresh, sequential M* and reference run on one H200. M* used source commit
`f574af84969a3e5191b5e781cd30e12bd029165f`; the reference used
`68f4d7a254b4bf41951d2e97f44b07f6cac2bf69` with a pinned dirty simulator
checkout. The source archive, source-matched Rust server and extension, model
files, configurations and runner inputs were checked before the run. The M*
serving gate was enabled only in a disposable copy. It remains disabled in the
PR source.

| Check | M* | Reference |
|---|---:|---:|
| DROID scenes passed | 54/56 | 55/56 |
| Cold session p50 | 271.65 ms | 137.83 ms |
| First warm call p50 | 546.25 ms | 263.12 ms |
| Second warm call p50 | 542.32 ms | 266.94 ms |

The simulation used seven tasks, eight scene seeds per task and sampler seed 0.
M* missed `lift_cube` seed 6 and `block_out_of_bowl` seed 1. The reference
missed `stack_block` seed 0. The matched session probe sent identical synthetic
observation bytes to each server and recorded four sessions after one warmup.
It compared 4,485 actuator values: overall maximum absolute difference
0.0911, 95th percentile 0.0542. The reference also returns `timestamp`
metadata, which was excluded from the actuator comparison.

The latency comparison uses bring-up settings: M* used `-O0` and the reference
used `compile=None` with action CUDA graphs disabled. The simulation and session
arms ran sequentially on the same VM. DROID and Qwen files were hash-verified;
the reference `.pth` VAE and M* Diffusers VAE were separately hash-verified,
without an equivalence claim. Scene outcomes and action differences do not
establish internal numerical parity or production readiness. The later
`532e656636` integration head was not measured by this run.

`source/` contains the paired summaries, individual episode records, matched
session records, configuration files, input hashes and the exact status/diff
of the reference simulator checkout. `provenance.json` gives SHA-256 for each
source file and the chart values. Run `python plot.py` with `rsvg-convert`
available to regenerate the SVG and PNG charts. `artifact-manifest.sha256`
checks all committed files except itself.
