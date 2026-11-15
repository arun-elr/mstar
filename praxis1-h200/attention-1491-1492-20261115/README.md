# Synthetic H200 attention wrapper timings for #1491 and #1492

These plots use the original per-sample JSON from the H200 runs at the exact revisions listed in `provenance.json`. Each report contains 20 warm GPU samples after five warmups for eager resource-wrapper calls and captured replay. The plotted values are their medians. These are synthetic BF16 QKV resource timings, not a checkpoint run, whole-engine capture, or served latency. The eager wrapper and captured replay paths do different work, so their ratio is not an end-to-end speedup.

Recreate the plots with `uv run --no-project --with matplotlib --with numpy python plot.py` in this directory. The script verifies each raw file hash, measured revision and median before drawing. To rerun the H200 measurement itself, use the pinned scripts and exact commands in [`attention-bench-20261115/README.md`](../attention-bench-20261115/README.md).

- `pr1491-raw.json` and `pr1492-raw.json`: original per-sample reports.
- `pr1491-attention-wrapper.png` and `pr1492-attention-wrapper.png`: PR-specific plots.
- `provenance.json`: input hashes, source revisions and run environment.
