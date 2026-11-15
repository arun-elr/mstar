# Qwen3.5-2B combined-stack A/B, 15 November 2026

This compares pinned main merge-base `f9073827da6111ba76c085b2a715e53cd0ef51c2`
with the full Praxis/episode/attention stack at `532e656636eb4eaa64d9fcd61e98d722a38d4449`.
It is a generic Qwen regression check, not a Praxis model latency test or a
measurement of any one PR. The newer upstream main commits are outside this
comparison.

The six fresh server processes ran on one H200 in base / stack / stack / base /
base / stack order. Both arms used identical Qwen3.5-2B config and checkpoint
files, source-matched Rust extension and server, `-O0`, and nonzero LLM graph
capture. Each process used eight warmups and 50 measured requests at
concurrency 1, 4 and 16, with 32 streamed output chunks per request. Four
short prompts repeat with prefix caching. The first interrupted attempt was
excluded; every number here comes from the completed six-run retry on one
boot, after which the VM was stopped.

| Concurrency | Base throughput | Stack throughput | Base p50 total | Stack p50 total |
|---|---:|---:|---:|---:|
| 1 | 510.7 | 510.8 chunks/s | 62.57 ms | 62.50 ms |
| 4 | 1,602.5 | 1,597.5 chunks/s | 77.26 ms | 77.38 ms |
| 16 | 4,106.0 | 4,034.6 chunks/s | 105.61 ms | 107.75 ms |

Each cell is the median of three fresh-process results. At concurrency 16,
the observed median change is 1.74% lower throughput and 2.03% higher p50
latency on the stack. The three-run ranges overlap, so this small repeated
prompt sample does not establish a general regression. Hash sets of generated
outputs match across arms for each prompt and concurrency, but prompt 0 has
two hashes within each arm at concurrency 4 and 16. Per-request outputs were
not identical.

`source/` has all six per-request JSONs, the benchmark program, configuration,
model-file hashes, run order, source-match logs, capture summary and derived
aggregate. `provenance.json` records the raw file SHA-256 values and per-run
statistics. Run `python plot.py` with `rsvg-convert` available to regenerate
the SVG and PNG chart. `artifact-manifest.sha256` checks every committed file
except itself.
