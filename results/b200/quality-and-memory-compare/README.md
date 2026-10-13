# Output quality and peak memory: vLLM-Omni and SGLang vs M* (B200)

Copied from the quality-compare handoff. Start with [`SUMMARY.md`](SUMMARY.md). This decodes each engine's full
1,021-frame stream and scores it against M*'s stock bf16 stream and the fp32 reference, using `quality.py`'s
metrics. It also takes a memory census (`mem_probe.py`). vLLM-Omni ran in its container (torch 2.13.0+cu130),
SGLang v0.5.21 on its VM (torch 2.13.0+cu130), and M* on `wan-bench-b200` (torch 2.12.1+cu132, `4b8b4789`).
cuDNN 9.20 everywhere. Date: 2026-10-13.

- `quality_compare.csv`, `quality_compare_raw.csv`, `quality_compare_frames.csv`, `memory_compare.csv`: results.
- `meta/`, `runs/`: per-stream metas, analysis logs, `mem_probe.jsonl`.
- `harness/`: the scripts as run.
- `vm-qc/`: the files from `~/qc` on `wan-bench-b200` that weren't in the handoff (memory-probe logs and JSONL, regen script and log).

Not committed:
- `${WAN_BENCH}/qc-frames/` (local, ~29 GB): uint8/float frame dumps of every stream.
- `~/qc/wan2{1,2}/*.npy` and `~/qc/cache/` on `wan-bench-b200` (~19 GB): the M*-side frame dumps and compile caches.
- Not hashed (regenerable with `harness/make_outputs.py`).
