# Wan 2.2 miscompiles under torch 2.13 (Inductor comprehensive padding)

Copied from the torch-2.13 handoff. Start with [`REPORT.md`](REPORT.md). It ran on VM `wan-bench-b200-vllm` (1× B200)
in the vLLM-Omni image (torch 2.13.0+cu130, cuDNN 9.20.0, diffusers 0.40.0) and in a torch 2.12.1+cu132
env. PR code at `4b8b4789`, rechecked at `c56b572b`. Date: 2026-10-13.

- `fix-inductor-padding.patch`: the fix. `repro_wan22_inductor_padding.py`: the minimal repro.
- `evidence/`: the minified 82-node graph, the Inductor output code with and without padding, and a kernel excerpt.
- `runs/`: the in-tree, repro and verify runs (torch 2.12 / 2.13 × PR / fix). Each is a short run
  (1–75 chunks), so all of them are kept. `scripts/`: the bisect and minifier helpers.
