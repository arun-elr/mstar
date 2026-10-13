# Samples: extra files

The sample videos, GIFs, contact sheets, `metrics.csv`, `prompts.md` and scripts are already on this branch in
[`benchmark/wan_vae_decoder/results/b200/samples/`](../../../benchmark/wan_vae_decoder/results/b200/samples/)
and [`benchmark/wan_vae_decoder/samples/`](../../../benchmark/wan_vae_decoder/samples/). Their media and
`metrics.csv` are byte-identical to the handoff's. This folder adds what wasn't committed:

- `runs/`: the per-step result JSONs (window timings, peak memory, md5s, config) for the **fox** prompt, from
  `${WAN_BENCH}/samples-work/`, and `tables.txt`, the full per-step tables for both prompts. The waves prompt's
  per-step JSONs were not copied locally (they would be in `~/samples` on VM `wan-bench-b200-vllm`); its
  numbers are in `tables.txt` and the committed `metrics.csv`.
- `original-handoff/`: the handoff versions of `SUMMARY.md` (published as `README.md`), `prompts.md`,
  `decode_step.py`, `tables.py` and `run_all.sh`, which differ from the committed, edited copies.

Run on one B200 (VM `wan-bench-b200-vllm`) with torch 2.12.1+cu130 and cuDNN 9.20, using PR code at `c56b572b`.
Date: 2026-10-13. One run per step per prompt.

Not committed: `${WAN_BENCH}/samples-work/gif_frame20.png`, a preview frame
(sha256 `e0c101a3aeb5b3789e8eebea99731ee08647be59c8ec01b05a08d09e16c176da`). The generated latents and raw
frames stay on the VM in `~/samples`.
