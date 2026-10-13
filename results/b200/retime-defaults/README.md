# Re-timing at the PR defaults (B200)

The in-tree streaming decoder re-timed after the defaults became
`windowed_decode_latents_per_call=1` with `cudnn.benchmark` off, on one B200
(`wan-bench-b200`, torch 2.12.1+cu132, cuDNN 9.20, diffusers 0.39), 2026-10-13.
2 fresh-process runs per configuration (fresh Inductor/Triton caches),
alternating order; same latents as every other B200 table (sha256 in
`../in-tree/README.md`). Summary: `benchmark/wan_vae_decoder/results/b200/retime.csv`
and `RESULTS.md` §0.

- `runs-663b7e9b/`: before the compact-cache commit. `retime.sh`: defaults and
  opt-in `cudnn.benchmark`, chunk (`bench_wan_decoder.py`) and served window
  (`bench_decode_nodes.py`). `retime2.sh`: M* before (`mstar_step0`; main
  `99f5108b` for the served window) and opt-in `=0`.
- `runs-913295e4/`: the final code (compact cache; `dd8d2512` after it only makes
  an import lazy). `retime3.sh`: defaults and opt-in `=0`, chunk and served window.
- `scripts/`: the three drivers, `retime_collect.py` (builds the CSVs) and the
  queue that ran the served rollouts afterwards.

The chunk runs use the PR branch's `bench_wan_decoder.py` (`mstar_warm` = the
checkout's config defaults); the served-window runs use the results branch's
`bench_decode_nodes.py` with `--preset default`.
