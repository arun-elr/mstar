# #1567 Chatterbox H200 benchmark, 2026-11-16

This is a current-head comparison of upstream main `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` and PR #1567 `1959fb2eac6f143f3fdeefe783cd49d517ba4b06`. Both used the same single H200, Python environment, Rust frontend binary, local Chatterbox checkpoint, and `config.yaml`. The opt-in fairness peek was off (`MSTAR_SPEC_PEEK_FOR_FAIRNESS=0`) and the consecutive speculation cap was 8. This measures the opt-in path; it does not measure the default peek path.

Four fresh server starts ran in main, PR, PR, main order, paired in adjacent starts. Each start sent one overlapping long/short mixed-CFG burst, then closed-loop mixed-CFG sweeps at concurrency 4, 8, and 16. The sweep sent 48, 48, and 96 measured requests respectively. Every measured request succeeded. The long burst request was still active when the eight short requests were submitted in every start.

The burst's short-request TTFA p50 rose from 0.716 to 1.112 s (+55.4% mean paired); the c8 sweep's TTFA p50 fell from 0.893 to 0.559 s (-37.4%). At c8, generated audio per wall second fell from 26.075 to 20.336 (-22.0% mean paired). The paired c8 throughput changes were -22.9% and -21.1%. This is a workload-dependent latency tradeoff with a measured throughput regression in this configuration. Two pairs show observed variation, not a confidence interval.

The client fixed prompts and seeds, but generated duration differed in 356 of 402 matched requests. Total c8 generated audio differed by about 1% per pair. The charts include a duration comparison so throughput and latency are not presented as isolated scheduler effects. A separate cap-tuning ablation, if published, is PR-only and does not change this main versus PR result.

## Plots and data

- `chatterbox_mixed_burst.png`: overlap burst, short TTFA and last-byte latency.
- `chatterbox_mixed_sweep.png`: sweep TTFA, throughput, and audio duration by concurrency.
- `chatterbox_cfg_lanes.png`: CFG=0 and default-CFG TTFA lanes.
- `chatterbox_audio_duration.png`: paired generated-audio differences.
- `summary.md`, `paired.csv`, `per_start.csv`, `request_audio_comparison.csv`, `audio_comparability.csv`: derived results.
- `main-*` and `pr-*`: per-request JSON and server logs; `manifest.csv` records the run order.
- `run.sh`, `config.yaml`, `burst_client.py`, `sweep_client.py`: exact server setup and traffic. The local model path and Python environment in `run.sh` and `config.yaml` need to be set for another host.
- `chart.py`: validates the manifest and per-request records, then regenerates the plots and derived data. With Matplotlib installed, run `python chart.py manifest.csv --out . --main-head 2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175 --pr-head 1959fb2eac6f143f3fdeefe783cd49d517ba4b06` from this directory.

`SHA256SUMS` covers the published artifacts except itself.
