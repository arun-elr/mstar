# #1567 cap setting check, 2026-11-16

This is a PR-only cap setting check on one H200 at #1567 head `1959fb2eac6f143f3fdeefe783cd49d517ba4b06`. The source, Chatterbox checkpoint, config, clients, Rust frontend binary, and `MSTAR_SPEC_PEEK_FOR_FAIRNESS=0` match the [main versus PR run](../current-main-h200/README.md). Four fresh starts ran cap 16, cap 32, cap 32, cap 16. Each start sent the same overlapping long/short mixed-CFG burst and c8/c16 closed-loop sweeps. All requests succeeded and the long request was active when the shorts arrived.

The earlier cap-8 PR starts are plotted as historical, unpaired context. Cap 16 and 32 did not recover the burst latency or c8 throughput seen in the main versus PR run: burst short TTFA p50 was 1.188 s at both caps, and c8 audio throughput was 19.023 and 20.056 audio-s/s respectively. The earlier cap-8 PR means were 1.112 s and 20.336 audio-s/s. In the paired 16/32 starts, cap 32's c8 TTFA p50 was 11.3% higher on average. These are two starts per cap, not a confidence interval or a causal estimate; output durations differed in 557 of 612 matched request comparisons.

## Plots and data

- `chatterbox_cap_ablation.png`: burst short TTFA and c8/c16 latency and throughput. Each dot is one fresh start; bars are means.
- `chatterbox_cap_audio_duration.png`: total generated audio duration by cell and cap.
- `summary.md`, `per_start.csv`, `paired.csv`, `request_audio_comparison.csv`, `audio_comparability.csv`: validated results and duration checks.
- `cap16-*` and `cap32-*`: raw request reports and server logs; `manifest.csv` records the run order.
- `run.sh`: exact server and client commands, with the local model path in the companion config to adjust for another host.
- `chart.py`: regenerates figures and derived data. Run `python chart.py --manifest manifest.csv --baseline-manifest ../current-main-h200/manifest.csv --out . --pr-head 1959fb2eac6f143f3fdeefe783cd49d517ba4b06` from this directory with Matplotlib installed.

`SHA256SUMS` covers the published artifacts except itself.
