# #1567 experimental readiness gate, 2026-11-16

This is a local experiment, not code in PR #1567. The control was PR head `1959fb2eac6f143f3fdeefe783cd49d517ba4b06`; the prototype added `prototype.patch`. Only `mstar/worker/speculation.py` differed between the two server source trees. The extra test changes in the patch were not loaded by the server. Both arms used one H200, the same Chatterbox checkpoint, config, Python environment, Rust frontend binary, cap 8, and `MSTAR_SPEC_PEEK_FOR_FAIRNESS=0`.

Four fresh starts ran PR, prototype, prototype, PR. Every start sent an overlapping long/short mixed-CFG burst and c8/c16 closed-loop sweeps. All requests succeeded; the long request remained active when the shorts arrived. The prototype checks for another ready key at each saturated cap step before yielding. It changes the documented peek-off behavior and performs readiness scans that can have side effects. The published PR does not contain this change.

The prototype did not recover the c8 throughput or burst latency measured against upstream main. Against the PR control in these two pairs, c8 audio throughput changed from 18.077 to 18.041 audio-s/s (-0.22% mean paired), while burst short TTFA p50 rose from 1.159 to 1.396 s (+20.21% mean paired). The two paired burst changes were +2.06% and +38.36%, so the mean is not a stable estimate. c8 TTFA p95 improved by 14.13%, but that does not offset the unresolved burst and throughput results. Generated duration differed in 273 of 306 paired requests. These are descriptive results from two pairs, not a causal estimate or confidence interval. The prototype remains uncommitted and is not proposed for the PR.

## Plots and data

- `chatterbox_readiness_ab.png`: paired burst/c8/c16 latency and throughput.
- `chatterbox_readiness_audio_duration.png`: paired total generated audio durations.
- `summary.md`, `per_start.csv`, `paired.csv`, `request_audio_comparison.csv`, `audio_comparability.csv`: validated derived results.
- `pr-*` and `proto-*`: per-request JSON and server logs; `manifest.csv` gives the run order.
- `prototype.patch`, `run.sh`: exact local code experiment and server/client commands. Apply the zero-context patch to PR head with `git apply --unidiff-zero prototype.patch`; the local paths in `run.sh` need adjusting for another host.
- `chart.py` and `mstar-pr1567-cap-ablation-charts.py`: parser and raw-request validator. Run `python chart.py --manifest manifest.csv --out . --pr-head 1959fb2eac6f143f3fdeefe783cd49d517ba4b06 --proto-head local-readiness-gate` from this directory with Matplotlib installed.

`SHA256SUMS` covers the published artifacts except itself.
