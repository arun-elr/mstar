# Cosmos3-Edge served end to end: re-run at the final PR defaults (B200)

`before` (main `99f5108b`) and `default` (the PR at `913295e4` with no overrides:
warm cache, channels-last, `windowed_decode_latents_per_call=1`,
`cudnn.benchmark` off, compact cache; `dd8d2512` after it only makes an import
lazy), one stream, kv (1 warm-up + 3 timed) and chained (1 warm-up + 2 timed),
on `wan-bench-b200` (1× B200, torch 2.12.1+cu132, flashinfer 0.7.1, cuDNN 9.20),
2026-10-13. Same request, checkpoint and scripts as
[`../e2e-cosmos3-edge`](../e2e-cosmos3-edge/) (`PHASES=c1`), with
`E2E_NO_FA3=1` so the DiT takes M*'s dense -> flashinfer fallback, as there.
`out-rerun/results.csv` is `scripts/analyze.py runs out-rerun`; the mp4 chunks
are not committed.

| one stream | before | default |
|---|---:|---:|
| kv: time to first frame | 1.37 s | 1.34 s |
| kv: total rollout | 14.26 s | 11.93 s (−16.4%) |
| kv: DiT / decoder per window | 1,287 / 542 ms | 1,293 / 240 ms |
| kv: decoder share of wall time | 30.4% | 16.1% |
| kv: peak allocated above idle | 5.31 GiB | 4.38 GiB |
| chained: time to first frame | 1.31 s | 1.27 s |
| chained: total rollout | 11.89 s | 9.04 s (−24.0%) |
| chained: DiT / decoder per window | 749 / 472 ms | 759 / 179 ms |
