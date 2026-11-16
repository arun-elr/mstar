**Cap 8 is historical context, not part of the cap 16/32 rotation.**

| Metric | cap 8 earlier mean | cap 16 mean | cap 32 mean | cap 32 vs 16 paired mean |
|---|---:|---:|---:|---:|
| Burst short TTFA p50 (s) | 1.112 | 1.188 | 1.188 | -0.05% |
| Burst short TTFA p95 (s) | 1.113 | 1.189 | 1.189 | -0.06% |
| c8 TTFA p50 (s) | 0.559 | 0.690 | 0.767 | +11.29% |
| c8 TTFA p95 (s) | 0.683 | 1.252 | 1.217 | -1.97% |
| c16 TTFA p50 (s) | 0.836 | 0.874 | 0.974 | +11.37% |
| c16 TTFA p95 (s) | 0.995 | 1.087 | 1.440 | +32.36% |
| c8 audio throughput (audio-s/s) | 20.336 | 19.023 | 20.056 | +5.44% |
| c16 audio throughput (audio-s/s) | 29.132 | 30.272 | 30.553 | +0.93% |
| Burst generated audio (audio s) | 56.180 | 55.940 | 56.180 | +0.43% |
| c8 generated audio (audio s) | 219.300 | 220.480 | 221.220 | +0.34% |
| c16 generated audio (audio s) | 442.960 | 440.640 | 444.940 | +0.98% |

| Comparison | Cell | Different request durations | Total audio change | Largest request difference |
|---|---|---:|---:|---:|
| between caps, rep 1 | burst | 9/9 | +2.29% | 0.640 s |
| between caps, rep 1 | c8 | 39/48 | -1.44% | 0.880 s |
| between caps, rep 1 | c16 | 88/96 | -0.43% | 0.920 s |
| between caps, rep 2 | burst | 7/9 | -1.43% | 0.440 s |
| between caps, rep 2 | c8 | 45/48 | +2.13% | 1.360 s |
| between caps, rep 2 | c16 | 92/96 | +2.40% | 1.640 s |
| within cap 16 | burst | 3/9 | +0.21% | 0.200 s |
| within cap 16 | c8 | 43/48 | -0.90% | 0.920 s |
| within cap 16 | c16 | 90/96 | -1.05% | 1.400 s |
| within cap 32 | burst | 8/9 | -3.43% | 0.640 s |
| within cap 32 | c8 | 42/48 | +2.69% | 0.960 s |
| within cap 32 | c16 | 91/96 | +1.77% | 1.640 s |

Duration check: 557/612 matched request comparisons differ. Throughput and latency changes are descriptive when output durations differ. Two starts per cap do not establish a confidence interval.
