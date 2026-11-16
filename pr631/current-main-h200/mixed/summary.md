| Concurrency | Metric | Main mean | PR mean | Mean paired change | Range of paired changes |
|---:|---|---:|---:|---:|---:|
| 1 | TTFA p50 | 150.55 ms | 149.45 ms | -0.73% | -1.32% to +0.05% |
| 1 | TTFA p95 | 174.09 ms | 172.42 ms | -0.89% | -5.03% to +1.50% |
| 1 | Audio throughput | 15.80 audio-s/s | 16.09 audio-s/s | +1.88% | +0.85% to +2.56% |
| 1 | Generated audio duration | 380.28 audio s | 354.40 audio s | -6.81% | -6.81% to -6.81% |
| 1 | Measured wall time | 24.07 s | 22.02 s | -8.52% | -9.13% to -7.59% |
| 8 | TTFA p50 | 554.93 ms | 518.43 ms | -6.57% | -8.26% to -4.09% |
| 8 | TTFA p95 | 717.64 ms | 699.18 ms | -2.60% | -8.26% to +4.49% |
| 8 | Audio throughput | 35.48 audio-s/s | 35.89 audio-s/s | +1.18% | -0.80% to +2.99% |
| 8 | Generated audio duration | 1153.29 audio s | 1083.76 audio s | -6.00% | -8.71% to -2.54% |
| 8 | Measured wall time | 32.51 s | 30.20 s | -7.11% | -7.99% to -5.37% |

Three matched fresh-start pairs per concurrency. Percentages are the mean of pairwise PR/main changes, not a confidence interval. Throughput is generated audio duration divided by wall time, so a duration change can affect it. See paired.csv for each pair.
