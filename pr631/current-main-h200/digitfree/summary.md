**Cannot isolate frontend overhead:** 10/92 matched main/PR requests generated different PCM lengths despite matched prompts and seed. TTFA and throughput below are descriptive; output-length and scheduling effects may contribute.

| Metric | Main mean | PR mean | Mean paired change | Pairwise range |
|---|---:|---:|---:|---:|
| TTFA p50 | 503.24 ms | 506.45 ms | +0.64% | +0.63% to +0.65% |
| TTFA p95 | 770.09 ms | 731.94 ms | -4.60% | -10.28% to +1.08% |
| Audio throughput | 34.83 audio-s/s | 34.46 audio-s/s | -1.06% | -2.98% to +0.86% |
| Generated audio duration | 534.56 audio s | 519.20 audio s | -2.87% | -6.45% to +0.71% |
| Measured wall time | 15.35 s | 15.06 s | -1.86% | -3.57% to -0.15% |

| Matched pair | Different request lengths | Main audio s | PR audio s | Largest request difference |
|---:|---:|---:|---:|---:|
| 1 | 7/46 | 534.04 | 537.84 | 22.32 s |
| 2 | 3/46 | 535.08 | 500.56 | 17.44 s |

| Same-arm starts | Different request lengths | First audio s | Second audio s | Largest request difference |
|---|---:|---:|---:|---:|
| main order 1 vs 4 | 6/46 | 534.04 | 535.08 | 17.44 s |
| pr order 2 vs 3 | 7/46 | 537.84 | 500.56 | 22.32 s |

Within-arm variability: 13/92 request durations differ across fresh starts. The two paired changes above are observed runs, not a confidence interval.
Digit-free 46-prompt file SHA256: `cee1ab88344ffc51e896e70f176da6747ae6579038a52868ecbf78c838ddee79`.
