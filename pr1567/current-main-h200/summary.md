Validated 4 fresh starts in 2 matched pairs.
| Metric | Main mean | PR mean | Mean paired change | Pairwise range |
|---|---:|---:|---:|---:|
| Burst short TTFA p50 (s) | 0.716 | 1.112 | +55.41% | +50.91% to +59.91% |
| Burst short TTFA p95 (s) | 0.765 | 1.113 | +45.61% | +41.54% to +49.67% |
| Burst short last-byte p50 (s) | 1.397 | 2.041 | +46.14% | +37.15% to +55.13% |
| Burst short last-byte p95 (s) | 1.398 | 2.107 | +50.75% | +46.39% to +55.10% |
| Burst long last-byte (s) | 5.833 | 6.109 | +5.42% | -3.53% to +14.37% |
| Burst audio-s/s | 9.637 | 9.196 | -3.94% | -12.06% to +4.18% |
| c4 TTFA p50 (s) | 0.714 | 0.418 | -41.41% | -42.82% to -40.01% |
| c8 TTFA p50 (s) | 0.893 | 0.559 | -37.38% | -37.60% to -37.16% |
| c16 TTFA p50 (s) | 1.406 | 0.836 | -39.96% | -45.57% to -34.36% |
| c4 TTFA p95 (s) | 1.491 | 0.575 | -61.41% | -63.74% to -59.08% |
| c8 TTFA p95 (s) | 1.123 | 0.683 | -39.14% | -39.73% to -38.55% |
| c16 TTFA p95 (s) | 1.826 | 0.995 | -45.49% | -46.49% to -44.49% |
| c4 cfg0 TTFA p50 (s) | 0.736 | 0.438 | -40.44% | -40.47% to -40.41% |
| c4 cfg0 TTFA p95 (s) | 1.491 | 0.561 | -62.34% | -63.74% to -60.94% |
| c4 default TTFA p50 (s) | 0.699 | 0.393 | -43.87% | -45.60% to -42.15% |
| c4 default TTFA p95 (s) | 1.490 | 0.567 | -61.97% | -64.88% to -59.06% |
| c8 cfg0 TTFA p50 (s) | 0.894 | 0.572 | -36.05% | -36.65% to -35.45% |
| c8 cfg0 TTFA p95 (s) | 1.195 | 0.690 | -42.14% | -45.77% to -38.52% |
| c8 default TTFA p50 (s) | 0.884 | 0.547 | -38.14% | -39.92% to -36.37% |
| c8 default TTFA p95 (s) | 1.058 | 0.678 | -35.64% | -38.93% to -32.36% |
| c16 cfg0 TTFA p50 (s) | 1.424 | 0.821 | -42.02% | -45.31% to -38.73% |
| c16 cfg0 TTFA p95 (s) | 1.826 | 1.000 | -45.22% | -46.51% to -43.93% |
| c16 default TTFA p50 (s) | 1.412 | 0.838 | -40.01% | -45.99% to -34.04% |
| c16 default TTFA p95 (s) | 1.826 | 0.995 | -45.51% | -46.49% to -44.53% |
| c4 audio-s/s | 14.425 | 13.535 | -6.18% | -8.26% to -4.09% |
| c8 audio-s/s | 26.075 | 20.336 | -22.00% | -22.89% to -21.12% |
| c16 audio-s/s | 32.496 | 29.132 | -10.00% | -15.84% to -4.16% |

| Pair | Cell | Request duration mismatches | Main audio s | PR audio s | Total duration change |
|---:|---|---:|---:|---:|---:|
| 1 | burst | 3/9 | 55.760 | 56.040 | +0.50% |
| 1 | c4 | 37/48 | 221.000 | 221.960 | +0.43% |
| 1 | c8 | 44/48 | 219.600 | 217.480 | -0.97% |
| 1 | c16 | 93/96 | 445.200 | 442.520 | -0.60% |
| 2 | burst | 6/9 | 56.000 | 56.320 | +0.57% |
| 2 | c4 | 41/48 | 222.760 | 223.400 | +0.29% |
| 2 | c8 | 43/48 | 223.200 | 221.120 | -0.93% |
| 2 | c16 | 89/96 | 444.760 | 443.400 | -0.31% |

Pairwise ranges show observed runs, not confidence intervals.
**Duration mismatch:** 356/402 matched requests differ. Throughput and latency comparisons cannot be read as isolated scheduler effects; use the per-request CSV and duration chart to inspect the change.
