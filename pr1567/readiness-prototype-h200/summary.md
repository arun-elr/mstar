**Experimental prototype. No production or review-readiness claim.**

| Metric | PR mean | Prototype mean | Mean paired change | Pairwise range |
|---|---:|---:|---:|---:|
| Burst short TTFA p50 (s) | 1.159 | 1.396 | +20.21% | +2.06% to +38.36% |
| Burst short TTFA p95 (s) | 1.160 | 1.440 | +23.84% | +2.06% to +45.63% |
| c8 TTFA p50 (s) | 0.626 | 0.629 | +0.49% | -1.75% to +2.73% |
| c8 TTFA p95 (s) | 1.155 | 1.001 | -14.13% | -26.49% to -1.77% |
| c16 TTFA p50 (s) | 0.886 | 0.845 | -4.53% | -7.05% to -2.00% |
| c16 TTFA p95 (s) | 1.351 | 1.081 | -19.85% | -24.14% to -15.55% |
| c8 audio throughput (audio-s/s) | 18.077 | 18.041 | -0.22% | -1.49% to +1.05% |
| c16 audio throughput (audio-s/s) | 26.702 | 27.486 | +2.93% | +2.27% to +3.60% |
| Burst generated audio (audio s) | 55.860 | 55.740 | -0.21% | -1.50% to +1.08% |
| c8 generated audio (audio s) | 221.600 | 220.800 | -0.36% | -0.95% to +0.24% |
| c16 generated audio (audio s) | 441.480 | 446.200 | +1.07% | +0.53% to +1.61% |

| Comparison | Cell | Different request durations | Total audio change | Largest request difference |
|---|---|---:|---:|---:|
| pair 1, PR vs prototype | burst | 7/9 | -1.50% | 0.440 s |
| pair 1, PR vs prototype | c8 | 42/48 | +0.24% | 0.960 s |
| pair 1, PR vs prototype | c16 | 84/96 | +0.53% | 1.480 s |
| pair 2, PR vs prototype | burst | 8/9 | +1.08% | 0.440 s |
| pair 2, PR vs prototype | c8 | 41/48 | -0.95% | 1.560 s |
| pair 2, PR vs prototype | c16 | 91/96 | +1.61% | 0.960 s |
| PR across starts | burst | 5/9 | -1.07% | 0.600 s |
| PR across starts | c8 | 44/48 | +0.51% | 0.960 s |
| PR across starts | c16 | 91/96 | -0.22% | 1.000 s |
| prototype across starts | burst | 8/9 | +1.52% | 0.480 s |
| prototype across starts | c8 | 45/48 | -0.69% | 0.880 s |
| prototype across starts | c16 | 93/96 | +0.85% | 1.240 s |

273/306 paired PR/prototype request durations differ. Latency and throughput changes are descriptive when output lengths differ. Two pairs show observed variation, not a confidence interval.
