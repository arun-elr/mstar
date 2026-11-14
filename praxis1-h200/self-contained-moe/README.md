# Praxis-1 self-contained MoE combine: H200 ABBA

This folder records the **fourth**, production-candidate experiment. The candidate has its own fused kernel in M\* source and does not import the Praxis reference at runtime. An earlier diagnostic experiment did import a reference kernel; its numbers and chart are separate and are not used here.

The comparison used a downstream copy of M\* `5902054c0f` with the self-contained candidate applied. It was **not measured on the current #1493 head**. Each of four fresh servers used one NVIDIA H200, the `runwayml/praxis-1-droid` checkpoint, the Python frontend, Triton attention, two Euler steps, M\* eager `-O0`, and no serving graph capture. The order was eager, fused, fused, eager. Each server ran five three-call sessions; the first session warmed up the server, leaving four measured sessions per server.

| Served call | Eager mean arm p50 | Fused mean arm p50 | Reduction |
|---|---:|---:|---:|
| Cold, 21 rows | 274.015 ms | 238.496 ms | 12.96% |
| Warm 1, 24 rows | 547.318 ms | 467.223 ms | 14.63% |
| Warm 2, 24 rows | 543.236 ms | 466.192 ms | 14.18% |

Each value is the mean of the two per-server medians, rather than a pooled median of all calls. The metric is client round-trip time. [The chart](moe-combine-h200-abba.png) plots those means and marks the two individual server medians. [The sanitized summary](summary.json) holds the four measured latencies per call and server, p50/p95 values, report checksums, and source checksums. `plot.py` regenerates the chart from that summary.

The 15 command hashes matched across all four arms, including warmup calls. The focused GPU suite reported **12 passed**. A separate gate checked bitwise BF16 eager/fused parity for 31, 1,284, and 3,468 token cases and successful CUDA graph capture and replay at each size. A DROID-shaped 1,284-token grouped MoE dispatch also matched bitwise while the eager fallback was made fatal. These are checks of the combine path; they do not establish whole-model numerical parity with the reference.

## Provenance

| Item | SHA-256 |
|---|---|
| Candidate `moe.py` | `24b0115abfa8f9fc89297d0b3e43380d24ca68713420bc675943664aaf4c9a0b8580` |
| Candidate fused kernel | `920e75620c2f1129b06a17210793ddfa290a6cadb1ce4a978c3e39dd7dedb011` |
| Matched client benchmark script | `65370861d3545910e63275e5b2d4bb63fdb875c958157e83a6c1efdacf75a06c` |
| ABBA launch script | `2406901282cb3ea276312fd4816b9aa6fba8843db24a312abab47a6f1a744dbd` |
| Client invocation script | `ba2ec5a3465819dc57b35ff142ad13b775a82e74d150fc40d8003c1d805c9f50` |
| GPU gate script | `5c5fb8ea1224d2bca1d87424c93dc31150bb5c5b887c1091e4fc5c7ae9e40fac` |

The four raw report checksums are in `summary.json`, keyed by experiment arm. The raw reports and server logs are retained separately because they contain local machine paths, runtime identifiers, and command payloads. No raw request or command arrays are included in this folder.

## Limits

Only two fresh servers were measured per mode, with four measured sessions per server. This is an A/B test of the combine change within the downstream M\* copy, not a comparison against main or a broad latency distribution. The serving benchmark used eager `-O0` and no graph capture; the isolated CUDA graph replay gate is a separate correctness check. The candidate has not had a new seven-task closed-loop simulation run or a current #1493 head run. The previous downstream run passed 6/7 simulation tasks, and warm latency after this change remains about 1.8 times the matched reference. The serving gate remains closed.
