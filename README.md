# PR #1556 H200 served A/B

Measured 16 November 2026 on one H200 (driver 580.173.02) with [Qwen3.5-0.8B revision `2fc0636`](https://huggingface.co/Qwen/Qwen3.5-0.8B/tree/2fc06364715b967f1860aea9cf38778875588b17). The weight file SHA256 is `04b1c301231dd422b8860db31311ab2721511346a32cb1e079c4c4e5f1fe4696`.

The comparison used upstream main `2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175` and PR #1556 `258b81371bc4626797177a5c93a14dad23405961`, in the order main, PR, PR, main. Each arm restarted the server. The same YAML, Python frontend, Rust graph runtime and communicator, SHM tensor transport, `MSTAR_COMPILE_LEVEL=0`, and checkpoint were used throughout. Each source archive's key file digests were checked against `git show` at its recorded commit. The exact launch command, environment, wheel hash and file digests are in each `raw/*-metadata.json`.

Eight fixed text prompts were repeated. The runner sent 64 output tokens per request with `ignore_eos=true`: 320 requests at concurrency 1, 1,280 at 8, and 2,560 at 16. Each cell had two warmup waves before timing. All 16,640 measured requests completed with exactly 64 generated tokens and no errors. Throughput divides completed requests or actual generated tokens by **measurement wall time**, excluding warmup. The raw JSON includes every request's token count and latency.

| Concurrency | Main requests/s | PR requests/s | PR vs main | Main tokens/s | PR tokens/s | Main p95 ms | PR p95 ms |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 10.60 | 10.57 | -0.28% | 678 | 677 | 103.4 | 102.9 |
| 8 | 63.22 | 63.97 | +1.20% | 4,046 | 4,094 | 140.1 | 136.0 |
| 16 | 104.41 | 105.06 | +0.62% | 6,682 | 6,724 | 168.4 | 167.1 |

Each table value is the mean of two server restarts. [Throughput](throughput.png) and [latency](latency.png) show both individual runs as faint points and their min-to-max span as whiskers. Those whiskers are **not** confidence intervals. The observed throughput differences are smaller than or similar to the run-to-run spread, so this bounded test does not establish a speedup or exact equivalence. Tokens/s is requests/s times 64 in this fixed-output workload.

The Python frontend's live check used [this client script](live.py). Two sequential calls with the same client ID returned 200 and identical sampled outputs without an explicit seed. A different client ID gave a different sampled output. A duplicate call while the first was running returned 409 after the documented grace period, and the first call completed 200. The [raw response](raw/pr1-live.json) and script preserve the check.

The [Rust frontend smoke](raw/rust-live.json) used the same PR code, checkpoint and client script. Sequential reuse returned 200 twice with identical sampled output, a concurrent duplicate returned 409 while the first call completed 200, and a different ID gave a different sampled output. The first same-ID sampled output also matched the Python frontend. Its [launch and binary provenance](raw/rust-metadata.json) is recorded separately. Rust frontend performance is not part of these plots.

The performance workload covers one text model, one GPU, one frontend, and 64-token outputs. It does not measure tensor parallelism, multimodal requests, or longer generations.

To reproduce, use the checked-out commit in each `raw/*-metadata.json`, place the pinned checkpoint at the path in [the YAML](qwen3_5_0.8b-local.yaml), start the server with the exact command and environment in that metadata, and run [the benchmark wrapper](bench.py), for example:

```sh
python bench.py --arm main1 --head 2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175 \
  --output main1-c8.json --prompts prompts.txt --concurrency 8 --requests 1280
```

Run `python plot.py` with `matplotlib` and `numpy` to rebuild the figures and [summary](summary.json) from `raw/`. The plot script fails if source heads, checkpoint/config identity, request counts, or 64-token outputs do not match the recorded workload.
