# Attention metadata replay on one H200

This checks the attention binding at [#1517 head `18c6c5c0`](https://github.com/mstar-project/mstar/pull/1517/commits/18c6c5c07edd3c23303c1f308ec8a0f08d155d46), which contains #1492 and #1491. The full source archive used on the VM had SHA-256 `d558863e9c834b8abc87aa25595da52f6d1f3a8af01071ce70f8308671c803de`. The [harness](gate.py) has SHA-256 `8a8121c1d8b66a56002b1cb8c014d2d6febd1e5036fe8af908f3aa7f0d0b98db`; [CPU preflight](preflight.json) from the archive and the exact Git checkout produced identical JSON.

The harness builds state declarations and read-keyed request metadata for two real requests plus a `-2` capture dummy. It captures a 256-token bucket with cold layouts, plans changed warm layouts on a separate CUDA stream with pinned nonblocking H2D staging, then replays. A separately created manager planned only for warm is the comparison on the same Q/K/V. Both paths use TRITON BF16 on one NVIDIA H200.

| Resource | Replay vs fresh warm max abs | Warm vs cold, same inputs max abs | Dummy query rows | Bucket padding |
|---|---:|---:|---:|---|
| Chunk `attn` | 0 | 2.6970 | 8 | zero |
| Expert `self_attn` | 0 | 2.40625 | 4 | zero |
| Expert `cross_attn` | 0 | 2.83594 | 4 | zero |

[Raw result](result.json) SHA-256 `46b211e4ec53ccda76a96b2b0a2f1e3ad044398acaa7157442c38f648e8bee3d`. Every output was BF16. Dummy queries are compared with the fresh warm manager; the caller discards them. The first run failed a [harness assertion](first-run.stderr) that incorrectly required those dummy rows to be zero. The source was unchanged; the corrected run passed. Only bucket padding after all planned spans must be zero.

The spot VM ran from 10:22:07 to 10:25:35 UTC on 2026-11-14, about 0.06 GPU-hours, and was confirmed stopped. This is a metadata and resource capture test. It does not run the full async worker, a model checkpoint, served CUDA graphs, or a performance comparison.

To rerun after checking out the exact source head, use the archived harness with `MSTAR_FLEX_BACKEND=TRITON OMP_NUM_THREADS=2 python gate.py --repo /path/to/checkout --mode gpu`. The run used torch `2.12.1+cu132`, a 30-minute process timeout and a 60-minute VM stop timer.
