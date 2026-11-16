# Generation control and weight checker H200 run, 2026-11-16

Live run on the stacked M* #1573 tree at `5b42f8e906edd29aa16970d95da5ee4a2ad162b7`, which includes #1542 at `25e29a868cc068ebbf645a9f915cbe55225e292a`. This predates the subsequent count/logging fix on #845 (`49ec`) propagated to #1542 (`d0d796`) and #1573 (`a7f7`). The new heads have CPU checks but no repeat GPU run in this asset.

One NVIDIA H200, one worker rank, SHM transport, default Rust control mesh. Python/FastAPI and a Rust frontend binary built from the tested source were run separately. The binary SHA-256 was `c9449061b3dc4f7a48ee6c8c36236ece6098d0af755dd34dc368b92c28735900`.

The real Orpheus FT checkpoint was `canopylabs/orpheus-3b-0.1-ft` at Hub revision `4206a56e5a68cf6cf96900a8a78acd3370c02eb6`. Its four safetensors shards were verified on both machines against these SHA-256 hashes:

1. `8feb9137e2a243d33827c2196ec81270179ec6e7acf8972a37f84b5eb6852450`
2. `69df9510f4b3e8d883b51e5d19f5f5b669d99ba9990f9e0e16f4fedd3d73709e`
3. `d24f6a0f618bd4aa09a23216fe55d086a6db92ee896f98740972702f265f026b`
4. `a1deaa1b3f38d5b9efe22f012dbb5c3f29e15f31b15dcf95e471a85bb03d2d21`

The tokenizer used `canopylabs/orpheus-3b-0.1-pretrained@bf0cce99761b2f5857b3d85829691f696bf20cb0`; SNAC used `hubertsiuzdak/snac_24khz@d73ad176a12188fcf4f360ba3bf2c2fbbe8f58ec`. All snapshots were loaded from offline cache after verification.

Both frontends returned 401 without a key, 200 for pause, 409 when checking weights while running, 200 for a paused checksum and matching compare, 409 for a deliberately wrong digest, and 400 for a checkpoint outside the allowlist. In-place reload of the **same** published checkpoint to version `v2` returned 200, then a second checksum matched the first; resume and the reported version were 200. One reload sample took 3.980 s on Python and 3.972 s on Rust. `/v1/audio/speech` returned a valid 24 kHz WAV before and after reload on both frontends; the files are in `audio/`. The deterministic request produced equal lengths and near-identical, non-byte-identical samples. No audio quality judgment is inferred from this check.

The three successful weight-checker calls per frontend took 7.672-8.020 s each. [This chart](weight-checker-latency.svg) plots those individual calls from the raw JSON; `plot_checker_latency.py` regenerates it. These are functional-probe wall times, not a repeated benchmark or serving-throughput result.

The [reload wall-time chart](same-checkpoint-reload-latency.svg) plots the one same-checkpoint reload call per frontend. `plot_reload_latency.py` regenerates it from the same raw JSON. It says nothing about changed-weight reloads or repeated latency.

The live script wrote a before-reload speech result to JSON, then overwrote that field on subsequent requests. The four original WAV files are intact. `audio_artifacts.json` records their hashes, sizes, sample rates and durations computed after the run; it does not reconstruct an HTTP status or timing from a file.

Qwen3.5-0.8B at Hub revision `2fc06364715b967f1860aea9cf38778875588b17` was also tested for auth, pause, checksum, comparison and resume on both frontends. Reload returned 400 with `Qwen3_5DenseModel does not support reloading weights in place`, as expected for that model.

The tested head emitted a Python logging `TypeError` while reporting the number of KV pages freed because it formatted a returned list with `%d`. The reload itself committed version `v2` and served after this logging error. The follow-up #845 fix was not included in this GPU run.

`raw/` contains the sanitized live request/response JSON, branch-hit log excerpts, config and checker scripts. Only checkpoint paths and local addresses were replaced with placeholders; status codes, timings, digests and worker messages are unchanged. The full unsanitized logs are retained locally. No multi-rank/TP reload, changed-weight output check, throughput, or repeated reload latency samples were run.
