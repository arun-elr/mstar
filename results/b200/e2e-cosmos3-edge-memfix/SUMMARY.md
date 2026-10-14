# End-to-end Cosmos3-Edge re-run of mstar-project/mstar#425 at its final code (with the decoder-cache memory fix)

## TL;DR

- **The memory fix held, and did better than expected.** Code: PR head `913295e492ee1628b74947f46aec0ea519a38f9e` ("Keep the Wan decoder cache in compact buffers"), served through the M* server on the same B200, with the same setup as `e2e-cosmos3/`.
  - **Growth per extra live stream at `windowed_decode_latents_per_call=1`:** 3.50 GiB → **1.33 GiB**, measured from concurrency 1 to 4. That is exactly the Wan 2.2 decoder cache, not the ~1.8 GiB expected. The one-call transient (about 3.05 GiB) is paid once, not per stream, because the decoder runs one stream's call at a time.
  - **At `=0`:** 18.7 GiB → **1.33 GiB** per extra stream.
- **Peak allocated above idle at `=1`, old → fixed:**
  - c1: 6.3 → 4.4 GiB
  - c4: 16.8 → 8.4 GiB
  - c8: 31.7 → 14.6 GiB
  - nvidia-smi at c8: 67.1 → 48.7 GB
- **`=0` no longer runs out of memory at concurrency 8:** it peaks at 28.3 GiB above idle (57.9 GiB allocated), where it used to OOM at 175.6 GiB. At c4 it now uses 22.9 GiB above idle, against 92.3 GiB before.
- **Timing at `=1` is within noise of the old after1 run:**
  - kv c1 total 11.91 s against 11.88 s (+0.25%), window gap 1.550 s against 1.550 s, time to first frame 1.39 s against 1.38 s;
  - kv c4 aggregate 17.52 against 17.50 fps;
  - chained c1 9.11 s against 9.01 s (one 9.20 s round; the other round and the dump took 9.02 s).
  - The decoder per window is 1.5–2.8% slower (236 against 232 ms at kv c1). Two things changed and I can't separate them: the fix's cache copy, and `vae_decode_cudnn_benchmark`, which is now off by default (old after1 forced it on).
- **The output is equivalent:**
  - after1fix's latents are bit-identical to the earlier session's in both modes, so the frame comparisons are like for like.
  - Its frames score the same against the fp32 whole-stream reference as old after1's: kv 53.46 dB against 53.46 dB (worst frame 49.43 against 49.52), chained 52.33 against 52.33 dB.
  - Its frames are 57.8 dB from old after1's (max 19 levels). That is the same size as the cuDNN-algorithm differences seen between launches before, not bit-exact because old after1 ran with `cudnn.benchmark` on.
- **New, and not caused by the PR: concurrency 8 now fails 3 of 8 streams with an `AssertionError` in the DiT's UniPC scheduler.**
  - It happens on the fixed code at both settings, and on the **old PR tree `60fbc5cd` re-run today**, which completed 8/8 in the earlier session.
  - It sits in the DiT path's KV-page-pressure hold/retry. The PR doesn't touch the DiT, worker, engine or conductor relative to upstream main.
  - The memory peaks at c8 are still valid: the old tree's c8 peak today, 60.23 GiB, equals its earlier 60.24 GiB, and the failures come about 67 s into the round. The c8 timings are comparable only with today's old-tree re-run, not with the earlier session.

GPU time: **27 minutes** on 1×B200 (`wan-bench-b200-vllm`):

- after1fix: 5.7 min timed launch + 2.9 min c8 launch
- after0fix: 6.0 min + 2.9 min
- old-tree c8 re-run (attributing the assertion): 2.9 min
- latent and frame comparison with reference decodes: 6.8 min

## Code and setup

| | |
|---|---|
| Branch head used | `origin/perf/wan-vae-decode` = `913295e492ee1628b74947f46aec0ea519a38f9e` (fork arun-elr/mstar), pushed 22:55:40 UTC. It appeared on my second poll, about 5 min after I started polling at 22:51 UTC. **No fallback patch was used.** |
| Fix contents | `mstar/model/components/wan_vae.py` is byte-identical to `memory-fix/fix.patch` applied to the previous head (md5 `ba353ee8…`). The commit also adds the patch's tests. |
| Code on the VM | `git archive 913295e4` into `~/e2e-cosmos3/mstar-fix` (committed files only, with a `REV` file). `~/oss/wt-wan-decoder` was not touched. No dependency changes since the old tree (`pyproject.toml` unchanged), so the same venv was used. |
| after1fix | PR head with **no overrides**. Its defaults (`Cosmos3Config` at `913295e4`): `windowed_decode_warm_cache=True`, `windowed_decode_max_warm_streams=0` (no cap), `windowed_decode_latents_per_call=1`, `vae_decode_channels_last=True`, **`vae_decode_cudnn_benchmark=False`**. `configs/cosmos3_edge.yaml` sets none of these (only commented examples). |
| after0fix | The same, with `windowed_decode_latents_per_call: 0` overlaid into `model_kwargs`. |
| before / after0 / after1 (old) | From `e2e-cosmos3/` (session of 19:40–20:20 UTC): upstream main `99f5108b`, and PR tree `60fbc5cd` with `cudnn_benchmark: true` and `=0` / `=1`. |
| Everything else | Identical to `e2e-cosmos3/SUMMARY.md`: the same VM, venv, HF snapshot (Edge @ `344d602b`), image (md5 `1dd31d4d…`), prompt, seeds 0..N-1, client flags (`stream_client.py`), memory and dump hook (`hook/sitecustomize.py`), flashinfer fallback for the DiT's dense attention (fa3-fwd still absent), server environment and phases. kv c1 ran 1 warm-up + 3 timed rollouts, kv c4 1 warm-up round + 2 timed, chained c1 1 warm-up + 2 timed, plus kv and chained dumps. c8 ran in a fresh launch: a c1 warm-up, then 1 round of 8 streams. |
| Scripts | `run_variant.sh` gained the two variant cases and nothing else. The old variants run exactly as before (`run_variant.sh.orig` is kept on the VM). `analyze.py` takes its variant list from `VARIANTS`. `run_fix.sh` chains the four launches through `run_all.sh`, which waits for an idle GPU. `cmp_fix.py` does the latent and frame check. |
| GPU checks | Before starting: `nvidia-smi` showed no compute apps and 0 MiB, with no tmux sessions and no containers. `run_all.sh` re-checked before each launch. One server at a time; every job ran in tmux. |

**Idle memory is 1.03 GiB higher** (29.57 against 28.54 GiB allocated). This comes from the PR's later commit `b9a63ca3`: with channels-last, the streaming node loads its own decoder-only VAE copy instead of sharing the whole-clip decoder's (documented in the config comment). It is a fixed cost, not per stream. The "above idle" figures below subtract each variant's own idle.

## Results (kv mode, 241 frames, 8 windows)

Definitions are as in `e2e-cosmos3/SUMMARY.md`: medians over the timed rollouts; "decoder per window" is the `vae_decoder_ar` node's CPU-timed total ÷ windows, from `--log-stats`; "peak" is `torch.cuda.max_memory_allocated` in the worker process during the phase.

### Concurrency 1 and 4

| | before | after1 (old) | **after1fix** | after0 (old) | **after0fix** |
|---|---|---|---|---|---|
| c1 time to first frame | 1.43 s | 1.38 s | **1.39 s** | 1.61 s | 1.53 s |
| c1 window gap (median) | 1.886 s | 1.550 s | **1.550 s** | 1.534 s | 1.538 s |
| c1 total rollout | 14.27 s | 11.88 s | **11.91 s** | 12.01 s | 11.94 s |
| c1 delivered fps | 16.89 | 20.28 | **20.23** | 20.06 | 20.19 |
| c1 DiT / decoder per window | 1287 / 534 ms | 1287 / 232 ms | **1294 / 236 ms** | 1294 / 244 ms | 1292 / 237 ms |
| c1 peak allocated above idle | 5.31 GiB | 6.31 GiB | **4.38 GiB** | 36.12 GiB | **18.93 GiB** |
| c4 window gap per stream | 8.55 s | 7.18 s | **7.19 s** | 7.13 s | 7.11 s |
| c4 rollout per stream / round wall | 64.2 / 65.0 s | 54.75 / 55.07 s | **54.75 / 55.03 s** | 55.18 / 55.65 s | 54.87 / 55.10 s |
| c4 fps per stream / aggregate | 3.75 / 14.84 | 4.40 / 17.50 | **4.40 / 17.52** | 4.37 / 17.32 | 4.39 / 17.50 |
| c4 decoder per window | 880 ms | 261 ms | **268 ms** | 295 ms | 265 ms |
| c4 peak allocated above idle | 5.41 GiB | 16.81 GiB | **8.38 GiB** | 92.32 GiB | **22.94 GiB** |
| c4 peak allocated / nvidia-smi peak | 33.97 GiB / 38.2 GB | 45.35 GiB / 50.9 GB | **37.95 GiB / 42.0 GB** | 120.86 GiB / 142.3 GB | 52.51 GiB / 60.4 GB |
| **growth per extra stream, (c4 − c1) / 3** | 0.03 GiB | 3.50 GiB | **1.33 GiB** | 18.73 GiB | **1.33 GiB** |

### Concurrency 8 (fresh launch; c1 warm-up, then 1 round)

| | before (earlier) | after1 (old, earlier) | **after1 old tree, re-run today** | **after1fix** | after0 (old, earlier) | **after0fix** |
|---|---|---|---|---|---|---|
| streams completed | 8/8 | 8/8 | **5/8 (3 AssertionError)** | **5/8 (3 AssertionError)** | 1/8 (7 OOM) | **5/8 (3 AssertionError), no OOM** |
| window gap (median, completed streams) | 10.73 s | 8.03 s | 7.12 s | 7.14 s | 1.56 s | 7.04 s |
| rollout per stream (median, completed) / round wall | 103.5 / 128.8 s | 88.6 / 109.5 s | 91.8 / 99.4 s | 92.2 / 99.8 s | 35.2 / 35.2 s | 92.2 / 99.8 s |
| decoder per window | 843 ms | 370 ms | 439 ms | 450 ms | 426 ms | 435 ms |
| peak allocated above idle | 5.52 GiB | 31.70 GiB | 31.69 GiB | **14.62 GiB** | 147.05 GiB (OOM) | **28.34 GiB** |
| peak allocated / nvidia-smi peak | 34.06 GiB / 38.3 GB | 60.24 GiB / 67.1 GB | 60.23 GiB / 68.0 GB | **44.19 GiB / 48.7 GB** | 175.59 GiB / 182.1 GB | **57.91 GiB / 64.1 GB** |
| **growth per extra stream, (c8 − c1) / 7** | 0.03 GiB | 3.63 GiB | 3.63 GiB (against old c1) | **1.46 GiB** | n/a | **1.34 GiB** |

- **c8 timings are comparable only within today's session,** because 3 streams die partway through every c8 round today (see below). Against today's old-tree re-run, after1fix is within noise: per-stream rollout 92.2 against 91.8 s, gap 7.14 against 7.12 s, decoder 450 against 439 ms. Aggregate fps is not meaningful with 3 failed streams, so I left it out; it is in the CSV.
- **The c8 memory peaks are valid:**
  - the old tree's c8 peak today (60.23 GiB) equals its earlier 8/8 run (60.24 GiB);
  - the 3 streams fail about 67 s into the 99.8 s round, after all 8 have held a warm cache since their first window (time to first frame ≤ 14 s).
- **Why c8 growth at `=1` reads 1.46 rather than 1.33:** going from c4 to c8, the peak above the 8 caches rises from 3.06 to about 4.0 GiB. That is likely the DiT's or the decoder's transient under the 8-way KV-pressure holds; it is not per-stream retention.
  - after1fix: c1 4.38 = 1 × 1.33 + 3.05; c4 8.38 = 4 × 1.33 + 3.06; c8 14.62 = 8 × 1.33 + 3.98.
  - The 3.05 GiB matches the memory-fix agent's compiled Wan 2.2 `=1` transient of 3.03 GiB.

### Chained mode, 241 frames, 10 windows, c1

| | before | after1 (old) | **after1fix** | after0 (old) | **after0fix** |
|---|---|---|---|---|---|
| time to first frame | 1.35 s | 1.32 s | **1.36 s** | 1.53 s | 1.47 s |
| window gap (median) | 1.172 s | 0.861 s | **0.866 s** | 0.854 s | 0.854 s |
| total rollout | 11.85 s | 9.01 s | **9.11 s** (rounds 9.20, 9.02; dump 9.02) | 9.13 s | 9.09 s |
| delivered fps | 20.34 | 26.76 | **26.45** | 26.39 | 26.50 |
| DiT / decoder per window | 744 / 467 ms | 753 / 173 ms | **760 / 178 ms** | 757 / 183 ms | 754 / 180 ms |
| peak allocated above idle | 5.29 GiB | 9.72 GiB | **5.79 GiB** | 33.95 GiB | **18.27 GiB** |

### Did the expectation hold?

- **Memory: yes, and better than expected.** At `=1`, per-extra-stream growth fell from 3.50 GiB to 1.33 GiB (c4) or 1.46 GiB (c8). The expectation was about 1.8 GiB, but the transient isn't per stream.
  - With the fix, `=0` holds the same 1.33 GiB per stream. Its higher peak is only its larger one-call transient: about 17.6 GiB, against 3.05 GiB for `=1`.
  - So `=0` now fits concurrency 8 on this config: 57.9 GiB allocated, against an OOM before.
- **Timing: yes, within noise of after1.**
  - Rollout-level numbers are within 0.3% at kv c1 and c4, and within the earlier session's noise at c8 against today's old tree.
  - Chained c1 is +1.2%, carried by one slow round.
  - The decoder node itself is 1.5–2.8% slower per window. That is consistent with the memory-fix agent's +0.7 to +1.6 ms per chunk for the cache copy, plus cuDNN's heuristic algorithm pick now that `cudnn.benchmark` is off by default.
- **`=0` is still not worth it end to end.** Its time to first frame stays 0.14 s slower (1.53 against 1.39 s), and it is no faster: 11.94 s against 11.91 s total (kv c1), 17.50 against 17.52 fps (c4). The default of 1 is right.

## The concurrency-8 AssertionError (new today, not from the PR)

- **Error:** `diffusers/schedulers/scheduling_unipc_multistep.py:1214 assert self.this_order > 0`, from `Cosmos3DiTSubmodule.forward_batched` → `_scheduler_step` (`submodules.py:1942` → `:474`). The worker then fails every request in that DiT batch: 4 requests, one of which had already finished, so the conductor ignores that one.
- **What it means:** a request's UniPC scheduler was stepped past its last timestep. In each case the failing batch contains a request whose rollout had completed about 0.3 s earlier. It comes after a run of `Not enough free pages` KV admissions (the pool has 1024 pages) and `OOM on node=dit … holding k of 8 requests so the retry is smaller`. So the hold/retry path seems to step a request once more after its last window.
- **Not the decoder or the fix:**
  - The PR doesn't change the DiT, worker, engine or conductor relative to upstream main `99f5108b`. Its `submodules.py` diff starts at the decoder classes.
  - The **old PR tree `60fbc5cd` re-run today with the old after1 overlay** fails the same way: 3 of 8 streams, 13 `AssertionError` lines, the same as after1fix and after0fix.
  - In the earlier session the same tree completed 8/8, with about 280 page-admission warnings and 12–14 holds. Today there are about 260 warnings and 9 holds, so this is timing-dependent.
  - I did not re-run upstream main at c8 today.
- **Logs:** `logs/runs-c8-fix/*/server.log` and `logs/runs-c8-recheck/after1/server.log`.
- It is worth a separate issue against the DiT's KV-pressure retry path. It is outside this PR.

## Output equivalence

Method: one untimed dump request per variant and mode (seed 0, c=1) saved the raw uint8 frames and the latents `vae_decoder_ar` received. `cmp_fix.py` compares latents first. It then decodes, per distinct latent stream, one stock diffusers `AutoencoderKLWan.decode` of the whole de-duplicated stream in fp32 (TF32 off) and in bf16, and scores each frame set against the reference built from its own latents. This is the same method as `ref_decode.py`, which produced `e2e-cosmos3/reference.json`; I re-decoded because only scores, not frames, were kept. The re-decoded reference reproduces every old score exactly (before 46.52, after0 53.36, after1 53.46 dB in kv).

**Latents:**

| | kv stream md5 | chained stream md5 |
|---|---|---|
| before, after0, after1 (earlier session) | `111b540f…` | `41f13f25…` |
| **after1fix** | **`111b540f…` (bit-identical, every window)** | **`41f13f25…` (bit-identical)** |
| after0fix | `d349b697…` (max latent diff 9.2) | `a00abf27…` (max 8.8) |

- **after0fix's DiT output differs, but not because of the decoder.** `d349b697…` is exactly the kv latent stream of the overlap session's non-MPS launches (`~/overlap/runs/{coloc,split}`), which ran the old PR tree `60fbc5cd`. Its MPS launches produced `111b540f…`.
- This is the DiT's launch-dependent bimodality that `overlap/REPORT.md` describes: two deterministic outcomes, 13.6 dB apart in frames. Here after0fix against after1fix frames is 13.56 dB.
- The DiT code is upstream main's, and the decoder never feeds back into it.

**Frames against the fp32 whole-stream decode of their own latents** (PSNR whole clip / worst frame / max abs diff / values off by more than 1):

| | before | after1 (old) | **after1fix** | after0 (old) | **after0fix** (own latents) | stock bf16 whole-stream decode |
|---|---|---|---|---|---|---|
| kv | 46.52 / 34.46 / 138 / 5.4% | 53.46 / 49.52 / 39 / 1.2% | **53.46 / 49.43 / 36 / 1.2%** | 53.36 / 49.48 / 36 / 1.3% | **53.60 / 49.85 / 29 / 1.1%** | 53.42 / 49.18 / 38 (`111b`); 53.74 / 49.09 / 32 (`d349`) |
| kv worst frame per window | 52.7 50.7 37.5 47.7 34.5 41.6 38.8 40.3 | 52.7 50.7 49.5 55.5 51.4 51.6 52.3 53.5 | **52.8 50.7 49.4 55.6 51.4 51.5 52.4 53.5** | 52.0 50.7 49.5 55.5 51.4 51.7 52.3 53.5 | 52.0 51.3 49.9 55.3 51.2 52.8 51.8 54.0 | |
| kv first frame of windows 2–8 | 53.2 37.5 47.7 43.1 42.2 38.8 42.8 | 53.3 50.8 55.5 55.9 54.8 53.4 54.7 | **53.3 50.9 55.6 55.9 54.9 53.5 54.7** | 52.9 50.7 55.5 56.0 54.9 53.5 54.7 | 53.2 52.7 55.9 54.8 54.3 52.4 55.1 | |
| chained | 41.34 / 31.25 / 236 / 15.5% | 52.33 / 47.94 / 32 / 2.0% | **52.33 / 47.91 / 38 / 2.0%** | 52.25 / 47.87 / 32 / 2.1% | **53.40 / 50.92 / 36 / 1.0%** | 52.20 / 47.35 / 48 (`41f1`); 54.15 / 50.60 / 33 (`a00a`) |

**Frame against frame, same latents:**

- **after1fix vs old after1:** kv 57.81 dB, worst frame 54.13 dB, max 19 levels; chained 57.41 / 53.09 / 18. Not bit-identical. Old after1 ran with `cudnn.benchmark` on, which picks conv algorithms per process. The earlier report saw 59–61 dB (≤14 levels) between two launches of the same code with benchmark on, and old after0 vs old after1 was 57.1 dB (19 levels).
- **after1fix vs old after0:** kv 57.09 dB, max 19.
- **after0fix vs the overlap session's co-located old-tree `=1` frames** (same `d349b697` latents): 57.33 dB, worst frame 52.77 dB, max 20 levels. Against the overlap's split launch: 57.33 dB, max 25.

**Conclusion:** the fixed code decodes the stream exactly as well as before. after1fix's frames are as close to the fp32 whole-stream decode as old after1's, and as close as one stock bf16 decode of the whole stream, window by window and at every window start. No seam or quality change shows up.

## Files

- `results.csv` / `results.json`: after1fix and after0fix timed phases (kv c1, kv c4, chained c1), with per-node timings, memory by PID and nvidia-smi peaks. `analyze.py` with `VARIANTS="after1fix after0fix"`.
- `results_c8.csv` / `results_c8.json`: the c8 launches. `results_c8_recheck_oldtree.csv` / `.json`: the old tree's c8 re-run today.
- `frames_latents.json` (`cmp_fix.py` output): latent md5s per stream and window, frame-vs-frame comparisons, and every frame set against the fp32 and bf16 whole-stream decodes, with per-frame PSNR. `after0fix_vs_overlap_frames.json`: the same-latents check against the overlap session's frames.
- `logs/runs-fix/<variant>/`, `logs/runs-c8-fix/<variant>/`, `logs/runs-c8-recheck/after1/`: `run.log`, `server.log`, `client.log`, `client_<phase>.json`, `stats.txt`, `mem_<phase>.json`, `smi.csv`, `phases.jsonl`, `config.yaml` (as served), `startup.json`.
- `logs/run_fix.log`, `logs/run_c8_recheck.log`, `logs/analysis_fix.log`, and the `analysis-fix-*.log` outputs.
- `scripts/`: the scripts as run, i.e. the `e2e-cosmos3` scripts plus the two variant cases in `run_variant.sh`, `VARIANTS` in `analyze.py`, `run_fix.sh` and `cmp_fix.py`. `cmp_fix.py` does the latent comparison that the overlap work's `cmp_latents.py` did (stream md5 and per-window max diff), on this run's paths.
- Frames, mp4s and latents stay on the VM under `~/e2e-cosmos3/runs-fix*/` and `runs-c8-*/`.

## VM state

`wan-bench-b200-vllm` is **running**, with nothing of mine on it: no tmux sessions, no processes, GPU empty, no containers. New on the VM:

- `~/e2e-cosmos3/mstar-fix` (the `913295e4` archive);
- `runs-fix/`, `runs-c8-fix/`, `runs-c8-recheck/`, `analysis-fix/`;
- the added scripts, with `run_variant.sh.orig` and `analyze.py.orig` kept next to the edited copies.
