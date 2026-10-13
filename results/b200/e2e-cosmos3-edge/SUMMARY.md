# End-to-end Cosmos3 streaming rollout through the M* server: before vs after the Wan VAE decoder change (mstar-project/mstar#425)

## TL;DR

- **Speed.** Measured through the M* server on one B200, with the shipped `configs/cosmos3_edge.yaml` (every node co-located on one GPU), the PR cuts the streaming decoder from about 530 ms to about 240 ms per window. The rollout gets faster as a result:
  - **kv mode, 8 windows, concurrency 1:** 14.27 s → 12.01 s (16.9 → 20.1 fps).
  - **chained mode, 10 windows:** 11.85 s → 9.13 s (20.3 → 26.4 fps).
  - **concurrency 4:** aggregate throughput 14.8 → 17.3 fps.

  The decoder's share of wall time falls from 30% to 16% (kv) and from 39% to 20% (chained). The DiT is untouched: same time per window, and bit-identical latents in all three variants.
- **Output quality improves, it does not regress.** The frames differ from upstream main, so I scored all three variants against one fp32 decode of the whole latent stream (TF32 off, stock diffusers):
  - **the PR (both settings):** 53.4 dB, worst frame 49.5 dB. Stock diffusers' own bf16 whole-stream decode scores 53.4 dB, worst frame 49.2 dB.
  - **upstream main's stateless context re-decode:** 46.5 dB, worst frame 34.5 dB, max error 138 levels (kv). In chained mode it gets 41.3 dB, worst frame 31.3 dB, max error 236 levels.

  No new seams appear at window boundaries.
- **Key finding: `windowed_decode_latents_per_call=0` (the PR default) runs out of memory at concurrency 8; `=1` does not.**
  - At concurrency 8 with `=0`, 7 of 8 streams fail with `torch.OutOfMemoryError` in `decode_causal`, at 175.6 GiB allocated on the 178 GiB B200. With `=1`, all 8 complete at a 60.2 GiB peak; upstream main peaks at 34.1 GiB.
  - Concurrency 4 completes with `=0`, but at a 120.9 GiB peak (142 GB in nvidia-smi), against 45.4 GiB with `=1` and 34.0 GiB on upstream main.
  - **Cause:** under `torch.compile` (the serving default), each warm stream's decoder cache tensors are slices of much larger compiled-graph output buffers, and they keep those buffers alive. The cache adds up to the documented 1.33 GiB per stream, but the storage held per stream is 18.7 GiB with `=0` and 3.47 GiB with `=1`. Eager mode holds exactly 1.33 GiB with either setting.

  This is a fixable retention bug, not an inherent cost of the one-call decode. A decoder-only probe confirms it (see [Memory](#memory)).
- **Recommendation for the PR:**
  - compact the cache after each compiled `DecoderStep` call (clone any `feat_cache` entry whose storage is larger than the tensor itself);
  - until then, default `windowed_decode_latents_per_call` to `1`.

  End to end, `=1` was as fast as `=0` or slightly faster everywhere: kv total 11.88 s vs 12.01 s, chained 9.01 s vs 9.13 s, concurrency 4 17.5 vs 17.3 fps. `=0` also has a slower time to first frame: 1.61 s, against 1.38 s with `=1` and 1.43 s on upstream main.

GPU time: about 42 minutes on 1×B200 (`wan-bench-b200-vllm`). That covers the timed runs (20 min), a latent-capture pass (4.5 min), the reference decode (3.3 min), the concurrency-8 probe (8.4 min), the decoder-only cache probe (1 min), and two failed environment attempts (4.6 min, explained under [Environment](#environment)).

## Setup

| | |
|---|---|
| Config | `configs/cosmos3_edge.yaml` as shipped at each revision (model `cosmos3_edge`). Every node sits in one worker process on GPU 0: dit, reasoner, vision_encoder, vae_encoder, vae_decoder, vae_decoder_ar. `enable_windowed_video: true`; KV pool of 1024 pages; startup warm-up requests as shipped. Only the decoder switches below are overlaid into `model_kwargs`. The served yamls are in `logs/runs/<variant>/config.yaml`. |
| Why Edge | This is the smallest config with the windowed streaming decoder (`Cosmos3VAEDecoderARSubmodule`): 4B parameters, 8.6 GB checkpoint. It is co-located like `cosmos3_nano_ar.yaml`, and it is what `benchmark/wan_vae_decoder/e2e_stream.sh` serves. Nano (33 GB) would also fit on one GPU, but leaves about 25 GB less headroom. |
| Model | `nvidia/Cosmos3-Edge` @ `344d602b128d1bbdacb43b08d0a3626f46343e29` (current `main`). Public, not gated, license OpenMDW-1.1 (`license: other`, `openmdw1.1-license`). `nvidia/Cosmos3-Nano` @ `e59a53c2…` is also public and ungated. |
| before | upstream main `99f5108b50ccd2289b875cc67b2cf776bc7c445b`, no overrides (stateless re-decode of 8 context latents ahead of each window) |
| after0 | PR HEAD `60fbc5cdd4f648e058acb20cd37e6d0ca8c84d56` with `windowed_decode_warm_cache: true, vae_decode_channels_last: true, windowed_decode_latents_per_call: 0, vae_decode_cudnn_benchmark: true` (equal to the PR's defaults, set explicitly) |
| after1 | the same, with `windowed_decode_latents_per_call: 1` |
| Code on the VM | `git archive` of each revision from the PR worktree, so committed files only. The worktree itself was not touched. |
| Prompt | `A robot arm is cleaning a plate in the kitchen, smooth natural motion.` (the prompt of `benchmark/cosmos3/bench_stream_video.py`) |
| Image (i2v) | `nvidia/Cosmos3-Edge` `assets/example_i2v_input.jpg` (md5 `1dd31d4d20b8f25fe724d3326f3b1135`) |
| Request | `/generate`, `streaming=true`, `model_kwargs={"size":"832x480","num_frames":241,"num_inference_steps":20,"guidance_scale":6.0,"fps":24.0,"seed":0,"window_mode":"kv"\|"chained","window_frames":29,"stream_video":true}`. Concurrent streams use seeds 0..N-1. |
| Windows | kv: 8 windows of 8 latents with no overlap (chunks of 29+6×32+20 frames, 64 latents padded from 61). chained: 10 windows with 2 latents of overlap (29+8×24+20 frames, 62 latents). |
| Phases per server | kv at c=1 (1 warm-up + 3 timed rollouts) · kv dump at c=1 (untimed, raw frames saved) · kv at c=4 (1 warm-up round + 2 timed rounds) · chained at c=1 (1 warm-up + 2 timed) · chained dump. A separate launch per variant ran kv at c=8: a c=1 warm-up, then 1 round of 8 concurrent streams. Peak-memory counters are reset between phases. |

### Environment

- **Software:** a fresh venv on the VM (`~/e2e-cosmos3/venv`) with torch 2.12.1+cu130, diffusers 0.39.0, flashinfer-python 0.7.1, transformers 5.19.0, torchcodec 0.17.0, mooncake-transfer-engine 0.3.13.post1, cuDNN 9.20.0.48, Python 3.12, driver 580.173.02.
- **Why not `~/samples/venv`:** it lacked flashinfer, mooncake, torchaudio and torchcodec, and I didn't want to change a venv another agent uses.
- **Deviation 1: no fa3-fwd.** The `fa3-fwd` 0.0.3 wheel (pulled in by `.[cosmos3]`) only has Hopper (sm90) kernels. The first DiT forward on the B200 aborted the worker with `no kernel image is available for execution on the device`. I uninstalled the wheel, so M* takes its documented fallback: `Attention backend 'dense' requested but the fa3-fwd kernel is unavailable … using the paged 'flashinfer' backend instead.` All three variants run the same DiT attention.
- **Deviation 2: flashinfer JIT.** flashinfer JIT-builds its kernels with ninja and nvcc, so the server runs with `venv/bin` and `/usr/local/cuda/bin` on `PATH`.
- **Measurement hook (not decoder code):** `scripts/hook/sitecustomize.py` sits first on the server's `PYTHONPATH`. Every 0.5 s, each CUDA process writes `torch.cuda.max_memory_allocated` and `max_memory_reserved` to a file, and a marker file resets the peaks between phases. Only while a dump flag is set (the untimed dump requests), it also saves the raw uint8 chunks before mp4 encoding and the latents handed to `vae_decoder_ar`.
- **The hook doesn't change the output:** the dump runs' mp4s are md5-identical to the timed rounds' mp4s for every variant. Its pass-through wrapper does show up in the concurrency-8 OOM traceback.
- **Other samples:** nvidia-smi sampled every 500 ms; `--log-stats-file` gives the per-node, per-walk CPU timings (DiT vs decoder).

## Results

Medians over the timed rollouts. "Window gap" is the time between consecutive streamed chunks. "DiT / decoder per window" is the node's total time over the request divided by the window count, from `--log-stats` (CPU-timed; the decoder node also runs a last no-op pass). Decoder share = decoder time ÷ wall time. At c ≥ 4 it is decoder time summed over the round's streams ÷ the round's wall time, the GPU-time share; the per-stream figure is in `results.csv`.

### kv mode, 241 frames, 8 windows

| | before | after (=0, PR default) | after (=1) |
|---|---|---|---|
| **c=1** time to first frame | 1.43 s | 1.61 s (+0.18 s) | **1.38 s** |
| c=1 window gap (median / p90) | 1.89 / 1.90 s | 1.53 / 1.56 s | 1.55 / 1.57 s |
| c=1 total rollout | 14.27 s | 12.01 s (−15.8%) | **11.88 s (−16.7%)** |
| c=1 delivered fps | 16.9 | 20.1 | **20.3** |
| c=1 DiT / decoder per window | 1287 / 534 ms | 1294 / 244 ms | 1287 / 232 ms |
| c=1 decoder share of wall time | 29.9% | 16.2% | 15.6% |
| **c=4** time to first frame (median over 8 streams) | 6.08 s | 6.36 s | 5.94 s |
| c=4 window gap per stream | 8.55 s | 7.13 s | 7.18 s |
| c=4 rollout per stream / round wall | 64.2 / 65.0 s | 55.2 / 55.6 s | 54.8 / 55.1 s |
| c=4 fps per stream / aggregate | 3.75 / 14.8 | 4.37 / 17.3 | 4.40 / **17.5** |
| c=4 decoder per window | 880 ms | 295 ms | 261 ms |
| c=4 decoder share of round GPU time | 43.3% | 17.0% | 15.2% |
| **c=8** streams completed | 8/8 | **1/8 (7 OOM)** | 8/8 |
| c=8 time to first frame / per-stream rollout (median) | 11.8 / 103.5 s | 14.3 / 35.2 s (the 1 survivor) | 11.6 / 88.6 s |
| c=8 round wall / aggregate fps | 128.8 s / 15.0 | 35.2 s / 6.8 | 109.5 s / **17.6** |
| c=8 decoder share of round GPU time | 41.9% | n/a | 21.6% |

### chained mode, 241 frames, 10 windows, c=1

| | before | after (=0) | after (=1) |
|---|---|---|---|
| time to first frame | 1.35 s | 1.53 s | **1.32 s** |
| window gap (median / p90) | 1.17 / 1.19 s | 0.85 / 0.87 s | 0.86 / 0.87 s |
| total rollout | 11.85 s | 9.13 s (−22.9%) | **9.01 s (−24.0%)** |
| delivered fps | 20.3 | 26.4 | **26.8** |
| DiT / decoder per window | 744 / 467 ms | 757 / 183 ms | 753 / 173 ms |
| decoder share of wall time | 39.4% | 20.0% | 19.2% |

**Run-to-run spread:** every variant was deterministic within its server. At c=1, total rollout time varied by at most 0.04 s across the timed kv rollouts and 0.05 s across the chained ones. The DiT's time per window is the same in all three variants at every concurrency.

**Concurrency doesn't buy throughput here:** going from c=1 to c=4 to c=8 barely raises aggregate fps (about 17.5 at best). The DiT already saturates the GPU at c=1 (20 steps with guidance, about 12k tokens per window), so concurrent streams mainly share it.

**Time to first frame with `=0`:** it is 0.18–0.23 s slower than both upstream main and `=1`, in both modes. From reading the code, the likely reason is this. `DecoderStep` runs a stream's first two decoder calls eagerly. With `=0`, the second call is the first window's whole 7-latent decode, so it runs uncompiled, while upstream main runs that window through a compiled `vae.decode`. With `=1`, only one latent runs eagerly after the clip-start latent.

## Memory

The worker process (all nodes) holds **28.5 GiB allocated at idle**: weights, the 1024-page KV pool and CUDA graphs. "Peak" is `torch.cuda.max_memory_allocated` in that process during the phase. nvidia-smi values are the phase maximum over 0.5 s samples, for the whole GPU. The B200 reports 178.35 GiB total capacity to PyTorch.

| phase | before | after (=0) | after (=1) |
|---|---|---|---|
| kv c=1: peak allocated (above idle) | 33.8 GiB (+5.3) | 64.7 GiB (+36.1) | 34.8 GiB (+6.3) |
| kv c=1: peak reserved / nvidia-smi peak | 35.8 GiB / 37.8 GB | 79.0 GiB / 82.0 GB | 38.2 GiB / 40.3 GB |
| chained c=1: peak allocated (above idle) | 33.8 GiB (+5.3) | 62.5 GiB (+33.9) | 38.3 GiB (+9.7) |
| kv c=4: peak allocated (above idle) | 34.0 GiB (+5.4) | 120.9 GiB (+92.3) | 45.4 GiB (+16.8) |
| kv c=4: peak reserved / nvidia-smi peak | 36.2 GiB / 38.2 GB | 137.8 GiB / 142.3 GB | 48.6 GiB / 50.9 GB |
| kv c=8: peak allocated (above idle) | 34.1 GiB (+5.5) | **175.6 GiB (+147.1), OOM** | 60.2 GiB (+31.7) |
| kv c=8: peak reserved / nvidia-smi peak | 36.3 GiB / 38.3 GB | 176.9 GiB / 182.1 GB | 64.4 GiB / 67.1 GB |
| OOM | none | **7 of 8 streams at c=8**; none at c ≤ 4 | none |

**The OOM, exactly** (`logs/runs-c8/after0/server.log`, 23 `out of memory` lines):

```
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 1.62 GiB. GPU 0 has a total capacity of 178.35 GiB
of which 489.75 MiB is free. Including non-PyTorch memory, this process has 177.86 GiB memory in use. Of the allocated
memory 175.56 GiB is allocated by PyTorch, with 120.75 MiB allocated in private pools (e.g., CUDA Graphs), and
1.18 GiB is reserved by PyTorch but unallocated.
  submodules.py:2826 Cosmos3VAEDecoderARSubmodule.forward -> pixels = self._decode_stream(new, feat_cache, clip_start=False)
  submodules.py:2895 _decode_stream -> wan_vae.py:173 decode_causal -> step(x[:, :, i : i + per_call], feat_cache)
```

The worker survived: one stream finished, and the other 7 returned the OOM as request errors.

**Why `=0` grows with streams:** the peak above idle is linear in the number of live streams: about 18.7 GiB per extra stream with `=0` and 3.5 GiB with `=1`, while upstream main stays flat at about 5.4 GiB. Allocation returns to idle when the streams end, so the memory is held per live stream between windows. `scripts/probe_cache.py` confirms this on the PR's decoder node alone: 4 streams, 8-latent kv windows at 832x480, random latents, bf16, `cosmos3_edge` VAE. Results are in `logs/probe/probe.jsonl`:

| decoder step | cache tensor bytes per stream | storage kept alive per stream (after window 1) | peak allocated, 4 streams |
|---|---|---|---|
| `=0`, compiled (served default) | 1.33 GiB | **18.70 GiB** | 82.6 GiB |
| `=1`, compiled | 1.33 GiB | 3.47 GiB | 17.8 GiB |
| `=0`, eager | 1.33 GiB | 1.33 GiB | 26.2 GiB |
| `=1`, eager | 1.33 GiB | 1.33 GiB | 8.9 GiB |

Under `torch.compile`, the conv-cache entries the compiled `WanDecoder3d.forward` writes back into `feat_cache` are views into larger output buffers, roughly a whole call's activation. Each live stream therefore pins those buffers until it ends. With `=0` a call covers a window's 6 new latents (24 frames), so far more is pinned than with `=1` (4 frames).

**Possible fix (not tested):** in `DecoderStep.__call__` (`mstar/model/components/wan_vae.py`), after the compiled call, replace any `feat_cache[i]` whose `untyped_storage().nbytes()` exceeds `numel() * element_size()` with `feat_cache[i].clone()`. Each stream would then hold its documented 1.33 GiB.

The eager rows show the real one-call transient: about 24 GiB for `=0` against about 4 GiB for `=1` over 4 interleaved streams. That transient is per call, not per stream. The extrapolation that a fixed `=0` would fit c=8 on this config is mine; it was not measured.

The PR's config comment quotes a 35.3 GiB peak for `=0` on one stream from the decoder-only bench. That figure likely includes this retention as well as the transient.

## Output equivalence

**Method:**

- One untimed request per variant and mode (seed 0, c=1) dumped the raw uint8 frames.
- A second server launch per variant dumped the latents `vae_decoder_ar` received. They are **bit-identical across before, after0 and after1** in both modes (kv stream `[1,48,64,30,52]`, md5 `111b540f…`; chained `[1,48,62,30,52]`, md5 `41f13f25…`). Every frame difference below therefore comes from the decoder.
- `scripts/ref_decode.py` decodes the whole de-duplicated latent stream once with stock diffusers `AutoencoderKLWan.decode`: in fp32 with TF32 off (the reference), and in bf16 (the bar for "same as one decode in bf16").

All 241 frames of every variant differ from the reference by at least one value; none is bit-exact.

| vs fp32 whole-stream decode | before | after (=0) | after (=1) | stock bf16 whole-stream decode |
|---|---|---|---|---|
| kv: PSNR, whole clip | 46.52 dB | 53.36 dB | 53.46 dB | 53.42 dB |
| kv: worst frame (index) | **34.46 dB (#127)** | 49.48 dB (#64) | 49.52 dB (#64) | 49.18 dB |
| kv: median frame | 51.25 dB | 53.94 dB | 53.92 dB | |
| kv: max abs diff / values off by >1 | 138 / 5.4% | 36 / 1.3% | 39 / 1.2% | 38 |
| kv: worst frame per window | 52.7 50.7 37.5 47.7 34.5 41.6 38.8 40.3 | 52.0 50.7 49.5 55.5 51.4 51.7 52.3 53.5 | 52.7 50.7 49.5 55.5 51.4 51.6 52.3 53.5 | |
| kv: first frame of windows 2–8 | 53.2 37.5 47.7 43.1 42.2 38.8 42.8 | 52.9 50.7 55.5 56.0 54.9 53.5 54.7 | 53.3 50.8 55.5 55.9 54.8 53.4 54.7 | 52.7 50.1 56.4 56.9 54.9 53.5 55.0 |
| chained: PSNR, whole clip | 41.34 dB | 52.25 dB | 52.33 dB | 52.20 dB |
| chained: worst frame (index) | **31.25 dB (#53)** | 47.87 dB (#207) | 47.94 dB (#207) | 47.35 dB |
| chained: max abs diff / values off by >1 | 236 / 15.5% | 32 / 2.1% | 32 / 2.0% | 48 |

- **The PR decodes the stream as accurately as one stock bf16 decode of the whole stream.** That is what the warm cache claims ("the frames are those of one decode of the whole stream").
- **Upstream main's error comes from its approximation.** The stateless path re-decodes only 8 latents of context, so each window starts from a truncated conv state. The error starts at the window's first frame and runs through the window, from window 3 on in both modes (worst windows: kv window 5 at 34.5 dB, chained window 3 at 31.3 dB).
- **after0 vs after1:** 57.1 dB (kv) and 56.5 dB (chained), max 19 and 28 levels; both are equally close to the reference.
- **Seam check:** I compared the mean absolute frame-to-frame change across each window boundary with the neighbouring in-window changes. The ratios are the same in all three variants to within 0.05: kv 3.33 1.22 1.44 0.81–0.83 1.03 0.88 0.96. So they track the content (the first boundary is a fast-motion moment in the scene), and the PR introduces no seam. The reference comparison at window-start frames (rows above) shows the PR matching the reference at boundaries (50.7–56.0 dB in kv), where upstream main drops to 37.5 dB.
- **md5 of the full 241-frame uint8 clip:**

  | | before | after0 | after1 |
  |---|---|---|---|
  | kv | `cf77efe4…` | `3ba75512…` | `98681929…` |
  | chained | `f9311827…` | `b3979e9a…` | `d1a9f5ea…` |

- **Determinism:** upstream main is bit-identical across two server launches. The after-variants are bit-identical within a launch (all rounds and the dump), but differ between launches by 59–61 dB, at most 14 levels. That fits `vae_decode_cudnn_benchmark` letting cuDNN pick conv algorithms per process. Expect it in A/B comparisons across restarts.

## Caveats

- **Edge, not Nano.** This is the smallest windowed config. A co-located Nano (`cosmos3_nano_ar.yaml`, about 25 GB more weights) has less headroom, so the `=0` OOM would come at the same number of streams or fewer. That is an extrapolation, not measured.
- **DiT attention ran on flashinfer paged, not fa3,** because the fa3-fwd wheel lacks sm100 kernels. That affects the DiT's absolute time and therefore the decoder's share, not the decoder numbers.
- **Node timings are CPU wall spans** from `--log-stats`. At c ≥ 4 they include waiting on the shared GPU thread, which is why the decoder's time per window rises with concurrency.
- **Concurrency 8 got one round** (no warm-up round at c=8; a c=1 warm-up preceded it), so treat its latencies as indicative. The OOM and memory figures are the point of that run.

## Commands

On the VM (`ssh bench@<vm-ip>`, working dir `~/e2e-cosmos3`; scripts as in `scripts/`):

```bash
# environment
uv venv --python 3.12 --seed venv
uv pip install --python venv/bin/python --torch-backend=cu130 "torch==2.12.1" torchvision torchaudio "diffusers==0.39.0" "./mstar-pr[cosmos3]"
uv pip uninstall --python venv/bin/python mstar-ai fa3-fwd        # code via PYTHONPATH; fa3-fwd has no sm100 kernels
HF_HOME=~/e2e-cosmos3/hf hf download nvidia/Cosmos3-Edge --revision 344d602b128d1bbdacb43b08d0a3626f46343e29
printf 344d602b128d1bbdacb43b08d0a3626f46343e29 > hf/hub/models--nvidia--Cosmos3-Edge/refs/main   # served with HF_HUB_OFFLINE=1

# runs (each waits until nothing else is on the GPU, then before -> after0 -> after1, one server at a time)
bash scripts/run_all.sh                                   # timed phases        -> runs/
RUNS=runs-latents PHASES=dump bash scripts/run_all.sh     # latent capture      -> runs-latents/
RUNS=runs-c8 PHASES=c8 bash scripts/run_all.sh            # concurrency 8       -> runs-c8/
venv/bin/python scripts/analyze.py runs analysis && venv/bin/python scripts/analyze.py runs-c8 analysis-c8
HF_HOME=$PWD/hf HF_HUB_OFFLINE=1 venv/bin/python scripts/ref_decode.py ~/e2e-cosmos3 analysis
MSTAR=$PWD/mstar-pr HF_HOME=$PWD/hf venv/bin/python scripts/probe_cache.py --set windowed_decode_latents_per_call=0   # also =1, and --eager
```

The server, as `scripts/run_variant.sh` launches it (`MSTAR` = `mstar-main` for before, `mstar-pr` for after0 and after1):

```bash
CUDA_VISIBLE_DEVICES=0 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True COSMOS3_GEN_CAPTURE_RES=192x320,480x832,640x640 \
PATH=$E/venv/bin:/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda HF_HOME=$E/hf HF_HUB_OFFLINE=1 \
PYTHONPATH=$E/hook:$MSTAR E2E_MEM_DIR=$R/mem E2E_DUMP_DIR=$R/dump \
python $MSTAR/mstar/api_server/entrypoint.py --config $R/config.yaml --socket-path-prefix $(mktemp -d)/ \
  --upload-dir $(mktemp -d)/ --port 8100 --mooncake-port 9100 --tensor-comm-protocol SHM --log-stats-file $R/stats.txt
```

The client (`scripts/stream_client.py`, adapted from `benchmark/cosmos3/bench_stream_video.py` with concurrency and per-chunk timestamps added), for example:

```bash
python scripts/stream_client.py --port 8100 --image <snapshot>/assets/example_i2v_input.jpg --mode kv --frames 241 \
  --concurrency 4 --warmup 1 --rounds 2 --save-dir $R/mp4/kv_c4 --out $R/client_kv_c4.json
```

`benchmark/wan_vae_decoder/e2e_stream.sh` was the template: same config, same overlay mechanism, same server environment. I didn't use `bench_decode_nodes.py` itself because it drives the node on synthetic latents; `probe_cache.py` builds the node the same way.

## Files

- `results.csv` and `results.json`: every phase for c=1, c=4 and chained, with per-node timings, memory by PID, and before-vs-after frame comparisons including per-frame PSNR.
- `results_c8.csv` and `results_c8.json`: the concurrency-8 runs.
- `reference.json`: latent identity, and every variant against the fp32 and bf16 whole-stream decodes (per-frame PSNR, per-window maxima).
- `logs/runs/<variant>/`: `run.log`, `server.log`, `client.log`, `client_<phase>.json` (every chunk's arrival time and frame count), `stats.txt` (`--log-stats`), `mem_<phase>.json`, `smi.csv`, `phases.jsonl`, `config.yaml`.
- `logs/runs-latents/`, `logs/runs-c8/`, `logs/probe/`, `logs/run_all*.log`, `logs/setup.log`, and `logs/attempts/` (the two failed environment attempts).
- `scripts/`: everything used, including `hook/sitecustomize.py`.
- Frames, mp4s and latents stay on the VM under `~/e2e-cosmos3/runs*/<variant>/{dump,mp4}`; nothing was committed.
- The VM is still running with nothing of mine on it: no tmux session, no processes, GPU empty. The venv and HF cache remain in `~/e2e-cosmos3`.
