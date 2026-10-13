# Wan VAE warm-decode memory: H200 census notes (H200 DIAGNOSTIC ONLY)

**H200 diagnostic only.** Every number here comes from the local H200. Treat the
ms values as H200-only; B200 numbers belong to the agent on the SGLang B200 VM,
which owns the fix, `fix.patch` and `SUMMARY.md`. Allocation sizes and the tensor
census don't depend on the GPU, so they carry over to B200.

- Code: pushed PR head `daa44149` (`perf/wan-vae-decode`, fork `origin`),
  exported with `git archive` to `${WAN_BENCH}/memfix/base` (untouched) and
  `${WAN_BENCH}/memfix/src` (candidate fix). Nothing in `wt-wan-decoder` was
  edited, committed or pushed.
- Env: `${MSTAR}/.venv` (torch 2.12.1+cu132, diffusers 0.39.0),
  Wan 2.1 VAE, latents `${WAN_BENCH}/sglang/inputs/wan21/latents.pt`, 832x480.
- Probe: `${WAN_BENCH}/memfix/probe.py` (same protocol as `mem_probe.py`:
  latent 0, 10 warm-up chunks, census, peak reset, 5 measured chunks; then the
  remaining 70 chunks timed; md5 over every uint8 chunk). Runner
  `${WAN_BENCH}/memfix/run1.sh` (fresh process, fresh compile caches).
  Raw output: `${WAN_BENCH}/memfix/runs/` (`census*.json`, `q1.jsonl`).
- Only Wan 2.1 at `windowed_decode_latents_per_call=1` was measured. Wan 2.2 and
  `=0/2/3` were not run before I was told to stop.

## 1. Census between chunks (PR code, Wan 2.1, `=1`, compiled)

`gc` over CUDA tensors, grouped by storage `data_ptr`, plus
`torch.cuda.memory_snapshot()` (with allocation history recorded):

| what | tensors | storages | logical | storage |
|---|---:|---:|---:|---:|
| decoder feat cache (`ar_feat_cache`) | **32** | 32 | **1.759 GiB** (= `feat_cache_bytes`) | **4.526 GiB** |
| `ar_tail` (8 context latents, fp32) | 1 | 1 | 3.0 MiB | 4.2 MiB |
| compile-time FakeTensors kept by dynamo/inductor | 299 | 1 (64 B) | "0.28 GiB" (not real) | ~0 |
| allocator blocks not backing any Python-visible tensor | — | 0 | — | 0 |

**332 vs 32 is mostly a counting artifact.** `mem_probe.py`'s `held_tensors`
counts 299 compile-time FakeTensors that share a 64-byte storage (accessing
`data_ptr()` on them warns "FakeTensor"). Their 0.28 GiB of "logical" bytes is
not GPU memory, and they explain the 2.04 vs 1.76 GiB logical gap.

M* holds exactly 32 cache entries, 1.759 GiB logical, the same as vLLM-Omni and
SGLang (1.76 GiB). There are no duplicate slots, no stale entries from the eager
first calls, no surviving warm-up copy (`DecoderStep`'s scratch clone is freed),
no pixel buffers (`stream_video` delivers each window, so `ar_pixels` stays
empty), and no compiled-graph static inputs or other hidden allocations:
`memory_snapshot` showed 0 active blocks outside Python-visible storages.

**The waste is the storage behind those 32 entries: 4.53 GiB for 1.76 GiB of
data.** Per entry, at `=1`:

| entries | shape | logical | storage | ratio |
|---|---|---:|---:|---:|
| 25–30 | [1, 96, 2, 480, 832] | 146.2 MiB | 438.8 MiB | 3x (6 frames) |
| 19–24 | [1, 192, 2, 240, 416] | 73.1 MiB | 219.4 MiB | 3x |
| 13–17 | [1, 384, 2, 120, 208] | 36.6 MiB | 73.1 MiB | 2x |
| 12 | [1, 192, 2, 120, 208] | 18.3 MiB | 36.6 MiB | 2x |
| 0–11, 18, 31 | low-res, conv_out, etc. | — | compact | 1x |

The view strides give it away: entry 30 has stride `[153354240, 1, 38338560, 79872, 96]`.
Its dim-0 stride is 6 frames for a 2-frame tensor, and `_base` is `None`
(inductor's `reinterpret_tensor`), so the entry doesn't look like a Python view.
At `=0` the buffers are 12+ frames, which is why held state reaches 10.2 GiB there.

## 2. Root cause

Diffusers caches `x[:, :, -2:].clone()`. M*'s activations are channels-last
(`to_channels_last`) at batch 1. With that layout the T-slice is dense, so
`clone()` (`preserve_format`) gets exactly the source's strides. Inductor's
`remove_noop_ops` (`torch/_inductor/fx_passes/post_grad.py`:
`register_noop_decomp([aten.alias, aten.clone])` plus a `same_meta` check on
size, stride and dtype) then deletes the clone as a no-op.

The compiled graph returns each cache slot as a view into the activation. That
activation is usually the fused conv-input buffer `cat([old cache, x])` (6
frames at `=1`). The view keeps the whole buffer alive until the next call
replaces the slot.

With contiguous (non-channels-last) activations the slice's strides differ from
a compact clone's, so the clone survives. That's why stock diffusers, vLLM-Omni
and SGLang hold compact caches, and why eager M* calls are compact too.

CPU repro (`${WAN_BENCH}/memfix/noop_cpu.py`): a compiled `silu` plus
`cache[0] = y[:, :, -2:].clone()` on a [1, 8, 6, 5, 7] input gives:
- contiguous input: storage 560 elements = logical 560;
- channels-last input: storage 1680 elements for 560 logical (the whole
  6-frame buffer).

## 3. Candidate fix (prototype in `${WAN_BENCH}/memfix/src`, diff vs `base`)

In `mstar/model/components/wan_vae.py`, compile a wrapper instead of
`decoder.forward`. After the decoder call, the wrapper re-clones every tensor
slot with an explicit memory format: `channels_last_3d` if the slot is
`is_contiguous(memory_format=channels_last_3d)`, else `contiguous_format`.

The explicit format gives compact strides. Where a slot is an oversized view the
strides differ, so inductor keeps the clone. Where a slot already owns its
storage (cat outputs, the contiguous mid-block entry 3) the strides match, so
inductor drops the clone again. Eager calls are untouched; diffusers' clone is
already compact there. `DecoderStep.__call__` is unchanged, and the existing
fake-`_fn` test still passes.

Test added to `test/modular/test_wan_vae_decode.py`:
`test_compiled_channels_last_stream_cache_owns_its_storage[wan21|wan22]`. It runs
a tiny channels-last VAE through `DecoderStep(compile=True)` on CPU and asserts
that every slot's `untyped_storage().nbytes() == numel * element_size`, that
storages are distinct, and that their sum equals `feat_cache_bytes`. It also
checks frames against the eager step with `atol=1e-5`. It **fails on the PR code
and passes with the fix**. The whole file passes: 24 tests, of which the new
test takes ~26 s of CPU compile.

### Measured effect (H200, Wan 2.1, `=1`, compiled, one process each)

| code | held between chunks (alloc) | cache storage | peak | transient (one chunk) | ms per 12 frames (median of 70) | md5 |
|---|---:|---:|---:|---:|---:|---|
| PR `daa44149` (run with allocation history on) | 4.39 | 4.53 | 8.23 | 3.46 | 101.97 (history on) | `b5f74b90…` |
| PR `daa44149` | 4.39 | 4.53 | 8.23 | 3.46 | **102.42** | `52353c01…` |
| clone fix | 1.63 | **1.76** | **6.57** | 4.56 | 104.75 | `1e777467…` |
| clone fix | 1.63 | 1.76 | 6.57 | 4.56 | **104.72** | `52353c01…` |
| in-place `copy_` into old slots (rejected) | 1.62 | 1.76 | 7.31 | 5.31 | 105.83 | `20c9f4f7…` |

All values are GiB except ms. The PR's 8.23 GiB peak matches the B200 probe's 8.23.

Takeaways:
- **Held state is fixed:** cache storage equals `feat_cache_bytes` (1.76 GiB),
  the same as vLLM-Omni and SGLang. Peak drops by 1.66 GiB at `=1`. At `=0` the
  saving should be about 8 GiB (10.2 GiB held down to about 2), but I didn't
  measure that.
- **md5 isn't stable between processes even on unchanged PR code**
  (`b5f74b90` vs `52353c01`), probably because `cudnn.benchmark` picks
  algorithms per process. The fix produced `52353c01`, identical to a PR run, so
  it's exact in that pair. For a clean exactness check, compare with
  `vae_decode_cudnn_benchmark=False` (deterministic), at every `=N`.
- **Cost: about +2.3 ms per chunk at `=1` on H200 (+2.2%), above the ~0.5% run
  noise.** That matches the bandwidth of copying ~1.5 GiB of cache per decoder
  call (read plus write), times 3 calls per chunk. At `=0` (one call per chunk)
  expect about a third of that. Before the fix the cache cost nothing because
  the copy was elided; eager diffusers pays the same copy.
- **Transient went up, 3.46 to 4.56 GiB.** The compiled graph holds the old
  cache (a graph input) for the whole call while it allocates the new compact
  slots. Inductor also schedules the cache-producing kernels late. In the kept
  output code (`${WAN_BENCH}/memfix/cache/2003105/i/z3/cz3vacrw….py`) the decoder
  runs as several graphs (the largest has 6 convs), and the slot-producing
  `fused_cat_select_slice_unsqueeze` kernels launch after the last conv, just
  before `return`. So the activation buffers each clone reads stay alive to the
  end of the graph. The in-place `copy_` variant made this worse: the
  functionalized copies also land at graph end.

## 4. Open ideas (not tried)

- **Clone at the cache write, not at graph end.** Placing the compaction right
  after its producer would let each conv-input buffer be freed right after its
  conv. That should bring the transient down toward eager's (~1.7 GiB). Options:
  - a `list` subclass for `feat_cache` whose `__setitem__` compacts (check that
    dynamo inlines it without graph breaks);
  - subclassing the diffusers blocks the way `to_channels_last` already swaps
    `__class__` (more code);
  - checking whether a graph break inside the decoder causes the end-of-graph
    placement (`TORCH_LOGS=graph_breaks`).
- **If +2% at `=1` is too much:** at `=0` the copy is a third as large per chunk.
  Re-decide the `windowed_decode_latents_per_call` default from B200 numbers with
  the fix applied.
- **Measure** `=0/2/3`, Wan 2.2, the served window (`bench_decode_nodes.py`,
  8 latents, 6 new) and 4 streams, all on B200.

## 5. State of this machine

- I killed my `memfix` tmux session; no process of mine is on the GPU. My last
  job ended at 20:18:41 UTC.
- **A `memfix` session I did not start** appeared at 20:18:47 UTC (pids 2003753,
  2003758, 2003781, same user). It runs `runq.sh q2.txt`, a 32-run before/after
  matrix over Wan 2.1/2.2 and `=0..3`, written into my scratch dir
  `${WAN_BENCH}/memfix` with my scripts, and was on the H200 (518 MiB) when I
  checked. Probably the duplicate launch of this task. I left it alone.
