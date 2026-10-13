# Design

The streaming decoder (`Cosmos3VAEDecoderARSubmodule`) decodes each window statelessly at the decode level. It re-decodes the last 8 latents of context from an empty causal-conv cache, then drops the context pixels. This is the same left-context pattern M\* uses for other causal decoders (`LeftContextChunkPolicy` for the Qwen3-Omni vocoder). Nothing large stays on the GPU between windows, and every window is self-contained, but each window runs 3.4× the useful conv work. This PR adds an optional warm path that sits next to the stateless decode, which remains the fallback.

![Before and after: per-window decode](design_before_after.png)

*Before, each window decodes 8 context latents plus its new ones from an empty cache. After, it decodes only the new latents, continuing the cache the previous window left.*

**Warm path.** With `windowed_decode_warm_cache: true`, a stream's first window runs the same decode, but against a feature cache the node owns (`new_feat_cache` and `decode_causal` in `mstar/model/components/wan_vae.py`). The cache holds the last frames each causal conv saw. It lives in the request's state as `ar_feat_cache`, next to the context tail and window counter the node already keeps there. Each later window decodes only its new latents and updates the cache in place.

The output equals one decode of the whole stream: bit-identical with `windowed_decode_latents_per_call: 1`, and within one 8-bit level at the default `0` (one decoder call per window). The cost is 1.76 GiB (Wan 2.1) or 1.33 GiB (Wan 2.2) per live stream at 832×480.

![Cache lifecycle](design_cache_lifecycle.png)

*A stream's cache lives exactly as long as its request on one decoder copy. A resumed session takes the stateless re-decode for one window.*

**Lifecycle.**
- **Created** on the stream's first window.
- **Freed** when the last window is decoded. Otherwise it goes with the rest of the request state when the engine removes the request (`cleanup_request`), which covers abort, failure and normal teardown. On shutdown, process exit frees it.
- **Sessions are unchanged.** The `SessionStore` keeps only the 8-latent tail, never the cache. A `resume_session` request's first window is the stateless re-decode behind the stored tail; later windows run warm.
- **Off switch.** `windowed_decode_warm_cache: false` restores the stateless decode for every window.

**Routing.** Before and after this PR, the conductor binds each request to one copy of every component at admission and keeps that binding for the request's whole life. So every window of a stream reaches the same decoder copy and finds its cache. Windows never move between copies: if a decoder copy is lost, its in-flight requests fail and the client retries or resumes the session. Moving a stream mid-rollout isn't supported on either path, because the context tail and window counter are per-copy as well.

![Multi-pod routing](design_multi_pod.png)

*Sticky binding keeps a stream on one decoder copy. Losing that copy fails the stream's request; nothing falls back mid-stream.*

**Multi-node.** Placement is set by `node_groups`. The decoder node can share the DiT's GPU (as in `cosmos3_nano_ar.yaml`), or run in its own pod with one or more copies. The cache never leaves the decoder GPU. What crosses pods per window doesn't change: window latents in (about 1–2 MB) and uint8 frames out (about 29 MB per 24 frames at 832×480). A decoder copy's memory grows with its number of live streams, so capacity planning should count concurrent windowed streams per copy.

**Configuration.** These are `Cosmos3Config` fields, set under `model_kwargs` in the YAML like every other Cosmos3 model switch:
- `windowed_decode_warm_cache` (default `true`)
- `windowed_decode_latents_per_call` (`0`)
- `vae_decode_channels_last` (`true`)
- `vae_decode_cudnn_benchmark` (`true`)

There are no new CLI flags.

```yaml
model_kwargs:
  enable_windowed_video: true
  windowed_decode_warm_cache: false   # stateless decode for every window
```

---

# Findings for the author (not for the PR body)

Compared: `upstream/main` at `99f5108b` with `wt-wan-decoder` HEAD at `523da8d1`. The K8s behaviour comes from the private worktrees `wt-k8s-live` (`fd632d5d`), `wt-k8s-dist`, `wt-k8s-rfc`, `wt-k8s-mgpu` and `wt-k8s-bench`. None of those ran Cosmos3 across pods; everything about Cosmos3 on K8s below is inferred from the code.

## Correction to the framing: the "stateless" decoder already had per-stream state

The header says "streaming Wan VAE decode (STATELESS)", but on `upstream/main` the node already keeps per-request state on the copy that serves the stream (`Cosmos3VAEDecoderARSubmodule.forward`, via `NodeSubmodule.request_state(rid)`):
- `ar_chunks`: the window counter that detects the last window
- `ar_tail`: the latent context to re-decode
- `ar_pixels`: the assembled video
- `ar_emitted`: the streamed frame count

So "any decoder copy can serve any window" was never true. "STATELESS" means no engine-managed resources and no decoder conv cache, and every window's decode starts from an empty cache. The warm cache is a larger item added to state that already depended on sticky routing; it doesn't create a new routing requirement. The header could say "per-request state only".

## Q1: config field vs CLI flag

**How M\* exposes model behaviour:**
- **YAML config fields are the convention.** `Cosmos3Model._load_config` (`mstar/model/cosmos3/cosmos3_model.py`) overlays YAML `model_kwargs` onto `Cosmos3Config` with `setattr`, and logs and ignores unknown keys. `docs/serving.rst` ("model_kwargs") describes them as "model parameters fixed at server start … baked into the model's config dataclass".
- **CLI flags are infrastructure only.** `mstar/api_server/entrypoint.py:main` covers `--config`, host and ports, transport, timeout, logging and the Rust frontend. `mstar/cli/main.py:build_parser` (`mstar serve <model> [--config]`) is the same. The K8s branches add only `--remote-workers` and `--live-membership`.
- **There is no generic override flag** (`--set`, `-o key=val` or `--config-override`) on `main` or in any worktree.
- **Env vars are developer escape hatches,** mostly undocumented. Examples: `COSMOS3_COMPILE_VAE` (`Cosmos3VAEDecoderSubmodule.__init__`), `COSMOS3_VAE_DECODE_FP32` (`_decode_dtype`) and `COSMOS3_DISABLE_CUDA_GRAPH`. One documented precedent pairs a config field with an env override: `compile_reasoner_decode` with `COSMOS3_REASONER_COMPILE=0` (`docs/models.rst`).
- **A related top-level YAML key** is `max_concurrent_requests`, enforced at conductor admission.

**Recommendation:**
- Keep it as the YAML key `model_kwargs.windowed_decode_warm_cache`, as the PR does. Users switch it off by editing or copying the config.
- A new `--` flag would break the convention, since no CLI flag touches model behaviour.
- If a no-edit kill switch is wanted, `COSMOS3_VAE_WARM_CACHE=0` would match the `COSMOS3_REASONER_COMPILE` precedent. This is optional.
- It shouldn't be a per-request knob: it's a capacity and memory decision.
- Whether the default is `true` or opt-in `false` is for the maintainer (see open questions).

## Q2: cache lifecycle and routing

**Keyed by** request id, inside one submodule instance, so per copy and per process: `forward` → `rid = engine_inputs.request_ids[0]` → `self.request_state(rid)["ar_feat_cache"]`. It is not keyed by session.

| Event | What frees or keeps the cache | Evidence |
|---|---|---|
| First window | Created: `new_feat_cache` + `_decode_stream(clip_start=True)` + `st.add` | `Cosmos3VAEDecoderARSubmodule.forward` |
| Each later window | Updated in place | `decode_causal` |
| Last window | `st.remove("ar_feat_cache")`, before the client finishes reading | `forward`; `test_cache_is_freed_after_the_last_window` |
| Normal teardown, abort | Conductor drain, then `REMOVE_REQUEST` → `Worker` remove handler → `EngineManager.remove_request` → `Engine.remove_request` → `NodeSubmodule.cleanup_request` pops `request_states` | `mstar/worker/worker.py`, `mstar/engine/engine.py:remove_request`; `test_cache_goes_with_an_aborted_request` |
| Error in any node | `Worker._fail_requests` → `FAIL_REQUESTS` → conductor sends `REMOVE_REQUEST` → same path | `mstar/worker/worker.py:_fail_requests` |
| Session end or timeout | Not involved: sessions hold only the latent tail | `SessionStore` (`mstar/model/cosmos3/sessions.py`) |
| KV offload or eviction | Not involved: `Engine.offload_request` moves engine resources only, never `request_states` | `mstar/engine/engine.py:offload_request` |
| Shutdown or worker death | Process exit; `DeadWorkerError` takes the instance down (single copy) | `Conductor._fail_all_requests` |
| Cap or eviction | **None**; see GAP 1 | — |

**Which decoder copy runs each window: sticky per request, fixed at admission.**
- **On `main`:** `Conductor._assign_worker_graphs_to_workers` picks a DP replica per node group at random, once per request. `_build_request_sharding_config` freezes it into the request's `ShardingConfig`, which routes every edge, including the `window_latents` streaming edge (`_split_inputs_to_workers`). The DiT and the decoders are separate node groups in `cosmos3_nano_ar.yaml`, so they're picked independently.
- **On `wt-k8s-live`:** `CopyPools.pick` (`mstar/conductor/copy_pool.py`) returns the bound copy for the request's whole life. It is least-loaded in live mode and random otherwise (`Conductor._build_pools`, overridable with `MSTAR_COPY_PICK`). Load is the number of active bindings, not memory.
- **Never least-loaded per window,** and there is no migration.
- **The generator's `SessionStore` pin** (`begin`) only blocks a second concurrent request on the same session within that DiT copy. It doesn't influence routing and doesn't touch the decoder cache.

**Next window lands on a copy without the cache (the key answer):**
- **Handled: the request state is present but the cache is absent.** `forward` falls back to the stateless re-decode behind `ar_tail`, trims, and continues warm from the cache that decode leaves (`test_missing_cache_falls_back_to_the_redecode`). In the tree this only happens on a stream's first window, fresh or resumed. Nothing removes the cache mid-stream.
- **Not handled: the window arrives on a copy with no request state at all.** This cannot happen today: routing is sticky, and a lost copy fails the request (503) instead of moving it. If it ever happened (a future migration or rebalancing feature), both the old and new code would decode wrongly, silently:
  - `ar_chunks` restarts at 0, so the window is treated as window 0.
  - The re-pinned overlap latents aren't trimmed, so frames are duplicated.
  - The first latent decodes as a clip-start frame.
  - The last window is never detected: assembled mode never emits, and the session tail is never stored.
  
  This is a latent GAP (GAP 2), not a regression from this PR.

**Resumed sessions:** the first window is the stateless re-decode behind the session's stored 8-latent tail; later windows are warm (`test_resumed_session_warms_up_from_the_stored_latents`). If the resumed request lands on a decoder copy that doesn't hold the session, see GAP 5.

## Q3: Kubernetes and multi-node

- **Where the decoder runs.** Placement is `node_groups`, so a config edit can place components on pods (`wt-k8s-dist`, `notes/phase3-distributed-components.md`). Three layouts are possible:
  - The decoder shares the DiT's pod and GPU, as in today's `cosmos3_nano_ar.yaml` (rank 0).
  - It runs in a separate pod.
  - It runs as several copies: a StatefulSet per group, scaled with `kubectl scale` (`wt-k8s-live`, `notes/live-membership.md`).
- **One stream stays on one decoder pod.** The binding is fixed at ingest (as on `main`) and made explicit by `CopyPools`.
- **Pod loss** (spot preemption, crash, rolling restart without drain):
  - With live membership, `Conductor._on_worker_lost` fails only the requests active on that copy, with 503 "retry it".
  - Without it (or for a single-copy component), the whole instance goes down (`DeadWorkerError`).
  - There is no one-window fallback on another copy. Exactness isn't at risk mid-stream, because the stream simply ends.
  - A retry is a new request: fresh cache, exact from its own start.
  - A `resume_session` retry needs the DiT copy that holds the session (otherwise the DiT raises "unknown or expired session"). Its first window is the stateless approximation, which is the same as before this PR.
- **Graceful scale-down.** `_on_worker_leaving` marks the copy DRAINING (no new requests) and releases it once nothing is bound. But `MSTAR_DRAIN_TIMEOUT_S` defaults to 120 s, and a rollout longer than that is killed anyway (GAP 6).
- **Memory capacity per decoder pod.** These are estimates, not measured on K8s. They assume a dedicated decoder GPU, about 10% headroom, and the single-stream peaks from `RESULTS.md` §2b (35.3 / 28.8 GiB at `latents_per_call=0`, 8.2 / 7.4 GiB at `=1`).

  | GPU (usable) | Wan 2.1, 1.76 GiB per stream | Wan 2.2, 1.33 GiB per stream |
  |---|---|---|
  | B200 (≈178 GiB) | ~70 (`=0`) to ~86 (`=1`) | ~98 to ~114 |
  | H200 (≈131 GiB) | ~47 to ~62 | ~67 to ~83 |

  Compute binds first on a dedicated GPU. A served window takes 112 ms (Wan 2.1) or 143 ms (Wan 2.2) on B200, so one copy sustains about 9 or 7 windows per second, shared across its streams. Co-located with the DiT, memory binds first: the DiT weights and the fixed KV pool (`resources.kv.max_num_pages`) leave far less room, and nothing reserves cache memory (GAP 1).
- **Scaling signal.** Use concurrent windowed streams per decoder copy (`in_flight` and `pending_steps` per copy from `/components` and `/metrics` on `wt-k8s-live`), not GPU utilization. Utilization is bursty (112 ms busy per window) and doesn't reflect cache memory.
- **Cross-node traffic is unchanged.** Per window: latents in, about 1.6 MB (Wan 2.1 `[1,16,8,60,104]` bf16) or 1.2 MB (Wan 2.2); uint8 frames out, about 28.8 MB per 24 frames at 480×832. The 1.3–1.8 GiB cache never leaves the decoder GPU. `PerRequestState.disag_shared_keys` (unimplemented) is the hook a future state transfer would use, and the cache isn't marked.
- **What the K8s RFC (`wt-k8s-rfc`, `notes/rfc-128-kubernetes.md`) would need to add.** It currently says "stateful features (sessions…) are per pod. A miss is slower, not wrong":
  - **Session affinity at the instance router (phase 2) and the copy pick (live membership).** A `resume_session` must reach the DiT copy, and ideally the decoder copy, that holds the session. For Cosmos3 a DiT miss is an error, not slower.
  - **A per-copy stream cap or memory budget** for the decoder (GAP 1), exported as a metric.
  - **Drain before scale-down,** with a drain timeout at least as long as the longest rollout, for decoder and DiT StatefulSets.
  - **Treat preemption as fail-and-retry**, with no mid-stream migration, unless request-state transfer is built.

## GAPs and the smallest fix for each

1. **No cap and no accounting for cache memory.** Each live stream adds 1.3–1.8 GiB outside any resource pool, on a GPU the DiT may share. The only bound is admission (`max_concurrent_requests` if set). *Fix:* add `windowed_decode_max_warm_streams: int = 0` (0 = no cap) to `Cosmos3Config` (`mstar/model/cosmos3/config.py`). In `Cosmos3VAEDecoderARSubmodule.forward`, create a cache only when `sum("ar_feat_cache" in s for s in self.request_states.values()) < cap`; otherwise take the stateless path, which already exists and is exact per window. Also log the live count and `feat_cache_bytes` (defined in `wan_vae.py`, unused outside a test). At minimum, document `max_concurrent_requests` in the config comment.
2. **No guard against a window arriving without its request state.** This can't happen today, and it affects the old code too, but it would be silent corruption. *Fix:* put the window index in the streamed chunk (`Cosmos3DiTSubmodule._finish_window` already has `window_index`), and have `forward` raise if it doesn't match `st["ar_chunks"]`.
3. **Docstring and test wording.** "a lost cache" (in `test_windowed_decode.py`) and "Where a stream has no cache" can read as cross-copy recovery. *Fix:* say the fallback covers a stream's first window (fresh or resumed), and that a stream is bound to one copy for its life.
4. **Transient activation peak at the default `windowed_decode_latents_per_call=0`.** It adds about 27 GiB (Wan 2.1) or 21 GiB (Wan 2.2) at served window size (`RESULTS.md`, "Served windows and memory"), on the DiT's GPU in `cosmos3_nano_ar.yaml`. This is a one-at-a-time peak, not per stream, but it can OOM next to the KV pool. `=1` keeps the peak at 7–8 GiB for about 6% more decode time. *Fix:* default `1` (or a small chunk such as 2–3), or call out the trade-off in the config comment; see the open questions.
5. **Pre-existing, not this PR: a session miss on the decoder is silent.** The DiT raises on an unknown session (`_session_tail`). The decoder's `SessionStore` never calls `begin`, so its entries are never pinned and count against capacity. They can be evicted before the DiT's, or be absent on another decoder copy, and then window 0 decodes with no context: the first latent decodes to 1 frame instead of 4, leaving a seam and a 3-frame shortfall. *Fix:* have the decoder call `begin` on session requests as the DiT does, and log or raise when `resume_latent_units > 0` and `get` misses.
6. **K8s (`wt-k8s-live` only).** `MSTAR_DRAIN_TIMEOUT_S=120` by default can cut long rollouts during scale-down. *Fix:* chart value per StatefulSet, sized to the longest rollout.

## Open questions for the maintainer

1. **Default on, or opt-in?** The PR defaults `windowed_decode_warm_cache: true`. It is exact and 5–6× faster, but it adds 1.3–1.8 GiB per live stream where the stateless decode keeps nothing. Opt-in keeps today's memory profile for existing deployments.
2. **Cache cap:** a per-copy stream cap with stateless overflow (GAP 1), or rely on `max_concurrent_requests`?
3. **Is sticky per-request routing a guarantee** the decoder may rely on? It's already relied on for `ar_tail` and `ar_chunks`. If migration is ever planned, should streams fail loudly (GAP 2), or should the request state travel (`disag_shared_keys`)?
4. **`windowed_decode_latents_per_call` default:** `0` (fastest, +21–27 GiB peak) or `1` (bit-exact against `vae.decode`, peak 7–8 GiB)?
5. **Header wording:** keep "STATELESS", or say "per-request state only"?
6. **Session affinity** for `resume_session` with more than one DiT or decoder copy: who owns it, the router or the copy pool?

## Notes for the PR agent

- The images and script are in this directory. `draw_design.py` uses matplotlib only, and `--out` writes PNG and SVG anywhere, for example `python draw_design.py --out benchmark/wan_vae_decoder/results/b200/charts`.
- Relative image paths don't resolve in a GitHub PR body. Once committed, use `https://github.com/mstar-project/mstar/blob/<branch>/benchmark/wan_vae_decoder/results/b200/charts/design_before_after.png?raw=true` (and likewise for the other two).
- The before/after figure uses the benchmark chunk (3 new latents, 12 frames), where 335 → 56 ms and 3.4× FLOPs were measured. Served windows have 6 new latents (112 ms per window, Wan 2.1).
