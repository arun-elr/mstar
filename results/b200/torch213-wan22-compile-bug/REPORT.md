# Wan 2.2 decodes wrong under torch 2.13 (PR mstar-project/mstar#425): root cause and fix

Investigated on `wan-bench-b200-vllm` (1× B200) inside the vLLM-Omni image
`vllm/vllm-omni@sha256:d9d81b77c0c5055425f388ddde756ec4206c53e141d1ca6d1cbee21b2ba24e9f`
(torch 2.13.0+cu130, CUDA 13.0, cuDNN 9.20.0, Triton 3.7.1, diffusers 0.40.0). The code is a copy of the PR
at `4b8b478953bf6b6fcd60884c3563225cb70685b9`, the pushed HEAD when I started. The worktree was not
touched. The pushed HEAD has since moved to `c56b572b`; the patch applies cleanly there and its CPU
tests pass (23/23).

## Summary

- **Root cause: a `torch.compile` (Inductor) miscompile, triggered by Inductor's comprehensive stride
  padding.** It is not diffusers, not cuDNN, and not any of the PR's five decode changes. Compiling the
  stock diffusers decoder alone is enough to reproduce it. In the generated code, `up_blocks[1]`'s
  shortcut input `x_copy = x.clone()` is read through `DupUp3D` with the padded channels-first strides
  that Inductor chose for the clone. But the buffer it actually reads was written channels-last: it was
  fused in place into the channels-last output of `up_blocks[0]`'s upsample `Conv2d`. The reader
  therefore gets wrong values, and it runs about 65K elements past the end of the buffer. The results
  are wrong frames, a wrong conv cache (so the error compounds, and turns into NaN within a few chunks),
  and sometimes a CUDA illegal memory access.
- **Why only torch 2.13.** The Inductor bug is also present in torch 2.12.1: the standalone repro
  miscompiles identically there. On 2.12.1, though, dynamo graph-breaks the decoder at diffusers'
  `if feat_cache[idx] == "Rep":` (in `WanResample`), so a decoder call becomes 16 graphs. Torch 2.13
  traces through that comparison, so the whole decoder call becomes one 559-op graph. Only that bigger
  graph puts the mid block and the first two up blocks together, which is what the bug needs.
- **Why only Wan 2.2.** The pattern (an `x.clone()` shortcut through `DupUp3D` around residual up
  blocks) exists only in the Wan 2.2 decoder (`is_residual=True`).
- **diffusers is ruled out.** `autoencoder_kl_wan.py` is byte-identical in 0.39.0, 0.40.0 and the
  container (sha256 `a38de0c53adcb81887a0b48d44e3fb59737d7b1e4779a666febe83be666e8fa8`).
- **Fix:** compile the Wan VAE decode with `options={"comprehensive_padding": False}`. On torch 2.13,
  Wan 2.2 is back to 55.5 dB (in-tree, uint8) and 57.0 dB (prototype, float, peak 1). On Wan 2.1, and
  on torch 2.12.1, the timing is unchanged within noise and the PSNR is bit-identical. The patch is
  `fix-inductor-padding.patch`.

## Is the in-tree M* path affected?

Yes. Every compiled Wan 2.2 decode in M* is affected on torch 2.13; nothing is affected on torch 2.12.1.

| path | compiled by default? | torch 2.13, Wan 2.2 | torch 2.13, Wan 2.1 | torch 2.12.1 (both VAEs) |
|---|---|---|---|---|
| `mstar_warm`: `Cosmos3VAEDecoderARSubmodule` warm-cache stream (`DecoderStep`, this PR) | yes (`COSMOS3_COMPILE_VAE` defaults to 1) | **wrong**: 13.85 dB uint8 (10-chunk run); the full-length run **crashed** with a CUDA illegal memory access | correct | correct |
| `mstar_warm_eager` | no | correct (55.39 dB) | correct | correct |
| `mstar_ar_ctx8`: stateless `Cosmos3VAEDecoderSubmodule` (`torch.compile(vae.decode)`, **pre-existing on main**) | yes | **wrong**: 11.29 dB uint8 | not run (same compile; Wan 2.1 has no residual up blocks) | correct (51.08 dB in the PR's B200 results) |
| `Wan22VaeDecoderSubmodule` | no (`compile_vae_decode=False`) | eager by default, so correct; with `compile_vae_decode=True` it goes through `DecoderStep`, which is affected | n/a | correct |
| benchmark prototype `px_*`, `*_compile` (`common.StreamingDecoder`) | n/a | **wrong**: `px_nopad_bench` 13.86 dB | correct | correct |

Cosmos3 serves the Wan2.2-TI2V-5B VAE (`Cosmos3Config.vae.z_dim = 48`), so on torch 2.13 its image and
video decodes would serve wrong frames at 480×832. This includes main's stateless decode, not only this
PR. The bug depends on Inductor padding a stride: the latent plane 30×52 = 1560 bf16 elements is padded
to 1600, and 60×104 = 6240 to 6272. Resolutions whose strides are already 128-byte aligned may not
trigger it. I did not test other resolutions; the guard is unconditional.

## The fix

Inductor's comprehensive padding is turned off for the Wan VAE decode compiles only. Nothing global is
changed, and the default mode is otherwise unchanged:

- `mstar/model/components/wan_vae.py`: adds `COMPILE_OPTIONS = {"comprehensive_padding": False}` and
  `compile_decode(fn)` (`torch.compile(fn, fullgraph=False, dynamic=False, options=COMPILE_OPTIONS)`).
  `DecoderStep` uses it.
- `mstar/model/cosmos3/submodules.py`: `Cosmos3VAEDecoderSubmodule` compiles `vae.decode` through
  `compile_decode` (this also fixes main's stateless path).
- `benchmark/wan_vae_decoder/common.py`: `StreamingDecoder` compiles with
  `options={**torch._inductor.list_mode_options(compile_mode or "default"), "comprehensive_padding": False}`.
  `torch.compile` takes a mode or options, not both. This also keeps the `px_ro` / `px_maxat` modes. Every
  compiled benchmark variant (`px_*`, `diffusers_bf16_compile`, `*_compile`) now uses the guard. On
  torch 2.13, the unguarded numbers for those variants on Wan 2.2 are invalid.
- `test/modular/test_wan_vae_decode.py`: adds `test_compiled_decodes_turn_off_inductor_stride_padding`,
  which checks that both compile sites pass the option. It fails without the fix (`KeyError: 'options'`)
  and passes with it.

I chose not to gate this on the torch version: the bug exists in 2.12.1 too, and only the graph split
hides it there. Turning the option off costs nothing measurable on either version:

| run (B200, ms median of 150 samples; uint8 PSNR dB) | PR as is | PR + fix |
|---|---|---|
| torch 2.13, Wan 2.2, `mstar_warm` | crash (10-chunk run: 64.89, 13.85 dB) | **73.36**, 55.48 dB |
| torch 2.13, Wan 2.2, `px_nopad_bench` | 71.72, 13.86 dB (float peak 1: 7.85) | **72.28**, 55.33 dB (float peak 1: 56.99) |
| torch 2.13, Wan 2.1, `mstar_warm` | 53.74, 54.65 dB | 53.83, 54.65 dB |
| torch 2.13, Wan 2.1, `px_nopad_bench` | 53.70, 55.25 dB | 53.31, 55.25 dB |
| torch 2.13, Wan 2.2, `mstar_ar_ctx8` (10 chunks) | 352.89, 11.29 dB | 352.60, 51.09 dB |
| torch 2.12.1, Wan 2.2, `mstar_warm` | 74.55, 55.40 dB | 74.27, 55.40 dB |
| torch 2.12.1, Wan 2.1, `mstar_warm` | 57.05, 54.42 dB | 56.74, 54.42 dB |

Notes on the table:

- The torch 2.12.1 rows come from a venv inside the same container on the same VM, with M*'s package
  versions (torch 2.12.1+cu132, diffusers 0.39.0, Triton 3.7.1, cuDNN 9.20.0.48). I did not run anything
  on `wan-bench-b200`.
- For reference, on M*'s VM Wan 2.2 `px_nopad_bench` is 72.99 ms at 56.56 dB (float, peak 1), so with
  the fix the torch 2.13 stack is about as fast on Wan 2.2 and slightly faster on Wan 2.1.
- The handoff's 72.04 ms for `px_nopad_bench` on Wan 2.2 measured a broken graph. Use 72.28 ms (fixed)
  instead.

Workarounds I considered and rejected:

- `torch._inductor.config.comprehensive_padding = False` set globally would also work, but it changes
  every other compile in the process (DiT, reasoner).
- `padding_stride_threshold=10**9` also fixes it, but it is less direct.
- Inducing a graph break between the up blocks would depend on dynamo's behavior and cost fusion.

## Evidence

### 1. Reproduction (PR copy, in the container)

`px_nopad_bench` on Wan 2.2 gave 13.87 dB (float peak 2), 7.85 dB (peak 1) and 13.86 dB (uint8), the
same as the handoff. In-tree `mstar_warm` on Wan 2.2 gave 13.85 dB, while `mstar_warm_eager` gave 55.39
dB. On Wan 2.1 both were correct (54.65 and 54.41 dB). See `runs/repro/` and `runs/intree/`.

### 2. Bisect (Wan 2.2, torch 2.13, `common.StreamingDecoder`; script `scripts/w22_bisect.py`)

PSNR is per chunk against the stock eager bf16 decode, which itself is 56.0–56.5 dB from fp32.

| config | result |
|---|---|
| eager + all five non-compile changes (channels-last, one call per chunk, native conv padding, cudnn.benchmark, bf16 upsample) | correct: 56.1–56.5 dB vs fp32 |
| **compile only** (stock diffusers forwards, channels-first, one latent per call, no cudnn.benchmark) | **wrong**: clip-start frame 56.7 dB; steady chunk 1: 9.6, chunk 2: 9.9, chunk 3 onward: **NaN** |
| compile + one call per chunk | wrong: 8.8–9.9 dB on every chunk (no NaN) |
| compile + cudnn.benchmark | wrong: 9.6–10.4 dB |
| compile + bf16 upsample | wrong: same as compile only |
| compile + channels-last | **CUDA illegal memory access** |
| compile + native conv padding | crashed (no result) |

The smallest wrong config is `torch.compile` alone. The error starts at the first steady compiled call;
the clip-start call is fine. A single compiled steady call made from a correct eager cache returns
12.1 dB frames and also writes wrong cache slots (5.3 dB), so the error compounds across chunks.
Per-latent calls reach NaN by the third chunk.

### 3. Localization (`scripts/localize.py`, `backends.py`, `spans.py`)

- **Per module class.** I compiled only `DupUp3D`, `WanResidualUpBlock`, `WanResidualBlock`,
  `WanResample`, `WanAttentionBlock`, `WanMidBlock`, `WanRMS_norm` or `WanCausalConv3d`, comparing each
  instance's output with eager on the same input. All agreed within bf16 rounding (rel-L2 ≤ 0.006).
  Only compiling the whole `WanDecoder3d` was wrong (12.1 dB).
- **Backends.** `eager`, `aot_eager` and `aot_eager_decomp_partition` are bit-exact on the whole
  decoder; `inductor` is wrong. That rules out dynamo, AOT and decompositions.
- **Inductor config.** `comprehensive_padding=False` fixes it (57.8 dB vs eager bf16, normal compile
  rounding), and so does `padding_stride_threshold=10**9`. Still wrong with `allow_buffer_reuse=False`,
  `inplace_buffers=False`, both together, `max_fusion_size=1`, `epilogue_fusion=False`,
  `loop_ordering_after_fusion=False`, `pattern_matcher=False`, or `pad_channels_last=False`.
- **Spans of top-level stages compiled as one graph.** The smallest failing span is mid_block +
  `up_blocks[0]` + `up_blocks[1]` (rel-L2 0.81). Mid + `up_blocks[0]`, and `up_blocks[0..1]` alone, are
  fine.
- **Generated code** for that span (`evidence/inductor_kernels_excerpt.txt`, full files
  `evidence/inductor_span14_{padded,nopad}.py`):
  - With padding: `buf95` (`up_blocks[0]`'s output, which is `up_blocks[1]`'s `x`) is
    `reinterpret_tensor(buf94, (1,1024,2,60,104), (12779520, 1, 6389760, 106496, 1024))`. That is
    channels-last, and it is written in place by kernel 21 at `r0_3 + 1024*x4`.
  - Kernel 30, which adds `DupUp3D(x_copy)` to the upsampled output, reads the same `buf95` at
    `104*(h//2) + 6272*(t//2) + 12544*c + (w//2)`. Those are padded channels-first strides, the layout of
    the clone. Its largest offset is about 12.845M, while `buf95` holds 12.78M elements.
  - Without padding, `buf95` is channels-first `(12779520, 12480, 6240, 104, 1)` and kernel 26 reads it
    with matching strides.
- **Not involved.** Each of the suspects listed in the brief was checked:
  - Patchify/unpatchify: it runs outside the compiled decoder.
  - The 48-channel / `channels_last_3d` path and native conv padding: the bug reproduces with stock
    channels-first forwards.
  - cuDNN algorithm choice: it reproduces without `cudnn.benchmark`, and cuDNN is 9.20.0 on both stacks.
  - Triton: the version is 3.7.1 on both stacks, and the faulty index math comes from Inductor.
  - diffusers API changes: the file is identical.

### 4. Minimal standalone repro (`repro_wan22_inductor_padding.py`, torch only, about 35 s on a B200)

I captured the failing span's FX graph and shrank it with torch's FX minifier, from 425 to 82 nodes
(`evidence/minified_graph_82_nodes.txt`). I then rewrote it as a readable function of about 30 lines
with random inputs: mid block's last conv → `up_blocks[0]` (Conv2d + `DupUp3D` shortcut) →
`up_blocks[1]` (`x.clone()`, a resnet, temporal and spatial upsample, + `DupUp3D(x_copy)`).

```
torch 2.13.0+cu130, NVIDIA B200
  compiled (default) vs eager: rel L2 1.0776
  compiled (comprehensive_padding=False) vs eager: rel L2 0.0037
MISCOMPILED                                   (exit 1)
torch 2.12.1+cu132, NVIDIA B200: identical numbers, MISCOMPILED
```

Smaller hand-written variants (conv2d → add → clone → `DupUp3D`, with and without
`force_layout_optimization`) do not reproduce it. The trigger depends on Inductor's layout decisions
across this larger graph.

### 5. Why torch 2.12.1 decodes correctly (`scripts/graphs.py`, `torch._dynamo.explain` on one steady call)

| torch | graphs per decoder call | graph breaks | break site |
|---|---|---|---|
| 2.12.1+cu132 | 16 (largest 112 ops) | 15 | `autoencoder_kl_wan.py:286 if feat_cache[idx] == "Rep":` ("torch.* op returned non-Tensor") |
| 2.13.0+cu130 | 1 (559 ops) | 0 | none |

On 2.12.1 the whole Wan 2.2 decoder compiles to 57.68 dB vs eager bf16 both with and without padding.

## Could CPU tests have caught this?

No. Neither the numerics nor a diffusers-version matrix would have caught it:

- The diffusers file is identical across 0.39 and 0.40, so a version matrix exercises the same code.
- The bug lives in Inductor's GPU path. Comprehensive padding is off on CPU (`disable_padding_cpu = True`),
  the channels-last conv layout optimization that creates the mismatched buffer is a GPU path, and CPU
  Inductor generates C++, not Triton. The PR's CPU tests also run tiny VAEs with `compile=False`.
- What a CPU test can do is pin the guard. The patch adds one: it monkeypatches `torch.compile` and
  checks that both Wan decode compile sites pass `comprehensive_padding=False`. It is cheap and worth
  keeping, so the option isn't dropped in a refactor.
- Catching the numerics needs a GPU check. A cheap option is to run `repro_wan22_inductor_padding.py`,
  or a compiled-vs-eager steady Wan 2.2 call at 480×832, wherever GPU tests run, especially on torch
  upgrades.

## Follow-ups

- Report the bug to PyTorch with `repro_wan22_inductor_padding.py`: Inductor reads a buffer with the
  padded layout of a clone that was fused away, while the buffer was written channels-last. Once a fixed
  torch ships, the guard can be version-gated or removed, with the GPU check above as the gate.
- `DecoderStep` still has its two older workarounds for compile problems seen on torch 2.12: it runs the
  "Rep" call eagerly, and it warms each new call signature once on a copy of the cache. I did not test
  whether either problem is related to this padding bug, so both stay in place.
- The PR's `RESULTS.md` and README numbers are from torch 2.12.1, where the guard changes nothing
  (identical PSNR, timing within ±0.3 ms), so they don't need re-running.

## Files

- `REPORT.md`: this file.
- `fix-inductor-padding.patch`: against the PR; applies to `4b8b4789` and `c56b572b` (4 files, +43/−3).
- `repro_wan22_inductor_padding.py`: standalone torch-only repro; exit code 1 means miscompiled.
- `evidence/`: Inductor generated code with and without padding for the failing span, the kernel
  excerpt, and the minimized 82-node graph.
- `runs/`: run JSONs (`bench_wan_decoder.py` schema) and `verify_summary.txt`.
- `scripts/`: investigation harness:
  - `dock.sh`: one container per job.
  - `run.py`: `bench_wan_decoder.py` with offline VAE snapshots.
  - `w22_bisect.py`, `localize.py`, `backends.py`, `spans.py`, `graphs.py`: the bisect and localization
    steps above.
  - `capture.py`, `run_graph.py`, `minify.py`: graph capture, standalone replay and minification.
  - `verify.sh`: the verification batch.

On the VM, everything is under `~/w22/`: `mstar` (PR copy), `fix/mstar` (patched copy), `venv212`
(torch 2.12.1 venv), `h` (scripts), `code` (generated code), `runs`, and `repro` (captured graph, plus a
1.2 GB file of real input tensors). The VM is left running and idle.
