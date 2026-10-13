# Prompt-to-video samples: Wan VAE decode, one ladder step at a time

Two Wan 2.2 TI2V-5B text-to-video clips, each decoded from the same latents by every step of the
PR's change ladder through M*'s in-tree streaming decoder node. The scenes are a fox running through
snow and waves on a rocky beach, at 1280×704, 121 frames, 24 fps (5 s).

**Result: no step after the warm cache changes the video visibly.** Measured against stock diffusers
`vae.decode`:

- Every step from the warm cache on keeps the worst frame at or above 56.6 dB PSNR (fox) and 50.0 dB
  (waves), with worst-frame SSIM ≥ 0.9989.
- That is closer to stock bf16 than the model card's own fp32 pipeline decode is (52.4 / 47.3 dB).
- The warm cache run eagerly is byte-identical to stock (same md5) on both clips. All remaining
  differences come from `torch.compile`'s fused bf16 rounding, which M* before also has.
- The step that differs is **M* before**: its stateless 8-latent context re-decode drops to
  35.8 dB (fox) and 27.0 dB (waves). The drops come at the first frames of every 3-latent chunk from
  chunk 3 on, once the context window is truncated. That is a seam every 0.5 s, visible in the
  contact sheets' `|M* before - stock|` row. With the warm cache there is no seam.

**Speed:** decode time per 3-latent chunk falls from 784 ms to 159 ms (4.9×). For the whole 5 s clip,
streamed decode time falls from 6.96 s to 1.87 s.

Setup:

- Model and generation: `Wan-AI/Wan2.2-TI2V-5B-Diffusers` @ `b8fff731`, diffusers `WanPipeline`,
  model-card settings (50 steps, guidance 5.0, seed 42).
- Hardware and stack: 1× B200, torch 2.12.1+cu130, cuDNN 9.20.0, diffusers 0.39.0.
- M* code: this branch at `c56b572b` (the decoder code is unchanged since).

Prompts, settings and commands: [`prompts.md`](prompts.md). Scripts: [`../../../samples/`](../../../samples/).

## What was decoded

Every row decodes the same latents per prompt. The node rows go through
`Cosmos3VAEDecoderARSubmodule.forward`, using `mstar/model/components/wan_vae.py` and the config
switches below:

- the input is one streamed request: the clip-start latent, then ten 3-latent windows, with
  overlap 0;
- decode in bf16, `COSMOS3_COMPILE_VAE` on (the default);
- one fresh process per step, streaming the clip twice. Pass 1 compiles and warms up; pass 2 is timed
  and its frames are kept. Both passes give the same md5 in every row.

The streaming path is the Cosmos3 node because M*'s `wan22` node decodes whole clips with
`vae.decode`. Both use the same Wan 2.2 VAE.

**Native padding has no switch of its own.** `ChannelsLastCausalConv3d` always pads inside the conv,
so it lands together with `vae_decode_channels_last`.

The reference for all metrics is stock diffusers `AutoencoderKLWan.decode`: the whole clip, eager,
bf16, fed the same bf16 VAE input the node computes.

## Per-step results

Metrics are on the raw uint8 frames, not the mp4s:

- **md5**: md5 of the uint8 frames.
- **px ≥1 / px ≥2**: share of pixel positions (frame, y, x) where any RGB value differs from stock by
  at least 1 or 2 levels.
- **PSNR**: per frame over RGB, peak 255. "mean" averages the frames that are not identical to stock;
  for the warm-cache row, 5 frames are identical (the eager clip-start and first calls). "worst" is
  the lowest frame, with its index in parentheses.
- **SSIM**: Gaussian 11×11, σ 1.5, per RGB channel; the worst frame is shown.

### Fox ("A red fox running through fresh snow in a pine forest, morning light, slow tracking shot")

| step | md5 (uint8 frames) | px ≥1 | px ≥2 | max diff | PSNR mean / worst (frame) | SSIM worst |
|---|---|---|---|---|---|---|
| stock diffusers `vae.decode` (reference) | `260894c1f61432dfc086a1d89bf8ef93` | 0 | 0 | 0 | ∞ / ∞ | 1.0000 |
| M* before: stateless re-decode, `windowed_decode_warm_cache=False` | `8d9ade325279a8548eb26d6004b58350` | 49.59 % | 16.298 % | 160 | 47.1 / 35.8 (49) | 0.9707 |
| + warm cache | `d41f77847dcb4e2ac124d5452caf3691` | 24.76 % | 0.446 % | 33 | 57.5 / 56.7 (52) | 0.9991 |
| + channels-last (with native padding) (= this PR's defaults) | `bb9534952cde3a5ea6ff369ab43c6d93` | 25.36 % | 0.449 % | 28 | 57.6 / 56.7 (52) | 0.9991 |
| + one decoder call per chunk (`latents_per_call=0`) | `f8f62793c8c94ae17eefb519715f5faa` | 24.61 % | 0.427 % | 30 | 57.8 / 56.7 (52) | 0.9991 |
| + `cudnn.benchmark` (`=0`, opt-in) | `de0b5aa77705e5ec8610fa51ece7098c` | 24.65 % | 0.430 % | 27 | 57.8 / 56.7 (52) | 0.9991 |
| *control: warm cache only, eager* | `260894c1f61432dfc086a1d89bf8ef93` | 0 | 0 | 0 | ∞ / ∞ (identical to stock) | 1.0000 |
| *context: model-card pipeline decode, fp32 VAE* | `b3b2814cd4ef5a29695564a1c1e972e3` | 61.95 % | 1.867 % | 29 | 52.9 / 52.4 (52) | 0.9981 |

### Waves ("Waves rolling onto a rocky beach at sunset, seagulls in the distance")

| step | md5 (uint8 frames) | px ≥1 | px ≥2 | max diff | PSNR mean / worst (frame) | SSIM worst |
|---|---|---|---|---|---|---|
| stock diffusers `vae.decode` (reference) | `b034ca5bff80f50ce0e74c2e9d599e5b` | 0 | 0 | 0 | ∞ / ∞ | 1.0000 |
| M* before: stateless re-decode, `windowed_decode_warm_cache=False` | `e10cfff5e7c22b89ce1e80fdbb748267` | 63.57 % | 33.572 % | 238 | 38.0 / 27.0 (49) | 0.9261 |
| + warm cache | `d0cef71a19aaee4a4945240b0887fa4d` | 40.01 % | 4.197 % | 46 | 52.1 / 50.0 (95) | 0.9989 |
| + channels-last (with native padding) (= this PR's defaults) | `a6ddcc6e6e62052b58bdb3a2a025abcf` | 41.17 % | 4.254 % | 48 | 52.3 / 50.1 (95) | 0.9989 |
| + one decoder call per chunk (`latents_per_call=0`) | `ad3da2186b2c3d5123d1f4a1e38feef2` | 40.31 % | 4.127 % | 40 | 52.5 / 50.1 (95) | 0.9989 |
| + `cudnn.benchmark` (`=0`, opt-in) | `b8d53ea7e2b6684610593963dc60cf1d` | 40.36 % | 4.135 % | 39 | 52.5 / 50.1 (95) | 0.9989 |
| *control: warm cache only, eager* | `b034ca5bff80f50ce0e74c2e9d599e5b` | 0 | 0 | 0 | ∞ / ∞ (identical to stock) | 1.0000 |
| *context: model-card pipeline decode, fp32 VAE* | `5e642643a5074710ce36efe65ed09a89` | 71.87 % | 8.275 % | 67 | 48.9 / 47.3 (99) | 0.9977 |

Where M* before differs: in chunks 0–2 (frames 0–36), while the whole stream still fits in its
8-latent context, its worst frame is 57.5 dB (fox) and 52.6 dB (waves). From chunk 3 on, each
window is re-decoded from an empty cache behind only 8 latents of context. Its worst-frame PSNR then
drops at each chunk's first frame and recovers over the chunk. For the waves, chunk 4 runs
from 27.0 dB at frame 49 up to 35.6 dB at frame 60.

**The waves rows show bigger numbers but the same picture.** Every compiled step's difference from
stock is bf16 rounding, scaled by how much fine texture there is. The ×10 heatmaps stay close to
black.

### Timing (pass 2, CUDA-synchronized per window; the two prompts agree within 1.5 ms per chunk)

ms/chunk is the median over windows 4–10. That skips the clip-start window, the first 3-latent window
(which the compiled warm path runs eagerly), and M* before's still-growing context.

| step | ms / chunk (fox / waves) | vs M* before | whole clip ms (fox / waves) | peak alloc (GiB) |
|---|---|---|---|---|
| stock diffusers `vae.decode` (whole clip, eager, not streamed) | – | – | 3978 / 3980 | 11.6 |
| M* before | 783.5 / 784.8 | 1.00× | 6964 / 6961 | 13.8 |
| + warm cache | 227.0 / 227.6 | 3.45× | 2354 / 2355 | 13.1 |
| + channels-last (with native padding) (= this PR's defaults) | 162.2 / 163.6 | 4.8× | 1744 / 1752 | 15.5 |
| + one decoder call per chunk | 159.4 / 158.0 | 4.9× / 5.0× | 1884 / 1881 | 34.7 |
| + `cudnn.benchmark` (= this PR) | 159.5 / 158.2 | 4.9× / 5.0× | 1874 / 1876 | 34.7 |
| *control: warm cache only, eager* | 394.1 / 394.2 | 2.0× | 3974 / 3975 | 11.3 |

Notes on the timing:

- **One call per chunk is a small win at this resolution.** It saves about 3–6 ms per chunk, but the
  stream's first 3-latent window now runs eagerly as a single call: 409 ms, against 236 ms at one
  latent per call. That one-time 170 ms is more than a 10-chunk clip wins back, so this clip's total
  is 1.87 s rather than 1.74 s. One call per chunk also raises the decode's peak allocation from
  15.5 to 34.7 GiB at 1280×704.
- **`cudnn.benchmark` makes no measurable difference here**: 159.4 vs 159.5 ms (fox) and 158.0 vs
  158.2 ms (waves).
- **The per-chunk speedup matches the 832×480 Wan 2.2 number.** That run measured 368.85 → 73.76 ms (5.0×)
  at 832×480; here it is 4.9× at 1280×704.
- **First (cold) pass**, which includes compile: M* before took 34.1 s / 30.6 s, because it compiles
  `vae.decode` for every input length (1, 4, 7, 10 and 11 latents). This PR's defaults took 6.9 s.

### End-to-end (optional item, composed rather than a server run)

These clips have no Cosmos3 checkpoint behind them, and M*'s `wan22` server decodes whole clips with
none of these switches, so a served before/after run would not exercise this code. Instead, this is
the diffusers pipeline's measured time to latents plus the streamed decode measured above. Both
numbers are warm, pass 2.

| prompt | text encode + 50 denoise steps | + M* before decode | + this PR's decode | decode share before → after |
|---|---|---|---|---|
| fox | 72.6 s | 79.6 s | 74.5 s | 8.8 % → 2.5 % |
| waves | 71.6 s | 78.5 s | 73.4 s | 8.9 % → 2.6 % |

## Media

Per prompt (`fox_*`, `waves_*`):

- `<prompt>_<i>_<step>.mp4`: one per ladder step, at the original 1280×704, H.264 CRF 20 (preset
  slow, yuv420p), 24 fps. Steps: 0 stock, 1 M* before, 2 warm cache, 3 channels-last, 4 one call per
  chunk, 5 cudnn.benchmark = this PR.
  - **Trimmed to keep the total under 20 MB:** each clip is frames 37–60 (1 s). That is chunks 3 and
    4, which contain M* before's worst seam at frame 49. The full-length clips at CRF 20 came to
    about 1.5 MB (fox) and 5.1 MB (waves) each, about 64 MB in all.
- `<prompt>_side_by_side.mp4`: stock | M* before | this PR | `|this PR - stock|` ×10 heatmap
  (inferno colormap, max over RGB). Whole clip, panels at 480×264 (1920×292 total), CRF 20.
- `<prompt>_side_by_side.gif`: the same four panels as a 2×2 grid, 480×296, 8 fps, whole clip. It is
  under 3 MB at 8 fps; at 10 fps the fox GIF was 3.2 MB.
- `<prompt>_contact_sheet.png`: frames 0, 49, 60 (middle) and 120 (last).
  - Rows: stock, M* before, this PR, `|M* before - stock|` ×10, `|this PR - stock|` ×10. The two
    heatmap rows and the frame-49 column are additions to the requested sheet; they show where
    M* before differs from stock.
  - Panels are 320×176.

H.264 at CRF 20 with 4:2:0 chroma changes far more than 1–2 levels by itself, so the mp4s and GIFs
are for viewing only. All numbers above come from the raw uint8 frames.

## Caveats

- Determinism across processes was not re-tested here. Every step's two in-process passes are
  bit-identical. The full-stream quality run (`../quality.csv`) found that `cudnn.benchmark` can change the output between
  processes by up to 4 uint8 levels.
- torch 2.12.1 graph-breaks the Wan decoder at diffusers' `"Rep"` cache check, which keeps it clear of the
  Inductor padding miscompile that breaks compiled Wan 2.2 decodes on torch 2.13.
- The latents are not committed; `run_all.sh` regenerates them (md5s in `prompts.md`).
