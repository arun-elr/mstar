# Audio evidence for mstar-project/mstar#429

The Qwen3-Omni Code2Wav chunk-boundary trim, heard and measured on one real utterance: `utt5_T83`, 83 codec frames (6.6 s), the longest reply in the PR's fixture `test/integration/data/qwen3_omni_talker_codes.safetensors`.

| File | What it is |
|---|---|
| `full_decode.wav` | One decode of all 83 frames (the reference) |
| `main_stream.wav` | M*'s streaming path at `upstream/main` 99f5108b |
| `fix_stream.wav` | M*'s streaming path at the PR head 00bec2ba |
| `boundary_zoom.png` | ±20 ms around the boundary at 3.977 s, for all three |
| `error_vs_full.png` | SNR against the full decode in 10 ms windows, by sample index |
| `spectrogram.png` | Spectrograms of main and the fix on one scale |
| `metrics.json` | Every number below |
| `make_audio_evidence.py` | Regenerates all of the above |
| `long/` | A 72 s before/after listening test of one real 897-frame reply (35 boundaries); see `long/README.md` |

All WAVs are 24 kHz, mono, 16-bit PCM.

## How they were made

`make_audio_evidence.py` makes a `git archive` of each ref and runs each tree's own code in a separate interpreter. That code is the edge's `LeftContextChunkPolicy` (25 new frames plus 25 frames of context) and `StreamBuffer`, then `Code2WavSubmodule`'s `prepare_inputs` (pad to 50 frames), `preprocess`, `forward_batched` and `postprocess` (the trim). It loads the real Code2Wav weights (only the Code2Wav shard) from the local Hugging Face cache. The decode runs on CPU in the checkpoint's bf16, as M* serves it, but eager rather than `torch.compile`d. M*'s three Triton kernels (RMSNorm, RoPE, sliding-window attention) are replaced by the torch equivalents that the PR's CPU tests use. Both trees' full decodes come out bit-identical. A run takes about 20 s on 16 CPU threads.

```
python make_audio_evidence.py --repo <mstar checkout> --main-ref upstream/main \
    --fix-ref <PR head> --out assets/pr429
```

## Measured

- **Length.** The full decode is 158,805 samples (83·1920 − 555). The fix emits exactly that, in chunks of 47445, 48000, 48000 and 15360 samples. Main emits 158,250, in chunks of 48000, 47445, 47445 and 15360.
- **Where main goes wrong.** Each chunk of main was cross-correlated against the full decode. They line up at offsets of 0, 0, +555 and +1110 samples, so main skips 555 samples (23.1 ms) at its boundaries at 3.977 s and 5.954 s. At 2.000 s it skips nothing. There, the last 555 samples of its first chunk were decoded with the zero padding as look-ahead, at −8.5 dB SNR against the full decode (the fix: 28.2 dB). Main's stream ends with another 555 such samples.
- **Overall SNR against the full decode:** main 3.9 dB (misaligned after 3.98 s), the fix 27.0 dB. The fix's remaining error comes from the 25-frame left context truncating the 72-frame attention window, not from the trim.
- **Size of the step at a boundary.** This is |x[n] − x[n−1]| at the boundary sample, against the median and maximum of the other steps within ±20 ms, in int16 units:

  | Boundary | main | fix | full decode, same point |
  |---|---|---|---|
  | ~2.0 s | 49 (0.5× median) | 51 (0.8×) | 53 (0.8×) |
  | ~4.0 s | **4608** (19× median, 1.8× the local max 2493) | 224 (0.8×) | 400 (1.5×) |
  | ~6.0 s | **2776** (25× median, 1.3× the local max 2196) | 706 (4.4×) | 728 (4.5×) |

  Main's two skips are a step discontinuity larger than any other sample step within ±20 ms. The fix's steps match the full decode's. No listening test was done.
- **The spectrogram** shows no clean vertical click lines on main at this resolution, so it is not embedded in the PR.
