# 72 s listening test for mstar-project/mstar#429

**Start with `long_ab.wav`.** It plays 10 s of `main`, then the same 10 s of text with the fix, and so on through the passage. One beep comes before each `main` segment and two before each fix segment. Every cut has a 10 ms fade, so the cuts add no clicks of their own.

| File | What it is |
|---|---|
| `long_ab.wav` | Back-to-back A/B, 2.5 min |
| `long_full.wav` | One decode of all 897 frames (the reference), 71.74 s |
| `long_main.wav` | M*'s streaming path at `upstream/main` 99f5108b, 70.97 s |
| `long_fix.wav` | M*'s streaming path at the PR head 00bec2ba, 71.74 s |
| `long_boundaries.txt` | Every boundary's timestamp in each file, with its step ratio |
| `long_metrics.json` | Every number below, per boundary |
| `long_talker_codes.safetensors` | The 897 codec frames |
| `make_long_listening_test.py` | Regenerates all of the above, using `../make_audio_evidence.py` |

All files are 24 kHz, mono, 16-bit PCM.

## Where the codes come from

They are one real reply from M*'s Qwen3-Omni Talker. M* `upstream/main` 99f5108b served it on a B200 with `configs/qwen3omni_colocated.yaml`, voice Ethan, default sampling. The request was a single TTS call: "Read the following passage aloud exactly as written, with nothing else:" followed by the passage below. The Thinker read it back word for word. The codes are the Code2Wav input windows, recorded by an import hook (no M* source edits) and stitched using the 25-frame overlap. They are one utterance, with no joins. The server's own streamed audio for this request has exactly `long_main.wav`'s length (1,703,370 samples) and is 39.9 dB from it, which is the bf16 floor. So `long_main.wav` is what M* plays today.

> A bicycle turns the motion of the rider's legs into forward travel. The pedals are fixed to cranks, and the cranks turn a toothed ring called the chainring. A metal chain runs from the chainring to a smaller set of sprockets on the rear wheel. When the rider pushes down on a pedal, the chain pulls the rear sprocket around, and the wheel rolls forward. Gears let the rider choose how far the bicycle travels for each turn of the pedals. A large chainring with a small rear sprocket covers more ground with each stroke but needs more force, which suits flat roads. A small chainring with a large rear sprocket needs less force, which helps on steep hills. Brakes press rubber pads or metal discs to slow the wheels. The frame, usually made of steel, aluminium, or carbon fibre, holds every part in place and carries the rider's weight. Tyres filled with air soften the bumps from the road and grip the surface. A bell or a light helps other people notice the rider. Regular care, such as oiling the chain and checking the tyre pressure, keeps the bicycle running smoothly for many years. Many cities now build separate lanes so that people can ride to work safely.

The three WAVs are decoded exactly as in `../README.md`: each tree's own chunking and trim code, the real Code2Wav weights, CPU, bf16.

## Measured

- **Length.** The full decode and the fix are both 1,721,685 samples (897·1920 − 555). `main` is 1,703,370. It skips 555 samples at each of boundaries 2–35, 18,870 samples or **0.786 s** of speech in all, and then emits 555 samples decoded from padding at the end. Cross-correlating every chunk against the full decode confirms where each lands. `main`'s chunks sit at offsets of 0, 0, 555, 1110, … up to 18,870 samples, as the trim arithmetic predicts. The fix's chunks sit at 0.
- **Boundary steps.** For each boundary, take |x[n] − x[n−1]| at the boundary sample and divide it by the median step within ±20 ms. A boundary counts as **clearly above normal** when its step is larger than every other step within ±20 ms.

  | | `main` | fix | full decode, same points |
  |---|---|---|---|
  | Median step ratio over 35 boundaries | 12.4× | 1.2× | 1.1× |
  | Clearly above normal, all 35 | **14 / 35** | 0 / 35 | 0 / 35 |
  | Clearly above normal, in speech (18; full decode above −35 dBFS within ±20 ms) | **10 / 18** | 0 / 18 | 0 / 18 |
  | Clearly above normal, in pauses or quiet stretches (17) | 4 / 17 | 0 / 17 | 0 / 17 |

- **PESQ-WB against the full decode** (16 kHz; the maximum is about 4.64). Over the whole file: `main` 3.59, fix 4.55. Over 10 s segments cut at the same text position: `main` 4.01 on average (3.73–4.33), fix 4.55 (4.51–4.57). The whole-file score for `main` also counts its growing misalignment. The per-segment score mostly counts the five skips in each segment.
- **Verdict:** audible at some boundaries (voiced segments). `main`'s step stands out from its surroundings at 10 of the 18 boundaries that fall in speech and at 4 of the 17 in pauses. Elsewhere it is a 23 ms timing gap, and over the passage those gaps remove 0.79 s of speech. This verdict rests on the step statistic and PESQ; no formal listening test was done.
