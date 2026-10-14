"""A 72 s before/after listening test for mstar-project/mstar#429.

Decodes one long real utterance (``long_talker_codes.safetensors``: 897 Talker
codec frames, 35 chunk boundaries) with the real Code2Wav weights three ways,
using each tree's own streaming code exactly as ``make_audio_evidence.py``
does (its decode worker, CPU, bf16):

  long_full.wav  one decode of all frames (reference)
  long_main.wav  M*'s streaming path at ``--main-ref``
  long_fix.wav   M*'s streaming path at ``--fix-ref``
  long_ab.wav    10 s of main, then the same 10 s of text with the fix, and so
                 on; one beep before each main segment, two before each fix
                 segment, 10 ms fades at every cut

and writes ``long_boundaries.txt`` and ``long_metrics.json``.

    python make_long_listening_test.py --repo . --main-ref upstream/main \\
        --fix-ref fix/code2wav-chunk-trim --out assets/pr429/long
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent if (HERE.parent / "make_audio_evidence.py").exists() else HERE))
from make_audio_evidence import SR, chunk_placement, write_wav  # noqa: E402

EVIDENCE = Path(sys.modules["make_audio_evidence"].__file__)
SEGMENT_S = 10.0
HALF = 480  # 20 ms


def steps_around(x, b, half=HALF):
    """|x[b] - x[b-1]|, and the other |first differences| within 20 ms of b."""
    seg = x[max(b - half, 0):min(b + half, x.size)].astype(np.float64)
    d = np.abs(np.diff(seg))
    k = b - max(b - half, 0) - 1
    return d[k], np.delete(d, k)


def boundary_table(full, main, fix, main_pl, fix_pl):
    """One row per boundary. Main's boundary b_main sits where it jumps from
    global sample (end of chunk) to global sample (start of next chunk); the
    fix's and the full decode's are at the same global sample."""
    rows = []
    for j in range(1, len(fix_pl)):
        b_fix = fix_pl[j][0]
        b_main, g_main, _ = main_pl[j]
        skipped = g_main - (main_pl[j - 1][1] + main_pl[j - 1][2])
        row = {"boundary": j, "fix_s": b_fix / SR, "main_s": b_main / SR, "main_skipped_samples": int(skipped)}
        lo, hi = max(b_fix - HALF, 0), min(b_fix + HALF, full.size)
        row["full_rms_dbfs"] = float(10 * np.log10(np.mean((full[lo:hi] / 32768.0) ** 2) + 1e-12))
        for name, x, b in (("main", main, b_main), ("fix", fix, b_fix), ("full", full, b_fix)):
            step, rest = steps_around(x, b)
            row[name] = {"step": float(step), "median": float(np.median(rest)), "max": float(rest.max()),
                         "ratio": float(step / max(np.median(rest), 1.0)),
                         "above_local_max": bool(step > rest.max())}
        rows.append(row)
    return rows


def global_to_index(placements, g):
    """Where a stream plays global sample g (or the next sample it does play)."""
    for s, g0, n in placements:
        if g < g0:
            return s
        if g < g0 + n:
            return s + (g - g0)
    return placements[-1][0] + placements[-1][2]


def beeps(n):
    t = np.arange(int(0.08 * SR)) / SR
    tone = 0.25 * np.sin(2 * np.pi * 1000 * t) * np.hanning(t.size)
    gap = np.zeros(int(0.06 * SR))
    out = np.concatenate([np.concatenate([tone, gap]) for _ in range(n)])
    return np.concatenate([np.zeros(int(0.15 * SR)), out, np.zeros(int(0.5 * SR) - out.size - int(0.15 * SR))])


def fade(x, ms=10):
    x = x.astype(np.float64) / 32768
    n = int(ms / 1000 * SR)
    ramp = 0.5 - 0.5 * np.cos(np.linspace(0, np.pi, n))
    x[:n] *= ramp
    x[-n:] *= ramp[::-1]
    return x


def ab_track(full_len, main, fix, main_pl, fix_pl):
    parts, segments, g = [], [], 0
    seg = int(SEGMENT_S * SR)
    while g < full_len:
        g_end = min(g + seg, full_len)
        for name, x, pl, n_beeps in (("main", main, main_pl, 1), ("fix", fix, fix_pl, 2)):
            a, b = global_to_index(pl, g), global_to_index(pl, g_end)
            parts += [beeps(n_beeps), fade(x[a:b])]
            segments.append({"stream": name, "text_from_s": g / SR, "text_to_s": g_end / SR,
                             "stream_index_from": int(a), "stream_index_to": int(b)})
        g = g_end
    track = np.concatenate(parts)
    return (np.clip(track, -1, 1) * 32767).astype(np.int16), segments


def measured_lags(x, full, placements, search=600):
    """For each chunk, the lag (global - index) that best correlates the whole
    chunk with the full decode, searched around the trim's lag."""
    out = []
    for s, g, n in placements:
        at, span = s + min(search, n // 4), n - 2 * min(search, n // 4)
        seg = x[at:at + span].astype(np.float64)
        best, best_corr = None, -np.inf
        for d in range(g - s - search, g - s + search + 1):
            if at + d < 0 or at + d + span > full.size:
                continue
            ref = full[at + d:at + d + span].astype(np.float64)
            corr = float(seg @ ref) / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-9)
            if corr > best_corr:
                best, best_corr = d, corr
        out.append(best)
    return out


def pesq_scores(full, main, fix, main_pl, fix_pl):
    """PESQ-WB against the full decode at 16 kHz: over the whole file, and per
    10 s of text with each stream cut at the same text position."""
    try:
        from pesq import pesq
        from scipy.signal import resample_poly
    except ImportError:
        return None

    def score(ref, deg):
        ref, deg = resample_poly(ref / 32768.0, 2, 3), resample_poly(deg / 32768.0, 2, 3)
        deg = deg[:ref.size] if deg.size >= ref.size else np.pad(deg, (0, ref.size - deg.size))
        return float(pesq(16000, ref, deg, "wb"))

    seg = int(SEGMENT_S * SR)
    out = {"whole_file": {}, "per_10s_segment": {}}
    for name, x, pl in (("main", main, main_pl), ("fix", fix, fix_pl)):
        out["whole_file"][name] = score(full, x)
        out["per_10s_segment"][name] = [
            score(full[g:g + seg], x[global_to_index(pl, g):global_to_index(pl, g + seg)])
            for g in range(0, full.size - seg + 1, seg)
        ]
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--main-ref", default="upstream/main")
    parser.add_argument("--fix-ref", required=True)
    parser.add_argument("--codes", type=Path, default=HERE / "long_talker_codes.safetensors")
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()

    from safetensors import safe_open
    with safe_open(str(args.codes), framework="pt") as f:
        (utt,) = f.keys()

    args.out.mkdir(parents=True, exist_ok=True)
    refs = {}
    with tempfile.TemporaryDirectory() as tmp:
        npz = {}
        for name, ref in (("main", args.main_ref), ("fix", args.fix_ref)):
            sha = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", ref], text=True).strip()
            refs[name] = {"ref": ref, "sha": sha}
            src = Path(tmp) / name
            src.mkdir()
            archive = subprocess.run(["git", "-C", str(args.repo), "archive", sha], check=True, capture_output=True)
            subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)
            npz[name] = Path(tmp) / f"{name}.npz"
            subprocess.run(
                [sys.executable, str(EVIDENCE), "--utt", utt, "--fix-ref", args.fix_ref,
                 "--_codes", str(args.codes.resolve()), "--_worker", str(src), str(npz[name])],
                check=True, env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, cwd=tmp,
            )
        m, f = np.load(npz["main"]), np.load(npz["fix"])
        assert np.array_equal(m["full"], f["full"]), "the two trees' full decodes differ"
        full, main_s, fix_s = f["full"], m["stream"], f["stream"]
        up, frames = int(f["upsample"]), int(f["num_frames"])
        main_pl = [p for p in chunk_placement(m["chunk_lens"], m["windows"], up) if p[2]]
        fix_pl = [p for p in chunk_placement(f["chunk_lens"], f["windows"], up) if p[2]]
        decode_info = {"main": json.loads(str(m["info"])), "fix": json.loads(str(f["info"]))}

    tail = frames * up - full.size
    lags = {"main": measured_lags(main_s, full, main_pl), "fix": measured_lags(fix_s, full, fix_pl)}
    # main's chunks start at their new frames; the fix's start ``tail`` earlier
    # and tile the full decode
    assert lags["main"] == [g - s for s, g, _ in main_pl], lags["main"]
    assert lags["fix"] == [0] * len(fix_pl), lags["fix"]
    fix_pl = [(s, s, n) for s, _, n in fix_pl]
    rows = boundary_table(full, main_s, fix_s, main_pl, fix_pl)
    missing = sum(r["main_skipped_samples"] for r in rows)
    speech = [r for r in rows if r["full_rms_dbfs"] > -35]

    def share(rs, name):
        return sum(r[name]["above_local_max"] for r in rs), len(rs)

    metrics = {
        "utterance": utt, "frames": frames, "refs": refs, "decoder_tail_samples": tail,
        "samples": {"full": int(full.size), "main": int(main_s.size), "fix": int(fix_s.size)},
        "boundaries": len(rows),
        "main_missing_samples": missing, "main_missing_s": missing / SR,
        "main_extra_pad_samples_at_end": int(main_s.size - (full.size - missing)),
        "threshold": "boundary step > the largest other sample step within +-20 ms",
        "above_threshold": {k: share(rows, k) for k in ("main", "fix", "full")},
        "above_threshold_speech": {k: share(speech, k) for k in ("main", "fix", "full")},
        "speech_rule": "full-decode RMS within +-20 ms of the boundary > -35 dBFS",
        "median_ratio": {k: float(np.median([r[k]["ratio"] for r in rows])) for k in ("main", "fix", "full")},
        "chunk_lag_vs_full_measured": lags,
        "pesq_wb_vs_full": pesq_scores(full, main_s, fix_s, main_pl, fix_pl),
        "decode": decode_info,
        "per_boundary": rows,
    }

    write_wav(args.out / "long_full.wav", full)
    write_wav(args.out / "long_main.wav", main_s)
    write_wav(args.out / "long_fix.wav", fix_s)
    ab, segments = ab_track(full.size, main_s, fix_s, main_pl, fix_pl)
    write_wav(args.out / "long_ab.wav", ab)
    metrics["ab_segments"] = segments

    lines = [
        f"# {utt}: {frames} codec frames, {full.size / SR:.3f} s. Times in seconds of each file.",
        "# fix and full decode: chunk boundaries every 2 s, 23.1 ms early (the decoder's 555-sample tail).",
        "# main: boundary 1 drops nothing; every later boundary skips 555 samples, so main drifts earlier.",
        "# step ratio: |x[n]-x[n-1]| at the boundary over the median |step| within +-20 ms;",
        "# '*': the step is larger than every other step within +-20 ms.",
        "# rms_dBFS: the full decode's level within +-20 ms (speech above -35).",
        "# boundary  fix_s     main_s    main-fix_ms  rms_dBFS  ratio_main  ratio_fix  ratio_full",
    ]
    for r in rows:
        lag = (r["main_s"] - r["fix_s"]) * 1000
        lines.append(
            f"{r['boundary']:>9d}  {r['fix_s']:8.3f}  {r['main_s']:8.3f}  {lag:11.1f}  {r['full_rms_dbfs']:8.1f}  "
            + "  ".join(f"{r[k]['ratio']:9.1f}{'*' if r[k]['above_local_max'] else ' '}" for k in ("main", "fix", "full"))
        )
    lines += ["", "# long_ab.wav: one beep = main, two beeps = fix; each pair starts at the same text position."]
    lines += [f"# {s['stream']:4s} text {s['text_from_s']:6.2f}-{s['text_to_s']:6.2f} s" for s in segments]
    (args.out / "long_boundaries.txt").write_text("\n".join(lines) + "\n")
    (args.out / "long_metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: o.item()) + "\n")
    print(json.dumps({k: metrics[k] for k in ("samples", "boundaries", "main_missing_samples", "main_missing_s",
                                              "main_extra_pad_samples_at_end", "above_threshold",
                                              "above_threshold_speech", "median_ratio", "pesq_wb_vs_full")},
                     default=lambda o: o.item()))


if __name__ == "__main__":
    main()
