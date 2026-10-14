"""``listen_spectrogram.mp4`` for mstar-project/mstar#429: ``long/long_ab.wav``
with a scrolling log-mel spectrogram of what is playing.

The A/B track is rebuilt from ``long/long_metrics.json``'s ``ab_segments``
with ``make_long_listening_test.py``'s own beep and fade code and asserted
bit-identical to ``long_ab.wav``, so every boundary drawn sits on the audio
sample it is at. Boundaries in a ``main`` segment are red when
``make_boundary_figures.py`` calls them a glitch (speech, and main's +-50 ms
SNR at least 10 dB below this PR's), grey otherwise; a main boundary that
falls on an A/B cut is never played and is not drawn. Boundaries in a
this-PR segment are green.

    python make_listen_video.py [--out listen_spectrogram.mp4] [--jobs N]

Needs numpy, matplotlib, librosa and ffmpeg (libx264, aac).
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "long"))
import make_boundary_figures as mbf  # noqa: E402
from make_long_listening_test import beeps, fade  # noqa: E402

W, H, FPS = 1280, 720, 30
WINDOW_S = 2.5
LEAD = 12000  # samples of beep lead-in before each segment
# Paul Tol's vibrant red and teal
BEFORE_C, AFTER_C, QUIET_C = "#CC3311", "#009988", "#9A9A9A"
LABEL = {"main": "BEFORE (main)", "fix": "AFTER (this PR)"}

_state: dict = {}


def build() -> dict:
    d = mbf.load()
    rows = mbf.boundary_rows(d)
    meta = json.loads((mbf.LONG / "long_metrics.json").read_text())
    ab = mbf.read_wav(mbf.LONG / "long_ab.wav")
    streams = {"main": d["main"], "fix": d["fix"]}
    raw = {k: np.round(v * 32768).astype(np.int16) for k, v in streams.items()}

    segs, parts, pos = [], [], 0
    for s in meta["ab_segments"]:
        a, b = s["stream_index_from"], s["stream_index_to"]
        n_beeps = 1 if s["stream"] == "main" else 2
        parts += [beeps(n_beeps), fade(raw[s["stream"]][a:b])]
        segs.append(dict(stream=s["stream"], a=a, b=b, lead=pos, start=pos + LEAD, end=pos + LEAD + b - a))
        pos += LEAD + b - a
    rebuilt = (np.clip(np.concatenate(parts), -1, 1) * 32767).astype(np.int16)
    assert np.array_equal(rebuilt.astype(np.float64) / 32768.0, ab), "segment map does not reproduce long_ab.wav"

    # main index -> full-decode index, per chunk
    lag = np.zeros(d["main"].size, np.int64)
    for s, n, l in d["chunks"]:
        lag[s:s + n] = l

    events = []
    for seg in segs:
        for r, bf, bm in zip(rows, d["b_ref"], d["b_main"]):
            p = bm if seg["stream"] == "main" else bf
            if not seg["a"] < p < seg["b"]:
                continue
            if seg["stream"] == "fix":
                kind, tag = "after", "no skip"
            elif r["glitch_main"]:
                kind, tag = "glitch", "23 ms skipped" if r["main_skipped_samples"] else "23 ms from padding"
            else:
                kind, tag = "quiet", "skip, quiet"
            events.append(dict(t=(seg["start"] + p - seg["a"]) / mbf.SR, boundary=r["boundary"], kind=kind, tag=tag))

    top = mbf.db_top(d)
    spec = mbf.logmel(ab) - top
    return dict(ab=ab, segs=segs, events=events, spec=spec, lag=lag, dur=ab.size / mbf.SR)


def source_time(st: dict, k: int) -> tuple[dict, float, bool]:
    """Segment, source time (s of the 72 s reply) and whether sample k is in the beep lead-in."""
    seg = next((s for s in st["segs"] if s["lead"] <= k < s["end"]), st["segs"][-1])
    i = min(max(k - seg["start"], 0) + seg["a"], seg["b"] - 1)
    g = i + (st["lag"][i] if seg["stream"] == "main" else 0)
    return seg, g / mbf.SR, k < seg["start"]


def _init(st: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    _state.update(st)
    fig = plt.figure(figsize=(W / 100, H / 100), dpi=100, facecolor="white")
    ban = fig.add_axes((0, 0.855, 1, 0.145))
    ban.set_axis_off()
    ban.set_xlim(0, 1)
    ban.set_ylim(0, 1)
    bg = ban.add_patch(Rectangle((0, 0), 1, 1, color=BEFORE_C, alpha=0.12, lw=0))
    title = ban.text(0.02, 0.6, "", fontsize=30, fontweight="bold", va="center")
    when = ban.text(0.98, 0.62, "", fontsize=20, va="center", ha="right", family="DejaVu Sans Mono")
    sub = ban.text(0.02, 0.17, "", fontsize=11.5, va="center", color="#333333")
    count = ban.text(0.98, 0.17, "", fontsize=11.5, va="center", ha="right", color="#333333")

    ax = fig.add_axes((0.075, 0.255, 0.905, 0.565))
    dt = mbf.HOP / mbf.SR
    n = st["spec"].shape[1]
    ax.imshow(st["spec"], origin="lower", aspect="auto", cmap="bone", vmin=-mbf.DB_RANGE, vmax=0,
              interpolation="nearest", extent=(-dt / 2, (n - 0.5) * dt, -0.5, mbf.N_MELS - 0.5))
    mbf.mel_axis(ax)
    ax.set_ylabel("mel frequency (Hz)", fontsize=11)
    ax.tick_params(labelsize=9.5)
    colour = {"glitch": BEFORE_C, "quiet": QUIET_C, "after": AFTER_C}
    for e in st["events"]:
        c = colour[e["kind"]]
        ax.axvline(e["t"], color=c, lw=3 if e["kind"] != "quiet" else 2,
                   ls="-" if e["kind"] != "quiet" else (0, (5, 3)), ymax=0.9)
        ax.text(e["t"], mbf.N_MELS * 0.955, f"{e['tag']}  #{e['boundary']}", color="white", fontsize=10.5,
                fontweight="bold", ha="center", va="center", clip_on=True,
                bbox=dict(boxstyle="round,pad=0.25", fc=c, ec="none"))
    ax.plot([0.5, 0.5], [0, 1], transform=ax.transAxes, color="#FFD700", lw=1.6, zorder=5)
    ax.set_ylim(-0.5, mbf.N_MELS - 0.5)
    ax.set_xlabel("seconds from the playhead (yellow line)", fontsize=10)

    strip = fig.add_axes((0.075, 0.07, 0.905, 0.085))
    for s in st["segs"]:
        strip.axvspan(s["start"] / mbf.SR, s["end"] / mbf.SR, color=BEFORE_C if s["stream"] == "main" else AFTER_C,
                      alpha=0.18, lw=0)
    for e in st["events"]:
        strip.plot([e["t"]] * 2, [0.05, 0.95] if e["kind"] != "quiet" else [0.3, 0.7], color=colour[e["kind"]],
                   lw=1.6 if e["kind"] != "quiet" else 1.0)
    strip.set_xlim(0, st["dur"])
    strip.set_ylim(0, 1)
    strip.set_yticks([])
    ticks = np.arange(0, st["dur"], 15)
    strip.set_xticks(ticks)
    strip.set_xticklabels([f"{int(t) // 60}:{int(t) % 60:02d}" for t in ticks], fontsize=9)
    strip.set_title("whole video: red = BEFORE (main), green = AFTER (this PR); ticks = chunk boundaries",
                    fontsize=9.5, loc="left", pad=3)
    head = strip.axvline(0, color="black", lw=2)
    _state.update(fig=fig, ax=ax, bg=bg, title=title, when=when, sub=sub, count=count, head=head)


def _frame(k: int) -> bytes:
    st = _state
    t = k / FPS
    seg, src, lead = source_time(st, int(round(t * mbf.SR)))
    who = LABEL[seg["stream"]]
    c = BEFORE_C if seg["stream"] == "main" else AFTER_C
    st["title"].set_text(f"NEXT: {who}" if lead else f"NOW PLAYING: {who}")
    st["title"].set_color(c)
    st["bg"].set_color(c)
    st["bg"].set_alpha(0.06 if lead else 0.14)
    st["when"].set_text(f"t = {src:4.1f} s of the 72 s reply")
    st["sub"].set_text("one beep = BEFORE, two beeps = AFTER; each pair plays the same 10 s of the reply")
    heard = sum(e["kind"] == "glitch" and e["t"] <= t for e in st["events"])
    st["count"].set_text(f"speech glitches heard so far: {heard} in BEFORE, 0 in AFTER")
    ax = st["ax"]
    ax.set_xlim(t - WINDOW_S / 2, t + WINDOW_S / 2)
    ticks = np.arange(-1.0, 1.01, 0.25)
    ax.set_xticks(t + ticks)
    ax.set_xticklabels([f"{x:+.2f}".replace("+0.00", "0") for x in ticks])
    st["head"].set_xdata([t, t])
    fig = st["fig"]
    fig.canvas.draw()
    return bytes(np.asarray(fig.canvas.buffer_rgba())[:, :, :3])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE / "listen_spectrogram.mp4")
    parser.add_argument("--jobs", type=int, default=max(mp.cpu_count() // 2, 1))
    parser.add_argument("--frames", type=str, default="", help="comma-separated frame numbers: write PNGs only")
    args = parser.parse_args()

    st = build()
    n_frames = int(np.ceil(st["dur"] * FPS))
    print(json.dumps({"duration_s": st["dur"], "frames": n_frames,
                      "glitch": [e["boundary"] for e in st["events"] if e["kind"] == "glitch"],
                      "quiet": [e["boundary"] for e in st["events"] if e["kind"] == "quiet"],
                      "after": [e["boundary"] for e in st["events"] if e["kind"] == "after"]}))
    if args.frames:
        import matplotlib.image
        _init(st)
        for k in map(int, args.frames.split(",")):
            img = np.frombuffer(_frame(k), np.uint8).reshape(H, W, 3)
            matplotlib.image.imsave(args.out.with_name(f"{args.out.stem}_frame{k:05d}.png"), img)
        return

    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-loglevel", "error", "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "-", "-i", str(mbf.LONG / "long_ab.wav"), "-map", "0:v", "-map", "1:a",
         "-c:v", "libx264", "-preset", "slow", "-crf", "23", "-tune", "animation", "-pix_fmt", "yuv420p",
         "-g", str(2 * FPS), "-c:a", "aac", "-b:a", "128k", "-movflags", "+faststart", str(args.out)],
        stdin=subprocess.PIPE)
    with mp.get_context("fork").Pool(args.jobs, initializer=_init, initargs=(st,)) as pool:
        for i, buf in enumerate(pool.imap(_frame, range(n_frames), chunksize=16)):
            ff.stdin.write(buf)
            if i % 600 == 0:
                print(f"frame {i}/{n_frames}", flush=True)
    ff.stdin.close()
    assert ff.wait() == 0


if __name__ == "__main__":
    main()
