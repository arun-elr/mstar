"""Boundary figures for mstar-project/mstar#429, from the long WAVs in ``long/``.

  spectrogram_boundaries.png  log-mel zoom (+-150 ms) on three speech boundaries:
                              full decode / main / this PR, and the aligned
                              differences |main - full| and |this PR - full|
  boundary_error_strip.png    local SNR against the full decode (+-50 ms) at
                              all 35 boundaries, main vs this PR

Alignment: ``main``'s chunks are placed back at their position in the full
decode (chunk 1 at 0, chunk 2 at 0, chunk k at 555*(k-2) samples later than
where main plays it). Each offset is re-measured by cross-correlation and
asserted. Placed this way, main is bit-identical to this PR's output except
the 555 samples it skips at each of boundaries 2-35 (left as zeros, i.e.
missing) and the last 555 samples of chunk 1 (decoded from padding); this is
asserted too.

    python make_boundary_figures.py [--out DIR]

Needs numpy, scipy, matplotlib and librosa.
"""

from __future__ import annotations

import argparse
import wave
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
LONG = HERE / "long"
SR = 24000
CHUNK = 25 * 1920  # samples per 25-frame chunk
TAIL = 555  # Code2Wav's right-edge tail: 50*1920 - len(decode of 50 frames)
SNR_HALF = 1200  # +-50 ms
RMS_HALF = 480  # +-20 ms, the speech rule of make_long_listening_test.py
SPEECH_DBFS = -35.0
GLITCH_DB = 10.0  # main's local SNR this far below this PR's counts as a glitch
ZOOM = (3, 7, 11)
ZOOM_HALF_S = 0.150

# log-mel settings, shared with make_listen_video.py
N_FFT, HOP, N_MELS, FMAX, DB_RANGE = 512, 120, 128, 12000.0, 80.0
MEL_TICKS_HZ = (100, 250, 500, 1000, 2000, 4000, 8000, 12000)

# Okabe-Ito
MAIN_C, FIX_C, GREY = "#D55E00", "#0072B2", "#999999"


def read_wav(path: Path) -> np.ndarray:
    with wave.open(str(path)) as w:
        assert w.getframerate() == SR and w.getnchannels() == 1 and w.getsampwidth() == 2
        return np.frombuffer(w.readframes(w.getnframes()), "<i2").astype(np.float64) / 32768.0


def main_chunks(n_main: int) -> list[tuple[int, int, int]]:
    """(start in main, length, lag) per chunk; full-decode index = main index + lag."""
    out, s, k = [], 0, 0
    while s < n_main:
        n = CHUNK if k == 0 else min(CHUNK - TAIL, n_main - s)
        out.append((s, n, 0 if k < 2 else TAIL * (k - 1)))
        s, k = s + n, k + 1
    return out


def measured_lag(x: np.ndarray, ref: np.ndarray, s: int, n: int, max_lag: int) -> int:
    seg = x[s + 2400:s + n - 2400]
    r = ref[s + 2400:min(s + n - 2400 + max_lag, ref.size)]
    m = r.size + seg.size
    xc = np.fft.irfft(np.fft.rfft(r, m) * np.conj(np.fft.rfft(seg, m)), m)
    return int(np.argmax(xc[:r.size - seg.size + 1]))


def load() -> dict:
    ref, main, fix = (read_wav(LONG / f"long_{n}.wav") for n in ("full", "main", "fix"))
    assert ref.size == fix.size
    chunks = main_chunks(main.size)
    placed, have = np.zeros_like(ref), np.zeros(ref.size, bool)
    for s, n, lag in chunks:
        if s + n + lag <= ref.size:
            assert measured_lag(main, ref, s, n, TAIL * len(chunks)) == lag, (s, lag)
        n = min(n, ref.size - s - lag)
        placed[s + lag:s + lag + n] = main[s:s + n]
        have[s + lag:s + lag + n] = True
    n_b = ref.size // CHUNK
    b_ref = np.array([j * CHUNK - TAIL for j in range(1, n_b + 1)])
    gaps = [(b, b + TAIL) for b in b_ref[1:]]
    assert np.array_equal(np.flatnonzero(~have), np.concatenate([np.arange(a, b) for a, b in gaps]))
    same = have.copy()
    same[CHUNK - TAIL:CHUNK] = False
    assert np.array_equal(placed[same], fix[same]), "main (placed) differs from this PR outside the skips"
    b_main = np.array([chunks[j][0] for j in range(1, n_b + 1)])
    return dict(ref=ref, main=main, fix=fix, placed=placed, chunks=chunks, b_ref=b_ref, b_main=b_main)


def local_snr(x: np.ndarray, ref: np.ndarray, b: int, half: int = SNR_HALF) -> float:
    r = ref[b - half:b + half]
    return float(10 * np.log10(np.sum(r ** 2) / np.sum((x[b - half:b + half] - r) ** 2)))


def boundary_rows(d: dict) -> list[dict]:
    ref, rows = d["ref"], []
    for j, (b, bm) in enumerate(zip(d["b_ref"], d["b_main"]), 1):
        rms = 10 * np.log10(np.mean(ref[b - RMS_HALF:b + RMS_HALF] ** 2) + 1e-12)
        snr_m, snr_f = local_snr(d["placed"], ref, b), local_snr(d["fix"], ref, b)
        speech = bool(rms > SPEECH_DBFS)
        rows.append({
            "boundary": j, "full_and_fix_s": round(b / SR, 6), "main_s": round(bm / SR, 6),
            "main_skipped_samples": 0 if j == 1 else TAIL, "full_rms_dbfs": round(float(rms), 2),
            "speech": speech, "snr_db_main": round(snr_m, 2), "snr_db_fix": round(snr_f, 2),
            "glitch_main": speech and snr_m <= snr_f - GLITCH_DB,
        })
    return rows


def logmel(x: np.ndarray) -> np.ndarray:
    """Power mel spectrogram in dB re full scale; frame k is centred on sample k*HOP."""
    import librosa
    m = librosa.feature.melspectrogram(y=x.astype(np.float32), sr=SR, n_fft=N_FFT, hop_length=HOP,
                                       n_mels=N_MELS, fmax=FMAX, power=2.0, center=True)
    return 10 * np.log10(np.maximum(m, 1e-20))


def mel_axis(ax) -> None:
    """Label a mel-bin y axis (bin index 0..N_MELS-1) in Hz."""
    import librosa
    centres = librosa.mel_frequencies(n_mels=N_MELS, fmax=FMAX)
    ax.set_yticks(np.interp(MEL_TICKS_HZ, centres, np.arange(N_MELS)))
    ax.set_yticklabels([f"{f // 1000}k" if f >= 1000 else str(f) for f in MEL_TICKS_HZ])


def db_top(d: dict) -> float:
    return float(logmel(d["ref"]).max())


def fig_spectrogram(d: dict, rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    top = db_top(d)
    vmin, vmax = -DB_RANGE, 0.0
    specs = {k: logmel(x) - top for k, x in (("ref", d["ref"]), ("main", d["main"]), ("fix", d["fix"]),
                                               ("dmain", d["placed"] - d["ref"]), ("dfix", d["fix"] - d["ref"]))}
    labels = [("ref", "full decode\n(reference)"), ("main", "main\n(as played)"), ("fix", "this PR"),
              ("dmain", "|main − ref|\n(aligned)"), ("dfix", "|this PR − ref|")]
    half = int(round(ZOOM_HALF_S * SR / HOP))
    fig, axes = plt.subplots(len(labels), len(ZOOM), figsize=(1400 / 150, 1260 / 150), dpi=150,
                             sharex=True, sharey=True, gridspec_kw=dict(hspace=0.12, wspace=0.05))
    for c, j in enumerate(ZOOM):
        r = rows[j - 1]
        b, bm = d["b_ref"][j - 1], d["b_main"][j - 1]
        for i, (key, label) in enumerate(labels):
            ax = axes[i, c]
            centre = bm if key == "main" else b
            k0 = int(round(centre / HOP))
            cols = specs[key][:, k0 - half:k0 + half + 1]
            t = (np.arange(k0 - half, k0 + half + 1) * HOP - centre) / SR * 1e3
            dt = HOP / SR * 1e3
            im = ax.imshow(cols, origin="lower", aspect="auto", cmap="magma", vmin=vmin, vmax=vmax,
                           interpolation="nearest", extent=(t[0] - dt / 2, t[-1] + dt / 2, -0.5, N_MELS - 0.5))
            ax.axvline(0, color="white", ls=(0, (4, 3)), lw=1.1)
            if key in ("ref", "fix", "dmain", "dfix"):
                ax.axvspan(0, TAIL / SR * 1e3, ymin=0.94, ymax=1.0, color="#56B4E9", lw=0)
            if c == 0:
                ax.set_ylabel(label, fontsize=8)
            ax.tick_params(labelsize=7)
        axes[0, c].set_title(f"boundary {j}: {r['full_and_fix_s']:.3f} s\n(main plays it at {r['main_s']:.3f} s)",
                             fontsize=8.5)
        axes[-1, c].set_xlabel("ms from the boundary", fontsize=8)
    mel_axis(axes[0, 0])
    fig.text(0.012, 0.5, "mel frequency (Hz)", rotation=90, va="center", fontsize=8.5)
    axes[0, 0].text(-4, N_MELS * 0.97, "23.1 ms main skips →", color="white",
                    fontsize=6.5, va="top", ha="right")
    fig.subplots_adjust(left=0.085, right=0.905, top=0.925, bottom=0.05)
    top_ax, bottom_ax = axes[0, -1].get_position(), axes[-1, -1].get_position()
    cax = fig.add_axes((0.918, bottom_ax.y0, 0.012, top_ax.y1 - bottom_ax.y0))
    cb = fig.colorbar(im, cax=cax)
    cb.set_label(f"dB re loudest reference bin (same {DB_RANGE:.0f} dB scale in every panel)", fontsize=7.5)
    cb.ax.tick_params(labelsize=7)
    fig.suptitle("Code2Wav streaming at three speech chunk boundaries: log-mel spectrogram, 24 kHz, "
                 f"n_fft {N_FFT}, hop {HOP} (5 ms), {N_MELS} mels to {FMAX / 1000:.0f} kHz",
                 fontsize=9.5, x=0.5, y=0.995)
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def fig_strip(rows: list[dict], out: Path) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    js = np.array([r["boundary"] for r in rows])
    fig, ax = plt.subplots(figsize=(1400 / 150, 480 / 150), dpi=150)
    for r in rows:
        if r["speech"]:
            ax.axvspan(r["boundary"] - 0.5, r["boundary"] + 0.5, color="#E8E8E8", lw=0, zorder=0)
    ax.plot(js, [r["snr_db_fix"] for r in rows], "s", color=FIX_C, ms=5.5, label="this PR", zorder=3)
    ax.plot(js, [r["snr_db_main"] for r in rows], "o", color=MAIN_C, ms=6, mfc="white", mew=1.6,
            label="main", zorder=4)
    ax.set_xlim(0.5, len(rows) + 0.5)
    ax.set_xticks(js)
    ax.tick_params(labelsize=7.5)
    ax.set_xlabel("chunk boundary (every 2 s of the 72 s reply)", fontsize=8.5)
    ax.set_ylabel("local SNR vs full decode\n(dB, ±50 ms)", fontsize=8.5)
    ax.set_ylim(0, max(r["snr_db_fix"] for r in rows) + 3)
    ax.grid(axis="y", color="#DDDDDD", lw=0.6)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    handles, labels = ax.get_legend_handles_labels()
    handles.append(Patch(color="#E8E8E8", label="speech boundary (full decode > −35 dBFS within ±20 ms)"))
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=3, frameon=False, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=150, facecolor="white")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=HERE)
    args = parser.parse_args()
    d = load()
    rows = boundary_rows(d)
    fig_spectrogram(d, rows, args.out / "spectrogram_boundaries.png")
    fig_strip(rows, args.out / "boundary_error_strip.png")
    for r in rows:
        print(f"{r['boundary']:2d} {r['full_and_fix_s']:7.3f} {r['main_s']:7.3f} {r['full_rms_dbfs']:6.1f} "
              f"{'speech' if r['speech'] else 'quiet ':6s} main {r['snr_db_main']:5.1f} fix {r['snr_db_fix']:5.1f}"
              f"{'  GLITCH' if r['glitch_main'] else ''}")


if __name__ == "__main__":
    main()
