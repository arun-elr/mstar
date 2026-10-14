"""Audio evidence for mstar-project/mstar#429: Code2Wav chunk-boundary trim.

Decodes one utterance of real Talker codes (the PR's fixture
``test/integration/data/qwen3_omni_talker_codes.safetensors``) with the real
Qwen3-Omni Code2Wav weights three ways:

  full_decode.wav  one decode of all frames (reference)
  main_stream.wav  M*'s streaming path at ``--main-ref``
  fix_stream.wav   M*'s streaming path at ``--fix-ref``

Each streaming path runs that tree's own code, imported from a ``git archive``
of the ref in a separate interpreter: the edge's ``LeftContextChunkPolicy``
(25 new + 25 context frames) and ``StreamBuffer``, then ``Code2WavSubmodule``'s
``prepare_inputs`` (pad to 50 frames) / ``preprocess`` / ``forward_batched`` /
``postprocess`` (trim). The decoder runs on CPU in the checkpoint's bf16,
eager rather than ``torch.compile``d. M*'s three Triton kernels (RMSNorm, RoPE,
sliding-window attention) are replaced by torch equivalents, the same ones the
PR's CPU tests use.

Writes the three 24 kHz 16-bit mono WAVs, ``boundary_zoom.png``,
``error_vs_full.png``, ``spectrogram.png`` and ``metrics.json`` to ``--out``.

    python make_audio_evidence.py --repo . --main-ref upstream/main \\
        --fix-ref fix/code2wav-chunk-trim --out assets/pr429

Needs torch, safetensors, numpy, matplotlib, M*'s dependencies, and the
Qwen3-Omni checkpoint's Code2Wav shard in the local Hugging Face cache
(``HF_HUB_CACHE`` / ``HF_HOME`` / ``~/.cache/huggingface/hub``).
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import time
import wave
from pathlib import Path

import numpy as np

HF_REPO = "Qwen/Qwen3-Omni-30B-A3B-Instruct"
CODES = "test/integration/data/qwen3_omni_talker_codes.safetensors"
SR = 24000
# Okabe-Ito
FULL_C, MAIN_C, FIX_C = "#000000", "#D55E00", "#0072B2"


def find_snapshot() -> Path:
    repo_dir = f"models--{HF_REPO.replace('/', '--')}"
    roots = []
    if os.environ.get("HF_HUB_CACHE"):
        roots.append(Path(os.environ["HF_HUB_CACHE"]))
    if os.environ.get("HF_HOME"):
        roots.append(Path(os.environ["HF_HOME"]) / "hub")
    roots.append(Path.home() / ".cache" / "huggingface" / "hub")
    for root in roots:
        snapshots = root / repo_dir / "snapshots"
        for snapshot in sorted(snapshots.iterdir()) if snapshots.is_dir() else []:
            index = snapshot / "model.safetensors.index.json"
            if not index.is_file():
                continue
            shards = {s for k, s in json.loads(index.read_text())["weight_map"].items() if k.startswith("code2wav.")}
            if shards and all((snapshot / s).is_file() for s in shards):
                return snapshot
    raise SystemExit(f"{HF_REPO} Code2Wav weights are not in the local Hugging Face cache")


# ---------------------------------------------------------------------------
# Worker: decode with one tree's code
# ---------------------------------------------------------------------------

def decode_worker(src: Path, codes_path: Path, utt: str, dtype_name: str, out: Path) -> None:
    sys.path.insert(0, str(src))
    import torch
    import torch.nn.functional as F
    from safetensors.torch import load_file

    import mstar
    from mstar.model.qwen3_omni.components import code2wav as code2wav_module
    from mstar.model.qwen3_omni.components.code2wav import Qwen3OmniMoeCode2Wav
    from mstar.model.qwen3_omni.config import Code2WavConfig, Qwen3OmniModelConfig
    from mstar.model.qwen3_omni.submodules import Code2WavSubmodule
    from mstar.model.submodule_base import InputMetadata, ModelInputsFromEngine
    from mstar.model.utils import ModuleAndPrefix, load_weights_from_hf_shards
    from mstar.streaming.chunk_policy import LeftContextChunkPolicy
    from mstar.streaming.stream_buffer import StreamBuffer

    assert Path(mstar.__file__).resolve().is_relative_to(src.resolve()), mstar.__file__
    torch.set_num_threads(os.cpu_count() or 1)

    def rms_norm(x, weight, eps=1e-6, out=None):
        x32 = x.float()
        return (x32 * torch.rsqrt(x32.pow(2).mean(-1, keepdim=True) + eps) * weight.float()).to(x.dtype)

    def apply_rope_pos_ids(q, k, position_ids, rope_theta=10000.0):
        half = q.shape[-1] // 2
        inv_freq = rope_theta ** (-torch.arange(half, dtype=torch.float32) / half)
        angle = position_ids.float()[:, None, None] * inv_freq
        cos, sin = angle.cos(), angle.sin()
        for t in (q, k):
            x1, x2 = t[..., :half].float(), t[..., half:].float()
            t.copy_(torch.cat((x1 * cos - x2 * sin, x2 * cos + x1 * sin), dim=-1))
        return q, k

    def sliding_window_attn(q, k, v, window, scale=None):
        pos = torch.arange(q.shape[1])
        allowed = (pos[None, :] <= pos[:, None]) & (pos[:, None] - pos[None, :] < window)
        out = F.scaled_dot_product_attention(
            *(t.transpose(1, 2).float() for t in (q, k, v)), attn_mask=allowed, scale=scale,
        )
        return out.transpose(1, 2).to(q.dtype)

    code2wav_module.rms_norm = rms_norm
    code2wav_module.apply_rope_pos_ids = apply_rope_pos_ids
    code2wav_module.sliding_window_attn = sliding_window_attn

    snapshot = find_snapshot()
    config = Code2WavConfig.from_dict(json.loads((snapshot / "config.json").read_text())["code2wav_config"])
    model = Qwen3OmniMoeCode2Wav(config)
    load_weights_from_hf_shards(repo_dir=snapshot, modules=[ModuleAndPrefix(model, prefix="code2wav")], device="cpu")
    model.eval()
    model.consolidate()
    # consolidate() wraps forward in torch.compile; run it eager on CPU
    del model.forward
    model.to(dtype=getattr(torch, dtype_name))
    param_dtypes = sorted({str(p.dtype) for p in model.parameters()} | {str(b.dtype) for b in model.buffers()})

    frames = load_file(str(codes_path))[utt].long()
    num_frames = frames.shape[0]
    t0 = time.time()
    with torch.no_grad():
        full = model(frames.t()[None], torch.arange(num_frames)[None])[0, 0]
    full = (full.clamp(-1, 1) * 32767).to(torch.int16)
    t_full = time.time() - t0

    submodule = Code2WavSubmodule(model, Qwen3OmniModelConfig(code2wav=config))
    cfg = submodule.config.code2wav
    buffer = StreamBuffer(
        request_id="r", edge_name="codec_tokens", from_partition="Talker",
        policy=LeftContextChunkPolicy(chunk=cfg.codec_chunk_frames, left_context=cfg.codec_left_context_frames),
    )
    for i, frame in enumerate(frames):
        buffer.pre_read_register(f"t{i}")
        buffer.put(f"t{i}", frame)
    buffer.signal_done()

    fwd_info = type("FwdInfo", (), {"request_id": "r", "rid_handle": "r"})()
    chunks, windows = [], []
    t0 = time.time()
    while buffer.has_chunk_ready():
        chunk = buffer.pop_chunk()
        meta = {"r": InputMetadata(stream_chunks={"codec_tokens": chunk.info})}
        # None names the captured 50-frame graph; anything else runs eager
        key = submodule.cg_key_info("code2wav_chunk", {"r": fwd_info}, per_request_input_metadata=meta)
        engine_inputs = ModelInputsFromEngine(
            request_ids=["r"], per_request_info={}, per_request_input_metadata=meta, captured=key is None,
        )
        item = submodule.prepare_inputs(
            "code2wav_chunk", fwd_info, {"codec_tokens": [chunk.data["data"]]}, input_metadata=meta["r"],
        )
        with torch.no_grad():
            outputs = submodule.forward_batched(
                "code2wav_chunk", engine_inputs, **submodule.preprocess("code2wav_chunk", engine_inputs, [item]),
            )["r"]
        submodule.postprocess("r", fwd_info, outputs, inputs=item)
        audio = outputs["audio_chunk"][0].reshape(-1) if "audio_chunk" in outputs else torch.empty(0, dtype=torch.int16)
        chunks.append(audio.numpy())
        windows.append((chunk.info.num_items, chunk.info.context_items))
    t_stream = time.time() - t0

    np.savez(
        out, full=full.numpy(), chunk_lens=np.array([c.size for c in chunks]),
        stream=np.concatenate(chunks), windows=np.array(windows), num_frames=num_frames,
        upsample=submodule.total_upsample,
        info=json.dumps({"param_dtypes": param_dtypes, "t_full_s": t_full, "t_stream_s": t_stream,
                         "torch": torch.__version__}),
    )


# ---------------------------------------------------------------------------
# Analysis and figures
# ---------------------------------------------------------------------------

def write_wav(path: Path, pcm: np.ndarray) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes(pcm.astype("<i2").tobytes())


def chunk_placement(chunk_lens, windows, upsample):
    """For each emitted chunk: (index in the stream, global sample its first
    sample decodes). The window's new frames start at frame ``new_start``; a
    chunk's audio starts there less however much the trim moved back."""
    placements, stream_pos, new_start = [], 0, 0
    for length, (num_items, context) in zip(chunk_lens, windows):
        placements.append((stream_pos, new_start * upsample, int(length)))
        stream_pos += int(length)
        new_start += num_items - context
    return placements


def best_lag(stream, full, at, span=4800, max_lag=2000):
    """Lag d maximising correlation of stream[at:at+span] with full[at+d:...]."""
    seg = stream[at:at + span].astype(np.float64)
    best, best_corr = 0, -np.inf
    for d in range(-max_lag, max_lag + 1):
        lo = at + d
        if lo < 0 or lo + span > full.size:
            continue
        ref = full[lo:lo + span].astype(np.float64)
        corr = float(seg @ ref) / (np.linalg.norm(seg) * np.linalg.norm(ref) + 1e-9)
        if corr > best_corr:
            best, best_corr = d, corr
    return best, best_corr


def jump_stats(x, b, half=480):
    """|x[b] - x[b-1]| against the |first difference| within 20 ms of b,
    excluding the boundary step itself."""
    x = x.astype(np.float64)
    d = np.abs(np.diff(x[max(b - half, 0):min(b + half, x.size)]))
    k = b - max(b - half, 0) - 1
    step = d[k]
    rest = np.delete(d, k)
    return {"step": step, "local_median": float(np.median(rest)), "local_max": float(rest.max()),
            "step_over_median": step / max(float(np.median(rest)), 1.0)}


def local_snr(stream, full, win=240):
    n = min(stream.size, full.size) // win * win
    sig = np.mean(full.astype(np.float64) ** 2)
    err = (stream[:n].astype(np.float64) - full[:n].astype(np.float64)).reshape(-1, win)
    t = (np.arange(n // win) + 0.5) * win / SR
    return t, 10 * np.log10(sig / np.maximum(np.mean(err ** 2, axis=1), 1e-3))


def analyse_and_plot(main_npz, fix_npz, out: Path, utt: str, refs: dict) -> dict:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})
    m, f = np.load(main_npz), np.load(fix_npz)
    assert np.array_equal(m["full"], f["full"]), "the two trees' full decodes differ"
    full, main, fix = f["full"], m["stream"], f["stream"]
    up = int(f["upsample"])
    tail = int(f["num_frames"]) * up - full.size
    main_pl = chunk_placement(m["chunk_lens"], m["windows"], up)
    fix_pl = chunk_placement(f["chunk_lens"], f["windows"], up)
    main_pl = [(s, g, n) for s, g, n in main_pl if n]
    fix_pl = [(s, g, n) for s, g, n in fix_pl if n]

    # Where each main chunk lands, measured by correlation with the full decode
    lags = [best_lag(main, full, s + n // 4)[0] for s, _, n in main_pl]
    main_bounds = [s for s, _, _ in main_pl[1:]]
    fix_bounds = [s for s, _, _ in fix_pl[1:]]
    full_bounds = fix_bounds  # the fix's chunks tile the full decode exactly

    jumps = {
        "main": [jump_stats(main, b) for b in main_bounds],
        "fix": [jump_stats(fix, b) for b in fix_bounds],
        "full": [jump_stats(full, b) for b in full_bounds],
    }

    def snr(a, b):
        n = min(a.size, b.size)
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        return 10 * np.log10(np.sum(b ** 2) / np.sum((a - b) ** 2))

    metrics = {
        "utterance": utt, "frames": int(f["num_frames"]), "refs": refs,
        "decoder_tail_samples": tail,
        "samples": {"full": int(full.size), "main": int(main.size), "fix": int(fix.size)},
        "chunk_samples": {"main": [int(n) for n in m["chunk_lens"]], "fix": [int(n) for n in f["chunk_lens"]]},
        "windows_frames_context": {"main": m["windows"].tolist(), "fix": f["windows"].tolist()},
        "main_chunk_measured_lag_vs_full": lags,
        "main_chunk_trim_lag_vs_full": [g - s for s, g, _ in main_pl],
        "boundaries_s": {"main": [b / SR for b in main_bounds], "fix": [b / SR for b in fix_bounds]},
        "snr_vs_full_db": {"main": snr(main, full), "fix": snr(fix, full)},
        # main's first chunk ends with ``tail`` samples that looked ahead into the zero padding
        "snr_last_tail_of_first_chunk_db": {
            name: snr(x[fix_bounds[0]:fix_bounds[0] + tail], full[fix_bounds[0]:fix_bounds[0] + tail])
            for name, x in (("main", main), ("fix", fix))
        },
        "boundary_jumps": jumps,
        "decode": {"main": json.loads(str(m["info"])), "fix": json.loads(str(f["info"]))},
    }

    for name, pcm in (("full_decode", full), ("main_stream", main), ("fix_stream", fix)):
        write_wav(out / f"{name}.wav", pcm)

    # 1. Boundary zoom: the first boundary between two full 50-frame windows
    b = fix_bounds[1]
    assert main_bounds[1] == b
    half = 480
    idx = np.arange(b - half, b + half)
    ms = (idx - b) / SR * 1000
    skipped = lags[2] - lags[1]
    fig, axes = plt.subplots(3, 1, figsize=(11, 6.4), sharex=True, sharey=True, dpi=100)
    rows = [("full decode (reference)", full, FULL_C, None),
            ("main, streamed", main, MAIN_C, f"plays full[{b - 1}] then full[{b + skipped}]: {skipped} samples "
             f"({skipped / SR * 1000:.1f} ms) skipped"),
            ("fix, streamed", fix, FIX_C, "continues with full[%d]" % b)]
    for ax, (label, x, color, note) in zip(axes, rows):
        if x is not full:
            ax.plot(ms, full[idx] / 32768, color="#999999", lw=1.0, ls="--", label="full decode")
        ax.plot(ms, x[idx] / 32768, color=color, lw=1.4, label=label)
        ax.axvline(0, color="#444444", lw=0.8, ls=":")
        ax.set_title(f"{label}: {note}" if note else label, loc="left", fontsize=10, color=color)
        ax.set_ylabel("amplitude")
    axes[0].axvspan(0, half / SR * 1000, color=MAIN_C, alpha=0.12, lw=0)
    axes[0].set_title(f"full decode (reference); shaded: part of the {skipped / SR * 1000:.1f} ms main never plays",
                      loc="left", fontsize=10)
    axes[1].legend(loc="lower left", fontsize=8, frameon=False)
    axes[-1].set_xlabel(f"ms from the chunk boundary at sample {b} ({b / SR:.3f} s)")
    fig.suptitle(f"Qwen3-Omni Code2Wav, real codes ({utt}, {int(f['num_frames'])} frames): "
                 f"boundary between streamed windows 2 and 3", fontsize=11)
    fig.tight_layout()
    fig.savefig(out / "boundary_zoom.png")
    plt.close(fig)

    # 2. Local error vs the full decode, by sample index
    fig, ax = plt.subplots(figsize=(11, 3.6), dpi=100)
    for label, x, color in (("main", main, MAIN_C), ("fix", fix, FIX_C)):
        t, s = local_snr(x, full)
        ax.plot(t, s, color=color, lw=1.2, label=label)
    for bb in full_bounds:
        ax.axvline(bb / SR, color="#444444", lw=0.8, ls=":")
    for (s, _, n), lag in zip(main_pl, lags):
        ax.text((s + n / 2) / SR, 1.02, f"main offset {lag:+d}", transform=ax.get_xaxis_transform(),
                ha="center", fontsize=8.5, color=MAIN_C)
    ax.set_xlim(0, full.size / SR)
    ax.set_xlabel("time (s), sample index of each stream; dotted: chunk boundaries")
    ax.set_ylabel("SNR vs full decode,\n10 ms windows (dB)")
    ax.legend(loc="lower left", frameon=False, ncol=2)
    fig.tight_layout()
    fig.savefig(out / "error_vs_full.png")
    plt.close(fig)

    # 3. Spectrograms on one scale
    def spec(x, n_fft=512, hop=64):
        x = x.astype(np.float64) / 32768
        frames = np.lib.stride_tricks.sliding_window_view(x, n_fft)[::hop] * np.hanning(n_fft)
        return 20 * np.log10(np.abs(np.fft.rfft(frames, axis=1)).T + 1e-7)

    specs = [spec(x) for x in (main, fix)]
    vmax = max(s.max() for s in specs)
    fig, axes = plt.subplots(2, 1, figsize=(11, 5.6), sharex=True, dpi=100)
    for ax, s, label, bounds, color in zip(axes, specs, ("main", "fix"), (main_bounds, fix_bounds), (MAIN_C, FIX_C)):
        ax.imshow(s, origin="lower", aspect="auto", cmap="viridis", vmin=vmax - 90, vmax=vmax,
                  extent=(0, s.shape[1] * 64 / SR, 0, SR / 2000))
        ax.plot([bb / SR for bb in bounds], [SR / 2000 * 0.985] * len(bounds), "v", color="white", ms=7)
        ax.set_title(f"{label}: streamed (white markers: chunk boundaries)", loc="left", fontsize=10, color=color)
        ax.set_ylabel("kHz")
    axes[-1].set_xlabel("time (s)")
    fig.tight_layout()
    fig.savefig(out / "spectrogram.png")
    plt.close(fig)

    (out / "metrics.json").write_text(json.dumps(metrics, indent=2, default=lambda o: o.item()) + "\n")
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--main-ref", default="upstream/main")
    parser.add_argument("--fix-ref", required=True)
    parser.add_argument("--utt", default="utt5_T83", help="fixture key (the longest: 3 chunk boundaries)")
    parser.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float32"])
    parser.add_argument("--out", type=Path, default=Path("assets/pr429"))
    parser.add_argument("--_worker", nargs=2, metavar=("SRC", "OUT"), help=argparse.SUPPRESS)
    parser.add_argument("--_codes", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()

    if args._worker:
        src, npz = map(Path, args._worker)
        decode_worker(src, args._codes or src / CODES, args.utt, args.dtype, npz)
        return

    args.out.mkdir(parents=True, exist_ok=True)
    refs = {}
    with tempfile.TemporaryDirectory() as tmp:
        npzs = {}
        for name, ref in (("main", args.main_ref), ("fix", args.fix_ref)):
            sha = subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", ref], text=True).strip()
            refs[name] = {"ref": ref, "sha": sha}
            src = Path(tmp) / name
            src.mkdir()
            archive = subprocess.run(["git", "-C", str(args.repo), "archive", sha], check=True, capture_output=True)
            subprocess.run(["tar", "-x", "-C", str(src)], input=archive.stdout, check=True)
            npzs[name] = Path(tmp) / f"{name}.npz"
        # the fixture exists only in the fix tree
        codes = Path(tmp) / "fix" / CODES
        for name in ("main", "fix"):
            src = Path(tmp) / name
            if not (src / CODES).exists():
                (src / CODES).parent.mkdir(parents=True, exist_ok=True)
                (src / CODES).write_bytes(codes.read_bytes())
            # CPU only; a fresh interpreter so each tree imports its own mstar
            env = {**os.environ, "CUDA_VISIBLE_DEVICES": ""}
            subprocess.run(
                [sys.executable, __file__, "--utt", args.utt, "--dtype", args.dtype,
                 "--fix-ref", args.fix_ref, "--_worker", str(src), str(npzs[name])],
                check=True, env=env, cwd=tmp,
            )
        metrics = analyse_and_plot(npzs["main"], npzs["fix"], args.out, args.utt, refs)
    print(json.dumps({k: metrics[k] for k in ("samples", "chunk_samples", "main_chunk_measured_lag_vs_full",
                                              "boundaries_s", "snr_vs_full_db", "snr_last_tail_of_first_chunk_db")},
                     indent=1, default=lambda o: o.item()))


if __name__ == "__main__":
    main()
