"""Output equivalence of every decode configuration over a whole latent stream.

  python quality.py decode --variant px_nopad_bench --latents L.pt --out DIR [--vae wan21] [--tag run2]
  python quality.py analyze --dir DIR --out results/b200 [--vae wan21] [--source-video CLIP]
  python quality.py coverage --latents L.pt --out results/b200 [--vae wan21]

``decode`` (one variant per process, like the timing benchmark) streams the
first latent plus every 3-latent chunk of ``L.pt`` through the variant and
writes its frames: ``<variant>.u8.npy`` (uint8 ``[F, H, W, 3]``, the
conversion both M* decoder nodes use) and, for variants that expose the
decoder's float output, ``<variant>.f.npy`` (``[F, 3, H, W]``; bf16 bit
patterns as int16, or fp32 for the reference), so the analysis sees exact
values. The M* node variants return uint8 frames only.

``analyze`` compares every variant with stock diffusers bf16
(``diffusers_bf16``, same dtype) and with the fp32 reference (``ref_fp32``,
TF32 off): max / mean abs error, share of uint8 values that differ, per-frame
PSNR (uint8, peak 255), SSIM (Gaussian 11x11, sigma 1.5, per RGB channel),
LPIPS (AlexNet, ``lpips`` package), and temporal flicker: the mean
|(f_t - f_t-1) - (s_t - s_t-1)| between a variant's frame-to-frame change and
the reference's, in uint8 levels. Drift and seams: the PSNR trend over chunks
and the PSNR of each chunk's first frame against the chunk's other frames.
Bit-exactness: the md5 of each uint8 frame stream and the values / pixels
that differ from stock, from the previous ladder step, and (M* node rows) from
``diffusers_bf16_node_input``, stock fed the node's own bf16 de-normalized
input. Worst frame (minimum PSNR / SSIM) and where it sits in its chunk.
Determinism: ``<variant>.run2`` files (a second fresh process) compared bit for
bit with the first. With ``--source-video``, the PSNR against the clip the
latents were encoded from (encode + decode), over the first encoded piece. Writes ``quality.csv`` and
``quality_frames.csv`` (for ``plot_results.py``'s ``per_frame_psnr.png``).

``coverage`` drives the in-tree node over irregular windows (the clip-start
latent alone, ragged tails, served-size windows) and compares each stream
with one stock decode of the same input: ``coverage.csv``.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import sys
from pathlib import Path

import bench_wan_decoder as B
import numpy as np
import torch
from common import FRAMES_PER_CHUNK, WAN21_REPO, WAN22_REPO, latent_stats, load_vae, mstar_path, to_uint8

STOCK, REF = "diffusers_bf16", "ref_fp32"
# Stock diffusers bf16 fed the latents exactly as M*'s decoder node de-normalizes
# them (in bf16): the reference that isolates the M* rows' decode from that step.
STOCK_NODE = "diffusers_bf16_node_input"
NODE_ROWS = ("mstar_ar_ctx8", "mstar_warm", "mstar_warm_nobench")
# (label, variant): the change ladder in order, then the in-tree node.
STEPS = [
    ("M* before (re-decode 8 context latents)", "mstar_ar_ctx8"),
    ("1. keep conv cache (+ compile)", "diffusers_bf16_compile"),
    ("2. + channels-last", "channels_last_compile"),
    ("3. + one call per chunk", "channels_last_one_call_compile"),
    ("4. + native conv padding", "px_nopad"),
    ("5. + cudnn.benchmark", "px_nopad_bench"),
    ("in-tree node (mstar_warm)", "mstar_warm"),
    ("in-tree node, cudnn.benchmark off (control)", "mstar_warm_nobench"),
]
# What each row is compared with as "the previous step".
PREV = {v: p for (_, p), (_, v) in zip(STEPS, STEPS[1:5], strict=False)}
PREV |= {"px_nopad_bench": "px_nopad", "mstar_warm": "px_nopad_bench", "mstar_warm_nobench": "mstar_warm"}


# --------------------------------------------------------------------------- decode

class FrameWriter:
    """Appends ``[1, 3, T, H, W]`` decoder outputs to the variant's .npy files."""

    def __init__(self, out: Path, name: str, n_frames: int, h: int, w: int, float_dtype):
        self.u8 = np.lib.format.open_memmap(out / f"{name}.u8.npy", "w+", np.uint8, (n_frames, h, w, 3))
        self.f = None
        if float_dtype is not None:
            np_dtype = np.float32 if float_dtype == torch.float32 else np.int16
            self.f = np.lib.format.open_memmap(out / f"{name}.f.npy", "w+", np_dtype, (n_frames, 3, h, w))
        self.n = 0

    def add(self, video: torch.Tensor):
        """``video``: float in [-1, 1], or uint8 (M* nodes)."""
        t = video.shape[2]
        u8 = video if video.dtype == torch.uint8 else to_uint8(video)
        self.u8[self.n: self.n + t] = u8[0].permute(1, 2, 3, 0).contiguous().cpu().numpy()
        if self.f is not None:
            f = video[0].permute(1, 0, 2, 3).contiguous().cpu()
            self.f[self.n: self.n + t] = (f if f.dtype == torch.float32 else f.view(torch.int16)).numpy()
        self.n += t

    def close(self):
        self.u8.flush()
        if self.f is not None:
            self.f.flush()


def decode(args):
    B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    torch._dynamo.config.cache_size_limit = 64
    raw = torch.load(args.latents)["raw"]
    n_chunks = args.chunks or (raw.shape[2] - 1) // 3
    n_frames = 1 + n_chunks * FRAMES_PER_CHUNK
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = args.variant + (f".{args.tag}" if args.tag else "")
    v = args.variant
    with torch.inference_mode():
        if B.node_overrides(v):
            over, compiled = B.node_overrides(v)
            _, start, chunk = B.mstar_warm_node(raw, args.device, compiled=compiled, n_steady=n_chunks,
                                                overrides=over)
            px = start()
            writer = FrameWriter(out, name, n_frames, px.shape[3], px.shape[4], None)
            writer.add(px)
            for c in range(n_chunks):
                writer.add(chunk(c))
        elif v.startswith("mstar_ar_ctx"):
            sys.path.insert(0, mstar_path())
            os.environ["COSMOS3_COMPILE_VAE"] = "0" if v.endswith("_eager") else "1"
            from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderSubmodule

            ctx = int(v.removeprefix("mstar_ar_ctx").removesuffix("_eager"))
            vae = load_vae(B.REPO, torch.float32, args.device)
            sub = Cosmos3VAEDecoderSubmodule(vae=vae, config=B.mstar_redecode_config())
            mean, std = latent_stats(vae, "cpu")
            norm = ((raw - mean) / std).to(args.device)
            px = sub._decode_pixels(norm[:, :, :1])
            writer = FrameWriter(out, name, n_frames, px.shape[3], px.shape[4], None)
            writer.add(px)
            tail = norm[:, :, :1]
            for c in range(n_chunks):
                stream_tail = torch.cat([tail, norm[:, :, 1 + 3 * c: 4 + 3 * c]], dim=2)
                writer.add(sub._decode_pixels(stream_tail)[:, :, -FRAMES_PER_CHUNK:])
                tail = stream_tail[:, :, -ctx:]
        else:
            if v == STOCK_NODE:
                sd, dtype = B.build_streaming(STOCK, args.device)
                z = node_input(raw, args.device)
            else:
                sd, dtype = B.build_streaming(v, args.device)
                z = raw.to(args.device, dtype)
            sd.reset()
            px = sd.first(z[:, :, :1])
            writer = FrameWriter(out, name, n_frames, px.shape[3], px.shape[4], dtype)
            writer.add(px)
            for c in range(n_chunks):
                writer.add(sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]))
    assert writer.n == n_frames, (writer.n, n_frames)
    writer.close()
    meta = {"variant": v, "tag": args.tag, "vae": B.REPO, "frames": n_frames, "chunks": n_chunks,
            "latents": str(args.latents), "precision_flags": B.precision_flags(), "env": B.env_info()}
    (out / f"{name}.meta.json").write_text(json.dumps(meta, indent=2))
    print(json.dumps({k: meta[k] for k in ("variant", "tag", "vae", "frames")}))


def plain_node(device, **overrides):
    """``Cosmos3VAEDecoderARSubmodule`` on its own fp32-loaded VAE (the node casts
    it to the decode dtype), eager unless ``compiled``; ``overrides`` set config."""
    sys.path.insert(0, mstar_path())
    os.environ["COSMOS3_COMPILE_VAE"] = "1" if overrides.pop("compiled", False) else "0"
    from mstar.model.cosmos3.config import Cosmos3Config
    from mstar.model.cosmos3.submodules import Cosmos3VAEDecoderARSubmodule

    vae = load_vae(B.REPO, torch.float32, device)
    cfg = Cosmos3Config()
    for k, val in overrides.items():
        setattr(cfg, k, val)
    return Cosmos3VAEDecoderARSubmodule(vae=vae, config=cfg), vae


def node_input(raw, device):
    """The VAE input M*'s node computes from the DiT's normalized latents."""
    sub, vae = plain_node(device, vae_decode_channels_last=False)
    mean, std = latent_stats(vae, "cpu")
    z = sub._vae_input(((raw - mean) / std).to(device)).contiguous()
    del sub, vae
    return z


def coverage(args):
    """The in-tree node over irregular windows (the clip-start latent alone, a
    ragged tail, served-size windows) against one stock decode of the same
    stream, both fed the node's own VAE input: with the warm cache only
    (eager, default layout, one latent per call, no cudnn.benchmark) and with
    this change's defaults (compiled)."""
    from types import SimpleNamespace

    sys.path.insert(0, mstar_path())
    from mstar.model.cosmos3 import constants as C

    B.REPO = WAN21_REPO if args.vae == "wan21" else WAN22_REPO
    torch._dynamo.config.cache_size_limit = 64
    splits = [(1, 3, 3, 3, 3, 2), (1, 2, 5, 4, 3), (8, 6, 1), (1,)]
    n_lat = max(sum(sp) for sp in splits)
    raw = torch.load(args.latents)["raw"][:, :, :n_lat]
    configs = {
        "warm cache only (eager, default layout, 1 latent per call, no cudnn.benchmark)": dict(
            windowed_decode_warm_cache=True, vae_decode_channels_last=False, windowed_decode_latents_per_call=1,
            vae_decode_cudnn_benchmark=False),
        "this change's defaults (compiled)": dict(
            compiled=True, windowed_decode_warm_cache=True, vae_decode_channels_last=True,
            windowed_decode_latents_per_call=0, vae_decode_cudnn_benchmark=True),
    }
    z = node_input(raw, args.device)
    stock = load_vae(B.REPO, torch.bfloat16, args.device)
    rows = []
    with torch.inference_mode():
        refs = {}
        for sp in splits:
            n = sum(sp)
            if n not in refs:
                refs[n] = to_uint8(stock.decode(z[:, :, :n]).sample).cpu()
        del stock
        for cname, over in configs.items():
            sub, vae = plain_node(args.device, **over)
            mean, std = latent_stats(vae, "cpu")
            norm = ((raw - mean) / std).to(args.device)
            for sp in splits:
                n = sum(sp)
                md = {"num_windows": len(sp), "overlap_latent_units": 0, "num_frames": 1 + (n - 1) * 4,
                      "stream_video": True}
                infos = SimpleNamespace(request_ids=["cov"],
                                        per_request_info={"cov": SimpleNamespace(step_metadata=md)})
                sub.cleanup_request("cov")
                outs, a = [], 0
                for w in sp:
                    outs.append(sub.forward(C.VIDEO_DECODE_AR_WALK, infos, norm[:, :, a: a + w])["video_output"][0])
                    a += w
                got = torch.cat(outs, dim=2).cpu()
                ref = refs[n]
                assert got.shape == ref.shape, (got.shape, ref.shape)
                d = (got.int() - ref.int()).abs()
                mse = float((d.double() ** 2).mean())
                rows.append({
                    "vae": args.vae, "config": cname, "windows": "+".join(map(str, sp)), "latents": n,
                    "frames": got.shape[2], "identical_to_stock": bool(d.max() == 0),
                    "u8_values_diff": int((d > 0).sum()), "max_abs_u8": int(d.max()),
                    "psnr_u8_vs_stock": round(10 * math.log10(255**2 / mse), 2) if mse else "inf",
                })
                print(json.dumps(rows[-1]))
            del sub, vae
            torch.cuda.empty_cache()
    append_csv(Path(args.out) / "coverage.csv", rows, args.vae)


def compare(args):
    """md5 of each uint8 frame stream in ``args.chain`` (in order) and what
    changes from the stream before it and from ``args.ref``: one row per stream."""
    d = Path(args.dir)
    names = args.chain.split(",")
    ref = np.load(d / f"{args.ref}.u8.npy", mmap_mode="r") if args.ref else None
    rows, prev = [], None
    for v in names:
        p = d / f"{v}.u8.npy"
        if not p.exists():
            print(f"missing {p}", file=sys.stderr)
            prev = None
            continue
        x = np.load(p, mmap_mode="r")
        n = x.size
        row = {"vae": args.vae, "variant": v, "md5_u8_stream": md5(x), "frames": x.shape[0]}
        for tag, other in (("previous", prev), ("ref", ref)):
            if other is None:
                row |= {f"identical_to_{tag}": "", f"pct_values_diff_vs_{tag}": "", f"pct_pixels_diff_vs_{tag}": ""}
                continue
            vals, pix = count_diff(x, other)
            row |= {f"identical_to_{tag}": vals == 0, f"pct_values_diff_vs_{tag}": round(100 * vals / n, 4),
                    f"pct_pixels_diff_vs_{tag}": round(100 * pix / (n // 3), 4)}
        row["previous"] = names[names.index(v) - 1] if prev is not None else ""
        row["ref"] = args.ref
        rows.append(row)
        print(json.dumps(row))
        prev = x
    append_csv(Path(args.out), rows, args.vae)


# --------------------------------------------------------------------------- analyze

def gaussian_window(device, size=11, sigma=1.5):
    x = torch.arange(size, device=device, dtype=torch.float32) - size // 2
    g = torch.exp(-(x**2) / (2 * sigma**2))
    g = g / g.sum()
    return (g[:, None] * g[None, :]).expand(3, 1, size, size).contiguous()


def ssim(a: torch.Tensor, b: torch.Tensor, win: torch.Tensor) -> torch.Tensor:
    """Per-frame SSIM of uint8-valued float frames ``[N, 3, H, W]`` (range 255)."""
    import torch.nn.functional as F

    c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
    conv = lambda x: F.conv2d(x, win, groups=3)  # noqa: E731
    mu_a, mu_b = conv(a), conv(b)
    s_aa = conv(a * a) - mu_a**2
    s_bb = conv(b * b) - mu_b**2
    s_ab = conv(a * b) - mu_a * mu_b
    m = ((2 * mu_a * mu_b + c1) * (2 * s_ab + c2)) / ((mu_a**2 + mu_b**2 + c1) * (s_aa + s_bb + c2))
    return m.mean(dim=(1, 2, 3))


def psnr_frames(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    mse = ((a - b) ** 2).mean(dim=(1, 2, 3))
    return 10 * torch.log10(255.0**2 / mse.clamp_min(1e-12)).masked_fill(mse == 0, float("inf"))


def mean_or_inf(t: torch.Tensor) -> float:
    """Mean over the finite values (identical frames have infinite PSNR)."""
    fin = t[torch.isfinite(t)]
    return float(fin.mean()) if len(fin) else math.inf


def load_float(d: Path, name: str) -> np.ndarray | None:
    p = d / f"{name}.f.npy"
    return np.load(p, mmap_mode="r") if p.exists() else None


def as_float(arr: np.ndarray, sl, device) -> torch.Tensor:
    t = torch.from_numpy(np.ascontiguousarray(arr[sl]))
    return (t.view(torch.bfloat16) if t.dtype == torch.int16 else t).to(device).float()


def md5(arr: np.ndarray) -> str:
    h = hashlib.md5()
    for i in range(0, arr.shape[0], 64):
        h.update(np.ascontiguousarray(arr[i: i + 64]).tobytes())
    return h.hexdigest()


def count_diff(a: np.ndarray, b: np.ndarray) -> tuple[int, int]:
    """(uint8 values, pixels) that differ between two frame streams."""
    vals = pix = 0
    for i in range(0, a.shape[0], 64):
        ne = np.asarray(a[i: i + 64]) != np.asarray(b[i: i + 64])
        vals += int(ne.sum())
        pix += int(ne.any(axis=-1).sum())
    return vals, pix


def worst(t: torch.Tensor) -> tuple[float, int]:
    i = int(torch.argmin(t))
    return float(t[i]), i


def frame_place(i: int) -> str:
    """'chunk c, frame k/12' and whether the frame touches a chunk boundary."""
    if i == 0:
        return "first latent"
    c, k = divmod(i - 1, FRAMES_PER_CHUNK)
    edge = " (boundary)" if k in (0, FRAMES_PER_CHUNK - 1) else ""
    return f"chunk {c}, frame {k + 1}/{FRAMES_PER_CHUNK}{edge}"


# prepare_latents.py encodes the clip in independent pieces of 1 + 4 * 30 frames;
# only the first piece lines up frame for frame with the decoded stream.
ENCODE_PIECE = 1 + 4 * 30


def source_frames(args, n_frames: int) -> np.ndarray | None:
    """The clip's first encoded piece, preprocessed as prepare_latents.py does."""
    if not args.source_video:
        return None
    from prepare_latents import ping_pong, read_frames

    n = min(n_frames, ENCODE_PIECE)
    return ping_pong(read_frames(args.source_video, 832, 480, n), n)


def analyze(args):
    import lpips

    d, out = Path(args.dir), Path(args.out)
    dev = args.device
    names = [v for _, v in STEPS] + [STOCK, STOCK_NODE]
    u8 = {v: np.load(d / f"{v}.u8.npy", mmap_mode="r") for v in names + [REF] if (d / f"{v}.u8.npy").exists()}
    fl = {v: load_float(d, v) for v in names + [REF]}
    n_frames = u8[STOCK].shape[0]
    src = source_frames(args, n_frames)
    stock_md5 = md5(u8[STOCK])
    win = gaussian_window(dev)
    net = lpips.LPIPS(net="alex", verbose=False).to(dev).eval()
    bs = args.batch
    per_frame = {}
    rows = []
    for label, v in [(lab, v) for lab, v in STEPS if v in u8] + [("stock diffusers bf16, eager", STOCK)]:
        acc = {k: [] for k in ("psnr_ref", "psnr_stock", "ssim_ref", "ssim_stock", "lpips_ref", "lpips_stock",
                               "psnr_src")}
        diff1 = diff2 = total = 0
        max_u8 = 0
        f_max = f_sum = 0.0
        f_n = 0
        flick_stock = flick_ref = 0.0
        prev = None
        for s in range(0, n_frames, bs):
            sl = slice(s, min(s + bs, n_frames))
            x = torch.from_numpy(np.ascontiguousarray(u8[v][sl])).to(dev).permute(0, 3, 1, 2).float()
            st = torch.from_numpy(np.ascontiguousarray(u8[STOCK][sl])).to(dev).permute(0, 3, 1, 2).float()
            rf = torch.from_numpy(np.ascontiguousarray(u8[REF][sl])).to(dev).permute(0, 3, 1, 2).float()
            ad = (x - st).abs()
            diff1 += int((ad >= 1).sum())
            diff2 += int((ad >= 2).sum())
            total += ad.numel()
            max_u8 = max(max_u8, int(ad.max()))
            if fl.get(v) is not None:
                fd = (as_float(fl[v], sl, dev) - as_float(fl[STOCK], sl, dev)).abs()
                f_max = max(f_max, float(fd.max()))
                f_sum += float(fd.double().sum())
                f_n += fd.numel()
            acc["psnr_ref"].append(psnr_frames(x, rf))
            if src is not None and s < len(src):
                k = min(sl.stop, len(src)) - s
                sf = torch.from_numpy(np.ascontiguousarray(src[s: s + k])).to(dev).permute(0, 3, 1, 2).float()
                acc["psnr_src"].append(psnr_frames(x[:k], sf))
            acc["psnr_stock"].append(psnr_frames(x, st))
            acc["ssim_ref"].append(ssim(x, rf, win))
            acc["ssim_stock"].append(ssim(x, st, win))
            with torch.no_grad():
                to_l = lambda t: t / 127.5 - 1  # noqa: E731
                acc["lpips_ref"].append(net(to_l(x), to_l(rf)).flatten())
                acc["lpips_stock"].append(net(to_l(x), to_l(st)).flatten())
            # Temporal flicker against stock / the reference, frame-to-frame deltas.
            cat_x = x if prev is None else torch.cat([prev[0], x])
            cat_s = st if prev is None else torch.cat([prev[1], st])
            cat_r = rf if prev is None else torch.cat([prev[2], rf])
            dx, ds, dr = cat_x.diff(dim=0), cat_s.diff(dim=0), cat_r.diff(dim=0)
            flick_stock += float((dx - ds).abs().double().sum())
            flick_ref += float((dx - dr).abs().double().sum())
            prev = (x[-1:], st[-1:], rf[-1:])
        a = {k: torch.cat(vs).cpu().double() for k, vs in acc.items() if vs}
        per_frame[v] = a
        n_delta = (n_frames - 1) * 3 * x.shape[2] * x.shape[3]
        steady = a["psnr_ref"][1:].view(-1, FRAMES_PER_CHUNK)  # [chunks, 12], first frame excluded
        chunk_mean = steady.mean(dim=1).numpy()
        slope = float(np.polyfit(np.arange(len(chunk_mean)), chunk_mean, 1)[0])
        seam = float((steady[:, 0] - steady[:, 1:].mean(dim=1)).mean())
        h = md5(u8[v])
        p = PREV.get(v)
        prev_vals, prev_pix = count_diff(u8[v], u8[p]) if p in u8 else ("", "")
        stock_vals, stock_pix = (0, 0) if h == stock_md5 else count_diff(u8[v], u8[STOCK])
        node_ref = u8.get(STOCK_NODE) if v in NODE_ROWS else None
        node_vals, node_pix = count_diff(u8[v], node_ref) if node_ref is not None else ("", "")
        worst_ps, worst_ps_i = worst(a["psnr_stock"])
        worst_pr, worst_pr_i = worst(a["psnr_ref"])
        worst_ss, worst_ss_i = worst(a["ssim_stock"])
        worst_sr, worst_sr_i = worst(a["ssim_ref"])
        rows.append({
            "vae": args.vae, "step": label, "variant": v, "md5_u8_stream": h,
            "identical_to_stock": h == stock_md5,
            "u8_values_diff_vs_stock": stock_vals, "pixels_diff_vs_stock": stock_pix,
            "u8_values_diff_vs_stock_node_input": node_vals, "pixels_diff_vs_stock_node_input": node_pix,
            "previous_step": p or "",
            "identical_to_previous": (prev_vals == 0) if p in u8 else "",
            "u8_values_diff_vs_previous": prev_vals, "pixels_diff_vs_previous": prev_pix,
            "max_abs_float_vs_stock": round(f_max, 5) if f_n else "",
            "mean_abs_float_vs_stock": f"{f_sum / f_n:.2e}" if f_n else "",
            "max_abs_u8_vs_stock": max_u8,
            "pct_u8_diff_ge1_vs_stock": round(100 * diff1 / total, 3),
            "pct_u8_diff_ge2_vs_stock": round(100 * diff2 / total, 4),
            "psnr_u8_vs_fp32": round(mean_or_inf(a["psnr_ref"]), 2),
            "psnr_u8_vs_stock": round(mean_or_inf(a["psnr_stock"]), 2),
            "worst_psnr_u8_vs_stock": round(worst_ps, 2), "worst_psnr_vs_stock_at": frame_place(worst_ps_i),
            "worst_psnr_u8_vs_fp32": round(worst_pr, 2), "worst_psnr_vs_fp32_at": frame_place(worst_pr_i),
            "worst_ssim_vs_stock": round(worst_ss, 5), "worst_ssim_vs_stock_at": frame_place(worst_ss_i),
            "worst_ssim_vs_fp32": round(worst_sr, 5), "worst_ssim_vs_fp32_at": frame_place(worst_sr_i),
            "psnr_u8_vs_source_first_121": round(float(a["psnr_src"].mean()), 2) if "psnr_src" in a else "",
            "frames_identical_to_stock": int((~torch.isfinite(a["psnr_stock"])).sum()),
            "ssim_vs_fp32": round(float(a["ssim_ref"].mean()), 5),
            "ssim_vs_stock": round(float(a["ssim_stock"].mean()), 5),
            "lpips_vs_fp32": f"{float(a['lpips_ref'].mean()):.2e}",
            "lpips_vs_stock": f"{float(a['lpips_stock'].mean()):.2e}",
            "flicker_vs_stock_u8": round(flick_stock / n_delta, 4),
            "flicker_vs_fp32_u8": round(flick_ref / n_delta, 4),
            "psnr_trend_db_per_chunk": round(slope, 4),
            "seam_db": round(seam, 3),
            "deterministic_run2": determinism(d, v),
            "frames": n_frames,
        })
    out.mkdir(parents=True, exist_ok=True)
    append_csv(out / "quality.csv", rows, args.vae)
    frame_rows = [{"vae": args.vae, "variant": v, "frame": i,
                   "psnr_u8_vs_fp32": round(float(a["psnr_ref"][i]), 3),
                   "psnr_u8_vs_stock": round(float(a["psnr_stock"][i]), 3)}
                  for v, a in per_frame.items() for i in range(n_frames)]
    append_csv(out / "quality_frames.csv", frame_rows, args.vae)
    for r in rows:
        print(json.dumps(r))


def determinism(d: Path, v: str) -> str:
    """'' (not checked), 'bit-identical', or the max abs difference of a second process."""
    a, b = d / f"{v}.u8.npy", d / f"{v}.run2.u8.npy"
    if not b.exists():
        return ""
    fa, fb = load_float(d, v), load_float(d, f"{v}.run2")
    if fa is not None and fb is not None:
        same = np.array_equal(fa, fb)
        if same:
            return "bit-identical"
        diff = max(float((as_float(fa, slice(i, i + 64), "cpu") - as_float(fb, slice(i, i + 64), "cpu")).abs().max())
                   for i in range(0, fa.shape[0], 64))
        return f"max abs float diff {diff:.3g}"
    ua, ub = np.load(a, mmap_mode="r"), np.load(b, mmap_mode="r")
    if np.array_equal(ua, ub):
        return "bit-identical (uint8)"
    return f"max uint8 diff {int(np.abs(ua.astype(np.int16) - ub.astype(np.int16)).max())}"


def append_csv(path: Path, rows: list[dict], vae: str):
    """Replace this VAE's rows in ``path``, keeping the other VAE's."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keep = []
    if path.exists():
        with open(path) as f:
            keep = [r for r in csv.DictReader(f) if r["vae"] != vae]
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n", extrasaction="ignore")
        w.writeheader()
        w.writerows(keep + rows)
    print(f"wrote {path} ({len(keep) + len(rows)} rows)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    a = sp.add_parser("decode")
    a.add_argument("--variant", required=True)
    a.add_argument("--latents", required=True)
    a.add_argument("--out", required=True)
    a.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    a.add_argument("--chunks", type=int, default=0, help="steady chunks (default: the whole stream)")
    a.add_argument("--tag", default="", help="suffix for a repeat run, e.g. run2")
    a.add_argument("--device", default="cuda")
    c = sp.add_parser("coverage")
    c.add_argument("--latents", required=True)
    c.add_argument("--out", required=True)
    c.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    c.add_argument("--device", default="cuda")
    e = sp.add_parser("compare")
    e.add_argument("--dir", required=True)
    e.add_argument("--chain", required=True, help="comma-separated variants, each compared with the one before")
    e.add_argument("--ref", default="", help="variant every stream is also compared with")
    e.add_argument("--out", required=True, help="CSV (rows of other VAEs are kept)")
    e.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    b = sp.add_parser("analyze")
    b.add_argument("--dir", required=True)
    b.add_argument("--out", required=True)
    b.add_argument("--vae", choices=["wan21", "wan22"], default="wan21")
    b.add_argument("--batch", type=int, default=12)
    b.add_argument("--source-video", help="the clip the latents were encoded from (reconstruction PSNR)")
    b.add_argument("--device", default="cuda")
    args = ap.parse_args()
    {"decode": decode, "analyze": analyze, "coverage": coverage, "compare": compare}[args.cmd](args)


if __name__ == "__main__":
    main()
