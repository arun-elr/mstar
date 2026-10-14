"""Output check for the fix runs: latents first, then frames.

Latents: the windows vae_decoder_ar received in each dump, new (runs-fix/<v>)
against the earlier session's (runs-latents/<v>). Frames: every dump against
the old after1/after0/before frames (runs/<v>) and against one decode of its
own whole latent stream with stock diffusers AutoencoderKLWan.decode in fp32
(TF32 off) and bf16, as ref_decode.py does. One reference is decoded per
distinct latent stream, so a frame set is only scored against the reference
of the latents it was decoded from.

  python cmp_fix.py ~/e2e-cosmos3 out_dir
"""
import glob
import hashlib
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from diffusers import AutoencoderKLWan
from huggingface_hub import snapshot_download

E, OUT = Path(sys.argv[1]), Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)
REV = "344d602b128d1bbdacb43b08d0a3626f46343e29"
# name -> (frames dir root, latents dir root)
SETS = {
    "before": ("runs/before", "runs-latents/before"),
    "after0": ("runs/after0", "runs-latents/after0"),
    "after1": ("runs/after1", "runs-latents/after1"),
    "after1fix": ("runs-fix/after1fix", "runs-fix/after1fix"),
    "after0fix": ("runs-fix/after0fix", "runs-fix/after0fix"),
}


def windows(root, mode):
    return [torch.load(p) for p in sorted(glob.glob(str(E / root / "dump" / f"{mode}_latents" / "*.pt")))]


def frames(root, mode):
    fs = sorted(glob.glob(str(E / root / "dump" / mode / "*.npy")))
    return (np.concatenate([np.load(f) for f in fs], axis=1), [np.load(f).shape[1] for f in fs]) if fs else (None, [])


def stream(ws):
    parts = [ws[0]["latents"]] + [w["latents"][:, :, w["overlap_latent_units"]:] for w in ws[1:]]
    return torch.cat(parts, dim=2)


def md5(t):
    return hashlib.md5(t.float().numpy().tobytes()).hexdigest()


def psnr(a, b):
    mse = np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2)
    return float("inf") if mse == 0 else 10 * math.log10(255.0**2 / mse)


def compare(x, ref, bounds):
    diff = np.abs(x.astype(np.int16) - ref.astype(np.int16))
    per = [psnr(x[:, t], ref[:, t]) for t in range(x.shape[1])]
    fin = [p for p in per if math.isfinite(p)]
    edges = list(zip([0, *bounds], [*bounds, x.shape[1]]))
    return {
        "identical": bool((diff == 0).all()), "max_abs_diff": int(diff.max()),
        "differing_fraction": float((diff > 0).mean()), "fraction_diff_gt1": float((diff > 1).mean()),
        "psnr_whole_db": psnr(x, ref), "psnr_frame_min_db": min(fin) if fin else float("inf"),
        "worst_frame": int(np.argmin([p if math.isfinite(p) else 1e9 for p in per])),
        "psnr_frame_median_db": float(np.median(fin)) if fin else float("inf"),
        "psnr_window_min_db": [round(min(per[a:b]), 2) if any(map(math.isfinite, per[a:b])) else "inf"
                               for a, b in edges],
        "per_window_max_abs_diff": [int(diff[:, a:b].max()) for a, b in edges],
        "psnr_first_frame_of_window_db": [round(per[b], 2) if math.isfinite(per[b]) else "inf" for b in bounds],
        "per_frame_psnr_db": [round(p, 2) if math.isfinite(p) else "inf" for p in per],
    }


def brief(c):
    return {k: v for k, v in c.items() if k != "per_frame_psnr_db"}


torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
vae_dir = Path(snapshot_download("nvidia/Cosmos3-Edge", revision=REV, allow_patterns=["vae/*"])) / "vae"
res = {"vae": f"nvidia/Cosmos3-Edge@{REV}/vae", "sets": {k: list(v) for k, v in SETS.items()}, "modes": {}}
for mode in ("kv", "chained"):
    ws = {k: windows(lr, mode) for k, (_, lr) in SETS.items()}
    ws = {k: w for k, w in ws.items() if w}
    lat = {k: stream(w) for k, w in ws.items()}
    m = {"latents_md5": {k: md5(t) for k, t in lat.items()},
         "latent_window_md5": {k: [md5(w["latents"])[:8] for w in ws[k]] for k in ws},
         "stream_latents_shape": {k: list(t.shape) for k, t in lat.items()}}
    m["latents_max_abs_diff_vs_after1"] = {
        k: float((t.float() - lat["after1"].float()).abs().max()) if t.shape == lat["after1"].shape else "shape"
        for k, t in lat.items()}
    print(f"== {mode}: latents md5 {m['latents_md5']}")
    print(f"   latents max|diff| vs old after1 {m['latents_max_abs_diff_vs_after1']}")
    fr = {k: frames(fr_root, mode) for k, (fr_root, _) in SETS.items()}
    fr = {k: v for k, v in fr.items() if v[0] is not None}
    bounds = list(np.cumsum(fr["after1"][1])[:-1])
    m["chunk_frames"] = {k: v[1] for k, v in fr.items()}
    m["frames_md5"] = {k: hashlib.md5(v[0].tobytes()).hexdigest() for k, v in fr.items()}
    # frames vs frames (only meaningful between sets decoded from identical latents)
    pairs = [("after1fix", "after1"), ("after0fix", "after0"), ("after0fix", "after1fix"), ("after1fix", "after0"),
             ("after1fix", "before")]
    m["frames_vs_frames"] = {}
    for a, b in pairs:
        if a in fr and b in fr and fr[a][0].shape == fr[b][0].shape:
            c = compare(fr[a][0], fr[b][0], bounds)
            c["same_latents"] = m["latents_md5"].get(a) == m["latents_md5"].get(b)
            m["frames_vs_frames"][f"{a}_vs_{b}"] = c
            print(f"   {a} vs {b}: same latents {c['same_latents']} psnr {c['psnr_whole_db']:.2f} "
                  f"min {c['psnr_frame_min_db']:.2f} max|d| {c['max_abs_diff']} identical {c['identical']}")
    # one fp32 / bf16 reference per distinct latent stream
    refs = {}
    n_frames = ws["after1"][0]["num_frames"]
    for h in sorted(set(m["latents_md5"].values())):
        k0 = next(k for k, v in m["latents_md5"].items() if v == h)
        refs[h] = {}
        for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
            vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=dtype).cuda().eval()
            mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
            std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
            z = (lat[k0].float() * std + mean).to("cuda", dtype)
            with torch.inference_mode():
                out = vae.decode(z).sample
            refs[h][name] = (out.float() / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)[0, :, :n_frames].cpu().numpy()
            del vae, out
            torch.cuda.empty_cache()
        print(f"   decoded references for latents {h[:8]} (from {k0})")
    m["reference_latents_md5"] = sorted(refs)
    m["bf16_ref_vs_fp32_ref"] = {h[:8]: brief(compare(r["bf16"], r["fp32"], bounds)) for h, r in refs.items()}
    for ref in ("fp32", "bf16"):
        m[f"vs_{ref}_whole_stream"] = {}
        for k, (x, _) in fr.items():
            h = m["latents_md5"].get(k)
            if h is None:
                continue
            c = compare(x, refs[h][ref], bounds)
            c["reference_latents_md5"] = h
            m[f"vs_{ref}_whole_stream"][k] = c
            print(f"   vs {ref} ({h[:8]}) {k:<10} psnr {c['psnr_whole_db']:.2f} worst {c['psnr_frame_min_db']:.2f} "
                  f"(#{c['worst_frame']}) median {c['psnr_frame_median_db']:.2f} max|d| {c['max_abs_diff']} "
                  f">1: {100 * c['fraction_diff_gt1']:.1f}% window min {c['psnr_window_min_db']}")
    res["modes"][mode] = m
(OUT / "cmp_fix.json").write_text(json.dumps(res, indent=1, default=str))
