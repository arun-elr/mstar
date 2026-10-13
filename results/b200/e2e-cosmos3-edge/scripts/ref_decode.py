"""Ground truth for the served frames: one decode of each rollout's whole latent
stream (what the streaming decoder approximates), from the latents the
vae_decoder_ar node received (runs-latents/<v>/dump/<mode>_latents), with stock
diffusers AutoencoderKLWan.decode in fp32 (TF32 off) and in bf16. Scores the
frames each variant served (runs/<v>/dump/<mode>) against it.

  python ref_decode.py ~/e2e-cosmos3 out_dir
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
VARIANTS = ("before", "after0", "after1")
REV = "344d602b128d1bbdacb43b08d0a3626f46343e29"


def windows(v, mode):
    return [torch.load(p) for p in sorted(glob.glob(str(E / "runs-latents" / v / "dump" / f"{mode}_latents" / "*.pt")))]


def frames(root, v, mode):
    return np.concatenate([np.load(f) for f in sorted(glob.glob(str(E / root / v / "dump" / mode / "*.npy")))], axis=1)


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
        "psnr_window_min_db": [round(min(per[a:b]), 2) for a, b in edges],
        "per_window_max_abs_diff": [int(diff[:, a:b].max()) for a, b in edges],
        "psnr_first_frame_of_window_db": [round(per[b], 2) for b in bounds],
        "per_frame_psnr_db": [round(p, 2) if math.isfinite(p) else "inf" for p in per],
    }


torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = False
vae_dir = Path(snapshot_download("nvidia/Cosmos3-Edge", revision=REV, allow_patterns=["vae/*"])) / "vae"
res = {"vae": f"nvidia/Cosmos3-Edge@{REV}/vae", "modes": {}}
for mode in ("kv", "chained"):
    ws = {v: windows(v, mode) for v in VARIANTS}
    base = ws["before"]
    same = {v: len(ws[v]) == len(base) and all(torch.equal(a["latents"], b["latents"]) for a, b in zip(ws[v], base))
            for v in VARIANTS}
    parts = [base[0]["latents"]] + [w["latents"][:, :, w["overlap_latent_units"]:] for w in base[1:]]
    lat = torch.cat(parts, dim=2)
    n_frames = base[0]["num_frames"]
    m = {"windows": len(base), "window_latents": [tuple(w["latents"].shape) for w in base],
         "overlap_latent_units": base[0]["overlap_latent_units"], "stream_latents": tuple(lat.shape),
         "latents_identical_to_before": same,
         "latents_md5": hashlib.md5(lat.float().numpy().tobytes()).hexdigest()}
    refs = {}
    for name, dtype in (("fp32", torch.float32), ("bf16", torch.bfloat16)):
        vae = AutoencoderKLWan.from_pretrained(str(vae_dir), torch_dtype=dtype).cuda().eval()
        mean = torch.tensor(vae.config.latents_mean).view(1, -1, 1, 1, 1)
        std = torch.tensor(vae.config.latents_std).view(1, -1, 1, 1, 1)
        z = (lat.float() * std + mean).to("cuda", dtype)
        with torch.inference_mode():
            out = vae.decode(z).sample
        refs[name] = (out.float() / 2 + 0.5).clamp(0, 1).mul(255).to(torch.uint8)[0, :, :n_frames].cpu().numpy()
        del vae, out
        torch.cuda.empty_cache()
    served = {v: frames("runs", v, mode) for v in VARIANTS}
    rerun = {v: frames("runs-latents", v, mode) for v in VARIANTS}
    m["served_md5"] = {v: hashlib.md5(x.tobytes()).hexdigest() for v, x in served.items()}
    m["served_equals_latent_pass"] = {v: bool(np.array_equal(served[v], rerun[v])) for v in VARIANTS}
    chunk = [np.load(f).shape[1] for f in sorted(glob.glob(str(E / "runs" / "before" / "dump" / mode / "*.npy")))]
    bounds = list(np.cumsum(chunk)[:-1])
    m["chunk_frames"] = chunk
    m["bf16_ref_vs_fp32_ref"] = {k: v for k, v in compare(refs["bf16"], refs["fp32"], bounds).items()
                                 if k != "per_frame_psnr_db"}
    m["vs_fp32_whole_stream"] = {v: compare(x, refs["fp32"], bounds) for v, x in served.items()}
    m["vs_bf16_whole_stream"] = {v: compare(x, refs["bf16"], bounds) for v, x in served.items()}
    res["modes"][mode] = m
    print(f"== {mode}: stream {m['stream_latents']} windows {m['windows']} overlap {m['overlap_latent_units']} "
          f"latents identical {same} served==rerun {m['served_equals_latent_pass']}")
    print(f"   bf16 ref vs fp32 ref: psnr {m['bf16_ref_vs_fp32_ref']['psnr_whole_db']:.2f} "
          f"min {m['bf16_ref_vs_fp32_ref']['psnr_frame_min_db']:.2f} max {m['bf16_ref_vs_fp32_ref']['max_abs_diff']}")
    for ref in ("vs_fp32_whole_stream", "vs_bf16_whole_stream"):
        for v, c in m[ref].items():
            print(f"   {ref} {v:<7} psnr {c['psnr_whole_db']:.2f} dB  worst frame {c['psnr_frame_min_db']:.2f} dB "
                  f"(#{c['worst_frame']})  median {c['psnr_frame_median_db']:.2f}  max|d| {c['max_abs_diff']}  "
                  f"window max|d| {c['per_window_max_abs_diff']}  window min psnr {c['psnr_window_min_db']}")
(OUT / "reference.json").write_text(json.dumps(res, indent=1, default=str))
