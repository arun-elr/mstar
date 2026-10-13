"""One decode config per process; per-chunk PSNR vs the fp32 reference (steady chunks 10..19)
and vs stock eager bf16 per-frame decode (every chunk, saved once by --save-stock).

  python w22_bisect.py --vae wan22 --flags compile,cl,onecall,nopad,bench,up16 [--save-stock]
"""
import argparse, json, math, sys, time
from pathlib import Path
import run  # noqa: F401  (hf_snapshot override)
import common, torch
from common import StreamingDecoder, load_vae, to_uint8

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="wan22")
ap.add_argument("--flags", default="")
ap.add_argument("--chunks", type=int, default=20)
ap.add_argument("--save-stock", action="store_true")
ap.add_argument("--mode", default=None)
ap.add_argument("--tag", default="")
a = ap.parse_args()
flags = set(f for f in a.flags.split(",") if f)
unknown = flags - {"compile", "cl", "onecall", "nopad", "bench", "up16", "fp32"}
assert not unknown, unknown
repo = common.WAN21_REPO if a.vae == "wan21" else common.WAN22_REPO
torch._dynamo.config.cache_size_limit = 64
if "up16" in flags: common.bf16_nearest_upsample()
if "nopad" in flags: common.patch_causal_conv_native_pad()
if "bench" in flags: torch.backends.cudnn.benchmark = True
dtype = torch.float32 if "fp32" in flags else torch.bfloat16
vae = load_vae(repo, dtype, "cuda")
sd = StreamingDecoder(vae, per_frame="onecall" not in flags, channels_last="cl" in flags,
                      compile_decoder="compile" in flags, compile_mode=a.mode)
raw = torch.load(f"/vo/inputs/{a.vae}/latents.pt")["raw"]
ref = torch.load(f"/vo/inputs/{a.vae}/ref_fp32_outputs.pt")["float"]
z = raw.to("cuda", dtype)
stock_p = Path(f"/w22/runs/stock_bf16_{a.vae}.pt")
def psnr(x, y):
    mse = torch.mean((x.double() - y.double()) ** 2).item()
    return float("inf") if mse == 0 else round(10 * math.log10(4.0 / mse) - 20 * math.log10(2), 2)
outs, ms = [], []
with torch.inference_mode():
    outs.append(sd.first(z[:, :, :1]).float().cpu())
    for c in range(a.chunks):
        torch.cuda.synchronize(); t = time.perf_counter()
        o = sd.chunk(z[:, :, 1 + 3 * c: 4 + 3 * c]); torch.cuda.synchronize()
        ms.append((time.perf_counter() - t) * 1e3)
        outs.append(o.float().cpu())
if a.save_stock:
    torch.save(outs, stock_p)
stock = torch.load(stock_p) if stock_p.exists() else None
res = {"vae": a.vae, "flags": sorted(flags), "mode": a.mode, "tag": a.tag,
       "torch": torch.__version__,
       "psnr_vs_fp32_steady": [psnr(outs[1 + 10 + k], ref[k]) for k in range(min(10, a.chunks - 10))],
       "psnr_vs_stock_bf16": [psnr(o, s) for o, s in zip(outs, stock)] if stock else None,
       "ms_last5": [round(x, 1) for x in ms[-5:]]}
print("RESULT " + json.dumps(res))
