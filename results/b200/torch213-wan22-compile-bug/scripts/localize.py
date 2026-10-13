"""Which Wan decoder module goes wrong under torch.compile?  Warm the cache eagerly
(clip-start latent + 6 steady latents, one per call), then for one steady latent compile
only the instances of one class and, inside each instance, compare compiled vs eager on
the same input and cloned cache.

  python localize.py --vae wan22 --classes DupUp3D,WanResidualUpBlock,...
"""
import argparse, json, math
import run  # noqa: F401
import common, torch
from diffusers.models.autoencoders import autoencoder_kl_wan as akw

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="wan22")
ap.add_argument("--classes", default="DupUp3D,WanResidualUpBlock,WanResidualBlock,WanResample,WanAttentionBlock,WanMidBlock,WanRMS_norm,WanCausalConv3d,WanDecoder3d")
ap.add_argument("--cl", action="store_true")
ap.add_argument("--latent", type=int, default=7)
ap.add_argument("--ncall", type=int, default=1, help="latents per decoder call")
a = ap.parse_args()
torch._dynamo.config.cache_size_limit = 256
repo = common.WAN21_REPO if a.vae == "wan21" else common.WAN22_REPO
vae = common.load_vae(repo, torch.bfloat16, "cuda")
sd = common.StreamingDecoder(vae, per_frame=True, channels_last=a.cl, compile_decoder=False)
z = torch.load(f"/vo/inputs/{a.vae}/latents.pt")["raw"].to("cuda", torch.bfloat16)
dec = vae.decoder

def clone_cache(c):
    return [t.clone() if torch.is_tensor(t) else t for t in c]

def psnr(x, y):
    mse = torch.mean((x.double() - y.double()) ** 2).item()
    return float("inf") if mse == 0 else round(10 * math.log10(4.0 / mse) - 20 * math.log10(2), 2)

def stats(x, y):
    d = (x.float() - y.float()).abs()
    return {"psnr": psnr(x.float().clamp(-1, 1), y.float().clamp(-1, 1)) if x.shape[1] <= 12 else None,
            "max_abs": round(d.max().item(), 4), "ref_max": round(y.float().abs().max().item(), 3),
            "rel_l2": float("%.3g" % ((x.float() - y.float()).norm() / y.float().norm().clamp_min(1e-12)).item()),
            "nan": bool(torch.isnan(x).any())}

with torch.inference_mode():
    sd.first(z[:, :, :1])
    for i in range(1, a.latent):
        sd.chunk(z[:, :, i:i + 1])
    cache0 = clone_cache(sd.feat)
    x = sd._prep(z[:, :, a.latent:a.latent + a.ncall])
    ref = dec(x, feat_cache=clone_cache(cache0), feat_idx=[0])

names = {}
for n, m in dec.named_modules():
    names[id(m)] = n or "decoder"
out = {}
for cls_name in a.classes.split(","):
    cls = getattr(akw, cls_name)
    insts = [m for m in dec.modules() if type(m) is cls or (cls_name == "WanCausalConv3d" and isinstance(m, cls))]
    torch._dynamo.reset()
    records = []
    saved = {}
    for m in insts:
        orig = m.forward
        saved[id(m)] = orig
        comp = torch.compile(orig, fullgraph=False, dynamic=False)
        def wrapped(*args, _orig=orig, _comp=comp, _m=m, **kw):
            kw_e = dict(kw)
            if "feat_cache" in kw and kw["feat_cache"] is not None:
                kw_e["feat_cache"] = clone_cache(kw["feat_cache"])
                kw_e["feat_idx"] = [kw["feat_idx"][0]]
            args_e = tuple(t.clone() if torch.is_tensor(t) else t for t in args)
            e = _orig(*args_e, **kw_e)
            _comp(*tuple(t.clone() if torch.is_tensor(t) else t for t in args), **{**kw, **({"feat_cache": clone_cache(kw["feat_cache"]), "feat_idx": [kw["feat_idx"][0]]} if kw.get("feat_cache") is not None else {})})
            c = _comp(*args, **kw)
            records.append({"module": names[id(_m)], **stats(c, e)})
            return c
        m.forward = wrapped
    with torch.inference_mode():
        y = dec(x, feat_cache=clone_cache(cache0), feat_idx=[0])
    for m in insts:
        m.forward = saved[id(m)]
    bad = [r for r in records if r["nan"] or r["rel_l2"] > 0.02]
    out[cls_name] = {"n_instances": len(insts), "decoder_out_vs_eager": stats(y, ref),
                     "worst": sorted(records, key=lambda r: -r["rel_l2"] if not r["nan"] else -1e9)[:4],
                     "n_bad": len(bad)}
    print("CLASS " + json.dumps({cls_name: out[cls_name]}), flush=True)
