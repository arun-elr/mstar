"""Whole WanDecoder3d compiled with different backends / inductor knobs, one warm steady call,
vs eager on the same input and cloned cache. Each spec runs in this process after dynamo reset.

  python backends.py --vae wan22 --specs eager,aot_eager,aot_eager_decomp_partition,inductor
  spec "inductor:key=val;key=val" sets torch._inductor.config keys (dotted ok) for that run
"""
import argparse, json, math
import run  # noqa: F401
import common, torch

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="wan22")
ap.add_argument("--specs", default="eager,aot_eager,aot_eager_decomp_partition,inductor")
ap.add_argument("--latent", type=int, default=7)
ap.add_argument("--ncall", type=int, default=1)
ap.add_argument("--cl", action="store_true")
ap.add_argument("--fp32", action="store_true")
a = ap.parse_args()
torch._dynamo.config.cache_size_limit = 256
repo = common.WAN21_REPO if a.vae == "wan21" else common.WAN22_REPO
dt = torch.float32 if a.fp32 else torch.bfloat16
vae = common.load_vae(repo, dt, "cuda")
sd = common.StreamingDecoder(vae, per_frame=True, channels_last=a.cl, compile_decoder=False)
z = torch.load(f"/vo/inputs/{a.vae}/latents.pt")["raw"].to("cuda", dt)
dec = vae.decoder
cc = lambda c: [t.clone() if torch.is_tensor(t) else t for t in c]
def stats(x, y):
    x, y = x.float(), y.float()
    mse = torch.mean((x.clamp(-1, 1).double() - y.clamp(-1, 1).double()) ** 2).item()
    return {"psnr": float("inf") if mse == 0 else round(10 * math.log10(1 / mse), 2),
            "max_abs": round((x - y).abs().max().item(), 4), "nan": bool(torch.isnan(x).any())}
with torch.inference_mode():
    sd.first(z[:, :, :1])
    for i in range(1, a.latent):
        sd.chunk(z[:, :, i:i + 1])
    cache0 = cc(sd.feat)
    x = sd._prep(z[:, :, a.latent:a.latent + a.ncall])
    c_ref = cc(cache0)
    ref = dec(x, feat_cache=c_ref, feat_idx=[0])
import torch._inductor.config as ic
for spec in a.specs.split(","):
    backend, _, kv = spec.partition(":")
    patches = {}
    for item in filter(None, kv.split(";")):
        k, v = item.split("=")
        patches[k] = eval(v)
    torch._dynamo.reset()
    try:
        with ic.patch(patches) if patches else torch.inference_mode():
            fn = torch.compile(dec.forward, backend=backend, fullgraph=False, dynamic=False)
            with torch.inference_mode():
                fn(x, feat_cache=cc(cache0), feat_idx=[0])
                c = cc(cache0)
                y = fn(x, feat_cache=c, feat_idx=[0])
                cache_ok = min((stats(p, q)["psnr"] for p, q in zip(c, c_ref) if torch.is_tensor(p)), default=None)
        r = {"spec": spec, **stats(y, ref), "cache_min_psnr": cache_ok}
    except Exception as e:
        r = {"spec": spec, "error": f"{type(e).__name__}: {str(e)[:300]}"}
    print("SPEC " + json.dumps(r), flush=True)
