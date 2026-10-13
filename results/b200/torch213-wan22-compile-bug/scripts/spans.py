"""Compile contiguous spans of the Wan decoder's top-level stages as one graph and compare the
span output (and the cache slots it writes) with eager on the same inputs.

stages: 0 conv_in, 1 mid_block, 2..5 up_blocks[0..3], 6 head (norm_out, SiLU, conv_out)
  python spans.py --vae wan22 --spans 0-7,2-6,...   (half-open)
"""
import argparse, json, math
import run  # noqa: F401
import common, torch
from diffusers.models.autoencoders.autoencoder_kl_wan import CACHE_T
import torch._inductor.config as ic

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="wan22")
ap.add_argument("--spans", required=True)
ap.add_argument("--latent", type=int, default=7)
ap.add_argument("--nopadcfg", action="store_true")
ap.add_argument("--dump", help="write the generated code of each span to DUMP_<span>.py")
a = ap.parse_args()
torch._dynamo.config.cache_size_limit = 256
repo = common.WAN21_REPO if a.vae == "wan21" else common.WAN22_REPO
vae = common.load_vae(repo, torch.bfloat16, "cuda")
sd = common.StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=False)
z = torch.load(f"/vo/inputs/{a.vae}/latents.pt")["raw"].to("cuda", torch.bfloat16)
dec = vae.decoder
cc = lambda c: [t.clone() if torch.is_tensor(t) else t for t in c]


def cached_conv(conv, x, fc, fi):
    idx = fi[0]
    cache_x = x[:, :, -CACHE_T:].clone()
    if cache_x.shape[2] < 2 and fc[idx] is not None:
        cache_x = torch.cat([fc[idx][:, :, -1].unsqueeze(2), cache_x], dim=2)
    x = conv(x, fc[idx])
    fc[idx] = cache_x
    fi[0] += 1
    return x


def stage(k, x, fc, fi):
    if k == 0:
        return cached_conv(dec.conv_in, x, fc, fi)
    if k == 1:
        return dec.mid_block(x, feat_cache=fc, feat_idx=fi)
    if 2 <= k <= 5:
        return dec.up_blocks[k - 2](x, feat_cache=fc, feat_idx=fi, first_chunk=False)
    return cached_conv(dec.conv_out, dec.nonlinearity(dec.norm_out(x)), fc, fi)


def span_fn(i, j):
    def f(x, fc, idx0):
        fi = [idx0]
        for k in range(i, j):
            x = stage(k, x, fc, fi)
        return x
    return f


def rel(x, y):
    x, y = x.float(), y.float()
    return float("%.3g" % ((x - y).norm() / y.norm().clamp_min(1e-12)).item())


with torch.inference_mode():
    sd.first(z[:, :, :1])
    for i in range(1, a.latent):
        sd.chunk(z[:, :, i:i + 1])
    cache0 = cc(sd.feat)
    x = sd._prep(z[:, :, a.latent:a.latent + 1])
    ins, idxs, fc, fi = [], [], cc(cache0), [0]
    h = x
    for k in range(7):
        ins.append(h)
        idxs.append(fi[0])
        h = stage(k, h, fc, fi)
    ins.append(h)
    idxs.append(fi[0])
    print("STAGE_SHAPES " + json.dumps([[list(t.shape), list(t.stride())] for t in ins]), flush=True)
for sp in a.spans.split(","):
    i, j = map(int, sp.split("-"))
    torch._dynamo.reset()
    with ic.patch({"comprehensive_padding": False} if a.nopadcfg else {}):
        fn = torch.compile(span_fn(i, j), dynamic=False)
        with torch.inference_mode():
            fe = cc(cache0)
            ye = span_fn(i, j)(ins[i], fe, idxs[i])
            if a.dump:
                from torch._inductor.utils import run_and_get_code
                _, codes = run_and_get_code(fn, ins[i], cc(cache0), idxs[i])
                for n, code in enumerate(codes):
                    open(f"{a.dump}_{sp}_{n}.py", "w").write(code)
            else:
                fn(ins[i], cc(cache0), idxs[i])
            fcmp = cc(cache0)
            yc = fn(ins[i], fcmp, idxs[i])
            torch.cuda.synchronize()
    slots = list(range(idxs[i], idxs[j]))
    cache_rel = max([rel(fcmp[s], fe[s]) for s in slots if torch.is_tensor(fe[s])] or [0])
    print("SPAN " + json.dumps({"span": sp, "out_rel_l2": rel(yc, ye), "cache_max_rel_l2": cache_rel,
                                "nan": bool(torch.isnan(yc).any())}), flush=True)
