"""Run a captured graph (capture.py) eager vs torch.compile.

Inputs that were nn.Parameters in the real decoder are held as parameters of a wrapper
module (as dynamo sees them when it inlines the VAE's modules); the rest are passed in.

  python run_graph.py PREFIX [--real] [inductor config key=val ...]
"""
import importlib.util
import json
import sys

import torch
import torch._inductor.config as ic

prefix = sys.argv[1]
real = "--real" in sys.argv
NO_PARAMS = "--no-params" in sys.argv
cfg = {k: eval(v) for k, v in (kv.split("=") for kv in sys.argv[2:] if not kv.startswith("--"))}
spec = importlib.util.spec_from_file_location("graph", prefix + "_graph.py")
mod = importlib.util.module_from_spec(spec)
sys.modules["graph"] = mod
spec.loader.exec_module(mod)
gm = mod.GraphModule()
meta = json.load(open(prefix + "_inputs.json"))
real_vals = torch.load(prefix + "_real.pt") if real else None
torch.manual_seed(0)
vals = []
for k, m in enumerate(meta):
    if "value" in m:
        vals.append(m["value"])
        continue
    if real:
        vals.append(real_vals[k].as_strided(m["shape"], m["stride"]).cuda())
        continue
    dt = getattr(torch, m["dtype"])
    shape, stride = m["shape"], m["stride"]
    n = 1 + sum((s - 1) * st for s, st in zip(shape, stride))
    base = torch.randn(n, device="cuda")
    if m.get("param"):
        fan_in = 1
        for s in shape[1:]:
            fan_in *= s
        if len(shape) == 1:
            base.zero_()           # conv / linear bias
        elif fan_in == 1:
            base.fill_(1.0)        # WanRMS_norm gamma, [C, 1, 1(, 1)]
        else:
            base /= fan_in ** 0.5  # conv weight
    vals.append(base.to(dt).as_strided(shape, stride))


class Wrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.slots = []
        for k, (m, v) in enumerate(zip(meta, vals)):
            if m.get("param") and not NO_PARAMS:
                self.register_parameter(f"p{k}", torch.nn.Parameter(v, requires_grad=False))
                self.slots.append(("p", f"p{k}"))
            else:
                self.slots.append(("a", None))

    def forward(self, *inputs):
        it = iter(inputs)
        return gm(*[getattr(self, n) if kind == "p" else next(it) for kind, n in self.slots])


w = Wrapper()
inputs = [v for m, v in zip(meta, vals) if NO_PARAMS or not m.get("param")]
with torch.inference_mode():
    ref = w(*inputs)
    with ic.patch(cfg):
        out = torch.compile(w, dynamic=False)(*inputs)
    torch.cuda.synchronize()
ref = ref if isinstance(ref, (tuple, list)) else [ref]
out = out if isinstance(out, (tuple, list)) else [out]
rels = []
for k, (o, r) in enumerate(zip(out, ref)):
    if torch.is_tensor(r):
        rels.append([float("%.3g" % ((o.float() - r.float()).norm() / r.float().norm().clamp_min(1e-12)).item()), k, list(r.shape)])
print("GRAPH " + json.dumps({"torch": torch.__version__, "cfg": cfg, "real": real, "n_params": sum(bool(m.get("param")) for m in meta),
                             "out0_rel_l2": rels[0][0], "worst_rel_l2": sorted(rels, reverse=True)[:3]}))
