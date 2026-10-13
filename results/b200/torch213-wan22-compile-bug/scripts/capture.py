"""Capture the dynamo FX graph of a decoder span (see spans.py) and write a standalone repro
that needs only torch: the graph source plus random inputs with the captured shapes/strides.

  python capture.py --vae wan22 --span 1-4 --out /w22/repro/wan22_span14
"""
import argparse
import json
import sys

sys.argv, _argv = [sys.argv[0], "--spans", "0-0"], sys.argv  # reuse spans.py setup only
import spans  # noqa: E402  (runs the eager warm-up for the default args)

sys.argv = _argv
import torch  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("--vae", default="wan22")
ap.add_argument("--span", default="1-4")
ap.add_argument("--out", required=True)
ap.add_argument("--save-real", action="store_true")
a = ap.parse_args()
i, j = map(int, a.span.split("-"))
captured = []


def backend(gm, example_inputs):
    captured.append((gm, example_inputs))
    return gm.forward


torch._dynamo.reset()
fn = torch.compile(spans.span_fn(i, j), backend=backend, dynamic=False)
with torch.inference_mode():
    fn(spans.ins[i], spans.cc(spans.cache0), spans.idxs[i])
assert len(captured) == 1, f"{len(captured)} graphs (graph breaks)"
gm, ex = captured[0]
src = gm.print_readable(print_output=False)
meta = [{"shape": list(t.shape), "stride": list(t.stride()), "dtype": str(t.dtype).replace("torch.", ""),
          "param": isinstance(t, torch.nn.Parameter)}
         if torch.is_tensor(t) else {"value": t} for t in ex]
if "--save-real" in _argv:
    torch.save([t.detach().clone() if torch.is_tensor(t) else t for t in ex], a.out + "_real.pt")
with open(a.out + "_graph.py", "w") as f:
    f.write("import torch\nfrom torch import device\n\n" + src + "\n")
with open(a.out + "_inputs.json", "w") as f:
    json.dump(meta, f)
print("CAPTURED", len(gm.graph.nodes), "nodes,", len(ex), "inputs")
