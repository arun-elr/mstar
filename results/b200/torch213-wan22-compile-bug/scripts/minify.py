"""Shrink a captured graph with torch's FX minifier: failing = compiled output differs from eager
by rel-L2 > 0.2 on some output (padding on), while the same graph is right with padding off."""
import sys
import time

import torch
import torch._inductor.config as ic
from torch._functorch.fx_minifier import minifier

sys.argv = [sys.argv[0], sys.argv[1], "--no-params"] + sys.argv[2:]
import run_graph as R  # noqa: E402  (loads the graph, builds inputs, runs the check once)

t0 = time.time()
n_calls = [0]


def max_rel(gm, inps, cfg):
    torch._dynamo.reset()
    with torch.inference_mode():
        ref = gm(*inps)
        with ic.patch(cfg):
            out = torch.compile(gm, dynamic=False)(*inps)
        torch.cuda.synchronize()
    ref = ref if isinstance(ref, (tuple, list)) else [ref]
    out = out if isinstance(out, (tuple, list)) else [out]
    return max([((o.float() - r.float()).norm() / r.float().norm().clamp_min(1e-12)).item()
                for o, r in zip(out, ref) if torch.is_tensor(r) and r.numel()] or [0.0])


def fails(gm, inps):
    n_calls[0] += 1
    try:
        bad = max_rel(gm, inps, {}) > 0.2
    except Exception as e:  # noqa: BLE001
        print(f"[{n_calls[0]}] exception {type(e).__name__}", flush=True)
        return False
    print(f"[{n_calls[0]} {time.time() - t0:.0f}s] nodes={len(gm.graph.nodes)} fails={bad}", flush=True)
    return bad


gm = torch.fx.symbolic_trace(R.gm) if not isinstance(R.gm, torch.fx.GraphModule) else R.gm
res = minifier(gm, list(R.inputs), fails, save_dir="/w22/repro/minifier", max_granularity=None)
print("MINIFIED", res, flush=True)
