"""How dynamo splits the Wan decoder's steady-state call into graphs (count, ops per graph,
graph-break reasons), on this torch."""
import json
import sys

import run  # noqa: F401
import common
import torch

vae_name = sys.argv[1] if len(sys.argv) > 1 else "wan22"
repo = common.WAN21_REPO if vae_name == "wan21" else common.WAN22_REPO
vae = common.load_vae(repo, torch.bfloat16, "cuda")
sd = common.StreamingDecoder(vae, per_frame=True, channels_last=False, compile_decoder=False)
z = torch.load(f"/vo/inputs/{vae_name}/latents.pt")["raw"].to("cuda", torch.bfloat16)
with torch.inference_mode():
    sd.first(z[:, :, :1])
    for i in range(1, 7):
        sd.chunk(z[:, :, i:i + 1])
    x = sd._prep(z[:, :, 7:8])
    cache = [t.clone() if torch.is_tensor(t) else t for t in sd.feat]
    torch._dynamo.reset()
    ex = torch._dynamo.explain(vae.decoder.forward)(x, feat_cache=cache, feat_idx=[0])
sizes = [sum(1 for n in g.graph.nodes if n.op == "call_function" or n.op == "call_method") for g in ex.graphs]
print("GRAPHS " + json.dumps({"torch": torch.__version__, "graph_count": ex.graph_count,
                              "graph_break_count": ex.graph_break_count, "ops_per_graph": sizes,
                              "break_reasons": sorted({str(b.reason)[:160] for b in ex.break_reasons})[:8],
                              "break_sites": sorted({f"{f.filename.split('/')[-1]}:{f.lineno} {f.line}" for b in ex.break_reasons for f in b.user_stack[-1:]})}))
