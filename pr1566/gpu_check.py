#!/usr/bin/env python3
"""Real CUDA execution of #1566 RAS resource, with fixed synthetic logits.

This is a sampler-path check, not full model or official-reference parity.
"""
import json
import time
import torch

from mstar.engine.resources.sampler.config import RASSamplingReqConfig
from mstar.engine.resources.sampler.resource import SamplerResource
from mstar.model.cosyvoice3.components.sampling import ras_sample_plain

DEVICE = torch.device("cuda")
WIDTH = 4
OUT = {"device": torch.cuda.get_device_name(0), "torch": torch.__version__, "cuda": torch.version.cuda}

def resource():
    return SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=DEVICE,
        max_repetition_window=WIDTH, enable_blocked_tokens=True, enable_ras=True,
    )

def cfg(seed, window, tau, temp, top_p):
    c = RASSamplingReqConfig(
        temperature=temp, top_k=7, top_p=top_p,
        repetition_window=window, ras_tau=tau,
    )
    c.apply_conductor_config(seed=seed, resource_key="lm_sampler")
    return c

def eager_reference():
    runs=[]
    for noise_seed in (345, 346, 347):
        configs={"a":cfg(123456789,4,.25,1.,.8),"b":cfg(987654321,2,.5,.7,0.),"c":cfg(42,0,0.,0.,.9)}
        res=resource()
        for rid,c in configs.items(): res.ingest_request(rid,c)
        history={rid:torch.full((WIDTH,),-1,dtype=torch.long,device=DEVICE) for rid in configs}
        counts={rid:0 for rid in configs}
        gen=torch.Generator(device=DEVICE).manual_seed(noise_seed)
        rows=[]
        for batch in (["a"],["a","b"],["b","a","c"],["c","b","a"],["a","c"]):
            logits=torch.randn(len(batch),11,device=DEVICE,generator=gen)*2
            blocked=torch.zeros_like(logits,dtype=torch.bool)
            if "c" in batch:
                blocked[batch.index("c"),:]=True
                blocked[batch.index("c"),9]=False
            if "b" in batch: blocked[batch.index("b"),10]=True
            want=ras_sample_plain(
                logits,seed=torch.tensor([configs[r].seed for r in batch],device=DEVICE),
                step=torch.tensor([counts[r] for r in batch],device=DEVICE),
                history=torch.stack([history[r] for r in batch]),
                temperature=torch.tensor([configs[r].temperature for r in batch],device=DEVICE),
                top_p=torch.tensor([configs[r].top_p for r in batch],device=DEVICE),
                top_k=torch.tensor([configs[r].top_k for r in batch],device=DEVICE),
                window=torch.tensor([configs[r].repetition_window for r in batch],device=DEVICE),
                repeat_threshold=torch.tensor([configs[r].repetition_window*configs[r].ras_tau for r in batch],device=DEVICE),
                blocked=blocked,
            )
            got=res.sample(batch,logits,blocked)
            torch.cuda.synchronize()
            if not torch.equal(want,got):
                raise AssertionError(f"eager mismatch seed={noise_seed} batch={batch}: {want.tolist()} != {got.tolist()}")
            rows.append({"batch":batch,"tokens":got.tolist()})
            for rid,tok in zip(batch,want,strict=True):
                history[rid]=torch.cat((history[rid][1:],tok.reshape(1)))
                counts[rid]+=1
        runs.append({"noise_seed":noise_seed,"rows":rows})
    OUT["eager_reference"]={"matched":True,"runs":runs}

def captured_replay():
    eager, captured = resource(), resource()
    captured.build_accelerator_graph_buffers(slots=[],max_bs=3,max_seq_len=0)
    for res in (eager,captured):
        res.ingest_request("a",cfg(100,4,.25,1.,.8))
        res.ingest_request("b",cfg(200,2,.5,0.,0.))
    bufs=captured._cg_buffers
    assert bufs is not None
    gen=torch.Generator(device=DEVICE).manual_seed(125)
    static_logits=torch.randn(3,15,device=DEVICE,generator=gen)
    static_blocked=torch.zeros(3,15,device=DEVICE,dtype=torch.bool)
    bufs.gather_static(["a","b"],3,0)
    bufs.gather_dynamic(["a","b"],3,0)
    assert captured._history is not None
    captured._history.gather_step(["a","b"],3)
    captured._cg_sampler=bufs.sampler_for(3,0)
    graph=torch.cuda.CUDAGraph()
    with torch.no_grad(),torch.cuda.graph(graph):
        static_out=captured.sample(["a","b"],static_logits,static_blocked)
    torch.cuda.synchronize()
    rows=[]
    for batch in (["a"],["b","a"],["a","b"],["b"]):
        logits=torch.randn(len(batch),15,device=DEVICE,generator=gen)
        blocked=torch.zeros_like(logits,dtype=torch.bool)
        if "b" in batch: blocked[batch.index("b"),14]=True
        want=eager.sample(batch,logits,blocked)
        bufs.gather_static(batch,3,0)
        bufs.gather_dynamic(batch,3,0)
        captured._history.gather_step(batch,3)
        captured._cg_sampler=bufs.sampler_for(3,0)
        static_logits.zero_(); static_logits[:len(batch)].copy_(logits)
        static_blocked.zero_(); static_blocked[:len(batch)].copy_(blocked)
        graph.replay()
        torch.cuda.synchronize()
        got=static_out[:len(batch)]
        if not torch.equal(got,want):
            raise AssertionError(f"graph mismatch batch={batch}: {want.tolist()} != {got.tolist()}")
        captured._history.scatter_step()
        captured._cg_sampler=None
        rows.append({"batch":batch,"tokens":got.tolist()})
    OUT["captured_replay"]={"matched":True,"rows":rows,"actual_cuda_graph":True}

def threshold_regression():
    res=SamplerResource(
        vocab_size=None, enable_repetition_penalty=False, device=DEVICE,
        max_repetition_window=25, enable_blocked_tokens=True, enable_ras=True,
    )
    res.ingest_request("threshold", cfg(7,25,.6,1.,0.))
    logits=torch.tensor([[5.,0.]],device=DEVICE)
    tokens=[]
    for step in range(16):
        got=res.sample(["threshold"],logits)
        tokens.append(int(got.item()))
    assert tokens == [0]*15+[1], tokens
    OUT["threshold_regression"]={"tokens":tokens,"redraw_at_step":15}

if __name__=="__main__":
    t=time.perf_counter()
    eager_reference()
    captured_replay()
    threshold_regression()
    OUT["elapsed_s"]=time.perf_counter()-t
    print(json.dumps(OUT,indent=2))
