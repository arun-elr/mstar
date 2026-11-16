#!/usr/bin/env python3
"""Audit and plot matched served #1566 runs from raw client JSON."""
import json
import statistics
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT=Path(__file__).resolve().parent
ARMS=[a for a in ("main1","pr1","pr2","main2") if (ROOT/a/"result.json").exists()]
if not {"main1","pr1"}.issubset(ARMS):
    raise RuntimeError("need at least one matched main/PR pair")
data={a:json.loads((ROOT/a/"result.json").read_text()) for a in ARMS}
cells=("c1","c4")
for a,d in data.items():
    assert d["tests"]["c1"]["requests"]==160
    assert d["tests"]["c4"]["requests"]==256
    assert len(d["tests"]["parity"])==3
    assert d["ref_wav_sha256"]==next(iter(data.values()))["ref_wav_sha256"]
def keys(d,cell):
    rows=d["tests"]["parity"] if cell=="parity" else d["tests"][cell]["rows"]
    return [(r["pcm_sha256"],r["pcm_samples"]) for r in rows]
reference=data["main1"]
parity={}
for cell in ("parity",*cells):
    baseline=keys(reference,cell)
    parity[cell]={a:sum(x!=y for x,y in zip(baseline,keys(d,cell),strict=True)) for a,d in data.items()}
rows=[]
for a,d in data.items():
    for cell in cells:
        x=d["tests"][cell]
        rows.append({
            "arm":a,"source":"main" if a.startswith("main") else "pr",
            "concurrency":int(cell[1:]),"requests":x["requests"],
            "warmup_requests":x["warmup_requests"],
            "wall_s":x["wall_s"],"client_cpu_s":x["client_cpu_s"],
            "client_cpu_pct_wall":100*x["client_cpu_s"]/x["wall_s"],
            "requests_per_s":x["requests_per_s"],
            "audio_s_total":x["audio_s_total"],
            "audio_s_per_s":x["audio_s_per_s"],
            "latency_p50_ms":1000*x["total_p50_s"],
            "latency_p95_ms":1000*x["total_p95_s"],
        })
comparisons=[]
for cell in cells:
    c=int(cell[1:])
    m=[r for r in rows if r["concurrency"]==c and r["source"]=="main"]
    p=[r for r in rows if r["concurrency"]==c and r["source"]=="pr"]
    for metric in ("requests_per_s","audio_s_per_s","latency_p50_ms","latency_p95_ms"):
        mv=[r[metric] for r in m]; pv=[r[metric] for r in p]
        comparisons.append({"concurrency":c,"metric":metric,"main_values":mv,"pr_values":pv,
                            "main_mean":statistics.fmean(mv),"pr_mean":statistics.fmean(pv),
                            "delta_pct":100*(statistics.fmean(pv)/statistics.fmean(mv)-1)})
summary={
    "measurement":"One H200, official Fun-CosyVoice3-0.5B-2512; Python frontend; Rust graph and SHM; MSTAR_COMPILE_LEVEL=0; default graph capture.",
    "heads":{"main":"2c8ffde8f7fd321c29aad3d1aad3e8cc8e8dc175","pr":"e6701a1c75866f3331a9b8f39149bc89fcda5eb9"},
    "arms":ARMS,"same_pcm_mismatch_count_by_cell_and_arm":parity,
    "rows":rows,"comparisons":comparisons,
    "limits":"Two boots per source at most; no statistical confidence interval. Client uses repeated data-URL reference audio. Throughput excludes warmup waves. Audio duration and PCM hashes audited across matched seeds.",
}
(ROOT/"summary.json").write_text(json.dumps(summary,indent=2)+"\n")
for x in comparisons:
    print(f"c{x['concurrency']} {x['metric']}: main {x['main_mean']:.3f}, PR {x['pr_mean']:.3f}, delta {x['delta_pct']:+.2f}%")
print("parity",parity)
colors={"main":"#3276ad","pr":"#e47a35"}
fig,axes=plt.subplots(1,2,figsize=(10.5,4.2))
for ax,metric,ylabel in zip(axes,("requests_per_s","latency_p50_ms"),("Completed requests/s","Median end-to-end latency (ms)")):
    for ci,c in enumerate((1,4)):
        for src,off in (("main",-.17),("pr",.17)):
            vals=[r[metric] for r in rows if r["concurrency"]==c and r["source"]==src]
            xs=[ci+off+(j-(len(vals)-1)/2)*.06 for j in range(len(vals))]
            ax.scatter(xs,vals,color=colors[src],s=48,zorder=3,label=src.upper() if ci==0 else None)
            ax.hlines(statistics.fmean(vals),ci+off-.1,ci+off+.1,color=colors[src],lw=2)
    ax.set_xticks([0,1],["c1","c4"])
    ax.set_ylabel(ylabel)
    ax.set_ylim(bottom=0)
    ax.grid(axis="y",alpha=.22)
axes[0].legend(frameon=False)
fig.suptitle("#1566 CosyVoice3 H200: matched fixed-seed 32-token-limit requests\nPer-arm points; source main vs PR; warmup excluded")
fig.tight_layout()
fig.savefig(ROOT/"served_performance.png",dpi=180,bbox_inches="tight")
