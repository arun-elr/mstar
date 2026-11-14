import json
from pathlib import Path
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

root = Path(__file__).parent
reports = {
    'Reference 68f4d7a': json.loads((root/'reference-matched-warm-590-script.json').read_text()),
    'M* Python 5902054c0f': json.loads((root/'mstar-590-warm-python.json').read_text()),
    'M* Rust 5902054c0f': json.loads((root/'mstar-590-warm-rust.json').read_text()),
}
assert len({r['script_sha256'] for r in reports.values()}) == 1
assert len({tuple(r['observation_sha256']) for r in reports.values()}) == 1
fig, ax = plt.subplots(figsize=(9.2,4.7),layout='constrained')
x=np.arange(3); width=.25
colors=['#2a9d8f','#e76f51','#457b9d']
for i,(name,r) in enumerate(reports.items()):
    vals=[r['summary'][str(j)]['p50_ms'] for j in range(3)]
    bars=ax.bar(x+(i-1)*width,vals,width,label=name,color=colors[i])
    ax.bar_label(bars,labels=[f'{v:.0f}' for v in vals],padding=3,fontsize=8)
ax.set_xticks(x,['Cold (21 rows)','Warm 1 (24 rows)','Warm 2 (24 rows)'])
ax.set_ylabel('Median request latency (ms)')
ax.set_ylim(0,630)
ax.set_title('Praxis DROID on one H200: matched observations and benchmark script')
ax.legend(fontsize=8,frameon=False,loc='upper left')
ax.text(0,-0.24,'4 measured sessions after 1 warmup; M* -O0, no graph capture. Reference compile=None.\nSame 46-frame synthetic stream and 3 observation SHA-256 hashes; two-step sampling.',
        transform=ax.transAxes,fontsize=8,va='top')
fig.savefig(root/'latency-h200-5902054c0f.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'latency-h200-5902054c0f.svg',bbox_inches='tight')

rows=[json.loads(line) for line in (root/'mstar-590-stage-bench.jsonl').read_text().splitlines()]
by=defaultdict(list)
for row in rows: by[row['stage']].append(row)
name_to_key=[('VAE encoder','Praxis1VaeEncoderSubmodule'),('Backbone','Praxis1BackboneSubmodule'),('Action expert','Praxis1ActionExpertSubmodule'),('Action head','Praxis1ActionHeadSubmodule')]
stages=[]
for call in (6,7,8):
    per={}
    for name,key in name_to_key:
        idx=range(2*call,2*call+2) if name in ('Backbone','Action expert') else (call,)
        per[name]=sum(by[key][j]['gpu_ms'] for j in idx)
    stages.append(per)
fig,ax=plt.subplots(figsize=(8.6,4.5),layout='constrained')
bottom=np.zeros(3)
for name,color in [('VAE encoder','#f4a261'),('Backbone','#457b9d'),('Action expert','#2a9d8f'),('Action head','#8d99ae')]:
    vals=np.array([s[name] for s in stages])
    ax.bar(np.arange(3),vals,bottom=bottom,color=color,label=name)
    bottom+=vals
for i,v in enumerate(bottom): ax.text(i,v+6,f'{v:.0f} ms',ha='center',fontsize=9)
ax.set_xticks(np.arange(3),['Cold','Warm 1','Warm 2'])
ax.set_ylabel('Sum of CUDA-event stage time (ms)')
ax.set_ylim(0,560)
ax.set_title('M* 5902054c0f steady-state stage attribution')
ax.legend(fontsize=8,frameon=False,loc='upper left')
ax.text(0,-0.25,'Diagnostic copy, -O0. CUDA synchronize around each submodule forward; 1 warmup session excluded.\nBackbone and expert each sum two Euler calls. This instrumentation changes timing.',
        transform=ax.transAxes,fontsize=8,va='top')
fig.savefig(root/'stages-h200-5902054c0f.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'stages-h200-5902054c0f.svg',bbox_inches='tight')
