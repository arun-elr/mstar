import json
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
root=Path(__file__).parent
d=json.loads((root/'mstar-590-attention-resource-bench.json').read_text())
fig,axes=plt.subplots(1,2,figsize=(11,4.8),layout='constrained',sharey=True)
labels=['Encoder cond','Encoder blank','Decoder','Expert cross','Expert self']
colors=['#2a9d8f','#e9c46a','#457b9d']
for ax,phase in zip(axes,['cold','warm']):
    x=np.arange(len(labels)); width=.25
    series={
      'Eager wrapper':[c['eager'][phase]['wrapper']['gpu_ms']['p50'] for c in d['cases']],
      'Direct same mask':[c['eager'][phase]['direct_same_mask']['gpu_ms']['p50'] for c in d['cases']],
      'Captured replay':[c['captured'][phase]['replay']['gpu_ms']['p50'] for c in d['cases']],
    }
    for i,(name,vals) in enumerate(series.items()):
        ax.bar(x+(i-1)*width,vals,width,color=colors[i],label=name)
    ax.set_yscale('log')
    ax.set_ylim(.008,4)
    ax.set_xticks(x,labels,rotation=28,ha='right')
    ax.set_title(f'{phase.capitalize()} layout')
    ax.grid(axis='y',which='both',alpha=.16)
axes[0].set_ylabel('Median CUDA-event latency (ms, log scale)')
axes[1].legend(fontsize=8,frameon=False,loc='upper left')
fig.suptitle('M* 5902054c0f attention resources on one H200: synthetic BF16 Q/K/V')
fig.text(.5,-.08,'TRITON backend; 20 warmups, 100 measured samples. Captured replay uses a fixed warm-size bucket.\nWrapper-only geometry and masks; this does not measure whole-engine or served-session performance.',ha='center',fontsize=8)
fig.savefig(root/'attention-resource-h200-5902054c0f.png',dpi=180,bbox_inches='tight')
fig.savefig(root/'attention-resource-h200-5902054c0f.svg',bbox_inches='tight')
