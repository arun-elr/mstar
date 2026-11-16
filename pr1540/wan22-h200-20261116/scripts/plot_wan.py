import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
root = Path(__file__).parent
report = json.loads((root / 'pr1540-quality.json').read_text())
fig, axes = plt.subplots(1, 2, figsize=(12, 4.3), sharex=True)
for c in report['comparisons']:
    if c['label'] == 'bf16_repeat_seed41':
        continue
    label = c['label'].replace('bf16_vs_fp8_seed', 'FP8 vs BF16, seed ').replace('fp8_repeat_seed', 'FP8 repeat, seed ').replace('bf16_vs_keep_first_last_seed', 'first/last BF16 vs BF16, seed ').replace('full_fp8_vs_keep_first_last_seed', 'first/last BF16 vs full FP8, seed ')
    x = range(len(c['per_frame_psnr_db']))
    axes[0].plot(x, c['per_frame_psnr_db'], label=label, linewidth=1.8)
    axes[1].plot(x, c['per_frame_ssim'], label=label, linewidth=1.8)
axes[0].set_ylabel('PSNR (dB)')
axes[1].set_ylabel('SSIM')
for ax in axes:
    ax.set_xlabel('decoded frame (0–32)')
    ax.grid(alpha=0.2)
axes[0].set_title('Decoded video quality by frame')
axes[1].set_title('Same inputs, 20 steps, 832×480')
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='lower center', ncol=2, fontsize=8, bbox_to_anchor=(0.5, -0.06))
fig.tight_layout(rect=(0, 0.12, 1, 1))
fig.savefig(root / 'wan22-quality-per-frame.svg', bbox_inches='tight')
fig.savefig(root / 'wan22-quality-per-frame.png', dpi=150, bbox_inches='tight')
import statistics
records = [json.loads(line) for line in (root / 'pr1540-forward.log').read_text().splitlines() if line.startswith('{')]
fig, ax = plt.subplots(figsize=(6.5, 4.2))
seeds = [r['seed'] for r in records]
positions = range(len(seeds))
width = 0.34
ax.bar([p-width/2 for p in positions], [r['bf16_ms_median'] for r in records], width=width, label='BF16')
ax.bar([p+width/2 for p in positions], [r['fp8_ms_median'] for r in records], width=width, label='dense FP8')
ax.set_xticks(list(positions), [str(s) for s in seeds])
ax.set_xlabel('seed')
ax.set_ylabel('median DiT forward time (ms)')
ax.set_title('Real Wan 2.2 checkpoint, H200, batch 2 latent forward')
ax.legend()
ax.grid(axis='y', alpha=0.2)
fig.tight_layout()
fig.savefig(root / 'wan22-dit-forward-latency.svg')
fig.savefig(root / 'wan22-dit-forward-latency.png', dpi=150)
