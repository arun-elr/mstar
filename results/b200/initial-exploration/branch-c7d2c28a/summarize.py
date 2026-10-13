"""Results dir of per-variant JSONs -> markdown table.

  python summarize.py results/<run>
"""

import json
import sys
from pathlib import Path

ORDER = [
    "ref_fp32", "diffusers_bf16", "diffusers_bf16_compile",
    "mstar_ar_ctx8_eager", "mstar_ar_ctx8", "mstar_ar_ctx4", "mstar_ar_ctx2", "mstar_ar_ctx1",
    "mstar_wan22_oneshot_81f", "mstar_wan22_oneshot_241f", "mstar_wan22_oneshot_241f_tiled",
    "rx_cl", "rx_cl_chunk", "rx_cl_compile", "rx_chunk_compile", "rx_cl_chunk_compile", "rx_full", "rx_full_fp16",
]


def main(run: Path):
    rows = {}
    for p in sorted(run.rglob("*.json")):
        if p.name.endswith(".latencies.json"):
            continue
        r = json.loads(p.read_text())
        key = r["variant"] if p.parent == run else f"{r['variant']} ({p.parent.name})"
        rows[key] = r
    keys = [k for k in ORDER if k in rows] + sorted(k for k in rows if k not in ORDER)
    env = next(iter(rows.values()))["env"] if rows else {}
    print(f"GPU: {env.get('gpu')}  torch {env.get('torch')}  cuDNN {env.get('cudnn')}  "
          f"diffusers {env.get('diffusers')}  mstar {env.get('mstar_commit')}\n")
    print("| variant | median ms/chunk | p90 ms | decoder fps (12/median) | peak alloc GiB | PSNR float dB | PSNR u8 dB | note |")
    print("|---|---:|---:|---:|---:|---:|---:|---|")
    for k in keys:
        r = rows[k]
        note = []
        if "steady_decode_latents" in r:
            note.append(f"decodes {r['steady_decode_latents']} latents/chunk")
        if "clip_frames" in r:
            note.append(f"whole {r['clip_frames']}-frame clip {r['clip_ms_median']:.0f} ms, per-12-frame share")
        if "conv_inputs_not_channels_last" in r:
            note.append(f"non-CL conv inputs {r['conv_inputs_not_channels_last']}")
        print(f"| {k} | {r['median_ms']} | {r['p90_ms']} | {r['decoder_fps']} | "
              f"{r.get('peak_allocated_gib', '')} | {r.get('psnr_float_db_vs_fp32') or ''} | "
              f"{r.get('psnr_u8_db_vs_fp32') or ''} | {'; '.join(note)} |")


if __name__ == "__main__":
    main(Path(sys.argv[1]))
