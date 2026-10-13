#!/bin/bash
# Latents for both prompts, then the decode ladder (one fresh process per step), then metrics + media.
#   WORK=./work PY=python bash run_all.sh     # WORK holds runs/ (latents, frames) and out/ (media, CSVs)
# Needs the Wan2.2-TI2V-5B-Diffusers snapshot in the HF cache (prompts.md: revision, install).
set -e
HERE="$(cd "$(dirname "$0")" && pwd)"
WORK="${WORK:-$HERE/work}"
PY="${PY:-python}"
export HF_HUB_OFFLINE=1
mkdir -p "$WORK" && cd "$WORK"
for p in fox waves; do
  [ -f runs/${p}_latents.pt ] || $PY "$HERE/gen_latents.py" --prompt-id $p --out runs/${p}_latents.pt
done
for p in fox waves; do
  for s in stock_bf16 stock_fp32 s0_mstar_before s1_warm_cache s2_channels_last s3_one_call s4_cudnn_benchmark c1_warm_cache_eager; do
    $PY "$HERE/decode_step.py" --step $s --latents runs/${p}_latents.pt --out runs/$p
  done
done
for p in fox waves; do $PY "$HERE/analyze.py" --run runs/$p --out out --prompt-id $p; done
$PY "$HERE/tables.py" out runs   # merges metrics_<prompt>.csv into metrics.csv
