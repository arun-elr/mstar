#!/bin/bash
# Latents for both prompts, then the decode ladder (one fresh process per step), then metrics + media.
set -e
export HF_HOME=$HOME/samples/hf HF_HUB_OFFLINE=1
cd ~/samples
PY=venv/bin/python
for p in fox waves; do
  [ -f runs/${p}_latents.pt ] || $PY gen_latents.py --prompt-id $p --out runs/${p}_latents.pt 2>&1 | grep -E '^\{|Error|error' || true
  test -f runs/${p}_latents.pt
done
for p in fox waves; do
  for s in stock_bf16 stock_fp32 s0_mstar_before s1_warm_cache s2_channels_last s3_one_call s4_cudnn_benchmark c1_warm_cache_eager; do
    $PY decode_step.py --step $s --latents runs/${p}_latents.pt --out runs/$p 2>&1 | grep -E '^\{|Error|error' || true
    test -f runs/$p/$s.json
  done
done
for p in fox waves; do
  $PY analyze.py --run runs/$p --out out --prompt-id $p 2>&1 | grep -E 'Error|error' || true
done
ls -la out
echo ALL_DONE
