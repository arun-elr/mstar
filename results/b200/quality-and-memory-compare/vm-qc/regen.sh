#!/bin/bash
set -uo pipefail
cd ~/mstar-pr/benchmark/wan_vae_decoder
export PATH=$HOME/mstar-pr/.venv/bin:$PATH MSTAR=$HOME/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
for vae in wan21 wan22; do
  for v in diffusers_bf16 ref_fp32; do
    export TORCHINDUCTOR_CACHE_DIR=$HOME/qc/cache/$vae-$v/i TRITON_CACHE_DIR=$HOME/qc/cache/$vae-$v/t
    rm -rf $HOME/qc/cache/$vae-$v
    echo "[qc] $(date +%T) decode $vae $v"
    python quality.py decode --vae $vae --latents runs/b200/$vae/latents.pt --out $HOME/qc/$vae --variant $v || echo "[qc] FAILED $vae $v"
  done
done
echo QC_REGEN_DONE
