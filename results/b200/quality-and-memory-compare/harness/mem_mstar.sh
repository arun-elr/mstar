#!/bin/bash
cd ~/mstar-pr/benchmark/wan_vae_decoder
export PATH=$HOME/mstar-pr/.venv/bin:$PATH MSTAR=$HOME/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=$PWD
for vae in wan21 wan22; do for v in mstar_warm mstar_warm_lpc1; do
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader); [ -z "$apps" ] || { echo "GPU busy, refusing: $apps"; exit 3; }
  export TORCHINDUCTOR_CACHE_DIR=$HOME/qc/cache/mem-$vae-$v/i TRITON_CACHE_DIR=$HOME/qc/cache/mem-$vae-$v/t; rm -rf $HOME/qc/cache/mem-$vae-$v
  echo "[mem] $(date +%T) $vae $v"
  python ~/qc/mem_probe.py --engine mstar --variant $v --vae $vae --latents runs/b200/$vae/latents.pt --out ~/qc/mem_probe2.jsonl 2>&1 | grep '^{\|Error' || echo "[mem] FAILED $vae $v"
done; done; echo MEM_DONE
