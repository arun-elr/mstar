#!/bin/bash
export PATH=$HOME/sgl-venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 PYTHONPATH=$HOME/bench
cd ~/bench
for job in "wan21 sglang_default" "wan21 sglang_best" "wan22 sglang_default" "wan22 sglang_default_bf16" "wan22 sglang_best"; do
  set -- $job
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader); [ -z "$apps" ] || { echo "GPU busy, refusing: $apps"; exit 3; }
  echo "[mem] $(date +%T) $1 $2"
  python ~/qc/mem_probe.py --engine sglang --variant $2 --vae $1 --latents ~/bench/inputs/$1/latents.pt --out ~/qc/mem_probe2.jsonl 2>&1 | grep '^{' || echo "[mem] FAILED $1 $2"
done; echo MEM_DONE
