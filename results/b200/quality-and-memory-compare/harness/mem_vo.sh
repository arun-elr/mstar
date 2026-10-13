#!/bin/bash
IMG=vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd
for vae in wan21 wan22; do for v in vllm_omni_default vllm_omni_best; do
  apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader); [ -z "$apps" ] || { echo "GPU busy, refusing: $apps"; exit 3; }
  echo "[mem] $(date +%T) $vae $v"
  docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=${TMPDIR} -v "$HOME/vo:/vo" -v "$HOME/vo-bench:/bench:ro" -v "$HOME/qc:/qc" -w /bench \
    -e PYTHONPATH=/bench -e HF_HUB_OFFLINE=1 --entrypoint python "$IMG" /qc/mem_probe.py --engine vllm_omni --variant $v --vae $vae \
    --latents /vo/inputs/$vae/latents.pt --out /qc/mem_probe2.jsonl 2>&1 | grep '^{' || echo "[mem] FAILED $vae $v"
done; done; echo MEM_DONE
