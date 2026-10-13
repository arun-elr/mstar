#!/usr/bin/env bash
# Kernel profiles of the in-tree node with cudnn.benchmark off, both VAEs (same method as job.sh's prof).
set -uo pipefail
ROOT=${WAN_BENCH}/h200
OUT=$ROOT/runs
IN=${WAN_BENCH}/sglang/inputs
exec > >(tee -a "$OUT/job.log") 2>&1
export PATH="${MSTAR}/.venv/bin:$PATH" MSTAR=$ROOT/src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
cd "$ROOT/src/benchmark/wan_vae_decoder"
for vae in wan22; do
  while others="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"; [ -n "$others" ]; do
    echo "[h200] $(date +%T) GPU busy, waiting 5 min: $others"; sleep 300
  done
  sub=profile; [[ $vae == wan22 ]] && sub=profile_wan22
  export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/bench-nobench-$vae/inductor" TRITON_CACHE_DIR="$ROOT/cache/bench-nobench-$vae/triton"
  rm -rf "$ROOT/cache/bench-nobench-$vae"
  echo "[h200] $(date +%T) bench ladder_wan22 mstar_warm_nobench"
  timeout 2400 python bench_wan_decoder.py --out "$OUT/ladder_wan22" --variant mstar_warm_nobench --vae "$vae" --ref "$IN/$vae/ref_fp32_outputs.pt" \
    --latents "$IN/$vae/latents.pt" || echo "[h200] FAILED (exit $?): bench ladder_wan22 mstar_warm_nobench"
  rm -rf "$ROOT/cache/bench-nobench-$vae"
done
echo "[h200] $(date +%T) NOBENCH22_DONE"
