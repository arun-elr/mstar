#!/usr/bin/env bash
# Phase 2 on the B200 VM, strictly sequential: Wan2.2 suite, then two Wan 2.1 ablations.
set -uo pipefail
cd ~/wan-bench
MSTAR=$HOME/mstar GPU=0 RUN=b200-wan22 SUITES=wan22 bash run_all.sh
OUT=$HOME/wan-bench/results/b200-wan21
exec >> "$OUT/run.log" 2>&1
export PATH="$HOME/mstar/.venv/bin:$PATH" MSTAR=$HOME/mstar HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor-cache"
d=$OUT/wan21
c=(--vae wan21 --latents $d/latents.pt --ref $d/ref_fp32_outputs.pt)
for v in rx_cl_compile rx_chunk_compile; do
  echo "[phase2] $(date +%T) bench --variant $v"
  timeout 2400 python bench_wan_decoder.py --out "$d" --variant $v "${c[@]}" || echo "[phase2] FAILED (exit $?): $v"
done
python summarize.py "$d" > "$d/summary.md"
echo "[phase2] $(date +%T) PHASE2_DONE"
