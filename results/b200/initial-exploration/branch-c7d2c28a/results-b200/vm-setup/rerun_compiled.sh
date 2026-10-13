#!/usr/bin/env bash
# Re-run the variants that failed in run_all.sh (Triton could not build its CUDA
# helper: python3.12-dev was missing), same args and env as run_all.sh.
set -uo pipefail
cd ~/wan-bench
OUT=$HOME/wan-bench/results/b200-wan21
exec >> "$OUT/run.log" 2>&1
export PATH="$HOME/mstar/.venv/bin:$PATH" MSTAR=$HOME/mstar HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor-cache"
d=$OUT/wan21
c=(--vae wan21 --latents $d/latents.pt --ref $d/ref_fp32_outputs.pt)
run() { local sub="$1"; shift; echo "[rerun] $(date +%T) bench $*"; timeout 2400 python bench_wan_decoder.py --out "$OUT/$sub" "$@" || echo "[rerun] FAILED (exit $?): $*"; }
run wan21 --variant diffusers_bf16_compile "${c[@]}"
run wan21 --variant mstar_ar_ctx8 "${c[@]}"
for k in 4 2 1; do run wan21 --variant "mstar_ar_ctx$k" "${c[@]}" --repeats 1; done
run wan21 --variant rx_cl_chunk_compile "${c[@]}"
run wan21 --variant rx_full "${c[@]}" --profile
run wan21 --variant rx_full_fp16 "${c[@]}"
run wan21/drift2 --variant diffusers_bf16 "${c[@]}"
python summarize.py "$d" > "$d/summary.md"
echo "[rerun] $(date +%T) RERUN_DONE"
