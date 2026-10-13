#!/usr/bin/env bash
# Batch 2 on one GPU, strictly sequential; each phase is its own invocation:
#   bash batch2.sh px        # TF32 check, beat-60 prototypes, FLOP count, kernel profiles
#   bash batch2.sh variance  # 3 independent runs x 8 variants, orders ABC / CBA / BCA
#   bash batch2.sh wan22     # best prototype on the Wan2.2 VAE (+ profiles)
#   bash batch2.sh pxfix     # rerun px_cg / px_ro after the common.py fixes
# Every variant is a fresh process with fresh Inductor/Triton caches (fresh compile).
# Expects run_all.sh's Wan 2.1 / Wan2.2 latents and fp32 references under $BASE.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MSTAR="${MSTAR:-$HOME/oss/mstar}"
BASE="${BASE:-$HERE/results}"
OUT="${OUT:-$BASE/b200-batch2}"
L21="$BASE/b200-wan21/wan21/latents.pt";  R21="$BASE/b200-wan21/wan21/ref_fp32_outputs.pt"
L22="$BASE/b200-wan22/wan22/latents.pt";  R22="$BASE/b200-wan22/wan22/ref_fp32_outputs.pt"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/batch2.log") 2>&1
export PATH="$MSTAR/.venv/bin:$PATH" MSTAR HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${GPU:-0}"
cd "$HERE"

fresh() {  # fresh <tag>: per-process compile caches
  export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$1/inductor" TRITON_CACHE_DIR="$OUT/cache/$1/triton"
  rm -rf "$OUT/cache/$1"
}
bench() {  # bench <subdir> <variant> <args...>
  local sub="$1" v="$2"; shift 2
  fresh "$sub-$v"
  echo "[batch2] $(date +%T) bench $sub $v $*"
  timeout "${VARIANT_TIMEOUT:-2400}" python bench_wan_decoder.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[batch2] FAILED (exit $?): bench $sub $v"
}
prof() {  # prof <subdir> <variant> <args...>
  local sub="$1" v="$2"; shift 2
  fresh "prof-$sub-$v"
  echo "[batch2] $(date +%T) profile $sub $v"
  timeout "${VARIANT_TIMEOUT:-2400}" python profile_kernels.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[batch2] FAILED (exit $?): profile $sub $v"
}

PX=(px_base px_cg px_nopad px_bench px_ro px_maxat px_nopad_bench px_cg_nopad px_cg_nopad_bench px_cg_nopad_bench_fp16)
VAR=(diffusers_bf16 diffusers_bf16_compile mstar_ar_ctx8 mstar_ar_ctx8_eager rx_cl_compile rx_cl_chunk_compile rx_full rx_full_fp16 px_nopad_bench)

case "${1:?phase}" in
px)
  c=(--vae wan21 --latents "$L21")
  bench tf32 ref_fp32 "${c[@]}" --ref "$R21" --warmup 10 --chunks 10 --repeats 1
  bench tf32 ref_fp32_tf32 "${c[@]}" --ref "$R21" --warmup 10 --chunks 10 --repeats 1
  for v in "${PX[@]}"; do
    extra=(); [[ $v == *nopad* || $v == px_base ]] && [[ $v != *cg* ]] && extra=(--layout-check)
    bench px "$v" "${c[@]}" --ref "$R21" "${extra[@]}"
  done
  fresh flops; python profile_kernels.py --flops "${c[@]}" --out "$OUT/profile" || echo "[batch2] FAILED flops"
  for v in diffusers_bf16 mstar_ar_ctx8 rx_full "${PX[@]}"; do prof profile "$v" "${c[@]}"; done
  ;;
pxfix)  # px_cg / px_ro timings after the cache-slot / CUDA-graph-tree fixes in common.py
  for v in px_cg px_ro; do bench px "$v" --vae wan21 --latents "$L21" --ref "$R21"; done
  ;;
variance)
  c=(--vae wan21 --latents "$L21" --ref "$R21")
  n=${#VAR[@]}
  orders=("${VAR[*]}" "$(printf '%s\n' "${VAR[@]}" | tac | tr '\n' ' ')" "${VAR[*]:$((n/3))} ${VAR[*]:0:$((n/3))}")
  for r in 1 2 3; do
    for v in ${orders[$((r-1))]}; do bench "variance/run$r" "$v" "${c[@]}"; done
  done
  ;;
wan22)
  c=(--vae wan22 --latents "$L22")
  for v in ${WAN22_VARIANTS:-px_base px_cg_nopad_bench}; do bench wan22 "$v" "${c[@]}" --ref "$R22"; done
  fresh flops22; python profile_kernels.py --flops "${c[@]}" --out "$OUT/profile_wan22" || echo "[batch2] FAILED flops22"
  for v in diffusers_bf16 mstar_ar_ctx8 ${WAN22_VARIANTS:-px_base px_cg_nopad_bench}; do prof profile_wan22 "$v" "${c[@]}"; done
  ;;
esac
echo "[batch2] $(date +%T) PHASE_DONE $1"
