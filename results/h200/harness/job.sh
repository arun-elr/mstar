#!/usr/bin/env bash
# H200 replay of the B200 Wan VAE decoder benchmark (PR mstar-project/mstar#425, c56b572b).
# One GPU job at a time, strictly sequential, one variant per process with fresh
# Inductor/Triton caches (as run_ablation.sh). Waits (re-checking every 5 min) if
# any other compute process is on the GPU. Never kills anything.
set -uo pipefail
ROOT=${WAN_BENCH}/h200
HERE=$ROOT/src/benchmark/wan_vae_decoder
OUT=$ROOT/runs
IN=${WAN_BENCH}/sglang/inputs
L21=$IN/wan21/latents.pt; R21=$IN/wan21/ref_fp32_outputs.pt
L22=$IN/wan22/latents.pt; R22=$IN/wan22/ref_fp32_outputs.pt
QDIR=$ROOT/quality
mkdir -p "$OUT"
exec > >(tee -a "$OUT/job.log") 2>&1
export PATH="${MSTAR}/.venv/bin:$PATH" MSTAR=$ROOT/src HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
cd "$HERE"

nvidia-smi -i 0 --query-gpu=name,driver_version,memory.used,memory.total,clocks.max.sm,power.limit --format=csv | tee "$OUT/gpu.csv"
nvidia-smi -i 0 --query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw,temperature.gpu --format=csv -lms 1000 > "$OUT/gpu_trace.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT
date +%s > "$OUT/start_epoch"

wait_idle() {
  local others
  while others="$(nvidia-smi -i 0 --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"; [ -n "$others" ]; do
    echo "[h200] $(date +%T) GPU busy, waiting 5 min: $others"
    sleep 300
  done
}
fresh() {
  export TORCHINDUCTOR_CACHE_DIR="$ROOT/cache/$1/inductor" TRITON_CACHE_DIR="$ROOT/cache/$1/triton"
  rm -rf "$ROOT/cache/$1"
}
bench() {  # bench <subdir> <variant> <args...>
  local sub="$1" v="$2"; shift 2
  wait_idle; fresh "$sub-$v"
  echo "[h200] $(date +%T) bench $sub $v $*"
  timeout 2400 python bench_wan_decoder.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[h200] FAILED (exit $?): bench $sub $v"
  rm -rf "$ROOT/cache/$sub-$v"
}
prof() {  # prof <subdir> <variant> <args...>
  local sub="$1" v="$2"; shift 2
  wait_idle; fresh "prof-$sub-$v"
  echo "[h200] $(date +%T) profile $sub $v"
  timeout 2400 python profile_kernels.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[h200] FAILED (exit $?): profile $sub $v"
  rm -rf "$ROOT/cache/prof-$sub-$v"
}
runs3() {  # same orders as run_ablation.sh: ABC / CBA / BCA
  local sub="$1"; shift
  local -a c=() v=()
  while [[ $1 != -- ]]; do c+=("$1"); shift; done; shift
  v=("$@")
  local n=${#v[@]}
  local -a orders=("${v[*]}" "$(printf '%s\n' "${v[@]}" | tac | tr '\n' ' ')" "${v[*]:$((n/3))} ${v[*]:0:$((n/3))}")
  for r in 1 2 3; do
    for x in ${orders[$((r-1))]}; do bench "$sub/run$r" "$x" "${c[@]}"; done
  done
}
qdec() {  # qdec <vae> <latents> <variant> [tag]
  local vae="$1" l="$2" v="$3" tag="${4:-}"
  wait_idle; fresh "quality-$vae-$v$tag"
  echo "[h200] $(date +%T) quality decode $vae $v $tag"
  timeout 2400 python quality.py decode --vae "$vae" --latents "$l" --out "$QDIR/$vae" --variant "$v" ${tag:+--tag "$tag"} \
    || echo "[h200] FAILED (exit $?): quality decode $vae $v $tag"
  rm -rf "$ROOT/cache/quality-$vae-$v$tag"
}

INTREE=(mstar_warm mstar_warm_eager mstar_ar_ctx8 px_nopad_bench)
PHASES="${PHASES:-intree ladder intree_wan22 refcheck profile quality drift}"
for phase in $PHASES; do
  echo "[h200] $(date +%T) PHASE $phase"
  case $phase in
  intree)
    runs3 intree --vae wan21 --latents "$L21" --ref "$R21" -- "${INTREE[@]}" ;;
  ladder)
    c=(--vae wan21 --latents "$L21" --ref "$R21")
    bench ladder diffusers_bf16 "${c[@]}"
    bench ladder diffusers_bf16_compile "${c[@]}"
    bench ladder channels_last_compile "${c[@]}"
    bench ladder channels_last_one_call_compile "${c[@]}"
    bench ladder px_nopad "${c[@]}" --layout-check
    bench ladder mstar_warm_nobench "${c[@]}" ;;
  intree_wan22)
    runs3 intree_wan22 --vae wan22 --latents "$L22" --ref "$R22" -- "${INTREE[@]}" ;;
  refcheck)  # H200 fp32 reference (TF32 off) vs the B200 one, first 10 steady chunks
    bench refcheck_wan21 ref_fp32 --vae wan21 --latents "$L21" --ref "$R21" --warmup 10 --chunks 10 --repeats 1
    bench refcheck_wan22 ref_fp32 --vae wan22 --latents "$L22" --ref "$R22" --warmup 10 --chunks 10 --repeats 1 ;;
  profile)
    for v in mstar_ar_ctx8 mstar_warm px_nopad_bench; do prof profile "$v" --vae wan21 --latents "$L21"; done
    for v in mstar_ar_ctx8 mstar_warm px_nopad_bench; do prof profile_wan22 "$v" --vae wan22 --latents "$L22"; done ;;
  quality)
    for vae in wan21 wan22; do
      l=$L21; [[ $vae == wan22 ]] && l=$L22
      for v in ref_fp32 diffusers_bf16 mstar_ar_ctx8 mstar_warm mstar_warm_nobench; do qdec "$vae" "$l" "$v"; done
      qdec "$vae" "$l" mstar_warm run2
      qdec "$vae" "$l" mstar_warm_nobench run2
      wait_idle; fresh "coverage-$vae"
      echo "[h200] $(date +%T) quality coverage $vae"
      python quality.py coverage --vae "$vae" --latents "$l" --out "$OUT/quality_results" \
        || echo "[h200] FAILED (exit $?): coverage $vae"
      echo "[h200] $(date +%T) output check $vae"
      python "$ROOT/check_outputs.py" --vae "$vae" --dir "$QDIR/$vae" --out "$OUT/quality_results" \
        --b200 "$HERE/results/b200/quality.csv" || echo "[h200] FAILED (exit $?): check_outputs $vae"
      [[ -n ${KEEP_FRAMES:-} ]] || rm -f "$QDIR/$vae"/*.npy
    done ;;
  followup)  # coverage wan21 (first attempt: quality.py does not create --out), nobench kernel profiles
    mkdir -p "$OUT/quality_results"
    wait_idle; fresh coverage-wan21
    echo "[h200] $(date +%T) quality coverage wan21"
    python quality.py coverage --vae wan21 --latents "$L21" --out "$OUT/quality_results" \
      || echo "[h200] FAILED (exit $?): coverage wan21"
    prof profile mstar_warm_nobench --vae wan21 --latents "$L21"
    prof profile_wan22 mstar_warm_nobench --vae wan22 --latents "$L22" ;;
  drift)  # same as the ladder's diffusers_bf16: clock / contention drift check
    bench drift diffusers_bf16 --vae wan21 --latents "$L21" --ref "$R21" ;;
  esac
done
date +%s > "$OUT/end_epoch"
echo "[h200] $(date +%T) JOB_DONE"
