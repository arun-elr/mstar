#!/usr/bin/env bash
# SGLang Wan VAE decoder under M*'s methodology, one GPU, strictly sequential.
#   bash run_sglang.sh screen <vae>          # one run per candidate setting
#   bash run_sglang.sh runs3 <vae> <label>:<flags> ...  # 3 fresh-process runs, orders ABC / CBA / BCA
#   bash run_sglang.sh profile <vae> <label>:<flags> ...
# <flags> are bench_sglang_wan.py flags joined with "," (e.g. sglang_best:--compile).
# Every process gets fresh Inductor/Triton caches (fresh compile, fresh Triton JIT).
# Env: SGL_VENV (setup.sh's venv, default ./sgl-venv), SGL_INPUTS (dir with
# <vae>/{latents.pt,ref_fp32_outputs.pt} from run_all.sh), OUT (default ./results), GPU.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
OUT="${OUT:-$HERE/results}"
VENV="${SGL_VENV:-$HERE/sgl-venv}"
PY="$VENV/bin/python"
# Activated-venv PATH: SGLang JIT-builds its lossless norm+SiLU post-op kernel with ninja.
export PATH="$VENV/bin:$PATH"
export HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1 CUDA_VISIBLE_DEVICES="${GPU:-0}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
cd "$HERE"

busy=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[[ $busy -eq 0 ]] || { echo "GPU has $busy compute processes; refusing"; exit 3; }
nvidia-smi --query-gpu=timestamp,utilization.gpu,clocks.sm,power.draw,temperature.gpu,memory.used \
  --format=csv -l 5 >> "$OUT/gpu_trace.csv" &
TRACE=$!
trap 'kill $TRACE 2>/dev/null' EXIT

inputs() { L="${SGL_INPUTS:?set SGL_INPUTS}/$1/latents.pt"; R="$SGL_INPUTS/$1/ref_fp32_outputs.pt"; }
fresh() {
  export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$1/inductor" TRITON_CACHE_DIR="$OUT/cache/$1/triton"
  rm -rf "$OUT/cache/$1"
}
one() {  # one <subdir> <vae> <label:flags> [extra...]
  local sub="$1" vae="$2" spec="$3"; shift 3
  local label="${spec%%:*}" flags="${spec#*:}"
  [[ $flags == "$spec" ]] && flags=""
  inputs "$vae"
  fresh "$sub-$label"
  echo "[sglang] $(date +%T) $sub $vae $label ${flags//,/ } $*"
  # shellcheck disable=SC2086
  timeout "${VARIANT_TIMEOUT:-2400}" "$PY" bench_sglang_wan.py --vae "$vae" --variant "$label" \
    --latents "$L" --ref "$R" --out "$OUT/$sub" ${flags//,/ } "$@" \
    || echo "[sglang] FAILED (exit $?): $sub $vae $label"
  rm -rf "$OUT/cache/$sub-$label"
}

phase="${1:?phase}"; vae="${2:?vae}"; shift 2
case "$phase" in
screen)
  for spec in "$@"; do one "screen_$vae" "$vae" "$spec" --layout-check; done ;;
runs3)
  n=$#; v=("$@")
  orders=("${v[*]}" "$(printf '%s\n' "${v[@]}" | tac | tr '\n' ' ')" "${v[*]:$((n/3))} ${v[*]:0:$((n/3))}")
  for r in 1 2 3; do
    for spec in ${orders[$((r-1))]}; do one "$vae/run$r" "$vae" "$spec"; done
  done ;;
profile)
  for spec in "$@"; do one "profile_$vae" "$vae" "$spec" --profile; done ;;
*) echo "unknown phase $phase" >&2; exit 2 ;;
esac
echo "[sglang] $(date +%T) PHASE_DONE $phase $vae"
