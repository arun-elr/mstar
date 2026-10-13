#!/usr/bin/env bash
# Single queued GPU job: Wan VAE streaming-decode benchmark on one GPU, one
# variant per process. Never kills anything; refuses to run on a GPU with other
# compute processes unless ALLOW_SHARED=1 (numbers are then labelled as shared-GPU).
#
#   GPU=0 VIDEO=clip.mp4 bash run_all.sh               # Wan 2.1 VAE
#   GPU=0 VIDEO=clip.mp4 WAN22=1 bash run_all.sh       # also the Wan2.2 VAE
#   GPU=0 VIDEO=clip.mp4 SUITES=wan22 bash run_all.sh  # only the Wan2.2 suite
#
# Env: GPU (index, default 0), MSTAR (checkout whose model code the M* variants
# import, default this repo), RUN (run name), ALLOW_SHARED, SUITES, WAN22,
# VIDEO (source clip; random latents without it), VARIANT_TIMEOUT (s, default 2400).
# Raw outputs go to runs/$RUN/ (gitignored).
set -uo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
MSTAR="${MSTAR:-$(cd "$HERE/../.." && pwd)}"
GPU="${GPU:-0}"
RUN="${RUN:-$(date +%Y%m%d-%H%M%S)}"
OUT="$HERE/runs/$RUN"
VIDEO="${VIDEO:-}"
VARIANT_TIMEOUT="${VARIANT_TIMEOUT:-2400}"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1

export PATH="$MSTAR/.venv/bin:$PATH" MSTAR HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="$GPU"
export TORCHINDUCTOR_CACHE_DIR="$OUT/inductor-cache"   # cold compile, measured honestly
cd "$HERE"

others="$(nvidia-smi -i "$GPU" --query-compute-apps=pid,process_name,used_memory --format=csv,noheader)"
if [ -n "$others" ] && [ "${ALLOW_SHARED:-0}" != 1 ]; then
  echo "[run_all] GPU $GPU has other compute processes; not starting (set ALLOW_SHARED=1 to override):"
  echo "$others"
  exit 2
fi
echo "[run_all] GPU $GPU apps at start: ${others:-none}"
nvidia-smi -i "$GPU" --query-gpu=name,driver_version,memory.used,memory.total,clocks.max.sm,power.limit --format=csv | tee "$OUT/gpu.csv"
nvidia-smi -i "$GPU" --query-gpu=timestamp,utilization.gpu,memory.used,clocks.sm,power.draw --format=csv -lms 1000 > "$OUT/gpu_trace.csv" &
SAMPLER=$!
trap 'kill $SAMPLER 2>/dev/null' EXIT

run() {  # run <results-subdir> <bench args...>
  local sub="$1"; shift
  echo "[run_all] $(date +%T) bench $*"
  timeout "$VARIANT_TIMEOUT" python bench_wan_decoder.py --out "$OUT/$sub" "$@" \
    || echo "[run_all] FAILED (exit $?): $*"
}

suite() {  # suite <wan21|wan22>
  local vae="$1" dir="$OUT/$1"
  local lat="$dir/latents.pt" ref="$dir/ref_fp32_outputs.pt"
  mkdir -p "$dir"
  echo "[run_all] $(date +%T) prepare latents ($vae)"
  if [ -f "$VIDEO" ]; then
    python prepare_latents.py --vae "$vae" --video "$VIDEO" --chunks 85 --out "$lat"
  else
    python prepare_latents.py --vae "$vae" --random --chunks 85 --out "$lat"
  fi
  local c=(--vae "$vae" --latents "$lat")
  run "$vae" --variant ref_fp32 "${c[@]}" --repeats 1 --save-ref "$ref"
  c+=(--ref "$ref")
  run "$vae" --variant diffusers_bf16 "${c[@]}" --profile
  run "$vae" --variant diffusers_bf16_compile "${c[@]}"
  run "$vae" --variant mstar_ar_ctx8_eager "${c[@]}"
  run "$vae" --variant mstar_ar_ctx8 "${c[@]}"
  run "$vae" --variant mstar_warm_eager "${c[@]}"
  run "$vae" --variant mstar_warm "${c[@]}"
  for k in 4 2 1; do run "$vae" --variant "mstar_ar_ctx$k" "${c[@]}" --repeats 1; done
  run "$vae" --variant mstar_wan22_oneshot_81f "${c[@]}"
  run "$vae" --variant mstar_wan22_oneshot_241f "${c[@]}"
  run "$vae" --variant mstar_wan22_oneshot_241f_tiled "${c[@]}"
  run "$vae" --variant channels_last "${c[@]}" --layout-check
  run "$vae" --variant channels_last_one_call "${c[@]}" --layout-check
  run "$vae" --variant channels_last_compile "${c[@]}"
  run "$vae" --variant one_call_compile "${c[@]}"
  run "$vae" --variant channels_last_one_call_compile "${c[@]}"
  run "$vae" --variant px_base "${c[@]}" --profile
  run "$vae" --variant px_nopad_bench "${c[@]}" --layout-check
  run "$vae" --variant px_fp16 "${c[@]}"
  run "$vae/drift" --variant diffusers_bf16 "${c[@]}"   # same as the 2nd run: clock/contention drift check
  python summarize.py "$OUT" --out "$OUT/summary"
}

SUITES="${SUITES:-wan21}"
[ "${WAN22:-0}" = 1 ] && SUITES="$SUITES wan22"
for s in $SUITES; do suite "$s"; done
echo "[run_all] $(date +%T) done -> $OUT"
