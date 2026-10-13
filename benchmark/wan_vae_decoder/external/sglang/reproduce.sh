#!/bin/bash
# Every SGLang number in the PR, strictly sequential on one GPU: the screen of settings,
# 3 fresh-process runs of the reported rows, kernel profiles, and the frame-stream md5s.
#   SGL_INPUTS=/path/to/inputs bash reproduce.sh      # then: summarize_sglang.py, below
# Env as run_sglang.sh, plus DIFFUSERS039 (setup.sh).
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
R=run_sglang.sh
W21=(sglang_default sglang_best:--quality,extra-high sglang_compile:--compile sglang_realtime:--path,realtime)
W22=(sglang_default sglang_default_bf16:--decode-precision,bf16
     sglang_best:--quality,extra-high,--decode-precision,bf16 sglang_compile_bf16:--compile,--decode-precision,bf16)
bash $R screen wan21 sglang_default sglang_xhigh:--quality,extra-high sglang_high:--quality,high \
  sglang_realtime:--path,realtime sglang_compile:--compile sglang_compile_xhigh:--compile,--quality,extra-high \
  sglang_compile_maxat:--compile,--compile-mode,max-autotune-no-cudagraphs
bash $R runs3 wan21 "${W21[@]}"
bash $R screen wan22 sglang_default sglang_default_bf16:--decode-precision,bf16 \
  sglang_xhigh_bf16:--quality,extra-high,--decode-precision,bf16 sglang_xhigh:--quality,extra-high \
  sglang_realtime:--path,realtime sglang_compile_bf16:--compile,--decode-precision,bf16 \
  sglang_compile_maxat_bf16:--compile,--compile-mode,max-autotune-no-cudagraphs,--decode-precision,bf16
bash $R runs3 wan22 "${W22[@]}"
bash $R profile wan21 "${W21[@]}"
bash $R profile wan22 "${W22[@]}"

VENV="${SGL_VENV:-$HERE/sgl-venv}"
export PATH="$VENV/bin:$PATH" HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
O="${OUT:-$HERE/results}/md5"; mkdir -p "$O"
m() {  # m <cache tag> <md5_frames.py args...>: fresh compile caches per process
  local tag=$1; shift
  export TORCHINDUCTOR_CACHE_DIR="$O/cache/$tag/i" TRITON_CACHE_DIR="$O/cache/$tag/t"
  python md5_frames.py "$@" --out "$O"; rm -rf "$O/cache/$tag"
}
for vae in wan21 wan22; do
  L="$SGL_INPUTS/$vae/latents.pt" bf=(); [[ $vae == wan22 ]] && bf=(--decode-precision bf16)
  m s1 stream --vae $vae --latents "$L" --label sglang_default
  m s2 whole --vae $vae --latents "$L" --label sglang_default_whole
  m s3 stream --vae $vae --latents "$L" --label sglang_best --quality extra-high "${bf[@]}"
  m s4 whole --vae $vae --latents "$L" --label sglang_best_whole --quality extra-high "${bf[@]}"
  PYTHONPATH="${DIFFUSERS039:-$HERE/diffusers039}" m s5 diffusers --vae $vae --latents "$L" --label diffusers_bf16
  if [[ $vae == wan21 ]]; then
    m s6 stream --vae $vae --latents "$L" --label sglang_compile --compile
    m s7 stream --vae $vae --latents "$L" --label sglang_realtime --path realtime
    python md5_frames.py compare --out "$O" --vae $vae diffusers_bf16 sglang_default sglang_best sglang_compile sglang_realtime
  else
    m s6 stream --vae $vae --latents "$L" --label sglang_default_bf16 --decode-precision bf16
    python md5_frames.py compare --out "$O" --vae $vae diffusers_bf16 sglang_default sglang_default_bf16 sglang_best
  fi
  python md5_frames.py compare --out "$O" --vae $vae sglang_default sglang_default_whole
  python md5_frames.py compare --out "$O" --vae $vae sglang_best sglang_best_whole
done
# CSVs: python summarize_sglang.py results --flops-dir ../flops --out results/summary
