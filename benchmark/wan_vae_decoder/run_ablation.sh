#!/usr/bin/env bash
# Ablations, run-to-run variance and kernel profiles on one GPU, strictly
# sequential; each phase is its own invocation:
#   bash run_ablation.sh px        # TF32 check, prototype ablations, FLOP count, kernel profiles
#   bash run_ablation.sh variance  # 3 independent runs x 9 variants, orders ABC / CBA / BCA
#   bash run_ablation.sh wan22     # prototypes on the Wan2.2 VAE, FLOP count, kernel profiles
#   bash run_ablation.sh intree    # in-tree node vs prototype, 3 runs each VAE (ABC / CBA / BCA),
#                                  # then M*'s decoder node on served 8-latent windows
#   bash run_ablation.sh ladder    # the change ladder through M*'s node, compiled and eager, both VAEs:
#                                  # 3 runs, a kernel profile and the frame-stream md5 per step
#   bash run_ablation.sh lpc       # windowed_decode_latents_per_call 1 / 2 / 3 / 0 on M*'s node, both VAEs:
#                                  # benchmark chunk and served window, 2 runs each, frame-stream md5
#   bash run_ablation.sh quality   # every ladder step decoded over the whole stream, both VAEs,
#                                  # compared with stock bf16 and fp32, plus window coverage
#                                  # (quality.py; needs lpips; VIDEO=<source clip> adds encode+decode PSNR)
# Every variant is a fresh process with fresh Inductor/Triton caches (fresh compile).
# Reuses run_all.sh's latents and fp32 references: SRC=runs/<run_all run> (needs
# its wan21/ suite, and wan22/ for the wan22 phase). Outputs go to $OUT.
set -uo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
MSTAR="${MSTAR:-$(cd "$HERE/../.." && pwd)}"
SRC="${SRC:?set SRC to a run_all.sh output dir}"
OUT="${OUT:-$SRC/ablation}"
L21="$SRC/wan21/latents.pt";  R21="$SRC/wan21/ref_fp32_outputs.pt"
L22="$SRC/wan22/latents.pt";  R22="$SRC/wan22/ref_fp32_outputs.pt"
mkdir -p "$OUT"
exec > >(tee -a "$OUT/ablation.log") 2>&1
export PATH="$MSTAR/.venv/bin:$PATH" MSTAR HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES="${GPU:-0}"
cd "$HERE"

fresh() {  # fresh <tag>: per-process compile caches
  export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$1/inductor" TRITON_CACHE_DIR="$OUT/cache/$1/triton"
  rm -rf "$OUT/cache/$1"
}
bench() {  # bench <subdir> <variant> <args...>; DUMP_DIR: run1's in-tree frames go there
  local sub="$1" v="$2"; shift 2
  fresh "$sub-$v"
  if [[ -n ${DUMP_DIR:-} && $sub == */run1 && $v == mstar_* ]]; then
    mkdir -p "$DUMP_DIR"; set -- "$@" --dump-u8 "$DUMP_DIR/$v.u8.npy"
  fi
  echo "[ablation] $(date +%T) bench $sub $v $*"
  timeout "${VARIANT_TIMEOUT:-2400}" python bench_wan_decoder.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[ablation] FAILED (exit $?): bench $sub $v"
}
prof() {  # prof <subdir> <variant> <args...>
  local sub="$1" v="$2"; shift 2
  fresh "prof-$sub-$v"
  echo "[ablation] $(date +%T) profile $sub $v"
  timeout "${VARIANT_TIMEOUT:-2400}" python profile_kernels.py --out "$OUT/$sub" --variant "$v" "$@" \
    || echo "[ablation] FAILED (exit $?): profile $sub $v"
}

runs3() {  # runs3 <subdir> <args> -- <variants...>: 3 runs in the orders ABC / CBA / BCA
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
runs2() {  # runs2 <subdir> <args> -- <variants...>: 2 runs, forward then reverse order
  local sub="$1"; shift
  local -a c=() v=()
  while [[ $1 != -- ]]; do c+=("$1"); shift; done; shift
  v=("$@")
  local -a orders=("${v[*]}" "$(printf '%s\n' "${v[@]}" | tac | tr '\n' ' ')")
  for r in 1 2; do
    for x in ${orders[$((r-1))]}; do bench "$sub/run$r" "$x" "${c[@]}"; done
  done
}
nodes() {  # nodes <tag> <args...>: bench_decode_nodes.py, fresh compile caches
  local tag="$1"; shift
  fresh "nodes-$tag"
  echo "[ablation] $(date +%T) nodes $tag $*"
  timeout "${VARIANT_TIMEOUT:-2400}" python bench_decode_nodes.py --out "$OUT/nodes" --tag "$tag" "$@" \
    || echo "[ablation] FAILED (exit $?): nodes $tag"
}

PX=(px_base px_cg px_nopad px_bench px_ro px_maxat px_nopad_bench px_cg_nopad px_cg_nopad_bench px_cg_nopad_bench_fp16)
VAR=(diffusers_bf16 diffusers_bf16_compile mstar_ar_ctx8 mstar_ar_ctx8_eager channels_last_compile
     channels_last_one_call_compile px_base px_fp16 px_nopad_bench)
INTREE=(mstar_warm mstar_warm_eager mstar_ar_ctx8 px_nopad_bench)
LADDER=(mstar_step0 mstar_step1 mstar_step2 mstar_step3 mstar_step5
        mstar_step0_eager mstar_step1_eager mstar_step2_eager mstar_step3_eager mstar_step5_eager)
QUALITY=(ref_fp32 diffusers_bf16 diffusers_bf16_node_input mstar_ar_ctx8 diffusers_bf16_compile channels_last_compile
         channels_last_one_call_compile px_nopad px_nopad_bench mstar_warm mstar_warm_nobench)

case "${1:?phase}" in
px)
  c=(--vae wan21 --latents "$L21")
  bench tf32 ref_fp32 "${c[@]}" --ref "$R21" --warmup 10 --chunks 10 --repeats 1
  bench tf32 ref_fp32_tf32 "${c[@]}" --ref "$R21" --warmup 10 --chunks 10 --repeats 1
  for v in "${PX[@]}"; do
    extra=(); [[ $v != *cg* ]] && extra=(--layout-check)
    bench px "$v" "${c[@]}" --ref "$R21" "${extra[@]}"
  done
  fresh flops; python profile_kernels.py --flops "${c[@]}" --out "$OUT/profile" || echo "[ablation] FAILED flops"
  for v in diffusers_bf16 mstar_ar_ctx8 "${PX[@]}"; do prof profile "$v" "${c[@]}"; done
  ;;
variance)
  runs3 variance --vae wan21 --latents "$L21" --ref "$R21" -- "${VAR[@]}"
  ;;
wan22)
  c=(--vae wan22 --latents "$L22")
  for v in ${WAN22_VARIANTS:-px_base px_nopad_bench}; do bench wan22 "$v" "${c[@]}" --ref "$R22"; done
  fresh flops22; python profile_kernels.py --flops "${c[@]}" --out "$OUT/profile_wan22" || echo "[ablation] FAILED flops22"
  for v in diffusers_bf16 mstar_ar_ctx8 ${WAN22_VARIANTS:-px_base px_nopad_bench}; do
    prof profile_wan22 "$v" "${c[@]}"
  done
  ;;
intree)
  runs3 intree --vae wan21 --latents "$L21" --ref "$R21" -- "${INTREE[@]}"
  runs3 intree_wan22 --vae wan22 --latents "$L22" --ref "$R22" -- "${INTREE[@]}"
  runs3 control --vae wan21 --latents "$L21" --ref "$R21" -- mstar_warm mstar_warm_nobench  # cudnn.benchmark off
  # Served windows: 8 latents, 2 re-pinned, so 6 new latents per decoder call.
  for vae in wan21 wan22; do
    l=$L21; [[ $vae == wan22 ]] && l=$L22
    w=(--node cosmos3_ar --vae "$vae" --size 832x480 --window 8 --overlap 2 --latents "$l")
    nodes "${vae}_default" "${w[@]}" --psnr-windows 4
    nodes "${vae}_per_call_1" "${w[@]}" --set windowed_decode_latents_per_call=1
  done
  ;;
ladder)
  for vae in ${LADDER_VAES:-wan21 wan22}; do
    l=$L21 r=$R21 sfx=""; [[ $vae == wan22 ]] && l=$L22 r=$R22 sfx=_wan22
    d="$OUT/ladder_frames/$vae"
    DUMP_DIR="$d" runs3 "ladder$sfx" --vae "$vae" --latents "$l" --ref "$r" -- "${LADDER[@]}"
    for v in "${LADDER[@]}"; do prof "profile_ladder$sfx" "$v" --vae "$vae" --latents "$l"; done
    fresh "ladder-stock-$vae"
    python quality.py decode --vae "$vae" --latents "$l" --out "$d" --variant diffusers_bf16_node_input \
      || echo "[ablation] FAILED (exit $?): stock decode $vae"
    for chain in "${LADDER[*]:0:5}" "${LADDER[*]:5}"; do
      python quality.py compare --vae "$vae" --dir "$d" --chain "${chain// /,}" --ref diffusers_bf16_node_input \
        --out "$OUT/ladder_frames_$vae.csv.part" || echo "[ablation] FAILED (exit $?): compare $vae"
      cat "$OUT/ladder_frames_$vae.csv.part" >> "$OUT/ladder_frames_$vae.csv"; rm -f "$OUT/ladder_frames_$vae.csv.part"
    done
    [[ -n ${KEEP_FRAMES:-} ]] || rm -f "$d"/*.npy
  done
  ;;
lpc)
  LPC=(1 2 3 0)
  for vae in ${LPC_VAES:-wan21 wan22}; do
    l=$L21 r=$R21 sfx=""; [[ $vae == wan22 ]] && l=$L22 r=$R22 sfx=_wan22
    d="$OUT/lpc_frames/$vae"
    DUMP_DIR="$d" runs2 "lpc$sfx" --vae "$vae" --latents "$l" --ref "$r" -- "${LPC[@]/#/mstar_warm_lpc}"
    w=(--node cosmos3_ar --vae "$vae" --size 832x480 --window 8 --overlap 2 --latents "$l")
    for run in 1 2; do
      order=("${LPC[@]}"); [[ $run == 2 ]] && order=(0 3 2 1)
      for n in "${order[@]}"; do
        nodes "${vae}_lpc${n}_run${run}" "${w[@]}" --set windowed_decode_latents_per_call="$n"
      done
    done
    fresh "lpc-stock-$vae"
    python quality.py decode --vae "$vae" --latents "$l" --out "$d" --variant diffusers_bf16_node_input \
      || echo "[ablation] FAILED (exit $?): stock decode $vae"
    chain="${LPC[*]/#/mstar_warm_lpc}"
    python quality.py compare --vae "$vae" --dir "$d" --chain "${chain// /,}" \
      --ref diffusers_bf16_node_input --out "$OUT/lpc_frames_$vae.csv" || echo "[ablation] FAILED: compare $vae"
    [[ -n ${KEEP_FRAMES:-} ]] || rm -f "$d"/*.npy
  done
  ;;
quality)
  # Frames go to $QDIR (several GB per variant); deleted after each VAE's analysis.
  for vae in ${QUALITY_VAES:-wan21 wan22}; do
    l=$L21; [[ $vae == wan22 ]] && l=$L22
    d="${QDIR:-$OUT/quality}/$vae"
    q() {  # q <variant> [tag]: one decode, fresh process and compile caches
      local tag="${2:-}"
      fresh "quality-$vae-$1$tag"
      echo "[ablation] $(date +%T) quality decode $vae $1 $tag"
      timeout "${VARIANT_TIMEOUT:-2400}" python quality.py decode --vae "$vae" --latents "$l" --out "$d" \
        --variant "$1" ${tag:+--tag "$tag"} || echo "[ablation] FAILED (exit $?): quality decode $vae $1 $tag"
    }
    for v in "${QUALITY[@]}"; do q "$v"; done
    q px_nopad_bench run2
    q mstar_warm run2
    echo "[ablation] $(date +%T) quality analyze $vae"
    fresh "coverage-$vae"
    python quality.py coverage --vae "$vae" --latents "$l" --out "$OUT/quality_results" \
      || echo "[ablation] FAILED (exit $?): quality coverage $vae"
    python quality.py analyze --vae "$vae" --dir "$d" --out "$OUT/quality_results" ${VIDEO:+--source-video "$VIDEO"} \
      || echo "[ablation] FAILED (exit $?): quality analyze $vae"
    [[ -n ${KEEP_FRAMES:-} ]] || rm -f "$d"/*.npy
    if [[ -n ${VIDEO:-} ]]; then  # encode + decode vs the clip: its first encoded piece (10 chunks)
      for v in ref_fp32 diffusers_bf16 mstar_ar_ctx8 px_nopad_bench mstar_warm; do
        fresh "recon-$vae-$v"
        python quality.py decode --vae "$vae" --latents "$l" --out "$d-recon" --variant "$v" --chunks 10 \
          || echo "[ablation] FAILED (exit $?): recon decode $vae $v"
      done
      python quality.py analyze --vae "$vae" --dir "$d-recon" --out "$OUT/quality_recon_results" --source-video "$VIDEO" \
        || echo "[ablation] FAILED (exit $?): recon analyze $vae"
      rm -f "$d-recon"/*.npy
    fi
  done
  ;;
*) echo "unknown phase $1" >&2; exit 2 ;;
esac
echo "[ablation] $(date +%T) PHASE_DONE $1"
