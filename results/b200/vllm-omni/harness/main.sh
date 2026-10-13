#!/bin/bash
# Full batch: 3 fresh-process runs per variant (orders ABC / CBA / BCA), Wan 2.1 then Wan 2.2,
# controls, whole-clip decode, kernel profiles.
cd "$HOME/vo-bench"
R=$HOME/vo/runs
nvidia-smi --query-gpu=timestamp,clocks.sm,power.draw,temperature.gpu,utilization.gpu --format=csv -l 5 \
  > "$R/main_smi.csv" &
smi=$!
log() { echo "=== $* $(date -u +%T)"; }
one() {  # out vae variant [args]
  log "$3 $2 -> $1"
  ./run1.sh "$@" 2>&1 | grep -E '^\{"variant"|Error|error|Traceback|refusing' | cut -c1-300
}
A=vllm_omni_default B=vllm_omni_best C=vllm_omni_best_cudnnbench
for vae in wan21 wan22; do
  one "main/$vae/run1" $vae $A; one "main/$vae/run1" $vae $B; one "main/$vae/run1" $vae $C
  one "main/$vae/run2" $vae $C; one "main/$vae/run2" $vae $B; one "main/$vae/run2" $vae $A
  one "main/$vae/run3" $vae $B; one "main/$vae/run3" $vae $C; one "main/$vae/run3" $vae $A
done
# the other vLLM-Omni configurations (one run each; how vllm_omni_best was chosen)
for v in vllm_omni_off vllm_omni_channels_last vllm_omni_lingbot_exact vllm_omni_lingbot_fused; do
  one screen/wan21 wan21 $v
done
# controls: M*'s own variants in this container, and the reference diffusers forwards through vLLM-Omni
one controls/wan21 wan21 diffusers_bf16
one controls/wan21 wan21 px_nopad_bench
one controls/wan22 wan22 vllm_omni_off
one controls/wan22 wan22 diffusers_bf16
one controls/wan22 wan22 px_nopad_bench
# whole-clip decode (Wan22Pipeline's vae.decode), 241 frames; Wan 2.2's deploy config turns tiling on
one oneshot/wan21 wan21 vllm_omni_oneshot_241f
one oneshot/wan22 wan22 vllm_omni_oneshot_241f_tiled
one oneshot/wan22 wan22 vllm_omni_oneshot_241f
# kernel profiles (6 steady chunks after 10 warm-up chunks, as profile_kernels.py)
for vae in wan21 wan22; do
  for v in $A $B $C; do
    log "profile $v $vae"
    ./run1.sh "profile/$vae" $vae $v --profile-kernels 2>&1 | grep -E '^\{"variant"|Error|Traceback' | cut -c1-400
  done
done
kill $smi
log done
