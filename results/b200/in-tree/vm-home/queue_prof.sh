#!/bin/bash
set -uo pipefail
while tmux has-session -t extra 2>/dev/null || tmux has-session -t quality 2>/dev/null; do sleep 20; done
cd ~/mstar-pr/benchmark/wan_vae_decoder
cp ~/stage3/profile_kernels.py ./profile_kernels.py.new && mv ./profile_kernels.py.new ./profile_kernels.py
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy, not starting"; exit 1; }
OUT=runs/b200/ablation
exec > >(tee -a "$OUT/ablation.log") 2>&1
export PATH=~/mstar/.venv/bin:$PATH MSTAR=~/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
fresh() { export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$1/inductor" TRITON_CACHE_DIR="$OUT/cache/$1/triton"; rm -rf "$OUT/cache/$1"; }
for vae in wan21 wan22; do
  sub=profile_intree; [ $vae = wan22 ] && sub=profile_intree_wan22
  for v in mstar_warm px_nopad_bench; do
    fresh "prof-$vae-$v"; echo "[prof] $(date +%T) profile $vae $v"
    python profile_kernels.py --out $OUT/$sub --variant $v --vae $vae --latents runs/b200/$vae/latents.pt --trace || echo "[prof] FAILED $vae $v"
  done
done
echo "[prof] $(date +%T) PROF_DONE"
