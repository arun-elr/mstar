#!/bin/bash
while tmux has-session -t analyze21 2>/dev/null; do sleep 10; done
cd ~/mstar-pr/benchmark/wan_vae_decoder
export PATH=~/mstar/.venv/bin:$PATH MSTAR=~/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/lpdeps
OUT=runs/b200/ablation
for vae in wan21 wan22; do
  export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/det-$vae/inductor TRITON_CACHE_DIR=$OUT/cache/det-$vae/triton; rm -rf $OUT/cache/det-$vae
  echo "[det] $(date +%T) decode $vae mstar_warm_nobench run2"
  python quality.py decode --vae $vae --latents runs/b200/$vae/latents.pt --out $OUT/quality/$vae --variant mstar_warm_nobench --tag run2 || echo "[det] FAILED $vae"
done
for vae in wan21 wan22; do
  echo "[det] $(date +%T) analyze $vae"
  python quality.py analyze --vae $vae --dir $OUT/quality/$vae --out $OUT/quality_results --source-video ~/wan-bench/assets/draw.mp4 || echo "[det] FAILED analyze $vae"
done
echo "[det] $(date +%T) DET_DONE"
