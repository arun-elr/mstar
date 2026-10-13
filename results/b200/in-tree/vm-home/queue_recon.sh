#!/bin/bash
cd ~/mstar-pr/benchmark/wan_vae_decoder
export PATH=~/mstar/.venv/bin:$PATH MSTAR=~/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/lpdeps
OUT=runs/b200/ablation
for vae in wan21 wan22; do
  for v in ref_fp32 diffusers_bf16 mstar_ar_ctx8 px_nopad_bench mstar_warm; do
    export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/recon-$vae-$v/inductor TRITON_CACHE_DIR=$OUT/cache/recon-$vae-$v/triton; rm -rf $OUT/cache/recon-$vae-$v
    echo "[recon] $(date +%T) decode $vae $v"
    python quality.py decode --vae $vae --latents runs/b200/$vae/latents.pt --out $OUT/quality_recon/$vae --variant $v --chunks 10 || echo "[recon] FAILED $vae $v"
  done
  python quality.py analyze --vae $vae --dir $OUT/quality_recon/$vae --out $OUT/quality_recon_results --source-video ~/wan-bench/assets/draw.mp4 || echo "[recon] FAILED analyze $vae"
  rm -f $OUT/quality_recon/$vae/*.npy
done
echo "[recon] $(date +%T) RECON_DONE"
