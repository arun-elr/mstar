#!/bin/bash
# Re-time the in-tree streaming decoder at the PR defaults (latents_per_call=1, cudnn.benchmark off).
set -uo pipefail
export MSTAR=$HOME/retime/mstar PATH=$HOME/mstar-pr/.venv/bin:$PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
OUT=$HOME/retime/out; SRC=$HOME/mstar-pr/benchmark/wan_vae_decoder/runs/b200
mkdir -p $OUT; exec > >(tee -a $OUT/retime.log) 2>&1
fresh(){ export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/$1/inductor TRITON_CACHE_DIR=$OUT/cache/$1/triton; rm -rf $OUT/cache/$1; }
chunk(){ # chunk <run> <vae> <variant>
  fresh chunk-$1-$2-$3; echo "[retime] $(date +%T) chunk run$1 $2 $3"
  (cd $MSTAR/benchmark/wan_vae_decoder && timeout 2400 python bench_wan_decoder.py --variant $3 --vae $2 \
    --latents $SRC/$2/latents.pt --ref $SRC/$2/ref_fp32_outputs.pt --out $OUT/chunk/run$1/$2) || echo "[retime] FAILED chunk $*"
}
served(){ # served <run> <vae>
  fresh served-$1-$2; echo "[retime] $(date +%T) served run$1 $2"
  (cd $HOME/retime/bench && timeout 2400 python bench_decode_nodes.py --out $OUT/nodes --tag ${2}_default_run$1 \
    --node cosmos3_ar --vae $2 --size 832x480 --window 8 --overlap 2 --latents $SRC/$2/latents.pt --preset default) \
    || echo "[retime] FAILED served $*"
}
for run in 1 2; do
  vaes="wan22 wan21"; [ $run = 2 ] && vaes="wan21 wan22"
  for vae in $vaes; do
    if [ $run = 1 ]; then chunk $run $vae mstar_warm; served $run $vae; chunk $run $vae mstar_warm_bench
    else chunk $run $vae mstar_warm_bench; served $run $vae; chunk $run $vae mstar_warm; fi
  done
done
echo "[retime] $(date +%T) done"
