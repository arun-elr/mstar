#!/bin/bash
# Final PR head (913295e4, compact cache): defaults (=1, cudnn.benchmark off) and opt-in =0, chunk and served window.
set -uo pipefail
while tmux has-session -t =retime2 2>/dev/null; do sleep 10; done
export PATH=$HOME/mstar-pr/.venv/bin:$PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PR=$HOME/retime/mstar-fix
OUT=$HOME/retime/out3; SRC=$HOME/mstar-pr/benchmark/wan_vae_decoder/runs/b200
mkdir -p $OUT; exec > >(tee -a $OUT/retime3.log) 2>&1
fresh(){ export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/$1/inductor TRITON_CACHE_DIR=$OUT/cache/$1/triton; rm -rf $OUT/cache/$1; }
chunk(){ fresh c3-$1-$2-$3; echo "[retime3] $(date +%T) chunk run$1 $2 $3"
  (cd $PR/benchmark/wan_vae_decoder && MSTAR=$PR timeout 2400 python bench_wan_decoder.py --variant $3 --vae $2 \
    --latents $SRC/$2/latents.pt --ref $SRC/$2/ref_fp32_outputs.pt --out $OUT/chunk/run$1/$2) || echo "[retime3] FAILED chunk $*"; }
served(){ local run=$1 vae=$2 tag=$3; shift 3; fresh s3-$run-$vae-$tag; echo "[retime3] $(date +%T) served run$run $vae $tag"
  (cd $HOME/retime/bench && MSTAR=$PR timeout 2400 python bench_decode_nodes.py --out $OUT/nodes --tag ${vae}_${tag}_run$run \
    --node cosmos3_ar --vae $vae --size 832x480 --window 8 --overlap 2 --latents $SRC/$vae/latents.pt "$@") || echo "[retime3] FAILED served $*"; }
for run in 1 2; do
  for vae in wan22 wan21; do
    if [ $run = 1 ]; then chunk $run $vae mstar_warm; chunk $run $vae mstar_warm_lpc0; served $run $vae default; served $run $vae lpc0 --set windowed_decode_latents_per_call=0
    else served $run $vae lpc0 --set windowed_decode_latents_per_call=0; served $run $vae default; chunk $run $vae mstar_warm_lpc0; chunk $run $vae mstar_warm; fi
  done
done
echo "[retime3] $(date +%T) done"
