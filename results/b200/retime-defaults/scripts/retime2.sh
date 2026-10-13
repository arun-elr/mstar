#!/bin/bash
# Same session as retime.sh: M* before and the opt-in =0 (cudnn.benchmark off), chunk and served window.
set -uo pipefail
while tmux has-session -t =retime 2>/dev/null; do sleep 10; done
export PATH=$HOME/mstar-pr/.venv/bin:$PATH HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
PR=$HOME/retime/mstar MAIN=$HOME/e2e-cosmos3/mstar-main
OUT=$HOME/retime/out; SRC=$HOME/mstar-pr/benchmark/wan_vae_decoder/runs/b200
exec > >(tee -a $OUT/retime2.log) 2>&1
fresh(){ export TORCHINDUCTOR_CACHE_DIR=$OUT/cache/$1/inductor TRITON_CACHE_DIR=$OUT/cache/$1/triton; rm -rf $OUT/cache/$1; }
chunk(){ fresh c2-$1-$2-$3; echo "[retime2] $(date +%T) chunk run$1 $2 $3"
  (cd $PR/benchmark/wan_vae_decoder && MSTAR=$PR timeout 2400 python bench_wan_decoder.py --variant $3 --vae $2 \
    --latents $SRC/$2/latents.pt --ref $SRC/$2/ref_fp32_outputs.pt --out $OUT/chunk/run$1/$2) || echo "[retime2] FAILED chunk $*"; }
served(){ local run=$1 vae=$2 tag=$3 mstar=$4; shift 4; fresh s2-$run-$vae-$tag; echo "[retime2] $(date +%T) served run$run $vae $tag"
  (cd $HOME/retime/bench && MSTAR=$mstar timeout 2400 python bench_decode_nodes.py --out $OUT/nodes --tag ${vae}_${tag}_run$run \
    --node cosmos3_ar --vae $vae --size 832x480 --window 8 --overlap 2 --latents $SRC/$vae/latents.pt "$@") || echo "[retime2] FAILED served $*"; }
for run in 1 2; do
  for vae in wan22 wan21; do
    if [ $run = 1 ]; then chunk $run $vae mstar_step0; chunk $run $vae mstar_warm_lpc0; served $run $vae main $MAIN; served $run $vae lpc0 $PR --set windowed_decode_latents_per_call=0
    else served $run $vae lpc0 $PR --set windowed_decode_latents_per_call=0; served $run $vae main $MAIN; chunk $run $vae mstar_warm_lpc0; chunk $run $vae mstar_step0; fi
  done
done
echo "[retime2] $(date +%T) done"
