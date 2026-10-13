#!/usr/bin/env bash
while tmux has-session -t main 2>/dev/null; do sleep 20; done
export PATH=$HOME/sgl-venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
O=$HOME/bench/results/md5; mkdir -p $O; cd ~/bench
L=$HOME/bench/inputs/wan21/latents.pt
m() { local tag=$1; shift; export TORCHINDUCTOR_CACHE_DIR=$O/cache/$tag/i TRITON_CACHE_DIR=$O/cache/$tag/t; echo "[md5] $(date +%T) $tag $*"; python md5_frames.py "$@" --out $O 2>&1 | grep -v "server_args:\|unexpected keys\|NCCL\|Tolerance" | tail -3; rm -rf $O/cache/$tag; }
m a stream --vae wan21 --latents $L --label sglang_default
m b whole --vae wan21 --latents $L --label sglang_default_whole
m c stream --vae wan21 --latents $L --label sglang_best --quality extra-high
m d whole --vae wan21 --latents $L --label sglang_best_whole --quality extra-high
m e stream --vae wan21 --latents $L --label sglang_compile --compile
m f stream --vae wan21 --latents $L --label sglang_realtime --path realtime
PYTHONPATH=$HOME/diffusers039 m g diffusers --vae wan21 --latents $L --label diffusers_bf16
python md5_frames.py compare --out $O --vae wan21 diffusers_bf16 sglang_default sglang_best sglang_compile sglang_realtime
python md5_frames.py compare --out $O --vae wan21 sglang_default sglang_default_whole
python md5_frames.py compare --out $O --vae wan21 sglang_best sglang_best_whole
echo MD5_DONE
