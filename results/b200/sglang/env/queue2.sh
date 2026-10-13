#!/usr/bin/env bash
while tmux has-session -t main 2>/dev/null || tmux has-session -t md5 2>/dev/null; do sleep 20; done
cd ~/bench
bash run_sglang.sh runs3 wan22 sglang_default sglang_default_bf16:--decode-precision,bf16 sglang_best:--quality,extra-high,--decode-precision,bf16 sglang_compile_bf16:--compile,--decode-precision,bf16
export PATH=$HOME/sgl-venv/bin:$PATH HF_HUB_OFFLINE=1 HF_HUB_DISABLE_TELEMETRY=1
O=$HOME/bench/results/md5; mkdir -p $O
L=$HOME/bench/inputs/wan22/latents.pt
m() { local tag=$1; shift; export TORCHINDUCTOR_CACHE_DIR=$O/cache/$tag/i TRITON_CACHE_DIR=$O/cache/$tag/t; echo "[md5] $(date +%T) $tag $*"; python md5_frames.py "$@" --out $O 2>&1 | grep -v "server_args:\|unexpected keys\|NCCL\|Tolerance" | tail -3; rm -rf $O/cache/$tag; }
m h stream --vae wan22 --latents $L --label sglang_default
m i whole --vae wan22 --latents $L --label sglang_default_whole
m j stream --vae wan22 --latents $L --label sglang_default_bf16 --decode-precision bf16
m k stream --vae wan22 --latents $L --label sglang_best --quality extra-high --decode-precision bf16
m l whole --vae wan22 --latents $L --label sglang_best_whole --quality extra-high --decode-precision bf16
PYTHONPATH=$HOME/diffusers039 m n diffusers --vae wan22 --latents $L --label diffusers_bf16
python md5_frames.py compare --out $O --vae wan22 diffusers_bf16 sglang_default sglang_default_bf16 sglang_best
python md5_frames.py compare --out $O --vae wan22 sglang_default sglang_default_whole
python md5_frames.py compare --out $O --vae wan22 sglang_best sglang_best_whole
echo MD5_22_DONE
bash run_sglang.sh profile wan21 sglang_default sglang_best:--quality,extra-high sglang_compile:--compile sglang_realtime:--path,realtime
bash run_sglang.sh profile wan22 sglang_default sglang_default_bf16:--decode-precision,bf16 sglang_best:--quality,extra-high,--decode-precision,bf16 sglang_compile_bf16:--compile,--decode-precision,bf16
echo QUEUE2_DONE
