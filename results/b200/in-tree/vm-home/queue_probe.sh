#!/bin/bash
while tmux has-session -t lpc 2>/dev/null; do sleep 15; done
cd ~/mstar-pr/benchmark/wan_vae_decoder
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || exit 1
export PATH=~/mstar/.venv/bin:$PATH MSTAR=~/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0
for vae in wan21 wan22; do for lpc in 0 1; do for mode in compiled eager; do
  export TORCHINDUCTOR_CACHE_DIR=${TMPDIR}/probe-cache-$vae-$lpc-$mode TRITON_CACHE_DIR=${TMPDIR}/probe-tcache-$vae-$lpc-$mode
  python cache_probe_tmp.py $vae $lpc $mode 2>&1 | grep "^{"
done; done; done
echo PROBE_DONE
