#!/bin/bash
while tmux has-session -t intree 2>/dev/null; do sleep 20; done
cd ~/mstar-pr/benchmark/wan_vae_decoder
for f in ~/stage/*; do cp "$f" "./$(basename "$f").new" && mv "./$(basename "$f").new" "./$(basename "$f")"; done
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[ "$n" = 0 ] || { echo "GPU busy ($n procs), not starting"; exit 1; }
MSTAR=~/mstar-pr SRC=runs/b200 GPU=0 PYTHONPATH=~/lpdeps bash run_ablation.sh quality
echo QUALITY_EXIT $?
