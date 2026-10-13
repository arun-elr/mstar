#!/bin/bash
set -uo pipefail
while tmux has-session -t quality 2>/dev/null; do sleep 20; done
cd ~/mstar-pr/benchmark/wan_vae_decoder
for f in ~/stage2/*; do cp "$f" "./$(basename "$f").new" && mv "./$(basename "$f").new" "./$(basename "$f")"; done
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ] || { echo "GPU busy, not starting"; exit 1; }
OUT=runs/b200/ablation
exec > >(tee -a "$OUT/ablation.log") 2>&1
export PATH=~/mstar/.venv/bin:$PATH MSTAR=~/mstar-pr HF_HUB_OFFLINE=1 CUDA_VISIBLE_DEVICES=0 PYTHONPATH=~/lpdeps
fresh() { export TORCHINDUCTOR_CACHE_DIR="$OUT/cache/$1/inductor" TRITON_CACHE_DIR="$OUT/cache/$1/triton"; rm -rf "$OUT/cache/$1"; }
L21=runs/b200/wan21/latents.pt R21=runs/b200/wan21/ref_fp32_outputs.pt
L22=runs/b200/wan22/latents.pt
# cudnn.benchmark off control, in-tree node, 3 runs ABC / CBA / BCA (n=2: AB, BA, BA)
for r in 1 2 3; do
  case $r in 1) o="mstar_warm mstar_warm_nobench";; *) o="mstar_warm_nobench mstar_warm";; esac
  for v in $o; do
    fresh "control-run$r-$v"; echo "[extra] $(date +%T) bench control/run$r $v"
    python bench_wan_decoder.py --out $OUT/control/run$r --variant $v --vae wan21 --latents $L21 --ref $R21 || echo "[extra] FAILED $v"
  done
done
for vae in wan21 wan22; do
  l=$L21; [ $vae = wan22 ] && l=$L22
  d=$OUT/quality/$vae; chmod u+w $d
  for v in diffusers_bf16_node_input mstar_warm_nobench; do
    fresh "quality-$vae-$v"; echo "[extra] $(date +%T) quality decode $vae $v"
    python quality.py decode --vae $vae --latents $l --out $d --variant $v || echo "[extra] FAILED decode $vae $v"
  done
  fresh "coverage-$vae"; echo "[extra] $(date +%T) coverage $vae"
  python quality.py coverage --vae $vae --latents $l --out $OUT/quality_results || echo "[extra] FAILED coverage $vae"
  echo "[extra] $(date +%T) analyze $vae"
  python quality.py analyze --vae $vae --dir $d --out $OUT/quality_results --source-video ~/wan-bench/assets/draw.mp4 || echo "[extra] FAILED analyze $vae"
done
echo "[extra] $(date +%T) EXTRA_DONE"
