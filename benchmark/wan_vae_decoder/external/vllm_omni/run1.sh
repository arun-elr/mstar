#!/bin/bash
# One benchmark process in a fresh vLLM-Omni container.
#   VO_DATA=/path/to/data run1.sh OUTDIR VAE VARIANT [bench_vllm_omni.py args...]
# VO_DATA holds inputs/<vae>/{latents.pt,ref_fp32_outputs.pt} (run_all.sh's latents and
# fp32 references) and vae/<vae>/ (the VAE folders: config.json + safetensors); results go
# to VO_DATA/runs/OUTDIR. The benchmark directory is mounted read-only as /bench.
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
BENCH="$(cd "$HERE/../.." && pwd)"
IMG="${VO_IMAGE:-vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd}"
DATA="${VO_DATA:?set VO_DATA}"
DIGEST=$(docker image inspect "$IMG" --format '{{index .RepoDigests 0}}')
out=$1 vae=$2 variant=$3
shift 3
apps=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -n "$apps" ]; then
  echo "GPU has compute processes, refusing: $apps" >&2
  exit 3
fi
ref=(--ref "/vo/inputs/$vae/ref_fp32_outputs.pt")
[[ " $* " == *" --profile-kernels "* ]] && ref=()
mkdir -p "$DATA/runs/$out"
docker run --rm --gpus "device=${GPU:-0}" --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$DATA:/vo" -v "$BENCH:/bench:ro" -w /bench -e PYTHONPATH=/bench -e VO_VAE_ROOT=/vo/vae \
  -e VLLM_OMNI_COMMIT=f69b1f2b19cece03f6736c7da093ea25de0e24fd \
  -e DOCKER_IMAGE="$IMG" -e DOCKER_IMAGE_DIGEST="$DIGEST" -e HF_HUB_OFFLINE=1 \
  --entrypoint python "$IMG" external/vllm_omni/bench_vllm_omni.py --variant "$variant" --vae "$vae" \
  --latents "/vo/inputs/$vae/latents.pt" "${ref[@]}" \
  --out "/vo/runs/$out" "$@"
