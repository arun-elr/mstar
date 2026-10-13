#!/bin/bash
# One benchmark process in a fresh vLLM-Omni container.
#   run1.sh OUTDIR VAE VARIANT [bench_vllm_omni.py args...]
set -euo pipefail
IMG=vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd
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
mkdir -p "$HOME/vo/runs/$out"
docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=${TMPDIR} \
  -v "$HOME/vo:/vo" -v "$HOME/vo-bench:/bench" -w /bench \
  -e VLLM_OMNI_COMMIT=f69b1f2b19cece03f6736c7da093ea25de0e24fd \
  -e DOCKER_IMAGE="$IMG" -e DOCKER_IMAGE_DIGEST="$DIGEST" -e HF_HUB_OFFLINE=1 \
  --entrypoint python "$IMG" bench_vllm_omni.py --variant "$variant" --vae "$vae" \
  --latents "/vo/inputs/$vae/latents.pt" "${ref[@]}" \
  --out "/vo/runs/$out" "$@"
