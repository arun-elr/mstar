#!/bin/bash
# dock.sh python args...   one container; M* copy $M (default /w22/mstar), interpreter $PY (default python)
set -euo pipefail
IMG=vllm/vllm-omni:nightly-f69b1f2b19cece03f6736c7da093ea25de0e24fd
M=${M:-/w22/mstar}
PY=${PY:-python}
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)
[ -z "$apps" ] || { echo "GPU busy: $apps" >&2; exit 3; }
exec docker run --rm --gpus all --ipc=host --user "$(id -u):$(id -g)" -e HOME=${TMPDIR} -e USER=bench \
  -v "$HOME/vo:/vo" -v "$HOME/w22:/w22" -w /w22/h -e MSTAR=$M -e HF_HUB_OFFLINE=1 \
  -e PYTHONPATH=$M:$M/benchmark/wan_vae_decoder -e PYTHONUNBUFFERED=1 \
  ${DOCK_ENV:-} --entrypoint $PY "$IMG" "$@"
