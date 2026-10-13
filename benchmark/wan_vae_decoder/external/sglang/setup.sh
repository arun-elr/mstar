#!/bin/bash
# The SGLang environment the numbers were measured in: sglang[diffusion]==0.5.21 from PyPI
# with the stack it pins (torch 2.13.0+cu130), the two VAEs at the pinned revisions, and
# diffusers 0.39.0 on the side for the stock-diffusers md5 (md5_frames.py diffusers).
#   SGL_VENV=./sgl-venv DIFFUSERS039=./diffusers039 bash setup.sh
# Needs python3.12 with headers (Triton), uv, and ninja comes with the venv.
set -euxo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
VENV="${SGL_VENV:-$HERE/sgl-venv}"
D39="${DIFFUSERS039:-$HERE/diffusers039}"
uv venv --python 3.12 "$VENV"
uv pip install --python "$VENV/bin/python" "sglang[diffusion]==0.5.21"
uv pip install --python "$VENV/bin/python" --target "$D39" --no-deps "diffusers==0.39.0"
uv pip freeze --python "$VENV/bin/python" > "$HERE/sgl_freeze.txt"
export HF_HUB_DISABLE_TELEMETRY=1
"$VENV/bin/python" - <<'PY'
from huggingface_hub import snapshot_download
for repo, rev in [("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"),
                  ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635")]:
    print(snapshot_download(repo, revision=rev, allow_patterns=["vae/*"]))
PY
"$VENV/bin/python" -c "import torch, triton, sglang; print('torch', torch.__version__, 'cudnn', torch.backends.cudnn.version(), 'triton', triton.__version__, 'sglang', sglang.__version__)"
"$VENV/bin/python" -c "from sglang.multimodal_gen.runtime.models.vaes.wanvae import cat_pad_channels_last_3d, dup_up3d_add; print('fused kernels:', cat_pad_channels_last_3d is not None, dup_up3d_add is not None)"
