#!/bin/bash
# SGLang-Omni (sglang.multimodal_gen) Wan VAE decoder bench: VM environment.
# SGLang pinned to the v0.5.21 release from PyPI, with the stack it pins (torch 2.13.0).
set -euxo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3.12-dev tmux rsync
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
cd ~
rm -rf ~/sgl-venv
uv venv --python 3.12 ~/sgl-venv
uv pip install --python ~/sgl-venv/bin/python "sglang[diffusion]==0.5.21"
uv pip freeze --python ~/sgl-venv/bin/python > ~/sgl_freeze.txt
export HF_HUB_DISABLE_TELEMETRY=1
~/sgl-venv/bin/python - <<'EOF'
from huggingface_hub import snapshot_download
for repo, rev in [("Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "0fad780a534b6463e45facd96134c9f345acfa5b"),
                  ("Wan-AI/Wan2.2-TI2V-5B-Diffusers", "b8fff7315c768468a5333511427288870b2e9635")]:
    print(snapshot_download(repo, revision=rev, allow_patterns=["vae/*"]))
EOF
~/sgl-venv/bin/python -c "import torch, triton, diffusers, sglang; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), torch.cuda.get_device_name(0)); print('triton', triton.__version__, 'diffusers', diffusers.__version__, 'sglang', sglang.__version__)"
~/sgl-venv/bin/python -c "from sglang.multimodal_gen.runtime.models.vaes.wanvae import AutoencoderKLWan, cat_pad_channels_last_3d, dup_up3d_add; print('wanvae ok, fused kernels:', cat_pad_channels_last_3d is not None, dup_up3d_add is not None)"
md5sum ~/sgl-venv/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/models/vaes/wanvae.py \
       ~/sgl-venv/lib/python3.12/site-packages/sglang/multimodal_gen/runtime/models/vaes/wan_vae_cuda_opt.py \
       ~/sgl-venv/lib/python3.12/site-packages/sglang/kernels/ops/diffusion/layout/wan_causal_cache_triton.py
echo SETUP_DONE
