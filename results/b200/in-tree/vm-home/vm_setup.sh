#!/bin/bash
set -euxo pipefail
sudo DEBIAN_FRONTEND=noninteractive apt-get update -q
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q ffmpeg rsync
curl -LsSf https://astral.sh/uv/0.13.0/install.sh | sh
export PATH="$HOME/.local/bin:$PATH"
[ -d ~/mstar ] || git clone -q https://github.com/mstar-project/mstar ~/mstar
cd ~/mstar && git fetch -q origin && git checkout -q 99f5108b50ccd2289b875cc67b2cf776bc7c445b
uv sync --frozen --extra wan22 --extra cosmos3
.venv/bin/python -c "import torch, diffusers; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), torch.cuda.get_device_name(0)); print('diffusers', diffusers.__version__)"
echo SETUP_DONE
