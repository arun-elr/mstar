#!/bin/bash
set -euxo pipefail
export PATH="$HOME/.local/bin:$PATH"
cd ~/mstar
rm -rf .venv
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python -r ~/vm_requirements.txt \
  --extra-index-url https://download.pytorch.org/whl/cu132 \
  --extra-index-url https://download.pytorch.org/whl/cpu \
  --index-strategy unsafe-best-match
uv pip install --python .venv/bin/python --no-deps -e .
uv pip freeze --python .venv/bin/python > ~/vm_freeze.txt
.venv/bin/python -c "import torch, diffusers, triton; print('torch', torch.__version__, 'cuda', torch.version.cuda, 'cudnn', torch.backends.cudnn.version(), torch.cuda.get_device_name(0)); print('diffusers', diffusers.__version__, 'triton', triton.__version__)"
.venv/bin/python -c "import mstar.model.cosmos3.submodules, mstar.model.wan22.submodules; print('mstar imports ok')"
echo ENV_DONE
