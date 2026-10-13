# vLLM-Omni Wan VAE streaming decode on one B200

Copied from the vLLM-Omni handoff. Start with [`SUMMARY.md`](SUMMARY.md). It ran on VM `wan-bench-b200-vllm`
(1× B200) inside `vllm/vllm-omni:nightly-f69b1f2b…` (digest
`sha256:d9d81b77c0c5055425f388ddde756ec4206c53e141d1ca6d1cbee21b2ba24e9f`): vLLM-Omni `f69b1f2b`,
torch 2.13.0+cu130, cuDNN 9.20.0, diffusers 0.40.0. Date: 2026-10-13.

- `runs.csv`, `variance.csv`, `profile.csv`: summary tables.
- `runs/main/wan2{1,2}/run{1,2,3}/`: 3 fresh-process runs × 3 variants per VAE. `runs/screen/`, `runs/controls/` and `runs/oneshot/` hold the screening, control and one-shot runs. `runs/profile/`: kernel tables. `runs/main.log`, `runs/main_smi.csv`: run log and GPU trace.
- `harness/`: the scripts as run. `harness/vo-bench/` is the copy of M*'s benchmark harness mounted into the container.
- `inputs/`: the VAE `config.json` files and `SHA256SUMS` of the latents and references (the same files as `../in-tree/README.md` lists).
- `env/vm-create/`: VM provisioning (`create_cmd.sh`, `cloud-init.yaml`), with IDs, key and address replaced by placeholders.

Not committed: the VAE weights at `${WAN_BENCH}/vo-vae/` (local copies of the HF files)

| file | sha256 |
|---|---|
| `wan21/diffusion_pytorch_model.safetensors` (507,591,892 B) | `d6e524b3fffede1787a74e81b30976dce5400c4439ba64222168e607ed19e793` |
| `wan22/diffusion_pytorch_model.safetensors` (2,818,777,808 B) | `62cd18f19438e35b32ac63020e2852f566e9b02f46b6cdbd87972a356e3c6f4b` |
