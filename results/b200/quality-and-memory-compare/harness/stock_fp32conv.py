"""M*'s stock diffusers bf16 float output (quality.py's diffusers_bf16.f.npy, bf16 bits as int16)
converted to uint8 two ways: M*'s to_uint8 on the bf16 tensor (must reproduce diffusers_bf16.u8.npy)
and SGLang's conversion, (x.float() / 2 + 0.5) * 255 clamped (writes stock_bf16_fp32conv.u8.npy)."""
import hashlib, sys
from pathlib import Path
import numpy as np, torch
sys.path.insert(0, "${MSTAR_PR}/benchmark/wan_vae_decoder")
from common import to_uint8

d = Path(sys.argv[1])
f = np.load(d / "diffusers_bf16.f.npy", mmap_mode="r")
st = np.load(d / "diffusers_bf16.u8.npy", mmap_mode="r")
out = np.lib.format.open_memmap(d / "stock_bf16_fp32conv.u8.npy", "w+", np.uint8, st.shape)
same_m = True
for i in range(0, f.shape[0], 32):
    x = torch.from_numpy(np.ascontiguousarray(f[i:i + 32])).view(torch.bfloat16)  # [n, 3, H, W]
    m = to_uint8(x).permute(0, 2, 3, 1).numpy()
    same_m &= bool(np.array_equal(m, st[i:i + 32]))
    out[i:i + 32] = ((x.float() / 2 + 0.5) * 255).clamp(0, 255).to(torch.uint8).permute(0, 2, 3, 1).numpy()
out.flush()
h = hashlib.md5()
for i in range(0, out.shape[0], 64):
    h.update(np.ascontiguousarray(out[i:i + 64]).tobytes())
print(d.name, "to_uint8(bf16) reproduces stock u8:", same_m, "| fp32-conversion md5:", h.hexdigest())
