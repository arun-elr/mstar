"""profile_kernels.py at c56b572b only knows mstar_warm[_eager]; this adds
mstar_warm_nobench (the same node with vae_decode_cudnn_benchmark off) without
editing the benchmark code. Run from benchmark/wan_vae_decoder/."""

import sys

sys.path.insert(0, ".")
import bench_wan_decoder as B  # noqa: E402
import profile_kernels as P  # noqa: E402

_orig = P.make_runner


def make_runner(variant, raw, device):
    if variant == "mstar_warm_nobench":
        _, start, chunk = B.mstar_warm_node(raw, device, compiled=True, n_steady=(raw.shape[2] - 1) // 3,
                                            cudnn_benchmark=False)
        return start, chunk
    return _orig(variant, raw, device)


P.make_runner = make_runner
P.main()
