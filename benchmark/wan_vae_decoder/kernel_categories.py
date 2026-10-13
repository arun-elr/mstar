"""GPU kernel name -> profiler time category (heuristic, name based)."""

import re

CATEGORIES = [
    "convolution", "layout conversion", "norm / SiLU", "concat / pad", "upsample",
    "attention", "copies", "other elementwise", "other",
]


def category(name: str) -> str:
    n = name
    if re.search(r"nchwToNhwc|nhwcToNchw|nchw2nhwc|nhwc2nchw|convertTensor|transpose_readWrite", n):
        return "layout conversion"
    if n.startswith("triton_"):
        kind = n.split("_")[1]
        if re.search(r"_unsafe_index|upsample", n):
            return "upsample"
        if kind in ("per", "red"):
            return "norm / SiLU"  # Inductor reductions here are the RMS norms (+ fused SiLU/pad/cat)
        if re.search(r"constant_pad|_cat_|cat_|_cat$", n):
            return "concat / pad"
        return "other elementwise"
    if re.search(r"fmha|flash|attention|softmax", n, re.I):
        return "attention"
    if re.search(r"conv|implicit_gemm|xmma|fprop|cutlass|sm\d+_.*tensorop|cudnn", n, re.I):
        return "convolution"
    if re.search(r"upsample", n):
        return "upsample"
    if re.search(r"silu|NormTwo|norm_kernel|normalize|clamp_min|div_true|rsqrt|reduce_kernel", n):
        return "norm / SiLU"
    if re.search(r"CatArrayBatchedCopy|constant_pad|ConstantPad|FillFunctor|fill_kernel", n):
        return "concat / pad"
    if re.search(r"copy|Memcpy|Memset", n, re.I):
        return "copies"
    if re.search(r"elementwise", n):
        return "other elementwise"
    return "other"
