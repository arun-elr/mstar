"""M*'s kernel_categories.py plus SGLang's custom Wan VAE kernels.

M*'s ``category`` keys off the ``triton_`` prefix that Inductor gives its
kernels. SGLang's hand-written Triton kernels use their own names, so without
the rules below they would land in "other":

  _cat_pad_cl3d_kernel           cat_pad_channels_last_3d: causal-conv input
                                 (cache frames + x + zero pad) in NDHWC, plus the
                                 compact next-chunk cache              -> concat / pad
  _fused_cat_pad_5d_kernel       causal_conv3d_cat_pad fallback fusion -> concat / pad
  _dup_up3d_add_kernel           main + DupUp3D(shortcut), Wan 2.2 residual
                                 up blocks (depth-to-space copy + add) -> other elementwise
                                 (what Inductor's fused add+copy of the same op
                                 falls under in M*'s profiles)
  _nearest_upsample_nhwc_kernel  NHWC nearest 2x upsample             -> upsample
                                 (already matched by M*'s "upsample" rule)
  _wan_rmsnorm_silu_kernel       fused RMSNorm+SiLU (quality gate on) -> norm / SiLU
                                 (already matched by M*'s "silu" rule)
  *norm_silu_post*               lossless norm post-ops (x/denom*scale*gamma, SiLU)
                                                                       -> norm / SiLU

Everything else is M*'s rule set, unchanged.
"""

import re

CATEGORIES = [
    "convolution", "layout conversion", "norm / SiLU", "concat / pad", "upsample",
    "attention", "copies", "other elementwise", "other",
]

SGLANG_RULES = [
    (r"_cat_pad_cl3d_kernel|_fused_cat_pad_5d_kernel", "concat / pad"),
    (r"_dup_up3d_add_kernel", "other elementwise"),
    (r"_nearest_upsample_nhwc_kernel", "upsample"),
    (r"_wan_rmsnorm_silu_kernel|norm_silu_post", "norm / SiLU"),
]


def mstar_category(name: str) -> str:
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


def category(name: str) -> str:
    for pat, cat in SGLANG_RULES:
        if re.search(pat, name):
            return cat
    return mstar_category(name)
