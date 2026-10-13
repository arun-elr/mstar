"""Group a bench_wan_decoder.py profiler table into kernel
categories, in ms per chunk (the profile covers 6 steady chunks).

  python profile_breakdown.py results-b200/b200-wan21/wan21/*.profile.txt
"""

import re
import sys
from collections import defaultdict

CHUNKS = 6
CATS = [
    ("layout conversion", r"nchwToNhwc|nhwcToNchw|nchw2nhwc|nhwc2nchw"),
    ("convolution", r"implicit_gemm|conv3d|conv2d|convolution(?!_6)|sm\d+_xmma|cudnn.*conv"),
    ("attention", r"fmha|flash|attention"),
    ("nearest upsample", r"upsample_nea|_unsafe_index"),
    ("pad / cat / copy", r"constant_pad|CatArrayBatched|cat_|copy_kernel|direct_copy"),
]


def to_ms(s: str) -> float:
    m = re.fullmatch(r"([\d.]+)(us|ms|s)", s)
    v, u = float(m[1]), m[2]
    return v / 1e3 if u == "us" else v * 1e3 if u == "s" else v


def breakdown(path):
    cats = defaultdict(float)
    total = None
    for line in open(path):
        if line.startswith("Self CUDA time total"):
            total = to_ms(line.split(":")[1].strip()) / CHUNKS
            continue
        cols = re.split(r"\s{2,}", line.strip())
        if len(cols) < 10 or not re.fullmatch(r"[\d.]+(us|ms|s)", cols[-5] or ""):
            continue
        name, self_cuda = cols[0], to_ms(cols[-5])
        cat = next((c for c, pat in CATS if re.search(pat, name)), "norm / SiLU / other elementwise")
        cats[cat] += self_cuda / CHUNKS
    return cats, total


if __name__ == "__main__":
    for p in sys.argv[1:]:
        cats, total = breakdown(p)
        listed = sum(cats.values())
        print(f"== {p}: {total:.1f} ms/chunk kernel time (top-60 rows cover {listed:.1f})")
        for c, v in sorted(cats.items(), key=lambda kv: -kv[1]):
            print(f"   {c:34s} {v:6.1f} ms")
