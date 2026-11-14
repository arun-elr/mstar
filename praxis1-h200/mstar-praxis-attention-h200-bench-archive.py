#!/usr/bin/env python3
"""Disposable DROID-shaped Praxis attention resource benchmark.

This measures synthetic BF16 Q/K/V through the resource wrappers. It is not a
served-session, checkpoint, model-quality, or whole-engine graph benchmark.
Run on an approved H200 only; importing this file does not allocate a GPU.
"""

from __future__ import annotations

import argparse
import json
import math
import os
from pathlib import Path
import statistics
import subprocess
import sys
import time


def arguments() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--repo", type=Path, required=True, help="checkout containing current #1492/#1517 resources")
    p.add_argument("--text-tokens", type=int, required=True, help="actual trimmed conditional text length")
    p.add_argument("--blank-tokens", type=int, required=True, help="actual trimmed blank text length")
    p.add_argument("--text-source", required=True, help="how the two text lengths were measured")
    p.add_argument("--samples", type=int, default=100)
    p.add_argument("--warmups", type=int, default=20)
    p.add_argument("--output", type=Path, required=True)
    return p.parse_args()


def summary(values: list[float]) -> dict[str, float]:
    x = sorted(values)

    def pct(p: float) -> float:
        at = (len(x) - 1) * p
        low = math.floor(at)
        return x[low] + (x[min(low + 1, len(x) - 1)] - x[low]) * (at - low)

    return {
        "min": x[0], "p50": pct(0.50), "p95": pct(0.95),
        "p99": pct(0.99), "max": x[-1], "mean": statistics.fmean(x),
    }


def revision(repo: Path) -> str:
    return "5902054c0f9fa057a80ae5a7e15e67f19d4bcde2 (verified source archive)"


def round_up(n: int, block: int = 128) -> int:
    return (n + block - 1) // block * block


def main() -> None:
    args = arguments()
    if args.samples < 1 or args.warmups < 1 or min(args.text_tokens, args.blank_tokens) < 0:
        raise ValueError("samples and warmups must be positive; token counts cannot be negative")
    if os.getenv("MSTAR_FLEX_BACKEND", "TRITON") != "TRITON":
        raise RuntimeError("set MSTAR_FLEX_BACKEND=TRITON for the comparable pinned-backend run")
    os.environ["MSTAR_FLEX_BACKEND"] = "TRITON"  # before any mstar import
    sys.path.insert(0, str(args.repo.resolve()))

    import torch
    import torch.nn.functional as F
    from mstar.engine.resources.attn import flex as flex_module
    from mstar.engine.resources.attn.ragged.chunk_causal import FlexChunkCausalWrapper
    from mstar.engine.resources.attn.ragged.config import ChunkCausalLayout
    from mstar.engine.resources.attn.ragged.expert_flex import FlexExpertWrapper
    from mstar.engine.resources.attn.ragged.expert_mask import (
        ExpertCrossLayout, ExpertCrossMask, ExpertSelfLayout, ExpertSelfMask,
        cross_visibility, self_visibility,
    )

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; this script does not fall back to CPU")
    if flex_module._FLEX_BACKEND != "TRITON":
        raise RuntimeError(f"unexpected Flex backend: {flex_module._FLEX_BACKEND}")
    torch.manual_seed(12)
    device = torch.device("cuda:0")
    dtype = torch.bfloat16
    block = 128
    patch_tokens = 7 * 26
    chunk_tokens = 6 * patch_tokens
    expert_chunk = 6
    expert_future = 8
    results: dict = {
        "source": "synthetic BF16 QKV through current M* attention resources",
        "limits": ["not a checkpoint run", "not whole-engine capture", "not serving latency"],
        "environment": {
            "revision": revision(args.repo), "repo": str(args.repo.resolve()),
            "torch": torch.__version__, "cuda_runtime": torch.version.cuda,
            "python": sys.executable,
            "gpu": torch.cuda.get_device_name(0),
            "gpu_capability": list(torch.cuda.get_device_capability(0)),
            "flex_backend": flex_module._FLEX_BACKEND,
            "dtype": "bfloat16", "samples": args.samples, "warmups": args.warmups,
        },
        "geometry": {
            "canvas": [224, 832], "patch_tokens_per_frame": patch_tokens,
            "frames_per_chunk": 6, "chunk_tokens": chunk_tokens,
            "registers": 4, "block_size": block,
            "full_blocks": False,
            "text_tokens": args.text_tokens, "blank_tokens": args.blank_tokens,
            "text_source": args.text_source,
            "graph_bucket_source": "round_up(max(cold,warm),128), synthetic wrapper-only bucket",
        },
        "cases": [],
    }

    def qkv(q_len: int, kv_len: int, hq: int, hkv: int, d: int):
        return (
            torch.randn(q_len, hq, d, device=device, dtype=dtype),
            torch.randn(kv_len, hkv, d, device=device, dtype=dtype),
            torch.randn(kv_len, hkv, d, device=device, dtype=dtype),
        )

    def direct(wrapper, q, k, v):
        qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        return flex_module.flex_attention_masked(
            qh, kh, vh, block_mask=wrapper.block_mask, scale=wrapper.sm_scale,
            enable_gqa=wrapper.num_qo_heads != wrapper.num_kv_heads,
        )[0].transpose(0, 1).contiguous()

    def max_abs(a, b) -> float:
        return (a.float() - b.float()).abs().max().item()

    def expert_dense_oracle(mode, layout, q, k, v, d):
        chunks, glob, clean, noised = layout
        visible = (
            self_visibility(ExpertSelfMask(expert_chunk, expert_future), ExpertSelfLayout(chunks))
            if mode == "self" else cross_visibility(
                ExpertCrossMask(expert_chunk, expert_future, patch_tokens, 93, True),
                ExpertCrossLayout(chunks, glob, clean, noised),
            )
        ).to(device)
        qh, kh, vh = (x.transpose(0, 1).unsqueeze(0) for x in (q, k, v))
        return F.scaled_dot_product_attention(
            qh, kh, vh, attn_mask=visible, scale=d ** -0.5,
        )[0].transpose(0, 1)

    def assert_parity(a, b, label: str) -> float:
        err = max_abs(a, b)
        try:
            torch.testing.assert_close(a, b, atol=0.04, rtol=0.04)
        except AssertionError as exc:
            raise AssertionError(f"{label}: max_abs={err}; {exc}") from exc
        if not torch.isfinite(a).all() or not torch.isfinite(b).all():
            raise AssertionError(f"{label}: non-finite output")
        return err

    def time_cuda(call):
        for _ in range(args.warmups):
            call()
        torch.cuda.synchronize()
        gpu_ms: list[float] = []
        wall_ms: list[float] = []
        for _ in range(args.samples):
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            t0 = time.perf_counter()
            start.record()
            call()
            end.record()
            end.synchronize()
            wall_ms.append((time.perf_counter() - t0) * 1000)
            gpu_ms.append(start.elapsed_time(end))
        return {
            "gpu_ms": summary(gpu_ms), "wall_ms": summary(wall_ms),
            "raw_gpu_ms": gpu_ms, "raw_wall_ms": wall_ms,
        }

    def time_plan(wrapper, first, second):
        changed: list[float] = []
        cached: list[float] = []
        for iteration in range(args.warmups + args.samples):
            wrapper.plan(first)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            wrapper.plan(second)
            torch.cuda.synchronize()
            if iteration >= args.warmups:
                changed.append((time.perf_counter() - t0) * 1000)
            t0 = time.perf_counter()
            wrapper.plan(second)
            torch.cuda.synchronize()
            if iteration >= args.warmups:
                cached.append((time.perf_counter() - t0) * 1000)
        return {
            "changed_wall_ms": summary(changed), "cached_wall_ms": summary(cached),
            "raw_changed_wall_ms": changed, "raw_cached_wall_ms": cached,
        }

    def make_wrapper(kind: str, graph: bool, ceiling: int):
        common = dict(
            max_num_segments=1 if graph else None,
            max_total_tokens=ceiling if graph else None,
            device=device, accelerator_graph=graph, q_data_type=dtype,
        )
        if kind == "chunk":
            return FlexChunkCausalWrapper(24, 8, 128, block, False, **common)
        return FlexExpertWrapper(
            16, 16, 64, kind, expert_chunk, expert_future, patch_tokens,
            93, True, block, False, **common,
        )

    def lengths(kind, layout):
        if kind == "chunk":
            return layout.span, layout.span
        k, glob, clean, noised = layout
        q = k * (1 + expert_future + expert_chunk)
        kv = q if kind == "self" else glob + (clean + noised) * patch_tokens
        return q, kv

    def case(name: str, kind: str, cold_layout, warm_layout):
        layouts = {"cold": cold_layout, "warm": warm_layout}
        spans = {phase: lengths(kind, layout) for phase, layout in layouts.items()}
        ceiling = round_up(max(max(spans["cold"]), max(spans["warm"])), block)
        hq, hkv, d = (24, 8, 128) if kind == "chunk" else (16, 16, 64)
        item = {
            "name": name, "kind": kind,
            "heads": {"q": hq, "kv": hkv, "dim": d},
            "layouts": {phase: list(layout) for phase, layout in layouts.items()},
            "lengths": {phase: {"q": pair[0], "kv": pair[1]} for phase, pair in spans.items()},
            "graph_ceiling_q_and_kv": ceiling, "eager": {}, "captured": {},
        }
        print(f"benchmarking {name}: cold={spans['cold']}, warm={spans['warm']}, graph ceiling={ceiling}", file=sys.stderr, flush=True)

        eager = make_wrapper(kind, False, 0)
        item["eager_plan"] = time_plan(eager, [cold_layout], [warm_layout])
        for phase, layout in layouts.items():
            eager.plan([layout])
            q, k, v = qkv(*spans[phase], hq, hkv, d)
            got = eager.run(q, k, v)
            oracle = direct(eager, q, k, v)
            error = assert_parity(got, oracle, f"{name}/{phase}/direct-same-mask")
            independent_error = None
            if kind != "chunk":
                independent_error = assert_parity(
                    got, expert_dense_oracle(kind, layout, q, k, v, d),
                    f"{name}/{phase}/expert-dense-oracle",
                )
            item["eager"][phase] = {
                "wrapper_vs_direct_same_mask_max_abs": error,
                "expert_dense_oracle_max_abs": independent_error,
                "wrapper": time_cuda(lambda: eager.run(q, k, v)),
                "direct_same_mask": time_cuda(lambda: direct(eager, q, k, v)),
            }

        graph_wrapper = make_wrapper(kind, True, ceiling)
        qg, kg, vg = qkv(ceiling, ceiling, hq, hkv, d)
        graph_wrapper.plan([warm_layout])
        for _ in range(args.warmups):
            graph_wrapper.run(qg, kg, vg)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured_output = graph_wrapper.run(qg, kg, vg)
        torch.cuda.synchronize()
        qg.normal_()
        kg.normal_()
        vg.normal_()
        item["captured_plan"] = time_plan(graph_wrapper, [warm_layout], [cold_layout])
        for phase, layout in layouts.items():
            graph_wrapper.plan([layout])
            torch.cuda.synchronize()
            graph.replay()
            torch.cuda.synchronize()
            q_real, kv_real = spans[phase]
            eager.plan([layout])
            want = eager.run(qg[:q_real], kg[:kv_real], vg[:kv_real])
            error = assert_parity(
                captured_output[:q_real], want,
                f"{name}/{phase}/captured-replan-vs-eager",
            )
            if torch.count_nonzero(captured_output[q_real:]).item():
                raise AssertionError(f"{name}/{phase}: captured padding output is nonzero")
            item["captured"][phase] = {
                "max_abs_vs_eager": error, "padding_rows": ceiling - q_real,
                "padding_zero": True,
                "replay": time_cuda(graph.replay),
            }
        results["cases"].append(item)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(results, indent=2) + "\n")

    for label, text_len in (("cond", args.text_tokens), ("blank", args.blank_tokens)):
        glob = 4 + patch_tokens + text_len
        case(
            f"encoder_{label}", "chunk",
            ChunkCausalLayout(glob, chunk_tokens, chunk_tokens, 0),
            ChunkCausalLayout(glob, chunk_tokens, 2 * chunk_tokens, chunk_tokens),
        )
    case(
        "decoder", "chunk",
        ChunkCausalLayout(4, 4 * chunk_tokens, 4 * chunk_tokens, 0),
        ChunkCausalLayout(4, 4 * chunk_tokens, 8 * chunk_tokens, 0),
    )
    glob = 4 + patch_tokens + args.text_tokens
    for mode in ("cross", "self"):
        case(f"expert_{mode}", mode, (1, glob, 0, 6), (2, glob, 6, 12))

    print(json.dumps({"output": str(args.output), "cases": len(results["cases"])}))


if __name__ == "__main__":
    main()
