"""Disposable #1517 attention metadata, plan, and CUDA replay check.

CPU preflight checks the exact source files and both state-step metadata paths.
GPU mode uses one captured bucket with two real requests and a -2 pad row,
stages a changed plan on another CUDA stream, and compares replay with eager.
It does not test a full model or a live API server.
"""

from __future__ import annotations

import argparse
import contextlib
import hashlib
import json
import os
import subprocess
import sys
import tarfile
import tempfile
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("MSTAR_FLEX_BACKEND", "TRITON")

SOURCE_SHA256 = {
    "mstar/engine/state/adapters.py": "41914c20d9deb7b8ebc4d28b8ecd7fccddcbb7972d316ccb2ae87c40200af155",
    "mstar/engine/state/contract.py": "ad87e9fb30175b6b4132f177f57754b072f1bb0dd8a98d07640cd6aa39bd3892",
    "mstar/engine/resources/attn/ragged/base.py": "e45a489164c862b4175c4996f9ef61b85cf7366ca72db2d8f0da58932847f653",
    "mstar/engine/resources/attn/ragged/config.py": "044f46f26fb1c402486850d18143a63c5598345569a844d07e47d95858f16a98",
    "mstar/engine/resources/attn/ragged/chunk_causal.py": "b8bd3f33218c92e6d9a3b0541e0c7a8357160012e3c05d2f71bb74ef0a881dd5",
    "mstar/engine/resources/attn/ragged/expert_flex.py": "b72424b069932c2e62c4e68c1fc9b4305844b189cfd384c85c9a5c9d4880d39a",
}
SOURCE_COMMIT = "18c6c5c07edd3c23303c1f308ec8a0f08d155d46"
SOURCE_ARCHIVE_SHA256 = "d558863e9c834b8abc87aa25595da52f6d1f3a8af01071ce70f8308671c803de"


def check_source(repo: Path) -> None:
    if (repo / ".git").exists():
        actual = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True,
        ).strip()
        if actual != SOURCE_COMMIT:
            raise ValueError(f"source HEAD {actual} != {SOURCE_COMMIT}")
    for relative, expected in SOURCE_SHA256.items():
        actual = hashlib.sha256((repo / relative).read_bytes()).hexdigest()
        if actual != expected:
            raise ValueError(f"{relative} SHA-256 {actual} != {expected}")


@contextlib.contextmanager
def source_tree(repo: Path | None, archive: Path | None):
    if repo is not None:
        if not (repo / ".git").exists():
            raise ValueError("--repo requires an exact Git checkout; use --archive for a Git-less source")
        check_source(repo)
        yield repo
        return
    assert archive is not None
    actual = hashlib.sha256(archive.read_bytes()).hexdigest()
    if actual != SOURCE_ARCHIVE_SHA256:
        raise ValueError(f"source archive SHA-256 {actual} != {SOURCE_ARCHIVE_SHA256}")
    with tempfile.TemporaryDirectory(prefix="mstar-attention-18c6c5c0-") as name:
        root = Path(name)
        with tarfile.open(archive, "r:gz") as source:
            source.extractall(root, filter="data")
        check_source(root)
        yield root


def make_info(rid, walk, key, values):
    from mstar.conductor.request_info import CurrentForwardPassInfo

    return CurrentForwardPassInfo(
        request_id=rid, graph_walk=walk, fwd_index=0, random_seed=0,
        max_tokens=1, step_metadata={key: values},
    )


def make_chunk(phase, device, resources=None):
    import torch
    from mstar.engine.accelerator_graph_runner import dummy_metadata
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.step import Segment, SubmoduleStep
    from mstar.engine.state import Array, ArrayStep, Attend, ChunkCausal, KVEntry, Read
    from mstar.engine.state.adapters import (
        CHUNK_CAUSAL_LAYOUT_KWARG,
        build_specs,
        node_handles,
        state_steps,
    )

    layouts = (
        ((4, 16, 8), (2, 8, 0)) if phase == "cold" else
        ((4, 8, 8), (2, 16, 8))
    )
    array = Array(
        "tokens", nodes={"dit"}, layers={0},
        entry=KVEntry(heads=2, head_dim=32), max_len=0,
        reads={"attn": Read(mask=ChunkCausal(8), op=Attend(q_heads=2))},
    )
    if resources is None:
        spec = build_specs([array])[0]
        resource = build_resource(spec, EngineResourceInfo(device=device, kv_dtype=torch.bfloat16))
    else:
        resource = resources["attn"]
    handles = node_handles(SimpleNamespace(declare_state=lambda: [array]), "dit", {"attn": resource})
    infos = {
        rid: make_info(
            rid, "encode", CHUNK_CAUSAL_LAYOUT_KWARG,
            {"tokens": {"main": {"attn": list(lengths)}}},
        ) for rid, lengths in zip(("r0", "r1"), layouts, strict=True)
    }
    if phase == "cold":
        infos.update(dummy_metadata([-2], "encode"))
    segments = tuple(
        Segment(rid, "main", sum(lengths))
        for rid, lengths in zip(("r0", "r1"), layouts, strict=True)
    ) + (Segment(-2, "main", 8),)
    step = SubmoduleStep(state={"tokens": ArrayStep(segments=segments)})
    planned = state_steps(step, handles, per_request_info=infos)
    assert len(planned.steps["attn"].layouts) == 3
    assert planned.steps["attn"].layouts[-1].noised_len == 0
    return (
        {"attn": resource}, planned, {"attn": "main"},
        sum(s.span for s in segments[:-1]), sum(s.span for s in segments),
    )


def make_expert(phase, device, resources=None):
    import torch
    from mstar.engine.accelerator_graph_runner import dummy_metadata
    from mstar.engine.resources.base import EngineResourceInfo, build_resource
    from mstar.engine.resources.step import Segment, SubmoduleStep
    from mstar.engine.state import (
        Array,
        ArrayStep,
        Attend,
        ExpertCross,
        ExpertSelf,
        KVEntry,
        Read,
    )
    from mstar.engine.state.adapters import (
        EXPERT_LAYOUT_KWARG,
        build_specs,
        node_handles,
        state_steps,
    )

    layouts = (
        ((1, 2, 0, 2), (2, 2, 2, 4)) if phase == "cold" else
        ((2, 2, 2, 4), (1, 2, 0, 2))
    )
    arrays = [
        Array(
            "expert_q", nodes={"expert"}, labels={"q"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"self_attn": Read(mask=ExpertSelf(2, 1), op=Attend(q_heads=2))},
        ),
        Array(
            "trunk_kv", nodes={"expert"}, labels={"kv"}, layers={0},
            entry=KVEntry(heads=2, head_dim=32), max_len=0,
            reads={"cross_attn": Read(
                mask=ExpertCross(2, 1, 1, 2), op=Attend(q_heads=2), queries="expert_q",
            )},
        ),
    ]
    if resources is None:
        resources = {
            spec.resource_key: build_resource(spec, EngineResourceInfo(device=device, kv_dtype=torch.bfloat16))
            for spec in build_specs(arrays)
        }
    handles = node_handles(SimpleNamespace(declare_state=lambda: arrays), "expert", resources)
    infos = {}
    for rid, (chunks, glob, clean, noised) in zip(("r0", "r1"), layouts, strict=True):
        infos[rid] = make_info(rid, "chunk", EXPERT_LAYOUT_KWARG, {
            "expert_q": {"q": {"self_attn": [chunks]}},
            "trunk_kv": {"kv": {"cross_attn": [chunks, glob, clean, noised]}},
        })
    if phase == "cold":
        infos.update(dummy_metadata([-2], "chunk"))
    query_segments = tuple(
        Segment(rid, "q", chunks * 4)
        for rid, (chunks, _, _, _) in zip(("r0", "r1"), layouts, strict=True)
    ) + (Segment(-2, "q", 4),)
    key_segments = tuple(
        Segment(rid, "kv", glob + clean + noised)
        for rid, (_, glob, clean, noised) in zip(("r0", "r1"), layouts, strict=True)
    ) + (Segment(-2, "kv", 5),)
    step = SubmoduleStep(state={
        "expert_q": ArrayStep(segments=query_segments),
        "trunk_kv": ArrayStep(segments=key_segments),
    })
    planned = state_steps(step, handles, per_request_info=infos)
    assert set(planned.steps) == {"self_attn", "cross_attn"}
    assert planned.steps["self_attn"].layouts[-1][0] == 0
    assert planned.steps["cross_attn"].layouts[-1][0] == 0
    return (
        resources, planned, {"self_attn": "q", "cross_attn": "q<-kv"},
        sum(s.span for s in query_segments[:-1]), sum(s.span for s in query_segments),
    )


def ctx(walk, phase):
    from mstar.engine.resources.step import BucketKey, SlotLease, StepContext

    return StepContext(
        request_ids=[0, 1], graph_walk=walk, slot=0, capture=phase == "cold",
        slot_lease=SlotLease(0, BucketKey(walk, 3, 256)),
    )


def cpu_preflight():
    import torch

    results = {}
    for name, factory, walk in (
        ("chunk", make_chunk, "encode"), ("expert", make_expert, "chunk"),
    ):
        resources = None
        wrappers = {}
        for phase in ("cold", "warm"):
            resources, planned, labels, real_q, planned_q = factory(
                phase, torch.device("cpu"), resources,
            )
            for key, resource in resources.items():
                resource.plan(planned.steps[key], ctx(walk, phase))
                wrapper = resource._wrapper(labels[key])
                assert wrapper.accelerator_graph
                if phase == "warm":
                    assert wrapper is wrappers[key]
                else:
                    wrappers[key] = wrapper
            results[f"{name}_{phase}"] = {
                "resources": sorted(resources), "real_query_rows": real_q,
                "dummy_query_rows": planned_q - real_q,
                "layouts": {key: str(planned.steps[key].layouts) for key in resources},
            }
    return results


def gpu_case(name, factory, walk):
    import torch

    device = torch.device("cuda")
    resources, planned, labels, real_q, _planned_q = factory("cold", device)
    plan_stream = torch.cuda.Stream()
    run_stream = torch.cuda.current_stream()
    with torch.cuda.stream(plan_stream):
        for key, resource in resources.items():
            resource.plan(planned.steps[key], ctx(walk, "cold"))
        ready = torch.cuda.Event()
        ready.record(plan_stream)
    run_stream.wait_event(ready)

    results = {}
    for key, resource in resources.items():
        torch.manual_seed(120 + len(key))
        q = torch.randn((256, 2, 32), device=device, dtype=torch.bfloat16)
        k = torch.randn_like(q)
        v = torch.randn_like(q)
        label = labels[key]
        # Compile/warm the same graph-mode wrapper before CUDA capture.
        for _ in range(3):
            resource.run(q, k, v, label=label)
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            captured = resource.run(q, k, v, label=label)
        torch.cuda.synchronize()

        q.normal_()
        k.normal_()
        v.normal_()
        # Hold Q/K/V fixed across the cold and warm masks. Changing the input
        # alone must not satisfy the changed-layout assertion.
        graph.replay()
        cold_same_input = captured.clone()
        _, replay_plan, _, new_real_q, new_planned_q = factory("warm", device, resources)
        fresh_resources, oracle_plan, _, oracle_real_q, oracle_planned_q = factory("warm", device)
        assert (oracle_real_q, oracle_planned_q) == (new_real_q, new_planned_q)
        with torch.cuda.stream(plan_stream):
            resource.plan(replay_plan.steps[key], ctx(walk, "warm"))
            fresh_resources[key].plan(oracle_plan.steps[key], ctx(walk, "warm"))
            replay_ready = torch.cuda.Event()
            replay_ready.record(plan_stream)
        run_stream.wait_event(replay_ready)
        graph.replay()
        replay = captured.clone()
        fresh_warm = fresh_resources[key].run(q, k, v, label=label).clone()
        torch.cuda.synchronize()
        diff = (replay.float() - fresh_warm.float()).abs().max().item()
        changed = (
            cold_same_input[:min(real_q, new_real_q)].float()
            - replay[:min(real_q, new_real_q)].float()
        ).abs().max().item()
        # The -2 dummy is deliberately mapped to a real global span by the
        # metadata fallback. Compare its rows with the independent oracle,
        # and require zero only after that span, in the bucket padding.
        pad_nonzero = torch.count_nonzero(replay[new_planned_q:]).item()
        oracle_pad_nonzero = torch.count_nonzero(fresh_warm[new_planned_q:]).item()
        dummy_nonzero = torch.count_nonzero(replay[new_real_q:new_planned_q]).item()
        if (
            not torch.isfinite(replay).all()
            or not torch.isfinite(fresh_warm).all()
            or replay.dtype != torch.bfloat16
            or fresh_warm.dtype != torch.bfloat16
            or changed == 0
            or pad_nonzero
            or oracle_pad_nonzero
            or diff > 0.001
        ):
            raise AssertionError((name, key, diff, changed, pad_nonzero))
        results[key] = {
            "replay_vs_fresh_warm_max_abs": diff,
            "warm_vs_cold_same_input_max_abs": changed,
            "replay_padded_nonzero": pad_nonzero,
            "fresh_warm_padded_nonzero": oracle_pad_nonzero,
            "dtype": str(replay.dtype),
            "real_query_rows": new_real_q,
            "dummy_query_rows": new_planned_q - new_real_q,
            "dummy_nonzero": dummy_nonzero,
        }
    return results


def main():
    parser = argparse.ArgumentParser()
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--repo", type=Path)
    source.add_argument("--archive", type=Path)
    parser.add_argument("--mode", choices=("preflight", "gpu"), required=True)
    args = parser.parse_args()
    with source_tree(args.repo, args.archive) as repo:
        sys.path.insert(0, str(repo))
        if args.mode == "preflight":
            result = cpu_preflight()
        else:
            import torch

            if not torch.cuda.is_available() or torch.cuda.get_device_name(0).find("H200") < 0:
                raise RuntimeError("this gate requires one H200")
            result = {
                "chunk": gpu_case("chunk", make_chunk, "encode"),
                "expert": gpu_case("expert", make_expert, "chunk"),
            }
    print(json.dumps({"mode": args.mode, "source": SOURCE_COMMIT, "result": result}, sort_keys=True))


if __name__ == "__main__":
    main()
